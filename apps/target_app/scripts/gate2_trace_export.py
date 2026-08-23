"""Gate 2 — throwaway LangSmith -> benchmark.schemas.traces.Trace normalizer.

Proves the app's traces are exportable into the system's own schema. The real
collector is Phase 4's; this script exists only to demonstrate the shape is
reachable, so it is deliberately small and makes no attempt at completeness.

Usage: gate2_trace_export.py [session_id]  (defaults to the first Gate-1 session)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _contract import REPO_ROOT, banner, load_contract  # noqa: E402
from langsmith import Client  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from benchmark.schemas.traces import Span, Trace, Turn  # noqa: E402

SESSIONS_FILE = Path(__file__).resolve().parent / ".last_sessions.json"

RUN_TYPE_TO_SPAN_TYPE = {
    "llm": "llm",
    "tool": "tool",
    "retriever": "retrieval",
    "chain": "chain",
    "prompt": "chain",
    "parser": "chain",
}


def span_type_for(run, is_root: bool) -> str:
    if is_root:
        return "agent"
    return RUN_TYPE_TO_SPAN_TYPE.get(run.run_type, "chain")


def find_root(client: Client, project: str, session_id: str, attempts: int = 30):
    """LangSmith ingestion is asynchronous — poll for the root run."""
    for attempt in range(attempts):
        for run in client.list_runs(project_name=project, is_root=True, limit=25):
            metadata = (run.extra or {}).get("metadata") or {}
            if metadata.get("session_id") == session_id:
                return run
        time.sleep(2)
        print(f"  waiting for LangSmith ingestion ({attempt + 1}/{attempts})…")
    raise SystemExit(f"no root run found for session_id={session_id}")


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def user_and_final(root) -> tuple[str, str]:
    messages = (root.inputs or {}).get("messages") or []
    humans = [m for m in messages if (m.get("type") or m.get("role")) in ("human", "user")]
    user = message_text(humans[0]) if humans else ""
    out_messages = (root.outputs or {}).get("messages") or []
    final = message_text(out_messages[-1]) if out_messages else ""
    return user, final


def normalize(root, runs, session_id: str) -> Trace:
    spans = []
    for run in runs:
        is_root = run.id == root.id
        spans.append(
            Span(
                span_id=str(run.id),
                parent_span_id=str(run.parent_run_id) if run.parent_run_id else None,
                name=run.name,
                span_type=span_type_for(run, is_root),
                start_time=run.start_time,
                end_time=run.end_time or run.start_time,
                inputs=run.inputs or {},
                outputs=run.outputs or {},
                attributes={
                    "run_type": run.run_type,
                    "model": ((run.extra or {}).get("metadata") or {}).get("ls_model_name"),
                    "tokens": (run.prompt_tokens or 0) + (run.completion_tokens or 0)
                    if run.run_type == "llm"
                    else None,
                    "error": run.error,
                },
            )
        )
    user, final = user_and_final(root)
    return Trace(
        trace_id=str(root.trace_id),
        input_id=session_id,
        mode="single_turn",
        turns=[Turn(turn_index=0, user_message=user, final_response=final, spans=spans)],
        status="app_error" if root.error else "ok",
        metadata={
            "langsmith_project": root.session_id and str(root.session_id),
            "app": "target_app",
            "start_time": root.start_time.isoformat(),
        },
    )


def main() -> int:
    contract = load_contract()
    sessions = json.loads(SESSIONS_FILE.read_text())
    session_id = sys.argv[1] if len(sys.argv) > 1 else sessions[0]["session_id"]

    banner(f"GATE 2 — LangSmith trace -> Trace schema (session {session_id})")
    client = Client()
    project = contract["langsmith_project"]

    root = find_root(client, project, session_id)
    runs = list(client.list_runs(project_name=project, trace_id=root.trace_id))
    trace = normalize(root, runs, session_id)

    # Round-trip through the schema to prove validity, not just construction.
    reparsed = Trace.model_validate_json(trace.model_dump_json())
    assert reparsed == trace

    turn = trace.turns[0]
    by_type: dict[str, int] = {}
    for span in turn.spans:
        by_type[span.span_type] = by_type.get(span.span_type, 0) + 1

    summary = {
        "trace_id": trace.trace_id,
        "input_id": trace.input_id,
        "mode": trace.mode,
        "status": trace.status,
        "turns": len(trace.turns),
        "span_count": len(turn.spans),
        "span_types": dict(sorted(by_type.items())),
        "span_names": sorted({s.name for s in turn.spans}),
        "user_message": turn.user_message[:120],
        "final_response_chars": len(turn.final_response),
    }
    print(json.dumps(summary, indent=2))
    assert summary["span_count"] > 0
    assert "retrieval" in by_type, "retriever span missing from the normalized trace"
    assert "llm" in by_type
    print("\nOK — LangSmith run tree normalized into benchmark.schemas.traces.Trace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
