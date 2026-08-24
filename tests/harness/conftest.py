"""Network-free doubles for the two SDKs the harness talks to.

docs/execution-plan.md ground rule 5 — unit tests never touch the network.
`FakeLangSmithClient` deliberately reproduces the *projection* semantics that
bit Phase 2: fields outside an explicit `select=` come back `None`, so a
collector that forgets to ask for a field it audits ends up auditing nothing.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

T0 = datetime(2026, 8, 23, 12, 0, 0, tzinfo=UTC)

# Every attribute of langsmith.schemas.Run the harness may ever read.
RUN_FIELDS = (
    "id",
    "name",
    "run_type",
    "parent_run_id",
    "trace_id",
    "start_time",
    "end_time",
    "inputs",
    "outputs",
    "error",
    "extra",
    "tags",
    "serialized",
    "status",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "session_id",
)


@dataclass
class FakeRun:
    id: str
    name: str
    run_type: str
    parent_run_id: str | None = None
    trace_id: str = "tr-1"
    start_time: datetime = T0
    end_time: datetime | None = None
    inputs: dict[str, Any] | None = field(default_factory=dict)
    outputs: dict[str, Any] | None = field(default_factory=dict)
    error: str | None = None
    extra: dict[str, Any] | None = field(default_factory=dict)
    tags: list[str] | None = field(default_factory=list)
    serialized: dict[str, Any] | None = None
    status: str = "success"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    session_id: str = "proj-1"

    def __post_init__(self):
        if self.end_time is None and self.start_time is not None:
            self.end_time = self.start_time + timedelta(milliseconds=250)


def _project(run: FakeRun, select: list[str] | None) -> FakeRun:
    """Mirror LangSmith: unselected fields are simply not returned."""
    if select is None:
        return run
    kept = {f: getattr(run, f) for f in RUN_FIELDS if f in select or f == "id"}
    blanked = {f: None for f in RUN_FIELDS if f not in kept}
    return FakeRun(**{**kept, **blanked, "id": run.id})


_SESSION_IN_FILTER = re.compile(r'"session_id"\s*:\s*"([^"]+)"')


class FakeLangSmithClient:
    """Stands in for `langsmith.Client` for the two calls the collector makes."""

    def __init__(
        self,
        runs: list[FakeRun],
        *,
        honor_metadata_filter: bool = True,
        reveal_schedule: list[list[FakeRun]] | None = None,
        drop_serialized: bool = False,
    ):
        self.runs = list(runs)
        self.honor_metadata_filter = honor_metadata_filter
        # Successive results for child-span fetches, to simulate LangSmith's
        # asynchronous child ingestion lagging the root.
        self.reveal_schedule = reveal_schedule
        self.drop_serialized = drop_serialized
        self.calls: list[dict] = []
        self._child_fetches = 0

    def list_runs(
        self,
        *,
        project_name: str | None = None,
        trace_id: Any = None,
        is_root: bool | None = None,
        run_type: str | None = None,
        select: list[str] | None = None,
        filter: str | None = None,  # noqa: A002 - mirrors the SDK's parameter name
        limit: int | None = None,
    ):
        self.calls.append(
            {
                "project_name": project_name,
                "trace_id": trace_id,
                "is_root": is_root,
                "run_type": run_type,
                "select": select,
                "filter": filter,
                "limit": limit,
            }
        )
        if trace_id is not None and self.reveal_schedule is not None:
            index = min(self._child_fetches, len(self.reveal_schedule) - 1)
            self._child_fetches += 1
            pool = self.reveal_schedule[index]
        else:
            pool = self.runs

        out = []
        for run in pool:
            if trace_id is not None and str(run.trace_id) != str(trace_id):
                continue
            if is_root is not None and (run.parent_run_id is None) != is_root:
                continue
            if run_type is not None and run.run_type != run_type:
                continue
            if filter and self.honor_metadata_filter:
                match = _SESSION_IN_FILTER.search(filter)
                if match:
                    metadata = (run.extra or {}).get("metadata") or {}
                    if metadata.get("session_id") != match.group(1):
                        continue
            projected = _project(run, select)
            if self.drop_serialized:
                projected.serialized = None
            out.append(projected)
        return out[:limit] if limit else out


# --------------------------------------------------------------- run builders

def metadata_extra(session_id: str, turn_index: int = 0, **extra: Any) -> dict:
    """A realistic `run.extra` — including the noise the collector must drop."""
    return {
        "metadata": {
            "session_id": session_id,
            "turn_index": turn_index,
            "ls_model_name": "gpt-5.1-mini",
            "thread_id": "thread-abc",
            "assistant_id": "target_app",
            **extra,
        },
        "runtime": {"library": "langchain", "sdk_version": "0.1.0"},
    }


def react_agent_runs(
    session_id: str,
    *,
    turn_index: int = 0,
    trace_id: str = "tr-1",
    user_message: str = "what is the refund window?",
    final_response: str = "30 days.",
    leak_metadata: bool = False,
    docs: list[dict] | None = None,
) -> list[FakeRun]:
    """One create_react_agent turn as LangSmith records it, noise spans included."""
    leak = {"fault_retriever": {"behavior": "stale"}} if leak_metadata else {}
    extra = metadata_extra(session_id, turn_index, **leak)
    root_id = f"root-{trace_id}"
    docs = docs if docs is not None else [{"doc_id": "refund-policy", "updated": "2026-01-02"}]
    return [
        FakeRun(
            id=root_id,
            name="target_app",
            run_type="chain",
            parent_run_id=None,
            trace_id=trace_id,
            inputs={"messages": [{"type": "human", "content": user_message}]},
            outputs={
                "messages": [
                    {"type": "human", "content": user_message},
                    {"type": "ai", "content": final_response},
                ]
            },
            extra=extra,
        ),
        FakeRun(
            id=f"agent-{trace_id}",
            name="agent",
            run_type="chain",
            parent_run_id=root_id,
            trace_id=trace_id,
            extra=extra,
        ),
        FakeRun(
            id=f"seq-{trace_id}",
            name="RunnableSequence",
            run_type="chain",
            parent_run_id=f"agent-{trace_id}",
            trace_id=trace_id,
            extra=extra,
        ),
        FakeRun(
            id=f"llm-{trace_id}",
            name="ChatOpenAI",
            run_type="llm",
            parent_run_id=f"seq-{trace_id}",  # nested under a noise span on purpose
            trace_id=trace_id,
            inputs={"messages": [[{"type": "human", "content": user_message}]]},
            outputs={"generations": [[{"text": final_response}]]},
            extra=extra,
            serialized={"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]},
            prompt_tokens=120,
            completion_tokens=17,
        ),
        FakeRun(
            id=f"write-{trace_id}",
            name="ChannelWrite<messages>",
            run_type="chain",
            parent_run_id=f"agent-{trace_id}",
            trace_id=trace_id,
            extra=extra,
        ),
        FakeRun(
            id=f"tools-{trace_id}",
            name="tools",
            run_type="chain",
            parent_run_id=root_id,
            trace_id=trace_id,
            extra=extra,
        ),
        FakeRun(
            id=f"tool-{trace_id}",
            name="rag_search",
            run_type="tool",
            parent_run_id=f"tools-{trace_id}",
            trace_id=trace_id,
            inputs={"input": {"query": "refund window"}},
            outputs={"output": json.dumps(docs)},
            extra=extra,
        ),
        FakeRun(
            id=f"retr-{trace_id}",
            name="corpus_search",
            run_type="retriever",
            parent_run_id=f"tool-{trace_id}",
            trace_id=trace_id,
            inputs={"query": "refund window", "k": 3},
            outputs={"output": docs},
            extra=extra,
        ),
    ]


def new_id() -> str:
    return uuid.uuid4().hex[:8]
