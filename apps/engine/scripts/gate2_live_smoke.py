"""Gate 2 — run the served Engine over the fixture traces with a real model.

Drives the app exactly the way the benchmark will: `langgraph_sdk` against the
`base_url` / `assistant_id` in `configs/engine.yaml`, model chosen through
`config.configurable[model_configurable_key]`. No imports from `engine`.

    uv run python scripts/gate2_live_smoke.py [model_id]

Reports which planted errors the Engine found. Imperfect recall is acceptable —
this gate fails on crashes, schema violations, and a lost seed board, not on a
missed finding.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _contract import (  # noqa: E402
    FIXTURES,
    banner,
    load_categories,
    load_contract,
    load_seed_board,
    summarize,
    validate_board,
)

TRACES_FILE = FIXTURES / "traces.json"
ANSWER_KEY = FIXTURES / "planted_errors.json"
LAST_RUN = Path(__file__).resolve().parent / ".last_run.json"


def run_engine(model: str | None = None) -> dict:
    """One Engine run through the LangGraph Server API. Returns the raw output."""
    from langgraph_sdk import get_sync_client  # noqa: PLC0415

    contract = load_contract()
    client = get_sync_client(url=contract["base_url"])

    trace_count = len(json.loads(TRACES_FILE.read_text())["traces"])
    config: dict = {
        # 2 supersteps + 1 per trace. The graph raises its own default, but a
        # caller driving a large corpus should set this explicitly.
        "recursion_limit": 2 * trace_count + 10,
    }
    if model:
        config["configurable"] = {contract["model_configurable_key"]: model}

    started = time.time()
    thread = client.threads.create()
    result = client.runs.wait(
        thread["thread_id"],
        contract["assistant_id"],
        input={
            "trace_file": str(TRACES_FILE),
            "seed_issueboard": load_seed_board(),
            "categories": load_categories(),
        },
        config=config,
    )
    if isinstance(result, dict) and result.get("__error__"):
        raise SystemExit(f"run failed: {result['__error__']}")
    print(f"  run finished in {time.time() - started:.1f}s")
    return result


def report_recall(board, model: str) -> dict:
    """Which planted errors were found, and what the Engine said about the clean ones."""
    key = json.loads(ANSWER_KEY.read_text())
    by_id = {issue.error_id: issue for issue in board.issues}
    by_trace: dict[str, list] = {}
    for occurrence in board.occurrences:
        by_trace.setdefault(occurrence.trace_id, []).append(occurrence)

    rows = []
    for planted in key["planted"]:
        hits = by_trace.get(planted["trace_id"], [])
        issues = [by_id[o.error_id] for o in hits if o.error_id in by_id]
        categories = sorted({i.category_id for i in issues})
        expected = set(planted["expected_categories"])
        rows.append(
            {
                "trace_id": planted["trace_id"],
                "planted": planted["label"],
                "found": bool(hits),
                "category_match": bool(expected & set(categories)),
                "reported_categories": categories,
                "reported_titles": [i.title for i in issues],
                "merged_into_seed": (
                    planted["expected_seed_error_id"] in {o.error_id for o in hits}
                    if planted["expected_seed_error_id"]
                    else None
                ),
            }
        )

    flagged_clean = {t: len(by_trace.get(t, [])) for t in key["clean_traces"]}
    summary = {
        "model": model,
        "planted_found": sum(r["found"] for r in rows),
        "planted_total": len(rows),
        "category_matches": sum(r["category_match"] for r in rows),
        "clean_traces_flagged": {t: n for t, n in flagged_clean.items() if n},
        "occurrences_by_trace": {t: len(o) for t, o in sorted(by_trace.items())},
        "detail": rows,
    }
    print(json.dumps(summary, indent=2))
    return summary


def smoke(model: str | None = None) -> dict:
    label = model or "(default)"
    banner(f"GATE 2 — live Engine run over the fixture traces  [model={label}]")
    payload = run_engine(model)

    assert "issues" in payload and "occurrences" in payload, f"unexpected output: {sorted(payload)}"
    assert "injection_mode" not in json.dumps(payload), "Engine output names an ablation field"

    board = validate_board(payload)
    print(json.dumps(summarize(board), indent=2))

    seed_ids = [i["error_id"] for i in load_seed_board()["issues"]]
    board_ids = [i.error_id for i in board.issues]
    assert board_ids[: len(seed_ids)] == seed_ids, "the seed board was not carried through"
    assert len(board_ids) == len(set(board_ids)), "duplicate error_id on the board"
    assert board.occurrences, "no occurrences at all — the run found nothing anywhere"

    banner("planted-error recall")
    recall = report_recall(board, model or "default")

    LAST_RUN.write_text(json.dumps({"model": model, "board": payload, "recall": recall}, indent=2))
    print(f"\nOK — schema-valid Issueboard(source='engine_predicted'); board written to {LAST_RUN}")
    return {"board": board, "recall": recall}


def main() -> int:
    smoke(sys.argv[1] if len(sys.argv) > 1 else None)
    banner("GATE 2 PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
