"""Gate 2 — run the served Engine over the fixture traces with a real model.

Drives the app exactly the way the benchmark will: `langgraph_sdk` against the
`base_url` / `assistant_id` in `configs/engine.yaml`, model chosen through
`config.configurable[model_configurable_key]`. No imports from `engine`.

    uv run python scripts/gate2_live_smoke.py [model_id] [analysis_concurrency]

Reports which planted errors the Engine found. Imperfect recall is acceptable —
this gate fails on crashes, schema violations, and a lost seed board, not on a
missed finding.
"""

from __future__ import annotations

import json
import os
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
# Mirrors engine.graph.DEFAULT_ANALYSIS_CONCURRENCY. Named here rather than
# imported: these scripts stand in for the benchmark and must not import the app.
DEFAULT_CONCURRENCY = 8
ANSWER_KEY = FIXTURES / "planted_errors.json"
LAST_RUN = Path(__file__).resolve().parent / ".last_run.json"


def recorded_models(client, thread_id: str) -> list[str]:
    """Read back, from the server, the model each run on this thread was given.

    The point of the model-swap gate is that the swap actually happened. Trusting
    the request we just sent proves nothing — the config could be dropped on the
    way in (LangGraph silently declines to inject a config whose node annotation
    it does not recognise, which is exactly the bug this project already hit
    once). So the assertion reads the run record the server persisted.
    """
    models = []
    for run in client.runs.list(thread_id):
        record = run if isinstance(run, dict) else getattr(run, "__dict__", {})
        config = (record.get("kwargs") or {}).get("config") or record.get("config") or {}
        configurable = config.get("configurable") or {}
        if "model" in configurable:
            models.append(configurable["model"])
    return models


def run_engine(model: str | None = None, concurrency: int = DEFAULT_CONCURRENCY) -> dict:
    """One Engine run through the LangGraph Server API.

    Returns {output, thread_id, recorded_models, seconds} — `recorded_models` so
    callers can verify server-side which model the run actually received, and
    `seconds` because the assignment's 300-trace deliverable lives or dies on
    the per-trace cost.
    """
    from langgraph_sdk import get_sync_client  # noqa: PLC0415

    contract = load_contract()
    client = get_sync_client(url=contract["base_url"])

    trace_count = len(json.loads(TRACES_FILE.read_text())["traces"])
    configurable: dict = {"analysis_concurrency": concurrency}
    if model:
        configurable[contract["model_configurable_key"]] = model
    config: dict = {
        "configurable": configurable,
        # 2 supersteps + ceil(n / concurrency). The graph raises its own
        # default, but a caller driving a large corpus should set this.
        "recursion_limit": 2 * trace_count + 10,
    }

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
    seconds = time.time() - started
    # Batched work does NOT scale per-trace: a batch costs roughly its SLOWEST
    # trace, so dividing by trace_count flatters a wide batch badly (six traces
    # at N=8 is one batch, and "10s/trace" would be an artefact of the batch
    # never being full). Report the batch facts and let the reader extrapolate
    # on ceil(n / N) batches.
    batches = -(-trace_count // concurrency)
    print(
        f"  run finished in {seconds:.1f}s — {trace_count} traces at "
        f"concurrency={concurrency} = {batches} batch(es), "
        f"{seconds / batches:.1f}s per batch (incl. consolidation)"
    )

    try:
        seen = recorded_models(client, thread["thread_id"])
    except Exception as exc:  # readback is best-effort; the run itself stands
        print(f"  (could not read the run config back: {type(exc).__name__}: {exc})")
        seen = []
    return {
        "output": result,
        "thread_id": thread["thread_id"],
        "recorded_models": seen,
        "seconds": seconds,
        "seconds_per_batch": seconds / batches,
        "batches": batches,
        "trace_count": trace_count,
        "concurrency": concurrency,
    }


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


def smoke(model: str | None = None, concurrency: int = DEFAULT_CONCURRENCY) -> dict:
    label = model or "(default)"
    banner(
        f"GATE 2 — live Engine run over the fixture traces  "
        f"[model={label} concurrency={concurrency}]"
    )
    run = run_engine(model, concurrency)
    payload = run["output"]

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

    # A hand-crafted clean trace has nothing to find. Anything reported against
    # one is a false positive, and false positives are the half of the scoring
    # picture that recall alone hides.
    #
    # This is a real quality signal, but it is a signal about a NON-DETERMINISTIC
    # model: it fired once in five live runs during development (an over-eager
    # reading of a clean trace at concurrency=1, where every trace sees the full
    # running-title list and the model goes looking for known modes everywhere).
    # The default stays 0 so a regression is loud; ENGINE_GATE_MAX_CLEAN_FLAGS
    # lets a caller who has decided to tolerate the flake say so out loud rather
    # than delete the check.
    tolerated = int(os.environ.get("ENGINE_GATE_MAX_CLEAN_FLAGS", "0"))
    flagged = recall["clean_traces_flagged"]
    assert len(flagged) <= tolerated, (
        f"issues reported against {len(flagged)} trace(s) with nothing planted in "
        f"them (tolerating {tolerated}): {flagged}"
    )
    if flagged:
        print(f"  WARNING: tolerated false positives on clean traces: {flagged}")

    if model:
        assert run["recorded_models"], "server kept no record of the run's model"
        assert set(run["recorded_models"]) == {model}, (
            f"requested model {model!r}, server recorded {run['recorded_models']}"
        )
        print(f"  server-side readback confirms model={model!r}")

    LAST_RUN.write_text(
        json.dumps(
            {
                "model": model,
                "recorded_models": run["recorded_models"],
                "concurrency": run["concurrency"],
                "seconds": round(run["seconds"], 1),
                "batches": run["batches"],
                "seconds_per_batch": round(run["seconds_per_batch"], 1),
                "board": payload,
                "recall": recall,
            },
            indent=2,
        )
    )
    print(f"\nOK — schema-valid Issueboard(source='engine_predicted'); board written to {LAST_RUN}")
    return {
        "board": board,
        "recall": recall,
        "recorded_models": run["recorded_models"],
        "seconds": run["seconds"],
        "seconds_per_batch": run["seconds_per_batch"],
        "batches": run["batches"],
    }


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else None
    concurrency = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CONCURRENCY
    smoke(model, concurrency)
    banner("GATE 2 PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
