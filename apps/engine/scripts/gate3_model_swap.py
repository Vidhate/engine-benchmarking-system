"""Gate 3 — the same Engine, twice, with only the model id changed.

This is the assignment's comparison axis. Everything except
`config.configurable["model"]` is held constant: same traces, same seed board,
same vocabulary, same prompts, same code. If a board delta needs a code change
to explain it, the comparison is not measuring the model.

    uv run python scripts/gate3_model_swap.py [large_model] [mini_model]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _contract import banner  # noqa: E402
from gate2_live_smoke import smoke  # noqa: E402

# The two arms. Deliberately named here rather than read from the app: the
# benchmark chooses the models, the Engine only obeys.
DEFAULT_ARMS = ("gpt-5.1", "gpt-5-mini")
RESULTS = Path(__file__).resolve().parent / ".model_swap.json"


def main() -> int:
    arms = tuple(sys.argv[1:3]) or DEFAULT_ARMS
    if len(arms) != 2:
        raise SystemExit("usage: gate3_model_swap.py [large_model] [mini_model]")

    results = {}
    for model in arms:
        outcome = smoke(model)
        board = outcome["board"]
        # Verified server-side inside smoke(): the persisted run record names
        # the requested model. Repeated here so the failure message names the arm.
        assert set(outcome["recorded_models"]) == {model}, (
            f"arm {model!r} was not honored; server recorded {outcome['recorded_models']}"
        )
        results[model] = {
            "recorded_models": outcome["recorded_models"],
            "seconds": round(outcome["seconds"], 1),
            "board_id": board.board_id,
            "issue_count": len(board.issues),
            "occurrence_count": len(board.occurrences),
            "new_issues": [
                {"error_id": i.error_id, "title": i.title, "category_id": i.category_id,
                 "severity": i.severity}
                for i in board.issues
                if not i.error_id.startswith("seed-")
            ],
            "planted_found": outcome["recall"]["planted_found"],
            "category_matches": outcome["recall"]["category_matches"],
            "clean_traces_flagged": outcome["recall"]["clean_traces_flagged"],
            "occurrences_by_trace": outcome["recall"]["occurrences_by_trace"],
        }

    banner("GATE 3 — model swap via run config only")
    print(json.dumps(results, indent=2))
    RESULTS.write_text(json.dumps(results, indent=2) + "\n")

    # Belt and braces: two different models producing a byte-identical board is
    # possible but unlikely, and it is precisely what a silently-ignored model
    # config looks like. The readback above is the real check; this catches the
    # case where the readback itself is reading something stale.
    board_ids = {row["board_id"] for row in results.values()}
    if len(board_ids) == 1:
        raise SystemExit(
            f"both arms produced the identical board {board_ids.pop()}. Either the "
            f"model config was not honored, or the two models genuinely agreed "
            f"issue-for-issue and occurrence-for-occurrence — verify before "
            f"reporting any comparison from this run."
        )

    print("\nside by side")
    print(f"{'model':<16} {'issues':>7} {'occurrences':>12} {'seconds':>9} {'planted':>9}")
    for model, row in results.items():
        print(
            f"{model:<16} {row['issue_count']:>7} {row['occurrence_count']:>12} "
            f"{row['seconds']:>9} {row['planted_found']:>7}/3"
        )

    banner("GATE 3 PASSED — no code changed between the two runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
