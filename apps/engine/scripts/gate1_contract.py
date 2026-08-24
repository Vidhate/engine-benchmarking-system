"""Gate 1 — the config contract and the output schema, offline.

Two things the benchmark side needs to be true before any live run matters:
  * `configs/engine.yaml` parses into `benchmark.schemas.configs.EngineAppConfig`
    and describes the app that is actually served (`langgraph.json`);
  * a board produced by the Engine's own consolidation code validates against
    `benchmark.schemas.issues.Issueboard` with no translation step.

This lives in `scripts/` rather than `tests/` because it imports `benchmark.*`,
which app code and app tests must never do.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _contract import (  # noqa: E402
    APP_DIR,
    CONFIG_PATH,
    banner,
    load_categories,
    load_contract,
    load_seed_board,
    summarize,
    validate_board,
)

PRODUCER = Path(__file__).resolve().parent / "_produce_board.py"


def check_contract() -> dict:
    banner("GATE 1a — configs/engine.yaml -> EngineAppConfig")
    contract = load_contract()
    print(json.dumps(contract, indent=2))

    served = json.loads((APP_DIR / "langgraph.json").read_text())
    assistant = contract["assistant_id"]
    assert assistant in served["graphs"], (
        f"{CONFIG_PATH} declares assistant_id={assistant!r}, "
        f"but langgraph.json serves {sorted(served['graphs'])}"
    )
    assert contract["base_url"].endswith(":2025"), "the Engine is served on port 2025"
    assert contract["model_configurable_key"] == "model"
    print(f"\nOK — assistant {assistant!r} is served by langgraph.json")
    return contract


def check_vocabulary() -> list[dict]:
    banner("GATE 1b — category vocabulary is names + descriptions only")
    categories = load_categories()
    for category in categories:
        assert set(category) == {"category_id", "name", "description"}, category
    ids = [c["category_id"] for c in categories]
    assert "other" in ids, "the escape-hatch category must always be present"
    print(json.dumps(ids, indent=2))
    print(f"\nOK — {len(categories)} categories, escape hatch present")
    return categories


def check_output_shape(categories: list[dict]) -> None:
    """Run the Engine's consolidation offline and validate the board it emits.

    The clustering decision is stubbed; the board assembly, the seed merge and
    the serialization are the real code paths, which is what the schema
    contract is about.
    """
    banner("GATE 1c — Engine board -> benchmark.schemas.issues.Issueboard")
    # The producer runs in its own process: it imports `engine`, this script
    # imports `benchmark`, and neither imports the other.
    completed = subprocess.run(
        [sys.executable, str(PRODUCER), json.dumps(load_seed_board()), json.dumps(categories)],
        capture_output=True,
        text=True,
        cwd=APP_DIR,
        check=False,
    )
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
        raise SystemExit("failed to produce a board")

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    # Asserted on the Engine's own payload, not on the round-tripped model:
    # `benchmark.schemas.issues.Issue` declares `injection_mode` itself (default
    # None), so re-serializing through it emits the key regardless.
    assert "injection_mode" not in json.dumps(payload), "Engine output names an ablation field"

    board = validate_board(payload)
    print(json.dumps(summarize(board), indent=2))
    assert len(board.issues) == 3, "seed(2) + one new issue"
    assert board.issues[0].error_id == "seed-tool-failure-hidden"
    assert len(board.occurrences) == 2
    assert all(issue.injection_mode is None for issue in board.issues)
    print("\nOK — board validates as Issueboard(source='engine_predicted')")


def main() -> int:
    contract = check_contract()
    categories = check_vocabulary()
    check_output_shape(categories)
    banner("GATE 1 PASSED")
    print(f"assistant_id={contract['assistant_id']} base_url={contract['base_url']}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
