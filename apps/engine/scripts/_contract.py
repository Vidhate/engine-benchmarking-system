"""Shared helper for the smoke scripts: the black-box contract, nothing else.

These scripts stand in for the benchmark harness, so they are held to the same
rule the harness is: no imports from `engine`. Everything they know about the
app comes from `configs/engine.yaml` and the LangGraph Server API.

They MAY import `benchmark.schemas` — that is the benchmark side of the
boundary, and validating the Engine's output against the real `Issueboard`
model is precisely what these scripts exist to prove. (Same ruling as Phase 2's
`apps/target_app/scripts/_contract.py`.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_DIR.parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "engine.yaml"
FIXTURES = APP_DIR / "tests" / "fixtures"
TAXONOMY_PATH = REPO_ROOT / "configs" / "taxonomy.yaml"

# Same .env the served app loads via langgraph.json.
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(REPO_ROOT))


def load_contract() -> dict[str, Any]:
    """Parse configs/engine.yaml through the Phase-0 EngineAppConfig model."""
    from benchmark.schemas.configs import EngineAppConfig  # noqa: PLC0415

    return EngineAppConfig(**yaml.safe_load(CONFIG_PATH.read_text())).model_dump()


def load_categories() -> list[dict[str, str]]:
    """The public category vocabulary: names + descriptions only.

    This is the whole taxonomy surface the Engine is allowed to see — concrete
    injected error definitions never appear in `configs/taxonomy.yaml`.
    """
    from benchmark.schemas.issues import ErrorCategory  # noqa: PLC0415

    raw = yaml.safe_load(TAXONOMY_PATH.read_text())["categories"]
    return [ErrorCategory(**item).model_dump() for item in raw]


def load_seed_board() -> dict[str, Any]:
    return json.loads((FIXTURES / "seed_issueboard.json").read_text())


def validate_board(payload: dict[str, Any]):
    """Validate a run output against the real Phase-0 Issueboard model."""
    from benchmark.schemas.issues import Issueboard  # noqa: PLC0415

    board = Issueboard.model_validate(payload)
    assert board.source == "engine_predicted", f"source is {board.source!r}"
    known = {issue.error_id for issue in board.issues}
    dangling = sorted({o.error_id for o in board.occurrences} - known)
    assert not dangling, f"occurrences reference unknown issues: {dangling}"
    return board


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def summarize(board) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for occurrence in board.occurrences:
        counts[occurrence.error_id] = counts.get(occurrence.error_id, 0) + 1
    return {
        "board_id": board.board_id,
        "source": board.source,
        "issue_count": len(board.issues),
        "occurrence_count": len(board.occurrences),
        "issues": [
            {
                "error_id": issue.error_id,
                "title": issue.title,
                "category_id": issue.category_id,
                "severity": issue.severity,
                "occurrences": counts.get(issue.error_id, 0),
            }
            for issue in board.issues
        ],
    }
