"""Shared helper for the smoke scripts: read the black-box contract, nothing else.

These scripts stand in for the benchmark harness, so they are held to the same
rule: no imports from `target_app`. Everything they know about the app comes
from `configs/target_app.yaml` and the LangGraph Server API.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "target_app.yaml"

# Same .env the served app loads via langgraph.json.
load_dotenv(REPO_ROOT / ".env")


def load_contract() -> dict:
    """Parse configs/target_app.yaml through the Phase-0 TargetAppConfig model."""
    sys.path.insert(0, str(REPO_ROOT))
    from benchmark.schemas.configs import TargetAppConfig  # noqa: PLC0415

    config = TargetAppConfig(**_parse_yaml(CONFIG_PATH.read_text()))
    return config.model_dump()


def _parse_yaml(text: str) -> dict:
    """Tiny loader for this one flat file (pyyaml is not a benchmark dependency)."""
    root: dict = {}
    section: dict | None = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indented = line.startswith(("  ", "\t"))
        key, _, value = line.strip().partition(":")
        value = value.strip().strip('"')
        target = section if indented and section is not None else root
        if value == "":
            section = {}
            root[key.strip()] = section
        else:
            target[key.strip()] = int(value) if re.fullmatch(r"-?\d+", value) else value
    return root


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
