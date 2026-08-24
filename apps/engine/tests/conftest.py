"""Unit tests never touch the network: no OpenAI key, no tracing exporter.

Every LLM-dependent code path takes a chat model as an argument, so tests pass
a `FakeChatModel` and assert on the deterministic code around it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.setdefault("OPENAI_API_KEY", "sk-unit-test-not-a-real-key")

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def traces_file() -> Path:
    return FIXTURES / "traces.json"


@pytest.fixture
def extra_fields_file() -> Path:
    return FIXTURES / "traces_with_extra_fields.json"


@pytest.fixture
def seed_board_payload() -> dict:
    return json.loads((FIXTURES / "seed_issueboard.json").read_text())


@pytest.fixture
def index(traces_file):
    from engine.traces import TraceIndex

    return TraceIndex.from_file(traces_file)


@pytest.fixture
def categories() -> list:
    from engine.models import Category

    return [
        Category(category_id=cid, name=cid, description=f"{cid} description")
        for cid in (
            "hallucination",
            "retrieval_failure",
            "tool_misuse",
            "instruction_violation",
            "formatting",
            "state_loss",
            "other",
        )
    ]
