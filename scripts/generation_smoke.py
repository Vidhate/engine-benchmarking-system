#!/usr/bin/env python3
"""Live-model smoke test for OpenAIPromptExpander.

NOT run in CI (scripts/ci.sh does not call this). Exercises a handful of
real expansions against the OpenAI API using benchmark.models.GENERATION_MODEL,
so you can eyeball that the live expander produces sane text before trusting
it for a full generation run.

Usage:
    # requires OPENAI_API_KEY in the environment or a .env file at repo root
    uv run python scripts/generation_smoke.py

Loads .env from the repo root if present (simple, no new dependency: a
two-line parser, not python-dotenv).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.generation.expander import OpenAIPromptExpander  # noqa: E402
from benchmark.schemas.inputs import Dimension, Persona  # noqa: E402


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    _load_dotenv(REPO_ROOT / ".env")

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY not set — skipping live smoke.\n"
            "Copy .env (with OPENAI_API_KEY set) to the repo root to run this.",
        )
        return 0

    expander = OpenAIPromptExpander()

    safe_dim = Dimension(
        dim_id="topic",
        name="query_topic",
        kind="safe",
        variations=["refund_request"],
    )
    adversarial_dim = Dimension(
        dim_id="jailbreak_persona_override",
        name="jailbreak_persona_override",
        kind="adversarial",
        variations=["dan_style_roleplay_request"],
    )
    persona = Persona(
        persona_id="frustrated_billing_customer",
        name="Frustrated Billing Customer",
        kind="target",
        description="A customer upset about a duplicate charge, wants a refund.",
        goals=["get a refund"],
    )

    print("=== safe single-turn expansion ===")
    print(expander.expand(safe_dim, "refund_request", seed=1))

    print("\n=== adversarial single-turn expansion ===")
    print(expander.expand(adversarial_dim, "dan_style_roleplay_request", seed=1))

    print("\n=== multi-turn scenario brief ===")
    print(expander.expand_scenario(persona, "topic", "refund_request", seed=1))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
