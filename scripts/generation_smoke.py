#!/usr/bin/env python3
"""Live-model smoke test for OpenAIPromptExpander.

NOT run in CI (scripts/ci.sh does not call this). Exercises a handful of
real expansions against the OpenAI API using benchmark.models.GENERATION_MODEL,
so you can eyeball that the live expander produces sane text before trusting
it for a full generation run.

The expander itself is app-agnostic: the target-app description comes
entirely from configs/generation/v0.yaml's `app_context` field, loaded here
via load_generation_config — the yaml is the only control surface for what
app these inputs are being generated for (see benchmark/generation/expander.py).

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

from benchmark.generation.config_loader import load_generation_config  # noqa: E402
from benchmark.generation.expander import OpenAIPromptExpander  # noqa: E402

V0_CONFIG_PATH = REPO_ROOT / "configs" / "generation" / "v0.yaml"


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

    # Pull the target-app description and a sample dim/persona straight from
    # the checked-in yaml config — the expander never hardcodes any of this.
    cfg = load_generation_config(V0_CONFIG_PATH)
    expander = OpenAIPromptExpander()

    safe_dim = cfg.safe_dims[0]
    adversarial_dim = cfg.adversarial_dims[0]
    persona = cfg.personas[0]

    print("=== safe single-turn expansion ===")
    print(expander.expand(safe_dim, safe_dim.variations[0], seed=1, app_context=cfg.app_context))

    print("\n=== adversarial single-turn expansion ===")
    print(
        expander.expand(
            adversarial_dim, adversarial_dim.variations[0], seed=1, app_context=cfg.app_context
        )
    )

    print("\n=== multi-turn scenario brief ===")
    print(
        expander.expand_scenario(
            persona, safe_dim.dim_id, safe_dim.variations[0], seed=1, app_context=cfg.app_context
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
