"""Gate: benchmark/generation/*.py stays app-agnostic.

The generators (and the OpenAI/mock expanders) must never hardcode which
target app they're generating inputs for — that description lives solely in
GenerationConfig.app_context (sourced from the yaml config). This is a
regression guard for the PR #2 review finding: OpenAIPromptExpander used to
hardcode "product-support assistant" in its prompts.
"""

from pathlib import Path

GENERATION_SRC = Path(__file__).parent.parent / "benchmark" / "generation"

# App-specific vocabulary that was previously hardcoded in expander.py's
# prompts, or that belongs to the v0.yaml product-support domain and must
# only ever appear in config data (configs/generation/*.yaml), never in the
# generic generator source.
FORBIDDEN_DOMAIN_TERMS = [
    "product-support",
    "support assistant",
    "refund",
    "ticket",
    "billing",
    "shipping",
    "subscription",
]


def test_no_hardcoded_domain_vocabulary_in_generation_source():
    violations = []
    for path in sorted(GENERATION_SRC.rglob("*.py")):
        text = path.read_text().lower()
        for term in FORBIDDEN_DOMAIN_TERMS:
            if term in text:
                violations.append(f"{path.relative_to(GENERATION_SRC.parent.parent)}: {term!r}")
    assert not violations, (
        "benchmark/generation/*.py must stay app-agnostic — app description "
        "belongs in GenerationConfig.app_context (yaml), not Python source:\n"
        + "\n".join(violations)
    )


def test_scan_actually_sees_generation_source_files():
    # Guard against this test silently passing because the scan found nothing.
    assert any(GENERATION_SRC.rglob("*.py")), "benchmark/generation/ package not found"
