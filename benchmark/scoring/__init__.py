"""Scoring — Stage V (docs/architecture/06-scoring.md).

Two-layer error matcher (exact-key resolution + argmax-overlap pairing) plus
four scorers (category detection, per-error localization, severity, and
description deviation), assembled into a BenchmarkReport by `score()`.
"""

from benchmark.scoring.matcher import compute_fallback_rate, pair_issues, resolve_occurrences
from benchmark.scoring.report import score
from benchmark.scoring.scorer_categories import score_categories
from benchmark.scoring.scorer_description import (
    DescriptionJudge,
    OpenAIDescriptionJudge,
    score_descriptions,
)
from benchmark.scoring.scorer_per_error import score_per_error
from benchmark.scoring.scorer_severity import score_severity

__all__ = [
    "DescriptionJudge",
    "OpenAIDescriptionJudge",
    "compute_fallback_rate",
    "pair_issues",
    "resolve_occurrences",
    "score",
    "score_categories",
    "score_descriptions",
    "score_per_error",
    "score_severity",
]
