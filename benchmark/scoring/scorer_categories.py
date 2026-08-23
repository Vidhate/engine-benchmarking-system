"""Scorer 1 — category-level trace classification (docs/architecture/06-scoring.md).

Deliberately matching-free: rewards Engine for flagging the right *kind* of
problem on the right traces even when its write-ups don't align 1:1 with the
injected definitions. Computed straight off occurrences, no error matching.
"""

from __future__ import annotations

from collections import defaultdict

from benchmark.schemas import CategoryScore, Issueboard
from benchmark.scoring.metrics import binary_prf1_kappa


def _category_trace_sets(board: Issueboard) -> dict[str, set[str]]:
    category_by_error: dict[str, str] = {i.error_id: i.category_id for i in board.issues}
    traces: dict[str, set[str]] = defaultdict(set)
    for occ in board.occurrences:
        category_id = category_by_error.get(occ.error_id)
        if category_id is not None:
            traces[category_id].add(occ.trace_id)
    return traces


def score_categories(
    gt: Issueboard, pred: Issueboard, trace_ids: list[str]
) -> list[CategoryScore]:
    """Per-category P/R/F1/kappa over the (trace x category) binary matrix."""
    gt_by_category = _category_trace_sets(gt)
    pred_by_category = _category_trace_sets(pred)
    categories = sorted(set(gt_by_category) | set(pred_by_category))

    scores = []
    for category_id in categories:
        result = binary_prf1_kappa(
            gt_by_category.get(category_id, set()),
            pred_by_category.get(category_id, set()),
            trace_ids,
        )
        scores.append(
            CategoryScore(
                category_id=category_id,
                precision=result.precision,
                recall=result.recall,
                f1=result.f1,
                cohens_kappa=result.cohens_kappa,
                support=result.support,
            )
        )
    return scores
