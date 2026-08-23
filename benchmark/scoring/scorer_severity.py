"""Scorer 3 — asymmetric severity loss over Layer-2 pairs
(docs/architecture/06-scoring.md).

Under-predicting severity is worse than over-predicting (a missed high-severity
issue costs more than a false alarm):

    loss(actual, predicted) = (actual - predicted)^2   if predicted < actual  (quadratic)
                             = alpha * (predicted - actual)                   (linear, alpha < 1)
    ordinal: low=0, medium=1, high=2
"""

from __future__ import annotations

from benchmark.schemas import ErrorMatch, Issueboard

_SEVERITY_ORDINAL = {"low": 0, "medium": 1, "high": 2}


def _severity_loss(actual: int, predicted: int, alpha: float) -> float:
    if predicted < actual:
        return (actual - predicted) ** 2
    return alpha * (predicted - actual)


def score_severity(
    matches: list[ErrorMatch], gt: Issueboard, pred: Issueboard, alpha: float = 0.5
) -> float:
    """Mean asymmetric severity loss over matched Layer-2 pairs. 0.0 = perfect
    (or no matched pairs to compare)."""
    gt_severity = {issue.error_id: _SEVERITY_ORDINAL[issue.severity] for issue in gt.issues}
    pred_severity = {issue.error_id: _SEVERITY_ORDINAL[issue.severity] for issue in pred.issues}

    losses = []
    for m in matches:
        if m.matched_error_id is None:
            continue
        actual = gt_severity.get(m.matched_error_id)
        predicted = pred_severity.get(m.predicted_error_id)
        if actual is None or predicted is None:
            continue
        losses.append(_severity_loss(actual, predicted, alpha))

    if not losses:
        return 0.0
    return sum(losses) / len(losses)
