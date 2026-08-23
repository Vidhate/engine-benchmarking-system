"""Scorer 3 — asymmetric severity loss over Layer-2 pairs
(docs/architecture/06-scoring.md).

ordinal: low=0, medium=1, high=2
loss = (actual - predicted)^2   if predicted < actual   (under-prediction, quadratic)
     = alpha * (predicted - actual)                      (over-prediction, linear slope)
"""

import pytest

from benchmark.schemas import ErrorMatch, Issue, Issueboard
from benchmark.scoring.scorer_severity import score_severity


def make_issue(error_id, severity, category_id="retrieval"):
    return Issue(
        error_id=error_id, title=error_id, description=error_id,
        category_id=category_id, severity=severity,
    )


def test_exact_severity_match_has_zero_loss():
    gt = Issueboard(source="ground_truth", issues=[make_issue("K1", "medium")])
    pred = Issueboard(source="engine_predicted", issues=[make_issue("P1", "medium")])
    matches = [ErrorMatch(predicted_error_id="P1", matched_error_id="K1", overlap=1)]
    assert score_severity(matches, gt, pred, alpha=0.5) == 0.0


def test_under_prediction_is_quadratic_and_worse_than_symmetric_over_prediction():
    # Same ordinal distance (2), opposite direction.
    gt_under = Issueboard(source="ground_truth", issues=[make_issue("K1", "high")])
    pred_under = Issueboard(source="engine_predicted", issues=[make_issue("P1", "low")])
    matches_under = [ErrorMatch(predicted_error_id="P1", matched_error_id="K1", overlap=1)]
    under_loss = score_severity(matches_under, gt_under, pred_under, alpha=0.5)

    gt_over = Issueboard(source="ground_truth", issues=[make_issue("K1", "low")])
    pred_over = Issueboard(source="engine_predicted", issues=[make_issue("P1", "high")])
    matches_over = [ErrorMatch(predicted_error_id="P1", matched_error_id="K1", overlap=1)]
    over_loss = score_severity(matches_over, gt_over, pred_over, alpha=0.5)

    assert under_loss == pytest.approx(4.0)  # (2-0)^2
    assert over_loss == pytest.approx(1.0)  # 0.5 * (2-0)
    assert under_loss > over_loss


def test_mean_over_matched_pairs_only_unmatched_excluded():
    gt = Issueboard(
        source="ground_truth",
        issues=[make_issue("K1", "high"), make_issue("K2", "low")],
    )
    pred = Issueboard(
        source="engine_predicted",
        issues=[make_issue("P1", "medium"), make_issue("P2", "medium")],
    )
    matches = [
        ErrorMatch(predicted_error_id="P1", matched_error_id="K1", overlap=1),  # under: (2-1)^2=1
        ErrorMatch(predicted_error_id="P2", matched_error_id=None, overlap=0),  # excluded
    ]
    assert score_severity(matches, gt, pred, alpha=0.5) == pytest.approx(1.0)


def test_no_matched_pairs_returns_zero():
    gt = Issueboard(source="ground_truth", issues=[])
    pred = Issueboard(source="engine_predicted", issues=[])
    matches = [ErrorMatch(predicted_error_id="P1", matched_error_id=None, overlap=0)]
    assert score_severity(matches, gt, pred, alpha=0.5) == 0.0
