"""P/R/F1 + hand-computed Cohen's kappa over a binary trace matrix."""

import pytest

from benchmark.scoring.metrics import binary_prf1_kappa


def test_perfect_agreement_gives_kappa_one():
    universe = [f"t{i}" for i in range(1, 11)]
    gt_positive = {"t1"}
    pred_positive = {"t1"}
    result = binary_prf1_kappa(gt_positive, pred_positive, universe)
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.cohens_kappa == 1.0
    assert result.support == 1


def test_low_prevalence_hand_computed_kappa():
    # 10-trace universe, 1 true positive; predicted set adds one false positive.
    # TP=1 FP=1 FN=0 TN=8
    # po = (TP+TN)/N = 9/10 = 0.9
    # pe = (1/10 * 2/10) + (9/10 * 8/10) = 0.02 + 0.72 = 0.74
    # kappa = (0.9 - 0.74) / (1 - 0.74) = 0.16/0.26
    universe = [f"t{i}" for i in range(1, 11)]
    gt_positive = {"t1"}
    pred_positive = {"t1", "t2"}
    result = binary_prf1_kappa(gt_positive, pred_positive, universe)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(2 / 3)
    assert result.cohens_kappa == pytest.approx(0.16 / 0.26)
    assert result.support == 1


def test_no_positives_at_all_is_defined_not_crashing():
    universe = ["t1", "t2"]
    result = binary_prf1_kappa(set(), set(), universe)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.cohens_kappa == 1.0  # po == pe == 1, perfect trivial agreement
    assert result.support == 0


def test_no_recall_when_all_positives_missed():
    universe = ["t1", "t2", "t3"]
    result = binary_prf1_kappa({"t1"}, set(), universe)
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0
    assert result.support == 1
