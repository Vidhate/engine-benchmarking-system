"""Regression: score() must use the TRUE trace universe (control/clean traces
included) for kappa, not the union of trace_ids appearing in occurrences.

Kappa depends on n = len(universe) via TN/po/pe. Deriving the universe from
occurrences alone silently drops every trace with zero occurrences in both
boards — i.e. every control/clean trace, the majority of any real dataset at
a realistic control_fraction. That collapses kappa's whole point: correcting
for chance agreement under low prevalence.
"""

import pytest

from benchmark.schemas import EngineConfig, Issue, Issueboard, IssueOccurrence, ScoringConfig
from benchmark.scoring.report import score


def make_issue(error_id, category_id, severity="medium"):
    return Issue(
        error_id=error_id, title=error_id, description=error_id,
        category_id=category_id, severity=severity,
    )


def test_kappa_reflects_the_full_trace_universe_including_clean_traces():
    # 100 traces total. K1 injected on t1, t2. P1 predicts t1, t2 correctly
    # plus a false positive on t3 (no ground-truth injection anywhere on
    # t3). t4..t100 (97 traces) are clean: zero occurrences in either board.
    #
    # Hand-computed for category "retrieval": TP=2, FP=1, FN=0, TN=97, n=100.
    #   precision = 2/3, recall = 1.0, f1 = 0.8
    #   po = (TP+TN)/n = 99/100 = 0.99
    #   gt_pos_rate = 2/100 = 0.02, pred_pos_rate = 3/100 = 0.03
    #   pe = 0.02*0.03 + 0.98*0.97 = 0.0006 + 0.9506 = 0.9512
    #   kappa = (0.99 - 0.9512) / (1 - 0.9512) = 0.0388 / 0.0488 = 97/122
    k1 = make_issue("K1", "retrieval")
    gt = Issueboard(
        source="ground_truth",
        issues=[k1],
        occurrences=[
            IssueOccurrence(error_id="K1", trace_id="t1"),
            IssueOccurrence(error_id="K1", trace_id="t2"),
        ],
    )
    p1 = make_issue("P1", "retrieval")
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1],
        occurrences=[
            IssueOccurrence(error_id="P1", trace_id="t1"),
            IssueOccurrence(error_id="P1", trace_id="t2"),
            IssueOccurrence(error_id="P1", trace_id="t3"),
        ],
    )
    trace_ids = [f"t{i}" for i in range(1, 101)]  # the TRUE universe: 100 traces
    cfg = ScoringConfig(description_mode="similarity")

    report = score(
        gt, pred, cfg, base_rates={}, trace_ids=trace_ids,
        engine_config=EngineConfig(model="test-model"),
    )

    by_cat = {s.category_id: s for s in report.category_scores}
    retrieval = by_cat["retrieval"]
    assert retrieval.precision == pytest.approx(2 / 3)
    assert retrieval.recall == pytest.approx(1.0)
    assert retrieval.f1 == pytest.approx(0.8)
    assert retrieval.cohens_kappa == pytest.approx(97 / 122)
    assert retrieval.support == 2


def test_score_requires_trace_ids_it_does_not_silently_derive_from_occurrences():
    # A trace-less occurrence-derived universe would give kappa=0.0 here (the
    # bug this regression guards against); calling score() without trace_ids
    # must fail loudly instead of silently falling back to that universe.
    k1 = make_issue("K1", "retrieval")
    gt = Issueboard(
        source="ground_truth", issues=[k1],
        occurrences=[IssueOccurrence(error_id="K1", trace_id="t1"),
                     IssueOccurrence(error_id="K1", trace_id="t2")],
    )
    p1 = make_issue("P1", "retrieval")
    pred = Issueboard(
        source="engine_predicted", issues=[p1],
        occurrences=[IssueOccurrence(error_id="P1", trace_id="t1"),
                     IssueOccurrence(error_id="P1", trace_id="t2"),
                     IssueOccurrence(error_id="P1", trace_id="t3")],
    )
    cfg = ScoringConfig(description_mode="similarity")
    with pytest.raises(TypeError):
        score(gt, pred, cfg, base_rates={})
