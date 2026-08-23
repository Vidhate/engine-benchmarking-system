"""Golden BenchmarkReport test — a small, fully hand-computed E_K/E_P pair
asserted field-by-field (docs/architecture/06-scoring.md, execution-plan.md
Phase 1 gate).

Fixture, by design:
- K1 (retrieval, high) injected on t1, t2. P1 (retrieval, medium) predicts
  both exactly, with a description IDENTICAL to K1's (so TF-IDF similarity
  is exactly 1.0 — a proven property from test_tfidf.py — keeping this test
  hand-computable without re-deriving TF-IDF weights by hand). Severity
  mismatch (medium vs high) exercises the asymmetric loss. P1 ALSO fires a
  spurious retrieval occurrence on t5 (no ground-truth injection there at
  all) — a same-category false positive that keeps Scorer 1's "retrieval"
  P/R/F1/kappa non-trivial (a plain exact match gives po=1, which forces
  kappa=1 regardless of the trace universe and would silently hide a
  universe-derivation bug).
- K2 (formatting, low) injected on t3. P2 (formatting, low) predicts it
  exactly, identical description, identical severity -> zero severity loss,
  perfect description score.
- P3 ("other", medium) fires on t4, which has no ground-truth injection at
  all -> straight to the FP/E_h-candidate pool, no fallback attempted.
- t6 is a CLEAN trace: zero occurrences in either board. It must still count
  in the trace universe passed to score() — dropping it (the old
  occurrence-derived-universe bug) changes retrieval's kappa from 2/3 to
  something else, so this fixture's kappa value only comes out right when
  the true universe (t1..t6, n=6) is threaded through correctly.
"""

import pytest

from benchmark.schemas import (
    EngineConfig,
    ErrorMatch,
    Issue,
    Issueboard,
    IssueOccurrence,
    OccurrenceMatch,
    ScoringConfig,
)
from benchmark.schemas.issues import OTHER_CATEGORY_ID
from benchmark.scoring.report import score

K1_DESC = "the retriever returned stale outdated documents to the user"
K2_DESC = "response used the wrong currency symbol"

TRACE_IDS = ["t1", "t2", "t3", "t4", "t5", "t6"]  # t6 is clean: zero occurrences anywhere


def build_boards():
    k1 = Issue(
        error_id="K1", title="stale docs", description=K1_DESC,
        category_id="retrieval", severity="high",
    )
    k2 = Issue(
        error_id="K2", title="bad currency", description=K2_DESC,
        category_id="formatting", severity="low",
    )
    gt = Issueboard(
        source="ground_truth",
        issues=[k1, k2],
        occurrences=[
            IssueOccurrence(error_id="K1", trace_id="t1"),
            IssueOccurrence(error_id="K1", trace_id="t2"),
            IssueOccurrence(error_id="K2", trace_id="t3"),
        ],
    )

    p1 = Issue(
        error_id="P1", title="stale docs found", description=K1_DESC,
        category_id="retrieval", severity="medium",
    )
    p2 = Issue(
        error_id="P2", title="currency issue", description=K2_DESC,
        category_id="formatting", severity="low",
    )
    p3 = Issue(
        error_id="P3", title="weird issue", description="something novel happened",
        category_id=OTHER_CATEGORY_ID, severity="medium",
    )
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1, p2, p3],
        occurrences=[
            IssueOccurrence(error_id="P1", trace_id="t1"),
            IssueOccurrence(error_id="P1", trace_id="t2"),
            IssueOccurrence(error_id="P1", trace_id="t5"),  # FP: no GT injection on t5 at all
            IssueOccurrence(error_id="P2", trace_id="t3"),
            IssueOccurrence(error_id="P3", trace_id="t4"),
        ],
    )
    return gt, pred


def test_golden_report_field_by_field():
    gt, pred = build_boards()
    cfg = ScoringConfig(severity_alpha=0.5, description_mode="similarity")
    base_rates = {"control_fraction": 0.3, "per_error_injection_counts": {"K1": 2, "K2": 1}}
    engine_config = EngineConfig(model="test-model")

    report = score(
        gt, pred, cfg, base_rates, trace_ids=TRACE_IDS, engine_config=engine_config
    )

    # --- identity / passthrough ---
    assert report.report_id  # content-hash stamped, non-empty
    assert report.engine_config == engine_config
    assert report.base_rates == base_rates

    # --- Layer 1: occurrence resolution ---
    assert len(report.occurrence_matches) == 5
    by_key = {(m.trace_id, m.predicted_error_id): m for m in report.occurrence_matches}
    assert by_key[("t1", "P1")] == OccurrenceMatch(
        trace_id="t1", predicted_error_id="P1", resolved_error_id="K1", method="exact_key"
    )
    assert by_key[("t2", "P1")] == OccurrenceMatch(
        trace_id="t2", predicted_error_id="P1", resolved_error_id="K1", method="exact_key"
    )
    assert by_key[("t5", "P1")] == OccurrenceMatch(
        trace_id="t5", predicted_error_id="P1", resolved_error_id=None, method="exact_key"
    )
    assert by_key[("t3", "P2")] == OccurrenceMatch(
        trace_id="t3", predicted_error_id="P2", resolved_error_id="K2", method="exact_key"
    )
    assert by_key[("t4", "P3")] == OccurrenceMatch(
        trace_id="t4", predicted_error_id="P3", resolved_error_id=None, method="exact_key"
    )
    assert report.matcher_fallback_rate == 0.0

    # --- Layer 2: issue pairing ---
    # P1's t5 occurrence never resolved (no key, no fallback candidate on t5),
    # so it doesn't affect the argmax: P1 still pairs to K1 on overlap=2.
    assert report.matches == [
        ErrorMatch(predicted_error_id="P1", matched_error_id="K1", overlap=2),
        ErrorMatch(predicted_error_id="P2", matched_error_id="K2", overlap=1),
        ErrorMatch(predicted_error_id="P3", matched_error_id=None, overlap=0),
    ]
    assert report.eh_candidates == ["P3"]

    # --- Scorer 1: category-level (n=6 traces, including clean t6) ---
    # retrieval: gt_pos={t1,t2}, pred_pos={t1,t2,t5} -> TP=2,FP=1,FN=0,TN=3
    #   precision=2/3, recall=1, f1=0.8
    #   po=5/6, pe=1/2 -> kappa=(5/6-1/2)/(1/2)=2/3
    by_cat = {s.category_id: s for s in report.category_scores}
    assert set(by_cat) == {"retrieval", "formatting", "other"}
    assert by_cat["retrieval"].precision == pytest.approx(2 / 3)
    assert by_cat["retrieval"].recall == pytest.approx(1.0)
    assert by_cat["retrieval"].f1 == pytest.approx(0.8)
    assert by_cat["retrieval"].cohens_kappa == pytest.approx(2 / 3)
    assert by_cat["retrieval"].support == 2
    assert by_cat["formatting"].precision == 1.0
    assert by_cat["formatting"].recall == 1.0
    assert by_cat["formatting"].f1 == 1.0
    assert by_cat["formatting"].cohens_kappa == pytest.approx(1.0)
    assert by_cat["formatting"].support == 1
    assert by_cat["other"].precision == 0.0
    assert by_cat["other"].recall == 0.0
    assert by_cat["other"].f1 == 0.0
    assert by_cat["other"].cohens_kappa == pytest.approx(0.0)
    assert by_cat["other"].support == 0

    # --- Scorer 2: per-known-error (independent of Layer 2 pairing) ---
    # K1's resolved traces are {t1,t2} only — P1's unresolved t5 occurrence
    # never attaches to K1, so K1's own P/R/F1/kappa stay perfect.
    by_err = {s.category_id: s for s in report.per_error_scores}
    assert set(by_err) == {"K1", "K2"}
    assert by_err["K1"].precision == 1.0
    assert by_err["K1"].recall == 1.0
    assert by_err["K1"].f1 == 1.0
    assert by_err["K1"].cohens_kappa == pytest.approx(1.0)
    assert by_err["K1"].support == 2
    assert by_err["K2"].precision == 1.0
    assert by_err["K2"].recall == 1.0
    assert by_err["K2"].f1 == 1.0
    assert by_err["K2"].support == 1

    # --- Scorer 3: severity ---
    # P1 under-predicts K1 (medium=1 vs high=2): (2-1)^2 = 1
    # P2 matches K2 exactly (low vs low): alpha * 0 = 0
    # mean over matched pairs = 0.5
    assert report.severity_loss == pytest.approx(0.5)

    # --- Scorer 4: description (similarity mode, identical text -> 1.0) ---
    assert report.description_scores == {"K1": pytest.approx(1.0), "K2": pytest.approx(1.0)}

    # --- Headline: separate keys, no composite scalar ---
    # category macros average over [formatting, other, retrieval] (sorted):
    #   precision: (1 + 0 + 2/3)/3 = 5/9 ; recall: (1 + 0 + 1)/3 = 2/3
    #   f1: (1 + 0 + 0.8)/3 = 0.6
    assert report.headline["category_precision_macro"] == pytest.approx(5 / 9)
    assert report.headline["category_recall_macro"] == pytest.approx(2 / 3)
    assert report.headline["category_f1_macro"] == pytest.approx(0.6)
    assert report.headline["matched_error_f1_macro"] == pytest.approx(1.0)
    assert report.headline["mean_severity_loss"] == pytest.approx(0.5)
    assert report.headline["mean_description_score"] == pytest.approx(1.0)
    assert "composite" not in report.headline
    assert "score" not in report.headline


def test_report_is_deterministic_content_hash():
    gt, pred = build_boards()
    cfg = ScoringConfig(description_mode="similarity")
    base_rates = {"control_fraction": 0.3}
    r1 = score(gt, pred, cfg, base_rates, trace_ids=TRACE_IDS)
    r2 = score(gt, pred, cfg, base_rates, trace_ids=TRACE_IDS)
    assert r1.report_id == r2.report_id
