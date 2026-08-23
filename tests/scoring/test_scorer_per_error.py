"""Scorer 2 — per-error trace classification from Layer-1 resolutions
(independent of Layer-2 issue pairing) (docs/architecture/06-scoring.md)."""

from benchmark.schemas import Issue, Issueboard, IssueOccurrence, ScoringConfig
from benchmark.scoring.matcher import resolve_occurrences
from benchmark.scoring.scorer_per_error import score_per_error


def make_issue(error_id, category_id, title="issue", description=None):
    return Issue(
        error_id=error_id, title=title, description=description or title,
        category_id=category_id, severity="medium",
    )


def test_per_error_perfect_localization():
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
        ],
    )
    occ_matches = resolve_occurrences(gt, pred, ScoringConfig())
    scores = score_per_error(gt, pred, occ_matches, ["t1", "t2", "t3"])
    by_error = {s.category_id: s for s in scores}
    assert by_error["K1"].precision == 1.0
    assert by_error["K1"].recall == 1.0
    assert by_error["K1"].support == 2


def test_per_error_scoring_uses_layer1_resolution_including_fallback():
    # P1 predicted in the wrong category but fallback-resolves to K1 -> counts
    # toward K1's recall even though Layer-2 pairing (same-category only)
    # would never pair P1 to K1.
    k1 = make_issue(
        "K1", "retrieval", "stale docs",
        description="the retriever returned stale outdated documents",
    )
    gt = Issueboard(
        source="ground_truth",
        issues=[k1],
        occurrences=[IssueOccurrence(error_id="K1", trace_id="t1")],
    )
    p1 = make_issue(
        "P1", "formatting", "stale docs",
        description="the retriever returned stale outdated documents",
    )
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1],
        occurrences=[IssueOccurrence(error_id="P1", trace_id="t1")],
    )
    cfg = ScoringConfig(text_fallback_threshold=0.5)
    occ_matches = resolve_occurrences(gt, pred, cfg)
    scores = score_per_error(gt, pred, occ_matches, ["t1"])
    by_error = {s.category_id: s for s in scores}
    assert by_error["K1"].recall == 1.0
    assert by_error["K1"].precision == 1.0


def test_per_error_missed_known_error_has_zero_recall():
    k1 = make_issue("K1", "retrieval")
    gt = Issueboard(
        source="ground_truth", issues=[k1],
        occurrences=[IssueOccurrence(error_id="K1", trace_id="t1")],
    )
    pred = Issueboard(source="engine_predicted", issues=[], occurrences=[])
    occ_matches = resolve_occurrences(gt, pred, ScoringConfig())
    scores = score_per_error(gt, pred, occ_matches, ["t1"])
    by_error = {s.category_id: s for s in scores}
    assert by_error["K1"].recall == 0.0
    assert by_error["K1"].support == 1
