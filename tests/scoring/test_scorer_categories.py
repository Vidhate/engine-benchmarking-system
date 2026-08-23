"""Scorer 1 — category-level (trace x category) binary classification.

Matching-free: rewards detecting the right *kind* of problem on the right
traces even when Engine's write-ups don't align 1:1 with the injected
definitions (docs/architecture/06-scoring.md)."""

from benchmark.schemas import Issue, Issueboard, IssueOccurrence
from benchmark.scoring.scorer_categories import score_categories


def make_issue(error_id, category_id, title="issue"):
    return Issue(
        error_id=error_id, title=title, description=title, category_id=category_id,
        severity="medium",
    )


def test_perfect_category_detection_scores_perfectly():
    gt = Issueboard(
        source="ground_truth",
        issues=[make_issue("K1", "retrieval")],
        occurrences=[IssueOccurrence(error_id="K1", trace_id="t1")],
    )
    pred = Issueboard(
        source="engine_predicted",
        issues=[make_issue("P1", "retrieval")],
        occurrences=[IssueOccurrence(error_id="P1", trace_id="t1")],
    )
    scores = score_categories(gt, pred, ["t1", "t2", "t3"])
    by_cat = {s.category_id: s for s in scores}
    assert by_cat["retrieval"].precision == 1.0
    assert by_cat["retrieval"].recall == 1.0
    assert by_cat["retrieval"].f1 == 1.0
    assert by_cat["retrieval"].support == 1


def test_scorer_is_matching_free_credit_given_even_without_error_level_alignment():
    # Different error definitions, same category, same trace -> full credit here
    # even though Layer 2 pairing might not line these up 1:1.
    gt = Issueboard(
        source="ground_truth",
        issues=[make_issue("K1", "retrieval", "stale docs")],
        occurrences=[IssueOccurrence(error_id="K1", trace_id="t1")],
    )
    pred = Issueboard(
        source="engine_predicted",
        issues=[make_issue("P1", "retrieval", "totally different write-up")],
        occurrences=[IssueOccurrence(error_id="P1", trace_id="t1")],
    )
    scores = score_categories(gt, pred, ["t1"])
    by_cat = {s.category_id: s for s in scores}
    assert by_cat["retrieval"].f1 == 1.0


def test_covers_categories_present_in_either_board():
    gt = Issueboard(
        source="ground_truth",
        issues=[make_issue("K1", "retrieval")],
        occurrences=[IssueOccurrence(error_id="K1", trace_id="t1")],
    )
    pred = Issueboard(
        source="engine_predicted",
        issues=[make_issue("P1", "formatting")],
        occurrences=[IssueOccurrence(error_id="P1", trace_id="t2")],
    )
    scores = score_categories(gt, pred, ["t1", "t2"])
    cats = {s.category_id for s in scores}
    assert cats == {"retrieval", "formatting"}
    by_cat = {s.category_id: s for s in scores}
    # retrieval: gt has it, pred doesn't -> recall 0
    assert by_cat["retrieval"].recall == 0.0
    assert by_cat["retrieval"].support == 1
    # formatting: pred has it, gt doesn't -> precision 0, support 0
    assert by_cat["formatting"].precision == 0.0
    assert by_cat["formatting"].support == 0
