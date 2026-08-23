"""Layer 2 — argmax-overlap issue pairing + the granularity-asymmetry gate
(docs/architecture/06-scoring.md)."""

from benchmark.schemas import Issue, Issueboard, IssueOccurrence, ScoringConfig
from benchmark.scoring.matcher import pair_issues, resolve_occurrences
from benchmark.scoring.scorer_per_error import score_per_error


def make_issue(error_id, category_id, title="issue", description=None, severity="medium"):
    return Issue(
        error_id=error_id,
        title=title,
        description=description or f"{title} description",
        category_id=category_id,
        severity=severity,
    )


def test_two_finer_predicted_issues_both_pair_to_the_one_known_error_no_penalty():
    # Known error K1 was injected on t1..t4. Engine splits it into two predicted
    # issues (P1 on t1,t2; P2 on t3,t4) — finer than known, should be free.
    k1 = make_issue("K1", "retrieval", title="stale docs")
    gt = Issueboard(
        source="ground_truth",
        issues=[k1],
        occurrences=[
            IssueOccurrence(error_id="K1", trace_id=t) for t in ("t1", "t2", "t3", "t4")
        ],
    )
    p1 = make_issue("P1", "retrieval", title="stale docs variant a")
    p2 = make_issue("P2", "retrieval", title="stale docs variant b")
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1, p2],
        occurrences=[
            IssueOccurrence(error_id="P1", trace_id="t1"),
            IssueOccurrence(error_id="P1", trace_id="t2"),
            IssueOccurrence(error_id="P2", trace_id="t3"),
            IssueOccurrence(error_id="P2", trace_id="t4"),
        ],
    )
    cfg = ScoringConfig()
    occ_matches = resolve_occurrences(gt, pred, cfg)
    matches = pair_issues(occ_matches, gt, pred)
    by_pred = {m.predicted_error_id: m for m in matches}
    assert by_pred["P1"].matched_error_id == "K1"
    assert by_pred["P1"].overlap == 2
    assert by_pred["P2"].matched_error_id == "K1"
    assert by_pred["P2"].overlap == 2


def test_coarser_predicted_issue_pairs_only_with_its_majority_partner():
    # K1 injected on 3 traces, K2 on 1 trace, both retrieval category (so K2's
    # trace never overlaps K1's — disjointness). Engine lumps everything into
    # one predicted issue P1 spanning all 4 traces.
    k1 = make_issue("K1", "retrieval", title="stale docs")
    k2 = make_issue("K2", "retrieval", title="empty docs")
    gt = Issueboard(
        source="ground_truth",
        issues=[k1, k2],
        occurrences=[
            IssueOccurrence(error_id="K1", trace_id="t1"),
            IssueOccurrence(error_id="K1", trace_id="t2"),
            IssueOccurrence(error_id="K1", trace_id="t3"),
            IssueOccurrence(error_id="K2", trace_id="t4"),
        ],
    )
    p1 = make_issue("P1", "retrieval", title="retrieval is broken")
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1],
        occurrences=[
            IssueOccurrence(error_id="P1", trace_id="t1"),
            IssueOccurrence(error_id="P1", trace_id="t2"),
            IssueOccurrence(error_id="P1", trace_id="t3"),
            IssueOccurrence(error_id="P1", trace_id="t4"),
        ],
    )
    cfg = ScoringConfig()
    occ_matches = resolve_occurrences(gt, pred, cfg)
    matches = pair_issues(occ_matches, gt, pred)
    assert len(matches) == 1
    m = matches[0]
    assert m.matched_error_id == "K1"  # majority partner (3 traces vs 1)
    assert m.overlap == 3
    assert not m.tie_broken_by_text

    # "Occurrence-level detection credit stays fair throughout" (spec): even
    # though Layer 2 pairing left K2 without an issue-level partner, Layer 1
    # still resolved P1's t4 occurrence to K2 via the exact key, so Scorer 2's
    # per-error recall for the minority known error is untouched by the lump.
    per_error_scores = score_per_error(gt, pred, occ_matches, ["t1", "t2", "t3", "t4"])
    by_error = {s.category_id: s for s in per_error_scores}
    assert by_error["K2"].recall == 1.0
    assert by_error["K2"].precision == 1.0


def test_tie_break_uses_text_similarity_of_descriptions():
    # K1 and K2 each get exactly 1 trace of overlap with P1 -> true tie.
    k1 = make_issue(
        "K1", "retrieval", title="stale docs",
        description="the retriever returned stale outdated documents to the customer",
    )
    k2 = make_issue(
        "K2", "retrieval", title="empty docs",
        description="the tool call failed with an unrelated timeout error",
    )
    gt = Issueboard(
        source="ground_truth",
        issues=[k1, k2],
        occurrences=[
            IssueOccurrence(error_id="K1", trace_id="t1"),
            IssueOccurrence(error_id="K2", trace_id="t2"),
        ],
    )
    p1 = make_issue(
        "P1", "retrieval", title="stale docs",
        description="the retriever returned stale outdated documents to the customer",
    )
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1],
        occurrences=[
            IssueOccurrence(error_id="P1", trace_id="t1"),
            IssueOccurrence(error_id="P1", trace_id="t2"),
        ],
    )
    cfg = ScoringConfig()
    occ_matches = resolve_occurrences(gt, pred, cfg)
    matches = pair_issues(occ_matches, gt, pred)
    m = matches[0]
    assert m.overlap == 1
    assert m.tie_broken_by_text
    assert m.matched_error_id == "K1"  # closer description text wins the tie


def test_predicted_issue_with_no_resolved_occurrences_has_no_match():
    gt = Issueboard(source="ground_truth", issues=[], occurrences=[])
    p1 = make_issue("P1", "retrieval", title="phantom")
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1],
        occurrences=[IssueOccurrence(error_id="P1", trace_id="t1")],
    )
    cfg = ScoringConfig()
    occ_matches = resolve_occurrences(gt, pred, cfg)
    matches = pair_issues(occ_matches, gt, pred)
    assert matches[0].matched_error_id is None
    assert matches[0].overlap == 0
