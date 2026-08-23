"""Layer 1 — exact-key (trace_id, category_id) occurrence resolution + the
wrong-category TF-IDF fallback (docs/architecture/06-scoring.md)."""

from benchmark.schemas import Issue, Issueboard, IssueOccurrence, ScoringConfig
from benchmark.schemas.issues import OTHER_CATEGORY_ID
from benchmark.scoring.matcher import compute_fallback_rate, resolve_occurrences


def make_issue(error_id, category_id, title="issue", description=None, severity="medium"):
    return Issue(
        error_id=error_id,
        title=title,
        description=description or f"{title} description",
        category_id=category_id,
        severity=severity,
    )


def test_exact_key_resolves_to_the_unique_injected_known_error():
    # disjointness invariant: two errors of one category never share a trace,
    # so (trace_id, category_id) uniquely identifies the injected known error.
    k1 = make_issue("K1", "retrieval", title="stale docs")
    k2 = make_issue("K2", "tool_misuse", title="wrong tool")
    gt = Issueboard(
        source="ground_truth",
        issues=[k1, k2],
        occurrences=[
            IssueOccurrence(error_id="K1", trace_id="t1"),
            IssueOccurrence(error_id="K1", trace_id="t2"),
            IssueOccurrence(error_id="K2", trace_id="t1"),
        ],
    )
    p1 = make_issue("P1", "retrieval", title="stale docs found")
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1],
        occurrences=[
            IssueOccurrence(error_id="P1", trace_id="t1"),
            IssueOccurrence(error_id="P1", trace_id="t2"),
        ],
    )
    matches = resolve_occurrences(gt, pred, ScoringConfig())
    assert len(matches) == 2
    for m in matches:
        assert m.predicted_error_id == "P1"
        assert m.method == "exact_key"
    resolved = {m.trace_id: m.resolved_error_id for m in matches}
    assert resolved == {"t1": "K1", "t2": "K1"}


def test_fallback_fires_and_matches_when_same_trace_has_different_category_injection():
    k1 = make_issue(
        "K1", "retrieval", title="stale documents",
        description="the retriever returned stale outdated documents to the user",
    )
    gt = Issueboard(
        source="ground_truth",
        issues=[k1],
        occurrences=[IssueOccurrence(error_id="K1", trace_id="t1")],
    )
    # predicted in the WRONG category ("formatting") but a near-identical write-up.
    p1 = make_issue(
        "P1", "formatting", title="stale documents",
        description="the retriever returned stale outdated documents to the user",
    )
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1],
        occurrences=[IssueOccurrence(error_id="P1", trace_id="t1")],
    )
    cfg = ScoringConfig(text_fallback_threshold=0.5)
    matches = resolve_occurrences(gt, pred, cfg)
    assert len(matches) == 1
    m = matches[0]
    assert m.method == "text_fallback"
    assert m.resolved_error_id == "K1"
    assert compute_fallback_rate(matches) == 1.0


def test_fallback_attempted_but_below_threshold_stays_unresolved():
    k1 = make_issue(
        "K1", "retrieval", title="stale documents",
        description="the retriever returned stale outdated documents",
    )
    gt = Issueboard(
        source="ground_truth",
        issues=[k1],
        occurrences=[IssueOccurrence(error_id="K1", trace_id="t1")],
    )
    p1 = make_issue(
        "P1", "formatting", title="unrelated",
        description="response used the wrong currency symbol entirely",
    )
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1],
        occurrences=[IssueOccurrence(error_id="P1", trace_id="t1")],
    )
    cfg = ScoringConfig(text_fallback_threshold=0.9)
    matches = resolve_occurrences(gt, pred, cfg)
    m = matches[0]
    assert m.method == "text_fallback"  # fallback path was attempted (it "fired")
    assert m.resolved_error_id is None  # but similarity too low to match
    assert compute_fallback_rate(matches) == 1.0


def test_no_key_match_and_no_other_category_injection_goes_to_fp_pool():
    gt = Issueboard(source="ground_truth", issues=[], occurrences=[])
    p1 = make_issue("P1", "retrieval", title="phantom issue")
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1],
        occurrences=[IssueOccurrence(error_id="P1", trace_id="t1")],
    )
    matches = resolve_occurrences(gt, pred, ScoringConfig())
    m = matches[0]
    assert m.resolved_error_id is None
    assert m.method == "exact_key"  # fallback never attempted — no cross-category injection
    assert compute_fallback_rate(matches) == 0.0


def test_other_category_predictions_never_use_fallback():
    k1 = make_issue("K1", "retrieval", title="stale docs")
    gt = Issueboard(
        source="ground_truth",
        issues=[k1],
        occurrences=[IssueOccurrence(error_id="K1", trace_id="t1")],
    )
    p1 = make_issue("P1", OTHER_CATEGORY_ID, title="something novel")
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1],
        occurrences=[IssueOccurrence(error_id="P1", trace_id="t1")],
    )
    matches = resolve_occurrences(gt, pred, ScoringConfig(text_fallback_threshold=0.0))
    m = matches[0]
    assert m.resolved_error_id is None
    assert m.method == "exact_key"
    assert compute_fallback_rate(matches) == 0.0


def test_fallback_rate_is_fraction_of_all_occurrences():
    k1 = make_issue("K1", "retrieval", title="stale docs", description="stale docs retrieval")
    gt = Issueboard(
        source="ground_truth",
        issues=[k1],
        occurrences=[
            IssueOccurrence(error_id="K1", trace_id="t1"),
            IssueOccurrence(error_id="K1", trace_id="t2"),
        ],
    )
    p1 = make_issue("P1", "retrieval", title="stale docs", description="stale docs retrieval")
    p2 = make_issue("P2", "formatting", title="stale docs", description="stale docs retrieval")
    pred = Issueboard(
        source="engine_predicted",
        issues=[p1, p2],
        occurrences=[
            IssueOccurrence(error_id="P1", trace_id="t1"),  # exact key
            IssueOccurrence(error_id="P2", trace_id="t2"),  # wrong category -> fallback
        ],
    )
    matches = resolve_occurrences(gt, pred, ScoringConfig(text_fallback_threshold=0.1))
    assert compute_fallback_rate(matches) == 0.5
