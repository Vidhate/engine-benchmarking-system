"""Deterministic BM25-style retrieval over the checked-in corpus."""

import pytest

from target_app.corpus import load_corpus
from target_app.retriever import Retriever, tokenize


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_corpus())


def test_tokenize_lowercases_splits_and_drops_stopwords():
    assert tokenize("The Refund Policy, for a 30-day window!") == [
        "refund",
        "policy",
        "30",
        "day",
        "window",
    ]


def test_tokenize_is_stable_for_equivalent_inputs():
    assert tokenize("Sync is STUCK") == tokenize("  sync   is  stuck.  ")


@pytest.mark.parametrize(
    ("query", "expected_doc_id"),
    [
        ("how do I request a refund", "refund-policy"),
        ("how much does the team plan cost per seat", "pricing-plans"),
        ("my notebook is stuck syncing with a spinner", "troubleshooting-sync"),
        ("I forgot my password and my account is locked", "troubleshooting-login"),
        ("can I edit notes on a plane without wifi", "offline-mode"),
        ("export all my notes as markdown", "data-export"),
        ("is there a linux desktop app", "supported-platforms"),
        ("restore a page I deleted last week", "version-history"),
        ("is my data encrypted", "security-and-privacy"),
        ("how do I share a page with a guest", "team-collaboration"),
        ("my card was declined, where is my invoice", "billing-and-invoices"),
    ],
)
def test_relevant_document_ranks_first(retriever, query, expected_doc_id):
    results = retriever.search(query, k=3)
    assert results[0].doc_id == expected_doc_id


BROAD_QUERY = "notes account plan page sync export refund team"


def test_search_respects_k(retriever):
    assert len(retriever.search(BROAD_QUERY, k=1)) == 1
    assert len(retriever.search(BROAD_QUERY, k=5)) == 5


def test_zero_scoring_documents_are_never_returned(retriever):
    """A narrow query must not pad its result list with unrelated docs."""
    results = retriever.search("refund", k=10)
    assert 0 < len(results) < 10
    assert all(r.score > 0 for r in results)


def test_scores_are_descending_and_positive(retriever):
    results = retriever.search("refund policy billing", k=4)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0


def test_search_is_deterministic(retriever):
    a = retriever.search("two factor authentication code rejected", k=3)
    b = retriever.search("two factor authentication code rejected", k=3)
    assert [(r.doc_id, round(r.score, 9)) for r in a] == [
        (r.doc_id, round(r.score, 9)) for r in b
    ]


def test_query_with_no_signal_returns_nothing(retriever):
    assert retriever.search("the and of a", k=3) == []
    assert retriever.search("", k=3) == []


def test_worst_matches_are_disjoint_from_best_matches(retriever):
    """The irrelevant-docs shim relies on this being a real inversion."""
    query = "refund policy money back"
    best = {r.doc_id for r in retriever.search(query, k=3)}
    worst = {r.doc_id for r in retriever.worst_matches(query, k=3)}
    assert not (best & worst)
    assert "refund-policy" in best


def test_results_carry_title_and_updated_date(retriever):
    top = retriever.search("refund", k=1)[0]
    assert top.title
    assert top.updated
    assert "30 days" in top.text
