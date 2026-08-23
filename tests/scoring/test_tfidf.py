"""Pure-python TF-IDF text similarity (no runtime deps)."""

from benchmark.scoring.tfidf import text_similarity


def test_identical_text_has_similarity_one():
    text = "the retriever returned stale documents for the refund query"
    assert text_similarity(text, text) == 1.0


def test_disjoint_vocabulary_has_similarity_zero():
    a = "retriever returned stale documents"
    b = "tool call failed with a timeout exception"
    assert text_similarity(a, b) == 0.0


def test_partial_overlap_is_between_zero_and_one():
    a = "the retriever returned stale documents for the refund query"
    b = "the retriever returned irrelevant documents about shipping"
    sim = text_similarity(a, b)
    assert 0.0 < sim < 1.0


def test_more_shared_terms_yield_higher_similarity():
    query = "retriever returned stale irrelevant documents for the user query"
    close = "retriever returned stale irrelevant documents for a query"
    far = "retriever returned stale results"
    assert text_similarity(query, close) > text_similarity(query, far)


def test_empty_text_similarity_is_zero_not_error():
    assert text_similarity("", "anything here") == 0.0
    assert text_similarity("", "") == 0.0


def test_case_and_punctuation_insensitive():
    a = "Retriever returned STALE docs!"
    b = "retriever returned stale docs"
    assert text_similarity(a, b) == 1.0
