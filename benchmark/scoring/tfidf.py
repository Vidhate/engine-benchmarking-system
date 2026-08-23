"""Pure-python TF-IDF text similarity — no runtime deps beyond stdlib.

Used by the Layer-1 wrong-category fallback and the Layer-2 tie-break (the only
two places text similarity enters scoring per docs/architecture/06-scoring.md).
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {term: count / total for term, count in counts.items()}


def tfidf_vectors(docs: list[str]) -> list[dict[str, float]]:
    """Smoothed TF-IDF vectors (sklearn-style idf), L2-normalized.

    Each doc's vector is comparable to any other's via plain dot product
    once normalized, so cosine_sim reduces to a dot product.
    """
    tokenized = [tokenize(doc) for doc in docs]
    n_docs = len(docs)
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    idf = {term: math.log((1 + n_docs) / (1 + count)) + 1 for term, count in df.items()}

    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        tf = _tf(tokens)
        weighted = {term: freq * idf[term] for term, freq in tf.items()}
        norm = math.sqrt(sum(w * w for w in weighted.values()))
        if norm == 0:
            vectors.append({})
        else:
            vectors.append({term: w / norm for term, w in weighted.items()})
    return vectors


def cosine_sim(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    # Iterate the smaller vector for efficiency.
    small, large = (vec_a, vec_b) if len(vec_a) <= len(vec_b) else (vec_b, vec_a)
    return sum(weight * large.get(term, 0.0) for term, weight in small.items())


def text_similarity(a: str, b: str) -> float:
    """Cosine similarity between two texts' TF-IDF vectors, over the 2-doc corpus."""
    vectors = tfidf_vectors([a, b])
    return cosine_sim(vectors[0], vectors[1])


def best_match(query: str, candidates: list[str]) -> tuple[int, float]:
    """Index + similarity of the candidate most similar to `query` (TF-IDF over
    the joint corpus of query + all candidates). Raises on empty candidates."""
    if not candidates:
        raise ValueError("best_match requires at least one candidate")
    vectors = tfidf_vectors([query, *candidates])
    query_vec, candidate_vecs = vectors[0], vectors[1:]
    sims = [cosine_sim(query_vec, vec) for vec in candidate_vecs]
    best_idx = max(range(len(sims)), key=lambda i: sims[i])
    return best_idx, sims[best_idx]
