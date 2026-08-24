"""Deterministic BM25 retrieval over the local corpus — no embeddings, no network.

Kept intentionally boring: the target app is a benchmark subject, so its
retrieval must be reproducible run-to-run (the ablation engine compares an
unarmed run against a shimmed one and needs the unarmed half to be stable).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from target_app.corpus import Document

_WORD = re.compile(r"[a-z0-9]+")

# Small, fixed stop list. Deliberately conservative: product vocabulary
# ("plan", "team", "page") must survive.
STOPWORDS = frozenset(
    """
    a about all am an and any are as at be been being but by can cannot could did do does
    doing done for from get got had has have having he her here hers him his how i if in
    into is it its just like me my no nor of off on once one only or our ours out over own
    please same she should so some such than that the their theirs them then there these
    they this those to too us very was we were what when where which while who whom why
    will with would you your yours
    """.split()
)

BM25_K1 = 1.5
BM25_B = 0.75
TAG_WEIGHT = 3  # tags and title are repeated this many times in the indexed field


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords."""
    return [tok for tok in _WORD.findall(text.lower()) if tok not in STOPWORDS]


@dataclass(frozen=True)
class RetrievedDoc:
    doc_id: str
    title: str
    updated: str
    text: str
    score: float

    def as_payload(self) -> dict[str, object]:
        """What the tool actually returns.

        The BM25 score is deliberately omitted: an `irrelevant_docs` run would
        show near-zero scores next to non-zero ones on an organic run, which is
        a statistical tell an Engine could learn instead of reading the docs.
        """
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "updated": self.updated,
            "content": self.text,
        }


def _indexed_field(doc: Document) -> str:
    boosted = " ".join([doc.title, " ".join(doc.tags)] * TAG_WEIGHT)
    return f"{boosted} {doc.text}"


class Retriever:
    """BM25 over `Document.text` (+ boosted title/tags). Archived text is never indexed."""

    def __init__(self, documents: list[Document]):
        self.documents = list(documents)
        self._by_id = {doc.doc_id: doc for doc in self.documents}
        self._tokens = {
            doc.doc_id: Counter(tokenize(_indexed_field(doc))) for doc in self.documents
        }
        self._lengths = {doc_id: sum(c.values()) for doc_id, c in self._tokens.items()}
        n = len(self.documents) or 1
        self._avg_len = (sum(self._lengths.values()) / n) or 1.0
        df: Counter[str] = Counter()
        for counts in self._tokens.values():
            df.update(counts.keys())
        self._idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def score(self, query: str, doc_id: str) -> float:
        counts = self._tokens[doc_id]
        length = self._lengths[doc_id]
        total = 0.0
        for term in tokenize(query):
            tf = counts.get(term, 0)
            if not tf:
                continue
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * length / self._avg_len)
            total += self._idf.get(term, 0.0) * (tf * (BM25_K1 + 1)) / denom
        return total

    def _ranked(self, query: str) -> list[tuple[str, float]]:
        # (-score, doc_id) sort => deterministic tie-breaking by doc_id.
        return sorted(
            ((doc_id, self.score(query, doc_id)) for doc_id in self._by_id),
            key=lambda pair: (-pair[1], pair[0]),
        )

    def _wrap(self, ranked: list[tuple[str, float]]) -> list[RetrievedDoc]:
        out = []
        for doc_id, score in ranked:
            doc = self._by_id[doc_id]
            out.append(
                RetrievedDoc(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    updated=doc.updated,
                    text=doc.text,
                    score=score,
                )
            )
        return out

    def search(self, query: str, k: int = 3) -> list[RetrievedDoc]:
        """Top-k documents by BM25. Zero-scoring documents are never returned."""
        hits = [pair for pair in self._ranked(query) if pair[1] > 0][:k]
        return self._wrap(hits)

    def worst_matches(self, query: str, k: int = 3) -> list[RetrievedDoc]:
        """Bottom-k documents — the corpus's *least* relevant answers to `query`."""
        ranked = self._ranked(query)
        return self._wrap(list(reversed(ranked))[:k])

    def stale_view(self, results: list[RetrievedDoc]) -> list[RetrievedDoc]:
        """Same documents, served from their archived (outdated) revision."""
        out = []
        for hit in results:
            doc = self._by_id[hit.doc_id]
            out.append(
                RetrievedDoc(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    updated=doc.stale_updated,
                    text=doc.stale_text,
                    score=hit.score,
                )
            )
        return out
