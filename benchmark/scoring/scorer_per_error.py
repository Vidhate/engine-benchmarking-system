"""Scorer 2 — per-error trace classification (docs/architecture/06-scoring.md).

For each known error: which traces did the Layer-1 resolutions say exhibit it?
Independent of Layer-2 issue pairing — uses every resolved occurrence (exact-key
*and* fallback), since Layer 1 is the full occurrence-resolution output and this
is the strictest localization test (right error, right traces), not a pairing test.
"""

from __future__ import annotations

from collections import defaultdict

from benchmark.schemas import CategoryScore, Issueboard, OccurrenceMatch
from benchmark.scoring.metrics import binary_prf1_kappa


def score_per_error(
    gt: Issueboard,
    pred: Issueboard,  # kept for interface symmetry with the other scorers
    occ_matches: list[OccurrenceMatch],
    trace_ids: list[str],
) -> list[CategoryScore]:
    gt_traces: dict[str, set[str]] = defaultdict(set)
    for occ in gt.occurrences:
        gt_traces[occ.error_id].add(occ.trace_id)

    resolved_traces: dict[str, set[str]] = defaultdict(set)
    for m in occ_matches:
        if m.resolved_error_id is not None:
            resolved_traces[m.resolved_error_id].add(m.trace_id)

    known_error_ids = sorted({issue.error_id for issue in gt.issues} | set(resolved_traces))

    scores = []
    for error_id in known_error_ids:
        result = binary_prf1_kappa(
            gt_traces.get(error_id, set()), resolved_traces.get(error_id, set()), trace_ids
        )
        scores.append(
            CategoryScore(
                category_id=error_id,
                precision=result.precision,
                recall=result.recall,
                f1=result.f1,
                cohens_kappa=result.cohens_kappa,
                support=result.support,
            )
        )
    return scores
