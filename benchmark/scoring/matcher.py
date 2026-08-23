"""Error matcher: E_P -> E_K, in two layers (docs/architecture/06-scoring.md).

Layer 1 (occurrence-level, exact key): the ablation engine's same-category
disjointness invariant means (trace_id, category_id) uniquely identifies the
injected known error, so the primary path is exact bookkeeping, not fuzzy
matching. A TF-IDF fallback only fires for the wrong-category case.

Layer 2 (issue-level, argmax overlap): pairs each predicted *issue* with the
known error its resolved occurrences concentrate on the most, restricted to
known errors sharing the predicted issue's category (the literal argmax_{K_i
in same category} from the spec) — ties broken by description similarity.
"""

from __future__ import annotations

from collections import defaultdict

from benchmark.schemas import ErrorMatch, Issue, Issueboard, OccurrenceMatch, ScoringConfig
from benchmark.schemas.issues import OTHER_CATEGORY_ID
from benchmark.scoring.tfidf import best_match

_ISSUE_TEXT = "{title} {description}"


def _issue_text(issue: Issue) -> str:
    return _ISSUE_TEXT.format(title=issue.title, description=issue.description)


def resolve_occurrences(
    gt: Issueboard, pred: Issueboard, cfg: ScoringConfig
) -> list[OccurrenceMatch]:
    """Layer 1: resolve every predicted occurrence's (trace_id, category_id) key."""
    issue_by_id = {issue.error_id: issue for issue in pred.issues}
    gt_issue_by_id = {issue.error_id: issue for issue in gt.issues}

    # (trace_id, category_id) -> known error id, guaranteed unique by the
    # same-category disjointness invariant.
    exact_key_map: dict[tuple[str, str], str] = {}
    # trace_id -> [(category_id, error_id), ...] for the cross-category fallback pool.
    traces_to_gt: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for occ in gt.occurrences:
        gt_issue = gt_issue_by_id.get(occ.error_id)
        if gt_issue is None:
            continue
        exact_key_map[(occ.trace_id, gt_issue.category_id)] = occ.error_id
        traces_to_gt[occ.trace_id].append((gt_issue.category_id, occ.error_id))

    matches: list[OccurrenceMatch] = []
    for occ in pred.occurrences:
        pred_issue = issue_by_id.get(occ.error_id)
        category_id = pred_issue.category_id if pred_issue else None

        if category_id is not None:
            key = (occ.trace_id, category_id)
            if key in exact_key_map:
                matches.append(
                    OccurrenceMatch(
                        trace_id=occ.trace_id,
                        predicted_error_id=occ.error_id,
                        resolved_error_id=exact_key_map[key],
                        method="exact_key",
                    )
                )
                continue

        if category_id == OTHER_CATEGORY_ID or category_id is None:
            matches.append(
                OccurrenceMatch(
                    trace_id=occ.trace_id,
                    predicted_error_id=occ.error_id,
                    resolved_error_id=None,
                    method="exact_key",
                )
            )
            continue

        # No exact key. Same trace has injections in other categories? -> fallback.
        other_category_candidates = [
            (cat, err_id)
            for cat, err_id in traces_to_gt.get(occ.trace_id, [])
            if cat != category_id
        ]
        if not other_category_candidates:
            matches.append(
                OccurrenceMatch(
                    trace_id=occ.trace_id,
                    predicted_error_id=occ.error_id,
                    resolved_error_id=None,
                    method="exact_key",
                )
            )
            continue

        query_text = _issue_text(pred_issue)
        candidate_texts = [
            _issue_text(gt_issue_by_id[err_id]) for _, err_id in other_category_candidates
        ]
        best_idx, similarity = best_match(query_text, candidate_texts)
        resolved_id = None
        if similarity >= cfg.text_fallback_threshold:
            resolved_id = other_category_candidates[best_idx][1]
        matches.append(
            OccurrenceMatch(
                trace_id=occ.trace_id,
                predicted_error_id=occ.error_id,
                resolved_error_id=resolved_id,
                method="text_fallback",
            )
        )

    return matches


def compute_fallback_rate(matches: list[OccurrenceMatch]) -> float:
    """Fraction of resolved occurrences whose method was the text fallback —
    "firing" means the fallback path engaged, independent of whether it
    ultimately matched (matcher reliability stat, docs/architecture/06-scoring.md)."""
    if not matches:
        return 0.0
    fired = sum(1 for m in matches if m.method == "text_fallback")
    return fired / len(matches)


def pair_issues(
    occ_matches: list[OccurrenceMatch], gt: Issueboard, pred: Issueboard
) -> list[ErrorMatch]:
    """Layer 2: per predicted issue, argmax over same-category known errors of
    resolved occurrence-set overlap. Ties broken by description similarity."""
    gt_issue_by_id = {issue.error_id: issue for issue in gt.issues}

    resolved_by_pred: dict[str, list[OccurrenceMatch]] = defaultdict(list)
    for m in occ_matches:
        if m.resolved_error_id is not None:
            resolved_by_pred[m.predicted_error_id].append(m)

    matches: list[ErrorMatch] = []
    for pred_issue in pred.issues:
        candidates = resolved_by_pred.get(pred_issue.error_id, [])
        # Restrict to known errors sharing this predicted issue's category —
        # the literal "argmax_{K_i in same category}" from the spec.
        trace_sets: dict[str, set[str]] = defaultdict(set)
        for m in candidates:
            known = gt_issue_by_id.get(m.resolved_error_id)
            if known is not None and known.category_id == pred_issue.category_id:
                trace_sets[m.resolved_error_id].add(m.trace_id)

        if not trace_sets:
            matches.append(
                ErrorMatch(predicted_error_id=pred_issue.error_id, matched_error_id=None)
            )
            continue

        max_overlap = max(len(traces) for traces in trace_sets.values())
        winners = [err_id for err_id, traces in trace_sets.items() if len(traces) == max_overlap]

        if len(winners) == 1:
            matches.append(
                ErrorMatch(
                    predicted_error_id=pred_issue.error_id,
                    matched_error_id=winners[0],
                    overlap=max_overlap,
                )
            )
            continue

        query_text = _issue_text(pred_issue)
        candidate_texts = [_issue_text(gt_issue_by_id[err_id]) for err_id in winners]
        best_idx, _ = best_match(query_text, candidate_texts)
        matches.append(
            ErrorMatch(
                predicted_error_id=pred_issue.error_id,
                matched_error_id=winners[best_idx],
                overlap=max_overlap,
                tie_broken_by_text=True,
            )
        )

    return matches
