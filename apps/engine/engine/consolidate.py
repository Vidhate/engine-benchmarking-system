"""Consolidation: raw findings + a cluster plan + the seed board -> Issueboard.

The LLM's job in the meta pass is only to *decide the clustering* — which raw
findings are the same failure mode, and which of them are failure modes the
seed board already names. Turning that decision into an issueboard is pure
code, so the invariants the benchmark depends on (one issue per failure mode,
seed issues gain occurrences instead of being duplicated, occurrences carry
`{trace_id, error_id}`) hold regardless of what the model returns.

That split is also why the seed merge is testable without a network: the merge
is a function, not a prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from engine.models import (
    OTHER_CATEGORY_ID,
    Category,
    Cluster,
    ConsolidationPlan,
    Issue,
    Issueboard,
    IssueOccurrence,
    RawFinding,
    SeedIssueboard,
)

MAX_DESCRIPTION_CHARS = 2000
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str, limit: int = 48) -> str:
    cleaned = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return cleaned[:limit].strip("-") or "issue"


def normalize_title(text: str) -> str:
    """Comparison key for titles: case-, punctuation- and spacing-insensitive."""
    return " ".join(_SLUG_RE.sub(" ", text.strip().lower()).split())


def valid_category(category_id: str | None, categories: Iterable[Category]) -> str:
    """Clamp a category to the vocabulary the Engine was given.

    The Engine may only speak the public taxonomy it was handed; an id it
    invented would be unscoreable, so it degrades to the `other` escape hatch
    rather than being passed through.
    """
    known = {c.category_id for c in categories}
    if category_id and category_id in known:
        return category_id
    return OTHER_CATEGORY_ID


def fallback_plan(findings: list[RawFinding]) -> ConsolidationPlan:
    """Deterministic clustering used when the consolidation LLM is unavailable
    or returns nothing usable: group by (category, normalized title)."""
    groups: dict[tuple[str, str], list[int]] = {}
    for index, finding in enumerate(findings):
        groups.setdefault((finding.category_id, normalize_title(finding.title)), []).append(index)
    clusters = []
    for (category_id, _), indices in groups.items():
        head = findings[indices[0]]
        clusters.append(
            Cluster(
                title=head.title,
                description=head.description,
                category_id=category_id,
                severity=_max_severity(findings[i].severity for i in indices),
                finding_indices=indices,
            )
        )
    return ConsolidationPlan(clusters=clusters)


_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


def _max_severity(values: Iterable[str]) -> str:
    ordered = sorted(values, key=lambda s: _SEVERITY_ORDER.get(s, 1))
    return ordered[-1] if ordered else "medium"


def complete_plan(plan: ConsolidationPlan, findings: list[RawFinding]) -> ConsolidationPlan:
    """Fold findings the plan never referenced into deterministic clusters.

    A finding the analysis pass reported and the meta pass silently forgot is
    lost recall the benchmark would score as a miss, so leftovers are grouped
    by (category, normalized title) rather than dropped. Deliberate *rejection*
    is not expressible in the plan schema — if it ever needs to be, it should
    be an explicit `rejected_indices` field, not silence.
    """
    claimed = {i for cluster in plan.clusters for i in cluster.finding_indices}
    leftover = [i for i in range(len(findings)) if i not in claimed]
    if not leftover:
        return plan
    extra = fallback_plan([findings[i] for i in leftover])
    remapped = [
        cluster.model_copy(update={"finding_indices": [leftover[i] for i in cluster.finding_indices]})
        for cluster in extra.clusters
    ]
    return ConsolidationPlan(clusters=[*plan.clusters, *remapped])


def assemble_board(
    plan: ConsolidationPlan,
    findings: list[RawFinding],
    seed_board: SeedIssueboard | None = None,
    categories: Iterable[Category] | None = None,
) -> Issueboard:
    """Turn a cluster plan into an `Issueboard(source="engine_predicted")`.

    Seed issues are carried over verbatim (id, title, description, category,
    severity). A cluster that names one of them via `matches_seed_error_id`
    attaches its occurrences to that existing issue; every other cluster
    becomes a new issue with a fresh, deterministic `error_id`.
    """
    seed = seed_board or SeedIssueboard()
    categories = list(categories or [])
    seed_ids = {issue.error_id for issue in seed.issues}

    issues: list[Issue] = [issue.model_copy(deep=True) for issue in seed.issues]
    occurrences: list[IssueOccurrence] = [o.model_copy(deep=True) for o in seed.occurrences]
    seen_occurrences = {(o.error_id, o.trace_id, o.span_id) for o in occurrences}
    used_ids = set(seed_ids)

    plan = complete_plan(plan, findings)

    for cluster in plan.clusters:
        members = _members(cluster, findings)
        if not members:
            continue
        if cluster.matches_seed_error_id in seed_ids:
            # Merge: the seed board already names this failure mode. The seed
            # issue's own text is authoritative — only occurrences are added.
            error_id = cluster.matches_seed_error_id
        else:
            error_id = _unique_id(f"ep-{slug(cluster.title)}", used_ids)
            used_ids.add(error_id)
            issues.append(
                Issue(
                    error_id=error_id,
                    title=cluster.title.strip() or "Unnamed issue",
                    description=cluster.description.strip()[:MAX_DESCRIPTION_CHARS],
                    category_id=valid_category(cluster.category_id, categories),
                    severity=_max_severity([cluster.severity]),  # type: ignore[arg-type]
                )
            )
        for finding in members:
            if not finding.trace_id:
                continue
            key = (error_id, finding.trace_id, finding.span_id)
            if key in seen_occurrences:
                continue
            seen_occurrences.add(key)
            occurrences.append(
                IssueOccurrence(
                    error_id=error_id,
                    trace_id=finding.trace_id,
                    turn_index=finding.turn_index,
                    span_id=finding.span_id,
                    evidence=finding.evidence or finding.description,
                )
            )

    board = Issueboard(source="engine_predicted", issues=issues, occurrences=occurrences)
    return board.model_copy(update={"board_id": board_id(board)})


def _members(cluster: Cluster, findings: list[RawFinding]) -> list[RawFinding]:
    """Resolve a cluster's indices: in range, de-duplicated, order preserved."""
    seen: set[int] = set()
    members = []
    for index in cluster.finding_indices:
        if not isinstance(index, int) or index < 0 or index >= len(findings) or index in seen:
            continue
        seen.add(index)
        members.append(findings[index])
    return members


def _unique_id(candidate: str, taken: set[str]) -> str:
    if candidate not in taken:
        return candidate
    suffix = 2
    while f"{candidate}-{suffix}" in taken:
        suffix += 1
    return f"{candidate}-{suffix}"


def board_id(board: Issueboard) -> str:
    """Content hash, matching the benchmark's own id convention (16 hex chars
    over the board's content with the id field excluded)."""
    payload = board.model_dump(mode="json")
    payload.pop("board_id", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
