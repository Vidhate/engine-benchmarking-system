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
import sys
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
        cluster.model_copy(
            update={"finding_indices": [leftover[i] for i in cluster.finding_indices]}
        )
        for cluster in extra.clusters
    ]
    return ConsolidationPlan(clusters=[*plan.clusters, *remapped])


def offset_plan(plan: ConsolidationPlan, offset: int, chunk_size: int) -> ConsolidationPlan:
    """Rebase a chunk-local plan onto the global findings list.

    Indices outside the chunk are dropped rather than shifted: a model that
    invents an index has not told us anything about a finding it never saw.
    """
    return ConsolidationPlan(
        clusters=[
            cluster.model_copy(
                update={
                    "finding_indices": [
                        index + offset
                        for index in cluster.finding_indices
                        if isinstance(index, int) and 0 <= index < chunk_size
                    ]
                }
            )
            for cluster in plan.clusters
        ]
    )


def fold_clusters(plan: ConsolidationPlan) -> ConsolidationPlan:
    """Merge clusters that name the same failure mode, in code.

    Chunked consolidation asks the model about disjoint slices of the findings,
    so the same failure mode can come back once per chunk. Identical
    (category, normalized title) pairs are the same issue by construction and
    are folded here; differently-worded duplicates are the LLM merge pass's job.
    """
    groups: dict[tuple[str, str], Cluster] = {}
    order: list[tuple[str, str]] = []
    for cluster in plan.clusters:
        key = (cluster.category_id, normalize_title(cluster.title))
        head = groups.get(key)
        if head is None:
            groups[key] = cluster.model_copy(deep=True)
            order.append(key)
            continue
        head.finding_indices = [*head.finding_indices, *cluster.finding_indices]
        head.severity = _max_severity([head.severity, cluster.severity])  # type: ignore[assignment]
        head.matches_seed_error_id = head.matches_seed_error_id or cluster.matches_seed_error_id
    return ConsolidationPlan(clusters=[groups[key] for key in order])


def apply_merge(merge: ConsolidationPlan, source: ConsolidationPlan) -> ConsolidationPlan:
    """Apply a second-stage plan whose `finding_indices` are *cluster* indices.

    The merge pass reuses `ConsolidationPlan` over cluster summaries, so one
    schema and one prompt shape serve both stages. Clusters the merge pass never
    mentions survive untouched — a forgotten cluster must not become a lost issue.
    """
    claimed: set[int] = set()
    merged: list[Cluster] = []
    for group in merge.clusters:
        members: list[Cluster] = []
        for index in dict.fromkeys(group.finding_indices):
            if not isinstance(index, int) or not 0 <= index < len(source.clusters):
                continue
            if index in claimed:
                continue
            claimed.add(index)
            members.append(source.clusters[index])
        if not members:
            continue
        head = members[0]
        merged.append(
            Cluster(
                title=group.title.strip() or head.title,
                description=group.description.strip() or head.description,
                category_id=group.category_id or head.category_id,
                severity=_max_severity([group.severity, *(m.severity for m in members)]),  # type: ignore[arg-type]
                finding_indices=[i for m in members for i in m.finding_indices],
                matches_seed_error_id=group.matches_seed_error_id or _first_seed_id(members),
            )
        )
    merged.extend(
        cluster for index, cluster in enumerate(source.clusters) if index not in claimed
    )
    return ConsolidationPlan(clusters=merged)


def _first_seed_id(clusters: list[Cluster]) -> str | None:
    return next((c.matches_seed_error_id for c in clusters if c.matches_seed_error_id), None)


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
    # One occurrence per (issue, trace). That pair IS what scoring consumes, so
    # a second sighting of the same failure mode in the same trace enriches the
    # existing occurrence's localization rather than adding a row — otherwise a
    # seed occurrence (no span_id) and a fresh finding (span_id set) would both
    # land and double-count the trace.
    by_pair: dict[tuple[str, str], IssueOccurrence] = {
        (o.error_id, o.trace_id): o for o in occurrences
    }
    used_ids = set(seed_ids)

    plan = complete_plan(plan, findings)
    # "Every finding index appears in exactly one cluster" — first claim wins,
    # so a model that lists the same finding under two clusters cannot inflate
    # the board with a duplicate issue.
    claimed: set[int] = set()
    coerced: list[str] = []

    for cluster in plan.clusters:
        members = _members(cluster, findings, claimed)
        if not members:
            continue
        if cluster.matches_seed_error_id in seed_ids:
            # Merge: the seed board already names this failure mode. The seed
            # issue's own text is authoritative — only occurrences are added.
            error_id = cluster.matches_seed_error_id
        else:
            error_id = _unique_id(f"ep-{slug(cluster.title)}", used_ids)
            used_ids.add(error_id)
            category_id = valid_category(cluster.category_id, categories)
            if category_id != cluster.category_id:
                # A category the model invented is a real prediction we cannot
                # map, not an absent one — different from a default, and kept.
                # Counted because the rate is a per-model quality signal: an
                # Engine that keeps inventing vocabulary is telling us something.
                coerced.append(f"{cluster.category_id!r} ({cluster.title})")
            issues.append(
                Issue(
                    error_id=error_id,
                    title=cluster.title.strip() or "Unnamed issue",
                    description=cluster.description.strip()[:MAX_DESCRIPTION_CHARS],
                    category_id=category_id,
                    severity=cluster.severity,
                )
            )
        for finding in members:
            if not finding.trace_id:
                continue
            pair = (error_id, finding.trace_id)
            existing = by_pair.get(pair)
            if existing is not None:
                _enrich(existing, finding)
                continue
            occurrence = IssueOccurrence(
                error_id=error_id,
                trace_id=finding.trace_id,
                turn_index=finding.turn_index,
                span_id=finding.span_id,
                evidence=finding.evidence or finding.description,
            )
            by_pair[pair] = occurrence
            occurrences.append(occurrence)

    if coerced:
        print(
            f"[engine] coerced {len(coerced)} out-of-vocabulary category/ies to "
            f"'{OTHER_CATEGORY_ID}': {'; '.join(coerced)}",
            file=sys.stderr,
        )

    board = Issueboard(source="engine_predicted", issues=issues, occurrences=occurrences)
    return board.model_copy(update={"board_id": board_id(board)})


def _enrich(occurrence: IssueOccurrence, finding: RawFinding) -> None:
    """Fill in localization the existing occurrence is missing. Never overwrites:
    the first sighting (a seed occurrence, or the first cluster to claim the
    trace) stays authoritative."""
    if occurrence.turn_index is None and finding.turn_index is not None:
        occurrence.turn_index = finding.turn_index
    if occurrence.span_id is None and finding.span_id:
        occurrence.span_id = finding.span_id
    if not occurrence.evidence:
        occurrence.evidence = finding.evidence or finding.description


def _members(
    cluster: Cluster, findings: list[RawFinding], claimed: set[int] | None = None
) -> list[RawFinding]:
    """Resolve a cluster's indices: in range, unclaimed, de-duplicated, in order.

    `claimed` is shared across the clusters of one plan, which is how the
    "exactly one cluster per finding" rule is enforced.
    """
    claimed = claimed if claimed is not None else set()
    members = []
    for index in cluster.finding_indices:
        if not isinstance(index, int) or index < 0 or index >= len(findings):
            continue
        if index in claimed:
            continue
        claimed.add(index)
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
    """A 16-hex-char content hash of the board, id field excluded.

    Same *shape* as the benchmark's `schemas.io.content_hash`, but NOT the same
    value: that function hashes the board after it has been parsed into
    `benchmark.schemas.issues.Issueboard`, whose `Issue` declares fields this
    app does not (`injection_mode`, serialized as null), so the canonical JSON
    differs. This id identifies the board the Engine produced; the benchmark
    should re-stamp on ingest rather than assume the two agree.
    """
    payload = board.model_dump(mode="json")
    payload.pop("board_id", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
