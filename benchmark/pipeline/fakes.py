"""Development stand-ins for stages that are not on main yet.

**This module is temporary scaffolding, not product.** `fake_run_ablation`
exists because Phase 5 (`benchmark/ablation/`) is being built in parallel: it
lets the miniature run and the CI integration test exercise every *other*
integration seam today, against the exact call shape the real stage will have.
It ships inside the package rather than in `tests/` because the live smoke
script needs it too, and a test double that two callers import from two places
drifts.

**At integration time it is deleted**, and the single line
`ablation_stage=fake_run_ablation` becomes
`ablation_stage=load_ablation_stage()`. Nothing else in the pipeline changes —
that is the property this module is designed to prove.

What the fake does NOT do: it does not ablate anything. Traces pass through
untouched, and the ground truth it plants is a label over unmodified traces.
So a run using it produces *wiring* evidence, never *quality* evidence: the
Engine is being scored against errors nobody injected, and low scores are the
expected outcome, not a finding.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.pipeline.export import write_leak_stripped_export
from benchmark.schemas import (
    AblationConfig,
    AblationRecord,
    AblationSplit,
    ErrorCategory,
    Issue,
    Issueboard,
    IssueOccurrence,
    TraceDataset,
)
from benchmark.schemas.inputs import InputDataset
from benchmark.schemas.io import derive, stamp_dataset_id
from benchmark.schemas.issues import OTHER_CATEGORY_ID
from benchmark.tracing.store import TraceStore

#: Alternated across planted errors so a fake run still exercises both arms of
#: the post-hoc content-vs-mechanism analysis downstream.
_MODES = ("replay_edit", "dependency_fault")

_MAX_PLANTED_ERRORS = 3


@dataclass
class FakeAblationResult:
    """Structurally identical to the pinned Phase-5 `AblationResult`."""

    ablated: TraceDataset
    ground_truth: Issueboard
    records: list[AblationRecord] = field(default_factory=list)
    split: AblationSplit = field(
        default_factory=lambda: AblationSplit(seed=0, control_fraction=0.0)
    )
    export_path: str = ""
    dropped_errors: list[str] = field(default_factory=list)


def _adversarial_first(inputs: InputDataset) -> dict[str, int]:
    """Rank inputs so planted errors land where the app is likeliest to slip.

    Purely cosmetic for a pass-through fake, but it makes the miniature run's
    numbers a shade less meaningless: an adversarial input's trace is where an
    organic issue might actually coincide with the planted label.
    """
    order: dict[str, int] = {}
    for spec in inputs.inputs:
        adversarial = bool(spec.fixed_adversarial_id) or "adv" in spec.dim_id.lower()
        order[spec.input_id] = (0 if adversarial else 1)
    return order


def split_inputs(inputs: InputDataset, cfg: AblationConfig) -> AblationSplit:
    """A seeded, provenance-stratified control/ablate split at input level.

    Stratifying on `dim_id` is the cheap half of what Phase 5 does properly;
    the point here is only that control inputs exist, are chosen the same way
    on every rerun, and are never touched afterwards.
    """
    by_stratum: dict[str, list[str]] = {}
    for spec in sorted(inputs.inputs, key=lambda s: s.input_id):
        by_stratum.setdefault(spec.dim_id, []).append(spec.input_id)

    rng = random.Random(cfg.seed)
    control: list[str] = []
    ablate: list[str] = []
    for stratum in sorted(by_stratum):
        members = list(by_stratum[stratum])
        rng.shuffle(members)
        n_control = int(round(len(members) * cfg.control_fraction))
        control.extend(members[:n_control])
        ablate.extend(members[n_control:])
    return AblationSplit(
        seed=cfg.seed,
        control_fraction=cfg.control_fraction,
        strata=["dim_id"],
        control_input_ids=sorted(control),
        ablate_input_ids=sorted(ablate),
    )


def fake_run_ablation(
    *,
    traces: TraceDataset,
    inputs: InputDataset,
    categories: list[ErrorCategory],
    cfg: AblationConfig,
    harness: Any = None,
    store: TraceStore | None = None,
    export_path: str | Path,
) -> FakeAblationResult:
    """Pass-through "ablation": a synthetic ground truth over untouched traces.

    Same call shape and same return shape as the pinned Phase-5
    `run_ablation`. `harness` and `store` are accepted and ignored — the real
    stage needs them to replay and re-run, this one has nothing to inject.

    Invariants it DOES honour, because downstream code depends on them:

    * control inputs carry no ground-truth occurrence at all;
    * no trace carries two occurrences of the same category (the exact-key
      matcher's disjointness invariant — scoring is unsound without it);
    * the ablated dataset points at the source dataset via `parent_dataset_id`;
    * the export written to `export_path` is leak-stripped and audited.
    """
    cfg = cfg or AblationConfig()
    split = split_inputs(inputs, cfg)
    ablate_ids = set(split.ablate_input_ids)

    rank = _adversarial_first(inputs)
    ablate_traces = sorted(
        (t for t in traces.traces if t.input_id in ablate_ids),
        key=lambda t: (rank.get(t.input_id, 1), t.trace_id),
    )

    usable = [c for c in categories if c.category_id != OTHER_CATEGORY_ID]
    planted = usable[: max(1, min(_MAX_PLANTED_ERRORS, len(usable)))] if usable else []

    issues: list[Issue] = []
    occurrences: list[IssueOccurrence] = []
    records: list[AblationRecord] = []
    for index, category in enumerate(planted):
        error_id = f"K{index + 1}"
        issues.append(
            Issue(
                error_id=error_id,
                title=f"planted {category.name}",
                description=(
                    f"Synthetic ground-truth entry standing in for a real {category.name} "
                    f"injection until Phase 5 lands. {category.description}"
                ),
                category_id=category.category_id,
                severity=("high", "medium", "low")[index % 3],
                injection_mode=_MODES[index % len(_MODES)],
            )
        )
    # Round-robin, so one trace never gets two errors — and therefore never two
    # of the same category, whatever the trace count happens to be.
    for position, trace in enumerate(ablate_traces):
        if not issues:
            break
        issue = issues[position % len(issues)]
        occurrences.append(
            IssueOccurrence(
                error_id=issue.error_id,
                trace_id=trace.trace_id,
                turn_index=0,
                evidence="(fake ablation: no content was actually modified)",
            )
        )
        records.append(
            AblationRecord(
                ablation_id=f"fake-{issue.error_id}-{trace.trace_id}",
                error_id=issue.error_id,
                trace_id=trace.trace_id,
            )
        )

    ground_truth = stamp_dataset_id(
        Issueboard(source="ground_truth", issues=issues, occurrences=occurrences)
    )
    # Pass-through: every trace survives, control and ablate alike, so the
    # trace universe handed to scoring is the real one.
    ablated = derive(TraceDataset(traces=list(traces.traces)), traces)
    written = write_leak_stripped_export(ablated, export_path)

    return FakeAblationResult(
        ablated=ablated,
        ground_truth=ground_truth,
        records=records,
        split=split,
        export_path=str(written),
        dropped_errors=[],
    )


#: Explicit marker, so the runner's "this run was faked" warning does not rest
#: on the word "fake" surviving in somebody's wrapper name.
fake_run_ablation.is_pipeline_fake = True  # type: ignore[attr-defined]
FakeAblationResult.is_pipeline_fake = True  # type: ignore[attr-defined]
