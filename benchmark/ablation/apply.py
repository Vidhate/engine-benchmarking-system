"""Step 4 — filter, sub-sample, inject, record.

docs/architecture/04-ablation-engine.md, step 4:

    1. apply filter               -> eligible traces (ABLATE SET ONLY)
    2. sub-sample if too large    -> target_count traces (seeded RNG)
    3. inject per mode
    4. store {trace_id, error_id} -> IssueOccurrences + AblationRecords

## Same-category disjointness — the exact-match key

Two errors of the **same category are never injected into the same trace**.
That is what makes `(trace_id, category_id)` uniquely identify the injected
error, which is what lets Layer-1 scoring match predictions to ground truth by
exact key instead of by text similarity (docs/architecture/06-scoring.md).
Enforced here, during sub-sampling, per **input** rather than per trace: a
trace's id changes when it is ablated, so "which categories has this already
received" is a property of the input, not of any one trace object.

## Composition rules (why they are what they are)

A `dependency_fault` **regenerates the whole trace**, so it must run before any
`replay_edit` on the same input — the other order would throw the replay's
corrupted turn away and leave an `AblationRecord` describing content that is
not in the shipped trace. And per input at most **one** of each mode:

* two dependency faults would need two shims armed on one run, which the
  declared `configurable` surface expresses one key at a time;
* two replay edits would have the second overwrite the first's corrupted turn
  on the single-turn traces that dominate the corpus.

So a compound trace is exactly "one mechanism fault + one content corruption,
different categories", and every occurrence on it is still exactly true.

Because the final trace id for an input is only known once its last injection
has run, occurrences and records are **re-stamped** at the end to point at the
trace that actually ships.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from benchmark.ablation import filters
from benchmark.ablation.agent import ProposedError
from benchmark.ablation.inject import (
    InjectionError,
    SelfCorrected,
    apply_dependency_fault,
    apply_replay_edit,
    thread_alive,
    thread_ref,
)
from benchmark.schemas.ablation import AblationRecord, AblationSpec, AblationSplit
from benchmark.schemas.inputs import InputDataset, InputSpec, Persona
from benchmark.schemas.io import derive, stamp_dataset_id
from benchmark.schemas.issues import Issue, Issueboard, IssueOccurrence
from benchmark.schemas.traces import Trace, TraceDataset

log = logging.getLogger("benchmark.ablation")

# Mode C first — it regenerates the whole trace (see the module docstring).
MODE_ORDER = {"dependency_fault": 0, "replay_edit": 1}


@dataclass
class _InputState:
    """What has already happened to one input, as ablations accumulate."""

    trace: Trace
    categories: set[str] = field(default_factory=set)
    modes: set[str] = field(default_factory=set)
    occurrences: list[IssueOccurrence] = field(default_factory=list)
    records: list[AblationRecord] = field(default_factory=list)


class ApplyOutcome(BaseModel):
    ablated: TraceDataset
    ground_truth: Issueboard
    records: list[AblationRecord] = Field(default_factory=list)
    injected: dict[str, int] = Field(default_factory=dict)
    dropped: dict[str, str] = Field(default_factory=dict)
    # Candidates the app self-corrected out from under, per error. Reported
    # because it bounds how much of the corpus the not-retracted check consumed
    # (see `inject.retraction_in` on the residual false-negative surface).
    self_corrected: dict[str, int] = Field(default_factory=dict)


def _rng(seed: int, error_id: str) -> random.Random:
    """A per-error stream: one error's sub-sample never shifts another's."""
    return random.Random(f"{seed}\x1f{error_id}")


def _ablation_id(error_id: str, index: int) -> str:
    return f"abl-{error_id}-{index:03d}"


def apply_ablations(
    specs: Sequence[AblationSpec],
    proposals: Mapping[str, ProposedError],
    traces: TraceDataset,
    inputs: InputDataset,
    split: AblationSplit,
    harness: Any,
    *,
    seed: int = 0,
    dataset_id: str = "",
    max_turns: int = 1,
) -> ApplyOutcome:
    """`[N,M,T]`, `E` -> `[N,M,T*]`, `[N,E_K]` + the audit trail."""
    inputs_by_id: dict[str, InputSpec] = {i.input_id: i for i in inputs.inputs}
    gen_cfg = inputs.generation_config
    personas: dict[str, Persona] = {
        p.persona_id: p for p in (*gen_cfg.personas, *gen_cfg.adversarial_personas)
    }
    ablate_ids = set(split.ablate_input_ids)

    # Control traces are carried through by reference and never touched, so
    # "byte-identical through the pipeline" is a property of the code path,
    # not something a later copy has to preserve.
    order = [t.input_id for t in traces.traces]
    if len(set(order)) != len(order):
        # The whole step keys on input_id — disjointness, the current trace,
        # and the final re-stamp. Two traces sharing one input would silently
        # lose an injection and mislabel the survivor's ground truth, so this
        # fails loudly rather than producing a plausible-looking board.
        duplicates = sorted({i for i in order if order.count(i) > 1})
        raise ValueError(
            f"the trace dataset has {len(order) - len(set(order))} duplicate input_id(s) "
            f"{duplicates[:5]}: step 4 keys its disjointness and lineage bookkeeping on "
            f"input_id, and cannot tell two traces of one input apart"
        )
    state: dict[str, _InputState] = {t.input_id: _InputState(trace=t) for t in traces.traces}

    liveness: dict[str, bool] = {}

    def replayable(trace: Trace) -> bool:
        # Single-turn replay_edit is a post-hoc edit — no thread involved, so
        # a dead collection-lifetime server never disqualifies the trace.
        if len(trace.turns) <= 1:
            return True
        ref = thread_ref(trace)
        if ref is None:
            return False
        if ref not in liveness:
            liveness[ref] = thread_alive(harness, ref)
        return liveness[ref]

    injected: dict[str, int] = {}
    dropped: dict[str, str] = {}
    burned: dict[str, int] = {}
    used_issues: dict[str, Issue] = {}

    for spec in sorted(specs, key=lambda s: (MODE_ORDER.get(s.mode, 9), s.error_id)):
        proposal = proposals.get(spec.error_id)
        if proposal is None:
            dropped[spec.error_id] = f"{spec.error_id}: no proposal behind this spec"
            continue
        category = proposal.issue.category_id

        current_traces = [state[input_id].trace for input_id in order]
        candidates = filters.eligible(current_traces, spec.filter, ablate_ids)

        pool = [
            trace
            for trace in candidates
            if category not in state[trace.input_id].categories
            and spec.mode not in state[trace.input_id].modes
            and (spec.mode != "replay_edit" or replayable(trace))
        ]
        _rng(seed, spec.error_id).shuffle(pool)

        count = 0
        for trace in pool:
            if count >= spec.target_count:
                break
            ablation_id = _ablation_id(spec.error_id, count)
            try:
                if spec.mode == "replay_edit":
                    new_trace, record = apply_replay_edit(
                        trace,
                        spec,
                        harness,
                        ablation_id=ablation_id,
                        dataset_id=dataset_id,
                        seed=seed,
                    )
                    # Read k back from the RECORD, not the spec: an unpinned
                    # corruption draws its turn per trace, so the spec's value
                    # is None and only the record knows what was corrupted.
                    applied_params = record.actions_applied[0].params
                    occurrence = IssueOccurrence(
                        error_id=spec.error_id,
                        trace_id=new_trace.trace_id,
                        turn_index=int(applied_params.get("turn_index") or 0),
                        evidence=str(applied_params.get("marker") or "")[:200],
                    )
                else:
                    input_spec = inputs_by_id.get(trace.input_id)
                    if input_spec is None:
                        log.warning(
                            "%s: trace %s names an input that is not in the dataset",
                            spec.error_id,
                            trace.trace_id,
                        )
                        continue
                    new_trace, record = apply_dependency_fault(
                        input_spec,
                        spec,
                        harness,
                        baseline=trace,
                        ablation_id=ablation_id,
                        dataset_id=dataset_id,
                        persona=personas.get(input_spec.persona_id or ""),
                        max_turns=max_turns,
                    )
                    occurrence = IssueOccurrence(
                        error_id=spec.error_id,
                        trace_id=new_trace.trace_id,
                        evidence=(record.before_after[0][1] if record.before_after else "")[:200],
                    )
            except SelfCorrected as exc:
                # Counted separately and reported: a burned candidate is the
                # app defending itself against the injection, which is a
                # property of the corpus worth knowing, not just a retry.
                burned[spec.error_id] = burned.get(spec.error_id, 0) + 1
                log.warning(
                    "%s: %s self-corrected the injection — trying the next candidate (%s)",
                    spec.error_id,
                    trace.trace_id,
                    exc,
                )
                continue
            except InjectionError as exc:
                log.warning(
                    "%s: injection into %s failed (%s: %s) — trying the next candidate",
                    spec.error_id,
                    trace.trace_id,
                    type(exc).__name__,
                    exc,
                )
                continue
            except Exception as exc:  # noqa: BLE001 - one bad trace must not kill the batch
                log.warning(
                    "%s: unexpected %s injecting into %s: %s",
                    spec.error_id,
                    type(exc).__name__,
                    trace.trace_id,
                    exc,
                )
                continue

            entry = state[trace.input_id]
            entry.trace = new_trace
            entry.categories.add(category)
            entry.modes.add(spec.mode)
            entry.occurrences.append(occurrence)
            entry.records.append(record)
            count += 1

        injected[spec.error_id] = count
        if count:
            used_issues[spec.error_id] = proposal.issue
        else:
            dropped[spec.error_id] = (
                f"{spec.error_id}: {len(pool)} candidate(s) survived disjointness and "
                f"none could be injected"
            )

    # Re-stamp: an input's occurrences and records must name the trace that
    # actually ships, which is only known now.
    occurrences: list[IssueOccurrence] = []
    records: list[AblationRecord] = []
    for input_id in order:
        entry = state[input_id]
        final_id = entry.trace.trace_id
        for occurrence in entry.occurrences:
            occurrences.append(occurrence.model_copy(update={"trace_id": final_id}))
        for record in entry.records:
            records.append(record.model_copy(update={"trace_id": final_id}))
        if entry.records:
            entry.trace.ablation_ids = sorted({r.ablation_id for r in entry.records})

    ablated = TraceDataset(traces=[state[input_id].trace for input_id in order])
    board = Issueboard(
        source="ground_truth",
        issues=[used_issues[eid] for eid in sorted(used_issues)],
        occurrences=sorted(occurrences, key=lambda o: (o.trace_id, o.error_id)),
    )
    return ApplyOutcome(
        ablated=derive(ablated, traces),
        ground_truth=stamp_dataset_id(board),
        records=sorted(records, key=lambda r: r.ablation_id),
        injected=injected,
        dropped=dropped,
        self_corrected=burned,
    )
