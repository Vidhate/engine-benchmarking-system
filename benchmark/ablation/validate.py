"""Step 3 — the quality gate, mode-aware, with the re-plan loop.

docs/architecture/04-ablation-engine.md, step 3:

    filter matches >= min_eligible?  ->  no: the error is not expressible here
    replay_edit:  actions run clean on a sample? turn-k spans consistent?
                  downstream replay succeeds? result schema-valid?
    dependency_fault: shim armable? fault ACTIVATES — visible in the
                  regenerated spans? (activation only, NEVER outcome)
    any failure  ->  surface ALL reasons, back to step 2

Three properties this implementation insists on:

1. **The gate runs inside the ablate set.** `min_eligible` counts traces the
   filter matched among ablate-set traces only — control inputs are not part of
   the population and never will be.
2. **Dry runs leave nothing behind.** Every sample injection runs with
   `store_result=False`, so a rejected spec cannot leave an unlabelled,
   fault-contaminated trace in the store for the corpus to inherit.
3. **The loop is bounded and the drop is logged.** Each error gets at most
   `MAX_REPLANS` re-plans; after that it is dropped with the reasons that
   killed it, and the drop is reported in `AblationResult.dropped_errors` —
   never silently.

The re-plan is not a retry: the surfaced reason picks the adaptation.

| failure | adaptation |
|---|---|
| too few eligible traces | relax the filter (`plan_ablation(attempt=n)`) |
| corruption lost / app self-corrected | re-author the corruption via the agent |
| fault never activated | rotate to the next behavior for that shim |
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from benchmark.ablation import filters
from benchmark.ablation.agent import AblationAgent, CorpusDigest, ProposedError
from benchmark.ablation.inject import (
    CorruptionLost,
    InjectionError,
    SelfCorrected,
    apply_dependency_fault,
    apply_replay_edit,
)
from benchmark.ablation.plan import MAX_REPLANS, plan_ablation, rotate_behavior
from benchmark.harness.faults import FaultNotActivated, UndeclaredFault
from benchmark.schemas.ablation import AblationSpec
from benchmark.schemas.inputs import InputSpec, Persona
from benchmark.schemas.traces import Trace

log = logging.getLogger("benchmark.ablation")

Stage = Literal["filter", "dry_run", "schema"]


class ValidationFailure(BaseModel):
    """One surfaced reason a spec did not pass, with the attempt that produced it."""

    error_id: str
    attempt: int
    stage: Stage
    reason: str


class ValidationOutcome(BaseModel):
    specs: list[AblationSpec] = []
    proposals: dict[str, ProposedError] = {}
    failures: list[ValidationFailure] = []
    dropped: dict[str, str] = {}


def _sample_for(spec: AblationSpec, candidates: Sequence[Trace]) -> Trace:
    return candidates[0]


def _schema_valid(trace: Trace) -> str | None:
    try:
        Trace.model_validate_json(trace.model_dump_json())
    except Exception as exc:  # noqa: BLE001 - any validation failure is the answer
        return f"the injected trace is not schema-valid: {type(exc).__name__}: {exc}"
    return None


def dry_run(
    spec: AblationSpec,
    candidates: Sequence[Trace],
    harness: Any,
    inputs_by_id: Mapping[str, InputSpec],
    *,
    dataset_id: str = "",
    personas: Mapping[str, Persona] | None = None,
    max_turns: int = 1,
) -> tuple[Stage, str] | None:
    """Inject into ONE sample and report the first failure, or None if clean."""
    sample = _sample_for(spec, candidates)
    try:
        if spec.mode == "replay_edit":
            injected, _record = apply_replay_edit(
                sample,
                spec,
                harness,
                ablation_id=f"dryrun-{spec.error_id}",
                dataset_id=dataset_id,
                store_result=False,
            )
        else:
            input_spec = inputs_by_id.get(sample.input_id)
            if input_spec is None:
                return (
                    "dry_run",
                    f"trace {sample.trace_id} names input {sample.input_id!r}, which is "
                    f"not in the input dataset — it cannot be re-run with a fault",
                )
            persona = (personas or {}).get(input_spec.persona_id or "")
            injected, _record = apply_dependency_fault(
                input_spec,
                spec,
                harness,
                baseline=sample,
                ablation_id=f"dryrun-{spec.error_id}",
                dataset_id=dataset_id,
                persona=persona,
                max_turns=max_turns,
                store_result=False,
            )
    except (FaultNotActivated, UndeclaredFault) as exc:
        return "dry_run", f"{type(exc).__name__}: {exc}"
    except InjectionError as exc:
        return "dry_run", f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - a dry run must never abort the batch
        return "dry_run", f"unexpected {type(exc).__name__}: {exc}"

    schema_problem = _schema_valid(injected)
    if schema_problem:
        return "schema", schema_problem
    return None


def _adapt(
    proposal: ProposedError,
    reason: str,
    agent: AblationAgent,
    digest: CorpusDigest,
    reasons: Sequence[str],
) -> tuple[ProposedError, str | None]:
    """Apply the reason-specific adaptation, or explain why there is none left."""
    lowered = reason.lower()
    if proposal.issue.injection_mode == "replay_edit" and (
        SelfCorrected.__name__.lower() in lowered
        or CorruptionLost.__name__.lower() in lowered
        or "retract" in lowered
    ):
        try:
            revised = agent.revise_corruption(proposal, digest, reasons)
        except Exception as exc:  # noqa: BLE001 - a failed revision drops the error
            return proposal, f"the corruption could not be re-authored: {type(exc).__name__}: {exc}"
        return proposal.model_copy(update={"corruption": revised}), None

    if proposal.issue.injection_mode == "dependency_fault" and proposal.fault is not None:
        if "undeclaredfault" in lowered:
            return proposal, "the shim is not part of the app's declared fault surface"
        if "notactivated" in lowered or "activat" in lowered:
            nxt = rotate_behavior(proposal.fault.shim, proposal.fault.behavior)
            if nxt is None:
                return proposal, (
                    f"every behavior in the vocabulary for shim "
                    f"{proposal.fault.shim!r} failed to activate"
                )
            return (
                proposal.model_copy(
                    update={"fault": proposal.fault.model_copy(update={"behavior": nxt})}
                ),
                None,
            )
    # Anything else (most often "too few eligible traces") is handled by the
    # next attempt's filter relaxation, which plan_ablation applies by attempt
    # number — no change to the proposal itself.
    return proposal, None


def validate_specs(
    proposals: Sequence[ProposedError],
    traces: Sequence[Trace],
    ablate_input_ids: set[str],
    harness: Any,
    inputs_by_id: Mapping[str, InputSpec],
    *,
    agent: AblationAgent,
    digest: CorpusDigest,
    min_eligible: int = 5,
    dataset_id: str = "",
    replayable_trace_ids: set[str] | None = None,
    personas: Mapping[str, Persona] | None = None,
    max_turns: int = 1,
    max_replans: int = MAX_REPLANS,
) -> ValidationOutcome:
    """Validate every proposal, re-planning up to `max_replans` times each."""
    outcome = ValidationOutcome()
    for proposal in proposals:
        error_id = proposal.issue.error_id
        current = proposal
        reasons: list[str] = []
        drop_reason: str | None = None

        for attempt in range(max_replans + 1):
            spec = plan_ablation(current, attempt=attempt)
            candidates = filters.eligible(traces, spec.filter, ablate_input_ids)
            if spec.mode == "replay_edit" and replayable_trace_ids is not None:
                candidates = [t for t in candidates if t.trace_id in replayable_trace_ids]

            if len(candidates) < min_eligible:
                reason = (
                    f"filter matched {len(candidates)} trace(s) in the ablate set, "
                    f"below min_eligible={min_eligible}"
                )
                outcome.failures.append(
                    ValidationFailure(
                        error_id=error_id, attempt=attempt, stage="filter", reason=reason
                    )
                )
                reasons.append(reason)
                drop_reason = reason
                if attempt == max_replans:
                    break  # no attempt left to spend an adaptation on
                current, hard_stop = _adapt(current, reason, agent, digest, reasons)
                if hard_stop:
                    drop_reason = hard_stop
                    break
                continue

            failure = dry_run(
                spec,
                candidates,
                harness,
                inputs_by_id,
                dataset_id=dataset_id,
                personas=personas,
                max_turns=max_turns,
            )
            if failure is None:
                outcome.specs.append(spec)
                outcome.proposals[error_id] = current
                drop_reason = None
                break

            stage, reason = failure
            outcome.failures.append(
                ValidationFailure(
                    error_id=error_id, attempt=attempt, stage=stage, reason=reason
                )
            )
            reasons.append(reason)
            drop_reason = reason
            if attempt == max_replans:
                break  # no attempt left to spend an adaptation on
            current, hard_stop = _adapt(current, reason, agent, digest, reasons)
            if hard_stop:
                drop_reason = hard_stop
                break

        if drop_reason is not None:
            message = (
                f"{error_id} ({current.issue.injection_mode}) dropped after "
                f"{len(reasons)} attempt(s): {drop_reason}"
            )
            log.warning("%s", message)
            outcome.dropped[error_id] = message
    return outcome
