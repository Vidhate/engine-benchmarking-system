"""Step 2 — plan the filter + injection strategy for one proposed error.

docs/architecture/04-ablation-engine.md, step 2: per error, a `TraceFilter`
selecting traces where the error *can plausibly exist*, plus either
`ablation_actions` (replay_edit) or a `FaultConfig` (dependency_fault).

Planning is deliberately **deterministic code**, not a second LLM call. The
creative half — what fabricated sentence to put in the assistant's mouth — was
authored in step 1 with the corpus in front of the agent (see `agent.py`).
What is left is mechanical, and being mechanical is what lets the step-3
re-plan loop be unit-tested without a model:

| attempt | what changes |
|---|---|
| 0 | the agent's filter steps, on top of the mode's *necessary* preconditions |
| 1 | the most specific agent step is dropped (filters fail by over-specifying) |
| 2 | every agent step is dropped — preconditions only |

The mode preconditions are never relaxed, because they are not heuristics:
`replay_edit` cannot fork a trace with no thread reference, and a dependency
fault cannot activate in a trace that never exercised that dependency.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchmark.ablation.agent import BEHAVIOR_VOCABULARY, ProposedError
from benchmark.harness.faults import SHIM_TO_SPAN_TYPE
from benchmark.schemas.ablation import AblationAction, AblationSpec, FilterStep, TraceFilter

MAX_REPLANS = 2  # after this the error is dropped with a logged reason


def mode_preconditions(proposal: ProposedError) -> list[FilterStep]:
    """The filter steps that are *necessary*, not heuristic, for this mode."""
    steps = [FilterStep(field="status", op="eq", value="ok")]
    if proposal.issue.injection_mode == "replay_edit":
        # Mode A forks the source trace's thread; a trace with no thread ref
        # cannot be replayed at all (see `inject.assert_threads_alive`).
        steps.append(FilterStep(field="metadata.thread_id", op="exists"))
        return steps
    fault = proposal.fault
    if fault is not None:
        span_type = SHIM_TO_SPAN_TYPE.get(fault.shim)
        if span_type:
            # Activation is only possible where the dependency actually ran.
            steps.append(FilterStep(field="span_types", op="eq", value=span_type))
    return steps


def _relaxed(steps: Sequence[FilterStep], attempt: int) -> list[FilterStep]:
    if attempt <= 0:
        return list(steps)
    if attempt == 1:
        return list(steps[:-1])
    return []


def rotate_behavior(shim: str, behavior: str) -> str | None:
    """The next behavior to try for a shim whose fault never activated."""
    vocabulary = BEHAVIOR_VOCABULARY.get(shim, ())
    if behavior not in vocabulary:
        return vocabulary[0] if vocabulary else None
    index = vocabulary.index(behavior) + 1
    return vocabulary[index] if index < len(vocabulary) else None


def replay_actions(proposal: ProposedError) -> list[AblationAction]:
    """The `str -> str` mutation for a `replay_edit`, as one recorded action.

    Everything the injector and the validator need travels in `params`, so an
    `AblationRecord` is a complete, self-contained account of what was done.
    """
    corruption = proposal.corruption
    if corruption is None:
        raise ValueError(f"{proposal.issue.error_id}: replay_edit plan without a corruption")
    return [
        AblationAction(
            target=f"turns[{corruption.turn_index}].final_response",
            transform="replace",
            params={
                "replacement": corruption.replacement,
                "marker": corruption.marker,
                "retraction_patterns": list(corruption.retraction_patterns),
                "turn_index": corruption.turn_index,
            },
        )
    ]


def plan_ablation(
    proposal: ProposedError,
    *,
    attempt: int = 0,
    target_count: int | None = None,
) -> AblationSpec:
    """`[N,M,T]`, `[E,C_E]` -> `AblationSpec`. See the table in the module docstring."""
    steps = mode_preconditions(proposal) + _relaxed(proposal.filter_steps, attempt)
    mode = proposal.issue.injection_mode
    if mode is None:
        raise ValueError(f"{proposal.issue.error_id}: proposal has no injection_mode")
    return AblationSpec(
        error_id=proposal.issue.error_id,
        mode=mode,
        filter=TraceFilter(steps=steps),
        ablation_actions=replay_actions(proposal) if mode == "replay_edit" else [],
        fault_config=proposal.fault if mode == "dependency_fault" else None,
        target_count=target_count if target_count is not None else proposal.target_count,
    )
