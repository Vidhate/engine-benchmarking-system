"""Step 3 — the quality gate and the bounded re-plan loop."""

from __future__ import annotations

from benchmark.ablation.agent import CorpusDigest, ScriptedAblationAgent
from benchmark.ablation.validate import validate_specs
from benchmark.schemas.ablation import FilterStep

from .conftest import FakeHarness, make_proposal


def _validate(
    proposals, traces, inputs, harness, *, agent=None, min_eligible=2, ablate_ids=None, **kwargs
):
    return validate_specs(
        proposals,
        traces.traces,
        ablate_ids if ablate_ids is not None else {i.input_id for i in inputs.inputs},
        harness,
        {i.input_id: i for i in inputs.inputs},
        agent=agent or ScriptedAblationAgent(),
        digest=CorpusDigest(),
        min_eligible=min_eligible,
        dataset_id="ds",
        **kwargs,
    )


def test_a_clean_spec_passes_on_the_first_attempt(traces, inputs, harness):
    outcome = _validate([make_proposal()], traces, inputs, harness)
    assert [s.error_id for s in outcome.specs] == ["E-hallucination-00"]
    assert outcome.failures == []
    assert outcome.dropped == {}


def test_the_gate_counts_only_traces_inside_the_ablate_set(traces, inputs, harness):
    """Control inputs are not part of the population, ever."""
    outcome = validate_specs(
        [make_proposal()],
        traces.traces,
        {"safe-00"},  # a one-input ablate set
        harness,
        {i.input_id: i for i in inputs.inputs},
        agent=ScriptedAblationAgent(),
        digest=CorpusDigest(),
        min_eligible=5,
        dataset_id="ds",
    )
    assert outcome.specs == []
    assert "below min_eligible=5" in outcome.dropped["E-hallucination-00"]


def test_a_deliberately_broken_spec_is_rejected_with_the_reason_surfaced(
    traces, inputs, harness
):
    """An over-specific filter: the error is not expressible in this corpus."""
    proposal = make_proposal(
        filter_steps=[
            FilterStep(field="turns[*].final_response", op="contains", value="quantum tunnelling"),
            FilterStep(field="span_names", op="eq", value="a_tool_this_app_does_not_have"),
        ]
    )
    outcome = _validate([proposal], traces, inputs, harness, min_eligible=99)
    assert outcome.specs == []
    reasons = [f.reason for f in outcome.failures]
    assert all("min_eligible=99" in r for r in reasons)
    assert [f.stage for f in outcome.failures] == ["filter", "filter", "filter"]
    assert [f.attempt for f in outcome.failures] == [0, 1, 2], "the loop is bounded at 2 re-plans"
    assert "E-hallucination-00" in outcome.dropped


def test_relaxation_rescues_a_spec_whose_only_problem_was_over_specification(
    traces, inputs, harness
):
    proposal = make_proposal(
        filter_steps=[FilterStep(field="span_names", op="eq", value="not_a_real_span")]
    )
    outcome = _validate([proposal], traces, inputs, harness)
    assert [s.error_id for s in outcome.specs] == ["E-hallucination-00"]
    assert outcome.failures, "the first attempt must still be reported"
    assert outcome.dropped == {}


def test_a_self_correcting_app_triggers_a_re_authored_corruption(
    target_cfg, store, traces, inputs
):
    harness = FakeHarness(target_cfg, store, self_corrects=True)
    agent = ScriptedAblationAgent()
    # An ablate set of multi-turn inputs only: every candidate has a downstream
    # that retracts, so relaxing the filter cannot rescue this one.
    outcome = _validate(
        [make_proposal()],
        traces,
        inputs,
        harness,
        agent=agent,
        min_eligible=1,
        ablate_ids={"mt-00", "mt-01"},
    )
    assert agent.revise_calls, "the agent must be asked to re-author the corruption"
    assert "SelfCorrected" in agent.revise_calls[0][1][0]
    assert len(agent.revise_calls) == 2, "one re-authoring per re-plan, bounded"
    assert outcome.specs == [], "a corpus that always self-corrects has no valid spec"
    assert "E-hallucination-00" in outcome.dropped


def test_a_relaxed_filter_can_rescue_an_error_the_first_attempt_self_corrected(
    target_cfg, store, traces, inputs
):
    """Relaxation is not just about counts: it moves the sample too."""
    harness = FakeHarness(target_cfg, store, self_corrects=True)
    proposal = make_proposal(
        filter_steps=[FilterStep(field="mode", op="eq", value="multi_turn")]
    )
    outcome = _validate([proposal], traces, inputs, harness, min_eligible=1)
    # attempt 0 samples a multi-turn trace and is retracted; attempt 1 drops the
    # mode step, samples a single-turn trace, and there is no downstream to
    # retract anything.
    assert [s.error_id for s in outcome.specs] == ["E-hallucination-00"]
    assert any("SelfCorrected" in f.reason for f in outcome.failures)


def test_a_fault_that_never_activates_rotates_the_behaviour_then_drops(
    target_cfg, store, traces, inputs
):
    harness = FakeHarness(target_cfg, store, fault_activates=False)
    proposal = make_proposal(mode="dependency_fault")
    outcome = _validate([proposal], traces, inputs, harness, min_eligible=1)
    assert outcome.specs == []
    assert len(harness.fault_runs) == 3, "one dry run per attempt, each a new behaviour"
    behaviours = [run["fault"]["behavior"] for run in harness.fault_runs]
    assert len(set(behaviours)) == 3, behaviours
    assert "E-retrieval_failure" not in outcome.dropped
    assert "E-hallucination-00" in outcome.dropped


def test_dry_runs_never_persist_a_rejected_injection(target_cfg, store, traces, inputs):
    before = set(store.list_ids())
    harness = FakeHarness(target_cfg, store, self_corrects=True)
    proposal = make_proposal(filter_steps=[FilterStep(field="mode", op="eq", value="multi_turn")])
    _validate([proposal], traces, inputs, harness, min_eligible=1)
    assert set(store.list_ids()) == before


def test_replay_edit_candidates_are_limited_to_live_threads(traces, inputs, harness):
    """Liveness limits MULTI-TURN replay candidates only: a multi-turn-filtered
    proposal with zero live threads is dropped..."""
    from benchmark.schemas.ablation import FilterStep

    proposal = make_proposal(
        filter_steps=[FilterStep(field="mode", op="eq", value="multi_turn")]
    )
    outcome = _validate(
        [proposal], traces, inputs, harness, replayable_trace_ids=set(), max_replans=0
    )
    assert outcome.specs == []
    assert "below min_eligible" in outcome.dropped["E-hallucination-00"]


def test_single_turn_replay_candidates_bypass_the_liveness_filter(traces, inputs, harness):
    """...while single-turn candidates survive dead threads: M=1 replay_edit is
    a post-hoc edit that never forks a thread (the crash-resume scenario —
    a corpus collected across dead server lifetimes stays fully eligible)."""
    outcome = _validate(
        [make_proposal()], traces, inputs, harness, replayable_trace_ids=set()
    )
    assert outcome.specs, "single-turn candidates must survive an empty liveness set"


def test_the_validated_proposal_is_returned_so_apply_uses_the_revised_text(
    traces, inputs, harness
):
    proposal = make_proposal()
    outcome = _validate([proposal], traces, inputs, harness)
    assert outcome.proposals["E-hallucination-00"].issue.title == proposal.issue.title
