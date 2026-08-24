"""The two injection mechanics, against a fake harness."""

from __future__ import annotations

import json

import pytest

from benchmark.ablation.inject import (
    ConsistencyError,
    CorruptionLost,
    DeadThreadRefs,
    InjectionError,
    SelfCorrected,
    apply_dependency_fault,
    apply_replay_edit,
    assert_threads_alive,
    corrupt_turn,
    retraction_in,
)
from benchmark.ablation.plan import plan_ablation
from benchmark.schemas.traces import Trace

from .conftest import MARKER, FakeHarness, make_proposal, make_trace, make_turn


def _spec(**kwargs):
    return plan_ablation(make_proposal(**kwargs))


# ------------------------------------------------------ turn-k consistency

def test_corrupting_a_turn_rewrites_its_last_llm_span_to_match():
    turn = make_turn("t", 0, final_response="Refunds within 30 days.")
    corrupted, rewritten, original = corrupt_turn(turn, "Refunds within 365 days, case NBX-1.")
    llm = [s for s in corrupted.spans if s.span_type == "llm"][0]
    assert llm.outputs["generations"][-1][-1]["text"] == corrupted.final_response
    assert llm.span_id in rewritten
    assert "30 days" in original, "the pre-edit llm output is what the record audits"


def test_corrupting_a_turn_also_fixes_the_agent_spans_message_list():
    turn = make_turn("t", 0)
    corrupted, rewritten, _ = corrupt_turn(turn, "brand new answer, case NBX-1")
    agent = [s for s in corrupted.spans if s.span_type == "agent"][0]
    assert agent.outputs["messages"][-1]["content"] == "brand new answer, case NBX-1"
    assert agent.span_id in rewritten


def test_the_original_turn_is_not_mutated():
    turn = make_turn("t", 0, final_response="original")
    corrupt_turn(turn, "replacement")
    assert turn.final_response == "original"
    assert turn.spans[-1].outputs["generations"][-1][-1]["text"] == "original"


def test_a_turn_with_no_llm_span_cannot_be_made_consistent():
    turn = make_turn("t", 0)
    turn.spans = [s for s in turn.spans if s.span_type != "llm"]
    with pytest.raises(ConsistencyError, match="no llm span"):
        corrupt_turn(turn, "whatever")


# ------------------------------------------------------- Mode A, M = 1

def test_single_turn_mode_a_is_a_post_hoc_edit_with_no_replay(harness, store):
    trace = store.get("trace-safe-00")
    ablated, record = apply_replay_edit(
        trace, _spec(), harness, ablation_id="abl-1", dataset_id="ds"
    )
    assert harness.replays == [], "M=1 has no downstream to regenerate"
    assert ablated.turns[0].final_response.startswith("I have escalated")
    assert MARKER in ablated.model_dump_json()
    assert ablated.trace_id != trace.trace_id
    assert ablated.ablation_ids == ["abl-1"]
    assert record.before_after[0] == (trace.turns[0].final_response,
                                      ablated.turns[0].final_response)
    assert store.exists(ablated.trace_id)


def test_the_source_trace_is_left_immutable(harness, store):
    trace = store.get("trace-safe-00")
    before = trace.model_dump_json()
    apply_replay_edit(trace, _spec(), harness, ablation_id="abl-1", dataset_id="ds")
    assert trace.model_dump_json() == before
    assert store.get("trace-safe-00").model_dump_json() == before


# ------------------------------------------------------- Mode A, M > 1

def test_multi_turn_mode_a_splices_the_regenerated_tail(harness, store):
    trace = store.get("trace-mt-00")
    assert len(trace.turns) == 3
    ablated, record = apply_replay_edit(
        trace, _spec(), harness, ablation_id="abl-2", dataset_id="ds"
    )
    assert len(harness.replays) == 1
    call = harness.replays[0]
    assert call["remaining"] == [t.user_message for t in trace.turns[1:]]
    assert call["corrupted_state"]["messages"][0]["content"].startswith("I have escalated")

    assert [t.turn_index for t in ablated.turns] == [0, 1, 2]
    assert ablated.turns[0].final_response.startswith("I have escalated")
    assert ablated.turns[1].final_response == "Understood — anything else?"
    assert record.actions_applied[0].params["regenerated_turns"] == 2
    assert ablated.metadata["replay_fork_checkpoint_id"]


def test_turns_before_k_survive_untouched(harness, store):
    trace = store.get("trace-mt-00")
    ablated, _ = apply_replay_edit(
        trace,
        plan_ablation(make_proposal(turn_index=1)),
        harness,
        ablation_id="abl-3",
        dataset_id="ds",
    )
    assert ablated.turns[0].model_dump() == trace.turns[0].model_dump()
    assert ablated.turns[1].final_response.startswith("I have escalated")


def test_a_downstream_self_correction_is_refused_not_recorded(target_cfg, store):
    """The app really does re-search and contradict injected content."""
    harness = FakeHarness(target_cfg, store, self_corrects=True)
    trace = store.get("trace-mt-00")
    with pytest.raises(SelfCorrected, match="retracted"):
        apply_replay_edit(trace, _spec(), harness, ablation_id="abl-4", dataset_id="ds")


def test_a_marker_that_does_not_survive_the_splice_is_refused(harness, store):
    spec = _spec()
    # A replacement that never contained the marker: the splice "succeeds" but
    # T* does not carry the corruption, so there is nothing to score against.
    spec.ablation_actions[0].params["replacement"] = "an ordinary looking answer"
    trace = store.get("trace-safe-00")
    with pytest.raises(CorruptionLost, match="marker"):
        apply_replay_edit(trace, spec, harness, ablation_id="abl-5", dataset_id="ds")


def test_a_turn_index_past_the_end_of_the_trace_is_refused(harness, store):
    trace = store.get("trace-safe-00")
    spec = plan_ablation(make_proposal(turn_index=4))
    with pytest.raises(InjectionError, match="outside"):
        apply_replay_edit(trace, spec, harness, ablation_id="abl-6", dataset_id="ds")


def test_retraction_detection_uses_defaults_plus_the_authored_patterns():
    turns = [make_turn("t", 0, final_response="There is no such case on file.")]
    assert retraction_in(turns, []) == "no such"
    quiet = [make_turn("t", 0, final_response="Sure, happy to help.")]
    assert retraction_in(quiet, []) is None
    assert retraction_in(quiet, ["happy to help"]) == "happy to help"


# ------------------------------------------------------- thread liveness

def test_a_corpus_of_dead_threads_fails_loudly_with_guidance(target_cfg, store, traces):
    harness = FakeHarness(target_cfg, store, live_threads=set())
    with pytest.raises(DeadThreadRefs, match="server lifetime"):
        assert_threads_alive(traces.traces, harness)


def test_partially_dead_threads_narrow_the_population_rather_than_aborting(
    target_cfg, store, traces
):
    alive = {"thread-trace-safe-00", "thread-trace-safe-01"}
    harness = FakeHarness(target_cfg, store, live_threads=alive)
    replayable = assert_threads_alive(traces.traces, harness)
    assert replayable == {"trace-safe-00", "trace-safe-01"}


def test_a_trace_with_no_thread_ref_is_never_replayable(target_cfg, store):
    harness = FakeHarness(target_cfg, store)
    trace = make_trace("t-x", "safe-00")
    trace.metadata.pop("thread_id")
    with pytest.raises(DeadThreadRefs):
        assert_threads_alive([trace], harness)


# ---------------------------------------------------------------- Mode C

def test_mode_c_regenerates_the_trace_and_records_activation_evidence(harness, store, inputs):
    baseline = store.get("trace-safe-00")
    input_spec = next(i for i in inputs.inputs if i.input_id == "safe-00")
    spec = plan_ablation(make_proposal(mode="dependency_fault"))
    ablated, record = apply_dependency_fault(
        input_spec, spec, harness, baseline=baseline, ablation_id="abl-c1", dataset_id="ds"
    )
    assert harness.fault_runs[0]["baseline"] == baseline.trace_id, (
        "activation must be a byte-diff against the unarmed baseline, never weak validation"
    )
    assert ablated.trace_id != baseline.trace_id
    assert record.before_after == [("", harness.activation_evidence[ablated.trace_id])]
    assert record.actions_applied == []
    assert store.exists(ablated.trace_id)


def test_mode_c_never_inspects_the_outcome(harness, store, inputs):
    """Ground truth is mechanism-level: the final answer is not evidence."""
    baseline = store.get("trace-safe-00")
    input_spec = next(i for i in inputs.inputs if i.input_id == "safe-00")
    spec = plan_ablation(make_proposal(mode="dependency_fault"))
    _, record = apply_dependency_fault(
        input_spec, spec, harness, baseline=baseline, ablation_id="abl-c2", dataset_id="ds"
    )
    evidence = json.loads(record.before_after[0][1])
    assert "output" in evidence, "evidence is the corrupted span, not the answer"


def test_a_fault_that_never_activates_propagates_the_harness_error(target_cfg, store, inputs):
    from benchmark.harness.faults import FaultNotActivated

    harness = FakeHarness(target_cfg, store, fault_activates=False)
    baseline = store.get("trace-safe-00")
    input_spec = next(i for i in inputs.inputs if i.input_id == "safe-00")
    spec = plan_ablation(make_proposal(mode="dependency_fault"))
    with pytest.raises(FaultNotActivated):
        apply_dependency_fault(
            input_spec, spec, harness, baseline=baseline, ablation_id="abl-c3", dataset_id="ds"
        )


def test_dry_runs_persist_nothing(harness, store):
    before = set(store.list_ids())
    trace = store.get("trace-safe-00")
    ablated, _ = apply_replay_edit(
        trace, _spec(), harness, ablation_id="dry", dataset_id="ds", store_result=False
    )
    assert set(store.list_ids()) == before
    assert isinstance(ablated, Trace)
