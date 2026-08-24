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

from .conftest import (
    MARKER,
    FakeHarness,
    make_proposal,
    make_span,
    make_trace,
    make_turn,
)


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


def test_only_the_root_agent_span_is_rewritten():
    """A nested agent span is a sub-agent's own record; forcing it to agree
    with the final answer manufactures consistency the app never produced."""
    turn = make_turn("t", 0)
    nested = make_span(
        "t-t0-subagent",
        "agent",
        "sub_agent",
        parent="t-t0-agent",
        outputs={"messages": [{"type": "ai", "content": "sub-agent said this"}]},
        offset_ms=50,
    )
    turn.spans.append(nested)
    corrupted, rewritten, _ = corrupt_turn(turn, "brand new answer")
    assert "t-t0-agent" in rewritten, "the root must be rewritten"
    assert nested.span_id not in rewritten
    survivor = next(s for s in corrupted.spans if s.span_id == nested.span_id)
    assert survivor.outputs["messages"][-1]["content"] == "sub-agent said this"


def test_the_rewritten_llm_spans_token_count_tracks_its_new_output():
    """A span whose completion changed but whose token count still describes
    the original text is a statistical outlier an Engine could learn."""
    turn = make_turn("t", 0, final_response="short")
    llm = [s for s in turn.spans if s.span_type == "llm"][0]
    llm.attributes["tokens"] = 100
    before_len = len(json.dumps(llm.outputs, sort_keys=True, default=str))

    long_answer = "x" * (before_len * 3)
    corrupted, _, _ = corrupt_turn(turn, long_answer)
    after = [s for s in corrupted.spans if s.span_type == "llm"][0]
    assert after.attributes["tokens"] > 100, after.attributes
    # duration is untouched: the model call really did take that long
    assert after.attributes.get("duration_ms") == llm.attributes.get("duration_ms")


def test_a_span_with_no_token_count_gains_none():
    turn = make_turn("t", 0)
    llm = [s for s in turn.spans if s.span_type == "llm"][0]
    llm.attributes.pop("tokens", None)
    corrupted, _, _ = corrupt_turn(turn, "replacement")
    after = [s for s in corrupted.spans if s.span_type == "llm"][0]
    assert "tokens" not in after.attributes, "inventing one is its own tell"


def test_the_marker_check_is_escape_safe_and_scoped_to_turn_k(harness, store):
    """A marker containing a quote is JSON-escaped in a raw dump and would
    never match its own literal."""
    trace = store.get("trace-safe-00")
    marker = 'case "NBX-99"'
    spec = plan_ablation(
        make_proposal(marker=marker, replacement=f"I filed {marker} for you.")
    )
    ablated, _ = apply_replay_edit(
        trace, spec, harness, ablation_id="abl-q", dataset_id="ds", store_result=False
    )
    assert marker in ablated.turns[0].final_response


def test_a_marker_that_only_appears_in_the_tail_does_not_count(target_cfg, store):
    """Scoped to turn k: an echo downstream is not the corruption surviving."""
    harness = FakeHarness(target_cfg, store)
    trace = store.get("trace-mt-00")
    spec = plan_ablation(make_proposal(turn_index=0))
    # The replacement loses its marker, but the fake tail happens to contain it.
    spec.ablation_actions[0].params["replacement"] = "an ordinary answer"
    spec.ablation_actions[0].params["marker"] = "Understood"
    with pytest.raises(CorruptionLost, match="turn 0"):
        apply_replay_edit(
            trace, spec, harness, ablation_id="abl-e", dataset_id="ds", store_result=False
        )


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
    with pytest.raises(InjectionError, match="not a corruptible turn"):
        apply_replay_edit(trace, spec, harness, ablation_id="abl-6", dataset_id="ds")


def test_an_unpinned_corruption_draws_its_turn_seeded_across_the_conversation(
    harness, store
):
    """Always corrupting turn 0 makes the injected position a constant."""
    trace = store.get("trace-mt-00")
    spec = plan_ablation(make_proposal())
    assert spec.ablation_actions[0].params["turn_index"] is None
    chosen = set()
    for seed in range(12):
        _, record = apply_replay_edit(
            trace, spec, harness, ablation_id="abl-k", dataset_id="ds",
            store_result=False, seed=seed,
        )
        chosen.add(record.actions_applied[0].params["turn_index"])
    assert len(chosen) > 1, f"k never varied: {chosen}"
    assert chosen <= {0, 1, 2}


def test_the_drawn_turn_is_stable_for_one_seed(harness, store):
    trace = store.get("trace-mt-00")
    spec = plan_ablation(make_proposal())
    picks = {
        apply_replay_edit(
            trace, spec, harness, ablation_id="abl-k", dataset_id="ds",
            store_result=False, seed=5,
        )[1].actions_applied[0].params["turn_index"]
        for _ in range(4)
    }
    assert len(picks) == 1, picks


def test_an_agent_pinned_turn_wins_over_the_draw(harness, store):
    trace = store.get("trace-mt-00")
    spec = plan_ablation(make_proposal(turn_index=1))
    for seed in range(6):
        _, record = apply_replay_edit(
            trace, spec, harness, ablation_id="abl-k", dataset_id="ds",
            store_result=False, seed=seed,
        )
        assert record.actions_applied[0].params["turn_index"] == 1


def test_only_turns_that_can_be_made_consistent_are_drawable(harness, store):
    from benchmark.ablation.inject import valid_turn_indices

    trace = store.get("trace-mt-00")
    trace.turns[0].spans = [s for s in trace.turns[0].spans if s.span_type != "llm"]
    assert valid_turn_indices(trace) == [1, 2]
    spec = plan_ablation(make_proposal())
    for seed in range(10):
        _, record = apply_replay_edit(
            trace, spec, harness, ablation_id="abl-k", dataset_id="ds",
            store_result=False, seed=seed,
        )
        assert record.actions_applied[0].params["turn_index"] != 0


def test_an_unconditional_first_person_retraction_is_caught():
    turns = [make_turn("t", 0, final_response="I apologise — I made an error earlier.")]
    assert retraction_in(turns, []) in ("i apologi", "i made an error")
    quiet = [make_turn("t", 0, final_response="Sure, happy to help.")]
    assert retraction_in(quiet, []) is None
    assert retraction_in(quiet, ["happy to help"]) == "happy to help"


def test_an_anchored_phrase_only_counts_near_the_markers_own_terms():
    """"no such" is a retraction of the fabricated id, and ordinary English
    everywhere else. The bare phrase used to fire on both."""
    retracts = [make_turn("t", 0, final_response="There is no such case as NBX-4471 on file.")]
    assert retraction_in(retracts, [], MARKER) == "no such"
    # Same phrase, nothing to do with the injected marker.
    innocent = [
        make_turn("t", 0, final_response="There's no such setting in the mobile app yet.")
    ]
    assert retraction_in(innocent, [], MARKER) is None


def test_the_previously_false_positive_prone_phrases_no_longer_fire_alone():
    for text in (
        "I've filed a correction request with billing.",
        "There is no such option on the free plan.",
        "That does not exist in the current release.",
    ):
        turns = [make_turn("t", 0, final_response=text)]
        assert retraction_in(turns, [], MARKER) is None, text


def test_the_scan_covers_span_output_not_just_the_final_response():
    """The app can contradict an injected claim in a retrieval span while the
    answer says nothing about it."""
    turn = make_turn("t", 0, final_response="Here's what I found.")
    turn.spans[-1].outputs = {
        "generations": [[{"text": "I made an error in my previous message."}]]
    }
    assert retraction_in([turn], []) == "i made an error"


# ------------------------------------------------------------- fork point

def test_the_fork_point_is_the_recorded_checkpoint_not_a_boundary_position(harness, store):
    """A thread's boundary list is not the trace's turn space.

    Measured live: the server writes several checkpoints per answer, so turn k
    of the trace is boundary 2k, and asking for `turn_index=k` either forked at
    the wrong turn or was refused outright. The collector already recorded a
    checkpoint id per turn; that id is the key.
    """
    from benchmark.ablation.inject import fork_point

    trace = store.get("trace-mt-00")
    assert harness.turn_boundaries("thread-trace-mt-00")[1][0] != "ckpt-trace-mt-00-1", (
        "the fixture must reproduce the duplicated-boundary layout"
    )
    checkpoint_id, message_id = fork_point(
        trace, 1, harness, trace.turns[1].final_response
    )
    assert (checkpoint_id, message_id) == ("ckpt-trace-mt-00-1", "msg-trace-mt-00-1")


def test_a_replay_forks_the_turn_it_says_it_forked(harness, store):
    trace = store.get("trace-mt-00")
    spec = plan_ablation(make_proposal(turn_index=1))
    apply_replay_edit(
        trace, spec, harness, ablation_id="abl-fp", dataset_id="ds", store_result=False
    )
    assert harness.replays[-1]["checkpoint_ref"] == "ckpt-trace-mt-00-1"
    state = harness.replays[-1]["corrupted_state"]
    assert state["messages"][0]["id"] == "msg-trace-mt-00-1"


def test_an_earlier_replays_fork_on_the_same_thread_does_not_move_the_fork_point(
    harness, store
):
    """Mode A forks the SAME thread, so a previous injection's regenerated
    answers are appended to the same history."""
    from benchmark.ablation.inject import fork_point

    trace = store.get("trace-mt-00")
    fork = trace.model_copy(deep=True, update={"trace_id": "regen-earlier"})
    store.put(fork)  # same thread_id, extra boundaries
    assert len(harness.turn_boundaries("thread-trace-mt-00")) == 12
    assert fork_point(trace, 1, harness, trace.turns[1].final_response) == (
        "ckpt-trace-mt-00-1",
        "msg-trace-mt-00-1",
    )


def test_a_trace_with_no_recorded_checkpoints_falls_back_to_the_answer_text(
    harness, store
):
    from benchmark.ablation.inject import fork_point

    trace = store.get("trace-mt-00")
    trace.metadata.pop("turn_checkpoints")
    checkpoint_id, _message_id = fork_point(
        trace, 1, harness, trace.turns[1].final_response
    )
    assert checkpoint_id == "ckpt-trace-mt-00-1"


def test_a_fallback_that_matches_two_different_answers_is_refused(harness, store):
    """Two turns ending in the same words are ordinary; forking at the wrong
    one would mislabel the ground-truth turn index silently."""
    from benchmark.ablation.inject import fork_point

    trace = store.get("trace-mt-00")
    trace.metadata.pop("turn_checkpoints")
    for turn in trace.turns:
        turn.final_response = "Anything else I can help with?"
    store.put(trace)
    with pytest.raises(InjectionError, match="different assistant messages"):
        fork_point(trace, 1, harness, "Anything else I can help with?")


def test_a_fork_point_that_no_longer_exists_is_refused_with_the_reason(harness, store):
    """The recorded id is authoritative; when the thread has neither it nor the
    answer text, the injection is refused rather than forked at a guess."""
    from benchmark.ablation.inject import fork_point

    trace = store.get("trace-mt-00")
    trace.metadata["turn_checkpoints"] = [
        {"turn_index": 1, "checkpoint_id": "ckpt-the-server-dropped"}
    ]
    with pytest.raises(InjectionError, match="no checkpoint on thread"):
        fork_point(trace, 1, harness, "an answer this thread never produced")


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
