"""The batch runner: [N] inputs -> [N,M] outputs + [N,M,T] traces."""

from __future__ import annotations

import json

from benchmark.harness.collector import LangSmithCollector
from benchmark.harness.ids import session_id_for, trace_id_for
from benchmark.harness.runner import Harness, Quarantine, run_harness
from benchmark.schemas.configs import TargetAppConfig
from benchmark.schemas.inputs import GenerationConfig, InputDataset, InputSpec, Persona
from benchmark.schemas.io import stamp_dataset_id
from benchmark.schemas.traces import Trace
from benchmark.tracing.store import LocalTraceStore
from tests.harness.conftest import FakeLangSmithClient, FakeTargetApp

CFG = TargetAppConfig(
    base_url="http://127.0.0.1:2024",
    assistant_id="target_app",
    langsmith_project="engine-bench-target",
    fault_configurable_keys={"retriever": "fault_retriever", "tool": "fault_tool"},
    max_turns_supported=8,
)

PERSONA = Persona(
    persona_id="new_user",
    name="New User",
    kind="target",
    description="A first-time user.",
    goals=["understand a feature"],
)


def make_inputs(n_single=3, n_multi=0, max_turns=3) -> InputDataset:
    inputs = [
        InputSpec(
            input_id=f"safe-{i}",
            mode="single_turn",
            dim_id="topic",
            variation=f"v{i}",
            prompt=f"question {i}",
        )
        for i in range(n_single)
    ]
    inputs += [
        InputSpec(
            input_id=f"mt-{i}",
            mode="multi_turn",
            dim_id="topic",
            variation=f"v{i}",
            persona_id=PERSONA.persona_id,
            scenario=f"scenario {i}",
        )
        for i in range(n_multi)
    ]
    cfg = GenerationConfig(personas=[PERSONA], mode="mixed", max_turns=max_turns)
    return stamp_dataset_id(InputDataset(generation_config=cfg, inputs=inputs))


def build(tmp_path, inputs, *, simulator_script=None, **app_kwargs):
    ls = FakeLangSmithClient([])
    app = FakeTargetApp(ls, **app_kwargs)
    collector = LangSmithCollector(
        CFG.langsmith_project,
        client=ls,
        cfg=CFG,
        poll_interval_s=0.0,
        sleep=lambda _s: None,
        root_timeout_s=0.0,
        child_timeout_s=0.0,
        settle_polls=1,
    )
    store = LocalTraceStore(tmp_path / "traces")
    quarantine = Quarantine(tmp_path / "quarantine")
    from benchmark.harness.simulator import DONE_TOKEN, ScriptedUserSimulator

    simulator = ScriptedUserSimulator(simulator_script or ["turn one", "turn two", DONE_TOKEN])
    harness = Harness(
        CFG,
        store,
        client=app,
        collector=collector,
        simulator=simulator,
        quarantine=quarantine,
        concurrency=2,
    )
    return harness, app, store, quarantine, inputs


# ------------------------------------------------------------------ happy path

def test_inputs_become_stored_schema_valid_traces_and_outputs(tmp_path):
    harness, app, store, _q, inputs = build(tmp_path, make_inputs(3))

    outputs, traces = harness.run_batch(inputs)

    assert len(traces.traces) == 3
    assert len(outputs.outputs) == 3
    for trace in traces.traces:
        assert store.exists(trace.trace_id)
        assert Trace.model_validate_json(store.get(trace.trace_id).model_dump_json())
        assert trace.status == "ok"
        assert trace.turns and trace.turns[0].spans
    assert {o.input_id for o in outputs.outputs} == {"safe-0", "safe-1", "safe-2"}


def test_the_session_id_is_the_hash_of_dataset_and_input_and_reaches_run_metadata(tmp_path):
    inputs = make_inputs(1)
    harness, app, store, _q, _ = build(tmp_path, inputs)
    harness.run_batch(inputs)

    expected = session_id_for(inputs.dataset_id, "safe-0")
    assert app.calls[0]["session_id"] == expected
    assert store.exists(trace_id_for(expected))


def test_datasets_carry_lineage_back_to_the_input_dataset(tmp_path):
    inputs = make_inputs(2)
    harness, _app, _store, _q, _ = build(tmp_path, inputs)

    outputs, traces = harness.run_batch(inputs)

    assert traces.parent_dataset_id == inputs.dataset_id
    assert outputs.parent_dataset_id == inputs.dataset_id
    assert traces.dataset_id and outputs.dataset_id
    assert traces.dataset_id != outputs.dataset_id


def test_results_are_emitted_in_a_deterministic_order(tmp_path):
    inputs = make_inputs(5)
    harness, _app, _store, _q, _ = build(tmp_path, inputs)
    _outputs, traces = harness.run_batch(inputs)
    assert [t.input_id for t in traces.traces] == sorted(t.input_id for t in traces.traces)


def test_inputs_run_concurrently_but_never_more_than_the_cap(tmp_path):
    import time

    def slow_answer(message):
        time.sleep(0.02)
        return f"answer to: {message}"

    inputs = make_inputs(8)
    harness, app, _store, _q, _ = build(tmp_path, inputs, answer=slow_answer)
    harness.run_batch(inputs)
    # Both halves matter: >1 proves the batch is actually parallel, <=2 proves
    # the semaphore caps it so the target app is not overrun.
    assert app.max_in_flight == 2


def test_no_leak_tokens_survive_a_whole_batch(tmp_path):
    inputs = make_inputs(3)
    harness, _app, store, _q, _ = build(tmp_path, inputs)
    harness.run_batch(inputs)

    for trace_id in store.list_ids():
        blob = json.dumps(
            store.get(trace_id).model_dump(mode="json", exclude={"ablation_ids"})
        ).lower()
        for token in ("fault_", "shim", "supportchatmodel", "ablat"):
            assert token not in blob


# ------------------------------------------------------------------ idempotency

def test_rerunning_the_same_batch_skips_inputs_that_already_have_an_ok_trace(tmp_path):
    inputs = make_inputs(3)
    harness, app, store, _q, _ = build(tmp_path, inputs)

    first_outputs, first_traces = harness.run_batch(inputs)
    calls_after_first = len(app.calls)

    second_outputs, second_traces = harness.run_batch(inputs)

    assert len(app.calls) == calls_after_first, "a completed input was re-invoked"
    assert harness.stats["skipped"] == 3
    assert [t.trace_id for t in second_traces.traces] == [t.trace_id for t in first_traces.traces]
    assert second_outputs.dataset_id == first_outputs.dataset_id
    assert len(store.list_ids()) == 3


def test_an_app_error_trace_is_retried_on_the_next_run(tmp_path):
    inputs = make_inputs(2)
    failing = session_id_for(inputs.dataset_id, "safe-1")
    harness, app, store, _q, _ = build(tmp_path, inputs, fail_sessions={failing})

    harness.run_batch(inputs)
    assert store.get(trace_id_for(failing)).status == "app_error"
    calls_after_first = len(app.calls)

    harness.run_batch(inputs)
    retried = [c for c in app.calls[calls_after_first:] if c["session_id"] == failing]
    assert retried, "an app_error trace must not be treated as done"
    assert harness.stats["skipped"] == 1  # only the ok one


# ------------------------------------------------------- failure handling

def test_app_errors_are_kept_as_organic_signal(tmp_path):
    inputs = make_inputs(2)
    failing = session_id_for(inputs.dataset_id, "safe-0")
    harness, _app, store, _q, _ = build(tmp_path, inputs, fail_sessions={failing})

    _outputs, traces = harness.run_batch(inputs)

    kept = next(t for t in traces.traces if t.input_id == "safe-0")
    assert kept.status == "app_error"
    assert store.exists(kept.trace_id)
    assert kept.turns[0].user_message == "question 0"


def test_uncollectable_traces_are_quarantined_with_a_reason_never_silently_dropped(tmp_path):
    inputs = make_inputs(2)
    silent = session_id_for(inputs.dataset_id, "safe-1")
    harness, _app, store, quarantine, _ = build(tmp_path, inputs, silent_sessions={silent})

    _outputs, traces = harness.run_batch(inputs)

    assert [t.input_id for t in traces.traces] == ["safe-0"]
    assert not store.exists(trace_id_for(silent))
    quarantined = quarantine.list_ids()
    assert quarantined == [silent]
    record = quarantine.get(silent)
    assert record["reason"]
    assert record["input_id"] == "safe-1"
    assert harness.stats["quarantined"] == 1


# ------------------------------------------------------------------ multi-turn

def test_a_persona_conversation_becomes_one_trace_with_per_turn_spans(tmp_path):
    from benchmark.harness.simulator import DONE_TOKEN

    inputs = make_inputs(0, n_multi=1)
    harness, app, store, _q, _ = build(
        tmp_path, inputs, simulator_script=["hi there", "and refunds?", DONE_TOKEN]
    )

    _outputs, traces = harness.run_batch(inputs)

    trace = traces.traces[0]
    assert trace.mode == "multi_turn"
    assert len(trace.turns) == 2
    assert [t.turn_index for t in trace.turns] == [0, 1]
    assert [t.user_message for t in trace.turns] == ["hi there", "and refunds?"]
    assert all(turn.spans for turn in trace.turns)
    # One thread for the whole conversation.
    assert len({c["thread_id"] for c in app.calls}) == 1


def test_the_conversation_stops_at_max_turns_even_without_a_done_token(tmp_path):
    inputs = make_inputs(0, n_multi=1, max_turns=2)
    harness, app, _store, _q, _ = build(
        tmp_path, inputs, simulator_script=["a", "b", "c", "d", "e"]
    )

    _outputs, traces = harness.run_batch(inputs)
    assert len(traces.traces[0].turns) == 2
    assert len(app.calls) == 2


def test_max_turns_is_capped_by_what_the_app_declares_it_supports(tmp_path):
    inputs = make_inputs(0, n_multi=1, max_turns=99)
    cfg = CFG.model_copy(update={"max_turns_supported": 2})
    harness, app, _store, _q, _ = build(tmp_path, inputs, simulator_script=["a", "b", "c", "d"])
    harness.cfg = cfg

    harness.run_batch(inputs)
    assert len(app.calls) == 2


def test_the_simulator_sees_the_persona_and_the_scenario(tmp_path):
    from benchmark.harness.simulator import DONE_TOKEN

    inputs = make_inputs(0, n_multi=1)
    harness, _app, _store, _q, _ = build(tmp_path, inputs, simulator_script=["hi", DONE_TOKEN])
    harness.run_batch(inputs)

    call = harness.simulator.calls[0]
    assert call["persona"].persona_id == "new_user"
    assert call["scenario"] == "scenario 0"


def test_a_multi_turn_input_with_an_unknown_persona_is_quarantined(tmp_path):
    inputs = make_inputs(0, n_multi=1)
    inputs.inputs[0].persona_id = "ghost"
    inputs = stamp_dataset_id(inputs)
    harness, app, _store, quarantine, _ = build(tmp_path, inputs)

    _outputs, traces = harness.run_batch(inputs)
    assert traces.traces == []
    assert app.calls == []
    assert "ghost" in quarantine.get(quarantine.list_ids()[0])["reason"]


# ------------------------------------------------------------ top-level entry

def test_run_harness_is_the_documented_top_level_entrypoint(tmp_path):
    inputs = make_inputs(2)
    ls = FakeLangSmithClient([])
    app = FakeTargetApp(ls)
    collector = LangSmithCollector(
        CFG.langsmith_project, client=ls, cfg=CFG, poll_interval_s=0.0,
        sleep=lambda _s: None, root_timeout_s=0.0, child_timeout_s=0.0, settle_polls=1,
    )
    store = LocalTraceStore(tmp_path / "traces")

    outputs, traces = run_harness(
        inputs, CFG, store, client=app, collector=collector, concurrency=2,
        quarantine=Quarantine(tmp_path / "q"),
    )
    assert len(traces.traces) == 2
    assert traces.parent_dataset_id == inputs.dataset_id
    assert all(o.responses for o in outputs.outputs)
