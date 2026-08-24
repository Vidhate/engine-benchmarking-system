"""The two APIs Phase 5 consumes: `replay` (Mode A) and `run_with_faults` (Mode C).

Contracts are pinned here so Phase 5 can build against them before any live run.
"""

from __future__ import annotations

import pytest

from benchmark.harness.collector import LangSmithCollector
from benchmark.harness.faults import FaultNotActivated, UndeclaredFault
from benchmark.harness.runner import Harness, Quarantine, replay, run_with_faults
from benchmark.schemas.ablation import FaultConfig
from benchmark.schemas.configs import TargetAppConfig
from benchmark.schemas.inputs import InputSpec
from benchmark.tracing.store import LocalTraceStore
from tests.harness.conftest import FakeLangSmithClient, FakeTargetApp

CFG = TargetAppConfig(
    base_url="http://127.0.0.1:2024",
    assistant_id="target_app",
    langsmith_project="engine-bench-target",
    fault_configurable_keys={"retriever": "fault_retriever", "tool": "fault_tool"},
)

SPEC = InputSpec(
    input_id="safe-0",
    mode="single_turn",
    dim_id="topic",
    variation="refund",
    prompt="what is the refund window?",
)


def build(tmp_path, **app_kwargs):
    ls = FakeLangSmithClient([])
    app = FakeTargetApp(ls, **app_kwargs)
    collector = LangSmithCollector(
        CFG.langsmith_project, client=ls, cfg=CFG, poll_interval_s=0.0,
        sleep=lambda _s: None, root_timeout_s=0.0, child_timeout_s=0.0, settle_polls=1,
    )
    store = LocalTraceStore(tmp_path / "traces")
    harness = Harness(
        CFG, store, client=app, collector=collector,
        quarantine=Quarantine(tmp_path / "q"), concurrency=1,
    )
    return harness, app, store


# ------------------------------------------------------------ run_with_faults

def test_run_with_faults_arms_the_declared_key_and_returns_the_regenerated_trace(tmp_path):
    harness, app, store = build(tmp_path)

    trace = harness.run_with_faults(
        SPEC, FaultConfig(shim="retriever", target="corpus_search", behavior="empty")
    )

    assert app.calls[0]["configurable"] == {"fault_retriever": {"behavior": "empty"}}
    assert store.exists(trace.trace_id)
    assert trace.turns[0].spans


def test_the_armed_fault_is_visible_in_the_relevant_span(tmp_path):
    harness, _app, _store = build(tmp_path)
    fault = FaultConfig(shim="retriever", target="corpus_search", behavior="empty")

    trace = harness.run_with_faults(SPEC, fault)

    retrieval = [s for s in trace.turns[0].spans if s.span_type == "retrieval"]
    assert retrieval and retrieval[-1].outputs["output"] == []


def test_an_armed_run_gets_its_own_session_so_it_never_overwrites_the_clean_trace(tmp_path):
    harness, _app, store = build(tmp_path)
    clean = harness.run_single_turn(SPEC, dataset_id="ds-1")
    armed = harness.run_with_faults(
        SPEC,
        FaultConfig(shim="retriever", target="corpus_search", behavior="empty"),
        dataset_id="ds-1",
    )
    assert clean.trace_id != armed.trace_id
    assert store.exists(clean.trace_id) and store.exists(armed.trace_id)


def test_activation_is_checked_against_a_baseline_when_one_is_given(tmp_path):
    harness, _app, _store = build(tmp_path)
    baseline = harness.run_single_turn(SPEC, dataset_id="ds-1")

    trace = harness.run_with_faults(
        SPEC,
        FaultConfig(shim="retriever", target="corpus_search", behavior="stale"),
        dataset_id="ds-1",
        baseline=baseline,
    )
    # Evidence is handed to Phase 5 out-of-band, never written into the Trace:
    # a trace that says where its own fault is would leak the ground truth.
    evidence = harness.activation_evidence[trace.trace_id]
    assert "2019" in evidence  # the stale revision the retrieval span recorded
    assert "activation" not in trace.model_dump_json().lower()


def test_a_fault_that_changes_nothing_is_reported_not_silently_accepted(tmp_path):
    harness, _app, _store = build(tmp_path)
    baseline = harness.run_single_turn(SPEC, dataset_id="ds-1")

    with pytest.raises(FaultNotActivated):
        harness.run_with_faults(
            SPEC,
            # The fake app models no behaviour under this name, so the span is
            # byte-identical to the baseline.
            FaultConfig(shim="retriever", target="corpus_search", behavior="no_op"),
            dataset_id="ds-1",
            baseline=baseline,
        )


def test_an_undeclared_shim_is_refused_before_any_app_call(tmp_path):
    harness, app, _store = build(tmp_path)
    with pytest.raises(UndeclaredFault):
        harness.run_with_faults(
            SPEC, FaultConfig(shim="llm_proxy", target="x", behavior="truncate_output")
        )
    assert app.calls == []


def test_run_with_faults_is_importable_as_a_module_level_function(tmp_path):
    harness, _app, _store = build(tmp_path)
    trace = run_with_faults(
        SPEC,
        FaultConfig(shim="retriever", target="corpus_search", behavior="empty"),
        harness=harness,
    )
    assert trace.trace_id


# --------------------------------------------------------------------- replay

def test_replay_forks_at_the_checkpoint_then_resumes_the_remaining_turns(tmp_path):
    harness, app, store = build(tmp_path)
    corrupted = {"messages": [{"role": "ai", "id": "m1", "content": "refunds take 365 days"}]}

    trace = harness.replay(
        "thread-7", "ckpt-3", corrupted, ["remind me of the refund window"], input_id="safe-0"
    )

    assert app.updates[0]["thread_id"] == "thread-7"
    assert app.updates[0]["checkpoint"] == {"checkpoint_id": "ckpt-3"}
    assert app.updates[0]["values"] == corrupted
    # The resumed run must start from the fork, not from the thread head.
    assert app.calls[0]["checkpoint"] == {"checkpoint_id": "fork-ckpt", "thread_id": "thread-7"}
    assert app.calls[0]["message"] == "remind me of the refund window"
    assert store.exists(trace.trace_id)


def test_the_regenerated_trace_records_the_lineage_phase_5_needs(tmp_path):
    harness, _app, _store = build(tmp_path)
    trace = harness.replay(
        "thread-7", "ckpt-3", {"messages": []}, ["one", "two"], input_id="safe-0"
    )

    assert trace.input_id == "safe-0"
    assert trace.mode == "multi_turn"
    assert len(trace.turns) == 2
    assert trace.metadata["thread_id"] == "thread-7"
    assert trace.metadata["source_checkpoint_id"] == "ckpt-3"
    assert trace.metadata["fork_checkpoint_id"] == "fork-ckpt"
    assert trace.metadata["replayed"] is True


def test_only_the_first_resumed_turn_carries_the_fork_checkpoint(tmp_path):
    harness, app, _store = build(tmp_path)
    harness.replay("thread-7", "ckpt-3", {"messages": []}, ["one", "two"], input_id="i")
    assert app.calls[0]["checkpoint"] is not None
    assert app.calls[1]["checkpoint"] is None


def test_a_checkpoint_ref_may_be_the_full_dict(tmp_path):
    harness, app, _store = build(tmp_path)
    harness.replay(
        "thread-7",
        {"checkpoint_id": "ckpt-3", "thread_id": "thread-7"},
        {"messages": []},
        ["one"],
        input_id="i",
    )
    assert app.updates[0]["checkpoint"]["checkpoint_id"] == "ckpt-3"


def test_replay_with_nothing_left_to_regenerate_is_refused_loudly(tmp_path):
    """M=1 Mode A has no downstream — that is a post-hoc edit, not a replay."""
    harness, _app, _store = build(tmp_path)
    with pytest.raises(ValueError, match="remaining_user_messages"):
        harness.replay("thread-7", "ckpt-3", {"messages": []}, [], input_id="i")


def test_the_same_replay_is_idempotent_in_the_store(tmp_path):
    harness, _app, store = build(tmp_path)
    args = ("thread-7", "ckpt-3", {"messages": [{"content": "x"}]}, ["one"])
    first = harness.replay(*args, input_id="i")
    second = harness.replay(*args, input_id="i")
    assert first.trace_id == second.trace_id
    assert len(store.list_ids()) == 1


def test_replay_is_importable_as_a_module_level_function(tmp_path):
    harness, _app, _store = build(tmp_path)
    trace = replay("thread-7", "ckpt-3", {"messages": []}, ["one"], harness=harness, input_id="i")
    assert trace.trace_id
