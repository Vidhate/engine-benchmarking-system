"""The Stage III entrypoint, end to end, fully mocked.

This is the surface Phase 7 codes against:

    run_ablation(traces, inputs, categories, cfg, harness, store, export_path)
        -> AblationResult(ablated, ground_truth, records, split, export_path,
                          dropped_errors)
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from benchmark.ablation import AblationEngine, AblationResult, run_ablation
from benchmark.ablation.agent import ScriptedAblationAgent
from benchmark.ablation.export import audit_export
from benchmark.schemas.configs import AblationConfig
from benchmark.schemas.issues import Issueboard
from benchmark.schemas.traces import Trace

from .conftest import FakeHarness, make_proposal


@pytest.fixture
def agent():
    return ScriptedAblationAgent(
        {
            "hallucination": [make_proposal("h0", "hallucination", target_count=3)],
            "retrieval_failure": [
                make_proposal(
                    "r0", "retrieval_failure", mode="dependency_fault", target_count=3
                )
            ],
            "other": [make_proposal("o0", "other", target_count=2)],
        }
    )


@pytest.fixture
def result(traces, inputs, categories, ablation_cfg, harness, store, tmp_path, agent):
    engine = AblationEngine(harness, store, ablation_cfg, agent=agent)
    return engine.run(traces, inputs, categories, tmp_path / "engine_traces.json")


def test_the_public_signature_is_the_one_phase_7_codes_against():
    parameters = list(inspect.signature(run_ablation).parameters)
    assert parameters == [
        "traces", "inputs", "categories", "cfg", "harness", "store", "export_path"
    ]
    fields = set(AblationResult.model_fields)
    assert {
        "ablated", "ground_truth", "records", "split", "export_path", "dropped_errors"
    } <= fields


def test_run_ablation_wires_the_default_agent_and_returns_a_result(
    monkeypatch, traces, inputs, categories, ablation_cfg, harness, store, tmp_path, agent
):
    monkeypatch.setattr("benchmark.ablation.engine.default_agent", lambda: agent)
    result = run_ablation(
        traces, inputs, categories, ablation_cfg, harness, store,
        tmp_path / "engine_traces.json",
    )
    assert isinstance(result, AblationResult)
    assert result.ground_truth.source == "ground_truth"
    assert result.records


def test_both_injection_modes_are_exercised(result):
    modes = {i.error_id: i.injection_mode for i in result.ground_truth.issues}
    assert "replay_edit" in modes.values()
    assert "dependency_fault" in modes.values()


def test_the_ground_truth_board_is_schema_valid_and_lineage_is_kept(result, traces):
    Issueboard.model_validate_json(result.ground_truth.model_dump_json())
    assert result.ablated.parent_dataset_id == traces.dataset_id
    assert result.ablated.dataset_id


def test_every_occurrence_names_a_shipped_trace_and_a_declared_issue(result):
    shipped = {t.trace_id for t in result.ablated.traces}
    declared = {i.error_id for i in result.ground_truth.issues}
    for occurrence in result.ground_truth.occurrences:
        assert occurrence.trace_id in shipped
        assert occurrence.error_id in declared


def test_control_inputs_are_untouched_and_byte_identical(result, traces):
    before = {t.input_id: t.model_dump_json() for t in traces.traces}
    control = set(result.split.control_input_ids)
    assert control, "the fixture config must hold some inputs back"
    for trace in result.ablated.traces:
        if trace.input_id in control:
            assert trace.model_dump_json() == before[trace.input_id]
            assert trace.ablation_ids == []
    injected = {t.input_id for t in result.ablated.traces if t.ablation_ids}
    assert not injected & control


def test_the_export_exists_passes_its_own_audit_and_parses_as_traces(
    result, target_cfg
):
    payload = json.loads(open(result.export_path).read())
    audit_export(payload, target_cfg)
    assert len(payload) == len(result.ablated.traces)
    for item in payload:
        Trace.model_validate(item)  # the Engine's local Trace is a subset of ours


def test_the_export_never_names_the_ground_truth(result):
    blob = open(result.export_path).read().lower()
    for tell in ("ablation", "injection_mode", "replay_edit", "dependency_fault", "thread_id"):
        assert tell not in blob, tell


def test_the_split_is_reported_rather_than_hidden(result, inputs):
    assert result.split.strata
    assert len(result.split.control_input_ids) + len(result.split.ablate_input_ids) == len(
        inputs.inputs
    )
    assert result.injected_counts, "per-error injection counts are part of the report"


def test_an_all_multi_turn_dead_thread_corpus_fails_loudly_before_injecting(
    target_cfg, store, categories, ablation_cfg, tmp_path, agent
):
    """Multi-turn replay genuinely needs live threads, so an ALL-multi-turn
    corpus whose threads are all dead must still abort loudly. (A corpus with
    single-turn traces proceeds instead — those are post-hoc edits; see
    test_a_single_turn_corpus_survives_dead_server_threads.)"""
    from benchmark.ablation.inject import DeadThreadRefs
    from tests.ablation.conftest import make_inputs, make_traces

    multi_inputs = make_inputs(n_safe=0, n_adv=0, n_multi=3)
    multi_traces = make_traces(multi_inputs)
    for t in multi_traces.traces:
        store.put(t)
    harness = FakeHarness(target_cfg, store, live_threads=set())
    engine = AblationEngine(harness, store, ablation_cfg, agent=agent)
    with pytest.raises(DeadThreadRefs, match="one server lifetime"):
        engine.run(multi_traces, multi_inputs, categories, tmp_path / "engine_traces.json")


def test_a_dropped_error_is_reported_with_its_reason(
    traces, inputs, categories, harness, store, tmp_path
):
    from benchmark.schemas.ablation import FilterStep

    impossible = make_proposal(
        "h0",
        "hallucination",
        filter_steps=[FilterStep(field="span_names", op="eq", value="tool_that_never_ran")],
    )
    agent = ScriptedAblationAgent({"hallucination": [impossible]})
    cfg = AblationConfig(seed=7, control_fraction=0.3, min_eligible=99, n_per_category=1)
    engine = AblationEngine(harness, store, cfg, agent=agent)
    result = engine.run(traces, inputs, categories, tmp_path / "engine_traces.json")
    assert result.dropped_errors
    assert "min_eligible=99" in result.dropped_errors[0]
    assert result.ground_truth.issues == []


def test_a_failing_agent_costs_its_category_not_the_paid_for_corpus(
    traces, inputs, categories, ablation_cfg, harness, store, tmp_path
):
    """By step 1 the whole corpus is already collected; one bad draw must not
    discard it."""
    from benchmark.ablation.agent import AgentResponseError

    class _HalfBroken:
        def propose(self, category, n, digest, allowed_modes):
            if category.category_id == "hallucination":
                raise AgentResponseError("the model's reply was not JSON")
            return [make_proposal(f"ok-{category.category_id}", category.category_id)]

        def revise_corruption(self, proposal, digest, reasons):  # pragma: no cover
            raise AssertionError("not reached")

    engine = AblationEngine(harness, store, ablation_cfg, agent=_HalfBroken())
    result = engine.run(traces, inputs, categories, tmp_path / "engine_traces.json")
    assert any("hallucination" in reason for reason in result.dropped_errors)
    assert result.ground_truth.issues, "the other categories still produced errors"
    assert Path(result.export_path).exists()


def test_the_config_knobs_reach_the_stages_that_use_them(
    traces, inputs, categories, harness, store, tmp_path, agent
):
    cfg = AblationConfig(
        seed=7, control_fraction=0.3, min_eligible=2, n_per_category=1,
        target_count=1, max_replans=0,
    )
    result = AblationEngine(harness, store, cfg, agent=agent).run(
        traces, inputs, categories, tmp_path / "engine_traces.json"
    )
    assert all(count <= 1 for count in result.injected_counts.values()), (
        f"target_count=1 was not honoured: {result.injected_counts}"
    )
    assert all(f.attempt == 0 for f in result.validation.failures), (
        "max_replans=0 must leave no second attempt"
    )


def test_the_not_retracted_checks_burn_rate_reaches_the_result(
    traces, inputs, categories, target_cfg, store, tmp_path
):
    """`retraction_in` has a documented residual false-negative surface. What
    bounds it is how many candidates it burned, so that number has to leave the
    engine rather than stay in a log line."""
    from benchmark.schemas.ablation import FilterStep

    # The dry run stays clean; every step-4 candidate after it self-corrects.
    harness = FakeHarness(target_cfg, store, self_corrects=True, self_corrects_after=1)
    proposal = make_proposal(
        "h0",
        "hallucination",
        turn_index=0,  # only a trace with a regenerated tail can retract at all
        filter_steps=[FilterStep(field="mode", op="eq", value="multi_turn")],
    )
    agent = ScriptedAblationAgent({"hallucination": [proposal]})
    cfg = AblationConfig(
        seed=7, control_fraction=0.0, min_eligible=1, n_per_category=1, max_replans=0
    )
    result = AblationEngine(harness, store, cfg, agent=agent).run(
        traces, inputs, categories, tmp_path / "engine_traces.json"
    )
    # step 1 re-stamps error ids per category, so the id here is the stamped one
    assert result.self_corrected_counts == {"E-hallucination-00": 2}
    assert result.injected_counts.get("E-hallucination-00", 0) == 0


def test_a_mode_c_only_run_never_probes_thread_liveness(
    traces, inputs, categories, ablation_cfg, target_cfg, store, tmp_path
):
    """A probe is a real round trip per thread, and `assert_threads_alive` would
    abort a valid dependency-fault run over threads it never intended to fork."""
    harness = FakeHarness(target_cfg, store)
    agent = ScriptedAblationAgent(
        {
            "retrieval_failure": [
                make_proposal("r0", "retrieval_failure", mode="dependency_fault",
                              target_count=2)
            ]
        }
    )
    result = AblationEngine(harness, store, ablation_cfg, agent=agent).run(
        traces, inputs, categories, tmp_path / "engine_traces.json"
    )
    assert harness.boundary_probes == [], harness.boundary_probes
    assert result.injected_counts.get("E-retrieval_failure-00", 0) > 0, (
        "the Mode-C run still did its work"
    )


def test_a_run_with_a_replay_edit_does_probe_up_front(
    traces, inputs, categories, ablation_cfg, target_cfg, store, tmp_path
):
    harness = FakeHarness(target_cfg, store)
    agent = ScriptedAblationAgent(
        {"hallucination": [make_proposal("h0", "hallucination", target_count=2)]}
    )
    AblationEngine(harness, store, ablation_cfg, agent=agent).run(
        traces, inputs, categories, tmp_path / "engine_traces.json"
    )
    assert harness.boundary_probes, "Mode A must fail loudly on a dead-thread corpus"


def test_the_run_is_reproducible_for_a_seed(
    traces, inputs, categories, ablation_cfg, target_cfg, tmp_path
):
    from benchmark.tracing.store import LocalTraceStore

    boards = []
    for run in range(2):
        store = LocalTraceStore(tmp_path / f"store{run}")
        for trace in traces.traces:
            store.put(trace.model_copy(deep=True))
        harness = FakeHarness(target_cfg, store)
        agent = ScriptedAblationAgent(
            {"hallucination": [make_proposal("h0", "hallucination", target_count=3)]}
        )
        engine = AblationEngine(harness, store, ablation_cfg, agent=agent)
        result = engine.run(traces, inputs, categories, tmp_path / f"export{run}.json")
        boards.append(
            sorted((o.error_id, o.trace_id) for o in result.ground_truth.occurrences)
        )
    assert boards[0] == boards[1]



def test_a_single_turn_corpus_survives_dead_server_threads(
    inputs, categories, ablation_cfg, store, tmp_path, agent, target_cfg
):
    """The crash-resume scenario: every thread is dead (the corpus was
    collected under earlier, now-gone server lifetimes), but the corpus is
    single-turn — where replay_edit is a post-hoc edit that never forks a
    thread. Mode A must remain fully eligible, no DeadThreadRefs, and the
    engine must not probe liveness at all (a probe is a real server call)."""
    from tests.ablation.conftest import make_inputs, make_traces

    single_inputs = make_inputs(n_multi=0)
    single_traces = make_traces(single_inputs)
    for t in single_traces.traces:
        store.put(t)
    harness = FakeHarness(target_cfg, store, live_threads=set())  # all dead
    engine = AblationEngine(harness, store, ablation_cfg, agent=agent)

    result = engine.run(
        single_traces, single_inputs, categories, tmp_path / "engine_traces.json"
    )

    modes = {i.injection_mode for i in result.ground_truth.issues}
    assert "replay_edit" in modes, "dead threads must not exclude single-turn Mode A"
    assert harness.boundary_probes == [], (
        "an all-single-turn corpus must not probe thread liveness"
    )
