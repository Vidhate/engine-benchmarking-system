"""The Stage III entrypoint, end to end, fully mocked.

This is the surface Phase 7 codes against:

    run_ablation(traces, inputs, categories, cfg, harness, store, export_path)
        -> AblationResult(ablated, ground_truth, records, split, export_path,
                          dropped_errors)
"""

from __future__ import annotations

import inspect
import json

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


def test_a_dead_thread_corpus_fails_loudly_before_anything_is_injected(
    target_cfg, store, traces, inputs, categories, ablation_cfg, tmp_path, agent
):
    from benchmark.ablation.inject import DeadThreadRefs

    harness = FakeHarness(target_cfg, store, live_threads=set())
    engine = AblationEngine(harness, store, ablation_cfg, agent=agent)
    with pytest.raises(DeadThreadRefs, match="one server lifetime"):
        engine.run(traces, inputs, categories, tmp_path / "engine_traces.json")


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
