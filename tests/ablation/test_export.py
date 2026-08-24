"""The leak-stripped Engine export and its no-leak audit."""

from __future__ import annotations

import json

import pytest

from benchmark.ablation.export import (
    METADATA_FIELDS,
    ExportLeak,
    audit_export,
    build_export,
    strip_trace,
    write_engine_export,
)
from benchmark.schemas.traces import TraceDataset

from .conftest import make_trace


@pytest.fixture
def ablated_trace():
    trace = make_trace("t-abl", "safe-00", turns=2)
    trace.ablation_ids = ["abl-E-hallucination-00-000"]
    trace.metadata.update(
        {
            "replayed": True,
            "source_checkpoint_id": "ckpt-7",
            "fork_checkpoint_id": "fork-7",
            "ablation_parent_trace_id": "trace-safe-00",
            "session_id": "s-abcdef",
            "langsmith_trace_ids": ["tr-1"],
        }
    )
    return trace


def test_stripping_keeps_only_the_allowlisted_fields(ablated_trace):
    payload = strip_trace(ablated_trace)
    assert set(payload) == {"trace_id", "input_id", "mode", "turns", "status", "metadata"}
    assert set(payload["metadata"]) <= set(METADATA_FIELDS)
    assert "ablation_ids" not in payload


def test_the_replay_surface_never_survives(ablated_trace):
    blob = json.dumps(build_export([ablated_trace]))
    for tell in ("thread_id", "checkpoint", "replayed", "ablation", "session_id", "langsmith"):
        assert tell not in blob, tell


def test_the_spans_themselves_are_exported_in_full(ablated_trace):
    payload = strip_trace(ablated_trace)
    span = payload["turns"][0]["spans"][0]
    assert set(span) >= {"span_id", "name", "span_type", "inputs", "outputs", "attributes"}
    assert payload["turns"][0]["final_response"]


def test_the_audit_passes_on_a_clean_export(ablated_trace, target_cfg):
    audit_export(build_export([ablated_trace]), target_cfg)


def test_the_audit_catches_a_field_that_rides_along(ablated_trace, target_cfg):
    payload = build_export([ablated_trace])
    payload[0]["ablation_ids"] = ["abl-1"]
    with pytest.raises(ExportLeak, match="non-allowlisted"):
        audit_export(payload, target_cfg)


def test_the_audit_catches_a_metadata_key_that_rides_along(ablated_trace, target_cfg):
    payload = build_export([ablated_trace])
    payload[0]["metadata"]["thread_id"] = "thread-1"
    with pytest.raises(ExportLeak, match="metadata keys"):
        audit_export(payload, target_cfg)


def test_the_audit_catches_a_fault_name_inside_an_exported_span(ablated_trace, target_cfg):
    payload = build_export([ablated_trace])
    payload[0]["turns"][0]["spans"][0]["outputs"] = {"output": "armed via fault_retriever"}
    with pytest.raises(ExportLeak, match="fingerprints"):
        audit_export(payload, target_cfg)


def test_the_audit_catches_a_declared_fault_key_used_as_a_dict_key(ablated_trace, target_cfg):
    payload = build_export([ablated_trace])
    payload[0]["turns"][0]["spans"][0]["attributes"] = {"fault_retriever": {"behavior": "stale"}}
    with pytest.raises(ExportLeak, match="fault_retriever"):
        audit_export(payload, target_cfg)


def test_an_armed_behaviour_name_can_be_added_to_the_scan(ablated_trace, target_cfg):
    payload = build_export([ablated_trace])
    payload[0]["turns"][0]["spans"][0]["outputs"] = {"output": "irrelevant_docs"}
    with pytest.raises(ExportLeak):
        audit_export(payload, target_cfg, extra_tokens=("irrelevant_docs",))


def test_writing_produces_the_engine_input_contract(tmp_path, ablated_trace, target_cfg):
    dataset = TraceDataset(dataset_id="ds-1", parent_dataset_id="ds-0", traces=[ablated_trace])
    path = write_engine_export(dataset, tmp_path / "nested" / "traces.json", target_cfg)
    payload = json.loads(path.read_text())
    assert isinstance(payload, list), "apps/engine reads a bare list or a {traces: [...]} dataset"
    assert payload[0]["trace_id"] == "t-abl"
    assert "ds-0" not in path.read_text(), "lineage says the set was derived"


def test_a_failed_audit_writes_nothing(tmp_path, ablated_trace, target_cfg):
    ablated_trace.turns[0].final_response = "armed with fault_llm truncate_output"
    target = tmp_path / "traces.json"
    with pytest.raises(ExportLeak):
        write_engine_export([ablated_trace], target, target_cfg)
    assert not target.exists(), "a poisoned export must never reach disk"


def test_the_export_of_a_control_trace_is_indistinguishable_in_shape(target_cfg):
    control = make_trace("t-clean", "safe-01", turns=2)
    ablated = make_trace("t-dirty", "safe-02", turns=2)
    ablated.ablation_ids = ["abl-1"]
    ablated.metadata["replayed"] = True
    exported = build_export([control, ablated])
    assert set(exported[0]) == set(exported[1])
    assert set(exported[0]["metadata"]) == set(exported[1]["metadata"])
