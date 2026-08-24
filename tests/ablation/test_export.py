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


# --------------------------------------------------------- time normalization

def _span_times(exported: dict) -> list[tuple[str, str]]:
    return [(s["start_time"], s["end_time"]) for t in exported["turns"] for s in t["spans"]]


def _max_offset_seconds(exported: dict) -> float:
    from datetime import datetime

    times = [datetime.fromisoformat(e) for _s, e in _span_times(exported)]
    starts = [datetime.fromisoformat(s) for s, _e in _span_times(exported)]
    return (max(times) - min(starts)).total_seconds()


def test_wall_clock_times_cannot_separate_ablated_from_control():
    """The measured leak: controls collected at 20:27, Mode-C re-runs at 20:52+.

    A trivial time threshold separated the two sets. Export times are therefore
    re-based to a fixed synthetic origin, so absolute wall clock carries no
    signal at all.
    """
    from datetime import UTC, datetime, timedelta

    control = make_trace("t-control", "safe-00", turns=2)
    ablated = make_trace("t-ablated", "safe-01", turns=2)
    # Push the "ablated" trace 25 minutes into the future, exactly as a
    # re-run collected later in the session does.
    for turn in ablated.turns:
        for span in turn.spans:
            span.start_time += timedelta(minutes=25)
            span.end_time += timedelta(minutes=25)

    exported = build_export([control, ablated])
    origins = {datetime.fromisoformat(_span_times(e)[0][0]) for e in exported}
    assert len(origins) == 1, f"traces start at different wall clocks: {origins}"
    assert all(o.tzinfo is not None for o in origins)
    # And the two are comparable in extent, so no threshold on max-time works.
    spans = [_max_offset_seconds(e) for e in exported]
    assert abs(spans[0] - spans[1]) < 1.0, spans
    _ = datetime.now(UTC)  # the export must not depend on when it ran


def test_a_mid_conversation_gap_is_collapsed_to_a_plausible_one():
    """Mode A regenerates the tail minutes later; inside one conversation that
    gap is a fingerprint no organic trace produces."""
    from datetime import datetime, timedelta

    from benchmark.ablation.export import INTER_TURN_GAP

    trace = make_trace("t-spliced", "safe-00", turns=3)
    # turns 1..2 were regenerated 25 minutes after turn 0 was collected
    for turn in trace.turns[1:]:
        for span in turn.spans:
            span.start_time += timedelta(minutes=25)
            span.end_time += timedelta(minutes=25)

    exported = build_export([trace])[0]
    gaps = []
    for earlier, later in zip(exported["turns"], exported["turns"][1:], strict=False):
        end = max(datetime.fromisoformat(s["end_time"]) for s in earlier["spans"])
        start = min(datetime.fromisoformat(s["start_time"]) for s in later["spans"])
        gaps.append((start - end).total_seconds())
    assert gaps, "the fixture must be multi-turn"
    assert all(abs(g - INTER_TURN_GAP.total_seconds()) < 0.001 for g in gaps), gaps


def test_intra_turn_deltas_and_real_durations_are_preserved():
    """Only the origin moves. Within a turn, every relative time survives."""
    from datetime import datetime

    trace = make_trace("t-1", "safe-00", turns=2)
    before = [
        [(s.start_time, s.end_time) for s in turn.spans] for turn in trace.turns
    ]
    exported = build_export([trace])[0]
    for turn_index, turn in enumerate(exported["turns"]):
        origin_before = min(s for s, _e in before[turn_index])
        origin_after = min(datetime.fromisoformat(s["start_time"]) for s in turn["spans"])
        for span_index, span in enumerate(turn["spans"]):
            start_before, end_before = before[turn_index][span_index]
            start_after = datetime.fromisoformat(span["start_time"])
            end_after = datetime.fromisoformat(span["end_time"])
            assert start_after - origin_after == start_before - origin_before
            assert end_after - start_after == end_before - start_before, "duration is real"


def test_normalization_leaves_the_source_trace_untouched(ablated_trace):
    before = ablated_trace.model_dump_json()
    build_export([ablated_trace])
    assert ablated_trace.model_dump_json() == before


def test_a_naive_timestamp_does_not_blow_up_the_export():
    """The export runs AFTER every paid-for injection, so a TypeError here
    throws away a whole live run. A naive timestamp is read as UTC."""
    from datetime import datetime

    trace = make_trace("t-naive", "safe-00", turns=2)
    for turn in trace.turns:
        for span in turn.spans:
            span.start_time = span.start_time.replace(tzinfo=None)
            span.end_time = span.end_time.replace(tzinfo=None)

    exported = build_export([trace])[0]
    times = [datetime.fromisoformat(v) for pair in _span_times(exported) for v in pair]
    assert times, "the fixture must have spans"
    assert all(t.tzinfo is not None for t in times), "the export is tz-aware"
    assert min(times).isoformat() == "2026-01-01T00:00:00+00:00"


def test_a_turn_with_no_spans_does_not_break_normalization():
    trace = make_trace("t-1", "safe-00", turns=2)
    trace.turns[0].spans = []
    exported = build_export([trace])[0]
    assert exported["turns"][0]["spans"] == []
    assert exported["turns"][1]["spans"], "the remaining turn still exports"


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


def test_the_audit_catches_a_token_inside_a_span_id(ablated_trace, target_cfg):
    """Span ids are exported verbatim, so they are part of the leak surface.

    Caught for real by a fixture that named its regenerated traces
    "replayed-N"; keeping it as an explicit case.
    """
    payload = build_export([ablated_trace])
    payload[0]["turns"][0]["spans"][0]["span_id"] = "replayed-1-t0-agent"
    with pytest.raises(ExportLeak, match="replayed"):
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
