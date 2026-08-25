"""The stand-in ablation stage and the leak audit on the Engine's trace file.

The fake exists only until Phase 5 merges, but it has to honour the invariants
downstream code actually depends on — otherwise the miniature run proves the
wiring works against a shape the real stage will not produce.
"""

from __future__ import annotations

import json

import pytest

from benchmark.pipeline.export import (
    ExportLeak,
    assert_export_file_clean,
    export_traces,
    write_leak_stripped_export,
)
from benchmark.pipeline.fakes import fake_run_ablation, split_inputs
from benchmark.schemas import AblationConfig, Trace, TraceDataset


@pytest.fixture
def result(tmp_path, tiny_inputs, tiny_traces, taxonomy):
    return fake_run_ablation(
        traces=tiny_traces,
        inputs=tiny_inputs,
        categories=taxonomy,
        cfg=AblationConfig(seed=11, control_fraction=0.34),
        harness=None,
        store=None,
        export_path=tmp_path / "traces.json",
    )


# ------------------------------------------------------------------- the split

def test_the_split_partitions_every_input_exactly_once(tiny_inputs):
    split = split_inputs(tiny_inputs, AblationConfig(seed=3, control_fraction=0.5))
    all_ids = {s.input_id for s in tiny_inputs.inputs}
    assert set(split.control_input_ids) | set(split.ablate_input_ids) == all_ids
    assert not set(split.control_input_ids) & set(split.ablate_input_ids)


def test_the_split_is_seeded_and_therefore_reproducible(tiny_inputs):
    cfg = AblationConfig(seed=3, control_fraction=0.5)
    assert split_inputs(tiny_inputs, cfg) == split_inputs(tiny_inputs, cfg)


def test_a_different_seed_can_move_inputs(tiny_inputs):
    a = split_inputs(tiny_inputs, AblationConfig(seed=1, control_fraction=0.5))
    b = split_inputs(tiny_inputs, AblationConfig(seed=999, control_fraction=0.5))
    assert a.control_input_ids != b.control_input_ids or a.seed != b.seed


# ------------------------------------------------------------- ground truth

def test_control_inputs_carry_no_ground_truth(result, tiny_traces):
    control = set(result.split.control_input_ids)
    by_trace = {t.trace_id: t.input_id for t in tiny_traces.traces}
    flagged = {by_trace[o.trace_id] for o in result.ground_truth.occurrences}
    assert not flagged & control, "the control set was labelled"


def test_no_trace_carries_two_errors_of_one_category(result):
    category = {i.error_id: i.category_id for i in result.ground_truth.issues}
    seen: set[tuple[str, str]] = set()
    for occ in result.ground_truth.occurrences:
        key = (occ.trace_id, category[occ.error_id])
        assert key not in seen, f"same-category disjointness broken on {key}"
        seen.add(key)


def test_every_occurrence_names_a_known_issue(result):
    known = {i.error_id for i in result.ground_truth.issues}
    assert {o.error_id for o in result.ground_truth.occurrences} <= known


def test_one_record_per_occurrence(result):
    assert len(result.records) == len(result.ground_truth.occurrences)


def test_planted_errors_span_both_injection_modes(result):
    modes = {i.injection_mode for i in result.ground_truth.issues}
    assert modes == {"replay_edit", "dependency_fault"}


def test_the_ablated_set_is_the_full_trace_universe(result, tiny_traces):
    """Scoring's kappa needs every trace, control ones included."""
    assert [t.trace_id for t in result.ablated.traces] == [
        t.trace_id for t in tiny_traces.traces
    ]


def test_the_ablated_set_points_back_at_its_source(result, tiny_traces):
    assert result.ablated.parent_dataset_id == tiny_traces.dataset_id
    assert result.ablated.dataset_id != tiny_traces.dataset_id


def test_the_ground_truth_board_is_stamped(result):
    assert result.ground_truth.board_id
    assert result.ground_truth.source == "ground_truth"


# ------------------------------------------------------------------- export

def test_the_export_is_written_and_loads_as_traces(result):
    payload = assert_export_file_clean(result.export_path)
    assert len(payload["traces"]) == len(result.ablated.traces)
    assert all(Trace.model_validate(raw) for raw in payload["traces"])


def test_the_export_carries_no_ablation_ids(result):
    payload = json.loads(open(result.export_path).read())
    assert all("ablation_ids" not in raw for raw in payload["traces"])


def test_the_export_never_names_the_ground_truth(tmp_path):
    poisoned = TraceDataset(
        traces=[
            Trace(
                trace_id="t1",
                input_id="i1",
                mode="single_turn",
                ablation_ids=["abl-1"],
                metadata={"injection_mode": "replay_edit"},
            )
        ]
    )
    with pytest.raises(ExportLeak, match="injection_mode"):
        write_leak_stripped_export(poisoned, tmp_path / "traces.json")


def test_a_stripped_ablation_id_alone_is_not_enough(tmp_path):
    """ablation_ids is dropped by the writer; a leak elsewhere still fails."""
    ok = TraceDataset(traces=[Trace(trace_id="t1", input_id="i1", mode="single_turn",
                                    ablation_ids=["abl-1"])])
    path = write_leak_stripped_export(ok, tmp_path / "traces.json")
    assert "abl-1" not in path.read_text()


def test_auditing_a_missing_export_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="engine export not found"):
        assert_export_file_clean(tmp_path / "nope.json")


def test_a_fault_key_echo_in_the_export_is_caught(tmp_path):
    path = tmp_path / "traces.json"
    path.write_text(
        json.dumps(
            {
                "dataset_id": "d",
                "parent_dataset_id": None,
                "traces": [
                    {
                        "trace_id": "t1",
                        "input_id": "i1",
                        "mode": "single_turn",
                        "turns": [],
                        "status": "ok",
                        "metadata": {"fault_retriever": {"behavior": "stale"}},
                    }
                ],
            }
        )
    )
    with pytest.raises(ExportLeak, match="fault_retriever"):
        assert_export_file_clean(path)


# ------------------------------- the shape the REAL Phase-5 export arrives in

def test_the_pipeline_reads_the_export_phase_5_actually_writes(tmp_path, tiny_traces):
    """`benchmark.ablation.write_engine_export` writes a BARE LIST of traces.

    The pipeline's own stand-in writes `{dataset_id, parent_dataset_id,
    traces: [...]}`, so every reader on this side was built around a dict and
    a real export would have died on `payload.get`. The Engine app has always
    accepted both shapes (`apps/engine/engine/traces.py::load_traces`); the
    audit, the deliverables check and the invoker now do too, because the
    format the ground-truth side writes is the one that has to be read.
    """
    from benchmark.ablation.export import write_engine_export  # noqa: PLC0415

    path = write_engine_export(tiny_traces, tmp_path / "traces.json")
    payload = json.loads(path.read_text())
    assert isinstance(payload, list), "this test is worthless if Phase 5 writes a dict"

    audited = assert_export_file_clean(path)
    assert [t["trace_id"] for t in export_traces(audited)] == [
        t.trace_id for t in tiny_traces.traces
    ]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ([{"trace_id": "a"}], ["a"]),
        ({"traces": [{"trace_id": "a"}]}, ["a"]),
        ({"dataset_id": "d", "traces": []}, []),
        ([], []),
    ],
)
def test_both_export_shapes_yield_the_same_trace_list(payload, expected):
    assert [t["trace_id"] for t in export_traces(payload)] == expected


def test_an_export_that_is_neither_shape_says_so():
    with pytest.raises(ExportLeak, match="unsupported"):
        export_traces("not a trace corpus")
