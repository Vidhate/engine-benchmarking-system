"""The seams the pipeline is wired against.

Phase 5 (`benchmark/ablation/`) is being built in parallel and is not on main
yet, so the pipeline codes against its PINNED contract:

    run_ablation(traces, inputs, categories, cfg, harness, store, export_path)
        -> AblationResult(ablated, ground_truth, records, split,
                          export_path, dropped_errors)

That contract lives here as a Protocol, the import of the real implementation
is lazy (so CI is green before Phase 5 merges), and a structural check runs at
the seam so a drifted implementation fails at the hand-off rather than three
stages downstream.
"""

from __future__ import annotations

import sys
import types

import pytest

from benchmark.pipeline.contracts import (
    AblationResult,
    AblationStageUnavailable,
    assert_ablation_result,
    load_ablation_stage,
)
from benchmark.pipeline.fakes import fake_run_ablation
from benchmark.schemas import AblationConfig, AblationSplit, Issueboard, TraceDataset

# --------------------------------------------------------------- lazy import

def test_the_real_stage_is_only_imported_on_demand():
    """Importing benchmark.pipeline must not require benchmark.ablation."""
    assert "benchmark.ablation" not in sys.modules


def test_a_missing_phase_5_names_itself(monkeypatch):
    monkeypatch.setitem(sys.modules, "benchmark.ablation", None)
    with pytest.raises(AblationStageUnavailable, match="benchmark.ablation"):
        load_ablation_stage()


def test_the_real_stage_is_returned_once_phase_5_lands(monkeypatch):
    module = types.ModuleType("benchmark.ablation")
    module.run_ablation = lambda **kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "benchmark.ablation", module)
    assert load_ablation_stage() is module.run_ablation


def test_a_phase_5_without_run_ablation_says_so(monkeypatch):
    monkeypatch.setitem(sys.modules, "benchmark.ablation", types.ModuleType("benchmark.ablation"))
    with pytest.raises(AblationStageUnavailable, match="run_ablation"):
        load_ablation_stage()


# ---------------------------------------------------------- structural check

class _Bare:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _complete_result(**overrides):
    fields = {
        "ablated": TraceDataset(),
        "ground_truth": Issueboard(source="ground_truth"),
        "records": [],
        "split": AblationSplit(seed=0, control_fraction=0.3),
        "export_path": "/tmp/traces.json",
        "dropped_errors": [],
    }
    fields.update(overrides)
    return _Bare(**fields)


def test_a_complete_result_passes_the_seam_check():
    assert_ablation_result(_complete_result())


@pytest.mark.parametrize(
    "missing",
    ["ablated", "ground_truth", "records", "split", "export_path", "dropped_errors"],
)
def test_every_pinned_field_is_required(missing):
    fields = _complete_result().__dict__
    del fields[missing]
    with pytest.raises(TypeError, match=missing):
        assert_ablation_result(_Bare(**fields))


def test_a_wrongly_typed_field_is_caught_at_the_seam():
    with pytest.raises(TypeError, match="ablated"):
        assert_ablation_result(_complete_result(ablated={"traces": []}))


def test_the_result_protocol_is_runtime_checkable():
    assert isinstance(_complete_result(), AblationResult)


# ------------------------------------------------------------- the fake stage

def test_the_shipped_fake_satisfies_the_pinned_call_shape(
    tmp_path, tiny_inputs, tiny_traces, taxonomy
):
    """The fake is called with EXACTLY the keywords the real stage declares."""
    result = fake_run_ablation(
        traces=tiny_traces,
        inputs=tiny_inputs,
        categories=taxonomy,
        cfg=AblationConfig(seed=1, control_fraction=0.5),
        harness=None,
        store=None,
        export_path=tmp_path / "traces.json",
    )
    assert_ablation_result(result)
