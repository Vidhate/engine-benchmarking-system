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

import json
import subprocess
import sys
import types

import pytest

import benchmark
from benchmark.pipeline.contracts import (
    ABLATION_RESULT_FIELDS,
    AblationResult,
    AblationStage,
    AblationStageUnavailable,
    assert_ablation_result,
    load_ablation_stage,
)
from benchmark.pipeline.fakes import fake_run_ablation
from benchmark.schemas import AblationConfig, AblationSplit, Issueboard, TraceDataset
from tests.pipeline.conftest import REPO_ROOT

# --------------------------------------------------------------- lazy import

#: Imports EVERY `benchmark.pipeline.*` submodule — the package `__init__` is
#: not the whole surface, and a top-level `import benchmark.ablation` in any
#: one of them breaks the property just as thoroughly.
_IMPORT_GRAPH_PROBE = """
import json, pkgutil, sys
import benchmark.pipeline
for info in pkgutil.iter_modules(benchmark.pipeline.__path__, "benchmark.pipeline."):
    __import__(info.name)
json.dump(sorted(m for m in sys.modules if m.startswith("benchmark.")), sys.stdout)
"""


@pytest.fixture(scope="session")
def pipeline_import_graph() -> frozenset[str]:
    """Every `benchmark.*` module that importing `benchmark.pipeline` reaches.

    Measured in a FRESH interpreter on purpose. The property under test is
    about `benchmark.pipeline`'s own module graph; `sys.modules` in the pytest
    process is a different question entirely, because it also holds whatever
    the rest of the session imported — `tests/ablation/*` collects first and
    imports the real Phase-5 package, which used to make this look violated
    in a full run and satisfied in isolation.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_GRAPH_PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"could not probe the pipeline import graph:\n{proc.stderr}"
    return frozenset(json.loads(proc.stdout))


@pytest.fixture
def ablation_slot(pipeline_import_graph):
    """Hands a test the `benchmark.ablation` module slot, empty, and restores it.

    Two things have to hold for the stand-in tests below to mean anything, and
    both live here so a regression in either reddens all of them:

    1. `benchmark.pipeline` must not import `benchmark.ablation` at module
       import time — checked against the fresh-interpreter graph above.
    2. Whatever the session already imported into that slot must be invisible
       while a test installs its own stand-in. Clearing `sys.modules` is not
       enough: `import a.b as c` binds through `getattr(a, "b")` and only
       falls back to `sys.modules`, so the parent package's attribute has to
       go too — and come back afterwards.
    """
    assert "benchmark.ablation" not in pipeline_import_graph, (
        "benchmark.pipeline reaches benchmark.ablation at module import time — Phase 5 "
        "must stay behind load_ablation_stage()"
    )
    missing = object()
    saved_module = sys.modules.pop("benchmark.ablation", missing)
    saved_attr = getattr(benchmark, "ablation", missing)
    if saved_attr is not missing:
        del benchmark.ablation

    def install(module) -> None:
        sys.modules["benchmark.ablation"] = module

    try:
        yield install
    finally:
        sys.modules.pop("benchmark.ablation", None)
        if saved_module is not missing:
            sys.modules["benchmark.ablation"] = saved_module
        if saved_attr is not missing:
            benchmark.ablation = saved_attr


def test_the_real_stage_is_only_imported_on_demand(pipeline_import_graph):
    """Importing benchmark.pipeline must not require benchmark.ablation."""
    assert "benchmark.ablation" not in pipeline_import_graph


def test_a_missing_phase_5_names_itself(ablation_slot):
    ablation_slot(None)
    with pytest.raises(AblationStageUnavailable, match="benchmark.ablation"):
        load_ablation_stage()


def test_the_real_stage_is_returned_once_phase_5_lands(ablation_slot):
    module = types.ModuleType("benchmark.ablation")
    module.run_ablation = lambda **kwargs: None  # type: ignore[attr-defined]
    ablation_slot(module)
    assert load_ablation_stage() is module.run_ablation


def test_a_phase_5_without_run_ablation_says_so(ablation_slot):
    ablation_slot(types.ModuleType("benchmark.ablation"))
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


def test_the_protocol_and_the_runtime_check_describe_the_same_contract():
    """One canonical field list; the Protocol's annotations must not drift."""
    assert set(AblationResult.__annotations__) == set(ABLATION_RESULT_FIELDS)


@pytest.mark.parametrize("missing", list(ABLATION_RESULT_FIELDS))
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


# ------------------------------------------------------- the REAL Phase-5 stage
#
# Phase 5 has merged, so the pinned contract stops being a promise and becomes
# a checkable fact. These import `benchmark.ablation` on purpose — inside the
# test bodies, so the module-graph property above is not quietly broken by an
# import at the top of this file.

def test_the_seam_loads_the_real_run_ablation():
    import benchmark.ablation  # noqa: PLC0415

    assert load_ablation_stage() is benchmark.ablation.run_ablation


def test_the_real_signature_is_the_pinned_one():
    """Same parameter NAMES in the same ORDER as `AblationStage.__call__`.

    The runner calls the stage with keywords, so the names are load-bearing on
    their own; the order matters because the pinned contract is also what the
    ablation package's own callers and docs quote positionally.
    """
    import inspect  # noqa: PLC0415

    pinned = [p for p in inspect.signature(AblationStage.__call__).parameters if p != "self"]
    actual = list(inspect.signature(load_ablation_stage()).parameters)
    assert actual == pinned


@pytest.mark.parametrize("field", list(ABLATION_RESULT_FIELDS))
def test_the_real_result_type_declares_every_pinned_field(field):
    from benchmark.ablation import AblationResult as RealAblationResult  # noqa: PLC0415

    assert field in RealAblationResult.model_fields


def test_the_real_result_passes_the_seam_check():
    """`assert_ablation_result` over an actual Phase-5 `AblationResult`.

    The fake satisfying the contract only ever proved the fake was written to
    match it. This is the check that matters at integration time, and it runs
    in CI without a server or a model because the shape is the whole claim.
    """
    from benchmark.ablation import AblationResult as RealAblationResult  # noqa: PLC0415

    result = RealAblationResult(
        ablated=TraceDataset(),
        ground_truth=Issueboard(source="ground_truth"),
        split=AblationSplit(seed=0, control_fraction=0.3),
        export_path="/tmp/traces.json",
    )
    assert assert_ablation_result(result) is result
    assert isinstance(result, AblationResult), "the runtime-checkable Protocol disagrees"


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
