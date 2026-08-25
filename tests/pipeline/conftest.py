"""Fixtures for the Phase 7 pipeline tests.

Everything here is offline by construction, and none of it is defined here:
the prompt expander, the harness and the Engine invoker all come from
`benchmark.pipeline.fakes`, which is the same module the `--fake-*` CLI flags
and `scripts/pipeline_smoke.py` reach for. One definition per double — a test
double written twice is a test double that drifts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.pipeline.config import load_taxonomy
from benchmark.pipeline.fakes import (
    FakeEngineInvoker,
    FakeExpander,
    FakeHarness,
    FakeHarnessFactory,
    make_trace,
)
from benchmark.schemas import (
    Dimension,
    GenerationConfig,
    InputDataset,
    InputSpec,
    TraceDataset,
)
from benchmark.schemas.io import derive, stamp_dataset_id

__all__ = [
    "FakeEngineInvoker",
    "FakeExpander",
    "FakeHarness",
    "FakeHarnessFactory",
    "make_trace",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
MINI_PIPELINE_CONFIG = REPO_ROOT / "configs" / "pipeline" / "mini.yaml"


@pytest.fixture
def taxonomy():
    return load_taxonomy(REPO_ROOT / "configs" / "taxonomy.yaml")


@pytest.fixture
def tiny_inputs() -> InputDataset:
    cfg = GenerationConfig(
        safe_dims=[Dimension(dim_id="topic", name="topic", kind="safe", variations=["a", "b"])],
        mode="single_turn",
        seed=7,
    )
    dataset = InputDataset(
        generation_config=cfg,
        inputs=[
            InputSpec(
                input_id=f"in-{i}",
                mode="single_turn",
                dim_id="topic" if i % 2 else "adv",
                variation=f"v{i}",
                prompt=f"prompt {i}",
            )
            for i in range(6)
        ],
    )
    return stamp_dataset_id(dataset)


@pytest.fixture
def tiny_traces(tiny_inputs) -> TraceDataset:
    return derive(
        TraceDataset(
            traces=[make_trace(f"tr-{s.input_id}", s.input_id) for s in tiny_inputs.inputs]
        ),
        tiny_inputs,
    )


@pytest.fixture
def fake_harness_factory():
    return FakeHarnessFactory()


# ------------------------------------------------- a completed miniature run

@pytest.fixture
def mini_cfg(tmp_path):
    """configs/pipeline/mini.yaml, with its artifacts redirected into tmp."""
    from benchmark.pipeline.config import load_pipeline_config

    loaded = load_pipeline_config(MINI_PIPELINE_CONFIG)
    return loaded.model_copy(
        update={
            "artifacts_root": str(tmp_path / "artifacts"),
            "expansion_cache": str(tmp_path / "cache"),
        }
    ).with_root(loaded.root)


@pytest.fixture
def mini_engine_invoker():
    return FakeEngineInvoker()


@pytest.fixture
def mini_run(mini_cfg, fake_harness_factory, mini_engine_invoker):
    """The whole pipeline, end to end, with every external seam faked."""
    from benchmark.pipeline.fakes import fake_run_ablation  # noqa: PLC0415
    from benchmark.pipeline.runner import run_pipeline  # noqa: PLC0415

    return run_pipeline(
        mini_cfg,
        ablation_stage=fake_run_ablation,
        engine_invoker=mini_engine_invoker,
        harness_factory=fake_harness_factory,
        expander=FakeExpander(),
    )
