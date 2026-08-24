"""The seams the pipeline is wired against.

Three stages of the pipeline are things the pipeline does not own: the Phase-4
harness (a live LangGraph server), the Phase-5 ablation engine, and the Engine
under test (another live server). Each is reached through a Protocol here, so
the whole assembly can be exercised against fakes with no network — and so
Phase 5, which is being built in parallel, can be coded against before it
exists.

**The Phase-5 contract is PINNED**:

    run_ablation(traces: TraceDataset, inputs: InputDataset,
                 categories: list[ErrorCategory], cfg: AblationConfig,
                 harness: Harness, store: TraceStore,
                 export_path: Path) -> AblationResult

    AblationResult: ablated (TraceDataset), ground_truth (Issueboard),
                    records (list[AblationRecord]), split (AblationSplit),
                    export_path (str), dropped_errors (list[str])

`benchmark.ablation` is imported lazily (`load_ablation_stage`) so this package
imports — and CI passes — before Phase 5 merges. `assert_ablation_result`
re-checks the shape at the hand-off: a drifted implementation should fail at
the seam that owns the contract, not three stages later inside `score()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from benchmark.pipeline.config import EngineStageConfig
from benchmark.schemas import (
    AblationConfig,
    AblationRecord,
    AblationSplit,
    ErrorCategory,
    Issueboard,
    OutputDataset,
    TraceDataset,
)
from benchmark.schemas.configs import TargetAppConfig
from benchmark.schemas.inputs import InputDataset
from benchmark.tracing.store import TraceStore


class AblationStageUnavailable(RuntimeError):
    """Phase 5 has not merged yet (or does not export `run_ablation`)."""


@runtime_checkable
class AblationResult(Protocol):
    """What Phase 5 hands back. Structural, not nominal — see module docstring."""

    ablated: TraceDataset
    ground_truth: Issueboard
    records: list[AblationRecord]
    split: AblationSplit
    export_path: str
    dropped_errors: list[str]


class AblationStage(Protocol):
    """The pinned `run_ablation` call shape.

    Called with keywords by the runner: the pinned contract fixes the parameter
    NAMES as well as their order, and keywords survive a signature that later
    grows an optional parameter in the middle.
    """

    def __call__(
        self,
        *,
        traces: TraceDataset,
        inputs: InputDataset,
        categories: list[ErrorCategory],
        cfg: AblationConfig,
        harness: Any,
        store: TraceStore,
        export_path: Path,
    ) -> AblationResult: ...


#: (attribute, expected type) — the pinned AblationResult shape.
_ABLATION_RESULT_FIELDS: tuple[tuple[str, type | tuple[type, ...]], ...] = (
    ("ablated", TraceDataset),
    ("ground_truth", Issueboard),
    ("records", list),
    ("split", AblationSplit),
    ("export_path", (str, Path)),
    ("dropped_errors", list),
)


def assert_ablation_result(result: Any) -> Any:
    """Check a Phase-5 result against the pinned contract, or raise TypeError.

    Cheap, and it buys the one thing a parallel-phase integration needs most: a
    failure that names the field that drifted, at the moment the object crosses
    into pipeline code.
    """
    for name, expected in _ABLATION_RESULT_FIELDS:
        if not hasattr(result, name):
            raise TypeError(
                f"ablation result {type(result).__name__} has no {name!r} — the pinned "
                f"Phase-5 contract is AblationResult(ablated, ground_truth, records, "
                f"split, export_path, dropped_errors)"
            )
        value = getattr(result, name)
        if not isinstance(value, expected):
            wanted = getattr(expected, "__name__", str(expected))
            raise TypeError(
                f"ablation result field {name!r} is {type(value).__name__}, expected {wanted}"
            )
    return result


def load_ablation_stage() -> AblationStage:
    """Import Phase 5's `run_ablation`, lazily and with a legible failure."""
    try:
        import benchmark.ablation as ablation  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules patching
        raise AblationStageUnavailable(
            "benchmark.ablation is not available — Phase 5 has not merged yet. Run the "
            "pipeline with an explicit ablation_stage (see benchmark.pipeline.fakes)."
        ) from exc
    if ablation is None:
        raise AblationStageUnavailable(
            "benchmark.ablation is not available — Phase 5 has not merged yet. Run the "
            "pipeline with an explicit ablation_stage (see benchmark.pipeline.fakes)."
        )
    stage = getattr(ablation, "run_ablation", None)
    if stage is None:
        raise AblationStageUnavailable(
            "benchmark.ablation exists but exports no `run_ablation` — the pinned Phase-5 "
            "entrypoint. Nothing else can stand in for it."
        )
    return stage


# --------------------------------------------------------------- the harness

class HarnessLike(Protocol):
    """The slice of the Phase-4 `Harness` the pipeline itself uses.

    The ablation stage gets the *whole* harness object (it needs `replay` and
    `run_with_faults`); the pipeline only ever calls `run_batch` and reads
    `stats`. Typing it narrowly is what lets the CI test hand over a fake.
    """

    stats: dict[str, int]

    def run_batch(self, inputs: InputDataset) -> tuple[OutputDataset, TraceDataset]: ...


class HarnessFactory(Protocol):
    """Builds the one harness used by BOTH the batch and the ablation stage.

    One object, one target-app server lifetime: Mode-A replay forks a LangGraph
    thread created during the batch, and that thread does not survive a server
    restart.
    """

    def __call__(self, cfg: TargetAppConfig, store: TraceStore) -> HarnessLike: ...


# ---------------------------------------------------------------- the Engine

@dataclass
class EngineInvocation:
    """One Engine run's result plus the provenance the manifest records."""

    board: Issueboard
    raw_output: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    thread_id: str = ""
    recorded_models: list[str] = field(default_factory=list)
    trace_count: int = 0


class EngineInvoker(Protocol):
    """Drives the Engine app. The real one speaks `langgraph_sdk` against
    `configs/engine.yaml`; the CI one returns a canned board."""

    def __call__(
        self,
        *,
        trace_file: Path,
        seed_board: Issueboard,
        categories: list[ErrorCategory],
        engine: EngineStageConfig,
    ) -> EngineInvocation: ...
