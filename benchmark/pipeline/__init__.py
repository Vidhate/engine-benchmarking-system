"""Stage VI — end-to-end benchmark assembly (docs/execution-plan.md, Phase 7).

One command turns configs into a `BenchmarkReport`:

    generate_inputs -> run_harness -> run_ablation -> Engine -> score()

    uv run python -m benchmark.pipeline run --config configs/pipeline/mini.yaml

Every stage writes its artifact JSON under `<artifacts_root>/<run_id>/`, and
`manifest.json` ties the run together: config hashes, dataset ids and their
lineage, model ids, counts, per-stage timings, and warnings.

**Server-lifetime choreography** — the pipeline manages both LangGraph servers
itself, through each app's own `scripts/serve.sh` (a subprocess, never an
import):

1. target app up  ->  harness batch  ->  ablation (replay + fault re-runs)  ->  down
2. engine up      ->  one Engine run over the leak-stripped export          ->  down

The harness batch and the ablation stage MUST share one target-app server
lifetime: Mode-A replay forks a LangGraph thread created during the batch, and
`langgraph dev`'s thread state does not survive a restart. Scoring and
rendering need no server at all.

**Phase 5 is pinned, not imported.** `benchmark.ablation.run_ablation` is
loaded lazily via `load_ablation_stage()`, and `benchmark.pipeline.fakes` ships
a stand-in with the same call shape so the miniature run and the CI test work
before Phase 5 merges. See `benchmark.pipeline.contracts` for the pinned
contract and the seam check that enforces it.
"""

from benchmark.pipeline.config import (
    EngineStageConfig,
    PipelineConfig,
    ServerSpec,
    load_engine_app_config,
    load_pipeline_config,
    load_seed_board,
    load_taxonomy,
)
from benchmark.pipeline.contracts import (
    AblationResult,
    AblationStage,
    AblationStageUnavailable,
    EngineInvocation,
    EngineInvoker,
    HarnessFactory,
    HarnessLike,
    assert_ablation_result,
    load_ablation_stage,
)
from benchmark.pipeline.deliverables import DeliverableCheck, check_deliverables, rescore_from_disk
from benchmark.pipeline.engine import EngineRunFailed, LangGraphEngineInvoker
from benchmark.pipeline.export import ExportLeak, assert_export_file_clean
from benchmark.pipeline.manifest import ArtifactPaths, RunArtifacts, RunManifest, StageTiming
from benchmark.pipeline.render import render_markdown, severity_confusion
from benchmark.pipeline.runner import PipelineRun, build_base_rates, run_pipeline, slice_inputs
from benchmark.pipeline.servers import ServerLifetime, ServerStartFailed

__all__ = [
    "AblationResult",
    "AblationStage",
    "AblationStageUnavailable",
    "ArtifactPaths",
    "DeliverableCheck",
    "EngineInvocation",
    "EngineInvoker",
    "EngineRunFailed",
    "EngineStageConfig",
    "ExportLeak",
    "HarnessFactory",
    "HarnessLike",
    "LangGraphEngineInvoker",
    "PipelineConfig",
    "PipelineRun",
    "RunArtifacts",
    "RunManifest",
    "ServerLifetime",
    "ServerSpec",
    "ServerStartFailed",
    "StageTiming",
    "assert_ablation_result",
    "assert_export_file_clean",
    "build_base_rates",
    "check_deliverables",
    "load_ablation_stage",
    "load_engine_app_config",
    "load_pipeline_config",
    "load_seed_board",
    "load_taxonomy",
    "render_markdown",
    "rescore_from_disk",
    "run_pipeline",
    "severity_confusion",
    "slice_inputs",
]
