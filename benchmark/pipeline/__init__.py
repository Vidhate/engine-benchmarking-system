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

**What gets scored is not what the Engine returned.** The Engine hands back the
UPDATED issueboard, which contains issues it was given. `benchmark.pipeline.scoring`
reduces that to the Engine's own delta — dropping seed issues it said nothing
about, the occurrence pairs it was handed, and occurrences naming traces that do
not exist — and scores that. Both boards are persisted: `predicted_issueboard.json`
verbatim (the assignment deliverable) and `scored_issueboard.json` next to it.

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
from benchmark.pipeline.engine import (
    EngineModelMismatch,
    EngineRunFailed,
    LangGraphEngineInvoker,
)
from benchmark.pipeline.export import ExportLeak, assert_export_file_clean
from benchmark.pipeline.manifest import ArtifactPaths, RunArtifacts, RunManifest, StageTiming
from benchmark.pipeline.render import render_markdown, severity_confusion
from benchmark.pipeline.runner import (
    AblationLineageBroken,
    PipelineRun,
    build_base_rates,
    run_pipeline,
    slice_inputs,
)
from benchmark.pipeline.scoring import ScoredBoard, prepare_scored_board, score_engine_delta
from benchmark.pipeline.servers import ServerLifetime, ServerStartFailed

__all__ = [
    "AblationLineageBroken",
    "AblationResult",
    "AblationStage",
    "AblationStageUnavailable",
    "ArtifactPaths",
    "DeliverableCheck",
    "EngineInvocation",
    "EngineInvoker",
    "EngineModelMismatch",
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
    "ScoredBoard",
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
    "prepare_scored_board",
    "render_markdown",
    "rescore_from_disk",
    "run_pipeline",
    "score_engine_delta",
    "severity_confusion",
    "slice_inputs",
]
