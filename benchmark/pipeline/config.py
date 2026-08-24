"""The pipeline's config surface — one YAML file per benchmark run.

A run is fully described by `configs/pipeline/<name>.yaml`: which generation
config to expand, which taxonomy the Engine is shown, where the two black-box
apps live (by config path, never by import), and the per-stage knobs. Paths in
the file are repo-root-relative; the loader finds the root by walking up to
`pyproject.toml` so the same config works from any working directory.

The root is deliberately NOT a model field: `benchmark.schemas.io.content_hash`
hashes every field it sees, and a manifest whose config hash changes with the
checkout directory would make "same config, same run" unverifiable across
machines. It lives on a private attribute instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr

from benchmark.models import ENGINE_MODEL_MINI
from benchmark.schemas.configs import AblationConfig, EngineAppConfig, ScoringConfig
from benchmark.schemas.inputs import InputMode
from benchmark.schemas.io import stamp_dataset_id
from benchmark.schemas.issues import OTHER_CATEGORY_ID, ErrorCategory, Issueboard

# The Engine's LangGraph server defaults to a recursion limit of 25, which caps
# a run at ~23 traces (2 supersteps + one per batch). Anything full-scale must
# pass its own; the compiled graph itself raises the default to 10 000.
DEFAULT_RECURSION_LIMIT = 10_000
# Ruled full-scale setting: 16 puts a 300-trace run at ~21 min against ~35 min
# at the default 8 (apps/engine/README.md, "Known gap: the straggler tax").
DEFAULT_ANALYSIS_CONCURRENCY = 16


class ServerSpec(BaseModel):
    """How to start/stop one black-box app's LangGraph server.

    A path to the app's own `scripts/serve.sh`, invoked as a subprocess. This
    is the whole mechanism: the pipeline never imports from `apps/`, and the
    script path lives in config rather than in code for the same reason every
    other fact about the apps does.
    """

    script: str
    label: str = ""


class HarnessStageConfig(BaseModel):
    concurrency: int = 8


class EngineStageConfig(BaseModel):
    """Knobs for the Engine invocation (apps/engine/README.md, "Invoking the Engine")."""

    model: str = ENGINE_MODEL_MINI
    analysis_concurrency: int = DEFAULT_ANALYSIS_CONCURRENCY
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    max_tool_calls_per_trace: int = 50
    seed: int = 0


class DeliverablesConfig(BaseModel):
    """The assignment's shape checks. `min_traces` is 300 at full scale and a
    handful in the miniature — the check is the same code either way."""

    min_traces: int = 300


class PipelineConfig(BaseModel):
    run_id: str
    generation_config: str
    taxonomy: str = "configs/taxonomy.yaml"
    target_app_config: str = "configs/target_app.yaml"
    engine_app_config: str = "configs/engine.yaml"
    seed_issueboard: str | None = None
    artifacts_root: str = "data/pipeline"
    expansion_cache: str = "data/expansion_cache"
    trace_store: str | None = None  # defaults to <run_dir>/trace_store
    # Miniature runs slice the generated grid rather than shipping a second
    # generation config: same inputs, fewer of them, deterministically chosen.
    max_inputs: int | None = None
    input_modes: list[InputMode] | None = None
    harness: HarnessStageConfig = Field(default_factory=HarnessStageConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    engine: EngineStageConfig = Field(default_factory=EngineStageConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    deliverables: DeliverablesConfig = Field(default_factory=DeliverablesConfig)
    servers: dict[Literal["target_app", "engine"], ServerSpec] = Field(default_factory=dict)

    _root: Path = PrivateAttr(default_factory=Path.cwd)

    @property
    def root(self) -> Path:
        return self._root

    def with_root(self, root: str | Path) -> PipelineConfig:
        clone = self.model_copy()
        clone._root = Path(root).resolve()
        return clone

    def resolve(self, path: str | Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else self._root / path

    @property
    def run_dir(self) -> Path:
        return self.resolve(self.artifacts_root) / self.run_id


def find_root(start: Path) -> Path:
    """The repo root: the nearest ancestor holding a pyproject.toml."""
    start = start.resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start.parent if start.is_file() else start


def load_pipeline_config(path: str | Path, root: str | Path | None = None) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pipeline config not found: {path}")
    cfg = PipelineConfig.model_validate(yaml.safe_load(path.read_text()) or {})
    return cfg.with_root(root if root is not None else find_root(path))


def load_taxonomy(path: str | Path) -> list[ErrorCategory]:
    """C_E — the public category vocabulary the Engine is shown.

    `other` is mandatory: without the escape hatch, an out-of-taxonomy (E_h)
    discovery has nowhere to go but a wrong category, and scoring then counts a
    real finding as a category error (docs/architecture/06-scoring.md).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"taxonomy not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    categories = [ErrorCategory.model_validate(item) for item in raw.get("categories", [])]
    if OTHER_CATEGORY_ID not in {c.category_id for c in categories}:
        raise ValueError(
            f"taxonomy {path} has no {OTHER_CATEGORY_ID!r} category — the escape hatch is "
            f"mandatory (docs/architecture/06-scoring.md)"
        )
    return categories


def load_engine_app_config(path: str | Path) -> EngineAppConfig:
    """configs/engine.yaml — the entire knowledge surface for the Engine app."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"engine app config not found: {path}")
    return EngineAppConfig.model_validate(yaml.safe_load(path.read_text()) or {})


def load_seed_board(path: str | Path | None) -> Issueboard:
    """The seed issueboard the Engine is handed, empty when none is provided.

    The assignment's input includes an issueboard and its output is the
    *updated* board, so both halves have to work: an empty board (the Engine
    starts from nothing) and a populated one (the Engine must attach
    occurrences to issues that already exist rather than duplicate them).

    `source` is forced to "seed" — a board handed in as ground truth or as a
    previous run's prediction is still, in this position, a seed. What is NOT
    tolerated is `injection_mode`: that field exists only on E_K entries and
    naming it to the Engine hands over which errors were manufactured.
    """
    if path is None:
        return stamp_dataset_id(Issueboard(source="seed"))
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"seed issueboard not found: {path}")
    board = Issueboard.model_validate_json(path.read_text())
    leaked = [i.error_id for i in board.issues if i.injection_mode is not None]
    if leaked:
        raise ValueError(
            f"seed issueboard {path} carries injection_mode on {leaked} — that field is "
            f"ground-truth-side only and must never reach the Engine"
        )
    return stamp_dataset_id(board.model_copy(update={"source": "seed"}))


def config_hashes(cfg: PipelineConfig) -> dict[str, str]:
    """Content hashes of every config that shaped a run, for the manifest."""
    from benchmark.schemas.io import content_hash  # noqa: PLC0415

    return {
        "pipeline": content_hash(cfg),
        "ablation": content_hash(cfg.ablation),
        "engine": content_hash(cfg.engine),
        "scoring": content_hash(cfg.scoring),
        "harness": content_hash(cfg.harness),
    }
