"""Config models — the black-box interfaces and stage knobs.

TargetAppConfig / EngineAppConfig are the ONLY things the benchmark knows about
the two apps (docs/execution-plan.md, "The black-box contract").
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TargetAppConfig(BaseModel):
    """configs/target_app.yaml — the entire knowledge surface for the target app."""

    base_url: str
    assistant_id: str
    langsmith_project: str  # v0 collection backend (see execution plan future work)
    # Declared shim surface (Mode C): shim kind -> config.configurable key name.
    fault_configurable_keys: dict[str, str] = Field(default_factory=dict)
    max_turns_supported: int = 8


class EngineAppConfig(BaseModel):
    """configs/engine.yaml — the entire knowledge surface for the Engine app."""

    base_url: str
    assistant_id: str
    model_configurable_key: str = "model"  # the Sol-vs-mini swap goes through here


class EngineConfig(BaseModel):
    """One Engine run's identity — recorded in the BenchmarkReport for attribution."""

    model: str  # the comparison axis
    app: EngineAppConfig | None = None
    max_tool_calls_per_trace: int = 50
    seed: int = 0


class AblationConfig(BaseModel):
    """Stage III knobs (docs/architecture/04-ablation-engine.md)."""

    seed: int = 0
    control_fraction: float = 0.3  # input-level split, stratified on provenance
    min_eligible: int = 5  # step-3 filter gate, within the ablate set
    n_per_category: int = 2  # step-1 proposals per category


class ScoringConfig(BaseModel):
    """Stage V knobs (docs/architecture/06-scoring.md)."""

    severity_alpha: float = 0.5  # linear slope for over-prediction (under is quadratic)
    description_mode: Literal["similarity", "judge"] = "judge"
    text_fallback_threshold: float = 0.5  # Layer-1 wrong-category fallback similarity gate
    headline_weights: dict[str, float] = Field(default_factory=dict)  # optional composite
