"""Input-side schemas — Stage I output (docs/architecture/01-data-schemas.md §1).

GenerationConfig lives here (not configs.py) because InputDataset embeds it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

DimensionKind = Literal["safe", "adversarial"]
PersonaKind = Literal["target", "adversarial"]
InputMode = Literal["single_turn", "multi_turn"]
GenerationMode = Literal["single_turn", "multi_turn", "mixed"]


class Dimension(BaseModel):
    """One orthogonal axis of the input space. D safe dims, A_c adversarial dims."""

    dim_id: str
    name: str  # e.g. "query_topic", "prompt_injection_style"
    kind: DimensionKind
    variations: list[str]  # V_D / V_AC concrete values


class Persona(BaseModel):
    persona_id: str
    name: str
    kind: PersonaKind  # P vs P_A
    description: str  # system prompt for the user-simulator LLM
    goals: list[str] = Field(default_factory=list)


class InputSpec(BaseModel):
    """One element of the [N] input dataset."""

    input_id: str
    mode: InputMode
    # provenance — which cell of the generation grid produced this
    # (Phase 5's stratified control/ablate split depends on these fields)
    dim_id: str
    variation: str
    persona_id: str | None = None  # multi-turn only
    fixed_adversarial_id: str | None = None  # if drawn from the A_F library
    # payload
    prompt: str | None = None  # single-turn: the literal user message
    scenario: str | None = None  # multi-turn: scenario brief for the simulator


class GenerationConfig(BaseModel):
    safe_dims: list[Dimension] = Field(default_factory=list)  # D with V_D
    adversarial_dims: list[Dimension] = Field(default_factory=list)  # A_c with V_AC
    fixed_adversarial: list[InputSpec] = Field(default_factory=list)  # A_F
    personas: list[Persona] = Field(default_factory=list)  # P
    adversarial_personas: list[Persona] = Field(default_factory=list)  # P_A
    mode: GenerationMode = "single_turn"
    max_turns: int = 1
    seed: int = 0  # reproducibility of LLM expansion


class InputDataset(BaseModel):
    dataset_id: str = ""  # content hash
    created_at: datetime | None = None
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)
    inputs: list[InputSpec] = Field(default_factory=list)  # len == N
