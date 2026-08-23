"""YAML config loading: configs/generation/*.yaml -> GenerationConfig."""

from __future__ import annotations

from pathlib import Path

import yaml

from benchmark.schemas.inputs import GenerationConfig


def load_generation_config(path: str | Path) -> GenerationConfig:
    """Parse a generation config YAML file into a validated GenerationConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"generation config not found: {path}")
    raw = yaml.safe_load(path.read_text())
    return GenerationConfig.model_validate(raw or {})
