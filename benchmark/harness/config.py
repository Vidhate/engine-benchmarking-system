"""The config-only knowledge surface for the target app.

docs/execution-plan.md ground rule 4: everything the benchmark knows about the
target app comes from `configs/target_app.yaml`, parsed through the Phase-0
`TargetAppConfig` model. Nothing else about the app may be hardcoded here.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from benchmark.schemas.configs import TargetAppConfig

DEFAULT_TARGET_APP_CONFIG = Path("configs/target_app.yaml")


def load_target_app_config(path: str | Path = DEFAULT_TARGET_APP_CONFIG) -> TargetAppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"target app config not found: {path}")
    return TargetAppConfig(**(yaml.safe_load(path.read_text()) or {}))
