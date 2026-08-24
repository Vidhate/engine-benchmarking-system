"""Model selection — the one axis the benchmark varies.

`config.configurable["model"]` (the key declared in `configs/engine.yaml`) picks
the model for a run. Nothing else about the Engine changes between arms, which
is what makes the Sol-vs-mini comparison a clean swap: same prompts, same tools,
same consolidation code, different model id.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

MODEL_CONFIGURABLE_KEY = "model"
MODEL_ENV_VAR = "ENGINE_MODEL"
# NOTE (same as apps/target_app): the plan names "gpt-5.1-mini"; this account's
# API returns 404 for that id, so the closest available small model is the
# default. Overridden per run via configurable, or globally via ENGINE_MODEL.
DEFAULT_MODEL = "gpt-5-mini"


def resolve_model_name(config: dict[str, Any] | None) -> str:
    """Run config wins, then ENGINE_MODEL, then the pinned default."""
    configurable = (config or {}).get("configurable") or {}
    requested = configurable.get(MODEL_CONFIGURABLE_KEY)
    if isinstance(requested, str) and requested.strip():
        return requested.strip()
    return os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL


def build_model(name: str) -> BaseChatModel:
    """Chat Completions (not the Responses API) so message content stays a
    plain string, matching how the target app's traces were produced."""
    return ChatOpenAI(model=name, use_responses_api=False)
