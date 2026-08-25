"""Model selection — the one axis the benchmark varies.

`config.configurable["model"]` (the key declared in `configs/engine.yaml`) picks
the model for a run. Nothing else about the Engine changes between arms, which
is what makes the Sol-vs-mini comparison a clean swap: same prompts, same tools,
same consolidation code, different model id.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

MODEL_CONFIGURABLE_KEY = "model"
MODEL_ENV_VAR = "ENGINE_MODEL"
# NOTE (same as apps/target_app): the plan names "gpt-5.1-mini"; this account's
# API returns 404 for that id, so the closest available small model is the
# default. Overridden per run via configurable, or globally via ENGINE_MODEL.
DEFAULT_MODEL = "gpt-5-mini"

# Per-request timeout, in seconds. The SDK's own default is 600 s, and that is
# not a theoretical number here: one hung analysis request stalled a whole
# batch for exactly 600 s before the transport gave up. A batch costs its
# SLOWEST trace (apps/engine/README.md, "the straggler tax"), so a single
# stuck request does not cost 600 s of one trace — it costs 600 s of the entire
# batch. 120 s is generous against the ~49 s a real batch takes.
REQUEST_TIMEOUT_S = 120
# Bounded retries on top of that. A retry is the right answer to the two
# failures this transport actually sees — a 429 and a dropped connection — and
# `max_retries` is the SDK's own exponential backoff, so this costs nothing on
# a healthy run. Two, not more: the Engine already runs `analysis_concurrency`
# requests at once, and a deep retry ladder on every one of them turns a
# provider blip into a much longer stall than the failure it is papering over.
MAX_RETRIES = 2


def resolve_model_name(config: Mapping[str, Any] | None) -> str:
    """Run config wins, then ENGINE_MODEL, then the pinned default.

    NOTE for callers: LangGraph only injects the run config into a node whose
    `config` parameter is annotated `RunnableConfig`. A `dict[str, Any]`
    annotation is silently ignored and the node sees no config at all — which
    looks exactly like "the caller did not pass a model", i.e. both arms of the
    comparison quietly run the default. See `graph.analyze_node`.
    """
    configurable = (config or {}).get("configurable") or {}
    requested = configurable.get(MODEL_CONFIGURABLE_KEY)
    if isinstance(requested, str) and requested.strip():
        return requested.strip()
    return os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL


def build_model(name: str) -> BaseChatModel:
    """Chat Completions (not the Responses API) so message content stays a
    plain string, matching how the target app's traces were produced.

    `timeout` and `max_retries` are NOT decoration: see the constants above.
    They are set here, on the one construction site, so every Engine node —
    analysis and consolidation alike — inherits the same bounded call.
    """
    return ChatOpenAI(
        model=name,
        use_responses_api=False,
        timeout=REQUEST_TIMEOUT_S,
        max_retries=MAX_RETRIES,
    )
