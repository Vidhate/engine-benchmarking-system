"""Mode C fault shims — the app-side half of the black-box fault contract.

Every dependency (retriever, action tool, LLM output) reads its fault
instruction from `config.configurable[<declared key>]` on the run. The
benchmark only ever knows the *key names*, which are published in
`configs/target_app.yaml`; the behaviours and their implementation live here.

No fault key present => completely normal behaviour.

Design rule (docs/architecture/04-ablation-engine.md): activation must be
*organically* visible in the trace spans — never announced by a marker the
Engine could pattern-match on.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from target_app.retriever import RetrievedDoc, Retriever

FAULT_RETRIEVER_KEY = "fault_retriever"
FAULT_TOOL_KEY = "fault_tool"
FAULT_LLM_KEY = "fault_llm"

FAULT_BEHAVIORS: dict[str, frozenset[str]] = {
    FAULT_RETRIEVER_KEY: frozenset({"irrelevant_docs", "empty", "stale"}),
    FAULT_TOOL_KEY: frozenset({"error", "timeout", "corrupted_result"}),
    FAULT_LLM_KEY: frozenset({"truncate_output"}),
}

MAX_TIMEOUT_DELAY_SECONDS = 5.0
DEFAULT_TIMEOUT_DELAY_SECONDS = 1.5
DEFAULT_TRUNCATE_FRACTION = 0.4

_WORD_SPAN = re.compile(r"\S+")


@dataclass(frozen=True)
class Fault:
    behavior: str
    params: dict[str, Any] = field(default_factory=dict)


def read_fault(config: Any, key: str) -> Fault | None:
    """Extract the fault armed under `key`, or None when the shim is disarmed.

    The value MUST be a mapping, e.g. `{"behavior": "stale"}`. A scalar
    (`"stale"`) is refused: `langchain_core.runnables.config.ensure_config`
    promotes every str/int/float/bool `configurable` entry into the run's
    tracing metadata, which would stamp "this trace was ablated" onto every
    span of every armed run. A mapping is not promoted.
    """
    if key not in FAULT_BEHAVIORS:
        raise ValueError(f"{key!r} is not a declared fault key")
    configurable = (config or {}).get("configurable") or {}
    value = configurable.get(key)
    if not value:
        return None

    if not isinstance(value, dict):
        raise ValueError(
            f"{key!r} must be armed with a mapping such as {{'behavior': ...}}, got "
            f"{type(value).__name__}; scalar values leak into tracing metadata"
        )

    behavior = value.get("behavior") or value.get("mode") or ""
    params = dict(value.get("params") or {})
    params.update({k: v for k, v in value.items() if k not in ("behavior", "mode", "params")})

    if behavior not in FAULT_BEHAVIORS[key]:
        raise ValueError(
            f"unknown {key} behavior {behavior!r}; expected one of "
            f"{sorted(FAULT_BEHAVIORS[key])}"
        )
    return Fault(behavior, params)


# ------------------------------------------------------------------ retriever

def apply_retriever_fault(
    fault: Fault | None,
    *,
    query: str,
    retriever: Retriever,
    k: int = 3,
) -> list[RetrievedDoc]:
    """Run retrieval, honouring an armed retriever fault."""
    if fault is None:
        return retriever.search(query, k=k)
    if fault.behavior == "empty":
        return []
    if fault.behavior == "irrelevant_docs":
        return retriever.worst_matches(query, k=k)
    if fault.behavior == "stale":
        return retriever.stale_view(retriever.search(query, k=k))
    raise ValueError(f"unhandled retriever behavior {fault.behavior!r}")


# ----------------------------------------------------------------------- tool

CORRUPTED_TICKET_ID = "NN-000000"
CORRUPTED_TEXT = "����"


def apply_tool_fault(fault: Fault | None, result: dict[str, Any]) -> dict[str, Any]:
    """Transform a normal tool result according to an armed tool fault."""
    if fault is None:
        return result

    if fault.behavior == "error":
        return {
            "status": "error",
            "error": "ticket_service_unavailable",
            "detail": "upstream ticketing service returned HTTP 503",
        }

    if fault.behavior == "timeout":
        delay = float(fault.params.get("delay_seconds", DEFAULT_TIMEOUT_DELAY_SECONDS))
        time.sleep(max(0.0, min(delay, MAX_TIMEOUT_DELAY_SECONDS)))
        return {
            "status": "error",
            "error": "timeout",
            "detail": "upstream ticketing service did not respond in time",
        }

    if fault.behavior == "corrupted_result":
        corrupted = dict(result)
        corrupted["ticket_id"] = CORRUPTED_TICKET_ID
        for key in ("subject", "priority", "queue"):
            if key in corrupted:
                corrupted[key] = CORRUPTED_TEXT
        corrupted["eta_hours"] = -1
        return corrupted

    raise ValueError(f"unhandled tool behavior {fault.behavior!r}")


# ------------------------------------------------------------------------ llm

def apply_llm_fault(fault: Fault | None, text: str) -> str:
    """Post-process an assistant message according to an armed LLM fault."""
    if fault is None or not text:
        return text
    if fault.behavior == "truncate_output":
        fraction = float(fault.params.get("fraction", DEFAULT_TRUNCATE_FRACTION))
        fraction = max(0.0, min(fraction, 1.0))
        spans = list(_WORD_SPAN.finditer(text))
        if not spans:
            return text
        keep = max(1, int(len(spans) * fraction))
        if keep >= len(spans):
            return text
        return text[: spans[keep - 1].end()]
    raise ValueError(f"unhandled llm behavior {fault.behavior!r}")
