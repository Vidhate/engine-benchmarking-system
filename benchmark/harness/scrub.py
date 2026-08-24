"""Leak scrubbing — the BLOCKING hand-off requirement from the Phase 2 review.

The Engine is supposed to find injected faults by *reading the trace*. Anything
that names the fault instead is a leak. The target app controls what it writes
into spans (apps/target_app/README.md, "Trace leak surface"), but it does not
control what the LangGraph server and LangSmith record *about* a run: run
metadata, `configurable` echoes, thread/checkpoint config, tags, and the
`serialized` model manifest all ride along with every span.

Two layers, and neither replaces the other:

1. **Allowlist copying** (benchmark/harness/collector.py) — the normalizer
   copies a fixed set of fields out of each run and never passes a LangSmith
   payload through wholesale. This is what actually prevents leaks.
2. **The audit here** — a scan of the *finished* Trace for structural fault
   fingerprints. It is a tripwire for layer 1: if a future field is added to
   the allowlist and drags a fault name with it, collection fails loudly
   instead of shipping a poisoned corpus.
"""

from __future__ import annotations

import json
import re
from typing import Any

from benchmark.schemas.configs import TargetAppConfig

# Structural fingerprints — tokens that can only come from the fault machinery,
# never from ordinary support-assistant text. Plain behaviour words ("empty",
# "stale", "error", "timeout") are deliberately excluded: corpus text uses them
# legitimately, e.g. "Emptying the Trash".
STRUCTURAL_TOKENS: tuple[str, ...] = (
    "fault_",
    "shim",
    "ablat",
    "irrelevant_docs",
    "corrupted_result",
    "truncate_output",
    # Known Phase-2 model-identity fingerprint. The app overrides both the run
    # name and the serialized manifest so this never ships; the token stays
    # here so a regression in that override is caught at collection time.
    "supportchatmodel",
)

# Any run-metadata key with this prefix is a `configurable` echo of an armed
# fault. None of them may reach a stored Trace.
LEAK_KEY_PREFIXES: tuple[str, ...] = ("fault_",)


class LeakDetected(Exception):
    """A fault fingerprint reached a normalized Trace (or a payload bound for one)."""


def leak_tokens(
    cfg: TargetAppConfig | None = None, extra: tuple[str, ...] | list[str] = ()
) -> tuple[str, ...]:
    """The token set to scan for.

    Config-declared fault key names are included so the scan follows the
    declared shim surface rather than a hardcoded copy of it; `extra` carries
    per-run additions (e.g. the behaviour string a Mode-C run armed).
    """
    declared = tuple(cfg.fault_configurable_keys.values()) if cfg else ()
    return tuple(dict.fromkeys(t.lower() for t in (*STRUCTURAL_TOKENS, *declared, *extra) if t))


def find_leaks(payload: Any, tokens: tuple[str, ...] | list[str] = STRUCTURAL_TOKENS) -> list[str]:
    """Tokens from `tokens` present anywhere in `payload`, serialized.

    Matching is left-anchored on a word boundary, not a bare substring: the
    token "fault_" is a substring of the entirely innocent "default_headers",
    and a scrubber that quarantines healthy traces gets switched off.
    """
    blob = json.dumps(payload, default=str).lower()
    found: set[str] = set()
    for token in tokens:
        # Report the whole offending word, not the stem that matched: a report
        # saying "fault_" is far less actionable than "fault_retriever".
        pattern = r"(?<![a-z0-9_])" + re.escape(token.lower()) + r"[a-z0-9_]*"
        found.update(match.group(0) for match in re.finditer(pattern, blob))
    return sorted(found)


def find_leaked_keys(payload: Any, prefixes: tuple[str, ...] = LEAK_KEY_PREFIXES) -> list[str]:
    """Dict keys anywhere in `payload` that look like a `configurable` echo."""
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(key, str) and key.lower().startswith(prefixes):
                    found.add(key)
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(payload)
    return sorted(found)


def assert_no_leak(
    payload: Any,
    *,
    where: str,
    tokens: tuple[str, ...] | list[str] = STRUCTURAL_TOKENS,
) -> None:
    """Raise LeakDetected if `payload` names a fault. Never returns a bool —
    a leak must not be something a caller can forget to check."""
    leaked_tokens = find_leaks(payload, tokens)
    leaked_keys = find_leaked_keys(payload)
    if leaked_tokens or leaked_keys:
        raise LeakDetected(
            f"fault fingerprints reached {where}: "
            f"tokens={leaked_tokens} metadata_keys={leaked_keys}"
        )
