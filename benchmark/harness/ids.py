"""Deterministic ids — what makes reruns idempotent and resumable.

`session_id = hash(dataset_id, input_id)` is stamped into the LangGraph run's
metadata, which is how the collector finds the run tree again in LangSmith.
The stored `Trace.trace_id` is derived from the same hash, so the runner can
ask `store.exists(trace_id)` *before* spending an app invocation — that is the
whole resumability mechanism (docs/architecture/03-trace-harness.md, "Scale
notes").
"""

from __future__ import annotations

import hashlib

# Field separator: without it, session_id("ab", "c") and session_id("a", "bc")
# would hash the same string.
_SEP = "\x1f"


def session_id_for(dataset_id: str, input_id: str, *, variant: str = "") -> str:
    """Deterministic run-metadata session id.

    `variant` separates re-runs of the same input that must NOT collide with
    the plain batch run — an armed Mode-C run, or a replay fork.
    """
    payload = _SEP.join(("v1", dataset_id, input_id, variant))
    return "s" + hashlib.sha256(payload.encode()).hexdigest()[:20]


def trace_id_for(session_id: str) -> str:
    """The stored Trace id for a session. Store-safe (no path separators)."""
    return f"trace-{session_id}"
