"""The leak-stripped Engine export, and the audit that proves it is stripped.

docs/architecture/04-ablation-engine.md, "Leak-proofing": the copy shipped to
Engine strips `ablation_ids`, `AblationRecord`s, `injection_mode`, split
membership, and any formatting artifact that would fingerprint an ablated
trace. `apps/engine/README.md` names the input contract: a JSON file of `Trace`
objects (a `{"traces": [...]}` dataset or a bare list) — nothing else crosses.

Stripping is an **allowlist rebuild**, never a delete-these-keys pass. A new
field added to `Trace` (or to a trace's metadata by a future collector) is
excluded by default and has to be added here deliberately, which is the only
version of this that stays correct as the schema moves.

## What gets dropped, and why each one is a leak

| dropped | why |
|---|---|
| `ablation_ids` | names the injection outright |
| `dataset_id` / `parent_dataset_id` | lineage says "this set was derived" |
| `metadata.thread_id`, `turn_checkpoints`, `*_checkpoint_id` | the replay
  surface: a fork checkpoint means a Mode-A trace |
| `metadata.replayed` | a boolean that *is* the ground truth |
| `metadata.ablation_parent_trace_id` | points straight at the original |
| `metadata.session_id`, `langsmith_*` | collection-side ids; the ablated
  variants hash differently, and none of it is app behavior |

`metadata.app` and `metadata.turn_count` survive: both are properties of the
conversation an Engine may legitimately read, and both are identical on control
and ablated traces.

## Known limitation, accepted deliberately

Semantic tells survive. A `stale`-armed retrieval span really does return
documents with old dates, because the app serves an archived revision rather
than a synthetic placeholder (`apps/target_app/README.md`). That is the error
being injected; scrubbing it would scrub the thing Engine is supposed to find.
No structural fingerprint accompanies it — no marker, no flag, no score — so
an Engine that spots it did so by reading the trace. Documented, not solved.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from benchmark.harness.scrub import find_leaked_keys, find_leaks, leak_tokens
from benchmark.schemas.configs import TargetAppConfig
from benchmark.schemas.traces import Trace, TraceDataset

# The exact key set each level of the export may carry.
TRACE_FIELDS: tuple[str, ...] = ("trace_id", "input_id", "mode", "turns", "status", "metadata")
TURN_FIELDS: tuple[str, ...] = ("turn_index", "user_message", "final_response", "spans")
SPAN_FIELDS: tuple[str, ...] = (
    "span_id",
    "parent_span_id",
    "name",
    "span_type",
    "start_time",
    "end_time",
    "inputs",
    "outputs",
    "attributes",
)
METADATA_FIELDS: tuple[str, ...] = ("app", "turn_count")

# Ground-truth vocabulary that must never appear anywhere in the export, on top
# of the harness's structural fault fingerprints.
ABLATION_TOKENS: tuple[str, ...] = (
    "ablation",
    "injection_mode",
    "replay_edit",
    "dependency_fault",
    "ground_truth",
    "control_input",
    "ablate_input",
    "fork_checkpoint",
    "source_checkpoint",
    "turn_checkpoints",
    "thread_id",
    "replayed",
)


class ExportLeak(Exception):
    """The Engine-facing export carries something only the ground truth should know."""


def _strip_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: metadata[key] for key in METADATA_FIELDS if key in metadata}


def strip_trace(trace: Trace) -> dict[str, Any]:
    """One `Trace` rebuilt from the allowlist, as plain JSON-able data."""
    payload = trace.model_dump(mode="json")
    out = {key: payload[key] for key in TRACE_FIELDS if key in payload}
    out["metadata"] = _strip_metadata(payload.get("metadata") or {})
    out["turns"] = [
        {
            **{key: turn[key] for key in TURN_FIELDS if key in turn and key != "spans"},
            "spans": [
                {key: span[key] for key in SPAN_FIELDS if key in span}
                for span in (turn.get("spans") or [])
            ],
        }
        for turn in (payload.get("turns") or [])
    ]
    return out


def build_export(traces: Iterable[Trace]) -> list[dict[str, Any]]:
    """The Engine-facing payload: a bare list of stripped traces."""
    return [strip_trace(trace) for trace in traces]


def _allowlist_violations(payload: Sequence[dict[str, Any]]) -> list[str]:
    violations: list[str] = []
    for trace in payload:
        stray = sorted(set(trace) - set(TRACE_FIELDS))
        if stray:
            violations.append(f"trace {trace.get('trace_id')}: extra fields {stray}")
        stray_meta = sorted(set(trace.get("metadata") or {}) - set(METADATA_FIELDS))
        if stray_meta:
            violations.append(f"trace {trace.get('trace_id')}: metadata keys {stray_meta}")
        for turn in trace.get("turns") or []:
            stray_turn = sorted(set(turn) - set(TURN_FIELDS))
            if stray_turn:
                violations.append(f"trace {trace.get('trace_id')}: turn fields {stray_turn}")
            for span in turn.get("spans") or []:
                stray_span = sorted(set(span) - set(SPAN_FIELDS))
                if stray_span:
                    violations.append(f"trace {trace.get('trace_id')}: span fields {stray_span}")
    return violations


def audit_export(
    payload: Sequence[dict[str, Any]],
    cfg: TargetAppConfig | None = None,
    *,
    extra_tokens: Sequence[str] = (),
) -> None:
    """Two independent checks; raise `ExportLeak` on either.

    1. **Field allowlist** — every key at every level is one this module names.
    2. **Token scan** — the harness's structural fault fingerprints
       (`benchmark/harness/scrub.py`) plus this module's ablation vocabulary,
       matched left-anchored on a word boundary so "fault_" cannot fire on
       "default_headers".

    Neither replaces the other: the allowlist stops a *new* field from riding
    along, the token scan catches a leak inside a field that is legitimately
    exported (a fault name that reached a span's outputs, say).
    """
    violations = _allowlist_violations(payload)
    if violations:
        raise ExportLeak("the Engine export carries non-allowlisted fields:\n" + "\n".join(
            violations[:12]
        ))

    tokens = leak_tokens(cfg, tuple(ABLATION_TOKENS) + tuple(extra_tokens))
    leaked = find_leaks(payload, tokens)
    keys = find_leaked_keys(payload)
    if leaked or keys:
        raise ExportLeak(
            f"ground-truth fingerprints reached the Engine export: "
            f"tokens={leaked} metadata_keys={keys}"
        )


def write_engine_export(
    traces: TraceDataset | Sequence[Trace],
    path: str | Path,
    cfg: TargetAppConfig | None = None,
    *,
    extra_tokens: Sequence[str] = (),
) -> Path:
    """Strip, audit, then write. A failed audit writes nothing.

    Ordering is the whole point: an export that is written first and audited
    second leaves a poisoned file on disk for a caller to pick up after the
    exception is logged and forgotten.
    """
    items = traces.traces if isinstance(traces, TraceDataset) else list(traces)
    payload = build_export(items)
    audit_export(payload, cfg, extra_tokens=extra_tokens)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return out
