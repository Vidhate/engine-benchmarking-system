"""Step 1 — propose concrete errors from the taxonomy plus the real corpus.

docs/architecture/04-ablation-engine.md, step 1: an agent with trace tools
explores the corpus and drafts concrete, app-specific errors under each
high-level category. "Grounding proposals in *real traces of this app* (not a
generic error list) is what makes injected errors plausible."

Exploration goes through the `TraceStore` (the Phase-0 tracing boundary), never
LangSmith: `build_digest` reads traces by id and summarizes what the app
actually does — its span types, tool and retriever span names, the documents it
retrieves, and a sample of real exchanges.

Mode assignment is bounded by the app's **declared** shim surface: a
`dependency_fault` is only proposable for a shim kind that maps onto a key in
`TargetAppConfig.fault_configurable_keys`. With no declared keys the benchmark
degrades gracefully to `replay_edit`-only (locked decision #1).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence

from benchmark.ablation.agent import (
    BEHAVIOR_VOCABULARY,
    AblationAgent,
    CorpusDigest,
    ProposedError,
)
from benchmark.harness.faults import SHIM_TO_CONFIG_KIND
from benchmark.schemas.configs import TargetAppConfig
from benchmark.schemas.issues import ErrorCategory, InjectionMode, Issue
from benchmark.schemas.traces import Trace
from benchmark.tracing.store import TraceStore

log = logging.getLogger("benchmark.ablation")

MAX_SAMPLE_EXCHANGES = 8
MAX_RESPONSE_CHARS = 600


def available_shims(cfg: TargetAppConfig) -> list[str]:
    """The `ShimKind`s this app declares a `configurable` key for."""
    return sorted(
        shim
        for shim, config_kind in SHIM_TO_CONFIG_KIND.items()
        if config_kind in cfg.fault_configurable_keys and shim in BEHAVIOR_VOCABULARY
    )


def allowed_modes(cfg: TargetAppConfig) -> list[InjectionMode]:
    """`replay_edit` always; `dependency_fault` only with a declared shim."""
    modes: list[InjectionMode] = ["replay_edit"]
    if available_shims(cfg):
        modes.append("dependency_fault")
    return modes


def _doc_ids(payload: object) -> list[str]:
    """Document identifiers out of a retrieval span's output, best effort."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (ValueError, TypeError):
            return []
    if isinstance(payload, dict):
        payload = payload.get("output") or payload.get("documents") or payload.get("docs") or []
    out: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                identifier = item.get("doc_id") or item.get("id") or item.get("source")
                if isinstance(identifier, str):
                    out.append(identifier)
    return out


def build_digest(
    store: TraceStore,
    trace_ids: Iterable[str],
    cfg: TargetAppConfig,
    *,
    app_context: str = "",
    max_samples: int = MAX_SAMPLE_EXCHANGES,
) -> CorpusDigest:
    """Summarize the corpus for the proposing agent, via TraceStore reads only."""
    digest = CorpusDigest(available_shims=available_shims(cfg), app_context=app_context)
    doc_ids: list[str] = []
    span_names: list[str] = []
    tool_names: list[str] = []

    for trace_id in sorted(trace_ids):
        try:
            trace = store.get(trace_id)
        except KeyError:
            log.warning("digest: trace %s is not in the store — skipping", trace_id)
            continue
        digest.n_traces += 1
        digest.modes[trace.mode] = digest.modes.get(trace.mode, 0) + 1
        for turn in trace.turns:
            for span in turn.spans:
                digest.span_types[span.span_type] = digest.span_types.get(span.span_type, 0) + 1
                span_names.append(span.name)
                if span.span_type == "tool":
                    tool_names.append(span.name)
                if span.span_type == "retrieval":
                    doc_ids.extend(_doc_ids(span.outputs))
        if len(digest.sample_exchanges) < max_samples and trace.turns:
            turn = trace.turns[0]
            digest.sample_exchanges.append(
                {
                    "trace_id": trace.trace_id,
                    "mode": trace.mode,
                    "turns": len(trace.turns),
                    "user_message": turn.user_message[:MAX_RESPONSE_CHARS],
                    "final_response": turn.final_response[:MAX_RESPONSE_CHARS],
                    "span_types": sorted({s.span_type for s in turn.spans}),
                }
            )

    digest.span_names = sorted(set(span_names))
    digest.tool_names = sorted(set(tool_names))
    digest.retrieved_doc_ids = sorted(set(doc_ids))
    return digest


def _usable(proposal: ProposedError, shims: Sequence[str]) -> str | None:
    """The reason this draft cannot be used, or None."""
    mode = proposal.issue.injection_mode
    if mode not in ("replay_edit", "dependency_fault"):
        return f"injection_mode is {mode!r}, which is not one of the two modes"
    if mode == "replay_edit":
        if proposal.corruption is None:
            return "replay_edit proposal carries no corruption to inject"
        if proposal.corruption.marker not in proposal.corruption.replacement:
            return "the corruption's marker does not appear in its replacement text"
        return None
    if proposal.fault is None:
        return "dependency_fault proposal carries no fault_config"
    if proposal.fault.shim not in shims:
        return (
            f"shim {proposal.fault.shim!r} is not part of the app's declared fault "
            f"surface (declared: {list(shims)})"
        )
    if not proposal.fault.behavior:
        return "fault_config has no behavior"
    return None


def propose_errors(
    store: TraceStore,
    trace_ids: Iterable[str],
    categories: Sequence[ErrorCategory],
    n_per_category: int,
    cfg: TargetAppConfig,
    agent: AblationAgent,
    *,
    app_context: str = "",
    digest: CorpusDigest | None = None,
) -> tuple[list[ProposedError], CorpusDigest]:
    """`[N,M,T]`, `C_E` -> `[E, C_E]` — concrete drafts, one batch per category.

    `error_id` is re-stamped here so ids are unique across categories however
    the agent numbered them, and unusable drafts are dropped with a logged
    reason rather than being carried into planning to fail later.
    """
    digest = digest or build_digest(store, trace_ids, cfg, app_context=app_context)
    modes = allowed_modes(cfg)
    shims = available_shims(cfg)

    out: list[ProposedError] = []
    for category in categories:
        drafts = agent.propose(category, n_per_category, digest, modes)
        for index, draft in enumerate(drafts[:n_per_category]):
            reason = _usable(draft, shims)
            if reason is not None:
                log.warning(
                    "dropping proposal %r for category %s: %s",
                    draft.issue.title,
                    category.category_id,
                    reason,
                )
                continue
            issue = Issue(
                **{
                    **draft.issue.model_dump(),
                    "error_id": f"E-{category.category_id}-{index:02d}",
                    "category_id": category.category_id,
                }
            )
            out.append(draft.model_copy(update={"issue": issue}))
    return out, digest


def proposed_issues(proposals: Sequence[ProposedError]) -> list[Issue]:
    """The `[E, C_E]` view — the Issues alone, `injection_mode` set on each."""
    return [p.issue for p in proposals]


def trace_ids_of(traces: Iterable[Trace]) -> list[str]:
    return [t.trace_id for t in traces]
