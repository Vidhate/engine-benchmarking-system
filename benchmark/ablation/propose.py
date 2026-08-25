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
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from benchmark.ablation.agent import (
    BEHAVIOR_VOCABULARY,
    AblationAgent,
    AgentTransportError,
    CorpusDigest,
    ProposedError,
)
from benchmark.ablation.filters import known_field, known_op
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


#: The mode-coverage floor step 1 aims for, as {mode: minimum proposals}.
#:
#: `dependency_fault: 1` is a REPORT requirement, not a taste: the benchmark's
#: content-vs-mechanism commentary compares how the system under test does on
#: errors planted in what the app *said* against errors planted in what
#: happened *to* it, and with zero mechanism errors that half of the report has
#: no data behind it. Measured need — two live mini runs proposed `replay_edit`
#: for all seven categories, because most of the taxonomy is content-shaped and
#: each category is drafted in its own independent call with no view of the
#: run's overall mode balance.
#:
#: It lives here rather than on `AblationConfig` deliberately. A floor of one
#: is a property of what a publishable report needs, identical for the mini and
#: the submission run, and a knob would only invite it being turned down to
#: zero on the run where it matters.
DEFAULT_MIN_PER_MODE: dict[InjectionMode, int] = {"dependency_fault": 1}

#: Words in a category's name or description that suggest its symptom is
#: something that happens TO the app rather than something the app says.
#: ORDERING ONLY — never an exclusion. Every category is still offered the
#: mechanism re-prompt; this just decides who gets asked first, so the usual
#: cost of the coverage pass is one extra agent call rather than seven.
_MECHANISM_HINTS: tuple[str, ...] = (
    "retriev", "search", "document", "corpus", "tool", "api", "call",
    "timeout", "truncat", "format", "cut off", "incomplete", "stale",
)


def _mechanism_affinity(category: ErrorCategory) -> int:
    text = f"{category.name} {category.category_id} {category.description}".lower()
    return sum(hint in text for hint in _MECHANISM_HINTS)


def _draft(
    agent: AblationAgent,
    category: ErrorCategory,
    n: int,
    digest: CorpusDigest,
    modes: Sequence[InjectionMode],
    dropped: dict[str, str],
    drop_key: str,
) -> list[ProposedError]:
    """One `agent.propose` call, with the retry policy and the drop bookkeeping.

    **One bad draw costs its category, never the run**: by the time step 1
    executes the whole corpus has already been collected and paid for, so a
    transport blip or a malformed reply must not discard it. Transport failures
    get one retry; a response the parser cannot use is not retried (a second
    identical prompt rarely helps).
    """
    try:
        return agent.propose(category, n, digest, modes)
    except AgentTransportError as exc:
        log.warning(
            "category %s: agent transport failure (%s) — retrying once",
            category.category_id,
            exc,
        )
        try:
            return agent.propose(category, n, digest, modes)
        except Exception as retry_exc:  # noqa: BLE001 - the category is dropped either way
            reason = (
                f"category {category.category_id} dropped: the proposing agent failed "
                f"twice ({type(retry_exc).__name__}: {retry_exc})"
            )
            log.warning("%s", reason)
            dropped[drop_key] = reason
            return []
    except Exception as exc:  # noqa: BLE001 - a bad reply must not kill the run
        reason = (
            f"category {category.category_id} dropped: the proposing agent returned "
            f"nothing usable ({type(exc).__name__}: {exc})"
        )
        log.warning("%s", reason)
        dropped[drop_key] = reason
        return []


def _sanitise(
    drafts: Sequence[ProposedError],
    category: ErrorCategory,
    shims: Sequence[str],
    *,
    n: int,
    start_index: int,
) -> list[ProposedError]:
    """Re-stamp ids, drop unusable drafts and hallucinated filter steps.

    `start_index` keeps the coverage pass's ids from colliding with the ids the
    first pass already handed out for the same category.
    """
    out: list[ProposedError] = []
    for offset, draft in enumerate(drafts[:n]):
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
                "error_id": f"E-{category.category_id}-{start_index + offset:02d}",
                "category_id": category.category_id,
            }
        )
        # One hallucinated field or operator costs that step, not the error:
        # letting either through would raise out of the middle of validation
        # and take the whole run with it.
        steps = []
        for step in draft.filter_steps:
            if not known_field(step.field):
                log.warning(
                    "%s: dropping filter step on unknown field %r", issue.error_id, step.field
                )
            elif not known_op(step.op):
                log.warning(
                    "%s: dropping filter step with unsupported op %r on %r",
                    issue.error_id,
                    step.op,
                    step.field,
                )
            else:
                steps.append(step)
        out.append(draft.model_copy(update={"issue": issue, "filter_steps": steps}))
    return out


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
    min_per_mode: Mapping[InjectionMode, int] | None = None,
) -> tuple[list[ProposedError], CorpusDigest, dict[str, str]]:
    """`[N,M,T]`, `C_E` -> `[E, C_E]` — concrete drafts, one batch per category.

    `error_id` is re-stamped here so ids are unique across categories however
    the agent numbered them, and unusable drafts are dropped with a logged
    reason rather than being carried into planning to fail later.

    Returns `(proposals, digest, dropped_categories)`. Every drop carries a
    reason out to `AblationResult.dropped_errors`.

    ## The mode-coverage pass

    Each category is drafted in its own call, so no single call can see the
    run's mode balance — and since most of a support-assistant taxonomy is
    content-shaped, the natural draw is `replay_edit` everywhere. After the
    first pass, any mode still short of `min_per_mode` gets a SECOND, narrower
    round: the same categories are re-asked with `allowed_modes` restricted to
    that one mode, most-plausible category first, stopping the moment the floor
    is met. Usually one extra call; at worst one per category.

    Three things it deliberately does not do:

    * it never runs for a mode the app cannot support — `allowed_modes(cfg)`
      already excludes `dependency_fault` when no shim is declared, so a
      shimless app takes no extra calls and raises no error;
    * it never forces a fault. The re-prompt asks; a category that cannot carry
      a mechanism error returns nothing usable and the pass moves on. The floor
      is a target, not a guarantee, because the only way to guarantee it would
      be to fabricate a proposal the app cannot actually inject;
    * it never hides the outcome. A coverage pass that ends short logs and
      records why, so a report built on one mode says so out loud instead of
      looking like a balanced run.
    """
    digest = digest or build_digest(store, trace_ids, cfg, app_context=app_context)
    modes = allowed_modes(cfg)
    shims = available_shims(cfg)
    floors = dict(DEFAULT_MIN_PER_MODE if min_per_mode is None else min_per_mode)

    out: list[ProposedError] = []
    dropped: dict[str, str] = {}
    per_category: Counter[str] = Counter()
    for category in categories:
        drafts = _draft(
            agent, category, n_per_category, digest, modes, dropped, category.category_id
        )
        kept = _sanitise(drafts, category, shims, n=n_per_category, start_index=0)
        per_category[category.category_id] += len(kept)
        out.extend(kept)

    for mode, floor in sorted(floors.items()):
        if mode not in modes:
            log.info(
                "mode-coverage: %s is not available for this app (declared shims: %s) — "
                "no coverage pass",
                mode,
                shims or "none",
            )
            continue
        have = sum(1 for p in out if p.issue.injection_mode == mode)
        if have >= floor:
            continue
        log.info(
            "mode-coverage: %d/%d %s proposal(s) after the first pass — re-prompting",
            have,
            floor,
            mode,
        )
        ranked = sorted(
            categories, key=lambda c: (-_mechanism_affinity(c), c.category_id)
        )
        for category in ranked:
            if have >= floor:
                break
            drafts = _draft(
                agent,
                category,
                floor - have,
                digest,
                [mode],
                dropped,
                f"{category.category_id}:{mode}-coverage",
            )
            kept = [
                p
                for p in _sanitise(
                    drafts,
                    category,
                    shims,
                    n=floor - have,
                    start_index=per_category[category.category_id],
                )
                if p.issue.injection_mode == mode
            ]
            if not kept:
                continue
            per_category[category.category_id] += len(kept)
            out.extend(kept)
            have += len(kept)
            log.info(
                "mode-coverage: category %s supplied %d %s proposal(s)",
                category.category_id,
                len(kept),
                mode,
            )
        if have < floor:
            reason = (
                f"mode coverage: only {have}/{floor} {mode!r} proposal(s) after re-prompting "
                f"every category — this run's errors are {mode}-poor and any "
                f"content-vs-mechanism comparison built on it is one-sided"
            )
            log.warning("%s", reason)
            dropped[f"{mode}-coverage"] = reason

    return out, digest, dropped


def proposed_issues(proposals: Sequence[ProposedError]) -> list[Issue]:
    """The `[E, C_E]` view — the Issues alone, `injection_mode` set on each."""
    return [p.issue for p in proposals]


def trace_ids_of(traces: Iterable[Trace]) -> list[str]:
    return [t.trace_id for t in traces]
