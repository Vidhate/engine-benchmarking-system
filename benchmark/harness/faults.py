"""Mode C — arming a declared dependency fault, and proving it activated.

docs/architecture/04-ablation-engine.md: ground truth for `dependency_fault` is
defined at the **mechanism** level ("retrieval returned irrelevant documents"),
never at the outcome level. Validation therefore checks **activation only** —
the fault is visible in the regenerated spans — and never whether the final
answer got worse.

The benchmark knows nothing about the app beyond `fault_configurable_keys` in
`configs/target_app.yaml`, so everything here is expressed in terms of that
declaration plus the shim kind on the `FaultConfig`.
"""

from __future__ import annotations

import json

from benchmark.schemas.ablation import FaultConfig
from benchmark.schemas.configs import TargetAppConfig
from benchmark.schemas.traces import Span, Trace

# AblationSpec's ShimKind vocabulary -> the shim-kind keys a TargetAppConfig
# declares. The right-hand side is a *config* key, not an app-internal name.
SHIM_TO_CONFIG_KIND: dict[str, str] = {
    "retriever": "retriever",
    "tool": "tool",
    "llm_proxy": "llm",
}

# Which span kind a fault of each shim must show up in.
SHIM_TO_SPAN_TYPE: dict[str, str] = {
    "retriever": "retrieval",
    "tool": "tool",
    "llm_proxy": "llm",
}


class UndeclaredFault(Exception):
    """The requested shim is not part of the app's declared fault surface."""


class FaultNotActivated(Exception):
    """The armed fault left no visible trace in the span it must corrupt."""


def fault_configurable(cfg: TargetAppConfig, fault: FaultConfig) -> dict[str, dict]:
    """`FaultConfig` -> the `config.configurable` payload for one run.

    Values are ALWAYS mappings. `langchain_core.runnables.config` copies every
    str/int/float/bool `configurable` entry into LangSmith-inheritable run
    metadata, so a scalar would stamp the fault name onto every span of the
    run; the target app refuses scalars for exactly that reason
    (apps/target_app/README.md, "Trace leak surface").
    """
    kind = SHIM_TO_CONFIG_KIND.get(fault.shim)
    key = cfg.fault_configurable_keys.get(kind) if kind else None
    if key is None:
        raise UndeclaredFault(
            f"shim {fault.shim!r} maps to config kind {kind!r}, which the target app "
            f"does not declare; declared: {sorted(cfg.fault_configurable_keys)}"
        )
    value: dict = {"behavior": fault.behavior}
    if fault.params:
        value["params"] = dict(fault.params)
    return {key: value}


def structural_behavior_tokens(fault: FaultConfig) -> tuple[str, ...]:
    """Behaviour names worth adding to the leak scan.

    Only structural identifiers (`irrelevant_docs`, `truncate_output`) qualify.
    Plain behaviour words — "empty", "stale", "error", "timeout" — are ordinary
    English that support-corpus text uses legitimately ("Emptying the Trash"),
    and scanning for them would quarantine healthy traces.
    """
    behavior = (fault.behavior or "").strip().lower()
    return (behavior,) if "_" in behavior else ()


def candidate_spans(trace: Trace, fault: FaultConfig) -> list[Span]:
    """Spans a fault of this shim kind could have corrupted, in time order.

    `fault.target` narrows to a named span when at least one span carries that
    name; otherwise every span of the shim's type is a candidate (the target
    may name an endpoint rather than a span).
    """
    span_type = SHIM_TO_SPAN_TYPE.get(fault.shim)
    typed = [s for turn in trace.turns for s in turn.spans if s.span_type == span_type]
    named = [s for s in typed if fault.target and s.name == fault.target]
    chosen = named or typed
    return sorted(chosen, key=lambda s: (s.end_time, s.span_id))


def activation_evidence(
    trace: Trace,
    fault: FaultConfig,
    *,
    baseline: Trace | None = None,
    weak_validation: bool = False,
) -> str:
    """Evidence that `fault` activated in `trace`, or raise `FaultNotActivated`.

    The returned string is what Phase 5 records as the `AblationRecord`'s
    activation evidence (`before_after` gets `("", <evidence>)`).

    **Phase 5's step-3 validation must pass `baseline`** — the unarmed trace
    for the same input. Only then is activation a byte-diff of the span the
    fault must corrupt. Without a baseline the check degrades to "the
    dependency ran at all" (plus a `delay_seconds` duration check), which a
    completely disarmed run also passes, so the caller has to say
    `weak_validation=True` to accept that. Passing neither is an error rather
    than a silent downgrade.
    """
    if baseline is None and not weak_validation:
        raise ValueError(
            "activation_evidence needs either baseline=<the unarmed trace for this "
            "input> for a real byte-diff, or weak_validation=True to acknowledge the "
            "weak form (which only proves the dependency was exercised, something a "
            "disarmed run also does). Phase 5 step-3 validation must use baseline."
        )
    spans = candidate_spans(trace, fault)
    span_type = SHIM_TO_SPAN_TYPE.get(fault.shim)
    if not spans:
        raise FaultNotActivated(
            f"trace {trace.trace_id} has no {span_type!r} span for target "
            f"{fault.target!r} — the dependency was never exercised, so the "
            f"armed {fault.shim} fault cannot have activated"
        )
    span = spans[-1]

    delay = (fault.params or {}).get("delay_seconds")
    if isinstance(delay, (int, float)):
        observed_ms = span.attributes.get("duration_ms")
        if observed_ms is None or observed_ms < delay * 1000:
            raise FaultNotActivated(
                f"span {span.name!r} ran for {observed_ms}ms but the armed fault "
                f"declares a delay of {delay}s — the delay never took effect"
            )

    evidence = json.dumps(span.outputs, sort_keys=True, default=str)
    if baseline is not None:
        baseline_spans = candidate_spans(baseline, fault)
        if baseline_spans:
            before = json.dumps(baseline_spans[-1].outputs, sort_keys=True, default=str)
            if before == evidence:
                raise FaultNotActivated(
                    f"the {span_type} span of trace {trace.trace_id} is byte-identical "
                    f"to the unarmed baseline {baseline.trace_id} — nothing was corrupted"
                )
    return evidence
