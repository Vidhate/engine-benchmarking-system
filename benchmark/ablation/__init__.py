"""Stage III — the ablation engine (docs/architecture/04-ablation-engine.md).

Manufactures ground truth by construction: every injected error is fully known
(what, where, how), so scoring against it is exact.

    [N, M, T]  ->  [N, M, T*], [N, E_K]

Public surface:

* `run_ablation(traces, inputs, categories, cfg, harness, store, export_path)`
  — the four-step loop end to end, returning an `AblationResult`.
* `AblationEngine` — the same loop with the LLM agent injectable (unit tests
  and the live smoke script use this; Phase 7 uses `run_ablation`).

Everything in this package lives on the **ground-truth side** of the leak
boundary. The only artifact that crosses to Engine is the file written by
`export.write_engine_export`, which is allowlist-rebuilt and audited before it
touches disk.
"""

from benchmark.ablation.agent import (
    AblationAgent,
    CorpusDigest,
    Corruption,
    OpenAIAblationAgent,
    ProposedError,
    ScriptedAblationAgent,
)
from benchmark.ablation.apply import ApplyOutcome, apply_ablations
from benchmark.ablation.engine import (
    AblationEngine,
    AblationResult,
    default_agent,
    run_ablation,
)
from benchmark.ablation.export import (
    ExportLeak,
    audit_export,
    build_export,
    strip_trace,
    write_engine_export,
)
from benchmark.ablation.filters import UnknownFilterField, eligible, matches, resolve
from benchmark.ablation.inject import (
    CorruptionLost,
    DeadThreadRefs,
    InjectionError,
    SelfCorrected,
    apply_dependency_fault,
    apply_replay_edit,
    assert_threads_alive,
)
from benchmark.ablation.plan import plan_ablation
from benchmark.ablation.propose import (
    DEFAULT_MIN_PER_MODE,
    allowed_modes,
    build_digest,
    propose_errors,
)
from benchmark.ablation.split import make_split, stratum_of
from benchmark.ablation.validate import ValidationFailure, ValidationOutcome, validate_specs

__all__ = [
    "AblationAgent",
    "AblationEngine",
    "AblationResult",
    "ApplyOutcome",
    "CorpusDigest",
    "Corruption",
    "DEFAULT_MIN_PER_MODE",
    "CorruptionLost",
    "DeadThreadRefs",
    "ExportLeak",
    "InjectionError",
    "OpenAIAblationAgent",
    "ProposedError",
    "ScriptedAblationAgent",
    "SelfCorrected",
    "UnknownFilterField",
    "ValidationFailure",
    "ValidationOutcome",
    "allowed_modes",
    "apply_ablations",
    "apply_dependency_fault",
    "apply_replay_edit",
    "assert_threads_alive",
    "audit_export",
    "build_digest",
    "build_export",
    "default_agent",
    "eligible",
    "make_split",
    "matches",
    "plan_ablation",
    "propose_errors",
    "resolve",
    "run_ablation",
    "strip_trace",
    "stratum_of",
    "validate_specs",
    "write_engine_export",
]
