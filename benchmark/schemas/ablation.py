"""Ablation artifacts — Stage III internal (docs/architecture/01-data-schemas.md §4).

Everything in this module lives on the ground-truth side of the leak boundary:
none of it is ever shipped to Engine.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from benchmark.schemas.issues import InjectionMode

FilterOp = Literal["eq", "ne", "contains", "regex", "gt", "lt", "exists"]
TransformKind = Literal["replace", "regex_sub", "llm_rewrite", "delete", "inject"]
ShimKind = Literal["llm_proxy", "retriever", "tool"]


class FilterStep(BaseModel):
    field: str  # JSONPath-ish selector into Trace
    op: FilterOp
    value: Any = None


class TraceFilter(BaseModel):
    """Declarative selection over trace properties — a list of predicate steps."""

    steps: list[FilterStep] = Field(default_factory=list)


class AblationAction(BaseModel):
    """A str-in -> str-out mutation applied at a located field (replay_edit mode)."""

    target: str  # JSONPath-ish selector (which field to mutate)
    transform: TransformKind
    params: dict[str, Any] = Field(default_factory=dict)  # replacement, regex, instruction…


class FaultConfig(BaseModel):
    """dependency_fault mode: which external-dependency shim to arm, and how.

    `shim`/`target` must map onto keys the target app declares in
    TargetAppConfig.fault_configurable_keys — the benchmark never knows more
    about the app than that declaration.
    """

    shim: ShimKind
    target: str  # e.g. tool name, retriever endpoint
    behavior: str  # "irrelevant_docs", "timeout", "truncate_output"…
    params: dict[str, Any] = Field(default_factory=dict)


class AblationSpec(BaseModel):
    """Filter + strategy for injecting one error. Produced by step 2, validated by step 3."""

    error_id: str  # FK -> Issue (the E_K entry this injects)
    mode: InjectionMode
    filter: TraceFilter = Field(default_factory=TraceFilter)
    ablation_actions: list[AblationAction] = Field(default_factory=list)  # replay_edit
    fault_config: FaultConfig | None = None  # dependency_fault
    target_count: int = 5  # sub-sample size after filtering


class AblationRecord(BaseModel):
    """Ground-truth provenance: exactly what was done to which trace."""

    ablation_id: str
    error_id: str
    trace_id: str
    actions_applied: list[AblationAction] = Field(default_factory=list)
    # (original, mutated) per action for replay_edit; activation evidence for
    # dependency_fault lands in `before_after` as ("", <observed corrupted span text>).
    before_after: list[tuple[str, str]] = Field(default_factory=list)


class AblationSplit(BaseModel):
    """Input-level control/ablate assignment, made once before any ablation.

    Ground-truth side only — stripped from everything Engine sees.
    """

    seed: int
    control_fraction: float
    strata: list[str] = Field(default_factory=list)  # stratification keys (mode, kind, dim)
    control_input_ids: list[str] = Field(default_factory=list)  # never ablated / re-run
    ablate_input_ids: list[str] = Field(default_factory=list)  # sole filter population
