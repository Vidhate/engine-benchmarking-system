"""Errors & Issueboard — Stage III + IV currency (docs/architecture/01-data-schemas.md §3)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high"]
BoardSource = Literal["seed", "ground_truth", "engine_predicted"]
InjectionMode = Literal["replay_edit", "dependency_fault"]

# Escape-hatch category id, always present in the vocabulary Engine receives, so
# out-of-taxonomy (E_h) discoveries are never shoehorned into a wrong category.
OTHER_CATEGORY_ID = "other"


class ErrorCategory(BaseModel):
    """C_E — the high-level taxonomy. Input to ablation, shared vocabulary for scoring."""

    category_id: str
    name: str  # e.g. "tool_misuse", "hallucination", "retrieval_failure", "other"
    description: str


class Issue(BaseModel):
    """One error definition. E_K entries are authored by ablation; E_P by Engine."""

    error_id: str
    title: str  # single-phrase title
    description: str  # verbose description
    category_id: str  # FK -> ErrorCategory
    severity: Severity
    # E_K entries ONLY — post-hoc analysis of which ablation kind benchmarks more
    # fairly. Stripped from anything Engine sees; never consumed by scorers.
    injection_mode: InjectionMode | None = None


class IssueOccurrence(BaseModel):
    error_id: str
    trace_id: str
    turn_index: int | None = None  # optional localization
    span_id: str | None = None
    evidence: str | None = None  # Engine's cited snippet / ablation's injected diff


class Issueboard(BaseModel):
    board_id: str = ""
    source: BoardSource
    issues: list[Issue] = Field(default_factory=list)
    occurrences: list[IssueOccurrence] = Field(default_factory=list)  # {trace_id, error_id}
