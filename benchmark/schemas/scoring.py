"""Scoring & benchmark report — Stage V output (docs/architecture/01-data-schemas.md §5)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from benchmark.schemas.configs import EngineConfig

OccurrenceMatchMethod = Literal["exact_key", "text_fallback"]


class OccurrenceMatch(BaseModel):
    """Layer 1: one predicted occurrence resolved via the exact (trace, category) key."""

    trace_id: str
    predicted_error_id: str
    resolved_error_id: str | None = None  # None -> FP / E_h candidate pool
    method: OccurrenceMatchMethod = "exact_key"  # fallback = wrong-category case


class ErrorMatch(BaseModel):
    """Layer 2: predicted issue -> known error, by argmax occurrence-set overlap."""

    predicted_error_id: str
    matched_error_id: str | None = None  # None -> no resolved occurrences (FP / E_h)
    overlap: int = 0  # winning |occurrences ∩ traces(K_i)| vote count
    tie_broken_by_text: bool = False


class CategoryScore(BaseModel):
    """P/R/F1/kappa over a binary trace matrix — used per category AND per known error."""

    category_id: str  # category id (scorer 1) or known error id (scorer 2)
    precision: float
    recall: float
    f1: float
    cohens_kappa: float
    support: int


class BenchmarkReport(BaseModel):
    report_id: str = ""
    engine_config: EngineConfig  # which model/agent produced E_P
    base_rates: dict[str, Any] = Field(default_factory=dict)  # control frac, injection counts
    matcher_fallback_rate: float = 0.0  # how often text fallback fired (reliability)
    occurrence_matches: list[OccurrenceMatch] = Field(default_factory=list)
    matches: list[ErrorMatch] = Field(default_factory=list)
    category_scores: list[CategoryScore] = Field(default_factory=list)  # scorer 1
    per_error_scores: list[CategoryScore] = Field(default_factory=list)  # scorer 2
    severity_loss: float = 0.0  # scorer 3 (asymmetric)
    description_scores: dict[str, float] = Field(default_factory=dict)  # scorer 4
    eh_candidates: list[str] = Field(default_factory=list)  # unmatched E_P ids for review
    headline: dict[str, float] = Field(default_factory=dict)  # macro-P/R/F1 etc.
