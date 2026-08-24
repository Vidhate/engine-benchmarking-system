"""The Engine's own view of its input and output.

These models are deliberately *local copies* of the shapes the benchmark
publishes, not imports of them: `apps/engine` is a black box that happens to
speak the same JSON, exactly like `apps/target_app`. Nothing here imports
`benchmark.*`.

Two consequences worth stating out loud:

* The trace models declare only the fields the Engine is allowed to look at.
  Everything else in the file — including any ablation bookkeeping a producer
  forgot to strip — is dropped at parse time by `extra="ignore"` and is
  therefore unreachable from Engine code (see `tests/test_no_leak.py`).
* The issueboard models mirror `benchmark.schemas.issues`, so the run output
  validates against `Issueboard` without translation on the benchmark side.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["low", "medium", "high"]
SpanType = Literal["llm", "tool", "retrieval", "chain", "agent"]

OTHER_CATEGORY_ID = "other"


class _Strict(BaseModel):
    """Base for input models: unknown keys are discarded, never retained."""

    model_config = ConfigDict(extra="ignore")


# --------------------------------------------------------------------------
# Input: traces
# --------------------------------------------------------------------------


class Span(_Strict):
    span_id: str
    parent_span_id: str | None = None
    name: str
    span_type: SpanType = "chain"
    start_time: str | None = None
    end_time: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)


class Turn(_Strict):
    turn_index: int
    user_message: str = ""
    final_response: str = ""
    spans: list[Span] = Field(default_factory=list)


class Trace(_Strict):
    trace_id: str
    input_id: str = ""
    mode: str = "single_turn"
    turns: list[Turn] = Field(default_factory=list)
    status: str = "ok"
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Input: category vocabulary (names + descriptions only)
# --------------------------------------------------------------------------


class Category(_Strict):
    category_id: str
    name: str = ""
    description: str = ""


# --------------------------------------------------------------------------
# Output: issueboard
# --------------------------------------------------------------------------


class Issue(BaseModel):
    error_id: str
    title: str
    description: str
    category_id: str
    severity: Severity


class IssueOccurrence(BaseModel):
    error_id: str
    trace_id: str
    turn_index: int | None = None
    span_id: str | None = None
    evidence: str | None = None


class Issueboard(BaseModel):
    board_id: str = ""
    source: str = "engine_predicted"
    issues: list[Issue] = Field(default_factory=list)
    occurrences: list[IssueOccurrence] = Field(default_factory=list)


class SeedIssueboard(_Strict):
    """The seed board as it arrives on the run input.

    Parsed leniently (`extra="ignore"`) so an `injection_mode` left on a seed
    issue by an upstream producer is dropped rather than carried through into
    the Engine's output.
    """

    board_id: str = ""
    source: str = "seed"
    issues: list[Issue] = Field(default_factory=list)
    occurrences: list[IssueOccurrence] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Intermediate: per-trace raw findings
# --------------------------------------------------------------------------


# NOTE ON DEFAULTS (applies to every LLM-facing schema below).
#
# A field the model may omit, that then arrives at scoring carrying a default,
# is a prediction the Engine never made being graded as though it had. Default
# `category_id="other"` and `severity="medium"` would do exactly that: silence
# becomes a scoreable answer, and the benchmark measures the default's luck
# against the ground-truth distribution instead of the model's judgement.
#
# So on these schemas: anything SCORED is required — no default, no fallback —
# and the model is forced to predict it. Anything the orchestrator already
# knows (trace_id) is kept off the model's schema entirely, so a construction
# bug fails loudly instead of quietly emitting "". Only unscored localization
# hints stay optional.


class FindingExtraction(BaseModel):
    """LLM-facing schema for one finding. Pydantic marks the fields without
    defaults `required` in the generated JSON schema, and OpenAI structured
    output enforces that, so the model cannot decline to answer.

    `trace_id` is deliberately ABSENT: the orchestrator knows which trace it
    asked about, and stamps it (`analysis.analyze_trace`). Asking the model for
    it invites a mis-attributed finding, which would corrupt the
    {trace_id, error_id} matrix scoring consumes.
    """

    title: str
    description: str
    category_id: str  # required — scored
    severity: Severity  # required — scored
    # Unscored localization hints: helpful for auditing an occurrence, never
    # graded, so absence here costs nothing and forcing them would only push
    # the model to invent a span id it could not find.
    evidence: str = ""
    span_id: str | None = None
    turn_index: int | None = None


class FindingExtractionList(BaseModel):
    """Structured-output envelope for one trace's findings."""

    findings: list[FindingExtraction] = Field(default_factory=list)


class RawFinding(BaseModel):
    """One unconsolidated observation, after the orchestrator stamps its trace.

    `trace_id` is required here precisely because it is not on the extraction
    schema — if the stamping step is ever skipped, construction raises rather
    than producing a finding attributed to "".
    """

    trace_id: str
    title: str
    description: str
    category_id: str
    severity: Severity
    evidence: str = ""
    span_id: str | None = None
    turn_index: int | None = None


# --------------------------------------------------------------------------
# Intermediate: the consolidation plan
# --------------------------------------------------------------------------


class Cluster(BaseModel):
    """One canonical failure mode, plus the raw findings that belong to it."""

    title: str
    description: str
    # Required for the same reason as on FindingExtraction, and more urgently:
    # the canonical Issue the benchmark scores takes its category and severity
    # from HERE, so a default on this schema is a defaulted prediction sitting
    # directly on the board.
    category_id: str
    severity: Severity
    # Index into the flat raw-findings list handed to the consolidation pass.
    # Structural bookkeeping, not a prediction — an empty grouping is a real,
    # meaningful answer, so the default stays.
    finding_indices: list[int] = Field(default_factory=list)
    # Set when this cluster is the same failure mode as an issue already on the
    # seed board; the cluster then attaches occurrences instead of creating a
    # new issue.
    matches_seed_error_id: str | None = None


class ConsolidationPlan(BaseModel):
    clusters: list[Cluster] = Field(default_factory=list)
