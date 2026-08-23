"""Trace schema — the [N, M, T] contract (docs/architecture/01-data-schemas.md §2).

This schema, not LangSmith's, is the system contract: the Phase 4 collector
normalizes LangSmith run trees into these models, and everything downstream
(ablation, Engine input, scoring) consumes only these.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

SpanType = Literal["llm", "tool", "retrieval", "chain", "agent"]
TraceMode = Literal["single_turn", "multi_turn"]
TraceStatus = Literal["ok", "app_error"]


class Span(BaseModel):
    span_id: str
    parent_span_id: str | None = None
    name: str  # "ChatOpenAI", "retriever.search", tool name…
    span_type: SpanType
    start_time: datetime
    end_time: datetime
    inputs: dict[str, Any] = Field(default_factory=dict)  # llm: messages; tool: args
    outputs: dict[str, Any] = Field(default_factory=dict)  # llm: completion; tool: result
    attributes: dict[str, Any] = Field(default_factory=dict)  # model, tokens, temperature…


class Turn(BaseModel):
    turn_index: int
    user_message: str
    final_response: str
    spans: list[Span] = Field(default_factory=list)  # tree via parent_span_id


class Trace(BaseModel):
    trace_id: str
    input_id: str  # FK -> InputSpec
    mode: TraceMode
    turns: list[Turn] = Field(default_factory=list)
    status: TraceStatus = "ok"
    metadata: dict[str, Any] = Field(default_factory=dict)  # app version, model, timestamps
    # INTERNAL ONLY — stripped before Engine sees traces (leak-proofing).
    ablation_ids: list[str] = Field(default_factory=list)


class TraceDataset(BaseModel):
    dataset_id: str = ""  # content hash; stamped via schemas.io.stamp_dataset_id
    parent_dataset_id: str | None = None  # ablated sets point at their source
    traces: list[Trace] = Field(default_factory=list)


class OutputRecord(BaseModel):
    """The [N, M] outputs view — one record per input, M final responses."""

    input_id: str
    trace_id: str
    responses: list[str] = Field(default_factory=list)


class OutputDataset(BaseModel):
    dataset_id: str = ""
    parent_dataset_id: str | None = None
    outputs: list[OutputRecord] = Field(default_factory=list)
