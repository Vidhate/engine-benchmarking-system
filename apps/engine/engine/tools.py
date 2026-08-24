"""The Engine's tool surface: four read-only trace-inspection tools.

`build_trace_tools(index)` is the ONLY place tools are constructed, and it
returns exactly `TRACE_TOOL_NAMES` — nothing else is dispatchable. That
property is asserted in `tests/test_tool_registry.py`.

Why a hand-built registry rather than an agent scaffold: Phase 2 dropped
`deepagents` after finding its scaffold kept filesystem and shell tools
registered on the ToolNode even when they were hidden from the model. The
same reasoning applies here with more force — this agent's whole job is to
read attacker-influenced text out of traces, so a tool it can be talked into
calling is a tool it must not have.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from engine.traces import TraceIndex

TRACE_TOOL_NAMES: frozenset[str] = frozenset(
    {"get_trace", "list_spans", "read_span", "search_text"}
)


def build_trace_tools(index: TraceIndex) -> list[BaseTool]:
    """Bind the four inspection operations to one loaded trace file."""

    @tool
    def get_trace(trace_id: str) -> str:
        """Overview of one trace: status, metadata, and every turn's user message,
        final response, and span table (span ids/names/types, no payloads).

        Args:
            trace_id: id of the trace to fetch.
        """
        return index.get_trace(trace_id)

    @tool
    def list_spans(trace_id: str, turn_index: int | None = None) -> str:
        """Span table for a trace: span_id, parent, name, type, whether the span
        errored, and a short preview of its output. Use it to pick spans to read.

        Args:
            trace_id: id of the trace.
            turn_index: optional — restrict to a single turn.
        """
        return index.list_spans(trace_id, turn_index)

    @tool
    def read_span(trace_id: str, span_id: str) -> str:
        """Read one span in full: its inputs, outputs and attributes.

        Args:
            trace_id: id of the trace.
            span_id: id of the span, as reported by list_spans or get_trace.
        """
        return index.read_span(trace_id, span_id)

    @tool
    def search_text(query: str, trace_id: str | None = None) -> str:
        """Case-insensitive substring search over trace text (turn messages and
        span inputs/outputs/attributes). Returns locations plus snippets. Use it
        to check whether a claim in an answer is actually supported anywhere in
        the trace.

        Args:
            query: substring to search for.
            trace_id: optional — restrict the search to one trace.
        """
        return index.search_text(query, trace_id)

    tools: list[BaseTool] = [get_trace, list_spans, read_span, search_text]
    registered = {t.name for t in tools}
    if registered != TRACE_TOOL_NAMES:
        raise RuntimeError(f"tool registry drift: {sorted(registered)}")
    return tools
