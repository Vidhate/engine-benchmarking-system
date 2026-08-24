"""The Engine's dispatchable surface must equal its declared surface.

Phase 2 dropped `deepagents` after finding its scaffold left filesystem and
shell tools registered on the ToolNode even when they were hidden from the
model. This Engine reads attacker-influenced text out of traces for a living,
so the same finding matters more here: any tool it can be talked into calling
is a tool it must not have.
"""

from __future__ import annotations

import json

from engine.analysis import _dispatch
from engine.tools import TRACE_TOOL_NAMES, build_trace_tools

FORBIDDEN = (
    "shell", "bash", "exec", "python", "run_command",
    "write_file", "read_file", "edit_file", "ls", "glob", "grep",
    "task", "subagent", "handoff", "todo", "web_search", "fetch",
)


def test_registry_is_exactly_the_four_trace_tools(index):
    assert {t.name for t in build_trace_tools(index)} == TRACE_TOOL_NAMES
    assert TRACE_TOOL_NAMES == {"get_trace", "list_spans", "read_span", "search_text"}


def test_no_filesystem_shell_or_delegation_tool_is_registered(index):
    names = {t.name.lower() for t in build_trace_tools(index)}
    assert not any(bad in name for name in names for bad in FORBIDDEN)


def test_the_model_is_bound_to_exactly_the_registry(index, categories):
    """What the graph binds is what the graph dispatches — no hidden extras."""
    from engine.analysis import analyze_trace
    from engine.models import RawFindingList
    from tests.fakes import FakeChatModel

    model = FakeChatModel(responses=[], structured=[RawFindingList()])
    analyze_trace(model, index, "trace-clean-pricing", [], categories)
    assert {t.name for t in model.bound_tools} == TRACE_TOOL_NAMES


def test_every_tool_reaches_its_index_operation(index):
    tools = {t.name: t for t in build_trace_tools(index)}
    assert json.loads(tools["get_trace"].invoke({"trace_id": "trace-clean-pricing"}))["mode"]
    assert json.loads(tools["list_spans"].invoke({"trace_id": "trace-clean-pricing"}))["spans"]
    span = tools["read_span"].invoke({"trace_id": "trace-clean-pricing", "span_id": "s-p-2"})
    assert json.loads(span)["span_type"] == "retrieval"
    hits = tools["search_text"].invoke({"query": "TicketServiceError"})
    assert json.loads(hits)["location_count"] >= 1


def test_tools_are_read_only(index, traces_file):
    """No tool takes a write argument, and the file is untouched after a run."""
    before = traces_file.read_bytes()
    for tool in build_trace_tools(index):
        fields = set(tool.args_schema.model_json_schema()["properties"])
        assert fields <= {"trace_id", "span_id", "turn_index", "query"}
    assert traces_file.read_bytes() == before


def test_a_hallucinated_tool_name_is_refused_not_reached(index):
    """A fabricated tool call comes back as an error message, not an execution."""
    registry = {t.name: t for t in build_trace_tools(index)}
    result = json.loads(_dispatch(registry, {"name": "write_file", "args": {"path": "/etc/x"}}))
    assert "no such tool" in result["error"]
    assert set(result["available"]) == TRACE_TOOL_NAMES


def test_a_bad_argument_is_reported_to_the_model_not_raised(index):
    registry = {t.name: t for t in build_trace_tools(index)}
    result = json.loads(_dispatch(registry, {"name": "read_span", "args": {"nope": 1}}))
    assert "error" in result
