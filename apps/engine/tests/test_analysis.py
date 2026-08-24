"""The two LLM passes, driven by a scripted fake model — no network."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from engine.analysis import analyze_trace, consolidate
from engine.models import (
    Cluster,
    ConsolidationPlan,
    RawFinding,
    RawFindingList,
    SeedIssueboard,
)
from tests.fakes import FakeChatModel, tool_call

TICKET_FINDING = RawFinding(
    title="Tool error reported to the user as success",
    description="create_ticket returned an error; the answer claims a ticket was created.",
    category_id="tool_misuse",
    severity="high",
    evidence="TicketServiceError: upstream ticketing API returned 503",
    span_id="s-t-2",
    turn_index=0,
)


def model(responses=None, structured=None) -> FakeChatModel:
    return FakeChatModel(responses=list(responses or []), structured=list(structured or []))


def fed_back_tool_messages(fake: FakeChatModel) -> list[ToolMessage]:
    """Tool results the loop actually put back in front of the model.

    `calls` snapshots the message list at each invoke, so the richest snapshot
    is the last tool-loop turn.
    """
    per_call = [[m for m in call if isinstance(m, ToolMessage)] for call in fake.calls]
    return max(per_call, key=len, default=[])


def emit_prompt(fake: FakeChatModel) -> str:
    return next(m for m in fake.calls[-1] if isinstance(m, HumanMessage)).content


# -- per-trace analysis ----------------------------------------------------


def test_a_clean_trace_yields_no_findings(index, categories):
    fake = model(responses=[AIMessage(content="Nothing wrong here.")],
                 structured=[RawFindingList()])
    assert analyze_trace(fake, index, "trace-clean-pricing", [], categories) == []


def test_tool_calls_are_executed_and_fed_back(index, categories):
    fake = model(
        responses=[
            tool_call("get_trace", {"trace_id": "trace-planted-ticket"}, "c1"),
            tool_call("read_span", {"trace_id": "trace-planted-ticket", "span_id": "s-t-2"}, "c2"),
            AIMessage(content="The ticket tool failed but the answer says it succeeded."),
        ],
        structured=[RawFindingList(findings=[TICKET_FINDING])],
    )
    findings = analyze_trace(fake, index, "trace-planted-ticket", [], categories)

    tool_messages = fed_back_tool_messages(fake)
    assert [m.tool_call_id for m in tool_messages] == ["c1", "c2"]
    assert "TicketServiceError" in tool_messages[1].content
    assert "TOOL read_span" in emit_prompt(fake)
    assert len(findings) == 1


def test_the_analysis_trace_id_overrides_whatever_the_model_reports(index, categories):
    """A model-supplied trace_id would let one trace's finding be filed against
    another trace's occurrences, silently corrupting the {trace_id, error_id} matrix."""
    stray = TICKET_FINDING.model_copy(update={"trace_id": "some-other-trace"})
    fake = model(responses=[AIMessage(content="done")],
                 structured=[RawFindingList(findings=[stray])])
    findings = analyze_trace(fake, index, "trace-planted-ticket", [], categories)
    assert findings[0].trace_id == "trace-planted-ticket"


def test_running_titles_and_categories_reach_the_analysis_prompt(index, categories):
    fake = model(responses=[AIMessage(content="ok")], structured=[RawFindingList()])
    analyze_trace(fake, index, "trace-clean-pricing", ["Truncated response"], categories)
    system = next(m for m in fake.calls[0] if isinstance(m, SystemMessage)).content
    assert "Truncated response" in system
    assert "tool_misuse" in system and "other" in system


def test_the_emit_step_sees_the_tool_log_and_has_no_tools(index, categories):
    """The reporting step must not be able to keep reading — its input is the
    log of what was actually inspected, so findings stay evidenced."""
    fake = model(
        responses=[tool_call("search_text", {"query": "14 days"}, "c1"),
                   AIMessage(content="The answer contradicts the retrieved policy.")],
        structured=[RawFindingList()],
    )
    analyze_trace(fake, index, "trace-planted-refund", [], categories)
    prompt = emit_prompt(fake)
    assert "TOOL search_text" in prompt
    assert "ANALYST NOTES" in prompt
    assert "14 days" in prompt
    # The emit call is tool-free: nothing new can be read at reporting time.
    assert not any(isinstance(m, ToolMessage) for m in fake.calls[-1])


def test_the_tool_budget_is_enforced(index, categories):
    fake = model(
        responses=[tool_call("get_trace", {"trace_id": "trace-clean-pricing"}, f"c{i}")
                   for i in range(10)],
        structured=[RawFindingList()],
    )
    analyze_trace(fake, index, "trace-clean-pricing", [], categories, max_tool_calls=3)
    prompt = emit_prompt(fake)
    assert prompt.count("TOOL get_trace") == 3
    assert "budget exhausted" in prompt
    assert len(fake.responses) == 7  # the loop stopped, it did not run out of script


def test_a_hallucinated_tool_call_does_not_stop_the_pass(index, categories):
    fake = model(
        responses=[tool_call("write_file", {"path": "/tmp/x"}, "c1"), AIMessage(content="ok")],
        structured=[RawFindingList()],
    )
    analyze_trace(fake, index, "trace-clean-pricing", [], categories)
    assert "no such tool" in fed_back_tool_messages(fake)[0].content


def test_a_dict_shaped_structured_result_is_accepted(index, categories):
    fake = model(responses=[AIMessage(content="ok")],
                 structured=[{"findings": [TICKET_FINDING.model_dump()]}])
    assert len(analyze_trace(fake, index, "trace-planted-ticket", [], categories)) == 1


def test_an_unusable_structured_result_yields_no_findings(index, categories):
    fake = model(responses=[AIMessage(content="ok")], structured=[None])
    assert analyze_trace(fake, index, "trace-planted-ticket", [], categories) == []


# -- consolidation ---------------------------------------------------------


def test_consolidation_skips_the_model_when_there_is_nothing_to_cluster(
    seed_board_payload, categories
):
    fake = model()  # no scripted responses: calling it would raise
    board = consolidate(fake, [], SeedIssueboard.model_validate(seed_board_payload), categories)
    assert fake.calls == []
    assert [i.error_id for i in board.issues] == [
        "seed-tool-failure-hidden",
        "seed-answers-without-retrieval",
    ]


def test_consolidation_prompt_carries_the_seed_board_and_the_findings(
    seed_board_payload, categories
):
    findings = [TICKET_FINDING.model_copy(update={"trace_id": "trace-planted-ticket"})]
    fake = model(structured=[ConsolidationPlan(
        clusters=[Cluster(title="t", description="d", category_id="tool_misuse", severity="high",
                          finding_indices=[0], matches_seed_error_id="seed-tool-failure-hidden")]
    )])
    consolidate(fake, findings, SeedIssueboard.model_validate(seed_board_payload), categories)
    system = next(m for m in fake.calls[0] if isinstance(m, SystemMessage)).content
    task = next(m for m in fake.calls[0] if isinstance(m, HumanMessage)).content
    assert "seed-tool-failure-hidden" in system
    assert "[0] trace=trace-planted-ticket" in task


def test_an_empty_plan_falls_back_to_deterministic_clustering(categories):
    findings = [
        TICKET_FINDING.model_copy(update={"trace_id": "t1"}),
        TICKET_FINDING.model_copy(update={"trace_id": "t2"}),
    ]
    fake = model(structured=[ConsolidationPlan()])
    board = consolidate(fake, findings, None, categories)
    assert len(board.issues) == 1
    assert {o.trace_id for o in board.occurrences} == {"t1", "t2"}


def test_a_dict_shaped_plan_is_accepted(categories):
    findings = [TICKET_FINDING.model_copy(update={"trace_id": "t1"})]
    fake = model(structured=[{"clusters": [
        {"title": "Tool error hidden", "description": "d", "category_id": "tool_misuse",
         "severity": "high", "finding_indices": [0]}
    ]}])
    board = consolidate(fake, findings, None, categories)
    assert board.issues[0].title == "Tool error hidden"


def test_a_junk_plan_falls_back_rather_than_raising(categories):
    findings = [TICKET_FINDING.model_copy(update={"trace_id": "t1"})]
    fake = model(structured=["not a plan"])
    board = consolidate(fake, findings, None, categories)
    assert len(board.issues) == 1


def test_the_fake_model_refuses_to_invent_structured_output(index, categories):
    """Guards the guard: an unscripted call must fail loudly, not return {}."""
    with pytest.raises(AssertionError, match="no scripted structured output"):
        analyze_trace(model(responses=[AIMessage(content="ok")]), index,
                      "trace-clean-pricing", [], categories)
