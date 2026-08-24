"""The served graph: the sequential loop, the model swap, and the run output."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage

from engine import graph as graph_module
from engine.llm import DEFAULT_MODEL, resolve_model_name
from engine.models import Cluster, ConsolidationPlan, RawFinding, RawFindingList
from tests.fakes import FakeChatModel

ALL_TRACE_IDS = [
    "trace-clean-pricing",
    "trace-clean-platforms",
    "trace-clean-export",
    "trace-planted-refund",
    "trace-planted-ticket",
    "trace-planted-truncated",
]

TICKET_FINDING = RawFinding(
    title="Tool error reported to the user as success",
    description="create_ticket errored; the answer claims a ticket was created.",
    category_id="tool_misuse",
    severity="high",
    evidence="TicketServiceError: 503",
    span_id="s-t-2",
)


@pytest.fixture
def scripted(monkeypatch):
    """Install one fake model for the whole run and hand it back to the test."""
    fake = FakeChatModel(responses=[], structured=[])
    monkeypatch.setattr(graph_module, "build_model", lambda name: fake)
    return fake


def script_for(fake, per_trace_findings: dict[str, list[RawFinding]], plan=None, n=6):
    """One AIMessage + one RawFindingList per trace, then the plan."""
    fake.responses.extend(AIMessage(content="reviewed") for _ in range(n))
    for trace_id in ALL_TRACE_IDS[:n]:
        fake.structured.append(RawFindingList(findings=per_trace_findings.get(trace_id, [])))
    if plan is not None:
        fake.structured.append(plan)


def run(traces_file, seed=None, categories=None, config=None):
    return graph_module.graph.invoke(
        {
            "trace_file": str(traces_file),
            "seed_issueboard": seed or {},
            "categories": [c.model_dump() for c in (categories or [])],
        },
        config=config or {},
    )


def test_the_run_output_is_an_issueboard_shaped_object(traces_file, scripted, categories):
    script_for(scripted, {}, plan=None)
    result = run(traces_file, categories=categories)
    assert set(result) == {"board_id", "source", "issues", "occurrences"}
    assert result["source"] == "engine_predicted"


def test_every_trace_is_analysed_once_in_order(traces_file, scripted, categories):
    script_for(scripted, {})
    run(traces_file, categories=categories)
    # Six analysis turns + six emit calls; consolidation is skipped (no findings).
    analysed = [
        call[1].content for call in scripted.calls if "Analyse trace" in str(call[1].content)
    ]
    assert [t.split("`")[1] for t in analysed] == ALL_TRACE_IDS


def test_findings_flow_through_consolidation_into_the_board(
    traces_file, scripted, categories, seed_board_payload
):
    script_for(
        scripted,
        {"trace-planted-ticket": [TICKET_FINDING]},
        plan=ConsolidationPlan(
            clusters=[
                Cluster(title="Tool error hidden", description="d", category_id="tool_misuse",
                        severity="high", finding_indices=[0],
                        matches_seed_error_id="seed-tool-failure-hidden")
            ]
        ),
    )
    result = run(traces_file, seed=seed_board_payload, categories=categories)

    assert [i["error_id"] for i in result["issues"]] == [
        "seed-tool-failure-hidden",
        "seed-answers-without-retrieval",
    ]
    assert [(o["error_id"], o["trace_id"]) for o in result["occurrences"]] == [
        ("seed-tool-failure-hidden", "trace-planted-ticket")
    ]


def test_running_titles_accumulate_across_traces(traces_file, scripted, categories):
    script_for(
        scripted,
        {
            "trace-clean-pricing": [TICKET_FINDING.model_copy(update={"title": "First mode"})],
            "trace-planted-ticket": [TICKET_FINDING.model_copy(update={"title": "Second mode"})],
        },
        plan=ConsolidationPlan(),
    )
    run(traces_file, categories=categories)
    analysis_prompts = [
        str(call[0].content)
        for call in scripted.calls
        if "automated error-analysis system" in str(call[0].content)
    ]
    assert len(analysis_prompts) == 6
    # The first trace sees an empty running list; later traces see what came before.
    assert "(none yet" in analysis_prompts[0]
    assert "First mode" in analysis_prompts[1]
    assert "First mode" in analysis_prompts[-1] and "Second mode" in analysis_prompts[-1]


def test_a_seed_board_survives_a_run_with_no_findings(traces_file, scripted, seed_board_payload):
    script_for(scripted, {})
    result = run(traces_file, seed=seed_board_payload)
    assert len(result["issues"]) == 2
    assert result["occurrences"] == []


def test_an_empty_seed_board_is_fine(traces_file, scripted, categories):
    script_for(scripted, {})
    assert run(traces_file, categories=categories)["issues"] == []


def test_one_failing_trace_does_not_abandon_the_rest(traces_file, monkeypatch, categories):
    """A flaky trace costs its own findings, not the whole run."""
    calls = {"n": 0}

    def flaky(model, index, trace_id, running_titles, categories, max_tool_calls):
        calls["n"] += 1
        if trace_id == "trace-planted-refund":
            raise RuntimeError("boom")
        return [TICKET_FINDING.model_copy(update={"trace_id": trace_id})]

    monkeypatch.setattr(graph_module, "build_model", lambda name: FakeChatModel())
    monkeypatch.setattr(graph_module, "analyze_trace", flaky)
    monkeypatch.setattr(
        graph_module, "consolidate", lambda m, f, s, c: _board_from(f)
    )
    result = run(traces_file, categories=categories)
    assert calls["n"] == 6
    assert len(result["occurrences"]) == 5


def test_a_failure_on_every_trace_raises_instead_of_reporting_no_errors(
    traces_file, monkeypatch, categories
):
    """An expired key must not be indistinguishable from a clean corpus."""

    def always_fails(*args, **kwargs):
        raise RuntimeError("401 Unauthorized")

    monkeypatch.setattr(graph_module, "build_model", lambda name: FakeChatModel())
    monkeypatch.setattr(graph_module, "analyze_trace", always_fails)
    with pytest.raises(Exception, match="401 Unauthorized"):
        run(traces_file, categories=categories)


def test_a_missing_trace_file_fails_loudly(tmp_path, scripted):
    with pytest.raises(Exception, match="trace_file not found"):
        run(tmp_path / "nope.json")


# -- the model swap --------------------------------------------------------


def test_the_model_comes_from_the_run_configurable(traces_file, monkeypatch, categories):
    seen: list[str] = []
    fake = FakeChatModel(responses=[], structured=[])
    script_for(fake, {})
    monkeypatch.setattr(graph_module, "build_model", lambda name: (seen.append(name), fake)[1])

    run(traces_file, categories=categories, config={"configurable": {"model": "gpt-5.1"}})
    assert set(seen) == {"gpt-5.1"}


def test_resolve_model_name_precedence(monkeypatch):
    monkeypatch.delenv("ENGINE_MODEL", raising=False)
    assert resolve_model_name(None) == DEFAULT_MODEL
    assert resolve_model_name({"configurable": {}}) == DEFAULT_MODEL
    assert resolve_model_name({"configurable": {"model": "  gpt-5.1  "}}) == "gpt-5.1"
    assert resolve_model_name({"configurable": {"model": ""}}) == DEFAULT_MODEL
    assert resolve_model_name({"configurable": {"model": 17}}) == DEFAULT_MODEL
    monkeypatch.setenv("ENGINE_MODEL", "from-env")
    assert resolve_model_name(None) == "from-env"
    # The run config still wins over the environment default.
    assert resolve_model_name({"configurable": {"model": "gpt-5.1"}}) == "gpt-5.1"


# -- graph shape -----------------------------------------------------------


def test_the_graph_has_only_the_three_orchestration_nodes():
    nodes = set(graph_module.graph.get_graph().nodes) - {"__start__", "__end__"}
    assert nodes == {"load", "analyze", "consolidate"}


def test_the_recursion_limit_allows_a_large_corpus():
    assert graph_module.RECURSION_LIMIT >= 1000


def _board_from(findings):
    from engine.consolidate import assemble_board, fallback_plan

    return assemble_board(fallback_plan(findings), findings, None, [])


def test_input_schema_is_the_declared_surface_only():
    schema = graph_module.EngineInput.__annotations__
    assert set(schema) == {"trace_file", "seed_issueboard", "categories"}
    assert "ablation" not in json.dumps(list(schema))
