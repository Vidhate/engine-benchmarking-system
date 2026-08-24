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


# -- review finding 1: consolidation must survive a raised error ----------


class ExplodingStructured:
    """A model whose structured-output call raises, `times` times, then (if a
    scripted plan is left) succeeds."""

    def __init__(self, times: int, plans=None, error=None):
        self.remaining = times
        self.plans = list(plans or [])
        self.error = error or RuntimeError("429 rate limited")
        self.attempts = 0

    def invoke(self, messages, **kwargs):
        self.attempts += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise self.error
        if not self.plans:
            raise AssertionError("no scripted plan left")
        return self.plans.pop(0)


class ExplodingModel(FakeChatModel):
    """FakeChatModel whose `with_structured_output` returns a scripted exploder."""

    exploder: object = None

    def with_structured_output(self, schema, **kwargs):
        return self.exploder


def exploding(times: int, plans=None, error=None):
    model = ExplodingModel(responses=[], structured=[])
    model.exploder = ExplodingStructured(times, plans, error)
    return model


def test_a_raising_consolidation_still_yields_a_valid_board(categories, capsys):
    """The whole run has already been paid for by the time consolidation runs;
    a 429 there must not discard it."""
    findings = [
        TICKET_FINDING.model_copy(update={"trace_id": "t1"}),
        TICKET_FINDING.model_copy(update={"trace_id": "t2"}),
    ]
    model = exploding(times=99)  # never succeeds
    board = consolidate(model, findings, None, categories)

    assert board.source == "engine_predicted"
    assert len(board.issues) == 1  # deterministic fallback grouped both findings
    assert {o.trace_id for o in board.occurrences} == {"t1", "t2"}
    assert "falling back to deterministic clustering" in capsys.readouterr().err


def test_consolidation_retries_once_before_falling_back(categories, capsys):
    findings = [TICKET_FINDING.model_copy(update={"trace_id": "t1"})]
    plan = ConsolidationPlan(
        clusters=[Cluster(title="Recovered on retry", description="d",
                          category_id="tool_misuse", severity="high", finding_indices=[0])]
    )
    model = exploding(times=1, plans=[plan])
    board = consolidate(model, findings, None, categories)

    assert model.exploder.attempts == 2
    assert [i.title for i in board.issues] == ["Recovered on retry"]
    err = capsys.readouterr().err
    assert "attempt 1 failed" in err
    assert "falling back" not in err


def test_a_persistently_failing_consolidation_does_not_lose_any_finding(categories):
    findings = [
        TICKET_FINDING.model_copy(update={"trace_id": f"t{i}", "title": f"Mode {i}"})
        for i in range(5)
    ]
    board = consolidate(exploding(times=99), findings, None, categories)
    assert len(board.issues) == 5
    assert {o.trace_id for o in board.occurrences} == {f"t{i}" for i in range(5)}


def test_the_seed_board_survives_a_failed_consolidation(seed_board_payload, categories):
    findings = [TICKET_FINDING.model_copy(update={"trace_id": "t1"})]
    seed = SeedIssueboard.model_validate(seed_board_payload)
    board = consolidate(exploding(times=99), findings, seed, categories)
    assert [i.error_id for i in board.issues][:2] == [
        "seed-tool-failure-hidden",
        "seed-answers-without-retrieval",
    ]


def test_malformed_structured_output_is_treated_as_a_failure_and_retried(categories, capsys):
    """A dict that is not a plan, or a plan with no clusters, is as useless as
    an exception and takes the same path."""
    findings = [TICKET_FINDING.model_copy(update={"trace_id": "t1"})]
    model = ExplodingModel(responses=[], structured=[])
    model.exploder = ExplodingStructured(0, plans=[ConsolidationPlan(), ConsolidationPlan()])
    board = consolidate(model, findings, None, categories)
    assert model.exploder.attempts == 2
    assert len(board.issues) == 1
    assert "falling back to deterministic clustering" in capsys.readouterr().err


# -- review finding 2: the large-N chunked path ---------------------------


def big_findings(n: int, modes: int = 4) -> list[RawFinding]:
    return [
        TICKET_FINDING.model_copy(
            update={
                "trace_id": f"trace-{i:04d}",
                "title": f"Failure mode {i % modes}",
                "category_id": "tool_misuse",
            }
        )
        for i in range(n)
    ]


def batch_sizes(model) -> list[int]:
    """How many findings each consolidation prompt actually carried."""
    sizes = []
    for call in model.calls:
        task = str(call[-1].content)
        if "Raw findings from this run" in task:
            sizes.append(task.count("] trace="))
    return sizes


def test_a_300_trace_run_consolidates_in_bounded_batches(categories):
    findings = big_findings(300)
    plans = []
    # One plan per batch: each batch clusters its own findings by mode.
    for start in range(0, 300, 80):
        chunk = findings[start : start + 80]
        plans.append(
            ConsolidationPlan(
                clusters=[
                    Cluster(
                        title=f"Failure mode {mode}",
                        description="d",
                        category_id="tool_misuse",
                        severity="high",
                        finding_indices=[i for i, f in enumerate(chunk)
                                         if f.title == f"Failure mode {mode}"],
                    )
                    for mode in range(4)
                ]
            )
        )
    # Then the cross-batch merge pass over the folded cluster summaries.
    plans.append(
        ConsolidationPlan(
            clusters=[
                Cluster(title=f"Failure mode {mode}", description="d",
                        category_id="tool_misuse", severity="high", finding_indices=[mode])
                for mode in range(4)
            ]
        )
    )
    model = FakeChatModel(responses=[], structured=plans)
    board = consolidate(model, findings, None, categories)

    assert batch_sizes(model) == [80, 80, 80, 60]
    assert all(size <= 80 for size in batch_sizes(model))
    assert len(board.issues) == 4, "four modes across 300 findings -> four issues"
    assert len(board.occurrences) == 300, "every trace keeps an occurrence"


def test_the_merge_pass_prompt_carries_summaries_not_findings(categories):
    findings = big_findings(160, modes=2)
    plans = [
        ConsolidationPlan(
            clusters=[
                Cluster(title=f"Failure mode {mode}", description="d",
                        category_id="tool_misuse", severity="high",
                        finding_indices=[i for i in range(80) if i % 2 == mode])
                for mode in range(2)
            ]
        )
        for _ in range(2)
    ]
    plans.append(ConsolidationPlan(clusters=[
        Cluster(title="Unified", description="d", category_id="tool_misuse",
                severity="high", finding_indices=[0, 1])
    ]))
    model = FakeChatModel(responses=[], structured=plans)
    consolidate(model, findings, None, categories)

    merge_prompt = str(model.calls[-1][-1].content)
    assert "Candidate issues from this run" in merge_prompt
    # Bounded by distinct failure modes, not by the 160 findings.
    assert "trace-0000" not in merge_prompt
    assert merge_prompt.count("[0]") == 1


def test_one_failing_batch_does_not_cost_the_other_batches(categories, capsys):
    """A 429 on batch 2 of 4 loses that batch's clustering, not the run."""
    findings = big_findings(300, modes=2)

    class FlakyBatch(ExplodingStructured):
        def __init__(self):
            super().__init__(0)
            self.n = 0

        def invoke(self, messages, **kwargs):
            self.attempts += 1
            self.n += 1
            if 3 <= self.n <= 4:  # both attempts of the second batch
                raise RuntimeError("429 rate limited")
            return ConsolidationPlan(
                clusters=[Cluster(title="Everything", description="d",
                                  category_id="tool_misuse", severity="high",
                                  finding_indices=list(range(80)))]
            )

    model = ExplodingModel(responses=[], structured=[])
    model.exploder = FlakyBatch()
    board = consolidate(model, findings, None, categories)

    assert "falling back to deterministic clustering" in capsys.readouterr().err
    # Every finding still reaches the board, however its batch was clustered.
    assert len(board.occurrences) == 300
    assert board.issues


def test_a_small_run_still_uses_a_single_consolidation_call(categories):
    findings = big_findings(10)
    model = FakeChatModel(responses=[], structured=[ConsolidationPlan(
        clusters=[Cluster(title="One", description="d", category_id="tool_misuse",
                          severity="high", finding_indices=list(range(10)))]
    )])
    consolidate(model, findings, None, categories)
    assert len(batch_sizes(model)) == 1
    assert model.calls[-1][-1].content.count("Candidate issues") == 0


def test_chunk_size_is_configurable_for_testing(categories):
    findings = big_findings(6, modes=1)
    plans = [
        ConsolidationPlan(clusters=[Cluster(title="M", description="d",
                                            category_id="tool_misuse", severity="high",
                                            finding_indices=[0, 1])])
        for _ in range(3)
    ]
    plans.append(ConsolidationPlan(clusters=[
        Cluster(title="M", description="d", category_id="tool_misuse",
                severity="high", finding_indices=[0])
    ]))
    model = FakeChatModel(responses=[], structured=plans)
    board = consolidate(model, findings, None, categories, chunk_size=2)
    assert batch_sizes(model) == [2, 2, 2]
    assert len(board.issues) == 1
    assert len(board.occurrences) == 6


def test_a_single_batch_folds_duplicate_clusters_like_a_multi_batch_run(categories):
    """Whether a corpus straddles the chunk boundary must not change the board.

    One call that names the same failure mode twice used to yield two issues,
    while the identical findings split across two batches yielded one.
    """
    findings = [
        TICKET_FINDING.model_copy(update={"trace_id": "t1"}),
        TICKET_FINDING.model_copy(update={"trace_id": "t2"}),
    ]
    duplicated = ConsolidationPlan(
        clusters=[
            Cluster(title="Tool error hidden", description="d", category_id="tool_misuse",
                    severity="low", finding_indices=[0]),
            Cluster(title="tool  error   hidden!", description="d", category_id="tool_misuse",
                    severity="high", finding_indices=[1]),
        ]
    )
    single = consolidate(
        FakeChatModel(responses=[], structured=[duplicated]), findings, None, categories
    )
    assert len(single.issues) == 1
    assert single.issues[0].severity == "high"
    assert {o.trace_id for o in single.occurrences} == {"t1", "t2"}

    # Same findings, same clusters, two batches -> identical board.
    split = consolidate(
        FakeChatModel(
            responses=[],
            structured=[
                ConsolidationPlan(clusters=[duplicated.clusters[0]]),
                ConsolidationPlan(
                    clusters=[duplicated.clusters[1].model_copy(update={"finding_indices": [0]})]
                ),
                ConsolidationPlan(clusters=[
                    Cluster(title="Tool error hidden", description="d",
                            category_id="tool_misuse", severity="high", finding_indices=[0])
                ]),
            ],
        ),
        findings,
        None,
        categories,
        chunk_size=1,
    )
    assert len(split.issues) == 1
    assert split.board_id == single.board_id
