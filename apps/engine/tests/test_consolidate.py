"""Clustering and the seed merge — the invariants the benchmark depends on.

All pure: the LLM only proposes a `ConsolidationPlan`; everything asserted here
happens in code, so a model that clusters badly still cannot produce a board
that duplicates a seed issue or loses the {trace_id, error_id} matrix.
"""

from __future__ import annotations

import pytest

from engine.consolidate import (
    assemble_board,
    board_id,
    complete_plan,
    fallback_plan,
    normalize_title,
    slug,
    valid_category,
)
from engine.models import (
    Cluster,
    ConsolidationPlan,
    Issue,
    IssueOccurrence,
    RawFinding,
    SeedIssueboard,
)

TICKET_MODE = "Tool error reported to the user as success"


def finding(trace_id: str, title: str = TICKET_MODE, **kwargs) -> RawFinding:
    defaults = {
        "description": "The tool call errored but the answer claims it worked.",
        "category_id": "tool_misuse",
        "severity": "high",
        "evidence": "TicketServiceError: upstream returned 503",
    }
    return RawFinding(trace_id=trace_id, title=title, **{**defaults, **kwargs})


@pytest.fixture
def seed(seed_board_payload) -> SeedIssueboard:
    return SeedIssueboard.model_validate(seed_board_payload)


# -- clustering ------------------------------------------------------------


def test_two_findings_of_the_same_mode_become_one_issue(categories):
    findings = [finding("trace-a"), finding("trace-b", title="Tool failure hidden from user")]
    plan = ConsolidationPlan(
        clusters=[
            Cluster(
                title=TICKET_MODE,
                description="A failed tool call is reported as a success.",
                category_id="tool_misuse",
                severity="high",
                finding_indices=[0, 1],
            )
        ]
    )
    board = assemble_board(plan, findings, None, categories)

    assert len(board.issues) == 1
    assert board.issues[0].title == TICKET_MODE
    assert {o.trace_id for o in board.occurrences} == {"trace-a", "trace-b"}
    assert {o.error_id for o in board.occurrences} == {board.issues[0].error_id}


def test_distinct_modes_stay_distinct_issues(categories):
    findings = [finding("trace-a"), finding("trace-b", title="Answer stops mid-sentence")]
    plan = ConsolidationPlan(
        clusters=[
            Cluster(title=TICKET_MODE, description="d", category_id="tool_misuse",
                    severity="high", finding_indices=[0]),
            Cluster(title="Truncated response", description="d", category_id="formatting",
                    severity="medium", finding_indices=[1]),
        ]
    )
    board = assemble_board(plan, findings, None, categories)
    assert len(board.issues) == 2
    assert {i.category_id for i in board.issues} == {"tool_misuse", "formatting"}


def test_repeated_finding_indices_do_not_duplicate_occurrences(categories):
    findings = [finding("trace-a")]
    plan = ConsolidationPlan(
        clusters=[Cluster(title="t", description="d", category_id="tool_misuse",
                          severity="high", finding_indices=[0, 0, 0])]
    )
    board = assemble_board(plan, findings, None, categories)
    assert len(board.occurrences) == 1


def test_two_findings_on_the_same_trace_and_span_collapse(categories):
    findings = [finding("trace-a", span_id="s-1"), finding("trace-a", span_id="s-1")]
    plan = ConsolidationPlan(
        clusters=[Cluster(title="t", description="d", category_id="tool_misuse",
                          severity="high", finding_indices=[0, 1])]
    )
    board = assemble_board(plan, findings, None, categories)
    assert len(board.occurrences) == 1


def test_out_of_range_indices_are_ignored_not_fatal(categories):
    findings = [finding("trace-a")]
    plan = ConsolidationPlan(
        clusters=[Cluster(title="t", description="d", category_id="tool_misuse",
                          severity="high", finding_indices=[0, 99, -3])]
    )
    board = assemble_board(plan, findings, None, categories)
    assert len(board.issues) == 1
    assert len(board.occurrences) == 1


def test_a_cluster_with_no_valid_findings_creates_no_issue(categories):
    plan = ConsolidationPlan(
        clusters=[Cluster(title="ghost", description="d", category_id="other",
                          severity="low", finding_indices=[42])]
    )
    board = assemble_board(plan, [finding("trace-a")], None, categories)
    # The ghost cluster is dropped; the unreferenced finding is recovered.
    assert [i.title for i in board.issues] == [TICKET_MODE]


# -- the seed merge --------------------------------------------------------


def test_seed_issue_gains_occurrences_without_being_duplicated(seed, categories):
    findings = [finding("trace-planted-ticket")]
    plan = ConsolidationPlan(
        clusters=[
            Cluster(
                title="Tool failure hidden",
                description="new wording for a known mode",
                category_id="tool_misuse",
                severity="high",
                finding_indices=[0],
                matches_seed_error_id="seed-tool-failure-hidden",
            )
        ]
    )
    board = assemble_board(plan, findings, seed, categories)

    assert [i.error_id for i in board.issues] == [i.error_id for i in seed.issues]
    matched = next(i for i in board.issues if i.error_id == "seed-tool-failure-hidden")
    # The seed issue's own text stays authoritative — only occurrences are added.
    assert matched.title == "Tool failure reported to the user as success"
    assert matched.description == seed.issues[0].description
    occurrences = [o for o in board.occurrences if o.error_id == "seed-tool-failure-hidden"]
    assert [o.trace_id for o in occurrences] == ["trace-planted-ticket"]


def test_seed_issues_survive_a_run_that_matched_none_of_them(seed, categories):
    board = assemble_board(ConsolidationPlan(), [], seed, categories)
    assert [i.error_id for i in board.issues] == [i.error_id for i in seed.issues]
    assert board.occurrences == []


def test_new_failure_modes_are_added_alongside_the_seed(seed, categories):
    findings = [finding("t1"), finding("t2", title="Answer stops mid-sentence",
                                       category_id="formatting", severity="medium")]
    plan = ConsolidationPlan(
        clusters=[
            Cluster(title="known", description="d", category_id="tool_misuse", severity="high",
                    finding_indices=[0], matches_seed_error_id="seed-tool-failure-hidden"),
            Cluster(title="Truncated response", description="d", category_id="formatting",
                    severity="medium", finding_indices=[1]),
        ]
    )
    board = assemble_board(plan, findings, seed, categories)
    assert len(board.issues) == 3
    assert board.issues[-1].title == "Truncated response"
    assert board.issues[-1].error_id.startswith("ep-")


def test_two_clusters_matching_one_seed_issue_still_yield_one_issue(seed, categories):
    findings = [finding("t1"), finding("t2")]
    plan = ConsolidationPlan(
        clusters=[
            Cluster(title="a", description="d", category_id="tool_misuse", severity="high",
                    finding_indices=[0], matches_seed_error_id="seed-tool-failure-hidden"),
            Cluster(title="b", description="d", category_id="tool_misuse", severity="high",
                    finding_indices=[1], matches_seed_error_id="seed-tool-failure-hidden"),
        ]
    )
    board = assemble_board(plan, findings, seed, categories)
    assert len(board.issues) == len(seed.issues)
    assert {o.trace_id for o in board.occurrences} == {"t1", "t2"}


def test_a_match_on_an_unknown_seed_id_becomes_a_new_issue(seed, categories):
    plan = ConsolidationPlan(
        clusters=[Cluster(title="invented", description="d", category_id="tool_misuse",
                          severity="high", finding_indices=[0],
                          matches_seed_error_id="seed-does-not-exist")]
    )
    board = assemble_board(plan, [finding("t1")], seed, categories)
    assert len(board.issues) == len(seed.issues) + 1
    assert board.issues[-1].error_id == "ep-invented"


def test_pre_existing_seed_occurrences_are_preserved_and_not_re_added(categories):
    seed = SeedIssueboard(
        issues=[Issue(error_id="s1", title="t", description="d",
                      category_id="tool_misuse", severity="high")],
        occurrences=[IssueOccurrence(error_id="s1", trace_id="t1", evidence="old")],
    )
    plan = ConsolidationPlan(
        clusters=[Cluster(title="t", description="d", category_id="tool_misuse", severity="high",
                          finding_indices=[0, 1], matches_seed_error_id="s1")]
    )
    board = assemble_board(plan, [finding("t1"), finding("t2")], seed, categories)
    assert len(board.occurrences) == 2
    assert next(o for o in board.occurrences if o.trace_id == "t1").evidence == "old"


def test_seed_extras_such_as_injection_mode_are_dropped(categories):
    """A seed issue carrying ablation bookkeeping must not leak it into the output."""
    seed = SeedIssueboard.model_validate(
        {
            "source": "seed",
            "issues": [
                {
                    "error_id": "s1",
                    "title": "t",
                    "description": "d",
                    "category_id": "tool_misuse",
                    "severity": "high",
                    "injection_mode": "dependency_fault",
                }
            ],
        }
    )
    board = assemble_board(ConsolidationPlan(), [], seed, categories)
    assert "injection_mode" not in board.model_dump_json()


# -- category clamping, ids, board shape -----------------------------------


def test_an_invented_category_degrades_to_other(categories):
    plan = ConsolidationPlan(
        clusters=[Cluster(title="t", description="d", category_id="made_up_category",
                          severity="low", finding_indices=[0])]
    )
    board = assemble_board(plan, [finding("t1")], None, categories)
    assert board.issues[0].category_id == "other"


def test_valid_category_passes_known_ids_through(categories):
    assert valid_category("formatting", categories) == "formatting"
    assert valid_category(None, categories) == "other"
    assert valid_category("formatting", []) == "other"


def test_error_ids_are_deterministic_and_unique(categories):
    plan = ConsolidationPlan(
        clusters=[
            Cluster(title="Same Title!", description="a", category_id="other",
                    severity="low", finding_indices=[0]),
            Cluster(title="same title", description="b", category_id="other",
                    severity="low", finding_indices=[1]),
        ]
    )
    board = assemble_board(plan, [finding("t1"), finding("t2")], None, categories)
    ids = [i.error_id for i in board.issues]
    assert ids == ["ep-same-title", "ep-same-title-2"]


def test_new_error_ids_never_collide_with_seed_ids(categories):
    seed = SeedIssueboard(
        issues=[Issue(error_id="ep-clash", title="t", description="d",
                      category_id="other", severity="low")]
    )
    plan = ConsolidationPlan(
        clusters=[Cluster(title="clash", description="d", category_id="other",
                          severity="low", finding_indices=[0])]
    )
    board = assemble_board(plan, [finding("t1")], seed, categories)
    assert [i.error_id for i in board.issues] == ["ep-clash", "ep-clash-2"]


def test_board_is_engine_predicted_with_a_content_hash_id(categories):
    board = assemble_board(ConsolidationPlan(), [], None, categories)
    assert board.source == "engine_predicted"
    assert len(board.board_id) == 16
    assert board_id(board.model_copy(update={"board_id": "x"})) == board.board_id


def test_occurrences_fall_back_to_the_description_when_evidence_is_empty(categories):
    plan = ConsolidationPlan(
        clusters=[Cluster(title="t", description="d", category_id="other",
                          severity="low", finding_indices=[0])]
    )
    board = assemble_board(plan, [finding("t1", evidence="")], None, categories)
    assert board.occurrences[0].evidence == "The tool call errored but the answer claims it worked."


def test_a_finding_without_a_trace_id_produces_no_occurrence(categories):
    plan = ConsolidationPlan(
        clusters=[Cluster(title="t", description="d", category_id="other",
                          severity="low", finding_indices=[0])]
    )
    board = assemble_board(plan, [finding("")], None, categories)
    assert board.occurrences == []


# -- plan completion / fallback -------------------------------------------


def test_fallback_groups_by_category_and_normalized_title():
    findings = [
        finding("t1", title="Tool error hidden"),
        finding("t2", title="tool  error   hidden!"),
        finding("t3", title="Tool error hidden", category_id="other"),
    ]
    plan = fallback_plan(findings)
    assert len(plan.clusters) == 2
    assert sorted(len(c.finding_indices) for c in plan.clusters) == [1, 2]


def test_fallback_takes_the_highest_severity_in_a_group():
    findings = [finding("t1", severity="low"), finding("t2", severity="high")]
    assert fallback_plan(findings).clusters[0].severity == "high"


def test_complete_plan_recovers_findings_the_model_forgot():
    findings = [finding("t1"), finding("t2", title="Truncated", category_id="formatting")]
    plan = ConsolidationPlan(
        clusters=[Cluster(title="t", description="d", category_id="tool_misuse",
                          severity="high", finding_indices=[0])]
    )
    completed = complete_plan(plan, findings)
    claimed = {i for c in completed.clusters for i in c.finding_indices}
    assert claimed == {0, 1}
    assert completed.clusters[-1].title == "Truncated"


def test_complete_plan_is_a_no_op_when_everything_is_claimed():
    findings = [finding("t1")]
    plan = ConsolidationPlan(
        clusters=[Cluster(title="t", description="d", category_id="tool_misuse",
                          severity="high", finding_indices=[0])]
    )
    assert complete_plan(plan, findings) is plan


def test_slug_and_normalize_title_helpers():
    assert slug("Tool error: reported as SUCCESS!") == "tool-error-reported-as-success"
    assert slug("!!!") == "issue"
    assert normalize_title("Tool  error!!  hidden") == normalize_title("tool error hidden")
