"""The served graph: the sequential loop, the model swap, and the run output."""

from __future__ import annotations

import json
import threading
import time

import pytest
from langchain_core.messages import AIMessage

from engine import graph as graph_module
from engine.llm import DEFAULT_MODEL, resolve_model_name
from engine.models import (
    Cluster,
    ConsolidationPlan,
    FindingExtraction,
    FindingExtractionList,
    RawFinding,
)
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
    trace_id="trace-planted-ticket",
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
        fake.structured.append(
            FindingExtractionList(
                findings=[
                    FindingExtraction(**f.model_dump(exclude={"trace_id"}))
                    for f in per_trace_findings.get(trace_id, [])
                ]
            )
        )
    if plan is not None:
        fake.structured.append(plan)


def run(traces_file, seed=None, categories=None, config=None, concurrency=1):
    """Invoke the graph.

    `concurrency` defaults to 1 so tests driving a *scripted* fake model stay
    deterministic — a queue of scripted responses has no meaning when eight
    workers pop from it at once. Concurrency itself is covered by the tests
    below, which instrument `analyze_trace` instead of scripting a model.
    """
    config = dict(config or {})
    if concurrency is not None:
        configurable = dict(config.get("configurable") or {})
        configurable.setdefault("analysis_concurrency", concurrency)
        config["configurable"] = configurable
    return graph_module.graph.invoke(
        {
            "trace_file": str(traces_file),
            "seed_issueboard": seed or {},
            "categories": [c.model_dump() for c in (categories or [])],
        },
        config=config,
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


def fail_traces(monkeypatch, failing: set[str]):
    """Make analysis raise for the named traces and succeed for the rest."""

    def selective(model, index, trace_id, running_titles, categories, max_tool_calls):
        if trace_id in failing:
            raise RuntimeError("401 Unauthorized")
        return [TICKET_FINDING.model_copy(update={"trace_id": trace_id})]

    monkeypatch.setattr(graph_module, "build_model", lambda name: FakeChatModel())
    monkeypatch.setattr(graph_module, "analyze_trace", selective)
    monkeypatch.setattr(graph_module, "consolidate", lambda m, f, s, c: _board_from(f))


def test_a_failure_on_every_trace_raises_instead_of_reporting_no_errors(
    traces_file, monkeypatch, categories
):
    """An expired key must not be indistinguishable from a clean corpus."""
    fail_traces(monkeypatch, set(ALL_TRACE_IDS))
    with pytest.raises(Exception, match="401 Unauthorized"):
        run(traces_file, categories=categories)


def test_a_failure_rate_above_the_threshold_calls_the_run_off(
    traces_file, monkeypatch, categories
):
    """2 of 6 is 33% — well past the point where 'the Engine found little' is a
    less likely story than 'the Engine barely ran'."""
    fail_traces(monkeypatch, {"trace-clean-pricing", "trace-planted-refund"})
    with pytest.raises(Exception, match="analysis failed on 2 of 6 traces"):
        run(traces_file, categories=categories)


def test_a_failure_rate_at_or_below_the_threshold_completes(
    traces_file, monkeypatch, categories
):
    """1 of 6 is under 20%: skip the flaky trace, keep the other five."""
    fail_traces(monkeypatch, {"trace-planted-refund"})
    result = run(traces_file, categories=categories)
    assert len(result["occurrences"]) == 5
    assert "trace-planted-refund" not in {o["trace_id"] for o in result["occurrences"]}


def test_the_failure_count_is_always_reported_on_stderr(
    traces_file, monkeypatch, categories, capsys
):
    """Below the threshold the run still succeeds — but silently succeeding is
    what made 299/300 failures look like 'found nothing'."""
    fail_traces(monkeypatch, {"trace-planted-refund"})
    run(traces_file, categories=categories)
    err = capsys.readouterr().err
    assert "1/6 traces failed analysis" in err
    assert "401 Unauthorized" in err


def test_a_clean_run_says_nothing_about_failures(traces_file, monkeypatch, categories, capsys):
    fail_traces(monkeypatch, set())
    run(traces_file, categories=categories)
    assert "failed analysis" not in capsys.readouterr().err


def test_the_threshold_is_a_rate_not_a_count(traces_file, monkeypatch, categories):
    """The same absolute count that is fatal at N=6 is tolerable at larger N;
    guard the constant so a future edit cannot silently make it a count."""
    assert 0 < graph_module.MAX_TRACE_FAILURE_RATE < 1


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


# -- review finding 6: the index cache -------------------------------------


def test_the_index_cache_survives_being_cleared_between_calls(traces_file, monkeypatch):
    """The cache is cleared before each insert; the loaded index must come from
    a local, or a concurrent clear turns into a KeyError mid-run."""
    graph_module._INDEX_CACHE.clear()
    first = graph_module.trace_index(str(traces_file))
    assert graph_module.trace_index(str(traces_file)) is first

    real_from_file = graph_module.TraceIndex.from_file

    def clear_after_load(path):
        index = real_from_file(path)
        graph_module._INDEX_CACHE.clear()
        return index

    graph_module._INDEX_CACHE.clear()
    monkeypatch.setattr(graph_module.TraceIndex, "from_file", staticmethod(clear_after_load))
    assert graph_module.trace_index(str(traces_file)).trace_ids == ALL_TRACE_IDS


# -- batched-parallel analysis --------------------------------------------


class Instrumented:
    """Stands in for `analyze_trace`, recording concurrency and inputs.

    Instrumenting the orchestration rather than scripting a model is the only
    way to observe in-flight counts and completion order; a scripted queue
    cannot tell you which worker popped what.
    """

    def __init__(self, delay=0.01, failing=(), delays=None):
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.delay = delay
        self.delays = delays or {}
        self.failing = set(failing)
        self.titles_seen: dict[str, list[str]] = {}
        self.started: list[str] = []
        self.finished: list[str] = []

    def __call__(self, model, index, trace_id, running_titles, categories, max_tool_calls):
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self.titles_seen[trace_id] = list(running_titles)
            self.started.append(trace_id)
        try:
            time.sleep(self.delays.get(trace_id, self.delay))
            if trace_id in self.failing:
                raise RuntimeError("401 Unauthorized")
            # One distinct title per trace: colliding titles would be folded
            # into one cluster and the ordering signal would vanish.
            return [TICKET_FINDING.model_copy(update={"trace_id": trace_id, "title": trace_id})]
        finally:
            with self.lock:
                self.in_flight -= 1
                self.finished.append(trace_id)


def instrument(monkeypatch, probe: Instrumented):
    monkeypatch.setattr(graph_module, "build_model", lambda name: FakeChatModel())
    monkeypatch.setattr(graph_module, "analyze_trace", probe)
    monkeypatch.setattr(graph_module, "consolidate", lambda m, f, s, c: _board_from(f))
    return probe


@pytest.mark.parametrize("concurrency,expected_max", [(1, 1), (2, 2), (8, 6)])
def test_max_in_flight_tracks_the_configured_batch_size(
    traces_file, monkeypatch, categories, concurrency, expected_max
):
    """N=1 stays strictly sequential; N>1 really overlaps. With N=8 and only six
    fixture traces the ceiling is the corpus, not the knob."""
    probe = instrument(monkeypatch, Instrumented(delay=0.05))
    run(traces_file, categories=categories, concurrency=concurrency)
    assert probe.max_in_flight == expected_max


def test_findings_follow_input_order_not_completion_order(
    traces_file, monkeypatch, categories
):
    """Randomised durations make completion order the reverse of input order;
    the board must not notice."""
    delays = {trace_id: 0.02 * (len(ALL_TRACE_IDS) - i)
              for i, trace_id in enumerate(ALL_TRACE_IDS)}
    probe = instrument(monkeypatch, Instrumented(delay=0.02, delays=delays))
    result = run(traces_file, categories=categories, concurrency=8)

    assert probe.finished != ALL_TRACE_IDS, "completion order was not shuffled; test is vacuous"
    assert [o["trace_id"] for o in result["occurrences"]] == ALL_TRACE_IDS


def test_the_same_corpus_gives_the_same_board_at_any_batch_size(
    traces_file, monkeypatch, categories
):
    boards = []
    for concurrency in (1, 3, 8):
        instrument(monkeypatch, Instrumented(delay=0))
        boards.append(run(traces_file, categories=categories, concurrency=concurrency))
    assert boards[0]["board_id"] == boards[1]["board_id"] == boards[2]["board_id"]


def test_running_titles_are_shared_between_batches_not_within(
    traces_file, monkeypatch, categories
):
    """Every trace in a batch sees the same list — otherwise what a trace sees
    would depend on which worker happened to finish first."""
    probe = instrument(monkeypatch, Instrumented(delay=0))
    run(traces_file, categories=categories, concurrency=3)

    first_batch = ALL_TRACE_IDS[:3]
    second_batch = ALL_TRACE_IDS[3:]
    assert all(probe.titles_seen[t] == [] for t in first_batch)
    # The second batch inherits every title the first one discovered.
    inherited = probe.titles_seen[second_batch[0]]
    assert len(inherited) == 3
    assert all(probe.titles_seen[t] == inherited for t in second_batch)


def test_at_n_equals_one_every_trace_sees_all_its_predecessors(
    traces_file, monkeypatch, categories
):
    probe = instrument(monkeypatch, Instrumented(delay=0))
    run(traces_file, categories=categories, concurrency=1)
    seen = [len(probe.titles_seen[t]) for t in ALL_TRACE_IDS]
    assert seen == [0, 1, 2, 3, 4, 5]


def test_a_worker_exception_does_not_kill_its_batchmates(
    traces_file, monkeypatch, categories
):
    """One failure inside a batch of eight costs that trace only."""
    probe = instrument(monkeypatch, Instrumented(delay=0, failing={"trace-planted-refund"}))
    result = run(traces_file, categories=categories, concurrency=8)

    assert set(probe.started) == set(ALL_TRACE_IDS), "every trace was still attempted"
    survivors = [t for t in ALL_TRACE_IDS if t != "trace-planted-refund"]
    assert [o["trace_id"] for o in result["occurrences"]] == survivors


def test_the_failure_rate_threshold_counts_across_workers(
    traces_file, monkeypatch, categories
):
    """Two concurrent failures out of six is still 33%, and still fatal."""
    instrument(
        monkeypatch,
        Instrumented(delay=0, failing={"trace-clean-pricing", "trace-planted-refund"}),
    )
    with pytest.raises(Exception, match="analysis failed on 2 of 6 traces"):
        run(traces_file, categories=categories, concurrency=8)


def test_the_batch_is_the_superstep_so_a_big_corpus_needs_few(
    traces_file, monkeypatch, categories
):
    """2 + ceil(n/N) supersteps: the reason the recursion limit stops being the
    binding constraint on a 300-trace run."""
    probe = instrument(monkeypatch, Instrumented(delay=0.05))
    run(traces_file, categories=categories, concurrency=8)
    assert probe.max_in_flight > 1
    assert graph_module.RECURSION_LIMIT >= 2 + -(-300 // 8)


# -- the concurrency knob --------------------------------------------------


def test_resolve_concurrency_precedence_and_clamping(monkeypatch):
    monkeypatch.delenv("ENGINE_ANALYSIS_CONCURRENCY", raising=False)
    resolve = graph_module.resolve_concurrency
    assert resolve(None) == graph_module.DEFAULT_ANALYSIS_CONCURRENCY
    assert resolve({"configurable": {}}) == graph_module.DEFAULT_ANALYSIS_CONCURRENCY
    assert resolve({"configurable": {"analysis_concurrency": 4}}) == 4
    assert resolve({"configurable": {"analysis_concurrency": "4"}}) == 4
    # Clamped, not refused: a typo should cost speed, not the whole run.
    assert resolve({"configurable": {"analysis_concurrency": 0}}) == 1
    assert resolve({"configurable": {"analysis_concurrency": -5}}) == 1
    assert resolve({"configurable": {"analysis_concurrency": 999}}) == 16
    assert resolve({"configurable": {"analysis_concurrency": "nope"}}) == 8
    assert resolve({"configurable": {"analysis_concurrency": None}}) == 8

    monkeypatch.setenv("ENGINE_ANALYSIS_CONCURRENCY", "2")
    assert resolve(None) == 2
    assert resolve({"configurable": {"analysis_concurrency": 5}}) == 5


def test_the_default_batch_size_is_what_the_readme_documents():
    assert graph_module.DEFAULT_ANALYSIS_CONCURRENCY == 8
    assert (graph_module.MIN_ANALYSIS_CONCURRENCY, graph_module.MAX_ANALYSIS_CONCURRENCY) == (1, 16)


def test_a_real_analysis_pass_runs_concurrently_end_to_end(
    traces_file, monkeypatch, categories
):
    """The genuine `analyze_trace` (tool loop + structured emit) under a
    content-addressed fake model, eight at a time."""
    seen: list[str] = []
    seen_lock = threading.Lock()

    def router(messages):
        text = str(messages[-1].content)
        trace_id = next(t for t in ALL_TRACE_IDS if t in text)
        with seen_lock:
            seen.append(trace_id)
        return FindingExtractionList(
            findings=[
                FindingExtraction(
                    **TICKET_FINDING.model_copy(update={"title": "Shared"}).model_dump(
                        exclude={"trace_id"}
                    )
                )
            ]
        )

    fake = FakeChatModel(responses=[], structured=[], router=router)
    monkeypatch.setattr(graph_module, "build_model", lambda name: fake)
    monkeypatch.setattr(graph_module, "consolidate", lambda m, f, s, c: _board_from(f))

    result = run(traces_file, categories=categories, concurrency=8)
    assert sorted(seen) == sorted(ALL_TRACE_IDS)
    assert [o["trace_id"] for o in result["occurrences"]] == ALL_TRACE_IDS


def five_trace_file(tmp_path, traces_file):
    """A 5-trace corpus, so 1 failure is exactly 20% — the threshold itself."""
    payload = json.loads(traces_file.read_text())
    payload["traces"] = payload["traces"][:5]
    path = tmp_path / "five.json"
    path.write_text(json.dumps(payload))
    return path


def test_exactly_the_threshold_still_completes(tmp_path, traces_file, monkeypatch, categories):
    """1 of 5 is 20.0%: the check is `> rate`, so the boundary itself passes."""
    path = five_trace_file(tmp_path, traces_file)
    instrument(monkeypatch, Instrumented(delay=0, failing={"trace-clean-pricing"}))
    result = run(path, categories=categories, concurrency=8)
    assert len(result["occurrences"]) == 4


def test_just_over_the_threshold_raises(tmp_path, traces_file, monkeypatch, categories):
    """2 of 5 is 40%, the next representable step up on this corpus."""
    path = five_trace_file(tmp_path, traces_file)
    instrument(
        monkeypatch,
        Instrumented(delay=0, failing={"trace-clean-pricing", "trace-clean-export"}),
    )
    with pytest.raises(Exception, match="analysis failed on 2 of 5 traces"):
        run(path, categories=categories, concurrency=8)


def test_the_boundary_is_inclusive_by_construction():
    """Pins the comparison itself, so the fixture arithmetic above cannot drift
    from the constant it is meant to exercise."""
    rate = graph_module.MAX_TRACE_FAILURE_RATE
    assert not 1 / 5 > rate, "1-of-5 must sit exactly on the threshold, not over it"
    assert 2 / 5 > rate
