"""LangSmith run trees -> our Trace schema, and everything that must NOT survive."""

from __future__ import annotations

from datetime import timedelta

import pytest

from benchmark.harness.collector import (
    SPAN_SELECT,
    IngestionTimeout,
    LangSmithCollector,
    TurnCoverageError,
    VacuousProjectionError,
)
from benchmark.harness.ids import trace_id_for
from benchmark.harness.scrub import LeakDetected
from benchmark.schemas.traces import Trace
from tests.harness.conftest import (
    FakeLangSmithClient,
    FakeRun,
    metadata_extra,
    react_agent_runs,
)


def make_collector(client, **kwargs):
    kwargs.setdefault("poll_interval_s", 0.0)
    kwargs.setdefault("sleep", lambda _s: None)
    return LangSmithCollector(project="engine-bench-target", client=client, **kwargs)


# --------------------------------------------------------------- normalization

def test_run_tree_becomes_a_schema_valid_single_turn_trace():
    runs = react_agent_runs("s-abc")
    collector = make_collector(FakeLangSmithClient(runs))

    trace = collector.collect("s-abc", input_id="safe-1", mode="single_turn")

    assert trace.trace_id == trace_id_for("s-abc")
    assert trace.input_id == "safe-1"
    assert trace.status == "ok"
    assert len(trace.turns) == 1
    turn = trace.turns[0]
    assert turn.user_message == "what is the refund window?"
    assert turn.final_response == "30 days."
    types = {s.span_type for s in turn.spans}
    assert {"agent", "llm", "tool", "retrieval"} <= types
    # Round-trips through the schema, not merely constructible.
    assert Trace.model_validate_json(trace.model_dump_json()) == trace


def test_framework_noise_spans_are_dropped_and_survivors_reparented():
    runs = react_agent_runs("s-abc")
    collector = make_collector(FakeLangSmithClient(runs))

    spans = collector.collect("s-abc", input_id="i", mode="single_turn").turns[0].spans
    names = {s.name for s in spans}
    by_id = {s.span_id: s for s in spans}

    assert "RunnableSequence" not in names, "framework wrapper survived the filter"
    assert "ChannelWrite<messages>" not in names
    assert {"agent", "tools", "ChatOpenAI", "rag_search", "corpus_search"} <= names
    # The llm span hung off the dropped RunnableSequence; it must re-attach to
    # the nearest *surviving* ancestor rather than dangle.
    llm = next(s for s in spans if s.name == "ChatOpenAI")
    assert llm.parent_span_id in by_id
    assert by_id[llm.parent_span_id].name == "agent"
    # No span points at an id that is not in the trace.
    assert all(s.parent_span_id is None or s.parent_span_id in by_id for s in spans)
    # Exactly one root (the agent span).
    assert [s.name for s in spans if s.parent_span_id is None] == ["target_app"]


@pytest.mark.parametrize(
    "name,run_type,noise",
    [
        # Framework plumbing — dropped.
        ("RunnableSequence", "chain", True),
        ("RunnableAssign<messages>", "chain", True),
        ("ChannelWrite<messages>", "chain", True),
        ("__start__", "chain", True),
        ("should_continue", "chain", True),
        ("Prompt", "chain", True),
        ("ChatPromptTemplate", "chain", True),
        ("StrOutputParser", "parser", True),
        ("AgentMiddleware.wrap_model_call", "chain", True),
        ("some_middleware", "chain", True),
        ("mystery", "annotation_queue", True),  # unknown run type
        # App semantics — kept.
        ("ChatOpenAI", "llm", False),
        ("rag_search", "tool", False),
        ("corpus_search", "retriever", False),
        ("agent", "chain", False),
        ("tools", "chain", False),
        ("call_model", "chain", False),
    ],
)
def test_the_documented_noise_filter_rule(name, run_type, noise):
    from benchmark.harness.collector import is_noise_span

    assert is_noise_span(name, run_type, is_root=False) is noise
    # The root run is always kept, whatever the framework called it.
    assert is_noise_span(name, run_type, is_root=True) is False


def test_span_attributes_are_allowlisted_never_copied_wholesale():
    runs = react_agent_runs("s-abc", leak_metadata=True)
    collector = make_collector(FakeLangSmithClient(runs))

    spans = collector.collect("s-abc", input_id="i", mode="single_turn").turns[0].spans
    llm = next(s for s in spans if s.name == "ChatOpenAI")

    assert llm.attributes["model"] == "gpt-5.1-mini"
    assert llm.attributes["tokens"] == 137
    assert llm.attributes["run_type"] == "llm"
    # Nothing from run.extra / run.tags / run.serialized is copied through.
    for span in spans:
        assert set(span.attributes) <= {
            "run_type",
            "model",
            "tokens",
            "error",
            "duration_ms",
        }, f"un-allowlisted attribute on {span.name}"


# ------------------------------------------------------------- leak scrubbing

def test_no_fault_token_or_fault_metadata_key_survives_into_the_stored_trace():
    """The BLOCKING Phase-2 hand-off requirement, as a fixture test.

    Every armed-run leak surface at once: the fault echoed in run metadata on
    every span, in run tags, and the subclass name in the llm manifest.
    """
    runs = react_agent_runs("s-armed", leak_metadata=True)
    for run in runs:
        run.tags = ["fault_retriever:stale", "shim=on"]
    for run in runs:
        if run.run_type == "llm":
            run.serialized = {"id": ["target_app", "shims", "SupportChatModel"]}

    collector = make_collector(FakeLangSmithClient(runs), audit_manifests=False)
    trace = collector.collect("s-armed", input_id="i", mode="single_turn")

    blob = trace.model_dump_json().lower()
    for token in (
        "fault_retriever",
        "fault_tool",
        "fault_llm",
        "irrelevant_docs",
        "corrupted_result",
        "truncate_output",
        "shim",
        "supportchatmodel",
    ):
        assert token not in blob, f"{token!r} leaked into the stored trace"

    def fault_keys(node):
        if isinstance(node, dict):
            return [k for k in node if k.startswith("fault_")] + [
                k for v in node.values() for k in fault_keys(v)
            ]
        if isinstance(node, list):
            return [k for v in node for k in fault_keys(v)]
        return []

    assert fault_keys(trace.model_dump(mode="json")) == []


def test_a_fault_token_in_span_payloads_is_a_loud_failure_not_a_silent_pass():
    """Positive control: the scrubber must be able to fail."""
    runs = react_agent_runs("s-leaky")
    tool = next(r for r in runs if r.run_type == "tool")
    tool.outputs = {"output": '{"note": "fault_retriever armed"}'}

    collector = make_collector(FakeLangSmithClient(runs), audit_manifests=False)
    with pytest.raises(LeakDetected) as excinfo:
        collector.collect("s-leaky", input_id="i", mode="single_turn")
    assert "fault_retriever" in str(excinfo.value)


def test_extra_leak_tokens_from_the_armed_behavior_are_scanned_too():
    runs = react_agent_runs("s-x")
    retr = next(r for r in runs if r.run_type == "retriever")
    retr.outputs = {"output": [{"doc_id": "d", "provenance": "stale_archive"}]}

    collector = make_collector(FakeLangSmithClient(runs), audit_manifests=False)
    collector.collect("s-x", input_id="i", mode="single_turn")  # not a leak by default
    with pytest.raises(LeakDetected):
        collector.collect(
            "s-x", input_id="i", mode="single_turn", extra_leak_tokens=("stale_archive",)
        )


# --------------------------------------------- explicit projection & vacuity

def test_span_fetch_asks_for_every_field_it_reads():
    runs = react_agent_runs("s-abc")
    client = FakeLangSmithClient(runs)
    make_collector(client).collect("s-abc", input_id="i", mode="single_turn")

    span_calls = [c for c in client.calls if c["trace_id"] is not None and c["run_type"] is None]
    assert span_calls, "no child-span fetch happened"
    for call in span_calls:
        assert call["select"] is not None, "runs fetched without an explicit projection"
        assert set(SPAN_SELECT) <= set(call["select"])


def test_manifest_audit_fails_loudly_when_serialized_comes_back_none():
    """Phase 2's lesson: list_runs does not project `serialized` by default.

    A getattr fallback would silently scan nothing; this must be a hard error.
    """
    runs = react_agent_runs("s-abc")
    client = FakeLangSmithClient(runs, drop_serialized=True)
    collector = make_collector(client, audit_manifests=True)

    with pytest.raises(VacuousProjectionError) as excinfo:
        collector.collect("s-abc", input_id="i", mode="single_turn")
    assert "serialized" in str(excinfo.value)


def test_manifest_audit_scans_the_manifest_it_fetched():
    runs = react_agent_runs("s-abc")
    for run in runs:
        if run.run_type == "llm":
            run.serialized = {"id": ["target_app", "shims", "SupportChatModel"]}
    collector = make_collector(FakeLangSmithClient(runs), audit_manifests=True)

    with pytest.raises(LeakDetected) as excinfo:
        collector.collect("s-abc", input_id="i", mode="single_turn")
    assert "supportchatmodel" in str(excinfo.value).lower()


def test_manifest_audit_requests_serialized_explicitly():
    runs = react_agent_runs("s-abc")
    client = FakeLangSmithClient(runs)
    make_collector(client, audit_manifests=True).collect("s-abc", input_id="i", mode="single_turn")

    manifest_calls = [c for c in client.calls if c["run_type"] == "llm"]
    assert manifest_calls, "manifest audit never fetched llm runs"
    assert all("serialized" in (c["select"] or []) for c in manifest_calls)


def test_projection_gap_on_a_field_the_normalizer_reads_fails_loudly():
    runs = react_agent_runs("s-abc")
    client = FakeLangSmithClient(runs)
    collector = make_collector(client, span_select=["id", "name", "parent_run_id", "trace_id"])

    with pytest.raises(VacuousProjectionError):
        collector.collect("s-abc", input_id="i", mode="single_turn")


# ------------------------------------------------------- ingestion polling

def test_child_spans_are_polled_for_until_ingestion_settles():
    """LangSmith child ingestion lags the root by up to ~30s."""
    full = react_agent_runs("s-abc")
    root_only = [full[0]]
    partial = full[:4]
    client = FakeLangSmithClient(
        full, reveal_schedule=[root_only, partial, full, full]
    )
    collector = make_collector(client)

    trace = collector.collect("s-abc", input_id="i", mode="single_turn")

    # 8 runs in, 2 framework wrappers dropped.
    assert len(trace.turns[0].spans) == 6
    child_fetches = [c for c in client.calls if c["trace_id"] is not None and not c["run_type"]]
    assert len(child_fetches) >= 3, "collector did not poll"


def test_ingestion_that_never_settles_is_a_bounded_loud_failure():
    full = react_agent_runs("s-abc")
    client = FakeLangSmithClient(full, reveal_schedule=[[full[0]]])
    ticks = iter([float(i) for i in range(0, 2000)])
    collector = make_collector(
        client, child_timeout_s=10.0, monotonic=lambda: next(ticks)
    )

    with pytest.raises(IngestionTimeout):
        collector.collect("s-abc", input_id="i", mode="single_turn")


def test_missing_root_run_is_a_bounded_loud_failure():
    client = FakeLangSmithClient(react_agent_runs("s-other"))
    ticks = iter([float(i) for i in range(0, 2000)])
    collector = make_collector(client, root_timeout_s=5.0, monotonic=lambda: next(ticks))

    with pytest.raises(IngestionTimeout):
        collector.collect("s-abc", input_id="i", mode="single_turn")


def test_a_server_that_ignores_the_metadata_filter_does_not_yield_a_wrong_trace():
    """Defensive: never trust the server-side filter to have been applied."""
    client = FakeLangSmithClient(react_agent_runs("s-other"), honor_metadata_filter=False)
    ticks = iter([float(i) for i in range(0, 2000)])
    collector = make_collector(client, root_timeout_s=5.0, monotonic=lambda: next(ticks))

    with pytest.raises(IngestionTimeout):
        collector.collect("s-abc", input_id="i", mode="single_turn")


# ------------------------------------------------------------------ turns

def test_multi_turn_roots_become_ordered_turns_of_one_trace():
    runs = react_agent_runs(
        "s-mt", turn_index=0, trace_id="tr-b", user_message="hi", final_response="hello"
    ) + react_agent_runs(
        "s-mt", turn_index=1, trace_id="tr-a", user_message="and refunds?", final_response="30 days"
    )
    collector = make_collector(FakeLangSmithClient(runs))

    trace = collector.collect("s-mt", input_id="mt-1", mode="multi_turn", expected_turns=2)

    assert [t.turn_index for t in trace.turns] == [0, 1]
    assert [t.user_message for t in trace.turns] == ["hi", "and refunds?"]
    assert all(t.spans for t in trace.turns)
    assert trace.metadata["turn_count"] == 2


def test_collector_waits_for_all_expected_turn_roots():
    turn0 = react_agent_runs("s-mt", turn_index=0, trace_id="tr-a")
    ticks = iter([float(i) for i in range(0, 2000)])
    # Turn 1's root never lands in LangSmith — a half-collected conversation
    # must be a loud failure, not a quietly truncated Trace.
    collector = make_collector(
        FakeLangSmithClient(turn0), root_timeout_s=5.0, monotonic=lambda: next(ticks)
    )
    with pytest.raises(IngestionTimeout):
        collector.collect("s-mt", input_id="i", mode="multi_turn", expected_turns=2)


def test_retry_duplicated_roots_collapse_to_the_latest_attempt_per_turn():
    """A retried turn leaves several roots carrying the SAME turn_index.

    Slicing the raw list would assemble {failed attempt, retry, turn 1} and
    silently drop turn 2 — a "3-turn" trace that never happened.
    """
    turn0_failed = react_agent_runs(
        "s-mt", turn_index=0, trace_id="tr-0a", user_message="hi", final_response="(timed out)"
    )
    turn0_retry = react_agent_runs(
        "s-mt", turn_index=0, trace_id="tr-0b", user_message="hi", final_response="hello"
    )
    turn1 = react_agent_runs(
        "s-mt", turn_index=1, trace_id="tr-1", user_message="refunds?", final_response="30 days"
    )
    turn2 = react_agent_runs(
        "s-mt", turn_index=2, trace_id="tr-2", user_message="thanks", final_response="anytime"
    )
    # The retry started later than the attempt it replaced.
    for run in turn0_retry:
        run.start_time = run.start_time + timedelta(seconds=30)
        run.end_time = run.end_time + timedelta(seconds=30)

    collector = make_collector(
        FakeLangSmithClient(turn0_failed + turn0_retry + turn1 + turn2)
    )
    trace = collector.collect("s-mt", input_id="i", mode="multi_turn", expected_turns=3)

    assert [t.turn_index for t in trace.turns] == [0, 1, 2]
    assert [t.user_message for t in trace.turns] == ["hi", "refunds?", "thanks"]
    assert trace.turns[0].final_response == "hello", "kept the failed attempt, not the retry"
    assert trace.metadata["langsmith_trace_ids"] == ["tr-0b", "tr-1", "tr-2"]


def test_duplicated_roots_do_not_let_the_wait_finish_before_every_turn_lands():
    turn0_a = react_agent_runs("s-mt", turn_index=0, trace_id="tr-0a")
    turn0_b = react_agent_runs("s-mt", turn_index=0, trace_id="tr-0b")
    turn1 = react_agent_runs("s-mt", turn_index=1, trace_id="tr-1")
    ticks = iter([float(i) for i in range(0, 2000)])
    collector = make_collector(
        FakeLangSmithClient(turn0_a + turn0_b + turn1),
        root_timeout_s=5.0,
        monotonic=lambda: next(ticks),
    )
    # Three roots exist, but they cover only turn indices {0, 1}.
    with pytest.raises(IngestionTimeout) as excinfo:
        collector.collect("s-mt", input_id="i", mode="multi_turn", expected_turns=3)
    assert "[0, 1]" in str(excinfo.value)


def test_a_turn_index_outside_the_expected_range_is_a_loud_failure():
    turn0 = react_agent_runs("s-mt", turn_index=0, trace_id="tr-0")
    stray = react_agent_runs("s-mt", turn_index=7, trace_id="tr-7")
    collector = make_collector(FakeLangSmithClient(turn0 + stray))

    with pytest.raises(TurnCoverageError, match="7"):
        collector.collect("s-mt", input_id="i", mode="single_turn", expected_turns=1)


# ------------------------------------------------ ingestion settle window

def test_a_burst_flushed_tree_is_not_accepted_at_its_first_plateau():
    """LangSmith flushes in bursts; a plateau is not the same as settled."""
    full = react_agent_runs("s-abc")
    root_only = [full[0]]
    burst = full[:4]
    schedule = [root_only, burst, burst, burst] + [full] * 8
    collector = make_collector(FakeLangSmithClient(full, reveal_schedule=schedule))

    trace = collector.collect("s-abc", input_id="i", mode="single_turn")

    assert len(trace.turns[0].spans) == 6, "settled on a mid-burst plateau"


def test_the_default_stability_window_covers_the_documented_ingestion_lag():
    collector = LangSmithCollector(project="p", client=FakeLangSmithClient([]))
    window_s = (collector.settle_polls - 1) * collector.poll_interval_s
    assert window_s >= 10.0, "a ~2s window is meaningless against a ~30s ingestion lag"
    assert collector.child_timeout_s > window_s


def test_a_tree_with_no_llm_run_at_all_never_settles():
    full = react_agent_runs("s-abc")
    without_llm = [r for r in full if r.run_type != "llm"]
    collector = make_collector(FakeLangSmithClient(without_llm), child_timeout_s=0.0)

    with pytest.raises(IngestionTimeout, match="llm span present=False"):
        collector.collect("s-abc", input_id="i", mode="single_turn")


def test_the_structural_floor_is_checked_after_normalization_and_is_configurable():
    """The app cannot answer without a model call, so too few llm spans means
    an incomplete tree however stable the raw run count looked."""
    full = react_agent_runs("s-abc")  # exactly one llm run, stable on every poll
    collector = make_collector(FakeLangSmithClient(full), min_llm_spans=2)

    with pytest.raises(IngestionTimeout, match="below the floor"):
        collector.collect("s-abc", input_id="i", mode="single_turn")

    # ...and the same tree is fine at the default floor of 1.
    ok = make_collector(FakeLangSmithClient(full)).collect(
        "s-abc", input_id="i", mode="single_turn"
    )
    assert sum(1 for s in ok.turns[0].spans if s.span_type == "llm") == 1


def test_caller_supplied_turn_text_overrides_the_derived_text():
    runs = react_agent_runs("s-abc")
    collector = make_collector(FakeLangSmithClient(runs))

    trace = collector.collect(
        "s-abc",
        input_id="i",
        mode="single_turn",
        hints=[{"turn_index": 0, "user_message": "authoritative", "final_response": "answer"}],
    )
    assert trace.turns[0].user_message == "authoritative"
    assert trace.turns[0].final_response == "answer"


# ------------------------------------------------------- app errors kept

def test_app_error_traces_are_kept_not_dropped():
    runs = react_agent_runs("s-err")
    runs[0].error = "TimeoutError: upstream took too long"
    collector = make_collector(FakeLangSmithClient(runs))

    trace = collector.collect("s-err", input_id="i", mode="single_turn")
    assert trace.status == "app_error"
    assert trace.turns, "an app_error trace still carries its turn"


def test_a_root_with_no_children_can_still_be_collected_when_not_required():
    root = FakeRun(
        id="root-x",
        name="target_app",
        run_type="chain",
        trace_id="tr-x",
        inputs={"messages": [{"type": "human", "content": "hi"}]},
        outputs={},
        error="app blew up before any child span",
        extra=metadata_extra("s-crash"),
    )
    collector = make_collector(FakeLangSmithClient([root]))

    trace = collector.collect(
        "s-crash", input_id="i", mode="single_turn", require_children=False
    )
    assert trace.status == "app_error"
    assert len(trace.turns[0].spans) == 1


def test_trace_metadata_is_an_allowlist_of_harness_facts():
    runs = react_agent_runs("s-abc", leak_metadata=True)
    collector = make_collector(FakeLangSmithClient(runs))

    trace = collector.collect(
        "s-abc", input_id="i", mode="single_turn", extra_metadata={"thread_id": "th-1"}
    )
    assert trace.metadata["session_id"] == "s-abc"
    assert trace.metadata["thread_id"] == "th-1"
    assert trace.metadata["langsmith_project"] == "engine-bench-target"
    assert not any(k.startswith("fault_") for k in trace.metadata)
