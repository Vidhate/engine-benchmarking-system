"""Steps 1 and 2 — proposing errors from the corpus, and planning their injection."""

from __future__ import annotations

import pytest

from benchmark.ablation.agent import BEHAVIOR_VOCABULARY, ScriptedAblationAgent
from benchmark.ablation.plan import mode_preconditions, plan_ablation, rotate_behavior
from benchmark.ablation.propose import (
    DEFAULT_MIN_PER_MODE,
    allowed_modes,
    available_shims,
    build_digest,
    propose_errors,
)
from benchmark.schemas.ablation import FaultConfig, FilterStep
from benchmark.schemas.configs import TargetAppConfig

from .conftest import make_proposal

# --------------------------------------------------------------- corpus digest


def test_the_digest_is_built_from_trace_store_reads(store, traces, target_cfg):
    digest = build_digest(store, [t.trace_id for t in traces.traces], target_cfg)
    assert digest.n_traces == len(traces.traces)
    assert digest.span_types["llm"] > 0
    assert "corpus_search" in digest.span_names
    assert digest.tool_names == ["rag_search"]
    assert "refund-policy" in digest.retrieved_doc_ids
    assert digest.sample_exchanges, "the agent must see real exchanges, not just counts"


def test_the_digest_survives_a_trace_that_is_not_in_the_store(store, traces, target_cfg):
    ids = [t.trace_id for t in traces.traces] + ["trace-that-never-landed"]
    digest = build_digest(store, ids, target_cfg)
    assert digest.n_traces == len(traces.traces)


def test_available_shims_come_only_from_the_declared_keys(target_cfg):
    assert available_shims(target_cfg) == ["llm_proxy", "retriever", "tool"]
    assert allowed_modes(target_cfg) == ["replay_edit", "dependency_fault"]


def test_without_declared_keys_the_benchmark_degrades_to_replay_edit_only():
    bare = TargetAppConfig(
        base_url="http://x", assistant_id="a", langsmith_project="p", fault_configurable_keys={}
    )
    assert available_shims(bare) == []
    assert allowed_modes(bare) == ["replay_edit"]


# -------------------------------------------------------------------- step 1

def test_propose_errors_restamps_ids_and_keeps_the_injection_mode(
    store, traces, categories, target_cfg
):
    agent = ScriptedAblationAgent(
        {
            "hallucination": [make_proposal("agent-numbered-it-0")],
            "retrieval_failure": [
                make_proposal(
                    "agent-numbered-it-0", "retrieval_failure", mode="dependency_fault"
                )
            ],
        }
    )
    proposals, _digest, _dropped = propose_errors(
        store, [t.trace_id for t in traces.traces], categories, 1, target_cfg, agent
    )
    ids = [p.issue.error_id for p in proposals]
    assert ids == ["E-hallucination-00", "E-retrieval_failure-00"]
    assert len(set(ids)) == len(ids), "ids must be unique across categories"
    assert [p.issue.injection_mode for p in proposals] == ["replay_edit", "dependency_fault"]
    assert all(p.issue.category_id == p.issue.error_id.split("-")[1] for p in proposals)


def test_a_proposal_whose_marker_is_not_in_its_replacement_is_dropped(
    store, traces, categories, target_cfg, caplog
):
    broken = make_proposal(marker="MISSING-999", replacement="no marker here")
    agent = ScriptedAblationAgent({"hallucination": [broken]})
    proposals, _digest, _dropped = propose_errors(
        store, [t.trace_id for t in traces.traces], categories, 1, target_cfg, agent
    )
    assert proposals == []


def test_a_dependency_fault_on_an_undeclared_shim_is_dropped(
    store, traces, categories, target_cfg
):
    bare = TargetAppConfig(
        base_url="http://x",
        assistant_id="a",
        langsmith_project="p",
        fault_configurable_keys={"retriever": "fault_retriever"},
    )
    proposal = make_proposal(
        "x", "retrieval_failure", mode="dependency_fault",
        fault=FaultConfig(shim="tool", target="create_ticket", behavior="error"),
    )
    agent = ScriptedAblationAgent({"retrieval_failure": [proposal]})
    proposals, _digest, _dropped = propose_errors(
        store, [t.trace_id for t in traces.traces], categories, 1, bare, agent
    )
    assert proposals == []


def test_the_agent_is_only_offered_the_modes_the_app_supports(
    store, traces, categories, target_cfg
):
    agent = ScriptedAblationAgent({})
    propose_errors(
        store, [t.trace_id for t in traces.traces], categories, 2, target_cfg, agent,
        min_per_mode={},
    )
    assert agent.propose_calls == [(c.category_id, 2) for c in categories]


# -------------------------------------------------------------------- step 2

def test_a_replay_edit_plan_carries_the_corruption_in_its_action():
    spec = plan_ablation(make_proposal())
    assert spec.mode == "replay_edit"
    assert spec.fault_config is None
    action = spec.ablation_actions[0]
    assert action.transform == "replace"
    assert action.params["marker"] == "NBX-4471"
    assert action.params["replacement"].startswith("I have escalated")


def test_a_dependency_fault_plan_carries_the_fault_config_and_no_actions():
    spec = plan_ablation(make_proposal(mode="dependency_fault"))
    assert spec.ablation_actions == []
    assert spec.fault_config is not None
    assert spec.fault_config.behavior == "irrelevant_docs"


def test_mode_preconditions_are_necessary_conditions_not_heuristics():
    replay = [(s.field, s.op) for s in mode_preconditions(make_proposal())]
    assert ("metadata.thread_id", "exists") in replay, "replay forks a thread"
    fault = [(s.field, s.value) for s in mode_preconditions(make_proposal(mode="dependency_fault"))]
    assert ("span_types", "retrieval") in fault, "a retriever fault needs a retrieval span"


def test_replanning_relaxes_the_agents_filter_but_never_the_preconditions():
    proposal = make_proposal(
        filter_steps=[
            FilterStep(field="span_types", op="eq", value="tool"),
            FilterStep(field="turns[*].final_response", op="contains", value="refund"),
        ]
    )
    first = plan_ablation(proposal, attempt=0)
    second = plan_ablation(proposal, attempt=1)
    third = plan_ablation(proposal, attempt=2)
    assert len(first.filter.steps) == 4
    assert len(second.filter.steps) == 3
    assert len(third.filter.steps) == 2
    preconditions = {(s.field, s.op) for s in mode_preconditions(proposal)}
    assert preconditions <= {(s.field, s.op) for s in third.filter.steps}


def test_behaviour_rotation_walks_the_vocabulary_and_then_gives_up():
    vocabulary = BEHAVIOR_VOCABULARY["retriever"]
    assert rotate_behavior("retriever", vocabulary[0]) == vocabulary[1]
    assert rotate_behavior("retriever", vocabulary[-1]) is None
    assert rotate_behavior("llm_proxy", "truncate_output") is None


def test_a_hallucinated_filter_field_costs_the_step_not_the_error(
    store, traces, categories, target_cfg
):
    """The agent authors filter fields as free text; one bad root must not
    raise out of the middle of validation and take the whole run with it.

    Only the ROOT is checked: a path that merely reaches nothing
    (`turns[*].nope`) is an ordinary "this trace does not have one" and
    resolves to `[]`, which is exactly what a filter step is for.
    """
    proposal = make_proposal(
        filter_steps=[
            FilterStep(field="tool_calls[*].name", op="exists"),
            FilterStep(field="span_types", op="eq", value="tool"),
        ]
    )
    agent = ScriptedAblationAgent({"hallucination": [proposal]})
    proposals, _digest, _dropped = propose_errors(
        store, [t.trace_id for t in traces.traces], categories, 1, target_cfg, agent
    )
    assert [s.field for s in proposals[0].filter_steps] == ["span_types"]


# ------------------------------------------------- the agent seam is guarded

class _ExplodingAgent:
    """An agent that fails a fixed number of times, then behaves."""

    def __init__(self, exc: Exception, fail_times: int = 99):
        self.exc = exc
        self.fail_times = fail_times
        self.calls = 0

    def propose(self, category, n, digest, allowed_modes):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return [make_proposal(f"ok-{category.category_id}", category.category_id)]

    def revise_corruption(self, proposal, digest, reasons):  # pragma: no cover
        raise AssertionError("not reached")


def _propose(agent, store, traces, categories, cfg):
    # `min_per_mode={}` turns the mode-coverage pass off: these tests are about
    # the FIRST pass's retry and drop policy, and a second round of calls would
    # make the call counts below measure two things at once. Coverage has its
    # own tests further down.
    return propose_errors(
        store, [t.trace_id for t in traces.traces], categories, 1, cfg, agent, min_per_mode={}
    )


def test_a_transport_failure_is_retried_once(store, traces, categories, target_cfg):
    from benchmark.ablation.agent import AgentTransportError

    agent = _ExplodingAgent(AgentTransportError("connection reset"), fail_times=1)
    proposals, _digest, dropped = _propose(agent, store, traces, categories, target_cfg)
    assert agent.calls == len(categories) + 1, "exactly one extra call, for the retry"
    assert dropped == {}
    assert len(proposals) == len(categories)


def test_a_persistent_transport_failure_drops_only_that_category(
    store, traces, categories, target_cfg
):
    from benchmark.ablation.agent import AgentTransportError

    agent = _ExplodingAgent(AgentTransportError("connection reset"))
    proposals, _digest, dropped = _propose(agent, store, traces, categories, target_cfg)
    assert proposals == []
    assert set(dropped) == {c.category_id for c in categories}
    assert all("failed twice" in reason for reason in dropped.values())


def test_a_non_json_reply_drops_the_category_without_a_retry(
    store, traces, categories, target_cfg
):
    from benchmark.ablation.agent import AgentResponseError

    agent = _ExplodingAgent(AgentResponseError("the model's reply was not JSON"))
    proposals, _digest, dropped = _propose(agent, store, traces, categories, target_cfg)
    assert agent.calls == len(categories), "a bad reply is not worth a second identical prompt"
    assert proposals == []
    assert all("nothing usable" in reason for reason in dropped.values())


def test_an_unexpected_agent_error_still_only_costs_its_category(
    store, traces, categories, target_cfg
):
    agent = _ExplodingAgent(ValueError("something nobody predicted"), fail_times=1)
    proposals, _digest, dropped = _propose(agent, store, traces, categories, target_cfg)
    # first category dies, the rest proceed
    assert len(dropped) == 1
    assert len(proposals) == len(categories) - 1


def test_a_hallucinated_operator_costs_the_step_not_the_error(
    store, traces, categories, target_cfg
):
    bad = FilterStep.model_construct(field="span_types", op="in", value=["tool"])
    good = FilterStep(field="status", op="eq", value="ok")
    proposal = make_proposal(filter_steps=[bad, good])
    agent = ScriptedAblationAgent({"hallucination": [proposal]})
    proposals, _digest, _dropped = _propose(agent, store, traces, categories, target_cfg)
    assert [s.op for s in proposals[0].filter_steps] == ["eq"]


def test_the_openai_parser_skips_a_step_with_an_invented_operator():
    """The Literal on FilterStep raises; one bad step must not lose the draft."""
    from benchmark.ablation.agent import OpenAIAblationAgent
    from benchmark.schemas.issues import ErrorCategory

    raw = {
        "injection_mode": "replay_edit",
        "title": "t",
        "description": "d",
        "severity": "high",
        "filter_steps": [
            {"field": "span_types", "op": "in", "value": ["tool"]},
            {"field": "status", "op": "eq", "value": "ok"},
        ],
        "corruption": {"replacement": "case NBX-1 is open", "marker": "NBX-1"},
    }
    proposal = OpenAIAblationAgent._to_proposal(
        ErrorCategory(category_id="hallucination", name="h", description="d"),
        0,
        raw,
        ["replay_edit"],
    )
    assert proposal is not None
    assert [s.op for s in proposal.filter_steps] == ["eq"]


def _parsed(raw: dict):
    from benchmark.ablation.agent import OpenAIAblationAgent
    from benchmark.schemas.issues import ErrorCategory

    return OpenAIAblationAgent._to_proposal(
        ErrorCategory(category_id="hallucination", name="h", description="d"),
        0,
        {
            "injection_mode": "replay_edit",
            "title": "t",
            "description": "d",
            "severity": "high",
            **raw,
        },
        ["replay_edit"],
    )


def test_the_agent_can_pin_the_turn_it_wants_corrupted():
    """`choose_turn_index` honours a pin, but nothing ever read one out of the
    model's reply — so the override was unreachable in the live path."""
    proposal = _parsed(
        {"corruption": {"replacement": "case NBX-1 is open", "marker": "NBX-1",
                        "turn_index": 2}}
    )
    assert proposal is not None
    assert proposal.corruption.turn_index == 2


def test_an_unpinned_or_unusable_turn_index_leaves_the_draw_alone():
    for body in (
        {"replacement": "case NBX-1 is open", "marker": "NBX-1"},
        {"replacement": "case NBX-1 is open", "marker": "NBX-1", "turn_index": "second"},
        {"replacement": "case NBX-1 is open", "marker": "NBX-1", "turn_index": -1},
        {"replacement": "case NBX-1 is open", "marker": "NBX-1", "turn_index": True},
        {"replacement": "case NBX-1 is open", "marker": "NBX-1", "turn_index": None},
    ):
        proposal = _parsed({"corruption": body})
        assert proposal is not None
        assert proposal.corruption.turn_index is None, body


def test_a_proposal_with_no_injection_mode_cannot_be_planned():
    proposal = make_proposal()
    proposal.issue.injection_mode = None
    with pytest.raises(ValueError, match="injection_mode"):
        plan_ablation(proposal)


# ------------------------------------------------------ step 1: mode coverage
#
# Measured need: across two live mini runs the agent proposed `replay_edit` for
# all seven categories and zero `dependency_fault`. Each category is drafted in
# its own call with no view of the run's mode balance, and most of a support
# taxonomy is content-shaped, so that is the natural draw — but the report's
# content-vs-mechanism half has nothing behind it without both modes.

SHIMLESS = TargetAppConfig(
    base_url="http://127.0.0.1:2024",
    assistant_id="target_app",
    langsmith_project="p",
    fault_configurable_keys={},
)


def _coverage(agent, store, traces, categories, cfg, n=1):
    return propose_errors(
        store, [t.trace_id for t in traces.traces], categories, n, cfg, agent
    )


def test_a_first_pass_with_no_mechanism_error_is_re_prompted_for_one(
    store, traces, categories, target_cfg
):
    """The path the live runs needed: content-only draw -> one narrow re-ask."""
    agent = ScriptedAblationAgent(
        {
            # Each category can draft either, but only the replay_edit one fits
            # in the n=1 first pass — exactly the live behaviour.
            c.category_id: [
                make_proposal(f"c-{c.category_id}", c.category_id, mode="replay_edit"),
                make_proposal(f"m-{c.category_id}", c.category_id, mode="dependency_fault"),
            ]
            for c in categories
        }
    )
    proposals, _digest, dropped = _coverage(agent, store, traces, categories, target_cfg)

    modes = [p.issue.injection_mode for p in proposals]
    assert modes.count("replay_edit") == len(categories), "the first pass is unchanged"
    assert modes.count("dependency_fault") >= 1, "the coverage floor was not met"
    assert not [k for k in dropped if k.endswith("-coverage")], dropped


def test_the_coverage_pass_asks_for_the_missing_mode_and_nothing_else(
    store, traces, categories, target_cfg
):
    """The re-prompt narrows `allowed_modes`, which IS the "specifically" part.

    It needs no new agent method: the existing `propose(category, n, digest,
    allowed_modes)` already carries the restriction, so a scripted agent and
    the live one both honour it the same way.
    """
    seen: list[list[str]] = []

    class ModeRecordingAgent(ScriptedAblationAgent):
        def propose(self, category, n, digest, allowed_modes):
            seen.append(list(allowed_modes))
            return super().propose(category, n, digest, allowed_modes)

    agent = ModeRecordingAgent(
        {
            c.category_id: [
                make_proposal(f"c-{c.category_id}", c.category_id, mode="replay_edit"),
                make_proposal(f"m-{c.category_id}", c.category_id, mode="dependency_fault"),
            ]
            for c in categories
        }
    )
    _coverage(agent, store, traces, categories, target_cfg)

    first_pass = seen[: len(categories)]
    assert all(set(m) == set(allowed_modes(target_cfg)) for m in first_pass)
    assert seen[len(categories):], "no coverage call was made"
    assert all(m == ["dependency_fault"] for m in seen[len(categories):])


def test_the_coverage_pass_stops_as_soon_as_the_floor_is_met(
    store, traces, categories, target_cfg
):
    """Bounded cost: one extra call in the ordinary case, not one per category."""
    agent = ScriptedAblationAgent(
        {
            c.category_id: [
                make_proposal(f"c-{c.category_id}", c.category_id, mode="replay_edit"),
                make_proposal(f"m-{c.category_id}", c.category_id, mode="dependency_fault"),
            ]
            for c in categories
        }
    )
    _coverage(agent, store, traces, categories, target_cfg)
    assert len(agent.propose_calls) == len(categories) + 1


def test_a_first_pass_that_already_covers_both_modes_is_not_re_prompted(
    store, traces, categories, target_cfg
):
    agent = ScriptedAblationAgent(
        {
            categories[0].category_id: [
                make_proposal("m-0", categories[0].category_id, mode="dependency_fault")
            ],
            **{
                c.category_id: [make_proposal(f"c-{c.category_id}", c.category_id)]
                for c in categories[1:]
            },
        }
    )
    _coverage(agent, store, traces, categories, target_cfg)
    assert len(agent.propose_calls) == len(categories), "an unnecessary agent call was spent"


def test_a_shimless_app_takes_no_coverage_pass_and_does_not_fail(store, traces, categories):
    """No declared shim means no fault is injectable — asking would be theatre.

    Locked decision #1: with no shim surface the benchmark degrades to
    replay_edit-only. That is a documented downgrade, not a failure, so it must
    not spend calls and must not record a coverage shortfall.
    """
    agent = ScriptedAblationAgent(
        {
            c.category_id: [
                make_proposal(f"c-{c.category_id}", c.category_id, mode="replay_edit"),
                make_proposal(f"m-{c.category_id}", c.category_id, mode="dependency_fault"),
            ]
            for c in categories
        }
    )
    assert allowed_modes(SHIMLESS) == ["replay_edit"]
    proposals, _digest, dropped = _coverage(agent, store, traces, categories, SHIMLESS)

    assert len(agent.propose_calls) == len(categories), "a shimless app was re-prompted"
    assert {p.issue.injection_mode for p in proposals} == {"replay_edit"}
    assert dropped == {}, dropped


def test_a_corpus_that_cannot_carry_a_mechanism_error_is_reported_not_forced(
    store, traces, categories, target_cfg
):
    """No forced fault. The floor is a target; the shortfall is surfaced."""
    agent = ScriptedAblationAgent(
        {c.category_id: [make_proposal(f"c-{c.category_id}", c.category_id)] for c in categories}
    )
    proposals, _digest, dropped = _coverage(agent, store, traces, categories, target_cfg)

    assert {p.issue.injection_mode for p in proposals} == {"replay_edit"}
    assert len(agent.propose_calls) == 2 * len(categories), "every category should be offered"
    shortfall = dropped.get("dependency_fault-coverage", "")
    assert "0/1" in shortfall and "one-sided" in shortfall, dropped


def test_coverage_proposals_do_not_collide_with_first_pass_error_ids(
    store, traces, categories, target_cfg
):
    """Both passes stamp `E-<category>-NN`; the second must not reuse an index."""
    agent = ScriptedAblationAgent(
        {
            c.category_id: [
                make_proposal(f"c-{c.category_id}", c.category_id, mode="replay_edit"),
                make_proposal(f"m-{c.category_id}", c.category_id, mode="dependency_fault"),
            ]
            for c in categories
        }
    )
    proposals, _digest, _dropped = _coverage(agent, store, traces, categories, target_cfg)
    ids = [p.issue.error_id for p in proposals]
    assert len(ids) == len(set(ids)), ids


def test_the_shipped_floor_asks_for_one_mechanism_error():
    assert DEFAULT_MIN_PER_MODE == {"dependency_fault": 1}
