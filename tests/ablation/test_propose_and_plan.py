"""Steps 1 and 2 — proposing errors from the corpus, and planning their injection."""

from __future__ import annotations

import pytest

from benchmark.ablation.agent import BEHAVIOR_VOCABULARY, ScriptedAblationAgent
from benchmark.ablation.plan import mode_preconditions, plan_ablation, rotate_behavior
from benchmark.ablation.propose import (
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
    proposals, _digest = propose_errors(
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
    proposals, _ = propose_errors(
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
    proposals, _ = propose_errors(
        store, [t.trace_id for t in traces.traces], categories, 1, bare, agent
    )
    assert proposals == []


def test_the_agent_is_only_offered_the_modes_the_app_supports(
    store, traces, categories, target_cfg
):
    agent = ScriptedAblationAgent({})
    propose_errors(store, [t.trace_id for t in traces.traces], categories, 2, target_cfg, agent)
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
    proposals, _ = propose_errors(
        store, [t.trace_id for t in traces.traces], categories, 1, target_cfg, agent
    )
    assert [s.field for s in proposals[0].filter_steps] == ["span_types"]


def test_a_proposal_with_no_injection_mode_cannot_be_planned():
    proposal = make_proposal()
    proposal.issue.injection_mode = None
    with pytest.raises(ValueError, match="injection_mode"):
        plan_ablation(proposal)
