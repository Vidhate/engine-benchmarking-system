"""Mode C arming shape + the activation check Phase 5's step-3 validation needs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from benchmark.harness.faults import (
    FaultNotActivated,
    UndeclaredFault,
    activation_evidence,
    fault_configurable,
    structural_behavior_tokens,
)
from benchmark.schemas.ablation import FaultConfig
from benchmark.schemas.configs import TargetAppConfig
from benchmark.schemas.traces import Span, Trace, Turn

CFG = TargetAppConfig(
    base_url="http://x",
    assistant_id="target_app",
    langsmith_project="p",
    fault_configurable_keys={"retriever": "fault_retriever", "tool": "fault_tool"},
)

T0 = datetime(2026, 8, 23, tzinfo=UTC)


def span(name, span_type, outputs=None, duration_ms=100, span_id=None):
    return Span(
        span_id=span_id or f"{name}-1",
        name=name,
        span_type=span_type,
        start_time=T0,
        end_time=T0 + timedelta(milliseconds=duration_ms),
        outputs=outputs or {},
        attributes={"duration_ms": duration_ms},
    )


def trace_with(*spans):
    return Trace(
        trace_id="t1",
        input_id="i1",
        mode="single_turn",
        turns=[Turn(turn_index=0, user_message="q", final_response="a", spans=list(spans))],
    )


# --------------------------------------------------------------------- arming

def test_the_declared_key_comes_from_config_not_from_the_fault_config():
    fault = FaultConfig(shim="retriever", target="corpus", behavior="empty")
    armed = fault_configurable(CFG, fault)
    assert armed == {"fault_retriever": {"behavior": "empty"}}


def test_armed_values_are_mappings_because_scalars_are_refused_by_the_app():
    armed = fault_configurable(
        CFG, FaultConfig(shim="tool", target="create_ticket", behavior="timeout",
                         params={"delay_seconds": 3})
    )
    assert armed == {"fault_tool": {"behavior": "timeout", "params": {"delay_seconds": 3}}}
    assert all(isinstance(v, dict) for v in armed.values())


def test_a_shim_the_app_does_not_declare_is_refused_before_any_run_happens():
    with pytest.raises(UndeclaredFault, match="llm_proxy"):
        fault_configurable(
            CFG, FaultConfig(shim="llm_proxy", target="x", behavior="truncate_output")
        )


def test_only_structural_behavior_names_join_the_leak_scan():
    """"empty"/"stale" are ordinary English; corpus text uses them legitimately."""
    assert structural_behavior_tokens(
        FaultConfig(shim="retriever", target="c", behavior="irrelevant_docs")
    ) == ("irrelevant_docs",)
    assert structural_behavior_tokens(
        FaultConfig(shim="retriever", target="c", behavior="stale")
    ) == ()


# ----------------------------------------------------------------- activation

def test_validation_strength_must_be_chosen_explicitly():
    """Baseline-less validation is near-vacuous, so it cannot be the default.

    Phase 5's step-3 validation must pass `baseline`; anything weaker has to
    say so out loud.
    """
    fault = FaultConfig(shim="retriever", target="corpus_search", behavior="empty")
    trace = trace_with(span("corpus_search", "retrieval", outputs={"output": []}))

    with pytest.raises(ValueError) as excinfo:
        activation_evidence(trace, fault)
    assert "baseline" in str(excinfo.value)
    assert "weak_validation" in str(excinfo.value)

    # Weak form: acknowledged explicitly.
    assert "output" in activation_evidence(trace, fault, weak_validation=True)
    # Strong form: a byte-diff against the unarmed run.
    unarmed = trace_with(
        span("corpus_search", "retrieval", outputs={"output": [{"doc_id": "refund-policy"}]})
    )
    assert "output" in activation_evidence(trace, fault, baseline=unarmed)


def test_activation_evidence_is_read_off_the_relevant_span():
    fault = FaultConfig(shim="retriever", target="corpus_search", behavior="empty")
    trace = trace_with(span("corpus_search", "retrieval", outputs={"output": []}))
    assert "output" in activation_evidence(trace, fault, weak_validation=True)


def test_a_dependency_that_was_never_exercised_cannot_have_activated():
    fault = FaultConfig(shim="retriever", target="corpus_search", behavior="empty")
    trace = trace_with(span("ChatOpenAI", "llm", outputs={"generations": []}))
    with pytest.raises(FaultNotActivated, match="retrieval"):
        activation_evidence(trace, fault, weak_validation=True)


def test_a_baseline_turns_activation_into_a_visible_diff():
    fault = FaultConfig(shim="retriever", target="corpus_search", behavior="empty")
    baseline = trace_with(
        span("corpus_search", "retrieval", outputs={"output": [{"doc_id": "refund-policy"}]})
    )
    armed = trace_with(span("corpus_search", "retrieval", outputs={"output": []}))

    evidence = activation_evidence(armed, fault, baseline=baseline)
    assert "output" in evidence

    with pytest.raises(FaultNotActivated, match="identical"):
        activation_evidence(baseline, fault, baseline=baseline)


def test_a_delay_parameter_must_show_up_in_the_span_duration():
    fault = FaultConfig(
        shim="tool", target="create_ticket", behavior="timeout", params={"delay_seconds": 3}
    )
    fast = trace_with(span("create_ticket", "tool", outputs={"output": "{}"}, duration_ms=120))
    slow = trace_with(span("create_ticket", "tool", outputs={"output": "{}"}, duration_ms=3400))

    activation_evidence(slow, fault, weak_validation=True)
    with pytest.raises(FaultNotActivated, match="delay"):
        activation_evidence(fast, fault, weak_validation=True)


def test_the_named_target_span_is_preferred_when_it_exists():
    fault = FaultConfig(shim="tool", target="create_ticket", behavior="error")
    trace = trace_with(
        span("rag_search", "tool", outputs={"output": "docs"}, span_id="a"),
        span("create_ticket", "tool", outputs={"output": '{"status": "error"}'}, span_id="b"),
    )
    assert "error" in activation_evidence(trace, fault, weak_validation=True)


def test_activation_looks_across_every_turn_of_a_multi_turn_trace():
    fault = FaultConfig(shim="retriever", target="corpus_search", behavior="empty")
    trace = Trace(
        trace_id="t",
        input_id="i",
        mode="multi_turn",
        turns=[
            Turn(
                turn_index=0,
                user_message="a",
                final_response="b",
                spans=[span("ChatOpenAI", "llm")],
            ),
            Turn(
                turn_index=1,
                user_message="c",
                final_response="d",
                spans=[span("corpus_search", "retrieval", outputs={"output": []})],
            ),
        ],
    )
    assert activation_evidence(trace, fault, weak_validation=True)
