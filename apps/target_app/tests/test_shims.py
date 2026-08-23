"""Mode C shims: fault instructions read from config.configurable.

No fault key present => completely normal behaviour. Callers only ever know
the key names declared in configs/target_app.yaml.
"""

import time

import pytest

from target_app.corpus import load_corpus
from target_app.retriever import Retriever
from target_app.shims import (
    FAULT_BEHAVIORS,
    FAULT_LLM_KEY,
    FAULT_RETRIEVER_KEY,
    FAULT_TOOL_KEY,
    Fault,
    apply_llm_fault,
    apply_retriever_fault,
    apply_tool_fault,
    read_fault,
)


@pytest.fixture(scope="module")
def retriever():
    return Retriever(load_corpus())


# --------------------------------------------------------------- key surface

def test_declared_keys_match_the_black_box_contract():
    assert FAULT_RETRIEVER_KEY == "fault_retriever"
    assert FAULT_TOOL_KEY == "fault_tool"
    assert FAULT_LLM_KEY == "fault_llm"
    assert FAULT_BEHAVIORS[FAULT_RETRIEVER_KEY] == frozenset(
        {"irrelevant_docs", "empty", "stale"}
    )
    assert FAULT_BEHAVIORS[FAULT_TOOL_KEY] == frozenset(
        {"error", "timeout", "corrupted_result"}
    )
    assert FAULT_BEHAVIORS[FAULT_LLM_KEY] == frozenset({"truncate_output"})


# ------------------------------------------------------------- read_fault

@pytest.mark.parametrize("config", [None, {}, {"configurable": {}}, {"configurable": None}])
def test_no_fault_key_means_no_fault(config):
    for key in FAULT_BEHAVIORS:
        assert read_fault(config, key) is None


def test_other_keys_are_ignored():
    config = {"configurable": {"thread_id": "abc", "fault_tool": "error"}}
    assert read_fault(config, FAULT_RETRIEVER_KEY) is None
    assert read_fault(config, FAULT_TOOL_KEY) == Fault("error", {})


def test_string_value_is_the_behavior():
    fault = read_fault({"configurable": {FAULT_RETRIEVER_KEY: "empty"}}, FAULT_RETRIEVER_KEY)
    assert fault == Fault("empty", {})


def test_dict_value_carries_params():
    config = {"configurable": {FAULT_TOOL_KEY: {"behavior": "timeout", "delay_seconds": 0.01}}}
    fault = read_fault(config, FAULT_TOOL_KEY)
    assert fault.behavior == "timeout"
    assert fault.params == {"delay_seconds": 0.01}


@pytest.mark.parametrize("value", ["", None, False])
def test_falsy_values_disarm_the_shim(value):
    assert read_fault({"configurable": {FAULT_LLM_KEY: value}}, FAULT_LLM_KEY) is None


def test_unknown_behavior_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        read_fault({"configurable": {FAULT_RETRIEVER_KEY: "explode"}}, FAULT_RETRIEVER_KEY)


def test_behavior_from_the_wrong_family_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        read_fault({"configurable": {FAULT_RETRIEVER_KEY: "timeout"}}, FAULT_RETRIEVER_KEY)


# -------------------------------------------------------- retriever shim

def test_unarmed_retriever_returns_relevant_docs(retriever):
    results = apply_retriever_fault(None, query="refund policy", retriever=retriever, k=3)
    assert results[0].doc_id == "refund-policy"


def test_empty_behavior_returns_no_documents(retriever):
    results = apply_retriever_fault(
        Fault("empty", {}), query="refund policy", retriever=retriever, k=3
    )
    assert results == []


def test_irrelevant_docs_returns_the_worst_matches(retriever):
    query = "refund policy money back"
    normal = {r.doc_id for r in retriever.search(query, k=3)}
    faulted = apply_retriever_fault(
        Fault("irrelevant_docs", {}), query=query, retriever=retriever, k=3
    )
    assert len(faulted) == 3
    assert not ({r.doc_id for r in faulted} & normal)
    assert "refund-policy" not in {r.doc_id for r in faulted}


def test_stale_keeps_doc_ids_but_swaps_in_the_archived_revision(retriever):
    query = "refund policy money back"
    normal = retriever.search(query, k=3)
    faulted = apply_retriever_fault(
        Fault("stale", {}), query=query, retriever=retriever, k=3
    )
    assert [r.doc_id for r in faulted] == [r.doc_id for r in normal]
    top = faulted[0]
    assert top.doc_id == "refund-policy"
    assert "90 days" in top.text  # the outdated policy
    assert "30 days" not in top.text
    assert top.updated == "2019-03-02"


def test_stale_docs_carry_no_fault_marker(retriever):
    """Activation must be organically visible, never fingerprintable."""
    faulted = apply_retriever_fault(
        Fault("stale", {}), query="refund", retriever=retriever, k=3
    )
    blob = " ".join(r.text + r.title + r.updated for r in faulted).lower()
    for tell in ("fault", "shim", "ablat", "inject", "archived"):
        assert tell not in blob


# ------------------------------------------------------------- tool shim

NORMAL_TICKET = {
    "status": "created",
    "ticket_id": "NN-4F2A19",
    "subject": "Refund request",
    "priority": "normal",
    "eta_hours": 24,
}


def test_unarmed_tool_passes_the_result_through():
    assert apply_tool_fault(None, NORMAL_TICKET) == NORMAL_TICKET


def test_error_behavior_returns_a_failure_without_a_ticket():
    result = apply_tool_fault(Fault("error", {}), NORMAL_TICKET)
    assert result["status"] == "error"
    assert "ticket_id" not in result
    assert result["error"]


def test_timeout_behavior_reports_a_timeout_and_actually_waits():
    started = time.monotonic()
    result = apply_tool_fault(Fault("timeout", {"delay_seconds": 0.05}), NORMAL_TICKET)
    elapsed = time.monotonic() - started
    assert elapsed >= 0.05
    assert result["status"] == "error"
    assert result["error"] == "timeout"
    assert "ticket_id" not in result


def test_timeout_delay_is_capped():
    started = time.monotonic()
    apply_tool_fault(Fault("timeout", {"delay_seconds": 900}), NORMAL_TICKET)
    assert time.monotonic() - started < 10


def test_corrupted_result_keeps_success_shape_but_garbles_the_payload():
    result = apply_tool_fault(Fault("corrupted_result", {}), NORMAL_TICKET)
    assert result["status"] == "created"
    assert result["ticket_id"] != NORMAL_TICKET["ticket_id"]
    assert result["eta_hours"] < 0
    assert result != NORMAL_TICKET


def test_tool_shim_does_not_mutate_the_input():
    before = dict(NORMAL_TICKET)
    apply_tool_fault(Fault("corrupted_result", {}), NORMAL_TICKET)
    apply_tool_fault(Fault("error", {}), NORMAL_TICKET)
    assert NORMAL_TICKET == before


# -------------------------------------------------------------- llm shim

LONG = (
    "You can request a refund within thirty days of the charge date. "
    "Open Settings, then Billing, then Request refund. "
    "Refunds settle in five to seven business days."
)


def test_unarmed_llm_leaves_text_alone():
    assert apply_llm_fault(None, LONG) == LONG


def test_truncate_output_cuts_the_text_short():
    out = apply_llm_fault(Fault("truncate_output", {}), LONG)
    assert len(out) < len(LONG)
    assert LONG.startswith(out)
    assert out  # never empty


def test_truncate_fraction_is_configurable():
    short = apply_llm_fault(Fault("truncate_output", {"fraction": 0.1}), LONG)
    long = apply_llm_fault(Fault("truncate_output", {"fraction": 0.8}), LONG)
    assert len(short) < len(long) < len(LONG)


def test_truncation_leaves_no_ellipsis_or_marker():
    out = apply_llm_fault(Fault("truncate_output", {}), LONG)
    assert not out.rstrip().endswith(("...", "…", "]"))
    assert "truncat" not in out.lower()


def test_truncating_short_or_empty_text_is_safe():
    assert apply_llm_fault(Fault("truncate_output", {}), "") == ""
    assert apply_llm_fault(Fault("truncate_output", {}), "Hi.") == "Hi."
