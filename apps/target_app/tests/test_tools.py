"""The two tools, and the fault keys reaching them through config.configurable."""

import json
import re

import pytest

from target_app.shims import FAULT_RETRIEVER_KEY, FAULT_TOOL_KEY
from target_app.tools import TOOLS, create_ticket, rag_search


def call(tool, args, configurable=None):
    return json.loads(tool.invoke(args, config={"configurable": configurable or {}}))


def test_the_app_exposes_exactly_two_tools():
    assert [t.name for t in TOOLS] == ["rag_search", "create_ticket"]


def test_fault_keys_are_not_part_of_the_tool_schemas():
    """Callers arm faults through run config, never through tool arguments."""
    for tool in TOOLS:
        schema = tool.args_schema.model_json_schema()
        properties = set(schema["properties"])
        assert "config" not in properties
        assert not properties & {FAULT_RETRIEVER_KEY, FAULT_TOOL_KEY}


# ------------------------------------------------------------------ rag_search

def test_rag_search_returns_relevant_documents():
    payload = call(rag_search, {"query": "how do I request a refund"})
    assert payload["query"] == "how do I request a refund"
    assert [r["doc_id"] for r in payload["results"]][0] == "refund-policy"
    assert "30 days" in payload["results"][0]["content"]


def test_rag_search_reports_a_miss_without_faults():
    payload = call(rag_search, {"query": "zzzz qqqq"})
    assert payload["results"] == []


def test_rag_search_empty_fault_returns_no_documents():
    payload = call(
        rag_search, {"query": "how do I request a refund"}, {FAULT_RETRIEVER_KEY: "empty"}
    )
    assert payload["results"] == []


def test_rag_search_irrelevant_docs_fault_swaps_the_documents():
    query = "how do I request a refund"
    normal = {r["doc_id"] for r in call(rag_search, {"query": query})["results"]}
    faulted = {
        r["doc_id"]
        for r in call(rag_search, {"query": query}, {FAULT_RETRIEVER_KEY: "irrelevant_docs"})[
            "results"
        ]
    }
    assert faulted and not (faulted & normal)


def test_rag_search_stale_fault_serves_the_outdated_revision():
    query = "how do I request a refund"
    payload = call(rag_search, {"query": query}, {FAULT_RETRIEVER_KEY: "stale"})
    top = payload["results"][0]
    assert top["doc_id"] == "refund-policy"
    assert top["updated"].startswith("2019")
    assert "90 days" in top["content"]


def test_rag_search_rejects_an_undeclared_behavior():
    with pytest.raises(ValueError, match="unknown"):
        call(rag_search, {"query": "refund"}, {FAULT_RETRIEVER_KEY: "nonsense"})


# ---------------------------------------------------------------- create_ticket

TICKET_ARGS = {"subject": "Refund for annual plan", "description": "Charged twice in June."}


def test_create_ticket_returns_a_fake_ticket_id():
    payload = call(create_ticket, TICKET_ARGS)
    assert payload["status"] == "created"
    assert re.fullmatch(r"NN-[0-9A-F]{6}", payload["ticket_id"])
    assert payload["subject"] == TICKET_ARGS["subject"]
    assert payload["eta_hours"] > 0


def test_ticket_ids_are_deterministic_per_request():
    first = call(create_ticket, TICKET_ARGS)["ticket_id"]
    second = call(create_ticket, TICKET_ARGS)["ticket_id"]
    other = call(create_ticket, {**TICKET_ARGS, "subject": "Sync stuck"})["ticket_id"]
    assert first == second != other


def test_create_ticket_error_fault():
    payload = call(create_ticket, TICKET_ARGS, {FAULT_TOOL_KEY: "error"})
    assert payload["status"] == "error"
    assert "ticket_id" not in payload


def test_create_ticket_timeout_fault():
    payload = call(
        create_ticket, TICKET_ARGS, {FAULT_TOOL_KEY: {"behavior": "timeout", "delay_seconds": 0.01}}
    )
    assert payload["error"] == "timeout"


def test_create_ticket_corrupted_result_fault():
    payload = call(create_ticket, TICKET_ARGS, {FAULT_TOOL_KEY: "corrupted_result"})
    assert payload["status"] == "created"
    assert payload["ticket_id"] == "NN-000000"
    assert payload["eta_hours"] == -1


def test_tool_fault_does_not_leak_into_the_retriever():
    payload = call(rag_search, {"query": "how do I request a refund"}, {FAULT_TOOL_KEY: "error"})
    assert payload["results"][0]["doc_id"] == "refund-policy"
