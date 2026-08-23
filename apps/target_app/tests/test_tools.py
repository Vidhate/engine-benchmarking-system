"""The two tools, and the fault keys reaching them through config.configurable."""

import inspect
import json
import re

import pytest
from langsmith import Client
from langsmith.run_helpers import tracing_context
from langsmith.run_trees import RunTree

from target_app import tools
from target_app.shims import FAULT_RETRIEVER_KEY, FAULT_TOOL_KEY, read_fault
from target_app.tools import TOOLS, _retrieve, create_ticket, rag_search


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
        rag_search,
        {"query": "how do I request a refund"},
        {FAULT_RETRIEVER_KEY: {"behavior": "empty"}},
    )
    assert payload["results"] == []


def test_rag_search_irrelevant_docs_fault_swaps_the_documents():
    query = "how do I request a refund"
    normal = {r["doc_id"] for r in call(rag_search, {"query": query})["results"]}
    faulted = {
        r["doc_id"]
        for r in call(
            rag_search, {"query": query}, {FAULT_RETRIEVER_KEY: {"behavior": "irrelevant_docs"}}
        )["results"]
    }
    assert faulted and not (faulted & normal)


def test_rag_search_stale_fault_serves_the_outdated_revision():
    query = "how do I request a refund"
    payload = call(rag_search, {"query": query}, {FAULT_RETRIEVER_KEY: {"behavior": "stale"}})
    top = payload["results"][0]
    assert top["doc_id"] == "refund-policy"
    assert top["updated"].startswith("2019")
    assert "90 days" in top["content"]


def test_rag_search_rejects_an_undeclared_behavior():
    with pytest.raises(ValueError, match="unknown"):
        call(rag_search, {"query": "refund"}, {FAULT_RETRIEVER_KEY: {"behavior": "nonsense"}})


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
    payload = call(create_ticket, TICKET_ARGS, {FAULT_TOOL_KEY: {"behavior": "error"}})
    assert payload["status"] == "error"
    assert "ticket_id" not in payload


def test_create_ticket_timeout_fault():
    payload = call(
        create_ticket, TICKET_ARGS, {FAULT_TOOL_KEY: {"behavior": "timeout", "delay_seconds": 0.01}}
    )
    assert payload["error"] == "timeout"


def test_create_ticket_corrupted_result_fault():
    payload = call(create_ticket, TICKET_ARGS, {FAULT_TOOL_KEY: {"behavior": "corrupted_result"}})
    assert payload["status"] == "created"
    assert payload["ticket_id"] == "NN-000000"
    assert payload["eta_hours"] == -1


def test_tool_fault_does_not_leak_into_the_retriever():
    payload = call(
        rag_search, {"query": "how do I request a refund"}, {FAULT_TOOL_KEY: {"behavior": "error"}}
    )
    assert payload["results"][0]["doc_id"] == "refund-policy"


def test_retrieval_payload_carries_no_relevance_score():
    """Scores would separate `irrelevant_docs` from organic misses statistically."""
    for doc in call(rag_search, {"query": "how do I request a refund"})["results"]:
        assert set(doc) == {"doc_id", "title", "updated", "content"}


# ------------------------------------------------------ what the span records

class _OfflineClient(Client):
    """A LangSmith client that records nothing anywhere — no network in tests."""

    def __init__(self):
        super().__init__(
            api_key="unit-test", api_url="http://localhost:1", auto_batch_tracing=False
        )

    def create_run(self, *args, **kwargs):
        return None

    def update_run(self, *args, **kwargs):
        return None


QUERY = "how do I request a refund"


def recorded_retrieval_span(configurable: dict) -> RunTree:
    """Trace the retrieval step exactly as rag_search invokes it, and capture the run."""
    fault = read_fault({"configurable": configurable}, FAULT_RETRIEVER_KEY)
    captured: list[RunTree] = []
    client = _OfflineClient()
    token = tools._ARMED_RETRIEVER_FAULT.set(fault)
    try:
        with tracing_context(enabled=True):
            _retrieve(
                QUERY,
                tools.DEFAULT_K,
                langsmith_extra={"on_end": captured.append, "client": client},
            )
    finally:
        tools._ARMED_RETRIEVER_FAULT.reset(token)
    spans = [run for run in captured if run.run_type == "retriever"]
    assert spans, "no retriever span was recorded"
    return spans[-1]


def test_the_tool_hands_the_fault_over_out_of_band(monkeypatch):
    """rag_search must arm the context var, never pass the fault as an argument."""
    seen: dict = {}

    def spy(query, k):
        seen["args"] = (query, k)
        seen["fault"] = tools._ARMED_RETRIEVER_FAULT.get()
        return []

    monkeypatch.setattr(tools, "_retrieve", spy)
    rag_search.invoke(
        {"query": QUERY},
        config={"configurable": {FAULT_RETRIEVER_KEY: {"behavior": "stale"}}},
    )
    assert seen["args"] == (QUERY, tools.DEFAULT_K)
    assert seen["fault"].behavior == "stale"
    # and the var is restored afterwards, so one run cannot bleed into the next
    assert tools._ARMED_RETRIEVER_FAULT.get() is None


# Structural giveaways. Behaviour words alone are not listed: legitimate corpus
# text contains e.g. "Emptying the Trash".
FORBIDDEN_TOKENS = (
    "fault",
    "shim",
    "ablat",
    "inject",
    "behavior",
    "irrelevant_docs",
    "corrupted_result",
    "truncate_output",
)


@pytest.mark.parametrize(
    "configurable",
    [
        {},
        {FAULT_RETRIEVER_KEY: {"behavior": "empty"}},
        {FAULT_RETRIEVER_KEY: {"behavior": "stale"}},
        {FAULT_RETRIEVER_KEY: {"behavior": "irrelevant_docs"}},
    ],
)
def test_retrieval_span_never_records_the_armed_fault(configurable):
    span = recorded_retrieval_span(configurable)
    blob = json.dumps(
        {"inputs": span.inputs, "outputs": span.outputs, "extra": span.extra}, default=str
    ).lower()
    leaked = [token for token in FORBIDDEN_TOKENS if token in blob]
    assert not leaked, f"retriever span leaked {leaked}: {blob[:400]}"


def test_the_retrieval_span_only_ever_takes_query_and_k():
    """Structural guard: nothing fault-shaped can become a traced argument."""
    assert set(inspect.signature(_retrieve.__wrapped__).parameters) == {"query", "k"}


def test_armed_faults_still_reach_the_traced_retrieval():
    """The leak test above would pass trivially if arming had stopped working."""
    empty = recorded_retrieval_span({FAULT_RETRIEVER_KEY: {"behavior": "empty"}})
    stale = recorded_retrieval_span({FAULT_RETRIEVER_KEY: {"behavior": "stale"}})
    assert empty.outputs["output"] == []
    assert stale.outputs["output"][0]["updated"] == "2019-03-02"
