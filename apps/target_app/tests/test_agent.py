"""Graph wiring and the LLM shim, with the model mocked (no network)."""

import asyncio
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from target_app import agent as agent_module
from target_app.agent import DEFAULT_MODEL, MODEL_ENV_VAR, _truncate_final_message, model_name
from target_app.shims import FAULT_LLM_KEY, Fault

ANSWER = (
    "Nimbus Notes refunds any paid plan within thirty days of the charge date. "
    "Open Settings, then Billing, then Request refund. "
    "The money lands back on your card in five to seven business days."
)


# ------------------------------------------------------------------ model id

def test_model_defaults_to_the_pinned_small_model(monkeypatch):
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    assert model_name() == DEFAULT_MODEL == "gpt-5-mini"


def test_model_is_overridable_by_env(monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "gpt-4.1-mini")
    assert model_name() == "gpt-4.1-mini"


# ------------------------------------------------------------- truncation unit

class _Response:
    """Stand-in for ModelResponse (a dataclass with `result`)."""

    def __init__(self, result):
        self.result = result


def test_truncate_targets_the_final_answer(monkeypatch):
    monkeypatch.setattr(agent_module, "replace", lambda resp, result: _Response(result))
    response = _Response([AIMessage(content=ANSWER)])
    out = _truncate_final_message(response, Fault("truncate_output", {}))
    text = out.result[0].content
    assert len(text) < len(ANSWER)
    assert ANSWER.startswith(text)


def test_truncate_skips_messages_that_carry_tool_calls(monkeypatch):
    monkeypatch.setattr(agent_module, "replace", lambda resp, result: _Response(result))
    tool_call_message = AIMessage(
        content="looking that up",
        tool_calls=[{"name": "rag_search", "args": {"query": "refund"}, "id": "call_1"}],
    )
    out = _truncate_final_message(_Response([tool_call_message]), Fault("truncate_output", {}))
    assert out.result[0].content == "looking that up"


# ------------------------------------------------------- graph, model mocked

@pytest.fixture
def mocked_llm(monkeypatch):
    """Replace the OpenAI call with a canned answer and record bound tools."""
    bound: list[list[str]] = []
    original_bind = ChatOpenAI.bind_tools

    def spy_bind_tools(self, tools, **kwargs):
        bound.append([getattr(t, "name", str(t)) for t in tools])
        return original_bind(self, tools, **kwargs)

    def fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=ANSWER))])

    async def fake_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return fake_generate(self, messages)

    monkeypatch.setattr(ChatOpenAI, "bind_tools", spy_bind_tools)
    monkeypatch.setattr(ChatOpenAI, "_generate", fake_generate)
    monkeypatch.setattr(ChatOpenAI, "_agenerate", fake_agenerate)
    return bound


def test_graph_binds_exactly_the_two_declared_tools(mocked_llm):
    graph = agent_module.build_graph()
    graph.invoke({"messages": [{"role": "user", "content": "how do refunds work?"}]})
    assert mocked_llm == [["rag_search", "create_ticket"]]


def test_unarmed_run_returns_the_full_answer(mocked_llm):
    graph = agent_module.build_graph()
    state = graph.invoke({"messages": [{"role": "user", "content": "how do refunds work?"}]})
    assert state["messages"][-1].content == ANSWER


def test_llm_fault_truncates_the_answer_when_armed_via_configurable(mocked_llm):
    graph = agent_module.build_graph()
    state = graph.invoke(
        {"messages": [{"role": "user", "content": "how do refunds work?"}]},
        config={"configurable": {FAULT_LLM_KEY: "truncate_output"}},
    )
    text = state["messages"][-1].content
    assert text != ANSWER
    assert ANSWER.startswith(text)
    assert len(text) < len(ANSWER)


def test_llm_fault_truncates_on_the_async_path_too(mocked_llm):
    """The LangGraph server drives the graph with ainvoke, so the async hook must exist."""
    graph = agent_module.build_graph()
    state = asyncio.run(
        graph.ainvoke(
            {"messages": [{"role": "user", "content": "how do refunds work?"}]},
            config={"configurable": {FAULT_LLM_KEY: "truncate_output"}},
        )
    )
    text = state["messages"][-1].content
    assert text != ANSWER
    assert ANSWER.startswith(text)


def test_async_run_without_faults_is_unaffected(mocked_llm):
    graph = agent_module.build_graph()
    state = asyncio.run(
        graph.ainvoke({"messages": [{"role": "user", "content": "how do refunds work?"}]})
    )
    assert state["messages"][-1].content == ANSWER


def test_replace_is_the_dataclasses_helper():
    """Guard the monkeypatch above against drifting away from the real import."""
    assert agent_module.replace is replace
