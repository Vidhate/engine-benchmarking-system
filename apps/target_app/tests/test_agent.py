"""Graph wiring, the LLM shim, and the tool allowlist — model mocked, no network."""

import asyncio

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from target_app import agent as agent_module
from target_app.agent import (
    ALLOWED_TOOL_NAMES,
    DEFAULT_MODEL,
    LLM_SPAN_NAME,
    MODEL_ENV_VAR,
    SupportChatModel,
    _degrade,
    model_name,
)
from target_app.shims import FAULT_LLM_KEY

ANSWER = (
    "Nimbus Notes refunds any paid plan within thirty days of the charge date. "
    "Open Settings, then Billing, then Request refund. "
    "The money lands back on your card in five to seven business days."
)
ARMED = {"configurable": {FAULT_LLM_KEY: {"behavior": "truncate_output"}}}


def chat_result(message: AIMessage) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=message)])


# ------------------------------------------------------------------ model id

def test_model_defaults_to_the_pinned_small_model(monkeypatch):
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    assert model_name() == DEFAULT_MODEL == "gpt-5-mini"


def test_model_is_overridable_by_env(monkeypatch):
    monkeypatch.setenv(MODEL_ENV_VAR, "gpt-4.1-mini")
    assert model_name() == "gpt-4.1-mini"


def test_the_served_model_is_the_shimmed_subclass():
    model = agent_module.build_model()
    assert isinstance(model, SupportChatModel)
    # the subclass must not be visible in the trace
    assert model.name == LLM_SPAN_NAME == "ChatOpenAI"


# ------------------------------------------- truncation inside the model call

@pytest.fixture
def armed(monkeypatch):
    """Pretend we are inside a graph run with fault_llm armed."""
    monkeypatch.setattr(agent_module, "get_config", lambda: ARMED)


@pytest.fixture
def unarmed(monkeypatch):
    monkeypatch.setattr(agent_module, "get_config", lambda: {"configurable": {}})


def test_degrade_truncates_the_generation_itself(armed):
    result = _degrade(chat_result(AIMessage(content=ANSWER)))
    text = result.generations[-1].message.content
    assert len(text) < len(ANSWER)
    assert ANSWER.startswith(text)


def test_degrade_is_a_no_op_when_unarmed(unarmed):
    result = _degrade(chat_result(AIMessage(content=ANSWER)))
    assert result.generations[-1].message.content == ANSWER


def test_degrade_outside_a_graph_run_is_a_no_op(monkeypatch):
    def outside():
        raise RuntimeError("not in a graph")

    monkeypatch.setattr(agent_module, "get_config", outside)
    result = _degrade(chat_result(AIMessage(content=ANSWER)))
    assert result.generations[-1].message.content == ANSWER


def test_degrade_skips_generations_that_carry_tool_calls(armed):
    message = AIMessage(
        content="looking that up",
        tool_calls=[{"name": "rag_search", "args": {"query": "refund"}, "id": "call_1"}],
    )
    assert _degrade(chat_result(message)).generations[-1].message.content == "looking that up"


def test_the_llm_span_sees_the_truncated_text(monkeypatch, armed):
    """`_generate` is what the LLM span records, so it must already be degraded."""
    monkeypatch.setattr(
        ChatOpenAI, "_generate", lambda self, m, stop=None, run_manager=None, **kw: chat_result(
            AIMessage(content=ANSWER)
        )
    )
    model = SupportChatModel(model="gpt-5-mini")
    recorded = model._generate([])
    assert recorded.generations[-1].message.content != ANSWER
    assert ANSWER.startswith(recorded.generations[-1].message.content)


# ------------------------------------------------------- graph, model mocked

@pytest.fixture
def mocked_llm(monkeypatch):
    """Replace the OpenAI call with a canned answer and record bound tools."""
    bound: list[list[str]] = []
    original_bind = ChatOpenAI.bind_tools
    scripted: list[AIMessage] = []

    def spy_bind_tools(self, tools, **kwargs):
        bound.append([getattr(t, "name", str(t)) for t in tools])
        return original_bind(self, tools, **kwargs)

    def fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = scripted.pop(0) if scripted else AIMessage(content=ANSWER)
        return chat_result(message)

    async def fake_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return fake_generate(self, messages)

    monkeypatch.setattr(ChatOpenAI, "bind_tools", spy_bind_tools)
    monkeypatch.setattr(ChatOpenAI, "_generate", fake_generate)
    monkeypatch.setattr(ChatOpenAI, "_agenerate", fake_agenerate)
    return {"bound": bound, "script": scripted}


ASK = {"messages": [{"role": "user", "content": "how do refunds work?"}]}


def test_graph_binds_exactly_the_two_declared_tools(mocked_llm):
    graph = agent_module.build_graph()
    graph.invoke(ASK)
    assert mocked_llm["bound"] == [["rag_search", "create_ticket"]]
    assert ALLOWED_TOOL_NAMES == {"rag_search", "create_ticket"}


def test_unarmed_run_returns_the_full_answer(mocked_llm):
    state = agent_module.build_graph().invoke(ASK)
    assert state["messages"][-1].content == ANSWER


def test_llm_fault_truncates_the_answer_when_armed_via_configurable(mocked_llm):
    state = agent_module.build_graph().invoke(ASK, config=ARMED)
    text = state["messages"][-1].content
    assert text != ANSWER
    assert ANSWER.startswith(text)


def test_llm_fault_truncates_on_the_async_path_too(mocked_llm):
    """The LangGraph server drives the graph with ainvoke."""
    state = asyncio.run(agent_module.build_graph().ainvoke(ASK, config=ARMED))
    text = state["messages"][-1].content
    assert text != ANSWER
    assert ANSWER.startswith(text)


def test_async_run_without_faults_is_unaffected(mocked_llm):
    state = asyncio.run(agent_module.build_graph().ainvoke(ASK))
    assert state["messages"][-1].content == ANSWER


# --------------------------------------------- dispatchable surface == declared

def tool_registry(graph) -> set[str]:
    """Every tool name the graph's ToolNode is able to dispatch."""
    for step in graph.nodes["tools"].node.steps:
        registry = getattr(step, "tools_by_name", None) or getattr(step, "_tools_by_name", None)
        if registry:
            return set(registry)
    raise AssertionError("no ToolNode registry found on the graph")


def test_the_graph_can_dispatch_nothing_but_the_two_declared_tools(mocked_llm):
    """The dispatchable surface must equal the declared surface, exactly."""
    assert tool_registry(agent_module.build_graph()) == {"rag_search", "create_ticket"}
    assert ALLOWED_TOOL_NAMES == {"rag_search", "create_ticket"}


def fabricate(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_x"}])


@pytest.mark.parametrize("blocked", ["execute", "write_file", "read_file", "task", "glob"])
def test_a_fabricated_call_to_an_unregistered_tool_cannot_run(mocked_llm, blocked, tmp_path):
    """Regression guard: Mode A rewrites assistant messages, so a fabricated tool
    call is a real input. It must never reach a shell or the filesystem."""
    victim = tmp_path / "pwned.txt"
    mocked_llm["script"].extend(
        [
            fabricate(
                blocked,
                {
                    "command": f"touch {victim}",
                    "file_path": str(victim),
                    "content": "x",
                    "pattern": "*",
                    "description": "x",
                    "subagent_type": "general-purpose",
                },
            ),
            AIMessage(content=ANSWER),
        ]
    )
    state = agent_module.build_graph().invoke(ASK)
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages, f"{blocked} produced no tool message at all"
    assert tool_messages[-1].status == "error", f"{blocked} was not rejected"
    assert not victim.exists(), f"{blocked} actually ran"


def test_the_declared_tools_still_dispatch(mocked_llm):
    mocked_llm["script"].extend(
        [fabricate("rag_search", {"query": "refund policy"}), AIMessage(content=ANSWER)]
    )
    state = agent_module.build_graph().invoke(ASK)
    tool_messages = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages and tool_messages[0].status != "error"
    assert "refund-policy" in tool_messages[0].content
