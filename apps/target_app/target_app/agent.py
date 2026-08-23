"""The target AI app: a Nimbus Notes product-support assistant.

A `create_react_agent` graph on a small OpenAI model with exactly two tools,
served through `langgraph.json` (`langgraph dev`). Persistence comes from the
LangGraph server, so the graph is compiled without its own checkpointer —
that is what makes thread time-travel (Mode A) available to the benchmark.

The prebuilt is deliberately minimal: it registers exactly the tools it is
handed, which keeps the app's dispatchable surface equal to its declared
surface. (An earlier revision used `deepagents`, whose built-in filesystem and
shell tools stayed registered on the ToolNode even when hidden from the model —
a fabricated tool call in a Mode A replay could have reached them.)
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langgraph.config import get_config
from langgraph.prebuilt import create_react_agent

from target_app.shims import FAULT_LLM_KEY, Fault, apply_llm_fault, read_fault
from target_app.tools import TOOLS

ALLOWED_TOOL_NAMES = frozenset(tool.name for tool in TOOLS)

MODEL_ENV_VAR = "TARGET_APP_MODEL"
# NOTE: the plan asks for "gpt-5.1-mini"; this account's API returns 404 for
# that id, so the closest available small model is pinned instead. Override
# with TARGET_APP_MODEL.
DEFAULT_MODEL = "gpt-5-mini"
LANGSMITH_PROJECT = "engine-bench-target"
LLM_SPAN_NAME = "ChatOpenAI"

SYSTEM_PROMPT = """\
You are the Nimbus Notes support assistant. Nimbus Notes is a cloud
note-taking workspace.

How to work:
1. For ANY question about Nimbus Notes features, pricing, policies, platforms,
   or troubleshooting, call `rag_search` first. Never answer such a question
   from memory.
2. Ground your answer in the returned documents and name the document titles
   you used.
3. If `rag_search` returns no documents, or the documents do not cover the
   question, say plainly that you could not find it in the knowledge base.
   Do not invent policy details, prices, or dates.
4. Call `create_ticket` only when the user asks for a human, a refund, or an
   account change that support must perform, or when the knowledge base
   cannot resolve their problem. Report the ticket id back to the user, and
   if ticket creation fails, tell the user it failed.
5. Be concise: a short answer plus the concrete steps. Two tool calls per
   turn is normally plenty.
"""

def configure_tracing() -> None:
    """Send traces to the LangSmith project declared in configs/target_app.yaml."""
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)


def model_name() -> str:
    return os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL


def build_model(name: str | None = None) -> ChatOpenAI:
    """Chat Completions (not the Responses API) so message content stays a plain string."""
    # `name` pins the LangSmith run name. The subclass must be indistinguishable
    # from a plain ChatOpenAI in the trace, or the class name is itself the tell
    # that says "this app was built for fault injection".
    return SupportChatModel(
        model=name or model_name(), use_responses_api=False, name=LLM_SPAN_NAME
    )


def _armed_fault(key: str) -> Fault | None:
    try:
        config = get_config()
    except RuntimeError:  # called outside a graph run
        return None
    return read_fault(config, key)


class SupportChatModel(ChatOpenAI):
    """Mode C LLM shim, applied *inside* the model call.

    A `base_url` swap to a degrading proxy is the fuller version of this hook;
    truncation is the pragmatic in-process equivalent, needing no second
    service. It deliberately happens here rather than in middleware so the LLM
    span itself records the degraded generation. Truncating after the span
    closed would leave "final answer != last llm span output" in every armed
    trace — a harness tell an Engine could learn instead of reading the trace.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return _degrade(super()._generate(messages, stop, run_manager, **kwargs))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return _degrade(await super()._agenerate(messages, stop, run_manager, **kwargs))


def _degrade(result: ChatResult) -> ChatResult:
    """Apply the armed LLM fault to a completion. Tool calls are left alone."""
    fault = _armed_fault(FAULT_LLM_KEY)
    if fault is None or not result.generations:
        return result
    generations = list(result.generations)
    generation = generations[-1]
    message = generation.message
    if not isinstance(message, AIMessage) or message.tool_calls:
        return result
    original = message.content if isinstance(message.content, str) else message.text
    truncated = apply_llm_fault(fault, original)
    if truncated == original:
        return result
    generations[-1] = replace_generation(generation, truncated)
    return ChatResult(generations=generations, llm_output=result.llm_output)


def replace_generation(generation: ChatGeneration, text: str) -> ChatGeneration:
    return ChatGeneration(
        message=generation.message.model_copy(update={"content": text}),
        generation_info=generation.generation_info,
    )


def build_graph():
    """The served graph. No checkpointer: the LangGraph server provides one."""
    configure_tracing()
    return create_react_agent(
        model=build_model(),
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
        name="target_app",
    )


graph = build_graph()
