"""The target AI app: a Nimbus Notes product-support assistant.

A `deepagents` agent on a small OpenAI model with exactly two tools, served
through `langgraph.json` (`langgraph dev`). Persistence comes from the
LangGraph server, so the graph is compiled without its own checkpointer —
that is what makes thread time-travel (Mode A) available to the benchmark.
"""

from __future__ import annotations

import os
from dataclasses import replace

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.config import get_config

from target_app.shims import FAULT_LLM_KEY, Fault, apply_llm_fault, read_fault
from target_app.tools import TOOLS

MODEL_ENV_VAR = "TARGET_APP_MODEL"
# NOTE: the plan asks for "gpt-5.1-mini"; this account's API returns 404 for
# that id, so the closest available small model is pinned instead. Override
# with TARGET_APP_MODEL.
DEFAULT_MODEL = "gpt-5-mini"
LANGSMITH_PROJECT = "engine-bench-target"

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

# Built-in deepagents scaffolding (filesystem, shell, subagents) is not part of
# a support assistant; excluding it keeps the model bound to exactly two tools.
_EXCLUDED_BUILTIN_TOOLS = frozenset(
    {"ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep", "execute", "task"}
)


def configure_tracing() -> None:
    """Send traces to the LangSmith project declared in configs/target_app.yaml."""
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)


def model_name() -> str:
    return os.environ.get(MODEL_ENV_VAR) or DEFAULT_MODEL


def build_model(name: str | None = None) -> ChatOpenAI:
    """Chat Completions (not the Responses API) so message content stays a plain string."""
    return ChatOpenAI(model=name or model_name(), use_responses_api=False)


class LlmFaultMiddleware(AgentMiddleware):
    """Mode C LLM shim: post-processes the model's own output.

    A `base_url` swap to a degrading proxy is the fuller version of this hook;
    truncation is implemented here as a pragmatic in-process equivalent that
    needs no second service.
    """

    name = "LlmFaultMiddleware"

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        response = handler(request)
        fault = _armed_llm_fault()
        return response if fault is None else _truncate_final_message(response, fault)

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        # The LangGraph server drives the graph asynchronously, so the async
        # twin is required — without it the run fails with NotImplementedError.
        response = await handler(request)
        fault = _armed_llm_fault()
        return response if fault is None else _truncate_final_message(response, fault)


def _armed_llm_fault() -> Fault | None:
    try:
        config = get_config()
    except RuntimeError:  # called outside a runnable context
        return None
    return read_fault(config, FAULT_LLM_KEY)


def _truncate_final_message(response: ModelResponse, fault: Fault) -> ModelResponse:
    """Truncate the assistant's answer. Messages carrying tool calls are left alone."""
    messages = list(response.result)
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, AIMessage) or message.tool_calls:
            continue
        original = message.content if isinstance(message.content, str) else message.text
        truncated = apply_llm_fault(fault, original)
        if truncated != original:
            messages[index] = message.model_copy(update={"content": truncated})
        break
    return replace(response, result=messages)


def build_graph():
    configure_tracing()
    register_harness_profile(
        "openai",
        HarnessProfile(
            excluded_tools=_EXCLUDED_BUILTIN_TOOLS,
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    return create_deep_agent(
        model=build_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[LlmFaultMiddleware()],
        name="target_app",
    )


graph = build_graph()
