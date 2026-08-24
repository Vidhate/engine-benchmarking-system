"""A scriptable stand-in for the chat model.

`analyze_trace` and `consolidate` take the model as an argument precisely so
this can be substituted: the tool loop, the structured-output handling, the
clustering merge and the seed merge are then all exercised with zero network.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeChatModel(BaseChatModel):
    """Replays a scripted list of AIMessages; `with_structured_output` returns
    the next scripted structured object.

    Attributes are plain lists so a test can assert on what the loop did:
    `calls` records every message list the model was invoked with, and
    `bound_tools` records the tool schemas bound to it.
    """

    responses: list[AIMessage] = []
    structured: list[Any] = []
    calls: list[list[BaseMessage]] = []
    bound_tools: list[Any] = []

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        response = self.responses.pop(0) if self.responses else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools, **kwargs):
        self.bound_tools.extend(tools)
        return self

    def with_structured_output(self, schema, **kwargs):
        return _StructuredFake(self)


class _StructuredFake:
    def __init__(self, parent: FakeChatModel) -> None:
        self._parent = parent

    def invoke(self, messages, **kwargs):
        self._parent.calls.append(list(messages))
        if not self._parent.structured:
            raise AssertionError("FakeChatModel: no scripted structured output left")
        return self._parent.structured.pop(0)


def tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )
