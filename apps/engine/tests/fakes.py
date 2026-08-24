"""A scriptable stand-in for the chat model.

`analyze_trace` and `consolidate` take the model as an argument precisely so
this can be substituted: the tool loop, the structured-output handling, the
clustering merge and the seed merge are then all exercised with zero network.
"""

from __future__ import annotations

import threading
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

    Thread-safe, because the analysis pass runs batches of traces concurrently.
    Script *order* is still only meaningful under `analysis_concurrency=1`; for
    concurrent tests set `router`, which picks the structured result from the
    prompt's content instead of from a queue position.
    """

    responses: list[AIMessage] = []
    structured: list[Any] = []
    calls: list[list[BaseMessage]] = []
    bound_tools: list[Any] = []
    # Optional callable(messages) -> structured object. Content-addressed, so a
    # concurrent test does not depend on which worker pops first.
    router: Any = None

    model_config = {"arbitrary_types_allowed": True}

    _lock: Any = None

    @property
    def lock(self) -> threading.Lock:
        if self._lock is None:
            self._lock = threading.Lock()
        return self._lock

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        with self.lock:
            self.calls.append(list(messages))
            response = self.responses.pop(0) if self.responses else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools, **kwargs):
        with self.lock:
            self.bound_tools.extend(tools)
        return self

    def with_structured_output(self, schema, **kwargs):
        return _StructuredFake(self)


class _StructuredFake:
    def __init__(self, parent: FakeChatModel) -> None:
        self._parent = parent

    def invoke(self, messages, **kwargs):
        parent = self._parent
        with parent.lock:
            parent.calls.append(list(messages))
            if parent.router is not None:
                return parent.router(messages)
            if not parent.structured:
                raise AssertionError("FakeChatModel: no scripted structured output left")
            return parent.structured.pop(0)


def tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )
