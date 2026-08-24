"""The `langgraph_sdk` adapter — the only channel to the target app.

docs/execution-plan.md, "The black-box contract": the benchmark drives the app
exclusively through `langgraph_sdk` against a URL from `configs/target_app.yaml`
and never imports app code. Everything this module knows about the app comes
from a `TargetAppConfig`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from benchmark.schemas.configs import TargetAppConfig


@dataclass
class AppResponse:
    """One turn's result, as the harness sees it.

    `error` is set for genuine app failures (5xx, timeout, graph error). Those
    are organic signal, not collector bugs — the trace is still kept, with
    `status="app_error"` (docs/architecture/03-trace-harness.md).
    """

    final_response: str
    messages: list[dict] = field(default_factory=list)
    error: str | None = None
    checkpoint_id: str | None = None
    thread_id: str = ""


@runtime_checkable
class TargetAppClient(Protocol):
    """The adapter surface the runner and the Phase-5 APIs depend on."""

    def create_thread(self) -> str: ...

    def invoke(
        self,
        thread_id: str,
        message: str,
        *,
        session_id: str,
        turn_index: int = 0,
        configurable: dict[str, Any] | None = None,
        checkpoint: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AppResponse: ...

    def get_state(self, thread_id: str) -> dict: ...

    def get_history(self, thread_id: str, limit: int = 100) -> list[dict]: ...

    def update_state(
        self, thread_id: str, values: Any, *, checkpoint: dict[str, Any] | None = None
    ) -> dict: ...


def message_text(message: Any) -> str:
    """Flatten a LangChain message's content to plain text."""
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def final_answer(messages: list[dict]) -> str:
    """The last assistant message that is an answer, not a tool call."""
    for message in reversed(messages or []):
        role = message.get("type") or message.get("role")
        if role in ("ai", "assistant") and not message.get("tool_calls"):
            return message_text(message)
    return ""


class LangGraphAppClient:
    """`TargetAppClient` over a LangGraph Server (`langgraph dev`)."""

    def __init__(
        self,
        cfg: TargetAppConfig,
        client: Any = None,
        *,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
        record_checkpoints: bool = True,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cfg = cfg
        self._client = client
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.record_checkpoints = record_checkpoints
        self._sleep = sleep

    @property
    def client(self) -> Any:
        if self._client is None:  # imported lazily: keeps unit tests network-free
            from langgraph_sdk import get_sync_client  # noqa: PLC0415

            self._client = get_sync_client(url=self.cfg.base_url)
        return self._client

    # ------------------------------------------------------------------ threads

    def create_thread(self) -> str:
        return self.client.threads.create()["thread_id"]

    def get_state(self, thread_id: str) -> dict:
        return self.client.threads.get_state(thread_id)

    def get_history(self, thread_id: str, limit: int = 100) -> list[dict]:
        return list(self.client.threads.get_history(thread_id, limit=limit))

    def update_state(
        self, thread_id: str, values: Any, *, checkpoint: dict[str, Any] | None = None
    ) -> dict:
        """Time-travel fork: rewrite state at `checkpoint`, returning the fork ref."""
        response = self.client.threads.update_state(
            thread_id, values=values, checkpoint=checkpoint
        )
        return response["checkpoint"]

    def current_checkpoint_id(self, thread_id: str) -> str | None:
        checkpoint = (self.get_state(thread_id) or {}).get("checkpoint") or {}
        return checkpoint.get("checkpoint_id")

    # ------------------------------------------------------------------- invoke

    def invoke(
        self,
        thread_id: str,
        message: str,
        *,
        session_id: str,
        turn_index: int = 0,
        configurable: dict[str, Any] | None = None,
        checkpoint: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AppResponse:
        run_metadata = {
            "session_id": session_id,
            "turn_index": turn_index,
            **(metadata or {}),
        }
        # Fault values are ALWAYS mappings — langchain promotes str/int/float/bool
        # `configurable` entries into inheritable tracing metadata, which would
        # stamp the fault name onto every span of the run (apps/target_app/README.md).
        config = {"configurable": dict(configurable)} if configurable else None

        last_error: str | None = None
        run: Any = None
        for attempt in range(self.max_retries + 1):
            try:
                run = self.client.runs.wait(
                    thread_id,
                    self.cfg.assistant_id,
                    input={"messages": [{"role": "user", "content": message}]},
                    config=config,
                    metadata=run_metadata,
                    checkpoint=checkpoint,
                )
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - any transport failure is retryable
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_retries:
                    self._sleep(self.retry_backoff_s * (2**attempt))

        if last_error is not None:
            return AppResponse("", [], error=last_error, thread_id=thread_id)

        messages = (run or {}).get("messages") or []
        if isinstance(run, dict) and "__error__" in run:
            return AppResponse(
                final_answer(messages),
                messages,
                error=str(run["__error__"]),
                thread_id=thread_id,
            )

        checkpoint_id = self.current_checkpoint_id(thread_id) if self.record_checkpoints else None
        return AppResponse(
            final_answer(messages), messages, checkpoint_id=checkpoint_id, thread_id=thread_id
        )
