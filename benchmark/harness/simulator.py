"""The user-simulator boundary: persona + scenario -> the next user message.

docs/architecture/03-trace-harness.md II.B — an LLM loaded with the persona's
description and the scenario brief converses with the target app until the
scenario resolves (it emits `[DONE]`) or `max_turns` is reached.

The LLM sits behind `UserSimulator` so unit tests never make a network call
(docs/execution-plan.md ground rule 5). `ScriptedUserSimulator` is the test
double; `OpenAIUserSimulator` is the live implementation, pinned to
`benchmark.models.PERSONA_SIM_MODEL`.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from benchmark.models import PERSONA_SIM_MODEL
from benchmark.schemas.inputs import Persona

DONE_TOKEN = "[DONE]"
_DONE_RE = re.compile(re.escape(DONE_TOKEN), re.IGNORECASE)

# History is a list of (user_message_the_simulator_sent, app_response).
History = Sequence[tuple[str, str]]


def is_done(text: str) -> bool:
    """True when the simulator signalled the scenario is resolved.

    Token-based on purpose: "I am done shopping" is a perfectly ordinary user
    message and must not end the conversation.
    """
    return bool(_DONE_RE.search(text or ""))


def strip_done(text: str) -> str:
    return _DONE_RE.sub("", text or "").strip()


@runtime_checkable
class UserSimulator(Protocol):
    def next_message(
        self, *, persona: Persona, scenario: str, history: History, turn_index: int
    ) -> str: ...


class ScriptedUserSimulator:
    """Deterministic, network-free simulator for unit tests.

    Running out of script is termination — a test that forgets to append
    `[DONE]` must not loop to `max_turns` and quietly pass.
    """

    def __init__(self, messages: Sequence[str]):
        self.messages = list(messages)
        self.calls: list[dict[str, Any]] = []

    def next_message(
        self, *, persona: Persona, scenario: str, history: History, turn_index: int
    ) -> str:
        self.calls.append(
            {
                "persona": persona,
                "scenario": scenario,
                "history": list(history),
                "turn_index": turn_index,
            }
        )
        if turn_index < len(self.messages):
            return self.messages[turn_index]
        return DONE_TOKEN


class OpenAIUserSimulator:
    """Live user-simulator on `PERSONA_SIM_MODEL`.

    Standard library only (urllib), matching `OpenAIPromptExpander`: no extra
    project dependency, and the network is touched only when `next_message` is
    actually called — never at import or construction time.
    """

    _ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        model: str = PERSONA_SIM_MODEL,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout

    def system_prompt(self, persona: Persona, scenario: str) -> str:
        goals = "\n".join(f"- {goal}" for goal in persona.goals) or "- (none stated)"
        return (
            "You are role-playing a real end user talking to a product-support "
            "assistant through a chat widget. Stay in character at all times and "
            "write only what the user would type — never narrate, never explain "
            "that you are simulating.\n\n"
            f"WHO YOU ARE: {persona.name}\n{persona.description}\n\n"
            f"YOUR GOALS:\n{goals}\n\n"
            f"YOUR SITUATION:\n{scenario}\n\n"
            "Write ONE short message per turn, in the persona's voice. When your "
            f"goals are met, or it is clear the assistant cannot help, reply with "
            f"{DONE_TOKEN} (optionally after a brief closing line) and nothing else."
        )

    def chat_messages(self, persona: Persona, scenario: str, history: History) -> list[dict]:
        """The simulator's own chat history — it is the *assistant* of this chat.

        It plays the app's user, so its own past messages are `assistant` turns
        and the app's replies are `user` turns.
        """
        system = self.system_prompt(persona, scenario)
        messages: list[dict] = [{"role": "system", "content": system}]
        for user_message, app_response in history:
            messages.append({"role": "assistant", "content": user_message})
            messages.append({"role": "user", "content": app_response})
        if len(messages) == 1:
            messages.append(
                {"role": "user", "content": "(the assistant is waiting — open the conversation)"}
            )
        return messages

    def next_message(
        self, *, persona: Persona, scenario: str, history: History, turn_index: int
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set — required for OpenAIUserSimulator")
        body = json.dumps(
            {"model": self.model, "messages": self.chat_messages(persona, scenario, history)}
        ).encode()
        request = urllib.request.Request(
            self._ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            payload = json.loads(response.read())
        return payload["choices"][0]["message"]["content"].strip()
