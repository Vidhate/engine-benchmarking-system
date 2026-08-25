"""The LLM-expander boundary: grid cell -> concrete text.

`PromptExpander` is the seam between the generation grid (Dimension/Persona
tuples) and the model that turns a grid cell into natural language. Two call
shapes:

- `expand`: single-turn grid cell -> concrete user prompt.
- `expand_scenario`: persona + grid cell -> multi-turn scenario brief.

`MockPromptExpander` is deterministic and network-free — it is the only
expander unit tests may use (see docs/execution-plan.md ground rule 5:
"No OpenAI calls in unit tests."). `OpenAIPromptExpander` is a live-model
skeleton exercised only by benchmark/generation/smoke.py.

## Transport: ChatOpenAI, and why it emits no LangSmith runs

The live expander used to speak raw `urllib` at the chat-completions endpoint.
A full-scale generation pass is ~400 sequential calls against a mid-tier model,
which is precisely the shape that trips a rate limit — and a 429 with no retry
budget does not slow the run down, it kills it partway through and leaves a
half-warmed cache. `ChatOpenAI` brings bounded retries with 429 backoff and a
real per-request timeout, so the transport is now its.

Importing `langchain_openai` here is an exception to the Phase-0 tracing
boundary (tests/test_tracing_boundary.py), granted to this file by name and to
`benchmark/ablation/agent.py`, on ONE condition: these calls must never produce
a LangSmith run. LangChain traces by default whenever `LANGSMITH_TRACING` is
set, and it is set — the harness needs it to collect the target app's traces. A
benchmark-side expansion landing in that project would pollute the collector's
own corpus with runs that are not traces of the app under test.

So every invocation goes through `_invoke_untraced`: `tracing_context(enabled
=False)` for the ambient LangSmith context, AND `callbacks: []` in the run
config so no inherited handler can re-attach one. The boundary test asserts
this wrapper is present in this file; do not inline a bare `.invoke()`.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from langchain_openai import ChatOpenAI
from langsmith import tracing_context

from benchmark.models import GENERATION_MODEL
from benchmark.schemas.inputs import Dimension, Persona


@runtime_checkable
class PromptExpander(Protocol):
    """Boundary Protocol implemented by both the mock and the OpenAI expander.

    `app_context` is the ONLY channel through which a target app's identity
    (domain, capabilities, vocabulary) reaches the expander — it is sourced
    from GenerationConfig.app_context (i.e. the yaml config), never
    hardcoded in the expander implementation. Empty app_context must still
    produce a sane, generic expansion.
    """

    def expand(self, dim: Dimension, variation: str, seed: int, app_context: str = "") -> str:
        """Turn one (dim, variation) grid cell into a concrete single-turn prompt."""
        ...

    def expand_scenario(
        self, persona: Persona, dim_id: str, variation: str, seed: int, app_context: str = ""
    ) -> str:
        """Turn a persona x (dim_id, variation) cell into a multi-turn scenario brief."""
        ...


class MockPromptExpander:
    """Deterministic, network-free expander used by all unit tests.

    Output is a pure function of its inputs (no randomness, no I/O), so
    identical (dim, variation, seed, app_context) or
    (persona, dim_id, variation, seed, app_context) tuples always produce
    byte-identical text.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def expand(self, dim: Dimension, variation: str, seed: int, app_context: str = "") -> str:
        self.calls.append(("expand", dim.dim_id, variation, seed, app_context))
        return (
            f"[mock:{dim.dim_id}:{variation}:{seed}] {dim.name} prompt for '{variation}' "
            f"(app_context={app_context!r})"
        )

    def expand_scenario(
        self, persona: Persona, dim_id: str, variation: str, seed: int, app_context: str = ""
    ) -> str:
        self.calls.append(
            ("expand_scenario", persona.persona_id, dim_id, variation, seed, app_context)
        )
        return (
            f"[mock:{persona.persona_id}:{dim_id}:{variation}:{seed}] "
            f"{persona.name} scenario for '{variation}' (app_context={app_context!r})"
        )


#: Passed as the run config on every invocation below. An empty callback list
#: is the second half of the tracing suppression: `tracing_context(enabled=
#: False)` turns off LangChain's own tracer, and this stops any handler a
#: caller installed from re-attaching one. See the module docstring.
NO_TRACING_CONFIG: dict[str, Any] = {"callbacks": []}


class OpenAIPromptExpander:
    """Live-model expander skeleton, backed by the OpenAI chat completions API.

    The model object is built lazily on first use, so the network is reached
    only when `.expand()`/`.expand_scenario()` are actually called — never at
    import or construction time — and pytest collection/import never triggers a
    request or requires a key. Exercised only by scripts/generation_smoke.py
    and a real generation run; never by CI.
    """

    def __init__(
        self,
        model: str = GENERATION_MODEL,
        api_key: str | None = None,
        # Was 30 s. A mid-tier reasoning model answering a 120-word expansion
        # can legitimately take longer than that, and a timeout fired at 30 s
        # bills for the call and throws the answer away.
        timeout: float = 120.0,
        # Rate limits are the concern, not flakiness: a generation pass is
        # hundreds of sequential calls, and a 429 with no retry budget ends the
        # run rather than slowing it. ChatOpenAI's backoff is exponential.
        max_retries: int = 4,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self._model: ChatOpenAI | None = None

    def _client(self) -> ChatOpenAI:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set — required for live OpenAIPromptExpander calls"
            )
        if self._model is None:
            self._model = ChatOpenAI(
                model=self.model,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
                # Chat Completions, and the same `temperature` the raw HTTP
                # body carried — this is a transport swap, and the request on
                # the wire is meant to be the one that was there before.
                # NOTE: langchain_openai drops `temperature` for gpt-5-family
                # models, which only accept the default. That is a fix, not a
                # regression: the old hand-rolled body sent it unconditionally
                # and the API rejected the request outright.
                use_responses_api=False,
                temperature=0.7,
            )
        return self._model

    def _invoke_untraced(self, system: str, user: str) -> str:
        """The ONLY call path to the model. Never emits a LangSmith run.

        Both suppressions are load-bearing; see the module docstring. If this
        is ever inlined into a bare `.invoke()`, tests/test_tracing_boundary.py
        fails, which is the point.
        """
        model = self._client()
        messages = [("system", system), ("user", user)]
        with tracing_context(enabled=False):
            reply = model.invoke(messages, config=NO_TRACING_CONFIG)
        return reply.content

    def _chat(self, system: str, user: str) -> str:
        return str(self._invoke_untraced(system, user)).strip()

    @staticmethod
    def _context_block(app_context: str) -> str:
        return app_context.strip() or "No further application context was provided."

    def expand(self, dim: Dimension, variation: str, seed: int, app_context: str = "") -> str:
        system = (
            "You generate one realistic single-turn user message for an AI "
            "application, given a description of that application and a grid "
            "cell (dimension + variation) it should instantiate. Return only "
            "the user message text, nothing else."
        )
        user = (
            f"Application context:\n{self._context_block(app_context)}\n\n"
            f"Dimension: {dim.name} ({dim.kind}).\n"
            f"Variation: {variation}\n"
            f"Seed: {seed}\n"
            "Write one concrete, natural user message for this grid cell, "
            "consistent with the application context above."
        )
        return self._chat(system, user)

    def expand_scenario(
        self, persona: Persona, dim_id: str, variation: str, seed: int, app_context: str = ""
    ) -> str:
        system = (
            "You write a short scenario brief for a user-simulator persona to "
            "follow across a multi-turn conversation with an AI application, "
            "given a description of that application. Return only the brief, "
            "nothing else."
        )
        user = (
            f"Application context:\n{self._context_block(app_context)}\n\n"
            f"Persona: {persona.name} — {persona.description}\n"
            f"Goals: {', '.join(persona.goals)}\n"
            f"Scenario dimension: {dim_id} / variation: {variation}\n"
            f"Seed: {seed}\n"
            "Write a 2-4 sentence scenario brief for a user-simulator to "
            "follow, consistent with the application context above."
        )
        return self._chat(system, user)
