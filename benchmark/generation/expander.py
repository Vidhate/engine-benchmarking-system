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
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol, runtime_checkable

from benchmark.models import GENERATION_MODEL
from benchmark.schemas.inputs import Dimension, Persona


@runtime_checkable
class PromptExpander(Protocol):
    """Boundary Protocol implemented by both the mock and the OpenAI expander."""

    def expand(self, dim: Dimension, variation: str, seed: int) -> str:
        """Turn one (dim, variation) grid cell into a concrete single-turn prompt."""
        ...

    def expand_scenario(self, persona: Persona, dim_id: str, variation: str, seed: int) -> str:
        """Turn a persona x (dim_id, variation) cell into a multi-turn scenario brief."""
        ...


class MockPromptExpander:
    """Deterministic, network-free expander used by all unit tests.

    Output is a pure function of its inputs (no randomness, no I/O), so
    identical (dim, variation, seed) or (persona, dim_id, variation, seed)
    tuples always produce byte-identical text.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def expand(self, dim: Dimension, variation: str, seed: int) -> str:
        self.calls.append(("expand", dim.dim_id, variation, seed))
        return f"[mock:{dim.dim_id}:{variation}:{seed}] {dim.name} prompt for '{variation}'"

    def expand_scenario(self, persona: Persona, dim_id: str, variation: str, seed: int) -> str:
        self.calls.append(("expand_scenario", persona.persona_id, dim_id, variation, seed))
        return (
            f"[mock:{persona.persona_id}:{dim_id}:{variation}:{seed}] "
            f"{persona.name} scenario for '{variation}'"
        )


class OpenAIPromptExpander:
    """Live-model expander skeleton, backed by the OpenAI chat completions API.

    Uses only the standard library (urllib) so no new project dependency is
    required to keep this skeleton importable. Reaches the network only when
    `.expand()`/`.expand_scenario()` are actually called — never at import or
    construction time — so pytest collection/import never triggers a request.
    Exercised only by benchmark/generation/smoke.py, which is not run in CI.
    """

    _ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        model: str = GENERATION_MODEL,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout

    def _chat(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set — required for live OpenAIPromptExpander calls"
            )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.7,
            }
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

    def expand(self, dim: Dimension, variation: str, seed: int) -> str:
        system = (
            "You write realistic single-turn user prompts for a product-support "
            "assistant. Return only the user message text, nothing else."
        )
        user = (
            f"Dimension: {dim.name} ({dim.kind}).\n"
            f"Variation: {variation}\n"
            f"Seed: {seed}\n"
            "Write one concrete, natural user message for this grid cell."
        )
        return self._chat(system, user)

    def expand_scenario(self, persona: Persona, dim_id: str, variation: str, seed: int) -> str:
        system = (
            "You write short scenario briefs describing what a user persona wants "
            "to accomplish across a multi-turn conversation with a product-support "
            "assistant. Return only the brief, nothing else."
        )
        user = (
            f"Persona: {persona.name} — {persona.description}\n"
            f"Goals: {', '.join(persona.goals)}\n"
            f"Scenario dimension: {dim_id} / variation: {variation}\n"
            f"Seed: {seed}\n"
            "Write a 2-4 sentence scenario brief for a user-simulator to follow."
        )
        return self._chat(system, user)
