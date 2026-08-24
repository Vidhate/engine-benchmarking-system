"""The ablation agent boundary — the one place an LLM enters Stage III.

docs/execution-plan.md ground rule 5: LLM-dependent behavior sits behind an
interface and is mocked in unit tests. `AblationAgent` is that interface, with
two call shapes:

* `propose` — draft `n` concrete, app-specific errors under one taxonomy
  category, grounded in a digest of *this app's real traces*, each already
  carrying the shape it will be injected with (a corruption for `replay_edit`,
  a `FaultConfig` for `dependency_fault`).
* `revise_corruption` — re-author a `replay_edit` corruption after step-3
  validation rejected it, with the reasons surfaced.

`ScriptedAblationAgent` is deterministic and network-free — the only agent unit
tests may use. `OpenAIAblationAgent` is the live implementation
(`ABLATION_AGENT_MODEL`), exercised solely by `scripts/ablation_smoke.py`.

## Why proposals carry their own injection payload

The doc splits step 1 (propose errors) from step 2 (plan ablations), and this
module respects that split — but the *creative* half of planning (what fabricated
sentence to put in the assistant's mouth) is the same act of authorship as
proposing the error, and needs the same corpus grounding. So the agent authors
it here and `plan.py` stays deterministic: filter assembly, target counts, and
the re-plan relaxations are code, which is what makes the validation loop
testable without a model.

## Self-correction is why corruptions carry a marker

The target app self-corrects: a replayed conversation may re-search the corpus
and contradict injected content. Corruptions must therefore be authored around
facts the corpus CANNOT refute — invented case/ticket references, fabricated
specifics absent from the doc store — never contradictions of retrievable
facts. `Corruption.marker` is the unfalsifiable token step-3 validation looks
for in `T*`, and `retraction_patterns` are the phrases that would mean the app
took it back downstream.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from benchmark.models import ABLATION_AGENT_MODEL
from benchmark.schemas.ablation import FaultConfig, FilterStep
from benchmark.schemas.issues import ErrorCategory, InjectionMode, Issue

# The generic dependency-fault vocabulary from
# docs/architecture/04-ablation-engine.md ("irrelevant/empty/stale docs",
# "tool error/timeout/corrupted result", "truncate/degrade the model"). This is
# the *proposal* vocabulary, not a claim about any app: whether a behavior does
# anything is settled by step-3 activation validation against the real app, and
# a behavior that does not activate is re-planned and then dropped.
BEHAVIOR_VOCABULARY: dict[str, tuple[str, ...]] = {
    "retriever": ("irrelevant_docs", "empty", "stale"),
    "tool": ("error", "timeout", "corrupted_result"),
    "llm_proxy": ("truncate_output",),
}

# Phrases that mean the app took an injected claim back. Checked against the
# REGENERATED tail only (docs/architecture/04-ablation-engine.md: the corrupted
# content must be present in T* and not explicitly retracted downstream).
DEFAULT_RETRACTION_PATTERNS: tuple[str, ...] = (
    "i apologi",
    "correction",
    "i was incorrect",
    "i made an error",
    "that was wrong",
    "disregard",
    "let me correct",
    "actually, the correct",
    "i misspoke",
    "does not exist",
    "no such",
    "i cannot find any record",
)


class Corruption(BaseModel):
    """The authored content of a `replay_edit` injection."""

    replacement: str  # the corrupted assistant response for the target turn
    marker: str  # the unfalsifiable token that proves the corruption is in T*
    retraction_patterns: list[str] = Field(default_factory=list)
    turn_index: int = 0  # which turn k to corrupt


class ProposedError(BaseModel):
    """One step-1 draft: the `Issue` plus everything its injection needs."""

    issue: Issue
    filter_steps: list[FilterStep] = Field(default_factory=list)
    corruption: Corruption | None = None  # replay_edit
    fault: FaultConfig | None = None  # dependency_fault
    target_count: int = 5


class CorpusDigest(BaseModel):
    """What the agent gets to see of the corpus, built from `TraceStore` reads."""

    n_traces: int = 0
    modes: dict[str, int] = Field(default_factory=dict)
    span_types: dict[str, int] = Field(default_factory=dict)
    span_names: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    sample_exchanges: list[dict[str, Any]] = Field(default_factory=list)
    available_shims: list[str] = Field(default_factory=list)
    app_context: str = ""


@runtime_checkable
class AblationAgent(Protocol):
    def propose(
        self,
        category: ErrorCategory,
        n: int,
        digest: CorpusDigest,
        allowed_modes: Sequence[InjectionMode],
    ) -> list[ProposedError]: ...

    def revise_corruption(
        self, proposal: ProposedError, digest: CorpusDigest, reasons: Sequence[str]
    ) -> Corruption: ...


# ------------------------------------------------------------------ scripted

class ScriptedAblationAgent:
    """Deterministic, network-free agent for unit tests and dry runs.

    Hands back whatever it was scripted with, keyed by `category_id`, and
    revises a corruption by appending a fresh marker suffix — enough for the
    re-plan loop to be exercised end to end without a model.
    """

    def __init__(self, proposals: dict[str, list[ProposedError]] | None = None):
        self.proposals = proposals or {}
        self.propose_calls: list[tuple[str, int]] = []
        self.revise_calls: list[tuple[str, list[str]]] = []
        self._revisions = 0

    def propose(
        self,
        category: ErrorCategory,
        n: int,
        digest: CorpusDigest,
        allowed_modes: Sequence[InjectionMode],
    ) -> list[ProposedError]:
        self.propose_calls.append((category.category_id, n))
        drafted = self.proposals.get(category.category_id, [])
        return [p for p in drafted if p.issue.injection_mode in allowed_modes][:n]

    def revise_corruption(
        self, proposal: ProposedError, digest: CorpusDigest, reasons: Sequence[str]
    ) -> Corruption:
        self.revise_calls.append((proposal.issue.error_id, list(reasons)))
        self._revisions += 1
        base = proposal.corruption
        if base is None:
            raise ValueError("revise_corruption called on a proposal with no corruption")
        marker = f"{base.marker}-R{self._revisions}"
        return Corruption(
            replacement=base.replacement.replace(base.marker, marker),
            marker=marker,
            retraction_patterns=list(base.retraction_patterns),
            turn_index=base.turn_index,
        )


# -------------------------------------------------------------------- OpenAI

_PROPOSE_SYSTEM = """\
You design *plausible, app-specific* failures to inject into an AI application's
traces, so a trace-analysis system can be benchmarked on finding them.

You are given: one error category, a digest of REAL traces from the app under
test, and the injection modes available. Draft concrete errors this app could
genuinely produce — never a generic error list.

Two injection modes:

* replay_edit — you author the corrupted assistant response for one turn, and
  the app organically regenerates everything after it. Author the corruption
  around facts the app's document corpus CANNOT refute: invented case or ticket
  references, fabricated internal specifics, numbers with no source. Do NOT
  contradict a fact the app can look up again — it will re-search and correct
  you, and the injection is then not in the trace. Every corruption needs a
  `marker`: a short, unique, invented literal string (like a case id) that
  appears verbatim in the replacement and could not appear by chance.
* dependency_fault — an external dependency is made to misbehave and the whole
  trace is regenerated with the fault active. Only mechanism-shaped errors:
  retrieval returning wrong/empty/stale documents, a tool erroring or timing
  out, a truncated model output.

Also draft a `filter`: 1-3 predicate steps that select traces where this error
could plausibly exist. Keep it LOOSE — an over-specific filter matches nothing
and the error gets dropped.

Return STRICT JSON only:
{"errors": [{
  "title": "...", "description": "...", "severity": "low|medium|high",
  "injection_mode": "replay_edit|dependency_fault",
  "filter_steps": [{"field": "...", "op": "eq|ne|contains|regex|gt|lt|exists",
                    "value": ...}],
  "corruption": {"replacement": "...", "marker": "...",
                 "retraction_patterns": ["..."]},
  "fault": {"shim": "retriever|tool|llm_proxy", "target": "...",
            "behavior": "...", "params": {}}
}]}
`corruption` is required for replay_edit and omitted for dependency_fault;
`fault` is the reverse. Never write the words fault_, shim, or ablation into
any user-visible text you author — that text ships inside the trace, and
naming the machinery would hand the answer away.

Filter fields you may use: status, mode, turn_count, span_count, span_types,
span_names, final_responses, user_messages, metadata.<key>,
turns[*].final_response, turns[*].user_message, turns[*].spans[*].span_type,
turns[*].spans[*].name.
"""

_REVISE_SYSTEM = """\
A corrupted assistant response you authored was rejected by validation. Rewrite
it so the rejection reasons no longer apply.

The most common reason is that the app SELF-CORRECTED: it re-searched its
document corpus and contradicted the injected claim downstream. The fix is
never to phrase the same contradiction more forcefully — it is to move the
fabrication onto ground the corpus cannot reach: an invented case/ticket
reference, an internal escalation id, a specific that simply is not in any
document.

Return STRICT JSON only:
{"replacement": "...", "marker": "...", "retraction_patterns": ["..."]}
"""


class OpenAIAblationAgent:
    """Live agent on `ABLATION_AGENT_MODEL`, via the chat completions API.

    Standard library only (urllib), and the network is reached only when a
    method is actually called — never at import or construction — so pytest
    collection never triggers a request.
    """

    _ENDPOINT = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        model: str = ABLATION_AGENT_MODEL,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout

    def _chat(self, system: str, user: str) -> dict:
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set — required for live OpenAIAblationAgent calls"
            )
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
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
        return json.loads(payload["choices"][0]["message"]["content"])

    @staticmethod
    def _digest_block(digest: CorpusDigest) -> str:
        return json.dumps(digest.model_dump(mode="json"), indent=2, default=str)[:12000]

    def propose(
        self,
        category: ErrorCategory,
        n: int,
        digest: CorpusDigest,
        allowed_modes: Sequence[InjectionMode],
    ) -> list[ProposedError]:
        user = (
            f"Error category: {category.name} ({category.category_id})\n"
            f"Category description: {category.description}\n"
            f"Injection modes available: {', '.join(allowed_modes)}\n"
            f"Dependency shims available: {', '.join(digest.available_shims) or 'none'}\n"
            f"Behavior vocabulary per shim: "
            f"{json.dumps({k: list(v) for k, v in BEHAVIOR_VOCABULARY.items()})}\n\n"
            f"Trace corpus digest:\n{self._digest_block(digest)}\n\n"
            f"Draft exactly {n} error(s) for this category."
        )
        payload = self._chat(_PROPOSE_SYSTEM, user)
        return [
            proposal
            for index, raw in enumerate(payload.get("errors", [])[:n])
            if (proposal := self._to_proposal(category, index, raw, allowed_modes)) is not None
        ]

    @staticmethod
    def _to_proposal(
        category: ErrorCategory,
        index: int,
        raw: dict,
        allowed_modes: Sequence[InjectionMode],
    ) -> ProposedError | None:
        mode = raw.get("injection_mode")
        if mode not in allowed_modes:
            return None
        issue = Issue(
            error_id=f"E-{category.category_id}-{index}",
            title=str(raw.get("title") or f"{category.name} error {index}"),
            description=str(raw.get("description") or ""),
            category_id=category.category_id,
            severity=raw.get("severity") if raw.get("severity") in
            ("low", "medium", "high") else "medium",
            injection_mode=mode,
        )
        steps = [
            FilterStep.model_validate(step)
            for step in (raw.get("filter_steps") or [])
            if isinstance(step, dict) and step.get("field") and step.get("op")
        ]
        corruption = None
        fault = None
        if mode == "replay_edit":
            body = raw.get("corruption") or {}
            if not body.get("replacement") or not body.get("marker"):
                return None
            corruption = Corruption(
                replacement=str(body["replacement"]),
                marker=str(body["marker"]),
                retraction_patterns=[str(p) for p in (body.get("retraction_patterns") or [])],
            )
        else:
            body = raw.get("fault") or {}
            if body.get("shim") not in BEHAVIOR_VOCABULARY or not body.get("behavior"):
                return None
            fault = FaultConfig(
                shim=body["shim"],
                target=str(body.get("target") or ""),
                behavior=str(body["behavior"]),
                params=dict(body.get("params") or {}),
            )
        return ProposedError(
            issue=issue, filter_steps=steps, corruption=corruption, fault=fault
        )

    def revise_corruption(
        self, proposal: ProposedError, digest: CorpusDigest, reasons: Sequence[str]
    ) -> Corruption:
        base = proposal.corruption
        if base is None:
            raise ValueError("revise_corruption called on a proposal with no corruption")
        user = (
            f"Error: {proposal.issue.title} — {proposal.issue.description}\n"
            f"Rejected replacement:\n{base.replacement}\n\n"
            f"Rejection reasons:\n- " + "\n- ".join(reasons) + "\n\n"
            f"Trace corpus digest:\n{self._digest_block(digest)}"
        )
        payload = self._chat(_REVISE_SYSTEM, user)
        replacement = str(payload.get("replacement") or "").strip()
        marker = str(payload.get("marker") or "").strip()
        if not replacement or not marker or marker not in replacement:
            raise ValueError(
                "revise_corruption returned no usable replacement/marker pair "
                "(the marker must appear verbatim in the replacement)"
            )
        return Corruption(
            replacement=replacement,
            marker=marker,
            retraction_patterns=[str(p) for p in (payload.get("retraction_patterns") or [])],
            turn_index=base.turn_index,
        )
