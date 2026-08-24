"""The two injection mechanics: Mode A (`replay_edit`) and Mode C (`dependency_fault`).

Everything here goes through the Phase-4 public API (`Harness.replay`,
`Harness.run_with_faults`, `Harness.locate_checkpoint`) — no LangGraph or
LangSmith type crosses this file.

## Mode A — replay_edit, and what the harness actually returns

`Harness.replay` returns the **regenerated turns only** (`k+1 … M`). This
module splices:

    T* = turns[0 … k-1] + corrupted turn k + regenerated tail

and rewrites turn *k*'s internal spans so the trace stays causally consistent:
the last llm span's output and the agent span's final assistant message must
equal the corrupted response. Leaving them alone would make
"final response != last llm span output" a fingerprint of every ablated trace —
a tell an Engine could learn instead of reading the trace.

**M = 1 is the degenerate case and the bulk of the corpus.** There is no
downstream to regenerate, so Mode A reduces to a consistency-managed post-hoc
edit; `Harness.replay` *refuses* an empty `remaining_user_messages` precisely so
that this path is taken deliberately rather than by accident.

## The app self-corrects — corrupt what the corpus cannot refute

Observed live: a replayed conversation may re-search the corpus and contradict
injected content. So corruptions are authored around invented case/ticket
references and fabricated specifics absent from the doc store (see
`agent.py`), and `retraction_in` checks the **regenerated tail** for an explicit
correction. A retraction means the injection did not survive into `T*`'s
downstream and the error must be re-planned, not recorded as ground truth.

## Threads are server-lifetime state

Replay forks a thread that must still exist in the LangGraph server's store.
A corpus collected in an earlier server lifetime has dead thread refs, and the
symptom (every replay failing to locate a checkpoint) is unhelpfully far from
the cause — so `assert_threads_alive` probes up front and fails loudly.

## Mode C — dependency_fault

`Harness.run_with_faults` is always called with the **baseline** (the unarmed
trace for the same input), which makes activation a byte-diff of the span the
fault must corrupt; `weak_validation` would accept "the dependency ran at all",
which a disarmed run also passes. Activation evidence arrives out of band on
`harness.activation_evidence[trace_id]` — never inside the Trace, which would
hand Engine the ground truth — and is stored as `AblationRecord.before_after`
`("", evidence)`. Ground truth is **mechanism-level**: there is no manifestation
verifier, and the final answer is never inspected.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Iterable, Sequence
from typing import Any

from benchmark.ablation.agent import DEFAULT_RETRACTION_PATTERNS
from benchmark.harness.ids import session_id_for, trace_id_for
from benchmark.schemas.ablation import AblationRecord, AblationSpec
from benchmark.schemas.inputs import InputSpec, Persona
from benchmark.schemas.traces import Trace, Turn

log = logging.getLogger("benchmark.ablation")


class InjectionError(Exception):
    """An injection could not be carried out on this trace."""


class DeadThreadRefs(InjectionError):
    """The source traces' threads are gone from the server — replay is impossible."""


class SelfCorrected(InjectionError):
    """The app retracted the injected content downstream; `T*` does not carry it."""


class CorruptionLost(InjectionError):
    """The corrupted content is not present in the assembled `T*`."""


class ConsistencyError(InjectionError):
    """Turn k has no span to make consistent with the corrupted response."""


# ------------------------------------------------------------ thread liveness

def thread_ref(trace: Trace) -> str | None:
    value = trace.metadata.get("thread_id")
    return value if isinstance(value, str) and value else None


def thread_alive(harness: Any, thread_id: str) -> bool:
    """True when the LangGraph server still has this thread's checkpoints."""
    try:
        return bool(harness.turn_boundaries(thread_id))
    except Exception as exc:  # noqa: BLE001 - any transport/lookup failure means "gone"
        log.debug("thread %s looks dead: %s: %s", thread_id, type(exc).__name__, exc)
        return False


def live_threads(traces: Sequence[Trace], harness: Any) -> set[str]:
    """The subset of `traces` whose thread the server can still fork."""
    alive: set[str] = set()
    checked: dict[str, bool] = {}
    for trace in traces:
        ref = thread_ref(trace)
        if ref is None:
            continue
        if ref not in checked:
            checked[ref] = thread_alive(harness, ref)
        if checked[ref]:
            alive.add(trace.trace_id)
    return alive


def assert_threads_alive(traces: Sequence[Trace], harness: Any) -> set[str]:
    """Probe thread liveness up front and fail loudly if nothing can be replayed."""
    alive = live_threads(traces, harness)
    if not alive and traces:
        raise DeadThreadRefs(
            f"none of the {len(traces)} candidate traces has a thread the LangGraph "
            f"server still holds. Threads are SERVER-LIFETIME state: a corpus "
            f"collected under an earlier `langgraph dev` process cannot be replayed. "
            f"Re-run the harness and the ablation engine within one server lifetime "
            f"(scripts/ablation_smoke.py does exactly that)."
        )
    if len(alive) < len(traces):
        log.warning(
            "%d/%d candidate traces have a live thread; the rest are not "
            "replay_edit-eligible in this server lifetime",
            len(alive),
            len(traces),
        )
    return alive


# ------------------------------------------------------- turn-k consistency

def _last_llm_span(turn: Turn):
    llm = [s for s in turn.spans if s.span_type == "llm"]
    return max(llm, key=lambda s: (s.end_time, s.span_id)) if llm else None


def _rewrite_llm_output(outputs: dict[str, Any], replacement: str) -> dict[str, Any]:
    """Put `replacement` where this llm span records its completion.

    The collector's allowlist copies one of `generations` / `message` /
    `messages` / `output` (benchmark/harness/collector.py `_OUTPUT_KEYS`), so
    the shape is rewritten in place rather than replaced wholesale — a span
    that suddenly changed its output *shape* would be as much of a tell as one
    whose text disagreed with the answer.
    """
    out = copy.deepcopy(outputs)
    generations = out.get("generations")
    if isinstance(generations, list) and generations:
        last_batch = generations[-1]
        if isinstance(last_batch, list) and last_batch and isinstance(last_batch[-1], dict):
            entry = last_batch[-1]
            if "text" in entry:
                entry["text"] = replacement
            message = entry.get("message")
            if isinstance(message, dict) and "content" in message:
                message["content"] = replacement
            if "text" not in entry and not isinstance(message, dict):
                entry["text"] = replacement
            return out
    message = out.get("message")
    if isinstance(message, dict):
        message["content"] = replacement
        return out
    messages = out.get("messages")
    if isinstance(messages, list) and messages:
        for entry in reversed(messages):
            if isinstance(entry, dict) and (entry.get("type") or entry.get("role")) in (
                "ai",
                "assistant",
            ):
                entry["content"] = replacement
                return out
    if "output" in out:
        if isinstance(out["output"], dict) and "content" in out["output"]:
            out["output"]["content"] = replacement
        else:
            out["output"] = replacement
        return out
    out["output"] = replacement
    return out


def _rewrite_agent_output(outputs: dict[str, Any], replacement: str) -> dict[str, Any]:
    out = copy.deepcopy(outputs)
    messages = out.get("messages")
    if isinstance(messages, list):
        for entry in reversed(messages):
            if isinstance(entry, dict) and (entry.get("type") or entry.get("role")) in (
                "ai",
                "assistant",
            ):
                entry["content"] = replacement
                break
    if isinstance(out.get("output"), str):
        out["output"] = replacement
    return out


def corrupt_turn(turn: Turn, replacement: str) -> tuple[Turn, list[str], str]:
    """Turn k with its response corrupted and its internal spans made consistent.

    Returns `(turn, rewritten_span_ids, original_llm_text)`.
    """
    corrupted = turn.model_copy(deep=True)
    corrupted.final_response = replacement

    llm_span = _last_llm_span(corrupted)
    if llm_span is None:
        raise ConsistencyError(
            f"turn {turn.turn_index} has no llm span, so the corrupted response cannot "
            f"be made consistent with the model call that produced it"
        )
    original_llm_text = json.dumps(llm_span.outputs, sort_keys=True, default=str)
    llm_span.outputs = _rewrite_llm_output(llm_span.outputs, replacement)
    rewritten = [llm_span.span_id]

    for span in corrupted.spans:
        if span.span_type == "agent":
            span.outputs = _rewrite_agent_output(span.outputs, replacement)
            rewritten.append(span.span_id)
    return corrupted, rewritten, original_llm_text


# ------------------------------------------------------ self-correction check

def retraction_in(turns: Iterable[Turn], patterns: Sequence[str]) -> str | None:
    """The first retraction phrase found in these turns' responses, or None."""
    haystacks = [turn.final_response.lower() for turn in turns]
    for pattern in (*DEFAULT_RETRACTION_PATTERNS, *(p.lower() for p in patterns)):
        needle = pattern.lower().strip()
        if needle and any(needle in text for text in haystacks):
            return pattern
    return None


# --------------------------------------------------------------------- Mode A

def _renumber(turns: Sequence[Turn], start: int) -> list[Turn]:
    return [turn.model_copy(deep=True, update={"turn_index": start + i}) for i, turn in
            enumerate(turns)]


def ablated_trace_id(dataset_id: str, input_id: str, ablation_id: str) -> str:
    """A fresh, deterministic trace id for an ablated trace.

    Same scheme as the harness's own ids (`benchmark/harness/ids.py`), so an
    ablated trace's identity is derived the same way as an organic one rather
    than being a decorated copy of its parent's id — the decoration itself
    would be a leak.
    """
    return trace_id_for(session_id_for(dataset_id, input_id, variant=f"ablate:{ablation_id}"))


def apply_replay_edit(
    trace: Trace,
    spec: AblationSpec,
    harness: Any,
    *,
    ablation_id: str,
    dataset_id: str = "",
    store_result: bool = True,
) -> tuple[Trace, AblationRecord]:
    """Corrupt turn k, make its spans consistent, regenerate `k+1 … M` organically."""
    if not spec.ablation_actions:
        raise InjectionError(f"{spec.error_id}: replay_edit spec has no ablation_actions")
    action = spec.ablation_actions[0]
    params = action.params
    replacement = str(params.get("replacement") or "")
    marker = str(params.get("marker") or "")
    turn_index = int(params.get("turn_index") or 0)
    patterns = [str(p) for p in (params.get("retraction_patterns") or [])]
    if not replacement:
        raise InjectionError(f"{spec.error_id}: replay_edit action has no replacement text")
    if not 0 <= turn_index < len(trace.turns):
        raise InjectionError(
            f"{spec.error_id}: turn_index {turn_index} is outside trace {trace.trace_id}'s "
            f"{len(trace.turns)} turn(s)"
        )

    source_turn = trace.turns[turn_index]
    original_response = source_turn.final_response
    corrupted, rewritten_spans, original_llm_text = corrupt_turn(source_turn, replacement)

    remaining = [turn.user_message for turn in trace.turns[turn_index + 1 :]]
    tail: list[Turn] = []
    replay_metadata: dict[str, Any] = {}
    if remaining:
        thread_id = thread_ref(trace)
        if thread_id is None:
            raise InjectionError(
                f"{spec.error_id}: trace {trace.trace_id} has no thread ref, so turns "
                f"{turn_index + 1}..{len(trace.turns) - 1} cannot be regenerated"
            )
        checkpoint_id, message_id = harness.locate_checkpoint(
            thread_id, original_response, turn_index=turn_index
        )
        regenerated = harness.replay(
            thread_id,
            checkpoint_id,
            {"messages": [{"role": "ai", "id": message_id, "content": replacement}]},
            remaining,
            input_id=trace.input_id,
            dataset_id=dataset_id,
            store_result=False,
        )
        tail = _renumber(regenerated.turns, turn_index + 1)
        replay_metadata = {
            "replay_source_checkpoint_id": regenerated.metadata.get("source_checkpoint_id"),
            "replay_fork_checkpoint_id": regenerated.metadata.get("fork_checkpoint_id"),
            "thread_id": regenerated.metadata.get("thread_id", thread_id),
        }

    turns = [t.model_copy(deep=True) for t in trace.turns[:turn_index]] + [corrupted] + tail
    ablated = Trace(
        trace_id=ablated_trace_id(dataset_id, trace.input_id, ablation_id),
        input_id=trace.input_id,
        mode="multi_turn" if len(turns) > 1 else "single_turn",
        turns=turns,
        status=trace.status,
        metadata={
            **trace.metadata,
            **replay_metadata,
            "turn_count": len(turns),
            # Ground-truth-side bookkeeping; stripped by the Engine export.
            "ablation_parent_trace_id": trace.trace_id,
        },
        ablation_ids=[*trace.ablation_ids, ablation_id],
    )

    if marker and marker.lower() not in ablated.model_dump_json().lower():
        raise CorruptionLost(
            f"{spec.error_id}: marker {marker!r} is not present in the assembled T* for "
            f"{trace.trace_id} — the corruption did not survive the splice"
        )
    retraction = retraction_in(tail, patterns)
    if retraction is not None:
        raise SelfCorrected(
            f"{spec.error_id}: the app retracted the injected content downstream of turn "
            f"{turn_index} (matched {retraction!r}); corrupt something the corpus cannot "
            f"refute instead"
        )

    record = AblationRecord(
        ablation_id=ablation_id,
        error_id=spec.error_id,
        trace_id=ablated.trace_id,
        actions_applied=[
            action.model_copy(
                deep=True,
                update={
                    "params": {
                        **params,
                        "rewritten_span_ids": rewritten_spans,
                        "regenerated_turns": len(tail),
                        "source_trace_id": trace.trace_id,
                    }
                },
            )
        ],
        before_after=[(original_response, replacement), (original_llm_text, replacement)],
    )
    if store_result:
        harness.store.put(ablated)
    return ablated, record


# --------------------------------------------------------------------- Mode C

def apply_dependency_fault(
    input_spec: InputSpec,
    spec: AblationSpec,
    harness: Any,
    *,
    baseline: Trace,
    ablation_id: str,
    dataset_id: str = "",
    persona: Persona | None = None,
    max_turns: int = 1,
    store_result: bool = True,
) -> tuple[Trace, AblationRecord]:
    """Arm the declared fault, re-run the input, record the activation evidence.

    Always passes `baseline`: activation is a byte-diff against the unarmed
    trace for the same input, never the weak "the dependency ran" form.
    """
    if spec.fault_config is None:
        raise InjectionError(f"{spec.error_id}: dependency_fault spec has no fault_config")
    # store_result=False on the way in: the trace is persisted once, below,
    # after its ablation bookkeeping is attached — otherwise the store would
    # briefly hold a version of the trace that does not know it was ablated.
    trace = harness.run_with_faults(
        input_spec,
        spec.fault_config,
        dataset_id=dataset_id,
        baseline=baseline,
        persona=persona,
        max_turns=max_turns,
        store_result=False,
    )
    evidence = harness.activation_evidence.get(trace.trace_id, "")
    if not evidence:
        raise InjectionError(
            f"{spec.error_id}: run_with_faults published no activation evidence for "
            f"{trace.trace_id} — without it the record cannot prove the mechanism fault"
        )
    trace.ablation_ids = [*trace.ablation_ids, ablation_id]
    trace.metadata = {**trace.metadata, "ablation_parent_trace_id": baseline.trace_id}
    if store_result:
        harness.store.put(trace)

    record = AblationRecord(
        ablation_id=ablation_id,
        error_id=spec.error_id,
        trace_id=trace.trace_id,
        actions_applied=[],
        # ("", evidence) is the documented shape for dependency_fault: there is
        # no "before" to diff in the record, the fault config IS the action.
        before_after=[("", evidence)],
    )
    return trace, record
