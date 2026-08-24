"""The two injection mechanics: Mode A (`replay_edit`) and Mode C (`dependency_fault`).

Everything here goes through the Phase-4 public API (`Harness.replay`,
`Harness.run_with_faults`, `Harness.turn_boundaries`) — no LangGraph or
LangSmith type crosses this file.

## Where to fork: an id, never a position

`Harness.turn_boundaries` returns one entry per *checkpoint* whose newest
message is a plain answer, and the live server writes **several checkpoints per
answer** — so trace turn k is boundary 2k, and a previous Mode-A replay forks
the same thread and appends its regenerated answers to the same history. The
boundary list is therefore not the trace's turn space, and
`locate_checkpoint(..., turn_index=k)` is only correct at k = 0 by accident.
`fork_point` resolves the fork from the checkpoint id the collector recorded
for that turn (`metadata.turn_checkpoints`), falling back to the answer text
and refusing an ambiguous match.

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

The check covers *all* regenerated content, not just the tail's final
responses: each regenerated turn's answer **and** every span output under it,
because the app can contradict an injected claim inside a retrieval or tool
span while the answer itself says nothing about it.

### Residual risk of the not-retracted check (accepted for v1)

It is a phrase matcher, so it is neither complete nor free:

* **False negatives it cannot see.** A paraphrased contradiction ("the record
  I'm looking at shows something different"), a span-level contradiction
  phrased in none of these words, and *silent supersession* — the app simply
  answers as if the injected claim was never made — all pass. Consequence: an
  `E_K` entry survives on a trace that no longer really carries the error.
  The bias is one-way and benign for scoring: it **understates** Engine's
  precision, and never invents an error Engine is punished for missing. It is
  the same direction as the `E_h` bias already documented in
  `docs/architecture/04-ablation-engine.md` ("precision as a lower bound").
* **False positives cost real corpus.** A phrase that fires on healthy English
  burns a validated spec's candidate and can drop an error the corpus could
  have carried. That is why the unconditional patterns are all first-person
  and self-referential, and the ambiguous ones ("no such", "does not exist")
  only count within `ANCHOR_WINDOW` characters of the injected marker's own
  distinctive terms (see `agent.py`).
* **It is measured, not guessed.** Every burned candidate is counted per error
  and reported out as `ApplyOutcome.self_corrected` ->
  `AblationResult.self_corrected_counts`, so the size of the surface this check
  consumes is a number in the run report rather than an unknown.

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
import random
import re
from collections.abc import Iterable, Sequence
from typing import Any

from benchmark.ablation.agent import (
    ANCHOR_WINDOW,
    ANCHORED_RETRACTION_PATTERNS,
    DEFAULT_RETRACTION_PATTERNS,
)
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


def _recorded_checkpoint(trace: Trace, turn_index: int) -> str | None:
    """The checkpoint the collector recorded for turn k, if it recorded one."""
    for entry in trace.metadata.get("turn_checkpoints") or []:
        if isinstance(entry, dict) and entry.get("turn_index") == turn_index:
            value = entry.get("checkpoint_id")
            return value if isinstance(value, str) and value else None
    return None


def fork_point(
    trace: Trace, turn_index: int, harness: Any, response_text: str = ""
) -> tuple[str, str]:
    """`(checkpoint_id, message_id)` to fork turn k at.

    **Not** `locate_checkpoint(..., turn_index=k)`. That `turn_index` indexes
    the *thread's* boundary list, which is not the trace's turn space, as the
    live run showed:

    * the server writes **several checkpoints per answer**, so one answer
      appears 2+ times in `turn_boundaries` — trace turn k is boundary 2k;
    * a previous Mode-A replay **forks the same thread**, so its regenerated
      answers are appended to the same history.

    Only k = 0 survived that by accident, which is why it stayed hidden while
    every corruption was pinned to turn 0. The collector already recorded which
    checkpoint each turn ended at (`metadata.turn_checkpoints`), so that **id**,
    never a position, is the key. Text is only the fallback, and it is refused
    as ambiguous unless every match is the same assistant message.
    """
    thread_id = thread_ref(trace)
    if thread_id is None:
        raise InjectionError(f"trace {trace.trace_id} has no thread ref to fork")
    boundaries = harness.turn_boundaries(thread_id)
    recorded = _recorded_checkpoint(trace, turn_index)
    if recorded is not None:
        for checkpoint_id, message_id, _text in boundaries:
            if checkpoint_id == recorded:
                return checkpoint_id, message_id
        log.warning(
            "trace %s records checkpoint %s for turn %d, but the thread no longer "
            "has it — falling back to matching on the answer text",
            trace.trace_id,
            recorded,
            turn_index,
        )
    wanted = (response_text or "").strip()
    matches = [(c, m) for c, m, text in boundaries if wanted and text == wanted]
    if not matches:
        raise InjectionError(
            f"no checkpoint on thread {thread_id} is turn {turn_index} of trace "
            f"{trace.trace_id} (recorded checkpoint {recorded!r}, "
            f"{len(boundaries)} boundary/boundaries on the thread)"
        )
    if len({message_id for _c, message_id in matches}) > 1:
        raise InjectionError(
            f"turn {turn_index} of trace {trace.trace_id} matches {len(matches)} different "
            f"assistant messages on thread {thread_id}; forking at the wrong one would "
            f"mislabel the ground-truth turn index"
        )
    return matches[0]


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
    """The llm span whose output IS the turn's final response.

    **Assumption**: the last-finishing llm call is the one that produced the
    answer. That holds for the target app (a `create_react_agent` loop, where
    tool calls are earlier llm spans and the answer is the final one) and for
    any agent whose last model call emits the reply. It would NOT hold for an
    architecture that runs a post-hoc critic/guardrail model after the answer,
    or that generates the answer and then summarizes it — there the true
    producing span is the second-to-last. Ordering is by `(end_time, span_id)`,
    so the span_id tiebreak keeps the choice deterministic when two llm spans
    share an end time (equal timestamps are common at collector resolution).
    """
    llm = [s for s in turn.spans if s.span_type == "llm"]
    return max(llm, key=lambda s: (s.end_time, s.span_id)) if llm else None


def _root_span(turn: Turn):
    """The turn's root agent span — the one with no surviving parent.

    Only the ROOT is rewritten for consistency. A nested agent-type span is a
    sub-agent's own record, and its output legitimately differs from the final
    answer; rewriting all of them would manufacture agreement the app never
    produced, which is its own tell.
    """
    agents = [s for s in turn.spans if s.span_type == "agent"]
    if not agents:
        return None
    ids = {s.span_id for s in turn.spans}
    roots = [s for s in agents if s.parent_span_id is None or s.parent_span_id not in ids]
    return min(roots or agents, key=lambda s: (s.start_time, s.span_id))


def valid_turn_indices(trace: Trace) -> list[int]:
    """Turns that can carry a Mode-A corruption (they have an llm span)."""
    return [i for i, turn in enumerate(trace.turns) if any(
        s.span_type == "llm" for s in turn.spans
    )]


def choose_turn_index(
    trace: Trace, requested: int | None, rng: random.Random
) -> int:
    """Which turn k to corrupt.

    An agent-supplied `requested` wins if it names a turn that can actually
    carry the corruption; otherwise k is drawn seeded-randomly from the valid
    turns. Always corrupting turn 0 would make "the injected turn" a constant
    across the whole ablated corpus — a positional tell, and a much weaker test
    of Engine (early-turn errors are the easiest kind to spot).
    """
    valid = valid_turn_indices(trace)
    if not valid:
        raise ConsistencyError(
            f"trace {trace.trace_id} has no turn with an llm span, so no turn can be "
            f"corrupted consistently"
        )
    if requested is not None:
        if requested not in valid:
            raise InjectionError(
                f"turn_index {requested} is not a corruptible turn of trace "
                f"{trace.trace_id} (valid: {valid})"
            )
        return requested
    return rng.choice(valid)


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
    llm_span.attributes = _rescale_attributes(llm_span.attributes, original_llm_text, replacement)
    rewritten = [llm_span.span_id]

    root = _root_span(corrupted)
    if root is not None:
        root.outputs = _rewrite_agent_output(root.outputs, replacement)
        rewritten.append(root.span_id)
    return corrupted, rewritten, original_llm_text


def _rescale_attributes(
    attributes: dict[str, Any], before: str, after: str
) -> dict[str, Any]:
    """Keep the rewritten llm span's `tokens` consistent with its new output.

    A span whose completion was replaced but whose token count still describes
    the *original* text is a statistical tell: an Engine (or a reviewer) can
    regress token count on output length and find every ablated span as an
    outlier. The count is scaled by the length ratio rather than dropped —
    dropping it would make "no token count" the tell instead, since every
    organic llm span has one.

    `duration_ms` is deliberately left alone: it measures how long the model
    call took, which really did happen, and the corruption does not claim
    otherwise.
    """
    tokens = attributes.get("tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or not before:
        return attributes
    ratio = len(after) / max(len(before), 1)
    out = dict(attributes)
    out["tokens"] = max(1, round(tokens * ratio))
    return out


# ------------------------------------------------------ self-correction check

# Words too common to anchor on. A marker like "case NBX-4471" anchors on
# "nbx-4471", not on "case".
_STOPWORDS = frozenset(
    "the a an and or of to in on for is are was were be been this that with case "
    "ref reference id number your our you we it its as at by from".split()
)


def marker_terms(marker: str) -> list[str]:
    """The distinctive tokens of a marker, for anchoring a retraction phrase."""
    tokens = re.findall(r"[a-z0-9][a-z0-9\-_/]*", marker.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2] or [marker.lower()]


def regenerated_text(turns: Iterable[Turn]) -> list[str]:
    """Every piece of app-authored text in the regenerated tail.

    Widened after review from "final responses only". The app can retract an
    injected claim without saying so in the answer — it re-searches the corpus
    and the contradiction shows up in a retrieval or tool span, or a later llm
    span reasons about it — so the scan covers each turn's final response AND
    its spans' outputs.
    """
    out: list[str] = []
    for turn in turns:
        out.append(turn.final_response)
        for span in turn.spans:
            if span.outputs:
                out.append(json.dumps(span.outputs, default=str))
    return [text.lower() for text in out if text]


def retraction_in(
    turns: Iterable[Turn], patterns: Sequence[str], marker: str = ""
) -> str | None:
    """The first retraction phrase in the regenerated content, or None.

    Two families (see `agent.py`): unconditional first-person phrases, and
    phrases that only count within `ANCHOR_WINDOW` characters of one of the
    marker's own distinctive terms.

    **Known false negatives, accepted for v1** (see the module docstring's
    "Residual risk"): a paraphrased contradiction, a span-level contradiction
    that uses none of these words, and silent supersession (the app simply
    answers differently and never mentions the injected claim) all pass this
    check. The bias direction is one-way and benign for scoring: a missed
    retraction means an `E_K` entry whose trace no longer really carries the
    error, which *understates* Engine's precision. It never invents an error
    Engine is then punished for missing.
    """
    haystacks = regenerated_text(turns)
    for pattern in (*DEFAULT_RETRACTION_PATTERNS, *(p.lower() for p in patterns)):
        needle = pattern.lower().strip()
        if needle and any(needle in text for text in haystacks):
            return pattern

    terms = marker_terms(marker) if marker else []
    if not terms:
        return None
    for pattern in ANCHORED_RETRACTION_PATTERNS:
        for text in haystacks:
            for match in re.finditer(re.escape(pattern), text):
                window = text[
                    max(0, match.start() - ANCHOR_WINDOW) : match.end() + ANCHOR_WINDOW
                ]
                if any(term in window for term in terms):
                    return pattern
    return None


# --------------------------------------------------------------------- Mode A

def _renumber(turns: Sequence[Turn], start: int) -> list[Turn]:
    return [turn.model_copy(deep=True, update={"turn_index": start + i}) for i, turn in
            enumerate(turns)]


def _marker_present(turn: Turn, marker: str) -> bool:
    """Is `marker` in this turn's own content, comparing real strings?

    Span payloads are walked as decoded values rather than serialized, so the
    comparison is against the text the app would actually show — a marker with
    a quote in it matches itself instead of its JSON escape.
    """
    needle = marker.lower()

    def walk(node: Any) -> bool:
        if isinstance(node, str):
            return needle in node.lower()
        if isinstance(node, dict):
            return any(walk(v) for v in node.values())
        if isinstance(node, (list, tuple)):
            return any(walk(v) for v in node)
        return False

    if needle in turn.final_response.lower():
        return True
    return any(walk(span.outputs) for span in turn.spans)


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
    seed: int = 0,
) -> tuple[Trace, AblationRecord]:
    """Corrupt turn k, make its spans consistent, regenerate `k+1 … M` organically."""
    if not spec.ablation_actions:
        raise InjectionError(f"{spec.error_id}: replay_edit spec has no ablation_actions")
    action = spec.ablation_actions[0]
    params = action.params
    replacement = str(params.get("replacement") or "")
    marker = str(params.get("marker") or "")
    patterns = [str(p) for p in (params.get("retraction_patterns") or [])]
    if not replacement:
        raise InjectionError(f"{spec.error_id}: replay_edit action has no replacement text")

    requested = params.get("turn_index")
    # Seeded per (trace, ablation): the same trace always draws the same k, so
    # a rerun is reproducible, but k varies across the corpus.
    rng = random.Random(f"{seed}\x1f{trace.trace_id}\x1f{ablation_id}")
    turn_index = choose_turn_index(
        trace, int(requested) if isinstance(requested, int) else None, rng
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
        checkpoint_id, message_id = fork_point(
            trace, turn_index, harness, original_response
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

    # Scoped to turn k, and matched against UNESCAPED strings. Searching the
    # whole trace's raw JSON would pass on a marker that merely echoed in a
    # regenerated tail turn while turn k itself lost it, and a marker
    # containing a quote or a backslash would be JSON-escaped in the dump and
    # never match its own literal.
    if marker and not _marker_present(ablated.turns[turn_index], marker):
        raise CorruptionLost(
            f"{spec.error_id}: marker {marker!r} is not present in turn {turn_index} of the "
            f"assembled T* for {trace.trace_id} — the corruption did not survive the splice"
        )
    retraction = retraction_in(tail, patterns, marker)
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
        mode="replay_edit",
        actions_applied=[
            action.model_copy(
                deep=True,
                update={
                    "params": {
                        **params,
                        # The turn actually corrupted, which for an unpinned
                        # corruption is drawn per trace — callers read k back
                        # from here rather than from the spec.
                        "turn_index": turn_index,
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
        mode="dependency_fault",
        actions_applied=[],
        # ("", evidence) is the documented shape for dependency_fault: there is
        # no "before" to diff in the record, the fault config IS the action.
        before_after=[("", evidence)],
    )
    return trace, record
