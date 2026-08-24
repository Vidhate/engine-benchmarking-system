"""LangSmith run trees -> our `Trace` schema. The only LangSmith-aware module.

Everything south of here reads the Phase-0 `TraceStore` (our schema, local
JSON), so replacing LangSmith later replaces this file and nothing else
(docs/execution-plan.md, "Tracing backend").

Three things this module is deliberately paranoid about:

1. **Allowlist copying.** A `Span` is *built*, never *cast*, from a LangSmith
   run: a fixed set of keys per span type for inputs/outputs, and a fixed set
   of five attributes. `run.extra`, `run.tags` and `run.serialized` are read
   for exactly one allowlisted value (`extra.metadata.ls_model_name`) and
   otherwise never copied — they are the channel that carries `configurable`
   echoes of an armed fault onto every span of a run.
2. **Explicit projection + anti-vacuity guards.** Runs are fetched with an
   explicit `select=`, and a field that must be populated but comes back
   `None` raises `VacuousProjectionError`. Phase 2 learned this the hard way:
   `list_runs` does not project `serialized` by default, so a `getattr`
   fallback silently audited nothing.
3. **Bounded, loud waiting.** LangSmith ingests child spans asynchronously and
   can lag the root by ~30s. The collector polls until the run tree settles
   and raises `IngestionTimeout` rather than normalizing a half-ingested tree
   into a quietly truncated Trace.

## Span filter rule (framework noise)

The corpus is meant to carry app semantics, not LangChain plumbing. A run is
kept when:

* it is the root run (recorded as the `agent` span), **or**
* its `run_type` is `llm`, `tool` or `retriever` (always meaningful), **or**
* its `run_type` is `chain`/`prompt`/`parser` **and** its name does not match
  `NOISE_NAME_PATTERNS` (the framework's own wrappers: Runnable*, Channel*,
  `__start__`/`__end__`, middleware `*.wrap_model_call`, output parsers,
  conditional-edge predicates).

Any other run type is dropped. When a span is dropped its surviving children
are re-parented onto the nearest surviving ancestor, so the tree stays
connected and no `parent_span_id` dangles.

Known limitation: the rule matches on names, so a *meaningful* graph node an
app happens to name `route`, `should_continue` or `_write` would be filtered
as plumbing. The pattern list is the place to fix that per app; nothing in the
target app's graph currently collides.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from benchmark.harness.ids import trace_id_for
from benchmark.harness.scrub import assert_no_leak, leak_tokens
from benchmark.schemas.configs import TargetAppConfig
from benchmark.schemas.traces import Span, Trace, TraceMode, TraceStatus, Turn

# Every field the normalizer reads. Asked for explicitly, then guarded.
SPAN_SELECT: list[str] = [
    "id",
    "name",
    "run_type",
    "parent_run_id",
    "trace_id",
    "start_time",
    "end_time",
    "inputs",
    "outputs",
    "error",
    "extra",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]

# The llm-manifest audit. `serialized` is the whole point of this projection.
MANIFEST_SELECT: list[str] = ["id", "name", "run_type", "serialized"]

RUN_TYPE_TO_SPAN_TYPE: dict[str, str] = {
    "llm": "llm",
    "tool": "tool",
    "retriever": "retrieval",
    "chain": "chain",
    "prompt": "chain",
    "parser": "chain",
}

ALWAYS_MEANINGFUL_RUN_TYPES = frozenset({"llm", "tool", "retriever"})
CHAINLIKE_RUN_TYPES = frozenset({"chain", "prompt", "parser"})

NOISE_NAME_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"Runnable(Sequence|Parallel|Lambda|Assign|Binding|Each|WithFallbacks).*",
        r"Channel(Read|Write).*",
        r"_(write|read)",
        r"__(start|end)__",
        r".*\.wrap_(model|tool)_call",
        r".*_middleware(\..*)?",
        r".*OutputParser",
        # `Prompt` is what create_react_agent names its prompt-assembly run;
        # its payload is the message list the very next llm span already
        # carries, so keeping it doubles the trace for no information.
        r"Prompt|(Chat)?PromptTemplate",
        r"should_continue|route|_route",
    )
)

# Inputs/outputs keys copied per span type. Anything not listed is dropped.
_INPUT_KEYS: dict[str, tuple[str, ...]] = {
    "llm": ("messages", "prompts", "input"),
    "tool": ("input", "args", "kwargs", "tool_input"),
    "retrieval": ("query", "k", "search_kwargs"),
    "chain": ("messages", "input"),
    "agent": ("messages", "input"),
}
_OUTPUT_KEYS: dict[str, tuple[str, ...]] = {
    "llm": ("generations", "message", "messages", "output"),
    "tool": ("output", "result"),
    "retrieval": ("output", "documents", "docs"),
    "chain": ("messages", "output"),
    "agent": ("messages", "output"),
}

_ALLOWED_ATTRIBUTES = ("run_type", "model", "tokens", "error", "duration_ms")


class IngestionTimeout(Exception):
    """LangSmith never produced the runs we waited for, within the bound."""


class VacuousProjectionError(Exception):
    """A field this collector reads or audits came back None where it must exist."""


class TurnCoverageError(Exception):
    """The session's root runs do not map onto the turns the harness drove.

    Raised when a root carries a turn_index outside `0..expected_turns-1` —
    a stale or colliding session, not a slow one, so waiting cannot fix it.
    """


@runtime_checkable
class TraceCollector(Protocol):
    """The seam Phase 5 depends on; `LangSmithCollector` is the v0 implementation."""

    def collect(self, session_id: str, **kwargs: Any) -> Trace: ...


@dataclass(frozen=True)
class TurnHint:
    """Authoritative turn text captured by the runner as it drove the app."""

    turn_index: int
    user_message: str
    final_response: str


# --------------------------------------------------------------- normalization

def is_noise_span(name: str | None, run_type: str | None, *, is_root: bool) -> bool:
    """The documented filter rule, as one predicate."""
    if is_root:
        return False
    if run_type in ALWAYS_MEANINGFUL_RUN_TYPES:
        return False
    if run_type not in CHAINLIKE_RUN_TYPES:
        return True
    return any(pattern.fullmatch(name or "") for pattern in NOISE_NAME_PATTERNS)


def _span_type(run: Any, *, is_root: bool) -> str:
    if is_root:
        return "agent"
    return RUN_TYPE_TO_SPAN_TYPE.get(run.run_type, "chain")


def _project(payload: dict | None, keys: Sequence[str]) -> dict:
    """Allowlist copy: only `keys`, only when present."""
    if not isinstance(payload, dict):
        return {}
    return {key: payload[key] for key in keys if key in payload}


def _model_name(run: Any) -> str | None:
    """The single value read out of run metadata, by fixed key.

    Everything else in `extra` (the `configurable` echo included) is dropped.
    """
    metadata = (getattr(run, "extra", None) or {}).get("metadata") or {}
    value = metadata.get("ls_model_name")
    return value if isinstance(value, str) else None


def _attributes(run: Any, span_type: str) -> dict[str, Any]:
    tokens = None
    if span_type == "llm":
        total = run.total_tokens
        if total is None:
            total = (run.prompt_tokens or 0) + (run.completion_tokens or 0) or None
        tokens = total
    duration_ms = None
    if run.start_time and run.end_time:
        duration_ms = round((run.end_time - run.start_time).total_seconds() * 1000)
    candidate = {
        "run_type": run.run_type,
        "model": _model_name(run),
        "tokens": tokens,
        "error": run.error,
        "duration_ms": duration_ms,
    }
    return {k: v for k, v in candidate.items() if k in _ALLOWED_ATTRIBUTES and v is not None}


def normalize_spans(runs: Sequence[Any], root_id: str) -> list[Span]:
    """Run tree -> filtered, re-parented `Span` list."""
    by_id = {str(run.id): run for run in runs}
    keep = {
        run_id: not is_noise_span(run.name, run.run_type, is_root=run_id == root_id)
        for run_id, run in by_id.items()
    }

    def surviving_parent(run_id: str) -> str | None:
        parent = by_id[run_id].parent_run_id
        while parent is not None:
            parent = str(parent)
            if parent not in by_id:  # parent not ingested / not fetched
                return None
            if keep.get(parent):
                return parent
            parent = by_id[parent].parent_run_id
        return None

    spans: list[Span] = []
    for run_id, run in by_id.items():
        if not keep[run_id]:
            continue
        is_root = run_id == root_id
        span_type = _span_type(run, is_root=is_root)
        spans.append(
            Span(
                span_id=run_id,
                parent_span_id=None if is_root else surviving_parent(run_id),
                name=run.name,
                span_type=span_type,
                start_time=run.start_time,
                end_time=run.end_time or run.start_time,
                inputs=_project(run.inputs, _INPUT_KEYS.get(span_type, ())),
                outputs=_project(run.outputs, _OUTPUT_KEYS.get(span_type, ())),
                attributes=_attributes(run, span_type),
            )
        )
    spans.sort(key=lambda s: (s.start_time, s.span_id))
    return spans


# ------------------------------------------------------------- turn text

def _message_text(message: Any) -> str:
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


def _derive_turn_text(root: Any) -> tuple[str, str]:
    """(user_message, final_response) from the root run's payloads.

    The *last* human message of the inputs, because on a stateful thread each
    run's input is only the new message, and the last non-tool-calling ai
    message of the outputs.
    """
    inputs = root.inputs if isinstance(root.inputs, dict) else {}
    outputs = root.outputs if isinstance(root.outputs, dict) else {}
    humans = [
        m
        for m in (inputs.get("messages") or [])
        if isinstance(m, dict) and (m.get("type") or m.get("role")) in ("human", "user")
    ]
    user = _message_text(humans[-1]) if humans else ""
    ai = [
        m
        for m in (outputs.get("messages") or [])
        if isinstance(m, dict)
        and (m.get("type") or m.get("role")) in ("ai", "assistant")
        and not m.get("tool_calls")
    ]
    final = _message_text(ai[-1]) if ai else ""
    return user, final


# ------------------------------------------------------------------ collector

class LangSmithCollector:
    """Fetches, filters, normalizes and audits a session's run trees."""

    def __init__(
        self,
        project: str,
        client: Any = None,
        *,
        cfg: TargetAppConfig | None = None,
        root_timeout_s: float = 90.0,
        # Child spans lag the root by up to ~30s and arrive in bursts, so a
        # plateau is not the same as settled. The stability window is
        # (settle_polls - 1) * poll_interval_s = 10s by default; the total
        # bound must comfortably exceed lag + window.
        child_timeout_s: float = 60.0,
        poll_interval_s: float = 2.5,
        settle_polls: int = 5,
        # Structural floor: the app cannot produce an answer without a model
        # call, so a turn with no llm span is an incomplete tree, however
        # stable the run count looked.
        min_llm_spans: int = 1,
        audit_manifests: bool = True,
        span_select: list[str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self.project = project
        self._client = client
        self.cfg = cfg
        self.root_timeout_s = root_timeout_s
        self.child_timeout_s = child_timeout_s
        self.poll_interval_s = poll_interval_s
        self.settle_polls = settle_polls
        self.min_llm_spans = min_llm_spans
        self.audit_manifests = audit_manifests
        self.span_select = span_select or SPAN_SELECT
        self._sleep = sleep
        self._monotonic = monotonic

    @property
    def client(self) -> Any:
        if self._client is None:  # imported lazily: keeps unit tests network-free
            from langsmith import Client  # noqa: PLC0415

            self._client = Client()
        return self._client

    # ---------------------------------------------------------------- fetching

    @staticmethod
    def _session_filter(session_id: str) -> str:
        # LangSmith's query language; the JSON blob is single-quoted inside it.
        return 'has(metadata, \'{"session_id": "' + session_id + '"}\')'

    @staticmethod
    def _metadata(run: Any) -> dict:
        return (getattr(run, "extra", None) or {}).get("metadata") or {}

    def _root_runs(self, session_id: str) -> list[Any]:
        runs = self.client.list_runs(
            project_name=self.project,
            is_root=True,
            filter=self._session_filter(session_id),
            select=self.span_select,
        )
        # Never trust the server-side filter to have been applied: an
        # unsupported filter that degrades to "return everything" would
        # otherwise hand us somebody else's trace.
        return [r for r in runs if self._metadata(r).get("session_id") == session_id]

    def _roots_by_turn(self, runs: Sequence[Any]) -> dict[int, Any]:
        """One root per turn — the LATEST attempt at each turn index.

        A turn the client retried leaves two or three roots sharing a
        turn_index. Slicing the raw list to `expected_turns` would assemble
        {failed attempt, retry, turn 1} and silently drop the real last turn,
        so retries collapse here before anything is ordered or sliced.
        """
        latest: dict[int, Any] = {}
        for run in runs:
            turn_index = self._metadata(run).get("turn_index")
            turn_index = turn_index if isinstance(turn_index, int) else 0
            previous = latest.get(turn_index)
            if previous is None or run.start_time > previous.start_time:
                latest[turn_index] = run
        return latest

    def wait_for_roots(self, session_id: str, expected_turns: int) -> list[Any]:
        deadline = self._monotonic() + self.root_timeout_s
        by_turn: dict[int, Any] = {}
        wanted = set(range(expected_turns)) if expected_turns else set()
        while True:
            by_turn = self._roots_by_turn(self._root_runs(session_id))
            covered = wanted <= set(by_turn) if wanted else bool(by_turn)
            if covered:
                break
            if self._monotonic() >= deadline:
                raise IngestionTimeout(
                    f"LangSmith has root runs for turn indices {sorted(by_turn)} of "
                    f"session_id={session_id} in project {self.project}, expected "
                    f"{sorted(wanted)}, after {self.root_timeout_s}s (a server-side "
                    f"metadata filter that is silently ignored looks exactly like this)"
                )
            self._sleep(self.poll_interval_s)

        if not expected_turns:
            return [by_turn[i] for i in sorted(by_turn)]
        # Exactly the turns the harness drove — an index outside the range is
        # a colliding or stale session, and waiting longer cannot fix it.
        stray = sorted(set(by_turn) - wanted)
        if stray:
            raise TurnCoverageError(
                f"session_id={session_id} has root runs at turn indices {stray}, "
                f"outside the {expected_turns} turn(s) this harness drove"
            )
        return [by_turn[i] for i in range(expected_turns)]

    def _span_runs(self, trace_id: Any) -> list[Any]:
        return list(
            self.client.list_runs(
                project_name=self.project, trace_id=trace_id, select=self.span_select
            )
        )

    def wait_for_spans(self, trace_id: Any, *, require_children: bool = True) -> list[Any]:
        """Poll until the run tree stops growing — child ingestion lags the root."""
        deadline = self._monotonic() + self.child_timeout_s
        previous = -1
        stable = 0
        runs: list[Any] = []
        while True:
            runs = self._span_runs(trace_id)
            count = len(runs)
            stable = stable + 1 if count == previous else 0
            previous = count
            settled = stable >= self.settle_polls - 1
            enough = count >= 2 and any(r.run_type == "llm" for r in runs)
            if settled and (enough or (not require_children and count >= 1)):
                return runs
            if self._monotonic() >= deadline:
                raise IngestionTimeout(
                    f"LangSmith run tree {trace_id} never settled: {count} runs after "
                    f"{self.child_timeout_s}s (child spans lag the root by up to ~30s; "
                    f"llm span present={any(r.run_type == 'llm' for r in runs)})"
                )
            self._sleep(self.poll_interval_s)

    # ------------------------------------------------------- vacuity guards

    def _guard_select(self) -> None:
        """A projection that omits a field we read is a bug, not a slow path.

        Without `extra` there is no run metadata, so session matching would
        find nothing and the collector would spin out its whole timeout before
        reporting a problem that is knowable up front.
        """
        missing = sorted(set(SPAN_SELECT) - set(self.span_select))
        if missing:
            raise VacuousProjectionError(
                f"span_select is missing fields the normalizer reads: {missing}"
            )

    def _guard_projection(self, runs: Sequence[Any], *, where: str) -> None:
        required = ("name", "run_type", "start_time")
        missing = [
            f"{run.id}.{field}"
            for run in runs
            for field in required
            if getattr(run, field, None) is None
        ]
        # A kept llm/tool/retriever run with `inputs is None` means the
        # projection dropped the payload, not that the run had no payload.
        missing += [
            f"{run.id}.inputs"
            for run in runs
            if run.run_type in ALWAYS_MEANINGFUL_RUN_TYPES and run.inputs is None
        ]
        if missing:
            raise VacuousProjectionError(
                f"fields came back None from LangSmith for {where}: {missing[:8]} — "
                f"select={self.span_select}; normalizing this tree would silently "
                f"drop the data it is supposed to carry"
            )

    def audit_llm_manifests(self, trace_id: Any, span_runs: Sequence[Any], tokens) -> int:
        """Fetch `serialized` explicitly for every llm run and scan it.

        `list_runs` does not project `serialized` by default. Fetching it and
        asserting every llm run came back with one is the difference between
        auditing the model manifest and auditing `None`.
        """
        llm_ids = sorted(str(r.id) for r in span_runs if r.run_type == "llm")
        if not llm_ids:
            return 0
        manifests = {
            str(run.id): run.serialized
            for run in self.client.list_runs(
                project_name=self.project,
                trace_id=trace_id,
                run_type="llm",
                select=MANIFEST_SELECT,
            )
        }
        populated = sorted(run_id for run_id in llm_ids if manifests.get(run_id))
        if populated != llm_ids:
            raise VacuousProjectionError(
                f"{len(llm_ids) - len(populated)}/{len(llm_ids)} llm runs in trace "
                f"{trace_id} came back with serialized=None despite an explicit "
                f"select={MANIFEST_SELECT} — the manifest audit would be scanning nothing"
            )
        assert_no_leak(manifests, where=f"llm manifest of trace {trace_id}", tokens=tokens)
        return len(populated)

    # ----------------------------------------------------------------- collect

    def collect(
        self,
        session_id: str,
        *,
        input_id: str,
        mode: TraceMode,
        expected_turns: int = 1,
        status: TraceStatus = "ok",
        hints: Sequence[TurnHint | dict] | None = None,
        extra_metadata: dict[str, Any] | None = None,
        require_children: bool = True,
        extra_leak_tokens: tuple[str, ...] | list[str] = (),
    ) -> Trace:
        self._guard_select()
        tokens = leak_tokens(self.cfg, tuple(extra_leak_tokens))
        hint_by_index = {
            (h.turn_index if isinstance(h, TurnHint) else h["turn_index"]): h
            for h in (hints or [])
        }

        roots = self.wait_for_roots(session_id, expected_turns)
        self._guard_projection(roots, where=f"root runs of session {session_id}")

        turns: list[Turn] = []
        langsmith_trace_ids: list[str] = []
        manifests_scanned = 0
        errored = status == "app_error"

        for index, root in enumerate(roots):
            span_runs = self.wait_for_spans(root.trace_id, require_children=require_children)
            self._guard_projection(span_runs, where=f"spans of trace {root.trace_id}")
            spans = normalize_spans(span_runs, str(root.id))

            user, final = _derive_turn_text(root)
            hint = hint_by_index.get(index)
            if hint is not None:
                user = hint.user_message if isinstance(hint, TurnHint) else hint["user_message"]
                final = (
                    hint.final_response if isinstance(hint, TurnHint) else hint["final_response"]
                )

            if require_children:
                llm_spans = sum(1 for s in spans if s.span_type == "llm")
                if llm_spans < self.min_llm_spans:
                    raise IngestionTimeout(
                        f"turn {index} of session {session_id} (trace {root.trace_id}) "
                        f"normalized to {llm_spans} llm span(s), below the floor of "
                        f"{self.min_llm_spans}: the app cannot produce an answer without "
                        f"a model call, so this run tree is still incomplete "
                        f"({len(spans)} span(s) total)"
                    )

            turns.append(
                Turn(turn_index=index, user_message=user, final_response=final, spans=spans)
            )
            langsmith_trace_ids.append(str(root.trace_id))
            errored = errored or bool(root.error)
            if self.audit_manifests:
                manifests_scanned += self.audit_llm_manifests(root.trace_id, span_runs, tokens)

        metadata: dict[str, Any] = {
            "session_id": session_id,
            "langsmith_project": self.project,
            "langsmith_trace_ids": langsmith_trace_ids,
            "turn_count": len(turns),
            "started_at": roots[0].start_time.isoformat() if roots else None,
            "llm_manifests_audited": manifests_scanned,
        }
        metadata.update(extra_metadata or {})

        trace = Trace(
            trace_id=trace_id_for(session_id),
            input_id=input_id,
            mode=mode,
            turns=turns,
            status="app_error" if errored else "ok",
            metadata=metadata,
        )
        # Tripwire on the allowlist above — see benchmark/harness/scrub.py.
        # `ablation_ids` is excluded: it is our own internal ground-truth field
        # (stripped before Engine sees anything), and its *name* matches the
        # "ablat" fingerprint, so scanning it would flag every healthy trace.
        assert_no_leak(
            trace.model_dump(mode="json", exclude={"ablation_ids"}),
            where=f"stored Trace {trace.trace_id}",
            tokens=tokens,
        )
        return trace
