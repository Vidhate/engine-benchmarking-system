"""Stage II top level: [N] inputs -> [N,M] outputs + [N,M,T] traces.

    InputDataset --(batch runner | persona simulator)--> target app
                 --(trace collector)--> TraceStore + (OutputDataset, TraceDataset)

Failure handling follows docs/architecture/03-trace-harness.md exactly:

* **App errors are kept.** A target app that times out or crashes produced a
  real, organic issue — part of the hidden-error set `E_h`. The trace is stored
  with `status="app_error"`, and a rerun retries it (only *ok* traces are done).
* **Collector failures are quarantined.** A trace we could not retrieve or
  normalize is our bug, not signal: it goes to a quarantine directory with a
  reason and a log line, and is never silently dropped.

Idempotency: `session_id = hash(dataset_id, input_id)` fixes the stored
`trace_id` up front, so a rerun can skip an input by a store lookup before
spending an app invocation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from benchmark.harness.client import LangGraphAppClient, TargetAppClient
from benchmark.harness.client import message_text as _text_of
from benchmark.harness.collector import (
    IngestionTimeout,
    LangSmithCollector,
    TurnHint,
)
from benchmark.harness.faults import (
    activation_evidence,
    fault_configurable,
    structural_behavior_tokens,
)
from benchmark.harness.ids import session_id_for, trace_id_for
from benchmark.harness.scrub import LeakDetected
from benchmark.harness.simulator import DONE_TOKEN, UserSimulator, is_done, strip_done
from benchmark.schemas.ablation import FaultConfig
from benchmark.schemas.configs import TargetAppConfig
from benchmark.schemas.inputs import InputDataset, InputSpec, Persona
from benchmark.schemas.io import derive
from benchmark.schemas.traces import (
    OutputDataset,
    OutputRecord,
    Trace,
    TraceDataset,
    TraceMode,
)
from benchmark.tracing.store import TraceStore

log = logging.getLogger("benchmark.harness")


class AmbiguousCheckpoint(LookupError):
    """Several turns end with the same assistant response — say which one."""

# Collection problems: our bug, so the result is quarantined rather than stored.
COLLECTION_FAILURES = (IngestionTimeout, LeakDetected, ValidationError, ValueError, KeyError)


class Quarantine:
    """Where traces we could not collect or validate go, with a reason."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if "/" in key or "\\" in key or key in ("", ".", ".."):
            raise ValueError(f"invalid quarantine key: {key!r}")
        return self.root / f"{key}.json"

    def put(self, key: str, *, reason: str, **payload: Any) -> Path:
        record = {"session_id": key, "reason": reason, **payload}
        path = self._path(key)
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
        log.error("quarantined %s: %s", key, reason)
        return path

    def discard(self, key: str) -> bool:
        """Drop a record whose input has since succeeded. True if one existed."""
        path = self._path(key)
        if not path.exists():
            return False
        path.unlink()
        log.info("cleared stale quarantine record %s", key)
        return True

    def get(self, key: str) -> dict:
        return json.loads(self._path(key).read_text())

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))


@dataclass
class _Outcome:
    input_id: str
    trace: Trace | None = None
    responses: list[str] = field(default_factory=list)
    skipped: bool = False
    quarantined: bool = False


class Harness:
    """Drives inputs against the target app and collects their traces."""

    def __init__(
        self,
        cfg: TargetAppConfig,
        store: TraceStore,
        *,
        client: TargetAppClient | None = None,
        collector: Any = None,
        simulator: UserSimulator | None = None,
        quarantine: Quarantine | None = None,
        concurrency: int = 8,
    ):
        self.cfg = cfg
        self.store = store
        self.client = client or LangGraphAppClient(cfg)
        self.collector = collector or LangSmithCollector(cfg.langsmith_project, cfg=cfg)
        self._simulator = simulator
        self.quarantine = quarantine or Quarantine(Path("data/quarantine"))
        self.concurrency = max(1, concurrency)
        self.stats: dict[str, int] = {}
        # Mode-C activation evidence, keyed by trace_id. Deliberately NOT part
        # of the Trace: a trace that names where its own fault is would hand
        # Engine the ground truth it is supposed to discover.
        self.activation_evidence: dict[str, str] = {}
        self._lock = threading.Lock()

    @property
    def simulator(self) -> UserSimulator:
        if self._simulator is None:
            from benchmark.harness.simulator import OpenAIUserSimulator  # noqa: PLC0415

            self._simulator = OpenAIUserSimulator()
        return self._simulator

    def _bump(self, key: str) -> None:
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + 1

    # ------------------------------------------------------------- collection

    def _collect(
        self,
        session_id: str,
        *,
        input_id: str,
        mode: TraceMode,
        hints: list[TurnHint],
        status: str,
        extra_metadata: dict[str, Any],
        extra_leak_tokens: tuple[str, ...] = (),
    ) -> Trace:
        return self.collector.collect(
            session_id,
            input_id=input_id,
            mode=mode,
            expected_turns=len(hints),
            status=status,
            hints=hints,
            extra_metadata=extra_metadata,
            require_children=status != "app_error",
            extra_leak_tokens=extra_leak_tokens,
        )

    @staticmethod
    def _error_only_trace(
        session_id: str, input_id: str, mode: TraceMode, hints: list[TurnHint], reason: str
    ) -> Trace:
        """An app failure with no retrievable run tree is still organic signal.

        Quarantining it would silently delete a genuine target-app failure from
        the corpus, which is the opposite of what the design asks for.
        """
        from benchmark.schemas.traces import Turn  # noqa: PLC0415

        return Trace(
            trace_id=trace_id_for(session_id),
            input_id=input_id,
            mode=mode,
            turns=[
                Turn(
                    turn_index=h.turn_index,
                    user_message=h.user_message,
                    final_response=h.final_response,
                )
                for h in hints
            ],
            status="app_error",
            metadata={"session_id": session_id, "collection_note": reason},
        )

    # ------------------------------------------------------------ single turn

    def run_single_turn(
        self,
        spec: InputSpec,
        *,
        dataset_id: str = "",
        session_id: str | None = None,
        configurable: dict[str, Any] | None = None,
        extra_leak_tokens: tuple[str, ...] = (),
        store_result: bool = True,
    ) -> Trace:
        if not spec.prompt:
            raise ValueError(f"single-turn input {spec.input_id!r} has no prompt")
        session_id = session_id or session_id_for(dataset_id, spec.input_id)
        thread_id = self.client.create_thread()
        response = self.client.invoke(
            thread_id, spec.prompt, session_id=session_id, turn_index=0, configurable=configurable
        )
        hints = [TurnHint(0, spec.prompt, response.final_response)]
        metadata = {
            "thread_id": thread_id,
            "app": self.cfg.assistant_id,
            "turn_checkpoints": [
                {"turn_index": 0, "checkpoint_id": response.checkpoint_id},
            ],
        }
        if response.error:
            log.warning("app error on %s (%s): %s", spec.input_id, session_id, response.error)
            try:
                trace = self._collect(
                    session_id,
                    input_id=spec.input_id,
                    mode="single_turn",
                    hints=hints,
                    status="app_error",
                    extra_metadata=metadata,
                    extra_leak_tokens=extra_leak_tokens,
                )
            except COLLECTION_FAILURES as exc:
                trace = self._error_only_trace(
                    session_id, spec.input_id, "single_turn", hints, f"{type(exc).__name__}: {exc}"
                )
        else:
            trace = self._collect(
                session_id,
                input_id=spec.input_id,
                mode="single_turn",
                hints=hints,
                status="ok",
                extra_metadata=metadata,
                extra_leak_tokens=extra_leak_tokens,
            )
        if store_result:
            self.store.put(trace)
        return trace

    # ------------------------------------------------------------- multi turn

    def _persona(self, inputs: InputDataset, spec: InputSpec) -> Persona:
        cfg = inputs.generation_config
        for persona in (*cfg.personas, *cfg.adversarial_personas):
            if persona.persona_id == spec.persona_id:
                return persona
        raise KeyError(
            f"input {spec.input_id!r} names persona {spec.persona_id!r}, which is not in "
            f"the dataset's generation_config"
        )

    def max_turns_for(self, requested: int) -> int:
        return max(1, min(requested or 1, self.cfg.max_turns_supported))

    def run_multi_turn(
        self,
        spec: InputSpec,
        persona: Persona,
        *,
        max_turns: int,
        dataset_id: str = "",
        session_id: str | None = None,
        configurable: dict[str, Any] | None = None,
        extra_leak_tokens: tuple[str, ...] = (),
        store_result: bool = True,
    ) -> Trace:
        session_id = session_id or session_id_for(dataset_id, spec.input_id)
        limit = self.max_turns_for(max_turns)
        thread_id = self.client.create_thread()
        scenario = spec.scenario or spec.prompt or ""

        history: list[tuple[str, str]] = []
        hints: list[TurnHint] = []
        checkpoints: list[dict] = []
        error: str | None = None

        for turn_index in range(limit):
            raw = self.simulator.next_message(
                persona=persona, scenario=scenario, history=history, turn_index=turn_index
            )
            message = strip_done(raw)
            if is_done(raw):
                if turn_index > 0:
                    break
                # A simulator that terminates before saying anything would
                # produce a zero-turn conversation; open with the scenario
                # instead so the input still yields a trace.
                message = message or scenario
                log.warning(
                    "simulator emitted %s on turn 0 for %s; opening with the scenario",
                    DONE_TOKEN,
                    spec.input_id,
                )
            response = self.client.invoke(
                thread_id,
                message,
                session_id=session_id,
                turn_index=turn_index,
                configurable=configurable,
            )
            hints.append(TurnHint(turn_index, message, response.final_response))
            checkpoints.append(
                {"turn_index": turn_index, "checkpoint_id": response.checkpoint_id}
            )
            history.append((message, response.final_response))
            if response.error:
                error = response.error
                log.warning("app error on %s turn %s: %s", spec.input_id, turn_index, error)
                break

        metadata = {
            "thread_id": thread_id,
            "app": self.cfg.assistant_id,
            "persona_id": persona.persona_id,
            "turn_checkpoints": checkpoints,
        }
        status = "app_error" if error else "ok"
        try:
            trace = self._collect(
                session_id,
                input_id=spec.input_id,
                mode="multi_turn",
                hints=hints,
                status=status,
                extra_metadata=metadata,
                extra_leak_tokens=extra_leak_tokens,
            )
        except COLLECTION_FAILURES:
            if not error:
                raise
            trace = self._error_only_trace(
                session_id, spec.input_id, "multi_turn", hints, error
            )
        if store_result:
            self.store.put(trace)
        return trace

    # ------------------------------------------------------------------ batch

    def _existing_ok_trace(self, trace_id: str, input_id: str) -> Trace | None:
        """The stored ok trace for this input, or None if it must be re-run.

        A stored file that will not parse is a corrupt artifact, not a reason
        to abort the whole batch: log it and let the input run again.
        """
        if not self.store.exists(trace_id):
            return None
        try:
            existing = self.store.get(trace_id)
        except Exception as exc:  # noqa: BLE001 - any unreadable artifact re-runs
            log.warning(
                "stored trace %s for %s is unreadable (%s: %s) — re-running the input",
                trace_id,
                input_id,
                type(exc).__name__,
                exc,
            )
            return None
        return existing if existing.status == "ok" else None

    def _run_one(self, inputs: InputDataset, spec: InputSpec) -> _Outcome:
        session_id = session_id_for(inputs.dataset_id, spec.input_id)
        trace_id = trace_id_for(session_id)
        existing = self._existing_ok_trace(trace_id, spec.input_id)
        if existing is not None:
            self._bump("skipped")
            log.info("skipping %s — ok trace %s already collected", spec.input_id, trace_id)
            return _Outcome(
                spec.input_id,
                trace=existing,
                responses=[t.final_response for t in existing.turns],
                skipped=True,
            )

        try:
            if spec.mode == "multi_turn":
                persona = self._persona(inputs, spec)
                trace = self.run_multi_turn(
                    spec,
                    persona,
                    max_turns=inputs.generation_config.max_turns,
                    session_id=session_id,
                )
            else:
                trace = self.run_single_turn(spec, session_id=session_id)
        except COLLECTION_FAILURES as exc:
            self.quarantine.put(
                session_id,
                reason=f"{type(exc).__name__}: {exc}",
                input_id=spec.input_id,
                dataset_id=inputs.dataset_id,
                mode=spec.mode,
            )
            self._bump("quarantined")
            return _Outcome(spec.input_id, quarantined=True)

        # The input produced a trace, so any quarantine record from an earlier
        # attempt is stale — leaving it would keep reporting a resolved fault.
        self.quarantine.discard(session_id)
        self._bump("ran")
        if trace.status == "app_error":
            self._bump("app_error")
        return _Outcome(
            spec.input_id, trace=trace, responses=[t.final_response for t in trace.turns]
        )

    def run_batch(self, inputs: InputDataset) -> tuple[OutputDataset, TraceDataset]:
        """[N] -> ([N,M], [N,M,T]), concurrently, resumably."""
        self.stats = {"ran": 0, "skipped": 0, "quarantined": 0, "app_error": 0}
        # The pool size is the concurrency semaphore: one task per input, at
        # most `concurrency` in flight, so the target app is never overrun.
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            outcomes = list(pool.map(lambda spec: self._run_one(inputs, spec), inputs.inputs))

        collected = sorted(
            (o for o in outcomes if o.trace is not None), key=lambda o: o.input_id
        )
        traces = TraceDataset(traces=[o.trace for o in collected])
        outputs = OutputDataset(
            outputs=[
                OutputRecord(
                    input_id=o.input_id, trace_id=o.trace.trace_id, responses=o.responses
                )
                for o in collected
            ]
        )
        log.info("batch complete: %s", self.stats)
        return derive(outputs, inputs), derive(traces, inputs)

    # ------------------------------------------------- public API for Phase 5

    def run_with_faults(
        self,
        input_spec: InputSpec,
        fault_config: FaultConfig,
        *,
        dataset_id: str = "",
        baseline: Trace | None = None,
        weak_validation: bool = False,
        persona: Persona | None = None,
        max_turns: int = 1,
        store_result: bool = True,
    ) -> Trace:
        """Mode C — arm a declared dependency fault, re-run, prove activation.

        Pass `baseline` (the unarmed trace for this input) for real activation
        validation; `weak_validation=True` acknowledges the weak form. Passing
        neither raises — see `benchmark.harness.faults.activation_evidence`.

        Raises `UndeclaredFault` before touching the app if the shim is not in
        `fault_configurable_keys`, and `FaultNotActivated` if the regenerated
        trace shows no corruption in the span the fault must corrupt.

        The trace is persisted **only after activation is proven**: an armed
        run whose fault did not activate is an unlabelled, fault-contaminated
        trace, and leaving one in the store would later feed it to Engine as
        organic signal.

        Activation evidence is published on `self.activation_evidence`, keyed
        by trace_id — never inside the Trace (that would leak ground truth).
        """
        if baseline is None and not weak_validation:
            raise ValueError(
                "run_with_faults needs either baseline=<the unarmed trace for this "
                "input> or an explicit weak_validation=True; see "
                "benchmark.harness.faults.activation_evidence for why"
            )
        configurable = fault_configurable(self.cfg, fault_config)
        variant = "fault:" + hashlib.sha256(
            json.dumps(fault_config.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()[:12]
        session_id = session_id_for(dataset_id, input_spec.input_id, variant=variant)
        tokens = structural_behavior_tokens(fault_config)

        # store_result=False on the way in: nothing is persisted until the
        # fault is proven to have activated.
        if input_spec.mode == "multi_turn":
            if persona is None:
                raise ValueError("multi_turn run_with_faults needs the input's persona")
            trace = self.run_multi_turn(
                input_spec,
                persona,
                max_turns=max_turns,
                session_id=session_id,
                configurable=configurable,
                extra_leak_tokens=tokens,
                store_result=False,
            )
        else:
            trace = self.run_single_turn(
                input_spec,
                session_id=session_id,
                configurable=configurable,
                extra_leak_tokens=tokens,
                store_result=False,
            )

        evidence = activation_evidence(
            trace, fault_config, baseline=baseline, weak_validation=weak_validation
        )
        self.activation_evidence[trace.trace_id] = evidence
        if store_result:
            self.store.put(trace)
        return trace

    def turn_boundaries(self, thread_id: str) -> list[tuple[str, str, str]]:
        """The thread's turn boundaries, oldest first.

        One entry per *answer*: `(checkpoint_id, message_id, text)` for each
        checkpoint whose newest message is a non-tool-calling assistant
        message. Intra-turn checkpoints — the ones ending in a tool call — are
        not turn boundaries and would otherwise shift every turn index.
        """
        boundaries: list[tuple[str, str, str]] = []
        for snapshot in reversed(self.client.get_history(thread_id)):  # oldest -> newest
            messages = (snapshot.get("values") or {}).get("messages") or []
            if not messages:
                continue
            last = messages[-1]
            if (last.get("type") or last.get("role")) != "ai" or last.get("tool_calls"):
                continue
            boundaries.append(
                (snapshot["checkpoint"]["checkpoint_id"], last["id"], _text_of(last).strip())
            )
        return boundaries

    def locate_checkpoint(
        self, thread_id: str, response_text: str = "", *, turn_index: int | None = None
    ) -> tuple[str, str]:
        """Find where to fork for a Mode-A edit, as `(checkpoint_id, message_id)`.

        Pass `turn_index` (0-based over the thread's answers), `response_text`,
        or both — both is the safest, since the text then double-checks the
        index. Matching on text alone and finding SEVERAL matches raises
        `AmbiguousCheckpoint` rather than picking one: two turns ending in the
        same words ("You're welcome!") are perfectly ordinary, and forking at
        the wrong one would mislabel the ground-truth turn index silently.

        Reading the thread's checkpoint history is the one piece of LangGraph
        knowledge Phase 5 would otherwise have to duplicate, so it lives here.
        """
        if not response_text and turn_index is None:
            raise ValueError("locate_checkpoint needs response_text, turn_index, or both")

        boundaries = self.turn_boundaries(thread_id)
        wanted = response_text.strip()

        if turn_index is not None:
            if not 0 <= turn_index < len(boundaries):
                raise KeyError(
                    f"thread {thread_id} has only {len(boundaries)} answer turn(s); "
                    f"turn_index={turn_index} is out of range"
                )
            checkpoint_id, message_id, text = boundaries[turn_index]
            if wanted and text != wanted:
                raise KeyError(
                    f"turn {turn_index} of thread {thread_id} does not end with the given "
                    f"response (it ends with {text[:80]!r})"
                )
            return checkpoint_id, message_id

        matches = [
            (index, checkpoint_id, message_id)
            for index, (checkpoint_id, message_id, text) in enumerate(boundaries)
            if text == wanted
        ]
        if not matches:
            raise KeyError(
                f"no checkpoint on thread {thread_id} ends with the given assistant response"
            )
        if len(matches) > 1:
            raise AmbiguousCheckpoint(
                f"{len(matches)} turns of thread {thread_id} end with the same assistant "
                f"response (turn indices {[m[0] for m in matches]}); pass turn_index to "
                f"say which one to fork at"
            )
        _index, checkpoint_id, message_id = matches[0]
        return checkpoint_id, message_id

    def replay(
        self,
        thread_ref: str,
        checkpoint_ref: str | dict,
        corrupted_state: Any,
        remaining_user_messages: list[str],
        *,
        input_id: str = "",
        dataset_id: str = "",
        store_result: bool = True,
    ) -> Trace:
        """Mode A — fork a thread at a checkpoint with edited state and resume.

        Returns the trace of the **regenerated turns only** (turns `k+1..M`).
        Phase 5 splices them onto `turns[0..k-1] + corrupted k` to assemble
        `T*`; `metadata` carries `thread_id`, `source_checkpoint_id` and
        `fork_checkpoint_id` so the splice is auditable.
        """
        if not remaining_user_messages:
            raise ValueError(
                "replay needs at least one entry in remaining_user_messages; a "
                "single-turn (M=1) Mode-A injection has no downstream to "
                "regenerate and is a consistency-managed post-hoc edit instead "
                "(docs/architecture/04-ablation-engine.md)"
            )
        checkpoint = (
            {"checkpoint_id": checkpoint_ref}
            if isinstance(checkpoint_ref, str)
            else dict(checkpoint_ref)
        )
        source_checkpoint_id = checkpoint.get("checkpoint_id")

        variant = "replay:" + hashlib.sha256(
            json.dumps(
                [thread_ref, source_checkpoint_id, corrupted_state, remaining_user_messages],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()[:12]
        session_id = session_id_for(dataset_id, input_id, variant=variant)

        fork = self.client.update_state(thread_ref, corrupted_state, checkpoint=checkpoint)

        hints: list[TurnHint] = []
        checkpoints: list[dict] = []
        error: str | None = None
        for turn_index, message in enumerate(remaining_user_messages):
            response = self.client.invoke(
                thread_ref,
                message,
                session_id=session_id,
                turn_index=turn_index,
                # Only the first resumed run starts from the fork; afterwards
                # the fork *is* the thread head.
                checkpoint=fork if turn_index == 0 else None,
            )
            hints.append(TurnHint(turn_index, message, response.final_response))
            checkpoints.append(
                {"turn_index": turn_index, "checkpoint_id": response.checkpoint_id}
            )
            if response.error:
                error = response.error
                break

        metadata = {
            "thread_id": thread_ref,
            "app": self.cfg.assistant_id,
            "source_checkpoint_id": source_checkpoint_id,
            "fork_checkpoint_id": (fork or {}).get("checkpoint_id"),
            "replayed": True,
            "turn_checkpoints": checkpoints,
        }
        trace = self._collect(
            session_id,
            input_id=input_id,
            mode="multi_turn" if len(hints) > 1 else "single_turn",
            hints=hints,
            status="app_error" if error else "ok",
            extra_metadata=metadata,
        )
        if store_result:
            self.store.put(trace)
        return trace


# ---------------------------------------------------------- module-level API

def run_harness(
    inputs: InputDataset,
    cfg: TargetAppConfig,
    store: TraceStore,
    *,
    client: TargetAppClient | None = None,
    collector: Any = None,
    simulator: UserSimulator | None = None,
    quarantine: Quarantine | None = None,
    concurrency: int = 8,
) -> tuple[OutputDataset, TraceDataset]:
    """The Stage II entrypoint: InputDataset -> (OutputDataset, TraceDataset)."""
    harness = Harness(
        cfg,
        store,
        client=client,
        collector=collector,
        simulator=simulator,
        quarantine=quarantine,
        concurrency=concurrency,
    )
    return harness.run_batch(inputs)


def replay(
    thread_ref: str,
    checkpoint_ref: str | dict,
    corrupted_state: Any,
    remaining_user_messages: list[str],
    *,
    harness: Harness,
    **kwargs: Any,
) -> Trace:
    """Mode A entrypoint — see `Harness.replay`."""
    return harness.replay(
        thread_ref, checkpoint_ref, corrupted_state, remaining_user_messages, **kwargs
    )


def run_with_faults(
    input_spec: InputSpec,
    fault_config: FaultConfig,
    *,
    harness: Harness,
    **kwargs: Any,
) -> Trace:
    """Mode C entrypoint — see `Harness.run_with_faults`."""
    return harness.run_with_faults(input_spec, fault_config, **kwargs)
