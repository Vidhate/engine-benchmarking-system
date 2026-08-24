"""Network-free fixtures for the ablation engine's unit tests.

docs/execution-plan.md ground rule 5 — no OpenAI call, no LangGraph server, no
LangSmith. Everything the ablation engine touches at runtime (the harness, the
proposing/planning LLM agent, the trace store) has a deterministic double here.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from benchmark.schemas.ablation import FaultConfig
from benchmark.schemas.configs import AblationConfig, TargetAppConfig
from benchmark.schemas.inputs import (
    Dimension,
    GenerationConfig,
    InputDataset,
    InputSpec,
    Persona,
)
from benchmark.schemas.io import stamp_dataset_id
from benchmark.schemas.issues import ErrorCategory
from benchmark.schemas.traces import Span, Trace, TraceDataset, Turn
from benchmark.tracing.store import LocalTraceStore

T0 = datetime(2026, 8, 24, 9, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------- trace fixtures

def make_span(
    span_id: str,
    span_type: str,
    name: str,
    *,
    parent: str | None = None,
    inputs: dict | None = None,
    outputs: dict | None = None,
    offset_ms: int = 0,
    attributes: dict | None = None,
) -> Span:
    start = T0 + timedelta(milliseconds=offset_ms)
    return Span(
        span_id=span_id,
        parent_span_id=parent,
        name=name,
        span_type=span_type,  # type: ignore[arg-type]
        start_time=start,
        end_time=start + timedelta(milliseconds=200),
        inputs=inputs or {},
        outputs=outputs or {},
        attributes=attributes or {},
    )


def make_turn(
    trace_id: str,
    turn_index: int,
    *,
    user_message: str = "what is the refund window?",
    final_response: str = "Refunds are available within 30 days.",
    with_retrieval: bool = True,
    with_tool: bool = True,
    docs: list[dict] | None = None,
) -> Turn:
    prefix = f"{trace_id}-t{turn_index}"
    docs = docs if docs is not None else [{"doc_id": "refund-policy", "updated": "2026-01-02"}]
    spans = [
        make_span(
            f"{prefix}-agent",
            "agent",
            "target_app",
            inputs={"messages": [{"type": "human", "content": user_message}]},
            outputs={
                "messages": [
                    {"type": "human", "content": user_message},
                    {"type": "ai", "content": final_response},
                ]
            },
            offset_ms=0,
        ),
    ]
    if with_tool:
        spans.append(
            make_span(
                f"{prefix}-tool",
                "tool",
                "rag_search",
                parent=f"{prefix}-agent",
                inputs={"input": {"query": "refund window"}},
                outputs={"output": json.dumps(docs)},
                offset_ms=100,
            )
        )
    if with_retrieval:
        spans.append(
            make_span(
                f"{prefix}-retr",
                "retrieval",
                "corpus_search",
                parent=f"{prefix}-tool" if with_tool else f"{prefix}-agent",
                inputs={"query": "refund window", "k": 3},
                outputs={"output": docs},
                offset_ms=150,
            )
        )
    spans.append(
        make_span(
            f"{prefix}-llm",
            "llm",
            "ChatOpenAI",
            parent=f"{prefix}-agent",
            inputs={"messages": [[{"type": "human", "content": user_message}]]},
            outputs={"generations": [[{"text": final_response}]]},
            offset_ms=300,
            attributes={"model": "gpt-5.1-mini", "tokens": 120},
        )
    )
    return Turn(
        turn_index=turn_index,
        user_message=user_message,
        final_response=final_response,
        spans=spans,
    )


def make_trace(
    trace_id: str,
    input_id: str,
    *,
    mode: str = "single_turn",
    turns: int = 1,
    with_retrieval: bool = True,
    with_tool: bool = True,
    status: str = "ok",
    thread_id: str | None = None,
) -> Trace:
    return Trace(
        trace_id=trace_id,
        input_id=input_id,
        mode=mode,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        turns=[
            make_turn(
                trace_id,
                i,
                user_message=f"question {i} about refunds?",
                final_response=f"Refunds are available within 30 days. (turn {i})",
                with_retrieval=with_retrieval,
                with_tool=with_tool,
            )
            for i in range(turns)
        ],
        metadata={
            "session_id": f"s-{trace_id}",
            "thread_id": thread_id or f"thread-{trace_id}",
            "app": "target_app",
            "turn_count": turns,
            "turn_checkpoints": [
                {"turn_index": i, "checkpoint_id": f"ckpt-{trace_id}-{i}"} for i in range(turns)
            ],
            "langsmith_project": "engine-bench-target",
        },
    )


# ---------------------------------------------------------------- input corpus

def make_inputs(n_safe: int = 6, n_adv: int = 4, n_multi: int = 2) -> InputDataset:
    safe_dim = Dimension(dim_id="topic", name="query_topic", kind="safe", variations=["a", "b"])
    adv_dim = Dimension(
        dim_id="injection", name="prompt_injection", kind="adversarial", variations=["x", "y"]
    )
    persona = Persona(
        persona_id="p1", name="Frustrated user", kind="target", description="grumpy", goals=["fix"]
    )
    cfg = GenerationConfig(
        safe_dims=[safe_dim],
        adversarial_dims=[adv_dim],
        personas=[persona],
        mode="mixed",
        max_turns=3,
        app_context="Nimbus Notes support assistant",
    )
    inputs: list[InputSpec] = []
    for i in range(n_safe):
        inputs.append(
            InputSpec(
                input_id=f"safe-{i:02d}",
                mode="single_turn",
                dim_id="topic",
                variation="ab"[i % 2],
                prompt=f"safe prompt {i}",
            )
        )
    for i in range(n_adv):
        inputs.append(
            InputSpec(
                input_id=f"adv-{i:02d}",
                mode="single_turn",
                dim_id="injection",
                variation="xy"[i % 2],
                prompt=f"adversarial prompt {i}",
            )
        )
    for i in range(n_multi):
        inputs.append(
            InputSpec(
                input_id=f"mt-{i:02d}",
                mode="multi_turn",
                dim_id="topic",
                variation="ab"[i % 2],
                persona_id="p1",
                scenario=f"scenario {i}",
            )
        )
    return stamp_dataset_id(InputDataset(generation_config=cfg, inputs=inputs))


def make_traces(inputs: InputDataset) -> TraceDataset:
    traces = []
    for spec in inputs.inputs:
        traces.append(
            make_trace(
                f"trace-{spec.input_id}",
                spec.input_id,
                mode=spec.mode,
                turns=3 if spec.mode == "multi_turn" else 1,
                # every third single-turn trace never touched the retriever, so
                # filters have something to actually discriminate on
                with_retrieval=not spec.input_id.endswith("02"),
                with_tool=not spec.input_id.endswith("02"),
            )
        )
    return stamp_dataset_id(TraceDataset(traces=traces))


# --------------------------------------------------------------- fake harness

class FakeHarness:
    """A `Harness` stand-in with the exact surface the ablation engine uses.

    Reproduces the parts of the real contract Phase 5 depends on:

    * `replay` returns the REGENERATED TURNS ONLY and refuses an empty
      `remaining_user_messages` (the M=1 degenerate case),
    * `run_with_faults` demands a `baseline` unless `weak_validation`,
    * activation evidence arrives out-of-band on `activation_evidence`,
    * `locate_checkpoint` reads a live thread and raises `KeyError` for a
      thread the server no longer has (a dead thread ref).
    """

    def __init__(
        self,
        cfg: TargetAppConfig,
        store: Any,
        *,
        live_threads: set[str] | None = None,
        self_corrects: bool = False,
        fault_activates: bool = True,
        replay_fails_for: set[str] | None = None,
    ):
        self.cfg = cfg
        self.store = store
        self.activation_evidence: dict[str, str] = {}
        self.live_threads = live_threads
        self.self_corrects = self_corrects
        self.fault_activates = fault_activates
        self.replay_fails_for = replay_fails_for or set()
        self.replays: list[dict] = []
        self.fault_runs: list[dict] = []
        self._counter = 0

    # -- thread liveness ---------------------------------------------------
    def _alive(self, thread_id: str) -> bool:
        return self.live_threads is None or thread_id in self.live_threads

    def turn_boundaries(self, thread_id: str):
        if not self._alive(thread_id):
            return []
        return [(f"ckpt-{thread_id}-{i}", f"msg-{thread_id}-{i}", f"answer {i}") for i in range(4)]

    def locate_checkpoint(self, thread_id, response_text="", *, turn_index=None):
        if not self._alive(thread_id):
            raise KeyError(f"thread {thread_id} has only 0 answer turn(s)")
        index = turn_index or 0
        return f"ckpt-{thread_id}-{index}", f"msg-{thread_id}-{index}"

    # -- Mode A ------------------------------------------------------------
    def replay(
        self,
        thread_ref,
        checkpoint_ref,
        corrupted_state,
        remaining_user_messages,
        *,
        input_id="",
        dataset_id="",
        store_result=True,
    ) -> Trace:
        if not remaining_user_messages:
            raise ValueError("replay needs at least one entry in remaining_user_messages")
        if input_id in self.replay_fails_for:
            raise RuntimeError("replay blew up for this input")
        self.replays.append(
            {
                "thread_ref": thread_ref,
                "checkpoint_ref": checkpoint_ref,
                "corrupted_state": corrupted_state,
                "remaining": list(remaining_user_messages),
                "input_id": input_id,
            }
        )
        self._counter += 1
        trace_id = f"replayed-{self._counter}"
        answer = "Understood — anything else?"
        if self.self_corrects:
            answer = "Correction: I made an error earlier; that reference does not exist."
        turns = [
            make_turn(
                trace_id,
                i,
                user_message=message,
                final_response=answer,
            )
            for i, message in enumerate(remaining_user_messages)
        ]
        trace = Trace(
            trace_id=trace_id,
            input_id=input_id,
            mode="multi_turn" if len(turns) > 1 else "single_turn",
            turns=turns,
            metadata={
                "thread_id": thread_ref,
                "source_checkpoint_id": checkpoint_ref,
                "fork_checkpoint_id": f"fork-{self._counter}",
                "replayed": True,
                "app": "target_app",
            },
        )
        if store_result:
            self.store.put(trace)
        return trace

    # -- Mode C ------------------------------------------------------------
    def run_with_faults(
        self,
        input_spec,
        fault_config: FaultConfig,
        *,
        dataset_id="",
        baseline=None,
        weak_validation=False,
        persona=None,
        max_turns=1,
        store_result=True,
    ) -> Trace:
        if baseline is None and not weak_validation:
            raise ValueError("run_with_faults needs either baseline=... or weak_validation=True")
        if fault_config.shim not in ("retriever", "tool", "llm_proxy"):
            from benchmark.harness.faults import UndeclaredFault

            raise UndeclaredFault(f"undeclared shim {fault_config.shim}")
        self.fault_runs.append(
            {
                "input_id": input_spec.input_id,
                "fault": fault_config.model_dump(),
                "baseline": None if baseline is None else baseline.trace_id,
            }
        )
        if not self.fault_activates:
            from benchmark.harness.faults import FaultNotActivated

            raise FaultNotActivated("the armed fault left no visible trace")
        self._counter += 1
        trace_id = f"faulted-{self._counter}"
        corrupted_docs = [{"doc_id": "webhook-setup", "updated": "2026-02-01"}]
        turns = [
            make_turn(
                trace_id,
                i,
                user_message=(input_spec.prompt or input_spec.scenario or "hello"),
                final_response="I could not find anything relevant about that.",
                docs=corrupted_docs,
            )
            for i in range(max_turns if input_spec.mode == "multi_turn" else 1)
        ]
        trace = Trace(
            trace_id=trace_id,
            input_id=input_spec.input_id,
            mode=input_spec.mode,
            turns=turns,
            metadata={
                "thread_id": f"thread-{trace_id}",
                "app": "target_app",
                "turn_checkpoints": [
                    {"turn_index": i, "checkpoint_id": f"ckpt-{trace_id}-{i}"}
                    for i in range(len(turns))
                ],
            },
        )
        self.activation_evidence[trace_id] = json.dumps(
            {"output": corrupted_docs}, sort_keys=True
        )
        if store_result:
            self.store.put(trace)
        return trace


# ----------------------------------------------------------- proposal factory

MARKER = "NBX-4471"
CORRUPT_TEXT = (
    "I have escalated this under case reference NBX-4471; the engineer assigned to it "
    "confirmed the charge will be reversed on the next billing cycle."
)


def make_proposal(
    error_id: str = "E-hallucination-00",
    category_id: str = "hallucination",
    *,
    mode: str = "replay_edit",
    marker: str = MARKER,
    replacement: str = CORRUPT_TEXT,
    turn_index: int = 0,
    filter_steps: list | None = None,
    fault: FaultConfig | None = None,
    target_count: int = 3,
    severity: str = "high",
):
    from benchmark.ablation.agent import Corruption, ProposedError
    from benchmark.schemas.issues import Issue

    corruption = (
        Corruption(replacement=replacement, marker=marker, turn_index=turn_index)
        if mode == "replay_edit"
        else None
    )
    if mode == "dependency_fault" and fault is None:
        fault = FaultConfig(shim="retriever", target="corpus_search", behavior="irrelevant_docs")
    return ProposedError(
        issue=Issue(
            error_id=error_id,
            title=f"{category_id} error",
            description="a concrete, app-specific failure",
            category_id=category_id,
            severity=severity,  # type: ignore[arg-type]
            injection_mode=mode,  # type: ignore[arg-type]
        ),
        filter_steps=filter_steps or [],
        corruption=corruption,
        fault=fault if mode == "dependency_fault" else None,
        target_count=target_count,
    )


# ------------------------------------------------------------------- fixtures

@pytest.fixture
def target_cfg() -> TargetAppConfig:
    return TargetAppConfig(
        base_url="http://127.0.0.1:2024",
        assistant_id="target_app",
        langsmith_project="engine-bench-target",
        fault_configurable_keys={
            "retriever": "fault_retriever",
            "tool": "fault_tool",
            "llm": "fault_llm",
        },
        max_turns_supported=8,
    )


@pytest.fixture
def categories() -> list[ErrorCategory]:
    return [
        ErrorCategory(category_id="hallucination", name="hallucination", description="made up"),
        ErrorCategory(
            category_id="retrieval_failure", name="retrieval_failure", description="bad docs"
        ),
        ErrorCategory(category_id="other", name="other", description="escape hatch"),
    ]


@pytest.fixture
def inputs() -> InputDataset:
    return make_inputs()


@pytest.fixture
def traces(inputs: InputDataset) -> TraceDataset:
    return make_traces(inputs)


@pytest.fixture
def store(tmp_path, traces):
    s = LocalTraceStore(tmp_path / "traces")
    for trace in traces.traces:
        s.put(copy.deepcopy(trace))
    return s


@pytest.fixture
def harness(target_cfg, store) -> FakeHarness:
    return FakeHarness(target_cfg, store)


@pytest.fixture
def ablation_cfg() -> AblationConfig:
    return AblationConfig(seed=7, control_fraction=0.3, min_eligible=2, n_per_category=1)
