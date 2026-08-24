"""Fixtures for the Phase 7 pipeline tests.

Everything here is offline by construction: a deterministic prompt expander, a
harness that replays canned traces instead of driving a LangGraph server, and
an Engine invoker that returns a canned board instead of calling one. Between
them they let the miniature integration test exercise the whole wiring with
zero network and zero servers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmark.pipeline.config import load_taxonomy
from benchmark.pipeline.contracts import EngineInvocation
from benchmark.schemas import (
    Dimension,
    GenerationConfig,
    InputDataset,
    InputSpec,
    Issue,
    Issueboard,
    IssueOccurrence,
    OutputDataset,
    OutputRecord,
    Span,
    Trace,
    TraceDataset,
    Turn,
)
from benchmark.schemas.io import derive, stamp_dataset_id

REPO_ROOT = Path(__file__).resolve().parents[2]
MINI_PIPELINE_CONFIG = REPO_ROOT / "configs" / "pipeline" / "mini.yaml"

_T0 = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


@pytest.fixture
def taxonomy():
    return load_taxonomy(REPO_ROOT / "configs" / "taxonomy.yaml")


def make_trace(trace_id: str, input_id: str, *, text: str = "answer", turns: int = 1) -> Trace:
    return Trace(
        trace_id=trace_id,
        input_id=input_id,
        mode="single_turn" if turns == 1 else "multi_turn",
        turns=[
            Turn(
                turn_index=i,
                user_message=f"question {i} for {input_id}",
                final_response=f"{text} {i}",
                spans=[
                    Span(
                        span_id=f"{trace_id}-s{i}",
                        name="ChatOpenAI",
                        span_type="llm",
                        start_time=_T0,
                        end_time=_T0,
                        inputs={"messages": [f"question {i}"]},
                        outputs={"content": f"{text} {i}"},
                    )
                ],
            )
            for i in range(turns)
        ],
        metadata={"session_id": f"sess-{trace_id}", "thread_id": f"thread-{trace_id}"},
    )


@pytest.fixture
def tiny_inputs() -> InputDataset:
    cfg = GenerationConfig(
        safe_dims=[Dimension(dim_id="topic", name="topic", kind="safe", variations=["a", "b"])],
        mode="single_turn",
        seed=7,
    )
    dataset = InputDataset(
        generation_config=cfg,
        inputs=[
            InputSpec(
                input_id=f"in-{i}",
                mode="single_turn",
                dim_id="topic" if i % 2 else "adv",
                variation=f"v{i}",
                prompt=f"prompt {i}",
            )
            for i in range(6)
        ],
    )
    return stamp_dataset_id(dataset)


@pytest.fixture
def tiny_traces(tiny_inputs) -> TraceDataset:
    return derive(
        TraceDataset(
            traces=[make_trace(f"tr-{s.input_id}", s.input_id) for s in tiny_inputs.inputs]
        ),
        tiny_inputs,
    )


# --------------------------------------------------------------------- fakes

class FakeExpander:
    """Deterministic, network-free `PromptExpander`."""

    def expand(self, dim, variation, seed, app_context="") -> str:
        return f"[{dim.dim_id}/{variation}] please help me with {variation}"

    def expand_scenario(self, persona, dim_id, variation, seed, app_context="") -> str:
        return f"[{persona.persona_id}/{dim_id}/{variation}] a conversation about {variation}"


class FakeHarness:
    """A `HarnessLike` that fabricates one canned trace per input.

    Stands in for the whole Phase-4 harness: no LangGraph server, no LangSmith,
    but the same `(OutputDataset, TraceDataset)` return and the same lineage.
    """

    #: A test double says so out loud: any faked stage makes a run's numbers
    #: evidence about wiring rather than about the Engine, and the runner
    #: reports that on every artifact it writes.
    is_pipeline_fake = True

    def __init__(self, cfg=None, store=None, *, app_error_inputs: tuple[str, ...] = ()):
        self.cfg = cfg
        self.store = store
        self.stats: dict[str, int] = {}
        self.app_error_inputs = app_error_inputs
        self.batches: list[InputDataset] = []

    def run_batch(self, inputs: InputDataset) -> tuple[OutputDataset, TraceDataset]:
        self.batches.append(inputs)
        traces = []
        outputs = []
        for spec in inputs.inputs:
            trace = make_trace(f"tr-{spec.input_id}", spec.input_id)
            if spec.input_id in self.app_error_inputs:
                trace = trace.model_copy(update={"status": "app_error"})
            if self.store is not None:
                self.store.put(trace)
            traces.append(trace)
            outputs.append(
                OutputRecord(
                    input_id=spec.input_id,
                    trace_id=trace.trace_id,
                    responses=[t.final_response for t in trace.turns],
                )
            )
        self.stats = {
            "ran": len(traces),
            "skipped": 0,
            "quarantined": 0,
            "app_error": len(self.app_error_inputs),
        }
        return (
            derive(OutputDataset(outputs=outputs), inputs),
            derive(TraceDataset(traces=traces), inputs),
        )


@pytest.fixture
def fake_harness_factory():
    made: list[FakeHarness] = []

    def factory(cfg, store):
        harness = FakeHarness(cfg, store)
        made.append(harness)
        return harness

    factory.made = made  # type: ignore[attr-defined]
    return factory


class FakeEngineInvoker:
    """An `EngineInvoker` that returns a board derived from the ground truth.

    `recall` picks how much of the ground truth it reproduces, so a test can
    ask for a perfect Engine, a blind one, or anything between — without a
    model. It also always adds one unmatched issue, which is what an E_h
    candidate looks like coming back from a real run.
    """

    is_pipeline_fake = True

    def __init__(self, ground_truth: Issueboard | None = None, *, recall: float = 1.0):
        self.ground_truth = ground_truth
        self.recall = recall
        self.calls: list[dict] = []

    def __call__(self, *, trace_file, seed_board, categories, engine) -> EngineInvocation:
        self.calls.append(
            {
                "trace_file": Path(trace_file),
                "seed_board": seed_board,
                "categories": categories,
                "engine": engine,
            }
        )
        gt = self.ground_truth or Issueboard(source="ground_truth")
        keep = int(round(len(gt.issues) * self.recall))
        issues = [
            Issue(
                error_id=f"P{n}",
                title=issue.title,
                description=issue.description,
                category_id=issue.category_id,
                severity=issue.severity,
            )
            for n, issue in enumerate(gt.issues[:keep])
        ]
        renamed = {issue.error_id: f"P{n}" for n, issue in enumerate(gt.issues[:keep])}
        occurrences = [
            IssueOccurrence(error_id=renamed[o.error_id], trace_id=o.trace_id)
            for o in gt.occurrences
            if o.error_id in renamed
        ]
        # An unmatched prediction: every real run produces some, and the report
        # has an E_h appendix precisely for them.
        first_trace = gt.occurrences[0].trace_id if gt.occurrences else "tr-unknown"
        issues.append(
            Issue(
                error_id="P-extra",
                title="unclassified oddity",
                description="something the Engine flagged that no injection explains",
                category_id="other",
                severity="low",
            )
        )
        occurrences.append(IssueOccurrence(error_id="P-extra", trace_id=first_trace))
        board = Issueboard(
            board_id="engine-side-hash-not-ours",
            source="engine_predicted",
            issues=list(seed_board.issues) + issues,
            occurrences=occurrences,
        )
        return EngineInvocation(
            board=board,
            raw_output=board.model_dump(mode="json"),
            seconds=1.5,
            thread_id="thread-fake",
            recorded_models=[engine.model],
            trace_count=len(TraceDataset.model_validate_json(Path(trace_file).read_text()).traces),
        )


# ------------------------------------------------- a completed miniature run

@pytest.fixture
def mini_cfg(tmp_path):
    """configs/pipeline/mini.yaml, with its artifacts redirected into tmp."""
    from benchmark.pipeline.config import load_pipeline_config

    loaded = load_pipeline_config(MINI_PIPELINE_CONFIG)
    return loaded.model_copy(
        update={
            "artifacts_root": str(tmp_path / "artifacts"),
            "expansion_cache": str(tmp_path / "cache"),
        }
    ).with_root(loaded.root)


@pytest.fixture
def mini_engine_invoker():
    return FakeEngineInvoker()


@pytest.fixture
def mini_run(mini_cfg, fake_harness_factory, mini_engine_invoker):
    """The whole pipeline, end to end, with every external seam faked."""
    from benchmark.pipeline.fakes import fake_run_ablation  # noqa: PLC0415
    from benchmark.pipeline.runner import run_pipeline  # noqa: PLC0415

    return run_pipeline(
        mini_cfg,
        ablation_stage=fake_run_ablation,
        engine_invoker=mini_engine_invoker,
        harness_factory=fake_harness_factory,
        expander=FakeExpander(),
    )
