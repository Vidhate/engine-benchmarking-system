"""The offline stand-ins for every stage the pipeline does not own.

Three of the pipeline's stages reach outside the process: the Phase-4 harness
(a live LangGraph target app plus LangSmith), the Phase-5 ablation engine (an
LLM agent driving that same app), and the Engine under test (a second live
server). This module holds one stand-in for each, plus a network-free prompt
expander, so the whole assembly runs end to end with no servers, no API keys
and no network at all.

They ship inside the package rather than in `tests/` because three callers
need them — `tests/pipeline/`, the `--fake-harness/--fake-ablation/--fake-engine`
CLI flags, and `scripts/pipeline_smoke.py` — and a test double defined twice
drifts. Every object here carries `is_pipeline_fake`, which is what makes the
runner stamp a FAKED warning onto the manifest and the rendered report: a run
that used any of them is evidence about *wiring*, never about the Engine.

`fake_run_ablation` is the one with a sharper edge. It does not ablate
anything: traces pass through untouched and the ground truth it plants is a
label over unmodified content, so the Engine is being scored against errors
nobody injected and low scores are the expected outcome, not a finding. The
real `benchmark.ablation.run_ablation` is the default everywhere now; this
stays as the CI double, because CI cannot spend an LLM agent and two servers
per run.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.pipeline.contracts import EngineInvocation
from benchmark.pipeline.export import write_leak_stripped_export
from benchmark.schemas import (
    AblationConfig,
    AblationRecord,
    AblationSplit,
    Dimension,
    ErrorCategory,
    Issue,
    Issueboard,
    IssueOccurrence,
    OutputDataset,
    OutputRecord,
    Persona,
    Span,
    Trace,
    TraceDataset,
    Turn,
)
from benchmark.schemas.configs import TargetAppConfig
from benchmark.schemas.inputs import InputDataset
from benchmark.schemas.io import derive, stamp_dataset_id
from benchmark.schemas.issues import OTHER_CATEGORY_ID
from benchmark.tracing.store import TraceStore

#: Alternated across planted errors so a fake run still exercises both arms of
#: the post-hoc content-vs-mechanism analysis downstream.
_MODES = ("replay_edit", "dependency_fault")

_MAX_PLANTED_ERRORS = 3


@dataclass
class FakeAblationResult:
    """Structurally identical to the pinned Phase-5 `AblationResult`."""

    ablated: TraceDataset
    ground_truth: Issueboard
    records: list[AblationRecord] = field(default_factory=list)
    split: AblationSplit = field(
        default_factory=lambda: AblationSplit(seed=0, control_fraction=0.0)
    )
    export_path: str = ""
    dropped_errors: list[str] = field(default_factory=list)


def _adversarial_first(inputs: InputDataset) -> dict[str, int]:
    """Rank inputs so planted errors land where the app is likeliest to slip.

    Purely cosmetic for a pass-through fake, but it makes the miniature run's
    numbers a shade less meaningless: an adversarial input's trace is where an
    organic issue might actually coincide with the planted label.
    """
    order: dict[str, int] = {}
    for spec in inputs.inputs:
        adversarial = bool(spec.fixed_adversarial_id) or "adv" in spec.dim_id.lower()
        order[spec.input_id] = (0 if adversarial else 1)
    return order


def split_inputs(inputs: InputDataset, cfg: AblationConfig) -> AblationSplit:
    """A seeded, provenance-stratified control/ablate split at input level.

    Stratifying on `dim_id` is the cheap half of what Phase 5 does properly;
    the point here is only that control inputs exist, are chosen the same way
    on every rerun, and are never touched afterwards.
    """
    by_stratum: dict[str, list[str]] = {}
    for spec in sorted(inputs.inputs, key=lambda s: s.input_id):
        by_stratum.setdefault(spec.dim_id, []).append(spec.input_id)

    rng = random.Random(cfg.seed)
    control: list[str] = []
    ablate: list[str] = []
    for stratum in sorted(by_stratum):
        members = list(by_stratum[stratum])
        rng.shuffle(members)
        n_control = int(round(len(members) * cfg.control_fraction))
        control.extend(members[:n_control])
        ablate.extend(members[n_control:])
    return AblationSplit(
        seed=cfg.seed,
        control_fraction=cfg.control_fraction,
        strata=["dim_id"],
        control_input_ids=sorted(control),
        ablate_input_ids=sorted(ablate),
    )


def fake_run_ablation(
    *,
    traces: TraceDataset,
    inputs: InputDataset,
    categories: list[ErrorCategory],
    cfg: AblationConfig,
    harness: Any = None,
    store: TraceStore | None = None,
    export_path: str | Path,
) -> FakeAblationResult:
    """Pass-through "ablation": a synthetic ground truth over untouched traces.

    Same call shape and same return shape as the pinned Phase-5
    `run_ablation`. `harness` and `store` are accepted and ignored — the real
    stage needs them to replay and re-run, this one has nothing to inject.

    Invariants it DOES honour, because downstream code depends on them:

    * control inputs carry no ground-truth occurrence at all;
    * no trace carries two occurrences of the same category (the exact-key
      matcher's disjointness invariant — scoring is unsound without it);
    * the ablated dataset points at the source dataset via `parent_dataset_id`;
    * the export written to `export_path` is leak-stripped and audited.
    """
    cfg = cfg or AblationConfig()
    split = split_inputs(inputs, cfg)
    ablate_ids = set(split.ablate_input_ids)

    rank = _adversarial_first(inputs)
    ablate_traces = sorted(
        (t for t in traces.traces if t.input_id in ablate_ids),
        key=lambda t: (rank.get(t.input_id, 1), t.trace_id),
    )

    usable = [c for c in categories if c.category_id != OTHER_CATEGORY_ID]
    planted = usable[: max(1, min(_MAX_PLANTED_ERRORS, len(usable)))] if usable else []

    issues: list[Issue] = []
    occurrences: list[IssueOccurrence] = []
    records: list[AblationRecord] = []
    for index, category in enumerate(planted):
        error_id = f"K{index + 1}"
        issues.append(
            Issue(
                error_id=error_id,
                title=f"planted {category.name}",
                description=(
                    f"Synthetic ground-truth entry standing in for a real {category.name} "
                    f"injection until Phase 5 lands. {category.description}"
                ),
                category_id=category.category_id,
                severity=("high", "medium", "low")[index % 3],
                injection_mode=_MODES[index % len(_MODES)],
            )
        )
    # Round-robin, so one trace never gets two errors — and therefore never two
    # of the same category, whatever the trace count happens to be.
    for position, trace in enumerate(ablate_traces):
        if not issues:
            break
        issue = issues[position % len(issues)]
        occurrences.append(
            IssueOccurrence(
                error_id=issue.error_id,
                trace_id=trace.trace_id,
                turn_index=0,
                evidence="(fake ablation: no content was actually modified)",
            )
        )
        records.append(
            AblationRecord(
                ablation_id=f"fake-{issue.error_id}-{trace.trace_id}",
                error_id=issue.error_id,
                trace_id=trace.trace_id,
            )
        )

    ground_truth = stamp_dataset_id(
        Issueboard(source="ground_truth", issues=issues, occurrences=occurrences)
    )
    # Pass-through: every trace survives, control and ablate alike, so the
    # trace universe handed to scoring is the real one.
    ablated = derive(TraceDataset(traces=list(traces.traces)), traces)
    written = write_leak_stripped_export(ablated, export_path)

    return FakeAblationResult(
        ablated=ablated,
        ground_truth=ground_truth,
        records=records,
        split=split,
        export_path=str(written),
        dropped_errors=[],
    )


#: Explicit marker, so the runner's "this run was faked" warning does not rest
#: on the word "fake" surviving in somebody's wrapper name.
fake_run_ablation.is_pipeline_fake = True  # type: ignore[attr-defined]
FakeAblationResult.is_pipeline_fake = True  # type: ignore[attr-defined]


# ============================================================================
# The other three offline seams: generation, the harness, the Engine.
# ============================================================================

#: One fixed instant for every canned span. Trace content has to be a pure
#: function of its input for a run's dataset_id to be reproducible, and a
#: wall clock is the usual way that quietly stops being true.
_T0 = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


class FakeExpander:
    """A network-free `PromptExpander`: no OpenAI call, same call shape."""

    is_pipeline_fake = True

    def expand(self, dim: Dimension, variation: str, seed: int, app_context: str = "") -> str:
        return f"[{dim.dim_id}/{variation}] please help me with {variation}"

    def expand_scenario(
        self, persona: Persona, dim_id: str, variation: str, seed: int, app_context: str = ""
    ) -> str:
        return f"[{persona.persona_id}/{dim_id}/{variation}] a conversation about {variation}"


def make_trace(trace_id: str, input_id: str, *, text: str = "answer", turns: int = 1) -> Trace:
    """One schema-valid canned `Trace`, with a span per turn."""
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


class FakeHarness:
    """A `HarnessLike` that fabricates one canned trace per input.

    Stands in for the whole Phase-4 harness: no LangGraph server, no LangSmith,
    but the same `(OutputDataset, TraceDataset)` return and the same lineage.
    Traces are written through to the store when there is one, so a downstream
    stage that reads the store rather than the dataset still finds them.
    """

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


class FakeHarnessFactory:
    """A `HarnessFactory` producing `FakeHarness`es, remembering each one."""

    is_pipeline_fake = True

    def __init__(self) -> None:
        self.made: list[FakeHarness] = []

    def __call__(self, cfg: TargetAppConfig, store: TraceStore) -> FakeHarness:
        harness = FakeHarness(cfg, store)
        self.made.append(harness)
        return harness


class FakeEngineInvoker:
    """An `EngineInvoker` that returns a board derived from the ground truth.

    `recall` picks how much of the ground truth it reproduces, so a caller can
    ask for a perfect Engine, a blind one, or anything between — without a
    model. It also always adds one unmatched issue, which is what an E_h
    candidate looks like coming back from a real run.

    `ground_truth` is settable after construction because that is the only
    order that works: the ablation stage has to run before there is a ground
    truth to mirror. Left unset (the CLI's `--fake-engine`, which has no way to
    reach inside the run), it predicts nothing but the unmatched issue — a
    blind Engine, which still exercises every seam downstream of it.
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
        exported = TraceDataset.model_validate_json(Path(trace_file).read_text())
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
        # has an E_h appendix precisely for them. It is pinned to a trace that
        # is actually in the export — a phantom trace id is a different defect
        # with its own guard, and this double should not manufacture one.
        first_trace = next(
            (o.trace_id for o in gt.occurrences),
            exported.traces[0].trace_id if exported.traces else "tr-unknown",
        )
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
            trace_count=len(exported.traces),
        )
