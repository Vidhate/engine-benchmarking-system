"""The end-to-end assembly: configs in, `BenchmarkReport` out.

    generate_inputs -> harness batch -> run_ablation -> Engine -> score()

Every stage the pipeline does not own is reached through a Protocol from
`benchmark.pipeline.contracts`, so the whole thing runs against fakes with no
network. Every stage writes its artifact, and `manifest.json` records what
produced what.

**Server-lifetime choreography** (see `benchmark.pipeline.servers`): the
harness batch and the ablation stage run inside ONE target-app server lifetime
— Mode-A replay forks a thread created during the batch, and `langgraph dev`
loses thread state on restart. The Engine's server is started afterwards, for
its own run only. Scoring and rendering need no server.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.generation.config_loader import load_generation_config
from benchmark.generation.expander import PromptExpander
from benchmark.generation.generators import generate_inputs
from benchmark.harness.config import load_target_app_config
from benchmark.pipeline.config import (
    PipelineConfig,
    config_hashes,
    load_engine_app_config,
    load_seed_board,
    load_taxonomy,
)
from benchmark.pipeline.contracts import (
    AblationStage,
    EngineInvocation,
    EngineInvoker,
    HarnessFactory,
    HarnessLike,
    assert_ablation_result,
)
from benchmark.pipeline.deliverables import DeliverableCheck, check_deliverables
from benchmark.pipeline.engine import LangGraphEngineInvoker
from benchmark.pipeline.export import assert_export_file_clean
from benchmark.pipeline.manifest import (
    ArtifactPaths,
    RunArtifacts,
    RunManifest,
    StageTiming,
    stage_timer,
    utcnow,
)
from benchmark.pipeline.render import render_markdown
from benchmark.pipeline.servers import ServerLifetime
from benchmark.schemas import (
    AblationRecord,
    AblationSplit,
    BenchmarkReport,
    EngineConfig,
    ErrorCategory,
    InputDataset,
    Issueboard,
    OutputDataset,
    TraceDataset,
)
from benchmark.schemas.io import stamp_dataset_id
from benchmark.scoring import score
from benchmark.scoring.scorer_description import DescriptionJudge
from benchmark.tracing.store import LocalTraceStore, TraceStore

log = logging.getLogger("benchmark.pipeline")


@dataclass
class PipelineRun:
    cfg: PipelineConfig
    run_dir: Path
    inputs: InputDataset
    outputs: OutputDataset
    traces: TraceDataset
    ablated: TraceDataset
    split: AblationSplit
    records: list[AblationRecord]
    export_path: Path
    seed_board: Issueboard
    ground_truth: Issueboard
    predicted: Issueboard
    engine: EngineInvocation
    report: BenchmarkReport
    manifest: RunManifest
    markdown: str
    deliverables: list[DeliverableCheck] = field(default_factory=list)


def _qualname(obj: Any) -> str:
    module = getattr(obj, "__module__", None) or type(obj).__module__
    name = getattr(obj, "__qualname__", None) or type(obj).__qualname__
    return f"{module}.{name}"


def _is_faked(obj: Any) -> bool:
    """Whether a stage (or its result) is a development stand-in.

    Checked on an explicit `is_pipeline_fake` marker rather than on the word
    "fake" appearing in a name: a wrapper closure around the fake would lose
    the name, and losing this warning is how a wiring run gets read as a
    quality result.
    """
    return bool(getattr(obj, "is_pipeline_fake", False))


def slice_inputs(inputs: InputDataset, cfg: PipelineConfig) -> InputDataset:
    """Apply the config's miniature slice, deterministically.

    A miniature run is the same generation config with fewer of its cells, not
    a second config that drifts from the real one. Selection is by sorted
    `input_id`, so it does not depend on grid iteration order.
    """
    specs = list(inputs.inputs)
    if cfg.input_modes:
        specs = [s for s in specs if s.mode in cfg.input_modes]
    if cfg.max_inputs is not None and len(specs) > cfg.max_inputs:
        specs = sorted(specs, key=lambda s: s.input_id)[: cfg.max_inputs]
    if not specs:
        raise ValueError(
            f"the slice (input_modes={cfg.input_modes}, max_inputs={cfg.max_inputs}) left "
            f"none of the {len(inputs.inputs)} generated input(s) — a benchmark over zero "
            f"inputs would run to completion and report nothing"
        )
    if len(specs) == len(inputs.inputs):
        return inputs
    return stamp_dataset_id(
        InputDataset(
            created_at=inputs.created_at,
            generation_config=inputs.generation_config,
            inputs=specs,
        )
    )


def default_harness_factory(cfg: PipelineConfig) -> HarnessFactory:
    """The real Phase-4 harness, built once and shared with the ablation stage.

    This is `benchmark.harness.run_harness`'s own body with the harness object
    kept: the ablation stage needs `replay` / `run_with_faults` on the SAME
    harness, against the same live server, as the batch that produced the
    threads it forks.
    """

    def factory(app_cfg, store: TraceStore) -> HarnessLike:
        from benchmark.harness import Harness, Quarantine  # noqa: PLC0415

        return Harness(
            app_cfg,
            store,
            quarantine=Quarantine(cfg.run_dir / "quarantine"),
            concurrency=cfg.harness.concurrency,
        )

    return factory


def build_base_rates(
    *,
    ground_truth: Issueboard,
    split: AblationSplit,
    records: list[AblationRecord],
    ablated: TraceDataset,
    dropped_errors: list[str],
) -> dict[str, Any]:
    """What the scores are relative to — the numbers a reader needs to judge them.

    Everything here comes off the ablation result, never off the predicted
    board: base rates describe the *benchmark*, and deriving them from the
    thing being measured would make a blind Engine look like an easy dataset.
    """
    n_traces = len(ablated.traces)
    labelled = {o.trace_id for o in ground_truth.occurrences}
    return {
        "n_traces": n_traces,
        "clean_traces": n_traces - len(labelled),
        "injected_traces": len(labelled),
        "injection_prevalence": round(len(labelled) / n_traces, 4) if n_traces else 0.0,
        "control_fraction": split.control_fraction,
        "control_inputs": len(split.control_input_ids),
        "ablate_inputs": len(split.ablate_input_ids),
        "split_strata": list(split.strata),
        "split_seed": split.seed,
        "injected_error_count": len(ground_truth.issues),
        "per_error_injection_counts": dict(
            sorted(Counter(o.error_id for o in ground_truth.occurrences).items())
        ),
        "per_error_record_counts": dict(sorted(Counter(r.error_id for r in records).items())),
        "injection_modes": dict(
            sorted(Counter(i.injection_mode or "unspecified" for i in ground_truth.issues).items())
        ),
        "dropped_errors": list(dropped_errors),
    }


def _engine_health_warnings(
    *, predicted: Issueboard, invocation: EngineInvocation, ablated: TraceDataset
) -> list[str]:
    """The only downstream signal of a partially failed Engine run.

    Trace-level analysis failures are logged to the Engine's stderr and skipped;
    the run still completes, just with a smaller board. Nothing in the output
    says so. So the pipeline says it: counts, side by side, in the log and in
    the manifest.
    """
    warnings: list[str] = []
    n_traces = len(ablated.traces)
    covered = len({o.trace_id for o in predicted.occurrences})
    log.warning(
        "ENGINE OUTPUT: %s issue(s), %s occurrence(s) over %s of %s trace(s) analysed "
        "(the Engine reports per-trace failures on stderr only — a smaller-than-expected "
        "board is the only signal that reaches here)",
        len(predicted.issues),
        len(predicted.occurrences),
        covered,
        n_traces,
    )
    if invocation.trace_count and invocation.trace_count != n_traces:
        warnings.append(
            f"the Engine was given {invocation.trace_count} traces but the ablated dataset "
            f"has {n_traces} — the export and the dataset have diverged"
        )
    if not predicted.occurrences:
        warnings.append(
            "the Engine returned no occurrences at all — check its stderr for per-trace "
            "analysis failures before reading the scores as a quality signal"
        )
    return warnings


def run_pipeline(
    cfg: PipelineConfig,
    *,
    ablation_stage: AblationStage,
    engine_invoker: EngineInvoker | None = None,
    harness_factory: HarnessFactory | None = None,
    expander: PromptExpander | None = None,
    store: TraceStore | None = None,
    servers: ServerLifetime | None = None,
    judge: DescriptionJudge | None = None,
    artifact_paths: ArtifactPaths | None = None,
) -> PipelineRun:
    """Configs -> BenchmarkReport, with every artifact and its lineage on disk.

    `ablation_stage` is required and has no default: until Phase 5 merges the
    only implementation is a fake, and a pipeline that silently defaults to a
    fake ablation would produce a report that looks exactly like a real one.
    Pass `benchmark.pipeline.contracts.load_ablation_stage()` for the real one.
    """
    timings: list[StageTiming] = []
    warnings: list[str] = []
    started_at = utcnow()

    artifacts = RunArtifacts(cfg.run_dir, artifact_paths)
    artifacts.write_json("pipeline_config", cfg.model_dump(mode="json"))

    categories: list[ErrorCategory] = load_taxonomy(cfg.resolve(cfg.taxonomy))
    seed_board = load_seed_board(
        cfg.resolve(cfg.seed_issueboard) if cfg.seed_issueboard else None
    )
    artifacts.write_model("seed_issueboard", seed_board)

    engine_app = load_engine_app_config(cfg.resolve(cfg.engine_app_config))
    engine_invoker = engine_invoker or LangGraphEngineInvoker(engine_app)
    servers = servers or ServerLifetime(cfg.root, {}, enabled=False)

    # ------------------------------------------------------- Stage I: inputs
    with stage_timer("generation", timings):
        generation_cfg = load_generation_config(cfg.resolve(cfg.generation_config))
        inputs = generate_inputs(
            generation_cfg, expander, cache_dir=cfg.resolve(cfg.expansion_cache)
        )
        inputs = slice_inputs(inputs, cfg)
    artifacts.write_model("inputs", inputs)
    log.info("inputs: %s specs, dataset_id=%s", len(inputs.inputs), inputs.dataset_id)

    target_app = load_target_app_config(cfg.resolve(cfg.target_app_config))
    store = store or LocalTraceStore(
        cfg.resolve(cfg.trace_store) if cfg.trace_store else cfg.run_dir / "trace_store"
    )
    harness = (harness_factory or default_harness_factory(cfg))(target_app, store)
    export_path = artifacts.claim("engine_input")

    # ------------ Stages II + III: ONE target-app server lifetime, in order --
    with servers.running("target_app"):
        with stage_timer("harness", timings):
            outputs, traces = harness.run_batch(inputs)
        artifacts.write_model("outputs", outputs)
        artifacts.write_model("traces", traces)
        log.info("traces: %s collected, harness stats=%s", len(traces.traces), harness.stats)

        with stage_timer("ablation", timings):
            result = assert_ablation_result(
                ablation_stage(
                    traces=traces,
                    inputs=inputs,
                    categories=categories,
                    cfg=cfg.ablation,
                    harness=harness,
                    store=store,
                    export_path=export_path,
                )
            )

    ablated: TraceDataset = result.ablated
    # Idempotent (the hash excludes the id field itself): stamping a board that
    # is already stamped is a no-op, and an unstamped one would otherwise reach
    # the manifest with an empty dataset id.
    ground_truth: Issueboard = stamp_dataset_id(result.ground_truth)
    records: list[AblationRecord] = list(result.records)
    split: AblationSplit = result.split
    if ablated.parent_dataset_id != traces.dataset_id:
        warnings.append(
            f"the ablated dataset's parent is {ablated.parent_dataset_id!r}, not the traces "
            f"it was built from ({traces.dataset_id!r}) — lineage is broken"
        )
    artifacts.write_model("ablated_traces", ablated)
    artifacts.write_model("ablation_split", split)
    artifacts.write_json(
        "ablation_records", [r.model_dump(mode="json") for r in records]
    )
    artifacts.write_model("ground_truth_issueboard", ground_truth)
    if result.dropped_errors:
        warnings.append(
            f"the ablation stage dropped {len(result.dropped_errors)} proposed error(s): "
            f"{result.dropped_errors}"
        )

    written_export = Path(result.export_path)
    if written_export != export_path:
        warnings.append(
            f"the ablation stage wrote its export to {written_export} rather than the "
            f"requested {export_path}"
        )
    # Defence in depth: a leaked ground-truth field does not crash anything, it
    # quietly turns the benchmark into a lookup exercise and makes the numbers
    # look BETTER. Audited at the boundary that consumes the file.
    assert_export_file_clean(written_export)

    # ------------------------------------ Stage IV: the Engine's own lifetime
    with servers.running("engine"), stage_timer("engine", timings):
        invocation = engine_invoker(
            trace_file=written_export,
            seed_board=seed_board,
            categories=categories,
            engine=cfg.engine,
        )
    # Idempotent re-stamp: the Engine's board_id is its own label, computed over
    # a model that has no `injection_mode` field, and is not byte-compatible
    # with ours. Never compared for equality — replaced.
    predicted = stamp_dataset_id(invocation.board)
    artifacts.write_model("predicted_issueboard", predicted)
    artifacts.write_json("engine_raw_output", invocation.raw_output)
    warnings += _engine_health_warnings(
        predicted=predicted, invocation=invocation, ablated=ablated
    )

    # ---------------------------------------------------------- Stage V: score
    with stage_timer("scoring", timings):
        base_rates = build_base_rates(
            ground_truth=ground_truth,
            split=split,
            records=records,
            ablated=ablated,
            dropped_errors=list(result.dropped_errors),
        )
        report = score(
            ground_truth,
            predicted,
            cfg.scoring,
            base_rates,
            # The FULL trace universe, control/clean traces included. Taken
            # from the ablated dataset, never from a union of occurrences:
            # kappa's correction for chance agreement is computed against n,
            # and dropping the clean majority silently inflates it.
            trace_ids=[t.trace_id for t in ablated.traces],
            engine_config=EngineConfig(
                model=cfg.engine.model,
                app=engine_app,
                max_tool_calls_per_trace=cfg.engine.max_tool_calls_per_trace,
                seed=cfg.engine.seed,
            ),
            judge=judge,
        )
    artifacts.write_model("report", report)

    stages = {
        "ablation": _qualname(ablation_stage),
        "engine_invoker": _qualname(engine_invoker),
        "harness": _qualname(harness),
        "servers": ",".join(sorted(servers.describe())) or "none",
    }
    if _is_faked(ablation_stage) or _is_faked(result):
        warnings.insert(
            0,
            f"the ablation stage was FAKED ({stages['ablation']}): traces were not "
            f"modified and the ground truth is a synthetic label over unmodified "
            f"traces. These scores are evidence about wiring, not about the Engine.",
        )

    manifest = RunManifest(
        run_id=cfg.run_id,
        created_at=started_at,
        config_hashes=config_hashes(cfg),
        dataset_ids={
            "inputs": inputs.dataset_id,
            "outputs": outputs.dataset_id,
            "traces": traces.dataset_id,
            "ablated_traces": ablated.dataset_id,
            "seed_issueboard": seed_board.board_id,
            "ground_truth_issueboard": ground_truth.board_id,
            "predicted_issueboard": predicted.board_id,
            "report": report.report_id,
        },
        lineage={
            "inputs": None,
            "outputs": outputs.parent_dataset_id,
            "traces": traces.parent_dataset_id,
            "ablated_traces": ablated.parent_dataset_id,
        },
        models={
            "engine": cfg.engine.model,
            "engine_recorded": ",".join(invocation.recorded_models),
        },
        counts={
            "inputs": len(inputs.inputs),
            "traces": len(ablated.traces),
            "raw_traces": len(traces.traces),
            "control_inputs": len(split.control_input_ids),
            "ablate_inputs": len(split.ablate_input_ids),
            "known_errors": len(ground_truth.issues),
            "known_occurrences": len(ground_truth.occurrences),
            "engine_issues": len(predicted.issues),
            "engine_occurrences": len(predicted.occurrences),
            "engine_traces_covered": len({o.trace_id for o in predicted.occurrences}),
            "eh_candidates": len(report.eh_candidates),
        },
        timings=timings,
        artifacts=artifacts.paths,
        stages=stages,
        harness_stats=dict(getattr(harness, "stats", {}) or {}),
        dropped_errors=list(result.dropped_errors),
        warnings=warnings,
    )

    markdown = render_markdown(
        report, ground_truth=ground_truth, predicted=predicted, manifest=manifest
    )
    artifacts.write_text("summary", markdown)

    artifacts.write_model("manifest", manifest)

    # Runs last, and off the run directory rather than off these objects: the
    # deliverable is what someone else can pick up from disk, not what this
    # process happens to be holding.
    # `judge=None` on purpose: the re-score is a reproducibility check, and
    # re-running an LLM judge would both fail to reproduce and double its cost.
    checks = check_deliverables(cfg.run_dir, min_traces=cfg.deliverables.min_traces)
    artifacts.write_json("deliverables", [c.model_dump(mode="json") for c in checks])
    for check in checks:
        (log.info if check.ok else log.error)(
            "deliverable %s: %s — %s", check.name, "ok" if check.ok else "FAILED", check.detail
        )
    log.info("report written to %s", artifacts.path("summary"))

    return PipelineRun(
        cfg=cfg,
        run_dir=cfg.run_dir,
        inputs=inputs,
        outputs=outputs,
        traces=traces,
        ablated=ablated,
        split=split,
        records=records,
        export_path=written_export,
        seed_board=seed_board,
        ground_truth=ground_truth,
        predicted=predicted,
        engine=invocation,
        report=report,
        manifest=manifest,
        markdown=markdown,
        deliverables=checks,
    )
