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
import random
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
from benchmark.pipeline.engine import (
    EngineModelMismatch,
    EngineRunFailed,
    LangGraphEngineInvoker,
)
from benchmark.pipeline.export import assert_export_file_clean
from benchmark.pipeline.manifest import (
    ArtifactPaths,
    RunArtifacts,
    RunManifest,
    StageTiming,
    stage_timer,
    utcnow,
)
from benchmark.pipeline.progress import Progress
from benchmark.pipeline.render import render_markdown
from benchmark.pipeline.scoring import ScoredBoard, score_engine_delta
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
from benchmark.scoring.scorer_description import DescriptionJudge
from benchmark.tracing.store import LocalTraceStore, TraceStore

log = logging.getLogger("benchmark.pipeline")


class AblationLineageBroken(RuntimeError):
    """The ablated dataset does not point back at the traces it came from."""


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
    #: The Engine's updated board, verbatim — the assignment's deliverable.
    predicted: Issueboard
    #: What was actually scored: the Engine's delta over the seed, restricted to
    #: the real trace universe. See benchmark/pipeline/scoring.py.
    scored: ScoredBoard
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

    Checked on an explicit `is_pipeline_fake` marker, plus the module the object
    comes from, rather than on the word "fake" appearing in a name: a wrapper
    closure around the fake would lose the name, and losing this warning is how
    a wiring run gets read as a quality result.
    """
    if getattr(obj, "is_pipeline_fake", False):
        return True
    module = getattr(obj, "__module__", None) or type(obj).__module__ or ""
    return module.startswith("benchmark.pipeline.fakes")


def _faked_stages(stages: dict[str, Any]) -> list[str]:
    """Every stage running against a stand-in, by name.

    ANY faked stage invalidates the run as evidence about the Engine, not just
    a faked ablation: a fake harness means the traces are invented, a fake
    invoker means the board is. They all get the same loud warning.
    """
    return sorted(name for name, obj in stages.items() if _is_faked(obj))


def _fake_warning(faked: list[str], implementations: dict[str, str]) -> str:
    named = ", ".join(f"{name} ({implementations.get(name, '?')})" for name in faked)
    return (
        f"FAKED stage(s): {named}. This run used development stand-ins, so its numbers "
        f"are evidence about wiring, not about the Engine."
    )


def slice_inputs(inputs: InputDataset, cfg: PipelineConfig) -> InputDataset:
    """Apply the config's slice, deterministically.

    A smaller run is the same generation config with fewer of its cells, not a
    second config that drifts from the real one. Three knobs, applied in order:
    `input_modes` filters, `max_inputs_per_mode` samples each mode (seeded, so
    the sample is reproducible), and `max_inputs` caps the total. Everything is
    ordered by `input_id` afterwards, so the resulting `dataset_id` does not
    depend on grid iteration order.

    Note what this does NOT do: it does not avoid the cost of *generating* the
    cells it drops. `generate_inputs` expands the whole grid before the pipeline
    sees it — expansion belongs to `benchmark/generation/`, and slicing is a
    pipeline-layer concern. The expansions are cached on disk, so only the first
    run pays for the cells it then discards.

    Ordering caveat: `max_inputs` is applied AFTER `max_inputs_per_mode`, and it
    takes the first N by sorted `input_id` across whatever the per-mode sample
    left. Because `input_id` is a content hash, that cut is effectively
    arbitrary with respect to mode — a `max_inputs` small enough to bite can
    therefore undo the per-mode balance the sample just established. Set one or
    the other when the mode mix matters; `configs/pipeline/full.yaml` uses only
    the per-mode caps for exactly this reason.
    """
    specs = list(inputs.inputs)
    if cfg.input_modes:
        specs = [s for s in specs if s.mode in cfg.input_modes]
    if cfg.max_inputs_per_mode:
        rng = random.Random(cfg.slice_seed)
        kept: list = []
        by_mode: dict[str, list] = {}
        for spec in sorted(specs, key=lambda s: s.input_id):
            by_mode.setdefault(spec.mode, []).append(spec)
        # Sorted mode order so the RNG draws in a fixed sequence.
        for mode in sorted(by_mode):
            group = by_mode[mode]
            limit = cfg.max_inputs_per_mode.get(mode)
            if limit is not None and len(group) > limit:
                group = sorted(rng.sample(group, limit), key=lambda s: s.input_id)
            kept.extend(group)
        specs = sorted(kept, key=lambda s: s.input_id)
    if cfg.max_inputs is not None and len(specs) > cfg.max_inputs:
        specs = sorted(specs, key=lambda s: s.input_id)[: cfg.max_inputs]
    if not specs:
        raise ValueError(
            f"the slice (input_modes={cfg.input_modes}, "
            f"max_inputs_per_mode={cfg.max_inputs_per_mode}, max_inputs={cfg.max_inputs}) "
            f"left none of the {len(inputs.inputs)} generated input(s) — a benchmark over "
            f"zero inputs would run to completion and report nothing"
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
    *, scored: Issueboard, invocation: EngineInvocation, ablated: TraceDataset
) -> list[str]:
    """The only downstream signal of a partially failed Engine run.

    Trace-level analysis failures are logged to the Engine's stderr and skipped;
    the run still completes, just with a smaller board. Nothing in the output
    says so. So the pipeline says it: counts, side by side, in the log and in
    the manifest.

    Counted on the SCORED board — the Engine's own delta, phantom trace ids and
    echoed seed occurrences already removed. A board that mostly repeats its
    input is exactly the case this warning exists to catch, and counting the
    echo would hide it.
    """
    warnings: list[str] = []
    n_traces = len(ablated.traces)
    covered = len({o.trace_id for o in scored.occurrences})
    log.warning(
        "ENGINE OUTPUT (its own contribution): %s issue(s), %s occurrence(s) over %s of %s "
        "trace(s) analysed (the Engine reports per-trace failures on stderr only — a "
        "smaller-than-expected board is the only signal that reaches here)",
        len(scored.issues),
        len(scored.occurrences),
        covered,
        n_traces,
    )
    if invocation.trace_count and invocation.trace_count != n_traces:
        warnings.append(
            f"the Engine was given {invocation.trace_count} traces but the ablated dataset "
            f"has {n_traces} — the export and the dataset have diverged"
        )
    if not scored.occurrences:
        warnings.append(
            "the Engine contributed no occurrences at all — check its stderr for per-trace "
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
    progress: Progress | None = None,
) -> PipelineRun:
    """Configs -> BenchmarkReport, with every artifact and its lineage on disk.

    `ablation_stage` is required and has no default: until Phase 5 merges the
    only implementation is a fake, and a pipeline that silently defaults to a
    fake ablation would produce a report that looks exactly like a real one.
    Pass `benchmark.pipeline.contracts.load_ablation_stage()` for the real one.

    `progress` defaults to a live `Progress()` (stderr, timestamped, on) so the
    CLI is followable out of the box; pass `Progress(quiet=True)` to silence it
    or a stream-backed one to capture it. It never writes to an artifact.
    """
    timings: list[StageTiming] = []
    warnings: list[str] = []
    started_at = utcnow()
    progress = progress or Progress()

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
    with progress.stage("generation") as stg, stage_timer("generation", timings):
        generation_cfg = load_generation_config(cfg.resolve(cfg.generation_config))
        inputs = generate_inputs(
            generation_cfg, expander, cache_dir=cfg.resolve(cfg.expansion_cache)
        )
        inputs = slice_inputs(inputs, cfg)
        stg.set_detail(f"{len(inputs.inputs)} inputs, dataset_id={inputs.dataset_id[:12]}")
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
        n_inputs = len(inputs.inputs)

        def _harness_done() -> int:
            stats = harness.stats
            return sum(stats.get(k, 0) for k in ("ran", "skipped", "quarantined"))

        with progress.stage("harness") as stg:
            with progress.poll("harness batch", _harness_done, n_inputs, interval=2.0):
                with stage_timer("harness", timings):
                    outputs, traces = harness.run_batch(inputs)
            stg.set_detail(f"{len(traces.traces)} traces, stats={harness.stats}")
        artifacts.write_model("outputs", outputs)
        artifacts.write_model("traces", traces)
        log.info("traces: %s collected, harness stats=%s", len(traces.traces), harness.stats)

        # `run_ablation`'s signature is pinned (benchmark/pipeline/contracts.py)
        # and exposes no progress callback or partial-result hook, so this
        # stage only gets a banner — there is no clean per-error signal to
        # poll the way `harness.stats` gives the batch one.
        with progress.stage("ablation") as stg, stage_timer("ablation", timings):
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
            stg.set_detail(
                f"{len(result.records)} records, {len(result.ground_truth.issues)} "
                f"ground-truth issue(s)"
            )

    ablated: TraceDataset = result.ablated
    # Idempotent (the hash excludes the id field itself): stamping a board that
    # is already stamped is a no-op, and an unstamped one would otherwise reach
    # the manifest with an empty dataset id.
    ground_truth: Issueboard = stamp_dataset_id(result.ground_truth)
    records: list[AblationRecord] = list(result.records)
    split: AblationSplit = result.split
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

    # Fail fast, BEFORE the Engine pass: a broken lineage link means the traces
    # the Engine is about to read for half an hour cannot be tied to the ground
    # truth they will be scored against. Spending the run to discover that at
    # the end is the expensive way to learn it.
    if ablated.parent_dataset_id != traces.dataset_id:
        raise AblationLineageBroken(
            f"the ablated dataset's parent is {ablated.parent_dataset_id!r}, not the traces "
            f"it was built from ({traces.dataset_id!r}) — the run would produce a report "
            f"whose ground truth cannot be traced to its corpus"
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
    n_traces = len(ablated.traces)
    concurrency = max(1, cfg.engine.analysis_concurrency)
    projected_batches = -(-n_traces // concurrency)  # ceil division
    try:
        with servers.running("engine"), progress.stage("engine") as stg:
            progress.note(
                f"    engine: {n_traces} trace(s), analysis_concurrency={concurrency}, "
                f"~{projected_batches} batch(es) projected"
            )
            with (
                stage_timer("engine", timings),
                progress.heartbeat(f"engine run ({cfg.engine.model})", interval=30.0),
            ):
                invocation = engine_invoker(
                    trace_file=written_export,
                    seed_board=seed_board,
                    categories=categories,
                    engine=cfg.engine,
                )
            stg.set_detail(
                f"{len(invocation.board.issues)} issue(s), "
                f"{len(invocation.board.occurrences)} occurrence(s), "
                f"{invocation.seconds:.0f}s server time"
            )
    except EngineRunFailed as exc:
        # The run is lost; the response must not be. A full-scale pass is
        # ~25 minutes of model time, and "it did not validate" is unactionable
        # without the payload that did not validate.
        if exc.raw_output is not None:
            path = artifacts.write_json("engine_raw_output", exc.raw_output)
            log.error("engine run failed; its raw output is at %s", path)
        raise

    # Idempotent re-stamp: the Engine's board_id is its own label, computed over
    # a model that has no `injection_mode` field, and is not byte-compatible
    # with ours. Never compared for equality — replaced.
    predicted = stamp_dataset_id(invocation.board)
    artifacts.write_model("predicted_issueboard", predicted)
    artifacts.write_json("engine_raw_output", invocation.raw_output)

    # Comparison integrity: what the SERVER recorded, not what we sent. A run
    # config LangGraph declined to inject looks exactly like a successful one.
    requested = cfg.engine.model
    if invocation.recorded_models is None:
        # Unreadable records: a server/SDK capability gap, not evidence of a
        # swap. The run stands, but the report must not claim a confirmation
        # it does not have.
        warnings.append(
            f"the server's run records could not be read, so {requested!r} is what was "
            f"requested, not what is confirmed to have run"
        )
    else:
        recorded = set(invocation.recorded_models)
        if not recorded:
            raise EngineModelMismatch(
                f"the server's run records are readable and none of them carries the "
                f"{engine_app.model_configurable_key!r} key, so the run asked for "
                f"{requested!r} and the server recorded no model at all. That is the "
                f"signature of a `configurable` entry LangGraph declined to inject: the "
                f"Engine fell back to its own default, and a model comparison built on "
                f"this would be two arms of the same model."
            )
        if recorded != {requested}:
            raise EngineModelMismatch(
                f"the run asked for model {requested!r} but the server recorded "
                f"{sorted(recorded)} — the model config did not take effect, and two arms "
                f"of a comparison would silently be the same model"
            )

    stage_objects = {
        "ablation": ablation_stage,
        "engine_invoker": engine_invoker,
        "harness": harness,
    }
    implementations = {name: _qualname(obj) for name, obj in stage_objects.items()}
    implementations["servers"] = ",".join(sorted(servers.describe())) or "none"
    faked = _faked_stages(stage_objects)
    # A stage wrapped in a closure loses its own identity; its RESULT does not.
    if "ablation" not in faked and _is_faked(result):
        faked = sorted([*faked, "ablation"])
    if faked:
        warnings.insert(0, _fake_warning(faked, implementations))

    # ---------------------------------------------------------- Stage V: score
    with progress.stage("scoring") as stg, stage_timer("scoring", timings):
        base_rates = build_base_rates(
            ground_truth=ground_truth,
            split=split,
            records=records,
            ablated=ablated,
            dropped_errors=list(result.dropped_errors),
        )
        # The faked-stage fact belongs in the report itself, not only in the
        # manifest: report.json travels on its own.
        base_rates["faked_stages"] = faked
        report, scored = score_engine_delta(
            ground_truth=ground_truth,
            predicted=predicted,
            seed=seed_board,
            # The FULL trace universe, control/clean traces included. Taken
            # from the ablated dataset, never from a union of occurrences:
            # kappa's correction for chance agreement is computed against n,
            # and dropping the clean majority silently inflates it.
            trace_ids=[t.trace_id for t in ablated.traces],
            cfg=cfg.scoring,
            base_rates=base_rates,
            engine_config=EngineConfig(
                model=cfg.engine.model,
                app=engine_app,
                max_tool_calls_per_trace=cfg.engine.max_tool_calls_per_trace,
                seed=cfg.engine.seed,
            ),
            judge=judge,
        )
        stg.set_detail(
            f"{len(report.category_scores)} categories scored, "
            f"{len(scored.board.occurrences)} occurrence(s) in the scored delta"
        )
    artifacts.write_model("report", report)
    # The Engine's own delta, as scored — the board the numbers describe, kept
    # next to the verbatim one the assignment asks for.
    artifacts.write_model("scored_issueboard", scored.board)
    warnings += scored.warnings()
    # Health is judged on the SCORED board: coverage counted before phantoms
    # and seed pairs come out would flatter a run that mostly echoed its input.
    warnings += _engine_health_warnings(
        scored=scored.board, invocation=invocation, ablated=ablated
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
            # "unreadable" rather than "" so the manifest distinguishes a
            # readback that could not happen from one that came back empty.
            "engine_recorded": (
                "unreadable"
                if invocation.recorded_models is None
                else ",".join(invocation.recorded_models)
            ),
        },
        counts={
            "inputs": len(inputs.inputs),
            "traces": len(ablated.traces),
            "raw_traces": len(traces.traces),
            "control_inputs": len(split.control_input_ids),
            "ablate_inputs": len(split.ablate_input_ids),
            "known_errors": len(ground_truth.issues),
            "known_occurrences": len(ground_truth.occurrences),
            # The board as returned (the assignment deliverable) …
            "engine_issues": len(predicted.issues),
            "engine_occurrences": len(predicted.occurrences),
            # … and what of it was actually the Engine's own contribution.
            # Coverage is computed AFTER phantoms and seed pairs come out, so
            # it counts real traces the Engine really said something about.
            "scored_issues": len(scored.board.issues),
            "scored_occurrences": len(scored.board.occurrences),
            "engine_traces_covered": len({o.trace_id for o in scored.board.occurrences}),
            "seed_carrier_issues": len(scored.carrier_error_ids),
            "dropped_seed_issues": len(scored.dropped_seed_issues),
            "dropped_seed_occurrences": scored.dropped_seed_occurrences,
            "phantom_occurrences": len(scored.phantom_occurrences),
            "eh_candidates": len(report.eh_candidates),
        },
        timings=timings,
        artifacts=artifacts.paths,
        stages=implementations,
        harness_stats=dict(getattr(harness, "stats", {}) or {}),
        dropped_errors=list(result.dropped_errors),
        warnings=warnings,
    )

    with progress.stage("render/deliverables") as stg:
        # The SCORED board, not the verbatim one: the prose describes the
        # numbers, and the numbers describe the Engine's delta.
        markdown = render_markdown(
            report, ground_truth=ground_truth, scored_board=scored.board, manifest=manifest
        )
        artifacts.write_text("summary", markdown)

        artifacts.write_model("manifest", manifest)

        # Runs last, and off the run directory rather than off these objects:
        # the deliverable is what someone else can pick up from disk, not what
        # this process happens to be holding.
        # `judge=None` on purpose: the re-score is a reproducibility check, and
        # re-running an LLM judge would both fail to reproduce and double its
        # cost.
        checks = check_deliverables(cfg.run_dir, min_traces=cfg.deliverables.min_traces)
        artifacts.write_json("deliverables", [c.model_dump(mode="json") for c in checks])
        for check in checks:
            (log.info if check.ok else log.error)(
                "deliverable %s: %s — %s", check.name, "ok" if check.ok else "FAILED", check.detail
            )
        log.info("report written to %s", artifacts.path("summary"))
        passed = sum(1 for c in checks if c.ok)
        stg.set_detail(f"{passed}/{len(checks)} deliverables ok")

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
        scored=scored,
        engine=invocation,
        report=report,
        manifest=manifest,
        markdown=markdown,
        deliverables=checks,
    )
