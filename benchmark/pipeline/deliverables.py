"""The assignment's deliverables, checked against what a run actually wrote.

The assignment asks for four concrete things:

1. a JSON file of >=300 traces, handed to the Engine;
2. an issueboard going IN;
3. the UPDATED issueboard coming OUT;
4. a standalone scoring function.

Each is checked here from the run directory alone — no in-memory objects, no
pipeline state. That is deliberate: a check that reads the same objects the run
just built proves the run is self-consistent, not that the deliverable exists on
disk in a shape someone else can use. The scoring check goes furthest and
actually re-scores from the artifacts, then compares against the report the run
wrote.

`min_traces` is a parameter, not a constant, so the miniature run exercises the
same code path at N=6 that the full run exercises at N>=300.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from benchmark.pipeline.export import ExportLeak, assert_export_file_clean
from benchmark.pipeline.manifest import ArtifactPaths, RunManifest
from benchmark.schemas import BenchmarkReport, Issueboard, ScoringConfig, TraceDataset
from benchmark.scoring import score
from benchmark.scoring.scorer_description import DescriptionJudge


class DeliverableCheck(BaseModel):
    name: str
    ok: bool
    detail: str


def _load(cls, path: Path):
    return cls.model_validate_json(path.read_text())


def rescore_from_disk(
    run_dir: str | Path,
    *,
    paths: ArtifactPaths | None = None,
    judge: DescriptionJudge | None = None,
) -> BenchmarkReport:
    """`score()` called with nothing but files — the standalone entrypoint.

    This is the assignment's "scoring function" deliverable in its usable form:
    ground truth, predictions, the trace universe and the scoring config, all
    read off disk. It is also how the pipeline proves its own report is
    reproducible rather than a side effect of the run that made it.

    A run configured for the LLM description judge is re-scored in similarity
    mode when no judge is injected — an LLM judge is not reproducible offline,
    and pretending otherwise would make this check meaningless.
    """
    run_dir = Path(run_dir)
    paths = paths or ArtifactPaths()
    ground_truth = _load(Issueboard, run_dir / paths.ground_truth_issueboard)
    predicted = _load(Issueboard, run_dir / paths.predicted_issueboard)
    ablated = _load(TraceDataset, run_dir / paths.ablated_traces)
    report = _load(BenchmarkReport, run_dir / paths.report)

    raw_cfg = json.loads((run_dir / paths.pipeline_config).read_text())
    scoring = ScoringConfig.model_validate(raw_cfg.get("scoring") or {})
    if scoring.description_mode == "judge" and judge is None:
        scoring = scoring.model_copy(update={"description_mode": "similarity"})

    return score(
        ground_truth,
        predicted,
        scoring,
        report.base_rates,
        trace_ids=[t.trace_id for t in ablated.traces],
        engine_config=report.engine_config,
        judge=judge,
    )


def _check(name: str, fn) -> DeliverableCheck:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - a failed deliverable is reported, not raised
        return DeliverableCheck(name=name, ok=False, detail=f"{type(exc).__name__}: {exc}")
    return DeliverableCheck(name=name, ok=True, detail=detail)


def check_deliverables(
    run_dir: str | Path,
    *,
    min_traces: int,
    paths: ArtifactPaths | None = None,
    judge: DescriptionJudge | None = None,
) -> list[DeliverableCheck]:
    run_dir = Path(run_dir)
    paths = paths or ArtifactPaths()

    def trace_file_scale() -> str:
        payload = json.loads((run_dir / paths.engine_input).read_text())
        traces = payload.get("traces", [])
        if len(traces) < min_traces:
            raise AssertionError(
                f"{paths.engine_input} carries {len(traces)} traces, the deliverable "
                f"needs at least {min_traces}"
            )
        return f"{len(traces)} traces in {paths.engine_input} (>= {min_traces})"

    def trace_file_schema() -> str:
        payload = assert_export_file_clean(run_dir / paths.engine_input)
        return f"{len(payload['traces'])} traces parse as Trace and name no ground truth"

    def issueboard_in() -> str:
        board = _load(Issueboard, run_dir / paths.seed_issueboard)
        if board.source != "seed":
            raise AssertionError(f"seed board source is {board.source!r}")
        return f"seed issueboard with {len(board.issues)} issue(s)"

    def issueboard_out() -> str:
        seed = _load(Issueboard, run_dir / paths.seed_issueboard)
        predicted = _load(Issueboard, run_dir / paths.predicted_issueboard)
        if predicted.source != "engine_predicted":
            raise AssertionError(f"predicted board source is {predicted.source!r}")
        predicted_ids = {i.error_id for i in predicted.issues}
        missing = sorted({i.error_id for i in seed.issues} - predicted_ids)
        if missing:
            raise AssertionError(
                f"the returned board dropped seed issue(s) {missing} — the deliverable is "
                f"the UPDATED board, not a replacement"
            )
        dangling = sorted({o.error_id for o in predicted.occurrences} - predicted_ids)
        if dangling:
            raise AssertionError(f"occurrences reference unknown issues: {dangling}")
        return (
            f"updated board: {len(predicted.issues)} issue(s) "
            f"({len(seed.issues)} carried over from the seed), "
            f"{len(predicted.occurrences)} occurrence(s)"
        )

    def standalone_scoring() -> str:
        rescored = rescore_from_disk(run_dir, paths=paths, judge=judge)
        original = _load(BenchmarkReport, run_dir / paths.report)
        drift = {
            key: (value, rescored.headline.get(key))
            for key, value in original.headline.items()
            if abs(value - rescored.headline.get(key, float("nan"))) > 1e-9
        }
        if drift:
            raise AssertionError(f"re-scoring from disk did not reproduce the report: {drift}")
        return f"score() re-run from artifacts reproduced all {len(original.headline)} headlines"

    def lineage() -> str:
        manifest = _load(RunManifest, run_dir / paths.manifest)
        traces = _load(TraceDataset, run_dir / paths.traces)
        ablated = _load(TraceDataset, run_dir / paths.ablated_traces)
        if ablated.parent_dataset_id != traces.dataset_id:
            raise AssertionError(
                f"the ablated dataset points at {ablated.parent_dataset_id!r}, not at the "
                f"traces it came from ({traces.dataset_id!r})"
            )
        if manifest.dataset_ids.get("report") != _load(
            BenchmarkReport, run_dir / paths.report
        ).report_id:
            raise AssertionError("the manifest does not name the report it shipped with")
        return f"lineage intact across {len(manifest.dataset_ids)} datasets"

    return [
        _check("traces_file_scale", trace_file_scale),
        _check("traces_file_schema_and_leak_free", trace_file_schema),
        _check("issueboard_in", issueboard_in),
        _check("issueboard_out_updated", issueboard_out),
        _check("standalone_scoring_entrypoint", standalone_scoring),
        _check("dataset_lineage", lineage),
    ]


__all__ = [
    "DeliverableCheck",
    "ExportLeak",
    "check_deliverables",
    "rescore_from_disk",
]
