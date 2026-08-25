"""On-disk artifact layout and the run manifest.

Every stage writes its artifact as JSON under `<artifacts_root>/<run_id>/`, and
`manifest.json` ties them together: config hashes, dataset ids and their
lineage, model ids, counts, per-stage timings, and any warnings the run raised.

Lineage is the point. A `BenchmarkReport` on its own is a number; a report
whose `parent` chain runs back through the predicted board, the export the
Engine read, the ablated dataset, the raw traces, the inputs, and the
generation config is a number you can argue with.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class StageTiming(BaseModel):
    stage: str
    seconds: float


class ArtifactPaths(BaseModel):
    """Filenames, relative to the run directory, in stage order."""

    pipeline_config: str = "pipeline_config.json"
    inputs: str = "inputs.json"
    outputs: str = "outputs.json"
    traces: str = "raw_traces.json"
    ablated_traces: str = "ablated_traces.json"
    #: THE assignment deliverable: the JSON trace file handed to the Engine.
    #: Leak-stripped, >=300 traces at full scale.
    engine_input: str = "traces.json"
    ablation_records: str = "ablation_records.json"
    ablation_split: str = "ablation_split.json"
    seed_issueboard: str = "seed_issueboard.json"
    ground_truth_issueboard: str = "ground_truth_issueboard.json"
    #: The Engine's updated board, verbatim — the assignment's deliverable.
    predicted_issueboard: str = "predicted_issueboard.json"
    #: What the report's numbers actually describe: the Engine's delta over the
    #: seed board, restricted to the real trace universe (pipeline/scoring.py).
    scored_issueboard: str = "scored_issueboard.json"
    engine_raw_output: str = "engine_raw_output.json"
    report: str = "report.json"
    summary: str = "report.md"
    deliverables: str = "deliverables.json"
    manifest: str = "manifest.json"
    #: Which stages have completed, and what each consumed and produced. Written
    #: incrementally, stage by stage, so a run that dies leaves a usable record;
    #: read only by `--resume` (benchmark/pipeline/resume.py). Not a dataset and
    #: not a deliverable — an interrupted run's bookmark.
    stage_checkpoints: str = "stage_checkpoints.json"


class RunManifest(BaseModel):
    run_id: str
    created_at: datetime | None = None
    config_hashes: dict[str, str] = Field(default_factory=dict)
    dataset_ids: dict[str, str] = Field(default_factory=dict)
    lineage: dict[str, str | None] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    timings: list[StageTiming] = Field(default_factory=list)
    artifacts: ArtifactPaths = Field(default_factory=ArtifactPaths)
    #: Which implementation actually ran each swappable stage. The fake
    #: ablation stage must be visible in the record of any run that used it.
    stages: dict[str, str] = Field(default_factory=dict)
    harness_stats: dict[str, int] = Field(default_factory=dict)
    dropped_errors: list[str] = Field(default_factory=list)
    #: Stages this run loaded from disk instead of executing (`--resume`), in
    #: pipeline order. A resumed run's timings describe the stages that actually
    #: ran, so a reader needs this list to know which numbers are missing — and
    #: which artifacts were produced by an earlier process.
    resumed_stages: list[str] = Field(default_factory=list)
    #: One line per resumable stage: resumed (and why it matched) or re-run (and
    #: why it did not). Present only on a `--resume` run.
    resume_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(t.seconds for t in self.timings)


class RunArtifacts:
    """Writes a run's artifacts under one directory, by logical name."""

    def __init__(self, run_dir: str | Path, paths: ArtifactPaths | None = None):
        self.run_dir = Path(run_dir)
        self.paths = paths or ArtifactPaths()
        self.written: dict[str, str] = {}

    def relative(self, name: str) -> str:
        try:
            return getattr(self.paths, name)
        except AttributeError as exc:
            raise KeyError(f"unknown artifact {name!r}") from exc

    def path(self, name: str) -> Path:
        return self.run_dir / self.relative(name)

    def _record(self, name: str) -> Path:
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.written[name] = self.relative(name)
        return path

    def write_model(self, name: str, model: BaseModel) -> Path:
        path = self._record(name)
        path.write_text(model.model_dump_json(indent=2) + "\n")
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        path = self._record(name)
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self._record(name)
        path.write_text(text)
        return path

    def claim(self, name: str) -> Path:
        """Reserve a path for something another component writes (the export)."""
        return self._record(name)


@contextmanager
def stage_timer(stage: str, timings: list[StageTiming]):
    """Record a stage's wall clock — including when the stage raises.

    A run that dies inside the Engine pass is exactly the run whose timings you
    want, so the record is written in a `finally`.
    """
    started = time.time()
    try:
        yield
    finally:
        timings.append(StageTiming(stage=stage, seconds=round(time.time() - started, 3)))


def utcnow() -> datetime:
    return datetime.now(UTC)
