"""Stage checkpointing — `--resume <run_dir>`.

A full-scale run is four to five hours, and three of its four expensive stages
are things you would never want to buy twice: ~400 expansion calls, ~2 h of
harness batch, ~25 min of Engine time. When the fourth one dies — and the ones
that die are the ones that touch a network — the whole run currently starts
again from an empty directory.

This module is the alternative. On `--resume`, each stage asks: *are my
artifacts already on disk, and are they the artifacts THIS run would have
produced?* If yes, they are loaded and the stage is skipped. If no, the stage
runs. Scoring and rendering are seconds, so they always re-run.

## Two different kinds of "no", and they must not be confused

**A stage's artifacts are missing or inconsistent** — the run was killed
mid-harness, the export never got written, the ablated dataset points at some
other corpus. That is ordinary: the stage re-runs. Nothing is lost and nothing
is at risk.

**The config does not describe this run directory** — a different generation
config, different ablation knobs, a different Engine model. That is NOT
ordinary, and it must never quietly re-run a stage into someone else's
directory: half the artifacts would come from one experiment and half from
another, the manifest would name a single config, and the resulting report
would be a number nobody could argue with because nobody could reconstruct it.
So `assert_same_run` hard-fails with the specific hashes that differ.

## Why there is a checkpoint file and not just lineage

Three of the four stages carry their own lineage: `traces.parent_dataset_id`
names the inputs, `ablated.parent_dataset_id` names the traces. The Engine
stage has none — an `Issueboard` is not a dataset with a parent, and a
predicted board cannot be checked against the corpus it was produced from,
because naming a trace that does not exist is a *legitimate* Engine output
(`scored.phantom_occurrences` exists precisely for that). Testing the board
against the trace universe would re-run a 25-minute stage over a real result.

So each stage appends an entry to `stage_checkpoints.json` when it completes,
naming what it consumed and what it produced. A resume needs BOTH: the
checkpoint (this stage really finished, here is what it was fed) and the
artifacts (and here they still are, still consistent). Artifacts with no
checkpoint re-run — a file that exists is not evidence that the stage that
writes it ran to completion.

The checkpoint also carries the handful of facts that live only in memory
during a run and would otherwise be lost across a resume: the harness's own
stats, the ablation stage's dropped errors, the Engine's recorded model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.generation.config_loader import load_generation_config
from benchmark.pipeline.config import PipelineConfig, config_hashes
from benchmark.pipeline.contracts import EngineInvocation
from benchmark.pipeline.export import export_traces
from benchmark.pipeline.manifest import ArtifactPaths, RunManifest
from benchmark.pipeline.progress import Progress
from benchmark.schemas import (
    AblationRecord,
    AblationSplit,
    InputDataset,
    Issueboard,
    OutputDataset,
    TraceDataset,
)
from benchmark.schemas.io import content_hash
from benchmark.tracing.store import TraceStore

log = logging.getLogger("benchmark.pipeline.resume")

#: The stages `--resume` can skip, in pipeline order. Scoring and rendering are
#: deliberately absent: they cost seconds, they are the stages most likely to
#: have changed since the run being resumed, and re-running them is how a
#: resumed run still gets a report that describes *this* code.
RESUMABLE_STAGES = ("generation", "harness", "ablation", "engine")

#: The config hashes compared against the manifest. `pipeline` is excluded from
#: the manifest half of the check because `--resume` legitimately rewrites
#: `artifacts_root` and `run_id` (that is how it points at a directory), and
#: both are fields of the hashed model. It IS compared against the run
#: directory's own `pipeline_config.json`, where those two fields can be
#: normalized away first — see `assert_same_run`.
STABLE_HASH_KEYS = ("ablation", "engine", "scoring", "harness")


class ResumeMismatch(RuntimeError):
    """`--resume` was pointed at a run directory this config does not describe.

    Always fatal. The alternative is a directory holding artifacts from two
    different experiments, and a manifest that names one config for both.
    """


@dataclass
class ResumedAblation:
    """An `AblationResult` rebuilt from disk.

    Structural, like everything else crossing that seam — it satisfies the
    pinned contract in `benchmark.pipeline.contracts` and is checked against it
    by `assert_ablation_result` exactly as a live result is.
    """

    ablated: TraceDataset
    ground_truth: Issueboard
    records: list[AblationRecord]
    split: AblationSplit
    export_path: str
    dropped_errors: list[str] = field(default_factory=list)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


class ResumeState:
    """Per-stage resume decisions for one run directory.

    Constructed disabled for an ordinary run, in which case every `try_*`
    method returns `None` immediately and `record` still writes the checkpoints
    — so a run that was never resumed is still resumable *later*, which is the
    only version of this feature that helps the run that unexpectedly dies.
    """

    def __init__(
        self,
        run_dir: str | Path,
        paths: ArtifactPaths | None = None,
        *,
        enabled: bool = False,
        progress: Progress | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.paths = paths or ArtifactPaths()
        self.enabled = enabled
        self.progress = progress or Progress(quiet=True)
        #: Stages loaded from disk this run, in the order they were skipped.
        self.loaded: list[str] = []
        #: Human-readable "why" for every resumable stage, for the manifest.
        self.notes: list[str] = []
        #: Harness stats recovered from a resumed harness checkpoint. Stays
        #: empty when the harness actually ran — the live object's own stats
        #: are the truth then.
        self.harness_stats: dict[str, int] = {}
        self._checkpoints: dict[str, Any] | None = None

    # ------------------------------------------------------- checkpoints

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / self.paths.stage_checkpoints

    def checkpoints(self) -> dict[str, Any]:
        if self._checkpoints is None:
            try:
                loaded = _read_json(self.checkpoint_path)
            except (OSError, ValueError):
                loaded = {}
            self._checkpoints = loaded if isinstance(loaded, dict) else {}
        return self._checkpoints

    def record(self, stage: str, **payload: Any) -> None:
        """Mark a stage complete. Called after the stage's artifacts are written.

        Order matters: an artifact on disk with no checkpoint entry means the
        process died between the two, and the stage re-runs. The reverse — a
        checkpoint with no artifact — is caught by the presence check.
        """
        entries = dict(self.checkpoints())
        entries[stage] = payload
        self._checkpoints = entries
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.write_text(json.dumps(entries, indent=2, default=str) + "\n")

    # ------------------------------------------------------------ helpers

    def _path(self, name: str) -> Path:
        return self.run_dir / getattr(self.paths, name)

    def _missing(self, *names: str) -> list[str]:
        return [getattr(self.paths, n) for n in names if not self._path(n).exists()]

    def _hit(self, stage: str, detail: str) -> None:
        self.loaded.append(stage)
        self.notes.append(f"{stage}: resumed from disk ({detail})")
        self.progress.resumed(stage, detail)

    def _miss(self, stage: str, reason: str) -> None:
        self.notes.append(f"{stage}: re-run ({reason})")
        log.info("resume: re-running %s — %s", stage, reason)

    def _begin(self, stage: str, *artifacts: str) -> dict[str, Any] | None:
        """The checks every stage shares: enabled, checkpointed, artifacts present."""
        if not self.enabled:
            return None
        entry = self.checkpoints().get(stage)
        if not isinstance(entry, dict):
            self._miss(stage, "no completed-stage checkpoint in the run directory")
            return None
        missing = self._missing(*artifacts)
        if missing:
            self._miss(stage, f"artifact(s) not on disk: {', '.join(missing)}")
            return None
        return entry

    # ------------------------------------------------------------- safety

    def assert_same_run(self, cfg: PipelineConfig) -> None:
        """Hard-fail unless `cfg` is the config this run directory was built by.

        Compared on content hashes rather than on paths or file mtimes: two
        checkouts of the same commit must resume each other's runs, and an
        edited YAML at the same path must not.
        """
        if not self.enabled:
            return
        stored_path = self._path("pipeline_config")
        if not stored_path.exists():
            raise ResumeMismatch(
                f"{self.run_dir} has no {self.paths.pipeline_config} — it is not a pipeline "
                f"run directory (or the run died before it wrote one, in which case there is "
                f"nothing to resume: drop --resume)"
            )
        try:
            stored = PipelineConfig.model_validate(_read_json(stored_path))
        except Exception as exc:  # noqa: BLE001 - a corrupt config is a legible failure
            raise ResumeMismatch(
                f"{stored_path} does not parse as a PipelineConfig ({type(exc).__name__}: "
                f"{exc}) — refusing to resume into a run directory that cannot be identified"
            ) from exc

        # `--resume` rewrites exactly these two fields to point the config at
        # the directory, so they are normalized away before comparing; every
        # other field still has to match.
        stored = stored.model_copy(
            update={"artifacts_root": cfg.artifacts_root, "run_id": cfg.run_id}
        ).with_root(cfg.root)
        drift = {
            key: (recorded, value)
            for key, recorded in config_hashes(stored).items()
            if (value := config_hashes(cfg).get(key)) != recorded
        }
        if drift:
            raise ResumeMismatch(
                f"--resume {self.run_dir} was given a config that does not match the one the "
                f"run directory records. Differing hashes (on disk -> given): "
                f"{ {k: f'{a} -> {b}' for k, (a, b) in sorted(drift.items())} }. Resuming "
                f"would mix artifacts from two different experiments into one manifest. Run "
                f"without --resume to start a fresh run, or point --config at the config this "
                f"directory was built with."
            )

        # The generation config's *contents*, not just its path: the pipeline
        # config names `configs/generation/v0.yaml`, and editing that file
        # changes the corpus without changing a single pipeline hash. The
        # authoritative record of what it contained is the copy embedded in
        # `inputs.json` by the run itself.
        inputs_path = self._path("inputs")
        if inputs_path.exists():
            recorded = InputDataset.model_validate_json(inputs_path.read_text()).generation_config
            current = load_generation_config(cfg.resolve(cfg.generation_config))
            if content_hash(recorded) != content_hash(current):
                raise ResumeMismatch(
                    f"the generation config has changed since {self.run_dir} was built "
                    f"({content_hash(recorded)} on disk, {content_hash(current)} now). Every "
                    f"artifact in that directory describes the old grid; resuming would score "
                    f"a corpus generated from one config against ground truth from another."
                )

        manifest_path = self._path("manifest")
        if manifest_path.exists():
            manifest = RunManifest.model_validate_json(manifest_path.read_text())
            recorded_hashes = manifest.config_hashes
            stale = {
                key: (recorded_hashes[key], config_hashes(cfg)[key])
                for key in STABLE_HASH_KEYS
                if key in recorded_hashes and recorded_hashes[key] != config_hashes(cfg)[key]
            }
            if stale:
                raise ResumeMismatch(
                    f"{self.paths.manifest} in {self.run_dir} records different stage configs "
                    f"from the ones given (manifest -> given): "
                    f"{ {k: f'{a} -> {b}' for k, (a, b) in sorted(stale.items())} }"
                )

    # -------------------------------------------------------- Stage I

    def try_generation(self, cfg: PipelineConfig) -> InputDataset | None:
        entry = self._begin("generation", "inputs")
        if entry is None:
            return None
        inputs = InputDataset.model_validate_json(self._path("inputs").read_text())
        if entry.get("inputs_dataset_id") != inputs.dataset_id:
            self._miss(
                "generation",
                f"{self.paths.inputs} is not the dataset the checkpoint names "
                f"({inputs.dataset_id!r} on disk, {entry.get('inputs_dataset_id')!r} recorded)",
            )
            return None
        # The slice is a pipeline-layer decision and is NOT part of the
        # generation config's hash, so it is checked here against what is on
        # disk: a resume with a narrower `input_modes` or a smaller cap must
        # re-slice rather than silently reuse a wider corpus.
        modes = {spec.mode for spec in inputs.inputs}
        if cfg.input_modes and not modes <= set(cfg.input_modes):
            self._miss(
                "generation",
                f"the stored inputs carry mode(s) {sorted(modes - set(cfg.input_modes))} that "
                f"this config's slice excludes",
            )
            return None
        for mode, limit in (cfg.max_inputs_per_mode or {}).items():
            count = sum(1 for spec in inputs.inputs if spec.mode == mode)
            if count > limit:
                self._miss(
                    "generation",
                    f"the stored inputs carry {count} {mode} input(s), above this config's "
                    f"cap of {limit}",
                )
                return None
        if cfg.max_inputs is not None and len(inputs.inputs) > cfg.max_inputs:
            self._miss(
                "generation",
                f"the stored inputs carry {len(inputs.inputs)}, above max_inputs="
                f"{cfg.max_inputs}",
            )
            return None
        self._hit("generation", f"{len(inputs.inputs)} inputs, dataset_id={inputs.dataset_id[:12]}")
        return inputs

    # ------------------------------------------------------- Stage II

    def try_harness(
        self, inputs: InputDataset, store: TraceStore
    ) -> tuple[OutputDataset, TraceDataset] | None:
        entry = self._begin("harness", "outputs", "traces")
        if entry is None:
            return None
        outputs = OutputDataset.model_validate_json(self._path("outputs").read_text())
        traces = TraceDataset.model_validate_json(self._path("traces").read_text())
        for name, dataset in (("outputs", outputs), ("traces", traces)):
            if dataset.parent_dataset_id != inputs.dataset_id:
                self._miss(
                    "harness",
                    f"the stored {name} descend from {dataset.parent_dataset_id!r}, not from "
                    f"the inputs this run holds ({inputs.dataset_id!r})",
                )
                return None
        # The dataset is a manifest; the ablation stage reads trace BODIES out
        # of the store. A trace file listed here but absent from the store
        # would fail several stages later, inside replay.
        absent = [t.trace_id for t in traces.traces if not store.exists(t.trace_id)]
        if absent:
            self._miss(
                "harness",
                f"{len(absent)} of {len(traces.traces)} trace(s) are named in "
                f"{self.paths.traces} but missing from the trace store (first: {absent[0]!r})",
            )
            return None
        self.harness_stats = dict(entry.get("stats") or {})
        self._hit("harness", f"{len(traces.traces)} traces, stats={self.harness_stats}")
        return outputs, traces

    # ------------------------------------------------------ Stage III

    def try_ablation(self, traces: TraceDataset, export_path: Path) -> ResumedAblation | None:
        """All five artifacts, or the stage re-runs as a unit.

        Ablation is not divisible on resume. The ablated dataset, the ground
        truth board, the records, the split and the export file are one
        consistent statement about which errors were injected where; loading
        four of them and regenerating the fifth would produce ground truth that
        does not describe the corpus it is scored against.
        """
        entry = self._begin(
            "ablation",
            "ablated_traces",
            "ground_truth_issueboard",
            "ablation_records",
            "ablation_split",
            "engine_input",
        )
        if entry is None:
            return None
        ablated = TraceDataset.model_validate_json(self._path("ablated_traces").read_text())
        if ablated.parent_dataset_id != traces.dataset_id:
            self._miss(
                "ablation",
                f"the stored ablated dataset descends from {ablated.parent_dataset_id!r}, not "
                f"from the traces this run holds ({traces.dataset_id!r})",
            )
            return None
        if entry.get("ablated_dataset_id") != ablated.dataset_id:
            self._miss(
                "ablation",
                f"{self.paths.ablated_traces} is not the dataset the checkpoint names "
                f"({ablated.dataset_id!r} on disk, {entry.get('ablated_dataset_id')!r} recorded)",
            )
            return None
        ground_truth = Issueboard.model_validate_json(
            self._path("ground_truth_issueboard").read_text()
        )
        records = [
            AblationRecord.model_validate(raw)
            for raw in _read_json(self._path("ablation_records"))
        ]
        split = AblationSplit.model_validate_json(self._path("ablation_split").read_text())
        # The export is the one artifact that crosses to the Engine, and the
        # only one whose consistency the lineage fields say nothing about.
        try:
            exported = export_traces(_read_json(self._path("engine_input")))
        except Exception as exc:  # noqa: BLE001 - an unreadable export is a re-run, not a crash
            self._miss("ablation", f"{self.paths.engine_input} does not parse ({exc})")
            return None
        if len(exported) != len(ablated.traces):
            self._miss(
                "ablation",
                f"{self.paths.engine_input} carries {len(exported)} trace(s) but the ablated "
                f"dataset has {len(ablated.traces)} — the export and the dataset have diverged",
            )
            return None
        self._hit(
            "ablation",
            f"{len(records)} records, {len(ground_truth.issues)} ground-truth issue(s)",
        )
        return ResumedAblation(
            ablated=ablated,
            ground_truth=ground_truth,
            records=records,
            split=split,
            export_path=str(export_path),
            dropped_errors=list(entry.get("dropped_errors") or []),
        )

    # ------------------------------------------------------- Stage IV

    def try_engine(self, ablated: TraceDataset) -> EngineInvocation | None:
        """Atomic: the predicted board AND the raw output, or the Engine re-runs.

        Identity comes from the checkpoint, not from the board. A predicted
        board naming a trace that does not exist is a legitimate Engine output
        — phantom occurrences are counted, not treated as corruption — so
        checking the board against the trace universe would re-buy 25 minutes
        of model time over a perfectly good result.
        """
        entry = self._begin("engine", "predicted_issueboard", "engine_raw_output")
        if entry is None:
            return None
        if entry.get("ablated_dataset_id") != ablated.dataset_id:
            self._miss(
                "engine",
                f"the stored board was produced from ablated dataset "
                f"{entry.get('ablated_dataset_id')!r}, not from the one this run holds "
                f"({ablated.dataset_id!r})",
            )
            return None
        predicted = Issueboard.model_validate_json(
            self._path("predicted_issueboard").read_text()
        )
        if entry.get("predicted_board_id") != predicted.board_id:
            self._miss(
                "engine",
                f"{self.paths.predicted_issueboard} is not the board the checkpoint names "
                f"({predicted.board_id!r} on disk, {entry.get('predicted_board_id')!r} recorded)",
            )
            return None
        raw_output = _read_json(self._path("engine_raw_output"))
        self._hit(
            "engine",
            f"{len(predicted.issues)} issue(s), {len(predicted.occurrences)} occurrence(s)",
        )
        return EngineInvocation(
            board=predicted,
            raw_output=raw_output if isinstance(raw_output, dict) else {},
            seconds=float(entry.get("seconds") or 0.0),
            thread_id=str(entry.get("thread_id") or ""),
            # Deliberately `[]` and never consulted: the runner skips the
            # model-verification block entirely for a resumed Engine stage and
            # says so in a warning, rather than re-asserting a confirmation
            # this process did not obtain. The original run's finding is kept
            # on the checkpoint under `recorded_models`.
            recorded_models=[],
            trace_count=int(entry.get("trace_count") or len(ablated.traces)),
        )

    # ------------------------------------------------------------ summary

    def resumed(self, stage: str) -> bool:
        return stage in self.loaded

    def warning(self) -> str | None:
        """The one line the report and the manifest carry about this resume."""
        if not self.loaded:
            return None
        return (
            f"RESUMED stage(s): {', '.join(self.loaded)}. Those artifacts were loaded from "
            f"{self.run_dir} rather than produced by this process; their timings are the "
            f"original run's, not this one's."
        )
