"""`--resume <run_dir>`: which stages get skipped, and which must never be.

The thing under test is a decision, not a computation — "do I already have
this?" — so every stage here is a counting wrapper around the offline fakes
from `benchmark.pipeline.fakes`. What each test asserts is how many times each
stage was *entered*, which is precisely the property a resume exists to change.

Two failure modes are worth more than the happy path, and both are covered
below:

* **resuming into the wrong directory.** A config that does not match the run
  directory must be refused, not merged. Half the artifacts from one experiment
  and half from another, under a manifest naming one config, is a report nobody
  can reconstruct — and it looks completely normal.
* **resuming a faked run.** The artifacts of a `--fake-*` run are byte-for-byte
  ordinary. If a resume read them without the checkpoint's record of what
  produced them, the FAKED warning would vanish and a wiring run would read as
  a result.
"""

from __future__ import annotations

import io
import json

import pytest

from benchmark.pipeline.fakes import (
    FakeEngineInvoker,
    FakeExpander,
    FakeHarnessFactory,
    fake_run_ablation,
)
from benchmark.pipeline.manifest import ArtifactPaths
from benchmark.pipeline.progress import Progress
from benchmark.pipeline.resume import ResumeMismatch, ResumeState
from benchmark.pipeline.runner import run_pipeline

PATHS = ArtifactPaths()


class Counters:
    """One counting wrapper per resumable stage, plus the objects to pass in."""

    def __init__(self) -> None:
        self.generation = 0
        self.harness = 0
        self.ablation = 0
        self.engine = 0
        self.ablation_fails = False

        outer = self

        class CountingExpander(FakeExpander):
            def expand(self, *a, **kw):
                outer.generation += 1
                return super().expand(*a, **kw)

            def expand_scenario(self, *a, **kw):
                outer.generation += 1
                return super().expand_scenario(*a, **kw)

        class CountingHarnessFactory(FakeHarnessFactory):
            def __call__(self, cfg, store):
                harness = super().__call__(cfg, store)
                run_batch = harness.run_batch

                def counted(inputs):
                    outer.harness += 1
                    return run_batch(inputs)

                harness.run_batch = counted
                return harness

        class CountingEngine(FakeEngineInvoker):
            def __call__(self, **kw):
                outer.engine += 1
                return super().__call__(**kw)

        def counting_ablation(**kw):
            outer.ablation += 1
            if outer.ablation_fails:
                raise RuntimeError("ablation stage killed (simulating an interrupted run)")
            return fake_run_ablation(**kw)

        # The marker, not the name, is what the runner reads — and a wrapper
        # closure would otherwise drop the fake's identity and lose the warning.
        counting_ablation.is_pipeline_fake = True

        self.expander = CountingExpander()
        self.harness_factory = CountingHarnessFactory()
        self.engine_invoker = CountingEngine()
        self.ablation_stage = counting_ablation

    def snapshot(self) -> dict[str, int]:
        return {
            "generation": self.generation,
            "harness": self.harness,
            "ablation": self.ablation,
            "engine": self.engine,
        }


def go(cfg, counters: Counters, *, resume: bool = False, progress: Progress | None = None):
    return run_pipeline(
        cfg,
        ablation_stage=counters.ablation_stage,
        engine_invoker=counters.engine_invoker,
        harness_factory=counters.harness_factory,
        expander=counters.expander,
        progress=progress or Progress(quiet=True),
        resume=resume,
    )


@pytest.fixture
def fresh(mini_cfg):
    """One complete offline run, and the counters that watched it."""
    counters = Counters()
    run = go(mini_cfg, counters)
    return mini_cfg, counters, run


# ------------------------------------------------- a fresh run is not a resume

def test_a_run_without_the_flag_executes_every_stage(fresh):
    _, counters, _ = fresh
    assert counters.harness == 1
    assert counters.ablation == 1
    assert counters.engine == 1
    assert counters.generation > 0


def test_a_run_without_the_flag_still_writes_its_checkpoints(fresh):
    """The run that needs resuming is the one nobody started with --resume."""
    cfg, _, _ = fresh
    entries = json.loads((cfg.run_dir / PATHS.stage_checkpoints).read_text())
    assert sorted(entries) == ["ablation", "engine", "generation", "harness"]
    assert entries["harness"]["inputs_dataset_id"]
    assert entries["engine"]["predicted_board_id"]


def test_a_run_without_the_flag_reports_no_resumed_stages(fresh):
    _, _, run = fresh
    assert run.manifest.resumed_stages == []
    assert "Resumed from disk" not in run.markdown


# --------------------------------------------- killed mid-run, then resumed

@pytest.fixture
def killed_after_harness(mini_cfg):
    """Generation and the harness completed; the ablation stage then died."""
    counters = Counters()
    counters.ablation_fails = True
    with pytest.raises(RuntimeError, match="killed"):
        go(mini_cfg, counters)
    return mini_cfg


def test_a_killed_run_leaves_only_the_stages_that_finished(killed_after_harness):
    entries = json.loads((killed_after_harness.run_dir / PATHS.stage_checkpoints).read_text())
    assert sorted(entries) == ["generation", "harness"]


def test_resume_skips_generation_and_harness_and_reruns_the_rest(killed_after_harness):
    counters = Counters()
    run = go(killed_after_harness, counters, resume=True)

    assert counters.generation == 0, "the expansion calls were paid for by the killed run"
    assert counters.harness == 0, "the harness batch was already on disk"
    assert counters.ablation == 1, "the stage that died must run"
    assert counters.engine == 1, "a stage that never ran cannot be resumed"
    assert run.manifest.resumed_stages == ["generation", "harness"]


def test_a_resumed_run_still_produces_a_complete_report(killed_after_harness):
    run = go(killed_after_harness, Counters(), resume=True)
    assert run.report.report_id
    assert [c.name for c in run.deliverables if not c.ok] == []
    assert "Resumed from disk" in run.markdown
    assert "generation, harness" in run.markdown


def test_the_resumed_traces_are_the_ones_the_killed_run_collected(killed_after_harness):
    before = json.loads((killed_after_harness.run_dir / PATHS.traces).read_text())
    run = go(killed_after_harness, Counters(), resume=True)
    assert run.traces.dataset_id == before["dataset_id"]
    assert run.ablated.parent_dataset_id == before["dataset_id"]


# --------------------------------------------- a complete run, resumed again

def test_a_full_artifact_resume_reruns_only_scoring_and_render(fresh):
    cfg, _, first = fresh
    counters = Counters()
    second = go(cfg, counters, resume=True)

    assert counters.snapshot() == {"generation": 0, "harness": 0, "ablation": 0, "engine": 0}
    assert second.manifest.resumed_stages == ["generation", "harness", "ablation", "engine"]
    # Scoring is not resumable and did run: the report is stamped fresh, and it
    # reproduces the original because the inputs to it are identical.
    assert second.report.headline == first.report.headline
    assert [t.stage for t in second.manifest.timings] == ["scoring"]


def test_a_full_artifact_resume_keeps_the_boards_it_loaded(fresh):
    cfg, _, first = fresh
    second = go(cfg, Counters(), resume=True)
    assert second.predicted.board_id == first.predicted.board_id
    assert second.ground_truth.board_id == first.ground_truth.board_id
    assert second.split == first.split


def test_a_resumed_engine_stage_never_claims_a_model_confirmation(fresh):
    cfg, _, _ = fresh
    second = go(cfg, Counters(), resume=True)
    assert second.manifest.models["engine_recorded"].startswith("resumed")
    assert any("resumed from disk" in w for w in second.manifest.warnings)


def test_resuming_a_faked_run_does_not_launder_away_the_faked_warning(fresh):
    """The artifacts of a faked run look exactly like a real one's."""
    cfg, _, first = fresh
    assert any(w.startswith("FAKED stage(s)") for w in first.manifest.warnings)

    second = go(cfg, Counters(), resume=True)
    faked = [w for w in second.manifest.warnings if w.startswith("FAKED stage(s)")]
    assert faked, "a resumed faked run reported itself as a clean one"
    assert "ablation" in faked[0] and "engine_invoker" in faked[0] and "harness" in faked[0]
    assert "resumed from disk" in second.manifest.stages["ablation"]


# ----------------------------------------------------- artifacts vs checkpoint

def test_artifacts_without_a_checkpoint_are_not_trusted(fresh):
    """A file on disk is not evidence that the stage that writes it finished."""
    cfg, _, _ = fresh
    (cfg.run_dir / PATHS.stage_checkpoints).unlink()
    counters = Counters()
    run = go(cfg, counters, resume=True)
    assert counters.snapshot() == {
        "generation": counters.generation,
        "harness": 1,
        "ablation": 1,
        "engine": 1,
    }
    assert run.manifest.resumed_stages == []


def test_a_missing_artifact_reruns_its_stage_but_not_the_earlier_ones(fresh):
    cfg, _, _ = fresh
    (cfg.run_dir / PATHS.predicted_issueboard).unlink()
    counters = Counters()
    run = go(cfg, counters, resume=True)
    assert counters.engine == 1
    assert counters.ablation == 0
    assert run.manifest.resumed_stages == ["generation", "harness", "ablation"]


def test_a_missing_export_file_reruns_the_whole_ablation_stage(fresh):
    """The five ablation artifacts are one statement; four of them is not a
    resumable stage, it is ground truth that may not describe its corpus."""
    cfg, _, _ = fresh
    (cfg.run_dir / PATHS.engine_input).unlink()
    counters = Counters()
    run = go(cfg, counters, resume=True)
    assert counters.ablation == 1
    assert counters.harness == 0, "the stage above it was intact and must not be re-bought"
    assert "ablation" not in run.manifest.resumed_stages


def test_a_rerun_stage_that_reproduces_its_output_does_not_invalidate_the_next_one(fresh):
    """Identity here is content, not provenance — and that is the point.

    The re-run ablation stage is deterministic, so it stamps the same
    `ablated.dataset_id`. The Engine's checkpoint names that id, so the board
    on disk still describes exactly the corpus that now exists, and re-running
    25 minutes of model time to arrive at the same input would be waste, not
    rigour. A stage that came back DIFFERENT would move the id and the Engine
    would re-run — which is the case the check actually exists for.
    """
    cfg, _, _ = fresh
    (cfg.run_dir / PATHS.engine_input).unlink()
    counters = Counters()
    run = go(cfg, counters, resume=True)
    assert counters.ablation == 1
    assert counters.engine == 0
    assert "engine" in run.manifest.resumed_stages


def test_an_ablation_that_comes_back_different_does_force_the_engine_to_rerun(fresh):
    cfg, _, _ = fresh
    (cfg.run_dir / PATHS.engine_input).unlink()
    counters = Counters()
    # A different ablation seed produces a different split, so a different
    # ablated dataset — the Engine's stored board no longer describes it.
    moved = cfg.model_copy(
        update={"ablation": cfg.ablation.model_copy(update={"control_fraction": 0.5})}
    ).with_root(cfg.root)
    # …which the config guard would refuse outright, so the guard is what is
    # demonstrated here: you cannot get a mixed corpus this way at all.
    with pytest.raises(ResumeMismatch):
        go(moved, counters, resume=True)
    assert counters.engine == 0


def test_a_tampered_predicted_board_reruns_the_engine(fresh):
    cfg, _, _ = fresh
    path = cfg.run_dir / PATHS.predicted_issueboard
    payload = json.loads(path.read_text())
    payload["board_id"] = "0000000000000000"
    path.write_text(json.dumps(payload))
    counters = Counters()
    go(cfg, counters, resume=True)
    assert counters.engine == 1


def test_traces_missing_from_the_store_rerun_the_harness(fresh):
    """The dataset is a manifest; the ablation stage reads trace bodies out of
    the store, and a body that is gone fails several stages later."""
    cfg, _, _ = fresh
    store_dir = cfg.run_dir / "trace_store"
    victim = sorted(store_dir.glob("*.json"))[0]
    victim.unlink()
    counters = Counters()
    run = go(cfg, counters, resume=True)
    assert counters.harness == 1
    assert "harness" not in run.manifest.resumed_stages
    assert run.manifest.resumed_stages[0] == "generation"


# ------------------------------------------------------------- hard failures

def test_a_config_that_does_not_match_the_directory_is_refused(fresh):
    cfg, _, _ = fresh
    other = cfg.model_copy(
        update={"ablation": cfg.ablation.model_copy(update={"seed": cfg.ablation.seed + 1})}
    ).with_root(cfg.root)
    with pytest.raises(ResumeMismatch, match="ablation"):
        go(other, Counters(), resume=True)


def test_the_refusal_names_the_hashes_that_differ(fresh):
    cfg, _, _ = fresh
    other = cfg.model_copy(
        update={"engine": cfg.engine.model_copy(update={"model": "some-other-model"})}
    ).with_root(cfg.root)
    with pytest.raises(ResumeMismatch) as exc:
        go(other, Counters(), resume=True)
    assert "engine" in str(exc.value)
    assert "->" in str(exc.value), "the message must name both hashes, not just complain"


def test_a_refused_resume_does_not_touch_the_directory(fresh):
    """Refusing after overwriting `pipeline_config.json` would destroy the one
    record that identifies the directory."""
    cfg, _, _ = fresh
    before = (cfg.run_dir / PATHS.pipeline_config).read_text()
    other = cfg.model_copy(
        update={"scoring": cfg.scoring.model_copy(update={"severity_alpha": 0.9})}
    ).with_root(cfg.root)
    with pytest.raises(ResumeMismatch):
        go(other, Counters(), resume=True)
    assert (cfg.run_dir / PATHS.pipeline_config).read_text() == before


def test_a_changed_generation_config_is_refused_even_though_no_hash_moves(mini_cfg, tmp_path):
    """The gap a hash comparison alone leaves open.

    The pipeline config names a generation YAML by PATH, so editing that file
    changes the whole corpus without moving a single pipeline hash. The
    authoritative record of what it said is the copy `inputs.json` embeds, and
    that is what the guard compares against.
    """
    copied = tmp_path / "generation.yaml"
    copied.write_text((mini_cfg.root / mini_cfg.generation_config).read_text())
    cfg = mini_cfg.model_copy(update={"generation_config": str(copied)}).with_root(mini_cfg.root)
    go(cfg, Counters())

    # Same path, same pipeline config, different contents: a different grid
    # seed means every expansion, and so every input_id, changes.
    copied.write_text(copied.read_text().replace("seed: 20260823", "seed: 11111111", 1))
    with pytest.raises(ResumeMismatch, match="generation config has changed"):
        go(cfg, Counters(), resume=True)


def test_resuming_a_directory_that_is_not_a_run_is_refused(tmp_path, mini_cfg):
    empty = tmp_path / "not-a-run"
    empty.mkdir()
    state = ResumeState(empty, enabled=True)
    with pytest.raises(ResumeMismatch, match="pipeline_config"):
        state.assert_same_run(mini_cfg)


def test_resume_is_inert_when_it_is_switched_off(mini_cfg, tmp_path):
    """Every guard above is skipped for an ordinary run — including the ones
    that would otherwise raise on an empty directory."""
    state = ResumeState(tmp_path, enabled=False)
    state.assert_same_run(mini_cfg)
    assert state.try_generation(mini_cfg) is None
    assert state.loaded == []


# ------------------------------------------------------------------ progress

def test_skipped_stages_are_announced_on_the_progress_stream(fresh):
    cfg, _, _ = fresh
    stream = io.StringIO()
    go(cfg, Counters(), resume=True, progress=Progress(stream=stream))
    printed = stream.getvalue()
    for stage in ("generation", "harness", "ablation", "engine"):
        assert f"↻ {stage} (resumed from disk)" in printed, printed
    # A resumed stage must not also print a completion banner: "✓ harness
    # (elapsed 0s)" says it ran and was instant, when it did not run at all.
    assert "✓ harness" not in printed
    assert "✓ scoring" in printed, "scoring always runs"
