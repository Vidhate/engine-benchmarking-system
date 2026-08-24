"""On-disk artifact layout + the manifest that ties a run together.

A benchmark number is only worth as much as its provenance: which configs,
which datasets, which model, how long. The manifest is that provenance, and it
lives next to the artifacts it describes.
"""

from __future__ import annotations

import json

import pytest

from benchmark.pipeline.manifest import (
    ArtifactPaths,
    RunArtifacts,
    RunManifest,
    StageTiming,
    stage_timer,
)
from benchmark.schemas import Issueboard


def test_the_engine_input_file_is_the_assignment_deliverable():
    """The file handed to the Engine is the assignment's `traces.json`."""
    assert ArtifactPaths().engine_input == "traces.json"


def test_every_artifact_path_is_relative_and_distinct():
    paths = ArtifactPaths().model_dump()
    assert all(not p.startswith("/") for p in paths.values())
    assert len(set(paths.values())) == len(paths)


def test_artifacts_write_under_the_run_dir(tmp_path):
    artifacts = RunArtifacts(tmp_path / "run")
    written = artifacts.write_model("seed_issueboard", Issueboard(source="seed"))
    assert written == tmp_path / "run" / "seed_issueboard.json"
    assert json.loads(written.read_text())["source"] == "seed"


def test_writing_creates_the_run_dir(tmp_path):
    artifacts = RunArtifacts(tmp_path / "deep" / "run")
    artifacts.write_json("engine_raw_output", {"a": 1})
    assert (tmp_path / "deep" / "run" / "engine_raw_output.json").exists()


def test_an_unknown_artifact_name_is_refused(tmp_path):
    with pytest.raises(KeyError, match="not_a_thing"):
        RunArtifacts(tmp_path).path("not_a_thing")


def test_relative_paths_are_recorded_for_the_manifest(tmp_path):
    artifacts = RunArtifacts(tmp_path)
    artifacts.write_text("summary", "# hi\n")
    assert artifacts.written["summary"] == "report.md"


def test_the_stage_timer_records_wall_clock():
    timings: list[StageTiming] = []
    with stage_timer("generation", timings):
        pass
    assert [t.stage for t in timings] == ["generation"]
    assert timings[0].seconds >= 0


def test_the_stage_timer_records_even_when_the_stage_raises():
    """A run that died in the Engine stage should still say how long it got."""
    timings: list[StageTiming] = []
    with pytest.raises(RuntimeError), stage_timer("engine", timings):
        raise RuntimeError("boom")
    assert [t.stage for t in timings] == ["engine"]


def test_the_manifest_round_trips(tmp_path):
    manifest = RunManifest(
        run_id="r",
        config_hashes={"pipeline": "abc"},
        dataset_ids={"inputs": "i1"},
        lineage={"traces": "i1"},
        models={"engine": "gpt-5.1-mini"},
        counts={"traces": 7},
        timings=[StageTiming(stage="generation", seconds=1.0)],
        warnings=["something looked off"],
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json(indent=2))
    assert RunManifest.model_validate_json(path.read_text()) == manifest
