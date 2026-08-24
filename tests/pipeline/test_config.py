"""Phase 7 — the pipeline's config surface.

Everything the pipeline knows about a run comes from one YAML file: which
generation config to expand, which taxonomy the Engine is shown, where the two
black-box apps live, and the per-stage knobs. Nothing about either app is
hardcoded in pipeline code.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchmark.pipeline.config import (
    PipelineConfig,
    load_engine_app_config,
    load_pipeline_config,
    load_seed_board,
    load_taxonomy,
)
from benchmark.schemas import OTHER_CATEGORY_ID, ErrorCategory, Issueboard
from benchmark.schemas.io import content_hash

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_checked_in_mini_config_parses():
    cfg = load_pipeline_config(REPO_ROOT / "configs" / "pipeline" / "mini.yaml")
    assert cfg.run_id
    assert cfg.root == REPO_ROOT
    assert cfg.resolve(cfg.generation_config).exists()
    assert cfg.resolve(cfg.taxonomy).exists()
    assert cfg.resolve(cfg.target_app_config).exists()
    assert cfg.resolve(cfg.engine_app_config).exists()


def test_the_checked_in_full_config_parses_and_is_full_scale():
    cfg = load_pipeline_config(REPO_ROOT / "configs" / "pipeline" / "full.yaml")
    # The ruled full-scale settings (docs/execution-plan Phase 7 + apps/engine README).
    assert cfg.engine.analysis_concurrency == 16
    assert cfg.engine.recursion_limit >= 10_000
    assert cfg.deliverables.min_traces >= 300
    assert cfg.resolve(cfg.generation_config).name == "v0.yaml"


def test_the_full_config_is_scoped_to_the_documented_run_size():
    """v0.yaml in `mixed` mode yields 1230 inputs — four times the scope the
    plan describes, and the multi-turn tail is the expensive part (a
    conversation costs several app invocations, not one)."""
    from benchmark.generation.config_loader import load_generation_config
    from benchmark.pipeline.runner import slice_inputs
    from benchmark.schemas import InputDataset, InputSpec

    cfg = load_pipeline_config(REPO_ROOT / "configs" / "pipeline" / "full.yaml")
    generation = load_generation_config(cfg.resolve(cfg.generation_config))

    single = sum(len(d.variations) for d in generation.safe_dims)
    single += sum(len(d.variations) for d in generation.adversarial_dims)
    single += len(generation.fixed_adversarial)
    multi = len(generation.personas) * sum(
        len(d.variations) for d in generation.safe_dims
    ) + len(generation.adversarial_personas) * (
        sum(len(d.variations) for d in generation.adversarial_dims)
        + len(generation.fixed_adversarial)
    )
    assert (single, multi) == (310, 920), "v0.yaml changed shape; re-check the slice"

    grid = InputDataset(
        inputs=[
            InputSpec(input_id=f"st-{i}", mode="single_turn", dim_id="d", variation="v")
            for i in range(single)
        ]
        + [
            InputSpec(input_id=f"mt-{i}", mode="multi_turn", dim_id="d", variation="v")
            for i in range(multi)
        ]
    )
    sliced = slice_inputs(grid, cfg)
    modes = {m: sum(1 for s in sliced.inputs if s.mode == m) for m in ("single_turn", "multi_turn")}
    assert modes["single_turn"] == 310, "every single-turn input runs"
    assert modes["multi_turn"] == 30, "the multi-turn tail is sampled, not run whole"
    assert len(sliced.inputs) >= cfg.deliverables.min_traces


def test_the_checked_in_submission_config_parses_and_is_full_scale():
    cfg = load_pipeline_config(REPO_ROOT / "configs" / "pipeline" / "submission.yaml")
    assert cfg.engine.analysis_concurrency == 16
    assert cfg.engine.recursion_limit >= 10_000
    assert cfg.deliverables.min_traces >= 300
    assert cfg.resolve(cfg.generation_config).name == "v0.yaml"
    assert cfg.input_modes == ["single_turn"]
    assert cfg.max_inputs_per_mode == {"single_turn": 300, "multi_turn": 0}


def test_the_submission_config_slices_to_exactly_300_single_turn_inputs():
    """The assignment's own deliverable number, single-turn only — no sampled
    multi-turn tail the way full.yaml has one (mirrors
    test_the_full_config_is_scoped_to_the_documented_run_size)."""
    from benchmark.generation.config_loader import load_generation_config
    from benchmark.pipeline.runner import slice_inputs
    from benchmark.schemas import InputDataset, InputSpec

    cfg = load_pipeline_config(REPO_ROOT / "configs" / "pipeline" / "submission.yaml")
    generation = load_generation_config(cfg.resolve(cfg.generation_config))

    single = sum(len(d.variations) for d in generation.safe_dims)
    single += sum(len(d.variations) for d in generation.adversarial_dims)
    single += len(generation.fixed_adversarial)
    multi = len(generation.personas) * sum(
        len(d.variations) for d in generation.safe_dims
    ) + len(generation.adversarial_personas) * (
        sum(len(d.variations) for d in generation.adversarial_dims)
        + len(generation.fixed_adversarial)
    )
    assert (single, multi) == (310, 920), "v0.yaml changed shape; re-check the slice"

    grid = InputDataset(
        inputs=[
            InputSpec(input_id=f"st-{i}", mode="single_turn", dim_id="d", variation="v")
            for i in range(single)
        ]
        + [
            InputSpec(input_id=f"mt-{i}", mode="multi_turn", dim_id="d", variation="v")
            for i in range(multi)
        ]
    )
    sliced = slice_inputs(grid, cfg)
    modes = {m: sum(1 for s in sliced.inputs if s.mode == m) for m in ("single_turn", "multi_turn")}
    assert modes == {"single_turn": 300, "multi_turn": 0}
    assert len(sliced.inputs) == 300 == cfg.deliverables.min_traces


def test_run_dir_is_run_id_under_artifacts_root(tmp_path):
    cfg = PipelineConfig(run_id="r1", generation_config="g.yaml", artifacts_root="data/p")
    cfg = cfg.with_root(tmp_path)
    assert cfg.run_dir == tmp_path / "data" / "p" / "r1"


def test_absolute_paths_are_left_alone(tmp_path):
    cfg = PipelineConfig(run_id="r", generation_config=str(tmp_path / "g.yaml"))
    cfg = cfg.with_root(Path("/nowhere"))
    assert cfg.resolve(cfg.generation_config) == tmp_path / "g.yaml"


def test_root_is_not_part_of_the_config_hash(tmp_path):
    """Two machines running the same config must stamp the same manifest hash."""
    cfg = PipelineConfig(run_id="r", generation_config="g.yaml")
    assert content_hash(cfg.with_root(tmp_path)) == content_hash(cfg.with_root(Path("/elsewhere")))
    assert "root" not in cfg.model_dump()


def test_the_root_is_found_by_walking_up_to_pyproject(tmp_path):
    root = tmp_path / "proj"
    (root / "configs" / "pipeline").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n")
    path = root / "configs" / "pipeline" / "x.yaml"
    path.write_text(yaml.safe_dump({"run_id": "x", "generation_config": "configs/g.yaml"}))
    assert load_pipeline_config(path).root == root


def test_an_explicit_root_wins(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text(yaml.safe_dump({"run_id": "x", "generation_config": "g.yaml"}))
    assert load_pipeline_config(path, root=tmp_path / "other").root == tmp_path / "other"


def test_a_missing_config_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="pipeline config not found"):
        load_pipeline_config(tmp_path / "nope.yaml")


def test_taxonomy_loads_as_categories_with_the_other_escape_hatch():
    categories = load_taxonomy(REPO_ROOT / "configs" / "taxonomy.yaml")
    assert all(isinstance(c, ErrorCategory) for c in categories)
    assert OTHER_CATEGORY_ID in {c.category_id for c in categories}


def test_taxonomy_without_other_is_refused(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        yaml.safe_dump({"categories": [{"category_id": "a", "name": "a", "description": "d"}]})
    )
    with pytest.raises(ValueError, match=OTHER_CATEGORY_ID):
        load_taxonomy(path)


def test_engine_app_config_comes_only_from_yaml():
    app = load_engine_app_config(REPO_ROOT / "configs" / "engine.yaml")
    assert app.base_url.startswith("http")
    assert app.assistant_id
    assert app.model_configurable_key


def test_no_seed_file_yields_an_empty_seed_board():
    board = load_seed_board(None)
    assert isinstance(board, Issueboard)
    assert board.source == "seed"
    assert board.issues == [] and board.occurrences == []
    assert board.board_id, "even an empty seed board carries a content-hash id"


def test_a_provided_seed_file_is_loaded_and_forced_to_source_seed(tmp_path):
    path = tmp_path / "seed.json"
    path.write_text(
        Issueboard(
            source="ground_truth",
            issues=[
                {
                    "error_id": "S1",
                    "title": "t",
                    "description": "d",
                    "category_id": "other",
                    "severity": "low",
                }
            ],
        ).model_dump_json()
    )
    board = load_seed_board(path)
    assert board.source == "seed"
    assert [i.error_id for i in board.issues] == ["S1"]


def test_a_seed_board_never_carries_injection_mode(tmp_path):
    """injection_mode is ground-truth-side only; a seed board goes to the Engine."""
    path = tmp_path / "seed.json"
    path.write_text(
        Issueboard(
            source="seed",
            issues=[
                {
                    "error_id": "S1",
                    "title": "t",
                    "description": "d",
                    "category_id": "other",
                    "severity": "low",
                    "injection_mode": "replay_edit",
                }
            ],
        ).model_dump_json()
    )
    with pytest.raises(ValueError, match="injection_mode"):
        load_seed_board(path)
