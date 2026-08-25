"""Gate: the submission grid is big enough, and the slice takes exactly 400/0.

`configs/pipeline/submission.yaml` asks for 400 single-turn inputs. If the grid
behind it holds fewer, `slice_inputs` does not fail — `max_inputs_per_mode` is a
CAP, not a target, so a 310-cell grid under a cap of 400 quietly yields 310 and
the run produces a 310-trace deliverable that nobody noticed was short. This
file is the thing that notices, and it does so without a single LLM call:
every count here is computed off the YAML.

The companion checks in tests/test_generation_v0_config.py still cover v0,
which remains the config with a multi-turn grid.
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from benchmark.generation.config_loader import load_generation_config
from benchmark.generation.expander import MockPromptExpander
from benchmark.generation.generators import generate_inputs
from benchmark.pipeline.config import load_pipeline_config
from benchmark.pipeline.runner import slice_inputs
from benchmark.schemas.inputs import GenerationConfig

ROOT = Path(__file__).parent.parent
GENERATION = ROOT / "configs" / "generation" / "submission.yaml"
PIPELINE = ROOT / "configs" / "pipeline" / "submission.yaml"

#: What configs/pipeline/submission.yaml asks the harness to collect.
TARGET_SINGLE_TURN = 400


@pytest.fixture(scope="module")
def cfg() -> GenerationConfig:
    return load_generation_config(GENERATION)


def grid_n(cfg: GenerationConfig) -> int:
    """(D x V_D) + (A_c x V_AC) + A_F — the single-turn cell count."""
    return (
        sum(len(d.variations) for d in cfg.safe_dims)
        + sum(len(d.variations) for d in cfg.adversarial_dims)
        + len(cfg.fixed_adversarial)
    )


# --------------------------------------------------------------- the grid

def test_the_submission_grid_parses(cfg):
    assert isinstance(cfg, GenerationConfig)


def test_the_single_turn_grid_clears_the_400_the_pipeline_asks_for(cfg):
    n = grid_n(cfg)
    assert n >= TARGET_SINGLE_TURN, (
        f"the grid yields {n} single-turn cells but configs/pipeline/submission.yaml caps "
        f"at {TARGET_SINGLE_TURN} — a cap above the grid is silently ignored, and the run "
        f"would ship a short deliverable"
    )


def test_the_grid_keeps_a_margin_over_the_cap(cfg):
    """Slack, so the sample drops cells rather than the cap being the grid.
    Same shape as v0's 310 -> 300."""
    assert grid_n(cfg) > TARGET_SINGLE_TURN


def test_the_grid_is_single_turn_only_so_nothing_is_generated_to_be_discarded(cfg):
    """The pipeline takes `multi_turn: 0`. A `mixed` config would still expand
    every persona-crossed cell first and pay for it."""
    assert cfg.mode == "single_turn"
    assert cfg.personas == []
    assert cfg.adversarial_personas == []


def test_every_dimension_has_unique_variations(cfg):
    for dim in (*cfg.safe_dims, *cfg.adversarial_dims):
        assert dim.variations, dim.dim_id
        assert len(dim.variations) == len(set(dim.variations)), f"{dim.dim_id} has duplicates"


def test_dim_ids_are_unique_across_safe_and_adversarial(cfg):
    ids = [d.dim_id for d in cfg.safe_dims] + [d.dim_id for d in cfg.adversarial_dims]
    assert len(ids) == len(set(ids))


def test_the_dimension_kinds_are_not_swapped(cfg):
    assert all(d.kind == "safe" for d in cfg.safe_dims)
    assert all(d.kind == "adversarial" for d in cfg.adversarial_dims)


def test_the_fixed_adversarial_library_is_well_formed(cfg):
    assert len(cfg.fixed_adversarial) >= 100
    ids = [e.input_id for e in cfg.fixed_adversarial]
    assert len(ids) == len(set(ids)), "fixed_adversarial input_ids must be unique"
    prompts = [e.prompt for e in cfg.fixed_adversarial]
    assert len(prompts) == len(set(prompts)), "two A_F entries carry the same prompt"
    for entry in cfg.fixed_adversarial:
        assert entry.mode == "single_turn"
        assert entry.prompt and entry.variation


def test_the_fixed_adversarial_library_is_a_full_base_x_framing_cross(cfg):
    """The v0 pattern: hand-authored attacks crossed with framings, with no
    gaps — a partial cross would make one attack over-represented."""
    bases, framings = set(), set()
    for entry in cfg.fixed_adversarial:
        _, base, framing = entry.variation.split("__")
        bases.add(base)
        framings.add(framing)
    assert len(bases) * len(framings) == len(cfg.fixed_adversarial)
    assert len(framings) == 8


def test_the_variations_are_content_and_not_numbered_padding(cfg):
    """Each variation is a grid cell an LLM expands into a distinct user
    message; `topic_27` would produce a prompt that tests nothing."""
    for dim in (*cfg.safe_dims, *cfg.adversarial_dims):
        for variation in dim.variations:
            assert not variation.rstrip("0123456789_").endswith(dim.dim_id), (
                f"{dim.dim_id}/{variation} looks like numbered filler"
            )
            # Low bar on purpose: `thai` and `greek` are perfectly good cells
            # on the language axis. The padding check above is the substantive
            # one; this only catches an empty or one-letter placeholder.
            assert len(variation) >= 4, f"{dim.dim_id}/{variation} is too short to mean anything"


def test_the_app_context_is_carried_over_from_v0(cfg):
    v0 = load_generation_config(ROOT / "configs" / "generation" / "v0.yaml")
    assert cfg.app_context == v0.app_context, (
        "the submission grid describes the same target app; a drifted app_context would "
        "generate inputs for a different one"
    )


def test_every_v0_variation_survives_into_the_submission_grid(cfg):
    """Grown from v0, not rewritten: the existing cells are known-good."""
    v0 = load_generation_config(ROOT / "configs" / "generation" / "v0.yaml")
    mine = {d.dim_id: set(d.variations) for d in (*cfg.safe_dims, *cfg.adversarial_dims)}
    for dim in (*v0.safe_dims, *v0.adversarial_dims):
        assert dim.dim_id in mine, f"{dim.dim_id} was dropped"
        missing = set(dim.variations) - mine[dim.dim_id]
        assert not missing, f"{dim.dim_id} lost v0 variations: {sorted(missing)}"


# ------------------------------------------------------------- the slice

def test_the_pipeline_config_points_at_this_grid():
    pipeline = load_pipeline_config(PIPELINE)
    assert Path(pipeline.generation_config).name == GENERATION.name


def test_the_pipeline_config_asks_for_400_single_turn_and_no_multi_turn():
    pipeline = load_pipeline_config(PIPELINE)
    assert pipeline.input_modes == ["single_turn"]
    assert pipeline.max_inputs_per_mode == {
        "single_turn": TARGET_SINGLE_TURN,
        "multi_turn": 0,
    }
    # `max_inputs` would be applied AFTER the per-mode cap and could undo it.
    assert pipeline.max_inputs is None


def test_min_traces_leaves_a_quarter_of_the_corpus_as_quarantine_slack():
    pipeline = load_pipeline_config(PIPELINE)
    assert pipeline.deliverables.min_traces == 300
    slack = TARGET_SINGLE_TURN - pipeline.deliverables.min_traces
    assert slack / TARGET_SINGLE_TURN == pytest.approx(0.25)


def test_the_slice_yields_exactly_400_single_turn_and_0_multi_turn(cfg, tmp_path):
    """The whole claim, end to end, through the real slicing code and a
    network-free expander."""
    pipeline = load_pipeline_config(PIPELINE)
    dataset = generate_inputs(
        cfg,
        expander=MockPromptExpander(),
        cache_dir=tmp_path,
        now=lambda: datetime(2026, 8, 24, tzinfo=UTC),
    )
    assert len(dataset.inputs) == grid_n(cfg)

    sliced = slice_inputs(dataset, pipeline)
    modes = [spec.mode for spec in sliced.inputs]
    assert modes.count("single_turn") == TARGET_SINGLE_TURN
    assert modes.count("multi_turn") == 0
    assert len(sliced.inputs) == TARGET_SINGLE_TURN
    ids = [spec.input_id for spec in sliced.inputs]
    assert len(ids) == len(set(ids))


def test_the_slice_is_reproducible(cfg, tmp_path):
    """Two arms of a comparison have to see the same 400 of the 420."""
    pipeline = load_pipeline_config(PIPELINE)
    dataset = generate_inputs(
        cfg,
        expander=MockPromptExpander(),
        cache_dir=tmp_path,
        now=lambda: datetime(2026, 8, 24, tzinfo=UTC),
    )
    first = slice_inputs(dataset, pipeline)
    second = slice_inputs(dataset, pipeline)
    assert first.dataset_id == second.dataset_id
    assert [s.input_id for s in first.inputs] == [s.input_id for s in second.inputs]


def test_the_grid_math_in_the_header_matches_the_file(cfg):
    """The comment block is the first thing anyone reads; a stale one is worse
    than none."""
    header = "\n".join(
        line for line in GENERATION.read_text().splitlines() if line.startswith("#")
    )
    d = sum(len(x.variations) for x in cfg.safe_dims)
    a = sum(len(x.variations) for x in cfg.adversarial_dims)
    f = len(cfg.fixed_adversarial)
    assert f"= {d} + {a} + {f} = {d + a + f}" in header


def test_the_pipeline_header_names_the_same_numbers():
    text = PIPELINE.read_text()
    cfg = load_generation_config(GENERATION)
    assert str(grid_n(cfg)) in text
    assert str(TARGET_SINGLE_TURN) in text


def test_the_yaml_holds_no_duplicate_keys():
    """`yaml.safe_load` silently keeps the last of a duplicated key, so a
    generated config can parse cleanly and still be wrong."""
    raw = yaml.compose(GENERATION.read_text())
    top_level = [key.value for key, _ in raw.value]
    assert len(top_level) == len(set(top_level)), top_level
