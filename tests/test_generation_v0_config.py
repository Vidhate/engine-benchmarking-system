"""Gate: configs/generation/v0.yaml parses into a GenerationConfig whose
single-turn N >= 300 — computed without any LLM/network calls."""

from datetime import UTC, datetime
from pathlib import Path

from benchmark.generation.config_loader import load_generation_config
from benchmark.generation.expander import MockPromptExpander
from benchmark.generation.generators import generate_inputs
from benchmark.schemas.inputs import GenerationConfig

V0_PATH = Path(__file__).parent.parent / "configs" / "generation" / "v0.yaml"


def load_v0() -> GenerationConfig:
    return load_generation_config(V0_PATH)


def test_v0_config_exists_and_parses():
    cfg = load_v0()
    assert isinstance(cfg, GenerationConfig)


def test_v0_single_turn_n_at_least_300():
    cfg = load_v0()
    n = (
        sum(len(d.variations) for d in cfg.safe_dims)
        + sum(len(d.variations) for d in cfg.adversarial_dims)
        + len(cfg.fixed_adversarial)
    )
    assert n >= 300


def test_v0_has_orthogonal_safe_dims_with_variations():
    cfg = load_v0()
    assert len(cfg.safe_dims) >= 1
    for dim in cfg.safe_dims:
        assert dim.kind == "safe"
        assert len(dim.variations) > 0
        assert len(dim.variations) == len(set(dim.variations)), f"{dim.dim_id} has duplicates"


def test_v0_has_custom_adversarial_dims_with_variations():
    cfg = load_v0()
    assert len(cfg.adversarial_dims) >= 1
    for dim in cfg.adversarial_dims:
        assert dim.kind == "adversarial"
        assert len(dim.variations) > 0
        assert len(dim.variations) == len(set(dim.variations))


def test_v0_fixed_adversarial_entries_well_formed():
    cfg = load_v0()
    assert len(cfg.fixed_adversarial) >= 70
    ids = [entry.input_id for entry in cfg.fixed_adversarial]
    assert len(ids) == len(set(ids)), "fixed_adversarial input_ids must be unique"
    for entry in cfg.fixed_adversarial:
        assert entry.mode == "single_turn"
        assert entry.prompt
        assert entry.variation


def test_v0_personas_present_for_mixed_mode():
    cfg = load_v0()
    assert cfg.mode == "mixed"
    assert len(cfg.personas) >= 1
    assert all(p.kind == "target" for p in cfg.personas)
    assert len(cfg.adversarial_personas) >= 1
    assert all(p.kind == "adversarial" for p in cfg.adversarial_personas)


def test_v0_dim_ids_unique_across_safe_and_adversarial():
    cfg = load_v0()
    ids = [d.dim_id for d in cfg.safe_dims] + [d.dim_id for d in cfg.adversarial_dims]
    assert len(ids) == len(set(ids))


def test_v0_generates_full_dataset_via_mock_expander(tmp_path):
    """End-to-end: v0.yaml -> generate_inputs (mocked) matches every count formula."""
    cfg = load_v0()
    ds = generate_inputs(
        cfg,
        expander=MockPromptExpander(),
        cache_dir=tmp_path,
        now=lambda: datetime(2026, 8, 23, tzinfo=UTC),
    )

    single_turn_n = (
        sum(len(d.variations) for d in cfg.safe_dims)
        + sum(len(d.variations) for d in cfg.adversarial_dims)
        + len(cfg.fixed_adversarial)
    )
    d1 = sum(len(d.variations) for d in cfg.safe_dims)
    d2 = sum(len(d.variations) for d in cfg.adversarial_dims) + len(cfg.fixed_adversarial)
    multi_turn_n = len(cfg.personas) * d1 + len(cfg.adversarial_personas) * d2

    assert sum(1 for i in ds.inputs if i.mode == "single_turn") == single_turn_n
    assert sum(1 for i in ds.inputs if i.mode == "multi_turn") == multi_turn_n
    assert len(ds.inputs) == single_turn_n + multi_turn_n
    assert ds.dataset_id

    ids = [i.input_id for i in ds.inputs]
    assert len(ids) == len(set(ids))
