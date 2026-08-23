"""Gate: disk expansion cache keyed by (config hash, dim_id, variation, persona_id, seed).

Same config + seed reruns must hit the cache and reproduce byte-identical
output without re-invoking the underlying expander.
"""

from benchmark.generation.cache import DiskExpansionCache
from benchmark.generation.expander import MockPromptExpander
from benchmark.schemas.inputs import Dimension, Persona


def make_dim(dim_id: str = "d1") -> Dimension:
    return Dimension(dim_id=dim_id, name="query_topic", kind="safe", variations=["refunds"])


def make_persona(persona_id: str = "p1") -> Persona:
    return Persona(persona_id=persona_id, name="Angry Alice", kind="target", description="…")


def test_cache_miss_then_hit_returns_identical_text(tmp_path):
    inner = MockPromptExpander()
    cache = DiskExpansionCache(expander=inner, cache_dir=tmp_path, config_hash="cfg1")
    dim = make_dim()

    first = cache.expand(dim, "refunds", seed=7)
    assert len(inner.calls) == 1

    second = cache.expand(dim, "refunds", seed=7)
    assert second == first
    # cache hit: inner expander not invoked again
    assert len(inner.calls) == 1


def test_cache_persists_across_instances(tmp_path):
    dim = make_dim()
    inner_a = MockPromptExpander()
    cache_a = DiskExpansionCache(expander=inner_a, cache_dir=tmp_path, config_hash="cfg1")
    text_a = cache_a.expand(dim, "refunds", seed=7)

    inner_b = MockPromptExpander()
    cache_b = DiskExpansionCache(expander=inner_b, cache_dir=tmp_path, config_hash="cfg1")
    text_b = cache_b.expand(dim, "refunds", seed=7)

    assert text_a == text_b
    assert len(inner_b.calls) == 0  # served entirely from disk


def test_different_config_hash_is_a_different_cache_entry(tmp_path):
    dim = make_dim()
    inner = MockPromptExpander()
    cache1 = DiskExpansionCache(expander=inner, cache_dir=tmp_path, config_hash="cfg1")
    cache2 = DiskExpansionCache(expander=inner, cache_dir=tmp_path, config_hash="cfg2")

    cache1.expand(dim, "refunds", seed=7)
    cache2.expand(dim, "refunds", seed=7)
    assert len(inner.calls) == 2  # both are cache misses — distinct keys


def test_different_seed_is_a_different_cache_entry(tmp_path):
    dim = make_dim()
    inner = MockPromptExpander()
    cache = DiskExpansionCache(expander=inner, cache_dir=tmp_path, config_hash="cfg1")
    a = cache.expand(dim, "refunds", seed=1)
    b = cache.expand(dim, "refunds", seed=2)
    assert a != b
    assert len(inner.calls) == 2


def test_expand_scenario_cache_hit(tmp_path):
    inner = MockPromptExpander()
    cache = DiskExpansionCache(expander=inner, cache_dir=tmp_path, config_hash="cfg1")
    persona = make_persona()

    first = cache.expand_scenario(persona, "d1", "refunds", seed=7)
    second = cache.expand_scenario(persona, "d1", "refunds", seed=7)
    assert first == second
    assert len(inner.calls) == 1


def test_expand_and_expand_scenario_do_not_collide(tmp_path):
    """Same (dim_id, variation, seed) via the two call shapes must not share a cache slot."""
    inner = MockPromptExpander()
    cache = DiskExpansionCache(expander=inner, cache_dir=tmp_path, config_hash="cfg1")
    dim = make_dim()
    persona = make_persona()

    cache.expand(dim, "refunds", seed=7)
    cache.expand_scenario(persona, "d1", "refunds", seed=7)
    assert len(inner.calls) == 2


def test_cache_writes_files_under_cache_dir(tmp_path):
    inner = MockPromptExpander()
    cache = DiskExpansionCache(expander=inner, cache_dir=tmp_path, config_hash="cfg1")
    cache.expand(make_dim(), "refunds", seed=7)
    assert any(tmp_path.iterdir())
