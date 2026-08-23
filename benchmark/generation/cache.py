"""Disk cache for LLM expansions, under data/generation_cache/ by default.

Keyed by (config content hash, dim_id, variation, persona_id, seed) — plus a
`kind` tag distinguishing `expand` from `expand_scenario` calls so the two
call shapes never collide on the same cache slot. Identical config + seed
reruns hit the cache and reproduce byte-identical output without
re-invoking the underlying (possibly network-backed) expander.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark.generation.expander import PromptExpander
from benchmark.schemas.inputs import Dimension, Persona

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "generation_cache"


@dataclass(frozen=True)
class ExpansionCacheKey:
    config_hash: str
    kind: str  # "expand" | "expand_scenario"
    dim_id: str
    variation: str
    persona_id: str | None
    seed: int


def _key_id(key: ExpansionCacheKey) -> str:
    canonical = json.dumps(asdict(key), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:24]


class DiskExpansionCache:
    """Wraps a PromptExpander with an on-disk cache.

    Implements the same call shapes as PromptExpander, so it is a drop-in
    replacement anywhere a PromptExpander is expected.
    """

    def __init__(
        self,
        expander: PromptExpander,
        config_hash: str,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ) -> None:
        self._expander = expander
        self._config_hash = config_hash
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_compute(self, key: ExpansionCacheKey, compute) -> str:
        path = self._cache_dir / f"{_key_id(key)}.json"
        if path.exists():
            return json.loads(path.read_text())["text"]
        text = compute()
        payload = json.dumps({"key": asdict(key), "text": text}, sort_keys=True, indent=2)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(payload)
        tmp_path.replace(path)
        return text

    def expand(self, dim: Dimension, variation: str, seed: int) -> str:
        key = ExpansionCacheKey(self._config_hash, "expand", dim.dim_id, variation, None, seed)
        return self._load_or_compute(key, lambda: self._expander.expand(dim, variation, seed))

    def expand_scenario(self, persona: Persona, dim_id: str, variation: str, seed: int) -> str:
        key = ExpansionCacheKey(
            self._config_hash, "expand_scenario", dim_id, variation, persona.persona_id, seed
        )
        return self._load_or_compute(
            key, lambda: self._expander.expand_scenario(persona, dim_id, variation, seed)
        )
