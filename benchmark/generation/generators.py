"""The three atomic generators + the top-level entrypoint
(docs/architecture/02-input-generation.md).

- generate_safe_inputs:      [D, V_D]        -> D x V_D single-turn prompts.
- generate_adversarial_inputs: [A_c, V_AC], [A_F] -> (A_c x V_AC) + A_F adversarial inputs.
- assemble_multi_turn:       [P], [P_A], D1, D2 -> (P x D1) + (P_A x D2) scenarios.
- generate_inputs:           GenerationConfig -> InputDataset (top-level).

Every InputSpec produced here carries full provenance (dim_id, variation,
persona_id, fixed_adversarial_id) and a deterministic input_id computed from
that provenance — Phase 5's stratified split depends on it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from benchmark.generation.cache import DEFAULT_CACHE_DIR, DiskExpansionCache
from benchmark.generation.expander import OpenAIPromptExpander, PromptExpander
from benchmark.schemas.inputs import Dimension, GenerationConfig, InputDataset, InputSpec, Persona
from benchmark.schemas.io import content_hash, stamp_dataset_id


def _input_id(
    kind: str,
    dim_id: str,
    variation: str,
    persona_id: str | None = None,
    fixed_adversarial_id: str | None = None,
) -> str:
    """Deterministic input_id from grid-cell provenance.

    Same (kind, dim_id, variation, persona_id, fixed_adversarial_id) always
    produces the same id, regardless of iteration order.
    """
    payload = {
        "kind": kind,
        "dim_id": dim_id,
        "variation": variation,
        "persona_id": persona_id,
        "fixed_adversarial_id": fixed_adversarial_id,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    return f"{kind}-{digest}"


def _require_persona_kind(personas: list[Persona], expected_kind: str, field_name: str) -> None:
    """Guard against a misconfigured YAML swapping `personas`/`adversarial_personas`."""
    for persona in personas:
        if persona.kind != expected_kind:
            raise ValueError(
                f"{field_name} must contain only kind={expected_kind!r} personas, "
                f"got persona_id={persona.persona_id!r} with kind={persona.kind!r}"
            )


def generate_safe_inputs(
    dims: list[Dimension], expander: PromptExpander, seed: int, *, expand: bool = True
) -> list[InputSpec]:
    """[D, V_D] -> D x V_D single-turn prompts.

    The expander turns each (dim, variation) grid cell into a concrete,
    natural user message. Pass expand=False to build the grid-cell identity
    (dim_id/variation/input_id) only, without spending an expander call per
    cell — used when only the pool's identity is needed (e.g. multi_turn-only
    generation, where the single-turn prompt text itself is never emitted).
    """
    out: list[InputSpec] = []
    for dim in dims:
        for variation in dim.variations:
            prompt = expander.expand(dim, variation, seed) if expand else None
            out.append(
                InputSpec(
                    input_id=_input_id("safe", dim.dim_id, variation),
                    mode="single_turn",
                    dim_id=dim.dim_id,
                    variation=variation,
                    prompt=prompt,
                )
            )
    return out


def generate_adversarial_inputs(
    custom_dims: list[Dimension],
    fixed_library: list[InputSpec],
    expander: PromptExpander,
    seed: int,
    *,
    expand: bool = True,
) -> list[InputSpec]:
    """[A_c, V_AC], [A_F] -> (A_c x V_AC) + A_F adversarial inputs.

    Custom adversarial dims are LLM-expanded like safe dims (expand=False
    skips that call, same as generate_safe_inputs); the fixed library is
    always a passthrough (its `prompt` text is reused as-is, no expander
    call either way), re-stamped with a deterministic input_id and its
    fixed_adversarial_id provenance.
    """
    out: list[InputSpec] = []
    for dim in custom_dims:
        for variation in dim.variations:
            prompt = expander.expand(dim, variation, seed) if expand else None
            out.append(
                InputSpec(
                    input_id=_input_id("adv", dim.dim_id, variation),
                    mode="single_turn",
                    dim_id=dim.dim_id,
                    variation=variation,
                    prompt=prompt,
                )
            )
    for entry in fixed_library:
        library_id = entry.fixed_adversarial_id or entry.input_id
        out.append(
            InputSpec(
                input_id=_input_id("fixed", entry.dim_id, entry.variation, None, library_id),
                mode="single_turn",
                dim_id=entry.dim_id,
                variation=entry.variation,
                fixed_adversarial_id=library_id,
                prompt=entry.prompt,
            )
        )
    return out


def assemble_multi_turn(
    personas: list[Persona],
    adversarial_personas: list[Persona],
    safe_pool: list[InputSpec],
    adversarial_pool: list[InputSpec],
    expander: PromptExpander,
    seed: int,
) -> list[InputSpec]:
    """[P],[P_A],D1,D2 -> (P x D1)+(P_A x D2) persona-crossed scenarios.

    Target personas P are crossed with the safe scenario pool D1; adversarial
    personas P_A are crossed with the adversarial scenario pool D2. Each
    resulting InputSpec carries a persona_id + scenario brief — no literal
    prompt (that's produced downstream by the Stage II user-simulator).

    Raises ValueError if `personas` contains a non-target persona or
    `adversarial_personas` contains a non-adversarial one — a misconfigured
    YAML swapping the two lists would otherwise silently cross an adversarial
    persona with the safe pool (or vice versa).
    """
    _require_persona_kind(personas, "target", "personas")
    _require_persona_kind(adversarial_personas, "adversarial", "adversarial_personas")

    out: list[InputSpec] = []
    for persona, pool in ((p, safe_pool) for p in personas):
        for item in pool:
            scenario = expander.expand_scenario(persona, item.dim_id, item.variation, seed)
            out.append(
                InputSpec(
                    input_id=_input_id(
                        "mt", item.dim_id, item.variation, persona.persona_id,
                        item.fixed_adversarial_id,
                    ),
                    mode="multi_turn",
                    dim_id=item.dim_id,
                    variation=item.variation,
                    persona_id=persona.persona_id,
                    fixed_adversarial_id=item.fixed_adversarial_id,
                    scenario=scenario,
                )
            )
    for persona, pool in ((p, adversarial_pool) for p in adversarial_personas):
        for item in pool:
            scenario = expander.expand_scenario(persona, item.dim_id, item.variation, seed)
            out.append(
                InputSpec(
                    input_id=_input_id(
                        "mt", item.dim_id, item.variation, persona.persona_id,
                        item.fixed_adversarial_id,
                    ),
                    mode="multi_turn",
                    dim_id=item.dim_id,
                    variation=item.variation,
                    persona_id=persona.persona_id,
                    fixed_adversarial_id=item.fixed_adversarial_id,
                    scenario=scenario,
                )
            )
    return out


def generate_inputs(
    cfg: GenerationConfig,
    expander: PromptExpander | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    now: Callable[[], datetime] | None = None,
) -> InputDataset:
    """GenerationConfig -> InputDataset (top-level entrypoint).

    mode="single_turn": only the single-turn safe+adversarial pools.
    mode="multi_turn":  only the persona-crossed multi-turn scenarios.
    mode="mixed":       both, concatenated.

    Expansions are routed through a disk cache keyed on this config's content
    hash, so identical config + seed reruns are byte-identical and hit the
    cache instead of re-invoking the expander.
    """
    expander = expander or OpenAIPromptExpander()
    config_hash = content_hash(cfg)
    cached = DiskExpansionCache(expander=expander, config_hash=config_hash, cache_dir=cache_dir)

    # Single-turn prompt text is only needed when it's actually emitted. A
    # multi_turn-only config only needs each pool item's (dim_id, variation)
    # identity for expand_scenario, so skip the (wasted) expand() calls.
    needs_single_turn_text = cfg.mode in ("single_turn", "mixed")

    safe_pool = generate_safe_inputs(
        cfg.safe_dims, cached, cfg.seed, expand=needs_single_turn_text
    )
    adversarial_pool = generate_adversarial_inputs(
        cfg.adversarial_dims, cfg.fixed_adversarial, cached, cfg.seed,
        expand=needs_single_turn_text,
    )

    inputs: list[InputSpec] = []
    if needs_single_turn_text:
        inputs.extend(safe_pool)
        inputs.extend(adversarial_pool)
    if cfg.mode in ("multi_turn", "mixed"):
        inputs.extend(
            assemble_multi_turn(
                cfg.personas,
                cfg.adversarial_personas,
                safe_pool,
                adversarial_pool,
                cached,
                cfg.seed,
            )
        )

    clock = now or (lambda: datetime.now(UTC))
    dataset = InputDataset(created_at=clock(), generation_config=cfg, inputs=inputs)
    return stamp_dataset_id(dataset)
