"""Step 0 — the input-level control/ablate split (locked design decision).

docs/architecture/04-ablation-engine.md, "Prevalence control":

* **Split at the INPUT level, not the trace level.** `dependency_fault`
  regenerates traces by re-running inputs; a trace-level split would let a
  "control" trace be silently replaced by a shimmed re-run.
* **Seeded and stratified** on Phase-3 provenance (mode, safe/adversarial,
  dimension) so the two sides have matched distributions — otherwise Engine
  can learn a distributional tell ("adversarial traces are the injected ones")
  rather than reading the trace.
* **Once, up front.** Filters and the `min_eligible` gate run inside the
  ablate set only; control inputs are never ablated and never re-run.

The split lives on the ground-truth side of the leak boundary and is stripped
from everything Engine sees.
"""

from __future__ import annotations

import hashlib
import random

from benchmark.schemas.ablation import AblationSplit
from benchmark.schemas.configs import AblationConfig
from benchmark.schemas.inputs import GenerationConfig, InputDataset, InputSpec


def _dimension_kind(spec: InputSpec, cfg: GenerationConfig) -> str:
    """safe vs adversarial, from the generation grid this input came out of.

    `InputSpec` carries the dimension *id*, not its kind, so the kind is
    recovered from the config the dataset embeds. An input drawn from the fixed
    adversarial library (`A_F`) is adversarial by construction even when its
    dimension is not declared anywhere.
    """
    if spec.fixed_adversarial_id:
        return "adversarial"
    for dim in cfg.adversarial_dims:
        if dim.dim_id == spec.dim_id:
            return "adversarial"
    for persona in cfg.adversarial_personas:
        if persona.persona_id == spec.persona_id:
            return "adversarial"
    return "safe"


def stratum_of(spec: InputSpec, cfg: GenerationConfig) -> str:
    """The stratification key: `mode|kind|dim_id`.

    The three axes the design names — single/multi-turn, safe/adversarial, and
    the dimension — as one stable string, so strata are reportable on the
    `AblationSplit` rather than being an implicit property of the code.
    """
    return f"{spec.mode}|{_dimension_kind(spec, cfg)}|{spec.dim_id}"


def _shuffled(ids: list[str], seed: int, stratum: str) -> list[str]:
    """A seeded shuffle whose result depends on the stratum as well as the seed.

    Sorting first makes the input order irrelevant; mixing the stratum name
    into the seed stops every stratum from drawing the same permutation index,
    which would correlate the assignment across strata.
    """
    salt = int(hashlib.sha256(f"{seed}\x1f{stratum}".encode()).hexdigest()[:8], 16)
    out = sorted(ids)
    random.Random(salt).shuffle(out)
    return out


def make_split(inputs: InputDataset, cfg: AblationConfig) -> AblationSplit:
    """Assign every `input_id` to exactly one side, once.

    Per-stratum quotas are allocated by **largest remainder**: each stratum
    gets `floor(n_s * f)` control slots, and the leftovers needed to reach the
    global target `round(N * f)` go to the strata with the largest fractional
    parts. Rounding each stratum independently would drop every stratum smaller
    than `1/f` out of the control set entirely — precisely the small,
    distinctive strata (a lone adversarial dimension) whose absence from
    control is a tell.
    """
    fraction = cfg.control_fraction
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"control_fraction must be in [0, 1], got {fraction}")

    gen_cfg = inputs.generation_config
    strata: dict[str, list[str]] = {}
    for spec in inputs.inputs:
        strata.setdefault(stratum_of(spec, gen_cfg), []).append(spec.input_id)

    total = len(inputs.inputs)
    target = round(total * fraction)

    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    for stratum, ids in strata.items():
        exact = len(ids) * fraction
        quotas[stratum] = int(exact)
        remainders.append((exact - int(exact), stratum))

    leftover = target - sum(quotas.values())
    # Largest fractional part first; the stratum name breaks ties so the
    # allocation never depends on dict ordering.
    for _remainder, stratum in sorted(remainders, key=lambda r: (-r[0], r[1]))[: max(leftover, 0)]:
        quotas[stratum] += 1

    control: list[str] = []
    ablate: list[str] = []
    for stratum in sorted(strata):
        ordered = _shuffled(strata[stratum], cfg.seed, stratum)
        take = min(quotas[stratum], len(ordered))
        control.extend(ordered[:take])
        ablate.extend(ordered[take:])

    return AblationSplit(
        seed=cfg.seed,
        control_fraction=fraction,
        strata=sorted(strata),
        control_input_ids=sorted(control),
        ablate_input_ids=sorted(ablate),
    )
