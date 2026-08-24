"""Gate: count formulas + provenance completeness for the three generators.

N = (D×V_D)+(A_c×V_AC)+A_F for single-turn; N = (P×D1)+(P_A×D2) for multi-turn.
"""

import pytest

from benchmark.generation.expander import MockPromptExpander
from benchmark.generation.generators import (
    assemble_multi_turn,
    generate_adversarial_inputs,
    generate_safe_inputs,
)
from benchmark.schemas.inputs import Dimension, InputSpec, Persona


def make_safe_dims(n_dims=2, n_var=3) -> list[Dimension]:
    return [
        Dimension(
            dim_id=f"safe{i}",
            name=f"dimension_{i}",
            kind="safe",
            variations=[f"v{i}-{j}" for j in range(n_var)],
        )
        for i in range(n_dims)
    ]


def make_adv_dims(n_dims=2, n_var=2) -> list[Dimension]:
    return [
        Dimension(
            dim_id=f"adv{i}",
            name=f"adversarial_dimension_{i}",
            kind="adversarial",
            variations=[f"a{i}-{j}" for j in range(n_var)],
        )
        for i in range(n_dims)
    ]


def make_fixed_library(n=4) -> list[InputSpec]:
    return [
        InputSpec(
            input_id=f"fixed-{i}",
            mode="single_turn",
            dim_id="adversarial_fixed",
            variation=f"category-{i}",
            prompt=f"fixed attack prompt {i}",
        )
        for i in range(n)
    ]


def make_persona(persona_id="p1", kind="target") -> Persona:
    return Persona(persona_id=persona_id, name="Persona", kind=kind, description="…")


# ---------------------------------------------------------------------------
# generate_safe_inputs: D x V_D
# ---------------------------------------------------------------------------


def test_generate_safe_inputs_count_matches_formula():
    dims = make_safe_dims(n_dims=3, n_var=4)  # D=3, V_D=4 -> 12
    out = generate_safe_inputs(dims, MockPromptExpander(), seed=1)
    assert len(out) == 12


def test_generate_safe_inputs_provenance_complete():
    dims = make_safe_dims(n_dims=1, n_var=2)
    out = generate_safe_inputs(dims, MockPromptExpander(), seed=1)
    for spec in out:
        assert spec.input_id
        assert spec.mode == "single_turn"
        assert spec.dim_id == "safe0"
        assert spec.variation
        assert spec.persona_id is None
        assert spec.fixed_adversarial_id is None
        assert spec.prompt


def test_generate_safe_inputs_ids_are_unique():
    dims = make_safe_dims(n_dims=3, n_var=4)
    out = generate_safe_inputs(dims, MockPromptExpander(), seed=1)
    assert len({s.input_id for s in out}) == len(out)


def test_generate_safe_inputs_deterministic_ids_for_same_grid_cell():
    dims = make_safe_dims(n_dims=1, n_var=1)
    a = generate_safe_inputs(dims, MockPromptExpander(), seed=1)
    b = generate_safe_inputs(dims, MockPromptExpander(), seed=1)
    assert a[0].input_id == b[0].input_id


# ---------------------------------------------------------------------------
# generate_adversarial_inputs: (A_c x V_AC) + A_F
# ---------------------------------------------------------------------------


def test_generate_adversarial_inputs_count_matches_formula():
    dims = make_adv_dims(n_dims=2, n_var=3)  # A_c=2, V_AC=3 -> 6
    fixed = make_fixed_library(n=5)  # A_F=5
    out = generate_adversarial_inputs(dims, fixed, MockPromptExpander(), seed=1)
    assert len(out) == 11


def test_generate_adversarial_inputs_llm_expanded_portion_provenance():
    dims = make_adv_dims(n_dims=1, n_var=2)
    out = generate_adversarial_inputs(dims, [], MockPromptExpander(), seed=1)
    for spec in out:
        assert spec.mode == "single_turn"
        assert spec.dim_id == "adv0"
        assert spec.fixed_adversarial_id is None
        assert spec.prompt


def test_generate_adversarial_inputs_fixed_portion_is_passthrough_with_provenance():
    fixed = make_fixed_library(n=2)
    out = generate_adversarial_inputs([], fixed, MockPromptExpander(), seed=1)
    assert len(out) == 2
    prompts = {s.prompt for s in out}
    assert prompts == {"fixed attack prompt 0", "fixed attack prompt 1"}
    for spec in out:
        assert spec.fixed_adversarial_id is not None
        assert spec.mode == "single_turn"


def test_generate_adversarial_inputs_ids_unique_across_both_sources():
    dims = make_adv_dims(n_dims=2, n_var=2)
    fixed = make_fixed_library(n=3)
    out = generate_adversarial_inputs(dims, fixed, MockPromptExpander(), seed=1)
    assert len({s.input_id for s in out}) == len(out)


# ---------------------------------------------------------------------------
# assemble_multi_turn: (P x D1) + (P_A x D2)
# ---------------------------------------------------------------------------


def test_assemble_multi_turn_count_matches_formula():
    safe_pool = generate_safe_inputs(make_safe_dims(2, 3), MockPromptExpander(), seed=1)  # D1=6
    adv_pool = generate_adversarial_inputs(
        make_adv_dims(1, 2), make_fixed_library(2), MockPromptExpander(), seed=1
    )  # D2=4
    personas = [make_persona("p1"), make_persona("p2")]  # P=2
    adv_personas = [make_persona("pa1", kind="adversarial")]  # P_A=1

    out = assemble_multi_turn(
        personas, adv_personas, safe_pool, adv_pool, MockPromptExpander(), seed=1
    )
    assert len(out) == 2 * 6 + 1 * 4  # (P x D1) + (P_A x D2)


def test_assemble_multi_turn_provenance_complete():
    safe_pool = generate_safe_inputs(make_safe_dims(1, 2), MockPromptExpander(), seed=1)
    persona = make_persona("p1")
    out = assemble_multi_turn([persona], [], safe_pool, [], MockPromptExpander(), seed=1)
    assert out
    for spec in out:
        assert spec.input_id
        assert spec.mode == "multi_turn"
        assert spec.dim_id
        assert spec.variation
        assert spec.persona_id == "p1"
        assert spec.scenario
        assert spec.prompt is None


def test_assemble_multi_turn_no_literal_prompt():
    safe_pool = generate_safe_inputs(make_safe_dims(1, 1), MockPromptExpander(), seed=1)
    out = assemble_multi_turn(
        [make_persona()], [], safe_pool, [], MockPromptExpander(), seed=1
    )
    assert all(s.prompt is None for s in out)


def test_assemble_multi_turn_ids_unique():
    safe_pool = generate_safe_inputs(make_safe_dims(2, 2), MockPromptExpander(), seed=1)
    personas = [make_persona("p1"), make_persona("p2")]
    out = assemble_multi_turn(personas, [], safe_pool, [], MockPromptExpander(), seed=1)
    assert len({s.input_id for s in out}) == len(out)


def test_assemble_multi_turn_rejects_adversarial_persona_in_personas():
    """A misconfigured YAML putting an adversarial persona under `personas:`
    must not be silently crossed with the safe pool."""
    safe_pool = generate_safe_inputs(make_safe_dims(1, 1), MockPromptExpander(), seed=1)
    bad_persona = make_persona("p1", kind="adversarial")
    with pytest.raises(ValueError, match="target"):
        assemble_multi_turn([bad_persona], [], safe_pool, [], MockPromptExpander(), seed=1)


def test_assemble_multi_turn_rejects_target_persona_in_adversarial_personas():
    adv_pool = generate_adversarial_inputs(make_adv_dims(1, 1), [], MockPromptExpander(), seed=1)
    bad_persona = make_persona("pa1", kind="target")
    with pytest.raises(ValueError, match="adversarial"):
        assemble_multi_turn([], [bad_persona], [], adv_pool, MockPromptExpander(), seed=1)


# ---------------------------------------------------------------------------
# app_context threading — generators stay app-agnostic; only the yaml-supplied
# app_context describes the target app to the expander.
# ---------------------------------------------------------------------------


def test_generate_safe_inputs_threads_app_context_to_expander():
    dims = make_safe_dims(n_dims=1, n_var=1)
    expander = MockPromptExpander()
    generate_safe_inputs(dims, expander, seed=1, app_context="A fictional payroll app.")
    assert expander.calls[0][-1] == "A fictional payroll app."


def test_generate_adversarial_inputs_threads_app_context_to_expander():
    dims = make_adv_dims(n_dims=1, n_var=1)
    expander = MockPromptExpander()
    generate_adversarial_inputs(dims, [], expander, seed=1, app_context="A fictional travel app.")
    assert expander.calls[0][-1] == "A fictional travel app."


def test_assemble_multi_turn_threads_app_context_to_expander():
    safe_pool = generate_safe_inputs(make_safe_dims(1, 1), MockPromptExpander(), seed=1)
    persona = make_persona("p1")
    expander = MockPromptExpander()
    assemble_multi_turn(
        [persona], [], safe_pool, [], expander, seed=1, app_context="A fictional CRM app."
    )
    assert expander.calls[0][-1] == "A fictional CRM app."
