"""Gate: PromptExpander boundary — mock is deterministic, no network calls."""

from benchmark.generation.expander import MockPromptExpander, PromptExpander
from benchmark.schemas.inputs import Dimension, Persona


def make_dim() -> Dimension:
    return Dimension(dim_id="d1", name="query_topic", kind="safe", variations=["refunds"])


def make_persona() -> Persona:
    return Persona(persona_id="p1", name="Angry Alice", kind="target", description="…")


def test_mock_implements_prompt_expander_protocol():
    assert isinstance(MockPromptExpander(), PromptExpander)


def test_mock_expand_is_deterministic_for_same_inputs():
    a = MockPromptExpander().expand(make_dim(), "refunds", seed=7)
    b = MockPromptExpander().expand(make_dim(), "refunds", seed=7)
    assert a == b
    assert isinstance(a, str) and a


def test_mock_expand_varies_with_variation():
    dim = make_dim()
    a = MockPromptExpander().expand(dim, "refunds", seed=7)
    b = MockPromptExpander().expand(dim, "shipping", seed=7)
    assert a != b


def test_mock_expand_varies_with_seed():
    dim = make_dim()
    a = MockPromptExpander().expand(dim, "refunds", seed=1)
    b = MockPromptExpander().expand(dim, "refunds", seed=2)
    assert a != b


def test_mock_expand_scenario_is_deterministic_and_persona_sensitive():
    persona = make_persona()
    other = Persona(persona_id="p2", name="Calm Bob", kind="target", description="…")
    a = MockPromptExpander().expand_scenario(persona, "d1", "refunds", seed=7)
    b = MockPromptExpander().expand_scenario(persona, "d1", "refunds", seed=7)
    c = MockPromptExpander().expand_scenario(other, "d1", "refunds", seed=7)
    assert a == b
    assert a != c


def test_mock_records_calls():
    expander = MockPromptExpander()
    expander.expand(make_dim(), "refunds", seed=7)
    expander.expand_scenario(make_persona(), "d1", "refunds", seed=7)
    assert len(expander.calls) == 2


def test_mock_expand_includes_app_context_when_provided():
    a = MockPromptExpander().expand(make_dim(), "refunds", seed=7, app_context="A payroll app.")
    b = MockPromptExpander().expand(make_dim(), "refunds", seed=7, app_context="A travel app.")
    assert "A payroll app." in a
    assert "A travel app." in b
    assert a != b


def test_mock_expand_scenario_includes_app_context_when_provided():
    persona = make_persona()
    a = MockPromptExpander().expand_scenario(
        persona, "d1", "refunds", seed=7, app_context="A payroll app."
    )
    b = MockPromptExpander().expand_scenario(
        persona, "d1", "refunds", seed=7, app_context="A travel app."
    )
    assert "A payroll app." in a
    assert "A travel app." in b
    assert a != b


def test_mock_expand_empty_app_context_does_not_crash():
    dim = make_dim()
    text = MockPromptExpander().expand(dim, "refunds", seed=7)  # no app_context passed
    assert isinstance(text, str) and text
    text2 = MockPromptExpander().expand(dim, "refunds", seed=7, app_context="")
    assert text2 == text
