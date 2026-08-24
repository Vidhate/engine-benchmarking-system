"""The persona user-simulator boundary (LLM behind an interface, mocked here)."""

from __future__ import annotations

import pytest

from benchmark.harness.simulator import (
    DONE_TOKEN,
    OpenAIUserSimulator,
    ScriptedUserSimulator,
    UserSimulator,
    is_done,
    strip_done,
)
from benchmark.schemas.inputs import Persona

PERSONA = Persona(
    persona_id="frustrated_billing_customer",
    name="Frustrated Billing Customer",
    kind="target",
    description="Upset about a duplicate charge. Curt.",
    goals=["get a refund", "understand why it happened"],
)


def test_scripted_simulator_satisfies_the_protocol():
    assert isinstance(ScriptedUserSimulator(["hi"]), UserSimulator)


def test_scripted_simulator_replays_its_script_in_order():
    sim = ScriptedUserSimulator(["first", "second", DONE_TOKEN])
    assert sim.next_message(persona=PERSONA, scenario="s", history=[], turn_index=0) == "first"
    assert sim.next_message(persona=PERSONA, scenario="s", history=[], turn_index=1) == "second"
    assert is_done(sim.next_message(persona=PERSONA, scenario="s", history=[], turn_index=2))


def test_scripted_simulator_records_what_it_was_asked():
    sim = ScriptedUserSimulator(["a", "b"])
    sim.next_message(persona=PERSONA, scenario="my scenario", history=[("u", "r")], turn_index=1)
    call = sim.calls[0]
    assert call["persona"].persona_id == "frustrated_billing_customer"
    assert call["scenario"] == "my scenario"
    assert call["history"] == [("u", "r")]


def test_scripted_simulator_terminates_when_its_script_runs_out():
    sim = ScriptedUserSimulator(["only one"])
    sim.next_message(persona=PERSONA, scenario="s", history=[], turn_index=0)
    assert is_done(sim.next_message(persona=PERSONA, scenario="s", history=[], turn_index=1))


@pytest.mark.parametrize(
    "text,done",
    [
        ("[DONE]", True),
        ("Great, thanks. [DONE]", True),
        ("  [done]  ", True),
        ("I am done shopping", False),
        ("still going", False),
    ],
)
def test_done_detection_is_token_based_not_word_based(text, done):
    assert is_done(text) is done


def test_strip_done_leaves_the_usable_message_behind():
    assert strip_done("Great, thanks. [DONE]") == "Great, thanks."
    assert strip_done("[DONE]") == ""


def test_openai_simulator_builds_a_persona_and_scenario_system_prompt():
    """The LLM is behind the interface; only prompt construction is unit-tested."""
    sim = OpenAIUserSimulator(api_key="unused-in-this-test")
    system = sim.system_prompt(PERSONA, "You were charged twice in June.")

    assert PERSONA.description in system
    assert "You were charged twice in June." in system
    for goal in PERSONA.goals:
        assert goal in system
    assert DONE_TOKEN in system, "the simulator must be told how to terminate"


def test_openai_simulator_maps_the_conversation_from_the_simulators_point_of_view():
    sim = OpenAIUserSimulator(api_key="unused-in-this-test")
    messages = sim.chat_messages(
        PERSONA, "scenario", [("I was charged twice", "Let me check that for you.")]
    )
    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    # The simulator *is* the user of the app, so its own lines are assistant
    # turns and the app's replies are user turns in its own chat history.
    assert messages[1] == {"role": "assistant", "content": "I was charged twice"}
    assert messages[2] == {"role": "user", "content": "Let me check that for you."}


def test_openai_simulator_never_touches_the_network_at_construction():
    OpenAIUserSimulator(api_key=None)  # must not raise


def test_openai_simulator_refuses_to_call_without_a_key():
    sim = OpenAIUserSimulator(api_key=None)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        sim.next_message(persona=PERSONA, scenario="s", history=[], turn_index=0)


def test_the_pinned_persona_model_is_used():
    from benchmark.models import PERSONA_SIM_MODEL

    assert OpenAIUserSimulator().model == PERSONA_SIM_MODEL
