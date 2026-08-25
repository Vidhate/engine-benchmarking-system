"""Gate: the Engine's chat model is a BOUNDED call — timeout and retries.

The failure this pins down was observed, not imagined: one analysis request
hung and the batch it belonged to stalled for exactly 600 s, the OpenAI SDK's
own default timeout. A batch costs its slowest trace, so a single stuck request
does not cost one trace's worth of time — it costs the whole batch's.

`build_model` is the ONE construction site for the Engine's model, so asserting
here covers analysis and consolidation together. These are constructor
assertions only: nothing in this file reaches the network (conftest.py sets a
dummy key precisely so construction is possible offline).
"""

from __future__ import annotations

from engine.llm import MAX_RETRIES, REQUEST_TIMEOUT_S, build_model


def test_the_model_carries_an_explicit_timeout_and_retry_budget():
    model = build_model("gpt-5-mini")
    # `request_timeout` is the field `timeout=` populates on ChatOpenAI.
    assert model.request_timeout == REQUEST_TIMEOUT_S
    assert model.max_retries == MAX_RETRIES


def test_the_timeout_is_well_under_the_sdk_default_that_stalled_a_batch():
    """600 s is the SDK default and the observed stall. Ours must be shorter."""
    assert 0 < REQUEST_TIMEOUT_S < 600
    assert MAX_RETRIES >= 1, "a bounded timeout with no retry turns a 429 into a lost trace"
    # A retry ladder is only bounded if the worst case is. Timeout x (1 + retries)
    # has to stay under the wall-clock a stalled batch can afford.
    assert REQUEST_TIMEOUT_S * (1 + MAX_RETRIES) <= 600


def test_the_bound_applies_to_whatever_model_the_run_asks_for():
    """The model id is the comparison axis; the transport settings are not."""
    for name in ("gpt-5-mini", "gpt-5.1"):
        model = build_model(name)
        assert model.model_name == name
        assert model.request_timeout == REQUEST_TIMEOUT_S
        assert model.max_retries == MAX_RETRIES


def test_chat_completions_is_still_the_api_in_use():
    """The transport hardening must not have flipped us onto the Responses API:
    message content staying a plain string is what makes Engine traces match
    how the target app's were produced."""
    assert build_model("gpt-5-mini").use_responses_api is False
