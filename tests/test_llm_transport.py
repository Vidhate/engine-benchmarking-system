"""Gate: the two benchmark-side LLM call sites are bounded AND untraced.

`benchmark/ablation/agent.py` and `benchmark/generation/expander.py` are the
only two places in the benchmark that talk to a model of their own. Both used
to hand-roll `urllib`; both now go through `ChatOpenAI`, which is the whole
reason for the swap:

* **bounded** — an explicit per-request timeout (the SDK's own default is 600 s,
  and a hung request stalls whatever batch it belongs to for all of it) and a
  retry budget with 429 backoff. Rate limits are the failure these two actually
  see: a generation pass is hundreds of sequential calls, and the ablation
  agent fans out across every category at once.
* **untraced** — `LANGSMITH_TRACING` is set during a real run because the
  harness needs it, and LangChain traces by default. An unsuppressed call here
  would put a benchmark-side run in the collector's own project: it pollutes
  the trace corpus with runs that are not traces of the app under test, and for
  the ablation agent it publishes which errors are about to be injected.

Nothing here reaches the network. `ChatOpenAI` and `tracing_context` are both
replaced with recorders, so what is asserted is exactly what the call site
*asks for* — which is the thing that can regress. tests/test_tracing_boundary.py
asserts the same suppression structurally, off the AST; this file asserts it
behaviourally, at the call. Losing either one would leave a gap.
"""

from __future__ import annotations

import contextlib

import pytest

from benchmark.ablation import agent as agent_mod
from benchmark.ablation.agent import AgentResponseError, AgentTransportError, OpenAIAblationAgent
from benchmark.generation import expander as expander_mod
from benchmark.generation.expander import OpenAIPromptExpander
from benchmark.schemas.inputs import Dimension, Persona


class Reply:
    def __init__(self, content):
        self.content = content


class RecordingChatOpenAI:
    """Stands in for `ChatOpenAI`. Records construction kwargs and each call."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        RecordingChatOpenAI.constructions.append(kwargs)

    constructions: list[dict] = []
    content: object = "{}"
    raises: Exception | None = None

    def invoke(self, messages, config=None):
        RecordingChatOpenAI.calls.append(
            {"messages": messages, "config": config, "traced": TRACING_DEPTH[0] == 0}
        )
        if RecordingChatOpenAI.raises is not None:
            raise RecordingChatOpenAI.raises
        return Reply(RecordingChatOpenAI.content)

    calls: list[dict] = []


#: Depth of the suppression context at the moment `invoke` is called. Zero
#: means the call was NOT wrapped, which is the regression this catches.
TRACING_DEPTH = [0]
TRACING_ARGS: list[dict] = []


@contextlib.contextmanager
def recording_tracing_context(**kwargs):
    TRACING_ARGS.append(kwargs)
    TRACING_DEPTH[0] += 1
    try:
        yield
    finally:
        TRACING_DEPTH[0] -= 1


@pytest.fixture
def transport(monkeypatch):
    """Both call sites, wired to the recorders. Reset per test."""
    RecordingChatOpenAI.constructions = []
    RecordingChatOpenAI.calls = []
    RecordingChatOpenAI.content = "{}"
    RecordingChatOpenAI.raises = None
    TRACING_DEPTH[0] = 0
    TRACING_ARGS.clear()
    for module in (agent_mod, expander_mod):
        monkeypatch.setattr(module, "ChatOpenAI", RecordingChatOpenAI)
        monkeypatch.setattr(module, "tracing_context", recording_tracing_context)
    return RecordingChatOpenAI


def make_dim() -> Dimension:
    return Dimension(dim_id="d1", name="query_topic", kind="safe", variations=["refunds"])


def make_persona() -> Persona:
    return Persona(persona_id="p1", name="Angry Alice", kind="target", description="…")


# --------------------------------------------------------- bounded calls

def test_the_ablation_agent_asks_for_a_timeout_and_a_retry_budget(transport):
    OpenAIAblationAgent(api_key="sk-test")._chat("system", "user")
    (built,) = transport.constructions
    assert built["timeout"] == 120.0
    assert built["max_retries"] == 4


def test_the_expander_asks_for_a_timeout_and_a_retry_budget(transport):
    transport.content = "hello"
    OpenAIPromptExpander(api_key="sk-test")._chat("system", "user")
    (built,) = transport.constructions
    assert built["timeout"] == 120.0
    assert built["max_retries"] == 4


def test_the_transport_swap_kept_the_request_on_the_wire_the_same(transport):
    """Chat Completions, `response_format` for the agent, `temperature` for the
    expander — a transport swap is not a behaviour change."""
    OpenAIAblationAgent(api_key="sk-test")._chat("system", "user")
    (agent_built,) = transport.constructions
    assert agent_built["use_responses_api"] is False
    assert agent_built["model_kwargs"] == {"response_format": {"type": "json_object"}}

    transport.constructions = []
    transport.content = "hello"
    OpenAIPromptExpander(api_key="sk-test")._chat("system", "user")
    (expander_built,) = transport.constructions
    assert expander_built["use_responses_api"] is False
    assert expander_built["temperature"] == 0.7


@pytest.mark.parametrize(
    "build",
    [
        lambda: OpenAIAblationAgent(api_key="sk-test"),
        lambda: OpenAIPromptExpander(api_key="sk-test"),
    ],
)
def test_no_client_is_built_until_a_call_is_actually_made(transport, build):
    """Import and construction stay offline — pytest collection must never
    build a client, let alone need a key."""
    build()
    assert transport.constructions == []


@pytest.mark.parametrize(
    "call",
    [
        lambda: OpenAIAblationAgent(api_key=None)._chat("s", "u"),
        lambda: OpenAIPromptExpander(api_key=None)._chat("s", "u"),
    ],
)
def test_a_missing_key_is_still_a_plain_runtime_error(transport, monkeypatch, call):
    """Not an AgentTransportError: no request was attempted, so retrying is not
    the answer — setting the key is."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        call()


# ------------------------------------------------------ untraced calls

@pytest.mark.parametrize(
    ("call", "content"),
    [
        (lambda: OpenAIAblationAgent(api_key="sk-test")._chat("s", "u"), "{}"),
        (lambda: OpenAIPromptExpander(api_key="sk-test")._chat("s", "u"), "hello"),
    ],
)
def test_every_call_site_invokes_inside_the_tracing_suppression(transport, call, content):
    transport.content = content
    call()
    (recorded,) = transport.calls
    assert recorded["traced"] is False, (
        "the model was invoked outside `tracing_context(enabled=False)` — this call "
        "would emit a LangSmith run into the collector's project"
    )
    assert TRACING_ARGS == [{"enabled": False}]
    # …and the context is exited again, so one benchmark-side call does not
    # silently mute the harness's own collection for the rest of the process.
    assert TRACING_DEPTH[0] == 0


@pytest.mark.parametrize(
    ("call", "content"),
    [
        (lambda: OpenAIAblationAgent(api_key="sk-test")._chat("s", "u"), "{}"),
        (lambda: OpenAIPromptExpander(api_key="sk-test")._chat("s", "u"), "hello"),
    ],
)
def test_every_call_site_also_passes_an_empty_callback_list(transport, call, content):
    """The second half of the suppression: a handler installed by a caller
    cannot re-attach a tracer to this run."""
    transport.content = content
    call()
    (recorded,) = transport.calls
    assert recorded["config"] is not None, "no run config was passed at all"
    assert recorded["config"].get("callbacks") == []


# ----------------------------------------------- unchanged error taxonomy

def test_a_transport_failure_is_still_an_agent_transport_error(transport):
    transport.raises = TimeoutError("connection reset")
    with pytest.raises(AgentTransportError, match="TimeoutError"):
        OpenAIAblationAgent(api_key="sk-test")._chat("s", "u")


def test_an_unparseable_reply_is_still_an_agent_response_error(transport):
    """Retrying does not fix a refusal, so it keeps the non-retryable class."""
    transport.content = "I'm sorry, I can't help with that."
    with pytest.raises(AgentResponseError, match="not JSON"):
        OpenAIAblationAgent(api_key="sk-test")._chat("s", "u")


def test_a_non_string_content_block_is_a_response_error_not_a_crash(transport):
    transport.content = [{"type": "text", "text": "{}"}]
    with pytest.raises(AgentResponseError, match="not the expected shape"):
        OpenAIAblationAgent(api_key="sk-test")._chat("s", "u")


# --------------------------------------------- prompts are unchanged text

def test_the_agent_still_sends_one_system_and_one_user_message(transport):
    OpenAIAblationAgent(api_key="sk-test")._chat("SYSTEM TEXT", "USER TEXT")
    (recorded,) = transport.calls
    assert recorded["messages"] == [("system", "SYSTEM TEXT"), ("user", "USER TEXT")]


def test_the_expanders_prompt_text_survives_the_swap(transport):
    """The expander's own prompts are the generation config's control surface;
    a transport swap must not have touched a word of them."""
    transport.content = "  a concrete user message  "
    expander = OpenAIPromptExpander(api_key="sk-test")
    text = expander.expand(make_dim(), "refunds", seed=3, app_context="A payroll app.")
    assert text == "a concrete user message", "the reply must still be stripped"
    (recorded,) = transport.calls
    _, (_, user) = recorded["messages"]
    assert "A payroll app." in user
    assert "Variation: refunds" in user
    assert "Seed: 3" in user


def test_the_expanders_scenario_prompt_survives_the_swap(transport):
    transport.content = "a brief"
    expander = OpenAIPromptExpander(api_key="sk-test")
    expander.expand_scenario(make_persona(), "d1", "refunds", seed=3, app_context="A travel app.")
    (recorded,) = transport.calls
    (_, system), (_, user) = recorded["messages"]
    assert "scenario brief" in system
    assert "Angry Alice" in user
    assert "A travel app." in user
