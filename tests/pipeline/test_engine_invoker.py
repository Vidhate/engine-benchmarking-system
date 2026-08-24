"""Driving the Engine through the LangGraph Server API.

Every fact in here was paid for once already (apps/engine/README.md, "Invoking
the Engine" + the Phase 6 hand-off notes), so each one gets a test:

* the run input is exactly {trace_file, seed_issueboard, categories};
* `recursion_limit` and `configurable` are passed EXPLICITLY — the server's
  default recursion limit of 25 caps a run at ~23 traces, and the default
  analysis concurrency of 8 costs ~14 minutes at 300 traces;
* the output IS the issueboard, and its `board_id` is re-stamped: the Engine's
  own hash is computed over a different model and is not byte-compatible.
"""

from __future__ import annotations

import json

import httpx
import pytest

from benchmark.pipeline.config import EngineStageConfig
from benchmark.pipeline.engine import EngineRunFailed, LangGraphEngineInvoker
from benchmark.schemas import EngineAppConfig, ErrorCategory, Issueboard
from benchmark.schemas.io import content_hash

APP = EngineAppConfig(
    base_url="http://127.0.0.1:2025", assistant_id="engine", model_configurable_key="model"
)

BOARD = {
    "board_id": "engine-side-hash",
    "source": "engine_predicted",
    "issues": [
        {
            "error_id": "P1",
            "title": "hallucinated refund window",
            "description": "the assistant invented a policy",
            "category_id": "hallucination",
            "severity": "high",
        }
    ],
    "occurrences": [{"error_id": "P1", "trace_id": "t1"}],
}


class FakeClient:
    def __init__(self, output=None):
        self.output = output if output is not None else BOARD
        self.threads = self
        self.runs = self
        self.created = 0
        self.calls: list[dict] = []

    def create(self):
        self.created += 1
        return {"thread_id": "thread-1"}

    def wait(self, thread_id, assistant_id, *, input, config):  # noqa: A002
        self.calls.append(
            {
                "thread_id": thread_id,
                "assistant_id": assistant_id,
                "input": input,
                "config": config,
            }
        )
        return self.output

    #: What the SERVER says it ran. Deliberately NOT derived from what `wait`
    #: was sent: a readback that echoes the request proves nothing, and the
    #: failure this guard exists for is exactly a request the server ignored.
    recorded = [{"kwargs": {"config": {"configurable": {"model": "server-side-model"}}}}]

    def list(self, thread_id):
        return self.recorded


@pytest.fixture
def trace_file(tmp_path):
    path = tmp_path / "traces.json"
    path.write_text(
        json.dumps(
            {
                "dataset_id": "d1",
                "traces": [
                    {"trace_id": f"t{i}", "input_id": f"i{i}", "mode": "single_turn", "turns": []}
                    for i in range(3)
                ],
            }
        )
    )
    return path


def invoke(client, trace_file, *, seed=None, engine=None):
    invoker = LangGraphEngineInvoker(APP, client=client)
    return invoker(
        trace_file=trace_file,
        seed_board=seed or Issueboard(source="seed"),
        categories=[ErrorCategory(category_id="other", name="other", description="d")],
        engine=engine or EngineStageConfig(model="gpt-5.1-mini", analysis_concurrency=16),
    )


def test_the_engine_sees_exactly_three_things(trace_file):
    client = FakeClient()
    invoke(client, trace_file)
    assert set(client.calls[0]["input"]) == {"trace_file", "seed_issueboard", "categories"}


def test_the_trace_file_is_an_absolute_path_not_inline_traces(trace_file):
    """A 300-trace corpus is never sent through the API or into graph state."""
    client = FakeClient()
    invoke(client, trace_file)
    sent = client.calls[0]["input"]["trace_file"]
    assert isinstance(sent, str) and sent.startswith("/")
    assert sent == str(trace_file.resolve())


def test_the_model_goes_through_the_declared_configurable_key(trace_file):
    client = FakeClient()
    invoke(client, trace_file, engine=EngineStageConfig(model="gpt-5.1"))
    configurable = client.calls[0]["config"]["configurable"]
    assert configurable[APP.model_configurable_key] == "gpt-5.1"


def test_analysis_concurrency_is_passed_explicitly(trace_file):
    """The default of 8 projects to ~35 min at 300 traces; 16 to ~21 min."""
    client = FakeClient()
    invoke(client, trace_file, engine=EngineStageConfig(analysis_concurrency=16))
    assert client.calls[0]["config"]["configurable"]["analysis_concurrency"] == 16


def test_the_recursion_limit_is_passed_explicitly(trace_file):
    """The server default of 25 silently caps a run at ~23 traces."""
    client = FakeClient()
    invoke(client, trace_file, engine=EngineStageConfig(recursion_limit=10_000))
    assert client.calls[0]["config"]["recursion_limit"] == 10_000


def test_the_output_is_the_issueboard(trace_file):
    result = invoke(FakeClient(), trace_file)
    assert result.board.source == "engine_predicted"
    assert [i.error_id for i in result.board.issues] == ["P1"]


def test_the_board_id_is_restamped_on_ingest(trace_file):
    """The Engine's own hash is over a different model and does not match ours."""
    result = invoke(FakeClient(), trace_file)
    assert result.board.board_id != BOARD["board_id"]
    assert result.board.board_id == content_hash(result.board)


def test_the_raw_output_is_kept_verbatim(trace_file):
    result = invoke(FakeClient(), trace_file)
    assert result.raw_output["board_id"] == "engine-side-hash"


def test_the_trace_count_is_read_off_the_export(trace_file):
    """The only downstream signal of partial Engine failure is a small board,
    so the run records what it was asked to analyse."""
    assert invoke(FakeClient(), trace_file).trace_count == 3


def test_a_failed_run_raises_rather_than_scoring_an_error_dict(trace_file):
    client = FakeClient(output={"__error__": {"message": "recursion limit"}})
    with pytest.raises(EngineRunFailed, match="recursion limit"):
        invoke(client, trace_file)


def test_a_non_board_output_raises(trace_file):
    with pytest.raises(EngineRunFailed, match="did not return an issueboard"):
        invoke(FakeClient(output={"nothing": "useful"}), trace_file)


def test_the_seed_board_is_sent_without_injection_mode(trace_file):
    seed = Issueboard(
        source="seed",
        issues=[
            {
                "error_id": "S1",
                "title": "t",
                "description": "d",
                "category_id": "other",
                "severity": "low",
            }
        ],
    )
    client = FakeClient()
    invoke(client, trace_file, seed=seed)
    payload = client.calls[0]["input"]["seed_issueboard"]
    assert "injection_mode" not in json.dumps(payload)
    assert payload["issues"][0]["error_id"] == "S1"


def test_the_readback_reports_the_server_not_the_request(trace_file):
    """Trusting the request we sent proves nothing — LangGraph can drop a
    config silently (apps/engine/README.md footgun). The fake's run records are
    fixed and unrelated to what `wait` received, so a readback that merely
    echoed the request would fail this."""
    result = invoke(
        FakeClient(), trace_file, engine=EngineStageConfig(model="what-we-asked-for")
    )
    assert result.recorded_models == ["server-side-model"]


def test_an_unreadable_run_record_is_None_not_empty(trace_file):
    """Absent evidence and evidence-of-absence are different answers, and the
    caller acts differently on them — so they must not share a value."""

    class NoList(FakeClient):
        def list(self, thread_id):
            raise RuntimeError("no runs endpoint")

    result = invoke(NoList(), trace_file)
    assert result.recorded_models is None
    assert result.board.issues


def test_records_without_the_model_key_read_back_as_empty(trace_file):
    """THE silent-config-drop signature: the server persisted the run, and the
    model key is simply not in the config it kept. Readable, and empty."""

    class Dropped(FakeClient):
        recorded = [{"kwargs": {"config": {"configurable": {"analysis_concurrency": 8}}}}]

    result = invoke(Dropped(), trace_file)
    assert result.recorded_models == []


def test_a_thread_with_no_runs_at_all_also_reads_back_as_empty(trace_file):
    class NoRuns(FakeClient):
        recorded: list = []

    assert invoke(NoRuns(), trace_file).recorded_models == []


def test_a_missing_export_file_is_caught_before_the_run(tmp_path):
    client = FakeClient()
    with pytest.raises(FileNotFoundError):
        invoke(client, tmp_path / "nope.json")
    assert client.calls == [], "the Engine was invoked against a file that does not exist"


# ------------------------------------------------------------------ timeouts

class SpyFactory:
    def __init__(self, client=None):
        self.client = client or FakeClient()
        self.kwargs: dict = {}

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.client


def test_the_client_is_built_with_an_explicit_read_timeout(trace_file):
    """`get_sync_client` defaults to a 300 s read timeout, and `runs.wait` holds
    ONE blocking request for the whole Engine pass — a 300-trace run is ~21 min,
    so the default kills it five minutes in, after all the money is spent."""
    factory = SpyFactory()
    invoker = LangGraphEngineInvoker(APP, client_factory=factory)
    invoker(
        trace_file=trace_file,
        seed_board=Issueboard(source="seed"),
        categories=[],
        engine=EngineStageConfig(timeout_s=7200.0),
    )
    timeout = factory.kwargs["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.read == 7200.0
    assert timeout.connect and timeout.connect < 60, "connect stays short — only reads are slow"


def test_the_read_timeout_comes_from_the_run_config(trace_file):
    factory = SpyFactory()
    LangGraphEngineInvoker(APP, client_factory=factory)(
        trace_file=trace_file,
        seed_board=Issueboard(source="seed"),
        categories=[],
        engine=EngineStageConfig(timeout_s=60.0),
    )
    assert factory.kwargs["timeout"].read == 60.0


def test_the_default_timeout_outlasts_a_full_scale_run():
    """~340 traces at concurrency 16 is ~25 min; the default must not be close."""
    assert EngineStageConfig().timeout_s >= 3 * 3600


def test_an_injected_client_is_used_as_is(trace_file):
    """Tests inject a client directly; no factory call, no timeout to apply."""
    factory = SpyFactory()
    invoker = LangGraphEngineInvoker(APP, client=FakeClient(), client_factory=factory)
    invoker(
        trace_file=trace_file,
        seed_board=Issueboard(source="seed"),
        categories=[],
        engine=EngineStageConfig(),
    )
    assert factory.kwargs == {}


# ------------------------------------------- surviving a bad response (I3)

def test_a_schema_invalid_response_carries_its_raw_payload(trace_file):
    """Hours of Engine time must never be lost to an unparseable reply: the
    exception carries the payload so the caller can persist it."""
    payload = {"issues": "not a list", "occurrences": []}
    with pytest.raises(EngineRunFailed) as caught:
        invoke(FakeClient(output=payload), trace_file)
    assert caught.value.raw_output == payload


def test_a_failed_run_carries_the_error_payload(trace_file):
    payload = {"__error__": {"message": "recursion limit"}}
    with pytest.raises(EngineRunFailed) as caught:
        invoke(FakeClient(output=payload), trace_file)
    assert caught.value.raw_output == payload


def test_a_board_that_is_not_engine_predicted_is_refused_at_ingest(trace_file):
    """Checked here, not only as an exit deliverable: a board claiming to be the
    seed or the ground truth must not reach scoring wearing a prediction's face."""
    payload = {**BOARD, "source": "ground_truth"}
    with pytest.raises(EngineRunFailed, match="engine_predicted"):
        invoke(FakeClient(output=payload), trace_file)


def test_the_refused_board_still_carries_its_payload(trace_file):
    payload = {**BOARD, "source": "ground_truth"}
    with pytest.raises(EngineRunFailed) as caught:
        invoke(FakeClient(output=payload), trace_file)
    assert caught.value.raw_output == payload


def test_duplicate_error_ids_are_refused_at_ingest(trace_file):
    """`error_id` is a key downstream, not a label: the seed-delta logic groups
    by it, the exact-key matcher resolves through it, and two issues sharing one
    would be silently conflated wherever it is used."""
    payload = {**BOARD, "issues": [BOARD["issues"][0], {**BOARD["issues"][0], "title": "again"}]}
    with pytest.raises(EngineRunFailed, match="duplicate error_id"):
        invoke(FakeClient(output=payload), trace_file)


def test_the_duplicate_id_failure_names_the_offenders(trace_file):
    payload = {**BOARD, "issues": [BOARD["issues"][0], BOARD["issues"][0]]}
    with pytest.raises(EngineRunFailed, match="P1") as caught:
        invoke(FakeClient(output=payload), trace_file)
    assert caught.value.raw_output == payload


def test_distinct_error_ids_pass(trace_file):
    payload = {
        **BOARD,
        "issues": [BOARD["issues"][0], {**BOARD["issues"][0], "error_id": "P2"}],
    }
    assert len(invoke(FakeClient(output=payload), trace_file).board.issues) == 2
