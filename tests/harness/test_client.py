"""The langgraph_sdk adapter: the only channel to the target app."""

from __future__ import annotations

import pytest

from benchmark.harness.client import AppResponse, LangGraphAppClient, TargetAppClient
from benchmark.schemas.configs import TargetAppConfig

CFG = TargetAppConfig(
    base_url="http://127.0.0.1:2024",
    assistant_id="target_app",
    langsmith_project="proj",
    fault_configurable_keys={"retriever": "fault_retriever"},
)


class FakeRuns:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def wait(self, thread_id, assistant_id, **kwargs):
        self.calls.append({"thread_id": thread_id, "assistant_id": assistant_id, **kwargs})
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeThreads:
    def __init__(self, checkpoint_id="ckpt-1", history=None):
        self.checkpoint_id = checkpoint_id
        self.history = history or []
        self.updates: list[dict] = []
        self.created = 0

    def create(self):
        self.created += 1
        return {"thread_id": f"thread-{self.created}"}

    def get_state(self, thread_id, **kwargs):
        return {"checkpoint": {"checkpoint_id": self.checkpoint_id}, "values": {"messages": []}}

    def get_history(self, thread_id, **kwargs):
        return self.history

    def update_state(self, thread_id, values, **kwargs):
        self.updates.append({"thread_id": thread_id, "values": values, **kwargs})
        return {"checkpoint": {"checkpoint_id": "fork-1", "thread_id": thread_id}}


class FakeSDK:
    def __init__(self, responses, **thread_kwargs):
        self.runs = FakeRuns(responses)
        self.threads = FakeThreads(**thread_kwargs)


def ok_run(text="30 days."):
    return {
        "messages": [
            {"type": "human", "content": "q"},
            {"type": "ai", "content": "", "tool_calls": [{"name": "rag_search"}]},
            {"type": "tool", "content": "docs"},
            {"type": "ai", "content": text},
        ]
    }


def make_client(sdk, **kwargs):
    kwargs.setdefault("sleep", lambda _s: None)
    return LangGraphAppClient(CFG, client=sdk, **kwargs)


def test_it_satisfies_the_target_app_client_protocol():
    assert isinstance(make_client(FakeSDK([])), TargetAppClient)


def test_invoke_stamps_the_session_id_into_run_metadata():
    sdk = FakeSDK([ok_run()])
    response = make_client(sdk).invoke("thread-1", "hello", session_id="s-abc", turn_index=2)

    call = sdk.runs.calls[0]
    assert call["assistant_id"] == "target_app"
    assert call["input"] == {"messages": [{"role": "user", "content": "hello"}]}
    assert call["metadata"]["session_id"] == "s-abc"
    assert call["metadata"]["turn_index"] == 2
    assert isinstance(response, AppResponse)
    assert response.final_response == "30 days."
    assert response.error is None


def test_invoke_returns_the_last_non_tool_calling_assistant_message():
    sdk = FakeSDK([ok_run("the final answer")])
    assert make_client(sdk).invoke("t", "q", session_id="s").final_response == "the final answer"


def test_fault_values_are_passed_as_configurable_mappings():
    sdk = FakeSDK([ok_run()])
    make_client(sdk).invoke(
        "t", "q", session_id="s", configurable={"fault_retriever": {"behavior": "empty"}}
    )
    configurable = sdk.runs.calls[0]["config"]["configurable"]
    assert configurable == {"fault_retriever": {"behavior": "empty"}}
    assert all(isinstance(v, dict) for v in configurable.values()), (
        "scalar configurable values are promoted into inheritable run metadata"
    )


def test_no_config_is_sent_when_nothing_is_armed():
    sdk = FakeSDK([ok_run()])
    make_client(sdk).invoke("t", "q", session_id="s")
    assert sdk.runs.calls[0]["config"] is None


def test_transient_failures_are_retried_then_succeed():
    sdk = FakeSDK([ConnectionError("boom"), ok_run("recovered")])
    response = make_client(sdk, max_retries=2).invoke("t", "q", session_id="s")
    assert response.final_response == "recovered"
    assert response.error is None
    assert len(sdk.runs.calls) == 2


def test_exhausted_retries_become_an_app_error_response_not_an_exception():
    sdk = FakeSDK([ConnectionError("boom"), ConnectionError("boom"), ConnectionError("boom")])
    response = make_client(sdk, max_retries=2).invoke("t", "q", session_id="s")
    assert response.error and "boom" in response.error
    assert response.final_response == ""


def test_a_server_reported_run_error_is_an_app_error_response():
    sdk = FakeSDK([{"__error__": {"error": "GraphRecursionError", "message": "loop"}}])
    response = make_client(sdk).invoke("t", "q", session_id="s")
    assert response.error and "GraphRecursionError" in response.error


def test_the_checkpoint_after_each_turn_is_recorded_for_phase_5_replay():
    sdk = FakeSDK([ok_run()], checkpoint_id="ckpt-42")
    response = make_client(sdk).invoke("t", "q", session_id="s")
    assert response.checkpoint_id == "ckpt-42"


def test_checkpoint_recording_can_be_switched_off():
    sdk = FakeSDK([ok_run()])
    response = make_client(sdk, record_checkpoints=False).invoke("t", "q", session_id="s")
    assert response.checkpoint_id is None


def test_create_thread_and_update_state_go_through_the_sdk():
    sdk = FakeSDK([])
    client = make_client(sdk)
    thread_id = client.create_thread()
    assert thread_id == "thread-1"

    forked = client.update_state(
        thread_id, {"messages": [{"role": "ai", "id": "m1", "content": "x"}]},
        checkpoint={"checkpoint_id": "ckpt-1"},
    )
    assert forked["checkpoint_id"] == "fork-1"
    assert sdk.threads.updates[0]["checkpoint"] == {"checkpoint_id": "ckpt-1"}


def test_the_client_never_imports_the_app():
    import benchmark.harness.client as module

    source = open(module.__file__).read()
    assert "import apps" not in source and "from apps" not in source


@pytest.mark.parametrize(
    "content,expected",
    [
        ("plain text", "plain text"),
        (
            [{"type": "text", "text": "block a"}, {"type": "text", "text": "block b"}],
            "block a block b",
        ),
    ],
)
def test_content_blocks_are_flattened(content, expected):
    sdk = FakeSDK([{"messages": [{"type": "ai", "content": content}]}])
    assert make_client(sdk).invoke("t", "q", session_id="s").final_response == expected
