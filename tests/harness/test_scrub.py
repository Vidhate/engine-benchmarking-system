"""The leak tripwire itself: it must fire on fingerprints and stay silent otherwise."""

import pytest

from benchmark.harness.scrub import (
    LeakDetected,
    assert_no_leak,
    find_leaked_keys,
    find_leaks,
    leak_tokens,
)
from benchmark.schemas.configs import TargetAppConfig

CFG = TargetAppConfig(
    base_url="http://x",
    assistant_id="a",
    langsmith_project="p",
    fault_configurable_keys={"retriever": "fault_retriever", "tool": "fault_tool"},
)


def test_declared_fault_keys_come_from_config_not_from_a_hardcoded_copy():
    tokens = leak_tokens(CFG)
    assert "fault_retriever" in tokens and "fault_tool" in tokens
    assert "shim" in tokens  # structural tokens are always included


def test_fingerprints_are_found_wherever_they_hide():
    payload = {
        "spans": [
            {"attributes": {"metadata": {"fault_llm": {"behavior": "truncate_output"}}}},
            {"tags": ["shims"], "serialized": {"id": ["x", "SupportChatModel"]}},
        ]
    }
    found = find_leaks(payload, leak_tokens(CFG))
    assert "fault_llm" in found
    assert "truncate_output" in found
    assert "shims" in found
    assert "supportchatmodel" in found


def test_innocent_words_that_merely_contain_a_stem_are_not_leaks():
    """A scrubber that quarantines healthy traces gets switched off."""
    payload = {
        "inputs": {"default_headers": {"x": 1}, "note": "the fault lies elsewhere"},
        "outputs": {"text": "Your subscription defaulted to the free plan."},
    }
    assert find_leaks(payload, leak_tokens(CFG)) == []
    assert_no_leak(payload, where="test", tokens=leak_tokens(CFG))


def test_configurable_echo_keys_are_found_by_prefix_at_any_depth():
    payload = {"a": [{"b": {"fault_retriever": {"behavior": "stale"}}}]}
    assert find_leaked_keys(payload) == ["fault_retriever"]
    assert find_leaked_keys({"defaults": 1, "faults": 2}) == []


def test_assert_no_leak_raises_rather_than_returning_a_boolean():
    with pytest.raises(LeakDetected) as excinfo:
        assert_no_leak({"x": "fault_tool"}, where="a stored Trace", tokens=leak_tokens(CFG))
    assert "a stored Trace" in str(excinfo.value)
    assert "fault_tool" in str(excinfo.value)
