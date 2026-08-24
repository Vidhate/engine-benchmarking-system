"""Deterministic session/trace ids + the config-only knowledge surface."""

from pathlib import Path

import pytest

from benchmark.harness.config import load_target_app_config
from benchmark.harness.ids import session_id_for, trace_id_for

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_session_id_is_a_pure_hash_of_dataset_and_input():
    a = session_id_for("ds-1", "safe-abc")
    b = session_id_for("ds-1", "safe-abc")
    assert a == b
    assert a != session_id_for("ds-2", "safe-abc")
    assert a != session_id_for("ds-1", "safe-abd")


def test_session_id_variant_separates_reruns_of_the_same_input():
    base = session_id_for("ds-1", "safe-abc")
    armed = session_id_for("ds-1", "safe-abc", variant="fault:retriever:empty")
    assert armed != base
    assert armed == session_id_for("ds-1", "safe-abc", variant="fault:retriever:empty")


def test_session_id_field_separator_prevents_collisions():
    # Naive "a" + "b" concatenation would make ("ab", "c") == ("a", "bc").
    assert session_id_for("ab", "c") != session_id_for("a", "bc")


def test_trace_id_is_derived_from_the_session_id_and_is_store_safe():
    sid = session_id_for("ds-1", "safe-abc")
    trace_id = trace_id_for(sid)
    assert trace_id_for(sid) == trace_id
    # LocalTraceStore rejects path separators; ids must survive it.
    assert "/" not in trace_id and "\\" not in trace_id
    assert trace_id not in ("", ".", "..")


def test_load_target_app_config_parses_the_checked_in_yaml():
    cfg = load_target_app_config(REPO_ROOT / "configs" / "target_app.yaml")
    assert cfg.base_url.startswith("http")
    assert cfg.assistant_id == "target_app"
    assert cfg.langsmith_project
    assert set(cfg.fault_configurable_keys) >= {"retriever", "tool", "llm"}
    assert cfg.max_turns_supported >= 1


def test_load_target_app_config_rejects_an_unknown_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_target_app_config(tmp_path / "nope.yaml")
