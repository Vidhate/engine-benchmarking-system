"""Gate: the TraceStore boundary — local filesystem store in our Trace schema."""

import pytest

from benchmark.tracing import LocalTraceStore, TraceStore
from tests.test_schemas_roundtrip import make_trace


def test_local_store_satisfies_protocol(tmp_path):
    assert isinstance(LocalTraceStore(tmp_path), TraceStore)


def test_put_get_roundtrip(tmp_path):
    store = LocalTraceStore(tmp_path)
    trace = make_trace("t1")
    store.put(trace)
    assert store.get("t1") == trace
    assert store.exists("t1")
    assert not store.exists("t2")


def test_list_and_iter_sorted(tmp_path):
    store = LocalTraceStore(tmp_path)
    for tid in ("t3", "t1", "t2"):
        store.put(make_trace(tid))
    assert store.list_ids() == ["t1", "t2", "t3"]
    assert [t.trace_id for t in store] == ["t1", "t2", "t3"]


def test_get_missing_raises(tmp_path):
    with pytest.raises(KeyError):
        LocalTraceStore(tmp_path).get("nope")


def test_put_overwrites_idempotently(tmp_path):
    store = LocalTraceStore(tmp_path)
    store.put(make_trace("t1"))
    updated = make_trace("t1")
    updated.status = "app_error"
    store.put(updated)
    assert store.get("t1").status == "app_error"
    assert store.list_ids() == ["t1"]


def test_rejects_path_traversal_ids(tmp_path):
    store = LocalTraceStore(tmp_path)
    with pytest.raises(ValueError):
        store.get("../escape")
