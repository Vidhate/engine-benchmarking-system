"""Gate: content-hash ids and parent lineage helpers behave deterministically."""

from datetime import UTC, datetime

import pytest

from benchmark.schemas import Issueboard, Trace, TraceDataset
from benchmark.schemas.io import content_hash, derive, load, save, stamp_dataset_id
from tests.test_schemas_roundtrip import make_input_dataset, make_issueboard, make_trace


def test_same_content_same_id():
    a = stamp_dataset_id(TraceDataset(traces=[make_trace()]))
    b = stamp_dataset_id(TraceDataset(traces=[make_trace()]))
    assert a.dataset_id == b.dataset_id != ""


def test_different_content_different_id():
    a = stamp_dataset_id(TraceDataset(traces=[make_trace("t1")]))
    b = stamp_dataset_id(TraceDataset(traces=[make_trace("t2")]))
    assert a.dataset_id != b.dataset_id


def test_id_field_does_not_affect_hash():
    ds = TraceDataset(traces=[make_trace()])
    assert content_hash(ds) == content_hash(ds.model_copy(update={"dataset_id": "zzz"}))


def test_stamp_works_for_board_and_report_ids():
    board = stamp_dataset_id(make_issueboard())
    assert board.board_id != ""


def test_derive_links_parent():
    parent = stamp_dataset_id(TraceDataset(traces=[make_trace()]))
    child = derive(TraceDataset(traces=[make_trace("t2")]), parent)
    assert child.parent_dataset_id == parent.dataset_id
    assert child.dataset_id not in ("", parent.dataset_id)


def test_derive_requires_stamped_parent():
    with pytest.raises(ValueError):
        derive(TraceDataset(), TraceDataset())  # parent unstamped


def test_derive_rejects_models_without_parent_field():
    parent = stamp_dataset_id(TraceDataset(traces=[make_trace()]))
    with pytest.raises(TypeError):
        derive(make_issueboard(), parent)  # Issueboard has no parent_dataset_id


def test_save_load_roundtrip(tmp_path):
    ds = stamp_dataset_id(TraceDataset(traces=[make_trace()]))
    path = save(ds, tmp_path / "nested" / "traces.json")
    assert load(TraceDataset, path) == ds


def test_save_load_preserves_datetime(tmp_path):
    trace = make_trace()
    path = save(trace, tmp_path / "t.json")
    restored = load(Trace, path)
    assert restored.turns[0].spans[0].start_time == trace.turns[0].spans[0].start_time


def test_board_hash_ignores_board_id_only():
    a = make_issueboard()
    b = Issueboard(
        board_id="different", source=a.source, issues=a.issues, occurrences=a.occurrences
    )
    assert content_hash(a) == content_hash(b)


def test_created_at_does_not_affect_content_hash():
    """created_at is volatile provenance metadata (wall-clock at generation time),
    not content — two datasets differing only in created_at must hash identically
    so real (non-test-clock) reruns of the same config+seed are reproducible."""
    ds = make_input_dataset()
    a = ds.model_copy(update={"created_at": datetime(2020, 1, 1, tzinfo=UTC)})
    b = ds.model_copy(update={"created_at": datetime(2030, 6, 15, 12, 30, tzinfo=UTC)})
    assert content_hash(a) == content_hash(b)
    assert stamp_dataset_id(a).dataset_id == stamp_dataset_id(b).dataset_id
