"""The trace-inspection layer: loading, indexing, and the four operations."""

from __future__ import annotations

import json

import pytest

from engine.traces import MAX_RESULT_CHARS, SNIPPET_CHARS, TraceIndex, load_traces, truncate

ALL_TRACE_IDS = [
    "trace-clean-pricing",
    "trace-clean-platforms",
    "trace-clean-export",
    "trace-planted-refund",
    "trace-planted-ticket",
    "trace-planted-truncated",
]


def test_loads_every_fixture_trace(index):
    assert index.trace_ids == ALL_TRACE_IDS


def test_loads_a_bare_list_payload(tmp_path, traces_file):
    payload = json.loads(traces_file.read_text())
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(payload["traces"]))
    assert [t.trace_id for t in load_traces(bare)] == ALL_TRACE_IDS


def test_rejects_an_unsupported_payload(tmp_path):
    path = tmp_path / "nope.json"
    path.write_text('"just a string"')
    with pytest.raises(ValueError, match="unsupported trace file payload"):
        load_traces(path)


def test_get_trace_reports_turns_and_span_table(index):
    view = json.loads(index.get_trace("trace-planted-ticket"))
    assert view["trace_id"] == "trace-planted-ticket"
    assert view["mode"] == "single_turn"
    assert len(view["turns"]) == 1
    turn = view["turns"][0]
    assert "escalate this to a human" in turn["user_message"]
    assert "NN-48213" in turn["final_response"]
    assert [s["span_id"] for s in turn["spans"]] == ["s-t-0", "s-t-1", "s-t-2", "s-t-3"]
    # The overview is an index, not a dump: span payloads require read_span.
    assert "TicketServiceError" not in json.dumps(view["turns"][0]["spans"])


def test_get_trace_covers_every_turn_of_a_multi_turn_trace(index):
    view = json.loads(index.get_trace("trace-clean-export"))
    assert [t["turn_index"] for t in view["turns"]] == [0, 1]


def test_get_trace_on_an_unknown_id_lists_what_exists(index):
    view = json.loads(index.get_trace("nope"))
    assert "unknown trace_id" in view["error"]
    assert view["known_trace_ids"] == ALL_TRACE_IDS


def test_list_spans_flags_the_errored_span(index):
    rows = json.loads(index.list_spans("trace-planted-ticket"))["spans"]
    errored = [r for r in rows if r["has_error"]]
    assert [r["span_id"] for r in errored] == ["s-t-2"]
    assert errored[0]["name"] == "create_ticket"
    assert errored[0]["span_type"] == "tool"


def test_list_spans_narrows_to_one_turn(index):
    rows = json.loads(index.list_spans("trace-clean-export", turn_index=1))["spans"]
    assert {r["turn_index"] for r in rows} == {1}
    assert [r["span_id"] for r in rows] == ["s-e-3", "s-e-4"]


def test_list_spans_on_a_missing_turn_says_so(index):
    result = json.loads(index.list_spans("trace-clean-export", turn_index=7))
    assert "no turn with turn_index=7" in result["error"]


def test_read_span_returns_full_payload_and_its_turn(index):
    span = json.loads(index.read_span("trace-planted-refund", "s-r-2"))
    assert span["span_type"] == "retrieval"
    assert span["turn_index"] == 0
    assert "within 14 days of purchase" in json.dumps(span["outputs"])


def test_read_span_on_an_unknown_span_lists_the_known_ones(index):
    result = json.loads(index.read_span("trace-planted-refund", "s-nope"))
    assert "no span 's-nope'" in result["error"]
    assert "s-r-2" in result["known_span_ids"]


def test_read_span_on_an_unknown_trace_reports_the_trace(index):
    result = json.loads(index.read_span("nope", "s-r-2"))
    assert "unknown trace_id" in result["error"]


def test_search_text_finds_the_contradiction_across_the_corpus(index):
    hits = json.loads(index.search_text("14 days"))
    assert hits["location_count"] >= 1
    assert {h["trace_id"] for h in hits["locations"]} == {"trace-planted-refund"}
    assert any(h["location"] == "span.outputs" for h in hits["locations"])


def test_search_text_scopes_to_one_trace(index):
    everywhere = json.loads(index.search_text("Nimbus"))
    scoped = json.loads(index.search_text("Nimbus", trace_id="trace-clean-pricing"))
    assert everywhere["location_count"] > scoped["location_count"]
    assert {h["trace_id"] for h in scoped["locations"]} == {"trace-clean-pricing"}


def test_search_text_is_case_insensitive(index):
    assert json.loads(index.search_text("TICKETSERVICEERROR"))["location_count"] >= 1


def test_search_text_reports_a_miss_as_zero_hits(index):
    miss = json.loads(index.search_text("quantum entanglement"))
    assert miss["location_count"] == 0
    assert miss["total_matches"] == 0


def test_search_text_separates_matching_fields_from_total_occurrences(index):
    """`location_count` counts FIELDS that matched; a field mentioning the query
    ten times is still one location. Conflating the two is how an agent decides
    "only one mention" about a span that repeats it throughout."""
    result = json.loads(index.search_text("Nimbus"))
    assert result["total_matches"] > result["location_count"]
    assert all(hit["matches_here"] >= 1 for hit in result["locations"])


def test_the_counts_describe_the_corpus_not_the_returned_page(index):
    """`locations` is capped by `limit`; both counts still report the truth
    about the whole search, which is the point of reporting them separately."""
    capped = json.loads(index.search_text("Nimbus", limit=2))
    assert len(capped["locations"]) == 2
    assert capped["location_count"] > 2
    assert capped["total_matches"] >= capped["location_count"]

    # With nothing dropped, the totals reconcile exactly.
    whole = json.loads(index.search_text("Nimbus", trace_id="trace-clean-pricing"))
    assert whole["location_count"] == len(whole["locations"])
    assert whole["total_matches"] == sum(h["matches_here"] for h in whole["locations"])


def test_matches_here_counts_every_occurrence_in_one_field(index):
    """The refund answer's own span output repeats "refund" more than once."""
    result = json.loads(index.search_text("refund", trace_id="trace-planted-refund"))
    assert max(hit["matches_here"] for hit in result["locations"]) > 1


def test_search_text_rejects_an_empty_query(index):
    assert "empty query" in json.loads(index.search_text("   "))["error"]


def test_search_text_on_an_unknown_trace_reports_it(index):
    assert "unknown trace_id" in json.loads(index.search_text("Nimbus", "nope"))["error"]


def test_search_snippets_are_bounded(index):
    hits = json.loads(index.search_text("Nimbus"))
    assert all(len(h["snippet"]) <= SNIPPET_CHARS for h in hits["locations"])


def test_truncate_marks_what_it_dropped():
    assert truncate("abc", 10) == "abc"
    result = truncate("x" * 100, 10)
    assert result.startswith("x" * 10)
    assert "90 more characters" in result


@pytest.mark.parametrize("multiplier", [1, 40])
def test_oversized_tool_results_stay_valid_json_within_budget(index, multiplier):
    """Truncation must never hand the model an unparseable object."""
    big = TraceIndex(index.traces * multiplier)
    results = [
        big.search_text("Nimbus", limit=10_000),
        big.get_trace("trace-planted-ticket"),
        big.list_spans("trace-planted-ticket"),
        big.read_span("trace-planted-ticket", "s-t-2"),
    ]
    for result in results:
        assert len(result) <= MAX_RESULT_CHARS + 200
        json.loads(result)  # parses, truncated or not


def test_item_shedding_reports_how_much_it_dropped(index):
    big = TraceIndex(index.traces * 40)
    payload = json.loads(big.search_text("Nimbus", limit=10_000))
    assert payload["location_count"] > len(payload["locations"])
    assert "of" in payload["truncated"] and "locations" in payload["truncated"]


def test_a_single_oversized_item_degrades_to_a_valid_envelope():
    from engine.traces import fit_json

    payload = {"locations": [{"snippet": "x" * 20_000}]}
    result = json.loads(fit_json(payload, "locations"))
    assert result["truncated"] is True
    assert isinstance(result["content"], str)
