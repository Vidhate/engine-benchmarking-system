"""The declarative TraceFilter engine (step 2's selection half)."""

from __future__ import annotations

import pytest

from benchmark.ablation.filters import (
    DERIVED_FIELDS,
    UnknownFilterField,
    eligible,
    matches,
    resolve,
)
from benchmark.schemas.ablation import FilterStep, TraceFilter

from .conftest import make_trace


@pytest.fixture
def trace():
    return make_trace("t1", "safe-00", turns=1)


@pytest.fixture
def bare_trace():
    return make_trace("t2", "safe-02", turns=1, with_retrieval=False, with_tool=False)


# ---------------------------------------------------------------- resolution

def test_resolve_reads_a_scalar_field(trace):
    assert resolve(trace, "status") == ["ok"]
    assert resolve(trace, "mode") == ["single_turn"]


def test_resolve_walks_lists_with_a_wildcard(trace):
    assert resolve(trace, "turns[*].turn_index") == [0]
    assert set(resolve(trace, "turns[*].spans[*].span_type")) == {
        "agent", "tool", "retrieval", "llm"
    }


def test_resolve_indexes_a_list_position(trace):
    assert resolve(trace, "turns[0].turn_index") == [0]
    assert resolve(trace, "turns[9].turn_index") == []


def test_resolve_reads_into_metadata_dicts(trace):
    assert resolve(trace, "metadata.app") == ["target_app"]


def test_a_missing_path_resolves_to_nothing_rather_than_raising(trace):
    assert resolve(trace, "metadata.nope") == []
    assert resolve(trace, "turns[*].spans[*].outputs.absent") == []


def test_derived_fields_are_documented_and_computed(trace, bare_trace):
    assert resolve(trace, "turn_count") == [1]
    assert set(resolve(trace, "span_types")) == {"agent", "tool", "retrieval", "llm"}
    assert "corpus_search" in resolve(trace, "span_names")
    assert resolve(bare_trace, "span_types") == ["agent", "llm"]
    assert "turn_count" in DERIVED_FIELDS


def test_an_unknown_root_field_is_a_planning_bug_not_a_silent_miss(trace):
    with pytest.raises(UnknownFilterField, match="typo_field"):
        resolve(trace, "typo_field.deeper")


# ---------------------------------------------------------------------- ops

def test_eq_matches_when_any_resolved_value_equals(trace):
    assert matches(trace, TraceFilter(steps=[FilterStep(field="status", op="eq", value="ok")]))
    assert matches(
        trace,
        TraceFilter(steps=[FilterStep(field="span_types", op="eq", value="retrieval")]),
    )


def test_ne_requires_that_no_resolved_value_equals(trace):
    assert matches(
        trace, TraceFilter(steps=[FilterStep(field="status", op="ne", value="app_error")])
    )
    assert not matches(
        trace, TraceFilter(steps=[FilterStep(field="span_types", op="ne", value="llm")])
    )


def test_contains_is_case_insensitive_substring_on_text(trace):
    step = FilterStep(field="turns[*].final_response", op="contains", value="REFUND")
    assert matches(trace, TraceFilter(steps=[step]))


def test_contains_also_works_on_collections(trace):
    step = FilterStep(field="span_types", op="contains", value="tool")
    assert matches(trace, TraceFilter(steps=[step]))


def test_regex_matches_anywhere_in_a_value(trace):
    step = FilterStep(field="turns[*].final_response", op="regex", value=r"\d+ days")
    assert matches(trace, TraceFilter(steps=[step]))


def test_gt_and_lt_compare_numerically(trace):
    assert matches(trace, TraceFilter(steps=[FilterStep(field="turn_count", op="gt", value=0)]))
    assert not matches(trace, TraceFilter(steps=[FilterStep(field="turn_count", op="gt", value=5)]))
    assert matches(trace, TraceFilter(steps=[FilterStep(field="turn_count", op="lt", value=2)]))


def test_gt_ignores_values_that_are_not_numbers(trace):
    assert not matches(trace, TraceFilter(steps=[FilterStep(field="status", op="gt", value=1)]))


def test_exists_is_about_presence_not_truthiness(trace, bare_trace):
    step = FilterStep(field="turns[*].spans[*].outputs.documents", op="exists")
    assert not matches(trace, TraceFilter(steps=[step]))
    present = FilterStep(field="metadata.thread_id", op="exists")
    assert matches(trace, TraceFilter(steps=[present]))


def test_exists_false_asserts_absence(bare_trace):
    step = FilterStep(field="span_types", op="exists", value=False)
    assert not matches(bare_trace, TraceFilter(steps=[step]))
    absent = FilterStep(field="metadata.persona_id", op="exists", value=False)
    assert matches(bare_trace, TraceFilter(steps=[absent]))


def test_steps_are_conjunctive(trace, bare_trace):
    f = TraceFilter(
        steps=[
            FilterStep(field="status", op="eq", value="ok"),
            FilterStep(field="span_types", op="eq", value="retrieval"),
        ]
    )
    assert matches(trace, f)
    assert not matches(bare_trace, f)


def test_an_empty_filter_matches_everything(trace, bare_trace):
    assert matches(trace, TraceFilter())
    assert matches(bare_trace, TraceFilter())


def test_an_unknown_operator_is_refused(trace):
    step = FilterStep.model_construct(field="status", op="wat", value="ok")
    with pytest.raises(ValueError, match="unsupported filter op"):
        matches(trace, TraceFilter(steps=[step]))


# ------------------------------------------------------------------ eligible

def test_eligible_selects_within_a_population_only(traces, inputs):
    ablate_ids = {i.input_id for i in inputs.inputs if i.input_id.startswith("safe")}
    f = TraceFilter(steps=[FilterStep(field="span_types", op="eq", value="retrieval")])
    picked = eligible(traces.traces, f, ablate_ids)
    assert picked, "the fixture corpus should have retrieval traces"
    assert all(t.input_id in ablate_ids for t in picked)
    assert all(t.input_id != "safe-02" for t in picked), "safe-02 never touched the retriever"


def test_eligible_returns_traces_in_a_stable_order(traces, inputs):
    ids = {i.input_id for i in inputs.inputs}
    f = TraceFilter()
    first = [t.trace_id for t in eligible(traces.traces, f, ids)]
    shuffled = list(reversed(traces.traces))
    second = [t.trace_id for t in eligible(shuffled, f, ids)]
    assert first == second == sorted(first)
