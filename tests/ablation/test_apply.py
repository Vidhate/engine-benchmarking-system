"""Step 4 — filter, sub-sample (disjointness-enforced), inject, record."""

from __future__ import annotations

import itertools
import random
from collections import Counter

from benchmark.ablation.apply import apply_ablations
from benchmark.ablation.plan import plan_ablation
from benchmark.ablation.split import make_split
from benchmark.schemas.configs import AblationConfig

from .conftest import FakeHarness, make_proposal


def _apply(proposals, traces, inputs, harness, split, *, seed=0, max_turns=3):
    specs = [plan_ablation(p) for p in proposals]
    return apply_ablations(
        specs,
        {p.issue.error_id: p for p in proposals},
        traces,
        inputs,
        split,
        harness,
        seed=seed,
        dataset_id=traces.dataset_id,
        max_turns=max_turns,
    )


def _all_split(inputs):
    """A split with nothing held back — for tests about step 4 alone."""
    return make_split(inputs, AblationConfig(seed=0, control_fraction=0.0))


# ------------------------------------------------------------------- basics

def test_injections_land_and_are_recorded(traces, inputs, harness):
    split = _all_split(inputs)
    proposal = make_proposal(target_count=3)
    outcome = _apply([proposal], traces, inputs, harness, split)
    assert outcome.injected == {"E-hallucination-00": 3}
    assert len(outcome.records) == 3
    assert len(outcome.ground_truth.occurrences) == 3
    assert [i.error_id for i in outcome.ground_truth.issues] == ["E-hallucination-00"]
    assert outcome.ground_truth.source == "ground_truth"


def test_the_ablated_dataset_keeps_one_trace_per_input_and_its_lineage(
    traces, inputs, harness
):
    split = _all_split(inputs)
    outcome = _apply([make_proposal(target_count=2)], traces, inputs, harness, split)
    assert len(outcome.ablated.traces) == len(traces.traces)
    assert [t.input_id for t in outcome.ablated.traces] == [t.input_id for t in traces.traces]
    assert outcome.ablated.parent_dataset_id == traces.dataset_id
    assert outcome.ablated.dataset_id != traces.dataset_id


def test_occurrences_reference_the_trace_that_actually_ships(traces, inputs, harness):
    split = _all_split(inputs)
    outcome = _apply([make_proposal(target_count=2)], traces, inputs, harness, split)
    shipped = {t.trace_id for t in outcome.ablated.traces}
    assert all(o.trace_id in shipped for o in outcome.ground_truth.occurrences)
    assert all(r.trace_id in shipped for r in outcome.records)


def test_the_sub_sample_respects_target_count(traces, inputs, harness):
    split = _all_split(inputs)
    outcome = _apply([make_proposal(target_count=2)], traces, inputs, harness, split)
    assert outcome.injected["E-hallucination-00"] == 2


def test_the_sub_sample_is_seeded_and_reproducible(traces, inputs, target_cfg, tmp_path):
    from benchmark.tracing.store import LocalTraceStore

    split = _all_split(inputs)
    picks = []
    for _ in range(2):
        store = LocalTraceStore(tmp_path / f"s{len(picks)}")
        for trace in traces.traces:
            store.put(trace.model_copy(deep=True))
        harness = FakeHarness(target_cfg, store)
        outcome = _apply([make_proposal(target_count=3)], traces, inputs, harness, split, seed=5)
        picks.append(
            sorted(r.actions_applied[0].params["source_trace_id"] for r in outcome.records)
        )
    assert picks[0] == picks[1]


def test_a_different_seed_picks_a_different_sub_sample(traces, inputs, harness):
    split = _all_split(inputs)
    a = _apply([make_proposal(target_count=3)], traces, inputs, harness, split, seed=1)
    b = _apply([make_proposal(target_count=3)], traces, inputs, harness, split, seed=77)
    picked = [
        sorted(r.actions_applied[0].params["source_trace_id"] for r in o.records) for o in (a, b)
    ]
    assert picked[0] != picked[1]


# ------------------------------------------------------- the split is honored

def test_control_inputs_are_never_selected(traces, inputs, harness):
    split = make_split(inputs, AblationConfig(seed=3, control_fraction=0.5))
    outcome = _apply([make_proposal(target_count=99)], traces, inputs, harness, split)
    control = set(split.control_input_ids)
    touched = {
        t.input_id for t in outcome.ablated.traces if t.ablation_ids
    }
    assert not touched & control


def test_control_traces_come_through_byte_identical(traces, inputs, harness):
    split = make_split(inputs, AblationConfig(seed=3, control_fraction=0.5))
    before = {t.input_id: t.model_dump_json() for t in traces.traces}
    outcome = _apply([make_proposal(target_count=99)], traces, inputs, harness, split)
    for trace in outcome.ablated.traces:
        if trace.input_id in set(split.control_input_ids):
            assert trace.model_dump_json() == before[trace.input_id], trace.input_id


# ------------------------------------------------ same-category disjointness

def test_no_trace_carries_two_errors_of_the_same_category(traces, inputs, harness):
    split = _all_split(inputs)
    proposals = [
        make_proposal("E-hallucination-00", "hallucination", target_count=99),
        make_proposal("E-hallucination-01", "hallucination", target_count=99),
    ]
    outcome = _apply(proposals, traces, inputs, harness, split)
    by_error = {p.issue.error_id: p.issue.category_id for p in proposals}
    seen = Counter(
        (o.trace_id, by_error[o.error_id]) for o in outcome.ground_truth.occurrences
    )
    assert seen and max(seen.values()) == 1, seen


def test_disjointness_holds_over_many_seeded_configurations(traces, inputs, target_cfg, tmp_path):
    """Property test: whatever the seed and whatever the error mix, the
    (trace_id, category_id) key stays unique — that is what makes Layer-1
    scoring an exact-key match instead of a text-similarity guess."""
    from benchmark.tracing.store import LocalTraceStore

    split = _all_split(inputs)
    rng = random.Random(2026)
    for run in range(12):
        store = LocalTraceStore(tmp_path / f"run{run}")
        for trace in traces.traces:
            store.put(trace.model_copy(deep=True))
        harness = FakeHarness(target_cfg, store)
        proposals = [
            make_proposal(
                f"E-{category}-{index:02d}",
                category,
                mode=rng.choice(["replay_edit", "dependency_fault"]),
                target_count=rng.randint(1, 6),
            )
            for index, category in enumerate(
                rng.choices(["hallucination", "retrieval_failure", "formatting"], k=5)
            )
        ]
        # de-duplicate error ids the random draw may have collided on
        proposals = list({p.issue.error_id: p for p in proposals}.values())
        outcome = _apply(
            proposals, traces, inputs, harness, split, seed=rng.randint(0, 10_000)
        )
        by_error = {p.issue.error_id: p.issue.category_id for p in proposals}
        keys = [(o.trace_id, by_error[o.error_id]) for o in outcome.ground_truth.occurrences]
        assert len(keys) == len(set(keys)), f"run {run}: duplicate exact key in {keys}"


def test_one_mode_c_and_one_mode_a_of_the_SAME_category_never_share_a_trace(
    traces, inputs, harness
):
    """The deterministic companion to the property test.

    The two modes are the one path that could sneak a second same-category
    injection onto a trace, because Mode C runs first and the trace it leaves
    behind is a fresh, fully eligible candidate for Mode A.
    """
    split = _all_split(inputs)
    proposals = [
        make_proposal("E-rf-fault", "retrieval_failure",
                      mode="dependency_fault", target_count=99),
        make_proposal("E-rf-content", "retrieval_failure", target_count=99),
    ]
    outcome = _apply(proposals, traces, inputs, harness, split)
    per_trace = Counter(o.trace_id for o in outcome.ground_truth.occurrences)
    assert per_trace, "the fixture should inject something"
    assert max(per_trace.values()) == 1, (
        f"a trace carries two retrieval_failure injections: {per_trace}"
    )
    # and each error still landed somewhere
    assert outcome.injected["E-rf-fault"] > 0


def test_a_trace_may_carry_compound_errors_from_different_categories(
    traces, inputs, harness
):
    split = _all_split(inputs)
    proposals = [
        make_proposal("E-hallucination-00", "hallucination", target_count=99),
        make_proposal(
            "E-retrieval_failure-00", "retrieval_failure",
            mode="dependency_fault", target_count=99,
        ),
    ]
    outcome = _apply(proposals, traces, inputs, harness, split)
    per_trace = Counter(o.trace_id for o in outcome.ground_truth.occurrences)
    assert max(per_trace.values()) == 2, "compound errors across categories must be possible"


def test_a_mechanism_fault_is_applied_before_a_content_corruption(traces, inputs, harness):
    """Mode C regenerates the whole trace; the other order would erase Mode A."""
    split = _all_split(inputs)
    proposals = [
        make_proposal("E-hallucination-00", "hallucination", target_count=99),
        make_proposal(
            "E-retrieval_failure-00", "retrieval_failure",
            mode="dependency_fault", target_count=99,
        ),
    ]
    outcome = _apply(proposals, traces, inputs, harness, split)
    compound = [
        trace
        for trace in outcome.ablated.traces
        if len(trace.ablation_ids) == 2
    ]
    assert compound, "the fixture corpus should produce at least one compound trace"
    replay_k = {
        record.trace_id: record.actions_applied[0].params["turn_index"]
        for record in outcome.records
        if record.mode == "replay_edit"
    }
    for trace in compound:
        # a replay_edit ran last, so the shipped trace is the spliced one —
        # at whichever turn that injection actually drew.
        k = replay_k[trace.trace_id]
        assert trace.turns[k].final_response.startswith("I have escalated")


def test_at_most_one_injection_per_mode_per_input(traces, inputs, harness):
    split = _all_split(inputs)
    proposals = [
        make_proposal("E-a", "hallucination", target_count=99),
        make_proposal("E-b", "formatting", target_count=99),
        make_proposal("E-c", "other", target_count=99),
    ]
    outcome = _apply(proposals, traces, inputs, harness, split)
    per_trace = Counter(o.trace_id for o in outcome.ground_truth.occurrences)
    assert max(per_trace.values()) == 1, "three replay_edits must not stack on one trace"


# -------------------------------------------------------------- failure paths

def test_an_injection_that_blows_up_moves_on_to_the_next_candidate(
    target_cfg, store, traces, inputs
):
    split = _all_split(inputs)
    harness = FakeHarness(target_cfg, store, replay_fails_for={"mt-00", "mt-01"})
    outcome = _apply(
        [make_proposal(target_count=len(traces.traces))], traces, inputs, harness, split
    )
    injected_inputs = {
        t.input_id for t in outcome.ablated.traces if t.ablation_ids
    }
    assert "mt-00" not in injected_inputs
    assert outcome.injected["E-hallucination-00"] >= len(traces.traces) - 2


def test_two_traces_for_one_input_fail_loudly(traces, inputs, harness):
    """Step 4 keys everything on input_id; a duplicate would silently lose an
    injection and mislabel the survivor's ground truth."""
    import pytest

    split = _all_split(inputs)
    doubled = traces.model_copy(deep=True)
    clone = doubled.traces[0].model_copy(deep=True, update={"trace_id": "trace-dup"})
    doubled.traces.append(clone)
    with pytest.raises(ValueError, match="duplicate input_id"):
        _apply([make_proposal()], doubled, inputs, harness, split)


def test_records_carry_their_injection_mode(traces, inputs, harness):
    split = _all_split(inputs)
    proposals = [
        make_proposal("E-a", "hallucination", target_count=2),
        make_proposal("E-b", "retrieval_failure", mode="dependency_fault", target_count=2),
    ]
    outcome = _apply(proposals, traces, inputs, harness, split)
    by_error = {r.error_id: r.mode for r in outcome.records}
    assert by_error == {"E-a": "replay_edit", "E-b": "dependency_fault"}


def test_a_self_corrected_candidate_is_counted_not_silently_skipped(
    traces, inputs, target_cfg, store
):
    """The not-retracted check burns candidates, and how many it burned bounds
    the residual risk documented in `inject.retraction_in`. Counting it is what
    turns that risk into a number in the report."""
    from benchmark.schemas.ablation import FilterStep

    harness = FakeHarness(target_cfg, store, self_corrects=True)
    split = _all_split(inputs)
    # Pinned to turn 0 of the conversations: only a trace with a regenerated
    # tail can self-correct at all.
    proposal = make_proposal(
        turn_index=0,
        target_count=3,
        filter_steps=[FilterStep(field="mode", op="eq", value="multi_turn")],
    )
    outcome = _apply([proposal], traces, inputs, harness, split)
    assert outcome.injected.get("E-hallucination-00", 0) == 0
    assert outcome.self_corrected == {"E-hallucination-00": 2}
    assert "E-hallucination-00" in outcome.dropped


def test_an_error_that_injects_nowhere_is_reported_not_silently_forgotten(
    traces, inputs, harness
):
    split = make_split(inputs, AblationConfig(seed=0, control_fraction=1.0))
    outcome = _apply([make_proposal(target_count=3)], traces, inputs, harness, split)
    assert outcome.injected["E-hallucination-00"] == 0
    assert "E-hallucination-00" in outcome.dropped
    assert outcome.ground_truth.issues == [], "an issue with no occurrence is not ground truth"


def test_every_ground_truth_issue_carries_its_injection_mode(traces, inputs, harness):
    split = _all_split(inputs)
    proposals = [
        make_proposal("E-hallucination-00", "hallucination", target_count=2),
        make_proposal(
            "E-retrieval_failure-00", "retrieval_failure",
            mode="dependency_fault", target_count=2,
        ),
    ]
    outcome = _apply(proposals, traces, inputs, harness, split)
    modes = {i.error_id: i.injection_mode for i in outcome.ground_truth.issues}
    assert modes == {
        "E-hallucination-00": "replay_edit",
        "E-retrieval_failure-00": "dependency_fault",
    }


def test_the_ablation_ids_on_a_shipped_trace_match_its_records(traces, inputs, harness):
    split = _all_split(inputs)
    outcome = _apply([make_proposal(target_count=99)], traces, inputs, harness, split)
    by_trace = {
        trace_id: sorted(r.ablation_id for r in group)
        for trace_id, group in itertools.groupby(
            sorted(outcome.records, key=lambda r: r.trace_id), key=lambda r: r.trace_id
        )
    }
    for trace in outcome.ablated.traces:
        assert trace.ablation_ids == by_trace.get(trace.trace_id, [])
