"""What actually gets scored: the ENGINE'S delta, restricted to the real traces.

Two rulings live here.

**Phantom trace ids (I2).** A predicted occurrence naming a trace that is not in
the dataset is not a false positive about that trace — there is no trace. It is
dropped before scoring, counted, and reported, because silently keeping it
would charge the Engine precision for a trace nobody ran.

**Seed-board semantics (I6).** The assignment hands the Engine a board and asks
for the UPDATED board back, so the returned board contains issues the Engine did
not author. Scoring the seed's own issues as Engine predictions would credit or
punish it for text it was given. The delta is:

* seed-originated issues that gained no new occurrence -> dropped entirely;
* occurrence pairs that were already on the seed board -> dropped;
* occurrences the Engine ADDED to a seed issue -> kept and scored (they are
  genuine predictions, and the exact key resolves them like any other);
* the seed issue that carries them -> kept only as a carrier: its
  description and severity are seed-authored, so it is excluded from Layer-2
  description/severity pairing and never counted as an E_h candidate.
"""

from __future__ import annotations

import pytest

from benchmark.pipeline.scoring import prepare_scored_board, score_engine_delta
from benchmark.schemas import (
    EngineConfig,
    Issue,
    Issueboard,
    IssueOccurrence,
    ScoringConfig,
)

TRACES = ["t1", "t2", "t3", "t4"]


def issue(error_id, category="tool_misuse", severity="high", description="d"):
    return Issue(
        error_id=error_id,
        title=f"title {error_id}",
        description=description,
        category_id=category,
        severity=severity,
    )


def occ(error_id, trace_id):
    return IssueOccurrence(error_id=error_id, trace_id=trace_id)


@pytest.fixture
def seed():
    return Issueboard(
        source="seed",
        issues=[issue("S-carrier"), issue("S-untouched", category="hallucination")],
        occurrences=[occ("S-carrier", "t1")],
    )


@pytest.fixture
def predicted(seed):
    """The updated board: the seed plus what the Engine did with it."""
    return Issueboard(
        source="engine_predicted",
        issues=[*seed.issues, issue("P-new", category="retrieval_failure")],
        occurrences=[
            occ("S-carrier", "t1"),  # pre-existing seed pair
            occ("S-carrier", "t2"),  # ENGINE-ADDED to a seed issue
            occ("P-new", "t3"),  # engine-authored issue
            occ("P-new", "tr-unknown"),  # phantom: no such trace
        ],
    )


# ------------------------------------------------------ phantom trace ids (I2)

def test_an_occurrence_on_a_trace_that_does_not_exist_is_dropped(seed, predicted):
    result = prepare_scored_board(predicted, seed, TRACES)
    assert all(o.trace_id in TRACES for o in result.board.occurrences)


def test_dropped_phantoms_are_counted_and_named(seed, predicted):
    result = prepare_scored_board(predicted, seed, TRACES)
    assert result.phantom_trace_ids == ["tr-unknown"]
    assert result.phantom_occurrences == [("P-new", "tr-unknown")]


def test_a_board_with_no_phantoms_reports_none(seed):
    clean = Issueboard(
        source="engine_predicted", issues=[issue("P1")], occurrences=[occ("P1", "t1")]
    )
    assert prepare_scored_board(clean, seed, TRACES).phantom_occurrences == []


def test_an_issue_left_with_only_phantoms_does_not_survive_as_a_prediction(seed):
    ghost = Issueboard(
        source="engine_predicted",
        issues=[issue("P-ghost")],
        occurrences=[occ("P-ghost", "tr-nope")],
    )
    result = prepare_scored_board(ghost, seed, TRACES)
    assert result.board.occurrences == []


# ------------------------------------------------------- the engine delta (I6)

def test_the_seed_s_own_occurrence_pairs_are_not_scored(seed, predicted):
    result = prepare_scored_board(predicted, seed, TRACES)
    pairs = {(o.error_id, o.trace_id) for o in result.board.occurrences}
    assert ("S-carrier", "t1") not in pairs, "the seed's own pair was scored as a prediction"
    assert result.dropped_seed_occurrences == 1


def test_an_engine_added_occurrence_on_a_seed_issue_is_kept(seed, predicted):
    """That is a genuine prediction: the Engine said this issue happens here."""
    result = prepare_scored_board(predicted, seed, TRACES)
    pairs = {(o.error_id, o.trace_id) for o in result.board.occurrences}
    assert ("S-carrier", "t2") in pairs


def test_a_seed_issue_that_gained_nothing_is_dropped_entirely(seed, predicted):
    result = prepare_scored_board(predicted, seed, TRACES)
    assert "S-untouched" not in {i.error_id for i in result.board.issues}
    assert result.dropped_seed_issues == ["S-untouched"]


def test_a_seed_issue_that_gained_occurrences_survives_as_a_carrier(seed, predicted):
    result = prepare_scored_board(predicted, seed, TRACES)
    assert "S-carrier" in {i.error_id for i in result.board.issues}
    assert result.carrier_error_ids == ["S-carrier"]


def test_engine_authored_issues_are_untouched(seed, predicted):
    result = prepare_scored_board(predicted, seed, TRACES)
    assert "P-new" in {i.error_id for i in result.board.issues}
    assert "P-new" not in result.carrier_error_ids


def test_an_empty_seed_board_changes_nothing(predicted):
    empty = Issueboard(source="seed")
    result = prepare_scored_board(predicted, empty, TRACES)
    assert result.carrier_error_ids == []
    assert result.dropped_seed_issues == []
    assert result.dropped_seed_occurrences == 0
    assert {i.error_id for i in result.board.issues} == {
        i.error_id for i in predicted.issues
    }


def test_the_scored_board_is_stamped_and_still_a_prediction(seed, predicted):
    result = prepare_scored_board(predicted, seed, TRACES)
    assert result.board.source == "engine_predicted"
    assert result.board.board_id


# ------------------------------------------------- carriers are not predictions

@pytest.fixture
def ground_truth():
    return Issueboard(
        source="ground_truth",
        issues=[
            issue("K1", category="tool_misuse", severity="low", description="known tool misuse"),
            issue("K2", category="retrieval_failure", severity="high"),
        ],
        occurrences=[occ("K1", "t2"), occ("K2", "t3")],
    )


def scored(predicted, seed, ground_truth, **kwargs):
    return score_engine_delta(
        ground_truth=ground_truth,
        predicted=predicted,
        seed=seed,
        trace_ids=TRACES,
        cfg=ScoringConfig(description_mode="similarity"),
        base_rates={},
        engine_config=EngineConfig(model="m"),
        **kwargs,
    )


def test_a_carrier_is_never_an_eh_candidate(seed, ground_truth):
    """The reviewer's case: a seed issue must not be reported as a discovery."""
    predicted = Issueboard(
        source="engine_predicted",
        issues=list(seed.issues),
        # An engine-added occurrence on a seed issue, on a trace with no
        # matching injection -> the carrier resolves to nothing.
        occurrences=[occ("S-carrier", "t4")],
    )
    report, result = scored(predicted, seed, ground_truth)
    assert result.carrier_error_ids == ["S-carrier"]
    assert "S-carrier" not in report.eh_candidates


def test_an_engine_added_occurrence_on_a_seed_issue_is_scored_not_a_false_positive(
    seed, ground_truth
):
    """K1 is a tool_misuse injection on t2; the seed carrier is tool_misuse too,
    so the exact key resolves it — the Engine gets the credit."""
    predicted = Issueboard(
        source="engine_predicted",
        issues=list(seed.issues),
        occurrences=[occ("S-carrier", "t1"), occ("S-carrier", "t2")],
    )
    report, _ = scored(predicted, seed, ground_truth)
    resolved = {m.resolved_error_id for m in report.occurrence_matches}
    assert "K1" in resolved
    assert all(m.resolved_error_id is not None for m in report.occurrence_matches)


def test_a_carrier_does_not_contribute_its_seed_authored_description(seed, ground_truth):
    """Scorer 4 would otherwise grade text the benchmark itself wrote."""
    predicted = Issueboard(
        source="engine_predicted",
        issues=list(seed.issues),
        occurrences=[occ("S-carrier", "t2")],
    )
    report, _ = scored(predicted, seed, ground_truth)
    assert report.description_scores == {}
    assert report.headline["mean_description_score"] == 0.0


def test_a_carrier_does_not_contribute_its_seed_authored_severity(seed, ground_truth):
    """S-carrier is 'high', K1 is 'low' — a real over-call, but not the Engine's."""
    predicted = Issueboard(
        source="engine_predicted",
        issues=list(seed.issues),
        occurrences=[occ("S-carrier", "t2")],
    )
    report, _ = scored(predicted, seed, ground_truth)
    assert report.severity_loss == 0.0
    assert report.headline["mean_severity_loss"] == 0.0


def test_an_engine_authored_issue_still_gets_its_description_and_severity_scored(
    seed, ground_truth
):
    predicted = Issueboard(
        source="engine_predicted",
        issues=[*seed.issues, issue("P-new", category="retrieval_failure", severity="high")],
        occurrences=[occ("P-new", "t3")],
    )
    report, _ = scored(predicted, seed, ground_truth)
    assert "K2" in report.description_scores


def test_the_delta_is_recorded_in_the_reports_base_rates(seed, predicted, ground_truth):
    report, _ = scored(predicted, seed, ground_truth)
    delta = report.base_rates["engine_delta"]
    assert delta["carrier_error_ids"] == ["S-carrier"]
    assert delta["dropped_seed_issues"] == ["S-untouched"]
    assert delta["dropped_seed_occurrences"] == 1
    assert delta["phantom_occurrences"] == 1
    assert delta["phantom_trace_ids"] == ["tr-unknown"]


def test_the_report_is_stamped_after_the_carrier_adjustment(seed, ground_truth):
    from benchmark.schemas.io import content_hash

    predicted = Issueboard(
        source="engine_predicted",
        issues=list(seed.issues),
        occurrences=[occ("S-carrier", "t2")],
    )
    report, _ = scored(predicted, seed, ground_truth)
    assert report.report_id == content_hash(report)


def test_scoring_still_sees_every_trace_including_clean_ones(seed, predicted, ground_truth):
    report, _ = scored(predicted, seed, ground_truth)
    # t4 is in no board; kappa still has to know it exists.
    supports = {s.category_id: s.support for s in report.category_scores}
    assert supports, "no categories scored"
    assert report.category_scores[0].cohens_kappa is not None
