"""Scorer 4 — description deviation over Layer-2 pairs
(docs/architecture/06-scoring.md).

"similarity" mode: TF-IDF cosine between descriptions (cheap, gameable).
"judge" mode: LLM judge behind the `DescriptionJudge` Protocol — mocked here,
NO network calls in unit tests.
"""

import pytest

from benchmark.schemas import ErrorMatch, Issue, Issueboard, ScoringConfig
from benchmark.scoring.scorer_description import (
    DescriptionJudge,
    OpenAIDescriptionJudge,
    score_descriptions,
)


def make_issue(error_id, description, category_id="retrieval"):
    return Issue(
        error_id=error_id, title=error_id, description=description,
        category_id=category_id, severity="medium",
    )


class FakeJudge:
    """A DescriptionJudge Protocol-compliant mock — no network calls."""

    def __init__(self, fixed_score: float = 0.75):
        self.fixed_score = fixed_score
        self.calls: list[tuple[str, str]] = []

    def score(self, gt_issue: Issue, pred_issue: Issue) -> float:
        self.calls.append((gt_issue.error_id, pred_issue.error_id))
        return self.fixed_score


def test_similarity_mode_scores_identical_descriptions_as_one():
    gt = Issueboard(
        source="ground_truth",
        issues=[make_issue("K1", "the retriever returned stale outdated documents")],
    )
    pred = Issueboard(
        source="engine_predicted",
        issues=[make_issue("P1", "the retriever returned stale outdated documents")],
    )
    matches = [ErrorMatch(predicted_error_id="P1", matched_error_id="K1", overlap=1)]
    cfg = ScoringConfig(description_mode="similarity")
    scores = score_descriptions(matches, gt, pred, cfg)
    assert scores == {"K1": pytest.approx(1.0)}


def test_similarity_mode_scores_unrelated_descriptions_low():
    gt = Issueboard(
        source="ground_truth", issues=[make_issue("K1", "retriever returned stale documents")]
    )
    pred = Issueboard(
        source="engine_predicted",
        issues=[make_issue("P1", "tool call failed with a network timeout")],
    )
    matches = [ErrorMatch(predicted_error_id="P1", matched_error_id="K1", overlap=1)]
    cfg = ScoringConfig(description_mode="similarity")
    scores = score_descriptions(matches, gt, pred, cfg)
    assert scores["K1"] == pytest.approx(0.0)


def test_unmatched_pairs_are_excluded():
    gt = Issueboard(source="ground_truth", issues=[])
    pred = Issueboard(source="engine_predicted", issues=[make_issue("P1", "phantom")])
    matches = [ErrorMatch(predicted_error_id="P1", matched_error_id=None, overlap=0)]
    cfg = ScoringConfig(description_mode="similarity")
    assert score_descriptions(matches, gt, pred, cfg) == {}


def test_many_to_one_pairs_are_averaged_per_known_error():
    gt = Issueboard(source="ground_truth", issues=[make_issue("K1", "aaa bbb ccc")])
    pred = Issueboard(
        source="engine_predicted",
        issues=[make_issue("P1", "aaa bbb ccc"), make_issue("P2", "xxx yyy zzz")],
    )
    matches = [
        ErrorMatch(predicted_error_id="P1", matched_error_id="K1", overlap=2),
        ErrorMatch(predicted_error_id="P2", matched_error_id="K1", overlap=1),
    ]
    cfg = ScoringConfig(description_mode="similarity")
    scores = score_descriptions(matches, gt, pred, cfg)
    # P1 vs K1 similarity == 1.0, P2 vs K1 similarity == 0.0 -> mean 0.5
    assert scores["K1"] == pytest.approx(0.5)


def test_judge_mode_uses_injected_judge_no_network():
    gt = Issueboard(source="ground_truth", issues=[make_issue("K1", "stale docs")])
    pred = Issueboard(source="engine_predicted", issues=[make_issue("P1", "stale docs found")])
    matches = [ErrorMatch(predicted_error_id="P1", matched_error_id="K1", overlap=1)]
    cfg = ScoringConfig(description_mode="judge")
    judge = FakeJudge(fixed_score=0.75)
    scores = score_descriptions(matches, gt, pred, cfg, judge=judge)
    assert scores == {"K1": 0.75}
    assert judge.calls == [("K1", "P1")]


def test_judge_mode_without_a_judge_raises_instead_of_calling_network():
    gt = Issueboard(source="ground_truth", issues=[make_issue("K1", "stale docs")])
    pred = Issueboard(source="engine_predicted", issues=[make_issue("P1", "stale docs found")])
    matches = [ErrorMatch(predicted_error_id="P1", matched_error_id="K1", overlap=1)]
    cfg = ScoringConfig(description_mode="judge")
    with pytest.raises(ValueError, match="judge"):
        score_descriptions(matches, gt, pred, cfg, judge=None)


def test_fake_judge_satisfies_the_description_judge_protocol():
    judge: DescriptionJudge = FakeJudge()
    assert isinstance(judge, DescriptionJudge)


def test_openai_judge_skeleton_never_calls_network_in_unit_tests():
    # Constructing the skeleton must not require the `openai` package or make
    # any network call; only invoking .score() would need a real client, and
    # we never do that here.
    judge = OpenAIDescriptionJudge()
    assert isinstance(judge, DescriptionJudge)
    gt_issue = make_issue("K1", "stale docs")
    pred_issue = make_issue("P1", "stale docs found")
    with pytest.raises(RuntimeError, match="openai"):
        judge.score(gt_issue, pred_issue)
