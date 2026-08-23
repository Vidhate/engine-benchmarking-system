"""Scorer 4 — description deviation over Layer-2 pairs
(docs/architecture/06-scoring.md).

Two interchangeable scoring backends behind the same [0, 1] contract:
- "similarity": TF-IDF cosine between descriptions — cheap, gameable, no deps.
- "judge": an LLM judge behind the `DescriptionJudge` Protocol. A trivial
  OpenAI-backed skeleton is provided (`OpenAIDescriptionJudge`) for later
  wiring; it requires the `openai` package (NOT a benchmark dependency) and
  is never exercised in unit tests — those inject a mock judge instead.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol, runtime_checkable

from benchmark.schemas import ErrorMatch, Issue, Issueboard, ScoringConfig
from benchmark.scoring.tfidf import text_similarity


@runtime_checkable
class DescriptionJudge(Protocol):
    """Behind-an-interface LLM judge for description quality. Implementations
    return a score in [0, 1]: does the predicted write-up identify the same
    root cause / failure surface as the ground-truth known error?"""

    def score(self, gt_issue: Issue, pred_issue: Issue) -> float: ...


class OpenAIDescriptionJudge:
    """Skeleton OpenAI-backed DescriptionJudge (docs/execution-plan.md pins the
    model in benchmark/models.py: DESCRIPTION_JUDGE_MODEL).

    This is intentionally a skeleton: constructing it never touches the
    network or requires `openai` to be installed (not a benchmark runtime
    dependency); only calling `.score()` for live judging would need a real
    client. Unit tests must inject a mock `DescriptionJudge` instead.
    """

    def __init__(self, model: str | None = None, client: object | None = None):
        if model is None:
            from benchmark.models import DESCRIPTION_JUDGE_MODEL

            model = DESCRIPTION_JUDGE_MODEL
        self.model = model
        self._client = client

    def _build_client(self):
        try:
            import openai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only without openai installed
            raise RuntimeError(
                "OpenAIDescriptionJudge.score() requires the `openai` package "
                "(not a benchmark runtime dependency) or an injected client. "
                "Unit tests should inject a mock DescriptionJudge instead."
            ) from exc
        raise RuntimeError(
            "OpenAIDescriptionJudge is a skeleton — the `openai` client is "
            "importable but no real client is wired up for live judge scoring."
        )

    def score(self, gt_issue: Issue, pred_issue: Issue) -> float:
        client = self._client or self._build_client()
        raise NotImplementedError(
            f"OpenAIDescriptionJudge skeleton has no live rubric call wired up "
            f"(client={client!r})."
        )


def score_descriptions(
    matches: list[ErrorMatch],
    gt: Issueboard,
    pred: Issueboard,
    cfg: ScoringConfig,
    judge: DescriptionJudge | None = None,
) -> dict[str, float]:
    """Per matched known error: mean description-deviation score in [0, 1]
    over every predicted issue paired to it (many-to-one averages)."""
    if cfg.description_mode == "judge" and judge is None:
        raise ValueError(
            "description_mode='judge' requires an explicit DescriptionJudge "
            "(inject a mock in tests; live scoring wires an OpenAIDescriptionJudge)."
        )

    gt_by_id = {issue.error_id: issue for issue in gt.issues}
    pred_by_id = {issue.error_id: issue for issue in pred.issues}

    scores_by_known: dict[str, list[float]] = defaultdict(list)
    for m in matches:
        if m.matched_error_id is None:
            continue
        gt_issue = gt_by_id.get(m.matched_error_id)
        pred_issue = pred_by_id.get(m.predicted_error_id)
        if gt_issue is None or pred_issue is None:
            continue

        if cfg.description_mode == "similarity":
            score = text_similarity(pred_issue.description, gt_issue.description)
        else:
            score = judge.score(gt_issue, pred_issue)

        scores_by_known[m.matched_error_id].append(score)

    return {
        error_id: sum(values) / len(values) for error_id, values in scores_by_known.items()
    }
