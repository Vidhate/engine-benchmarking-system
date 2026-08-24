"""What gets scored, as opposed to what the Engine returned.

The Engine's output is the **updated issueboard** — the assignment's deliverable,
persisted verbatim. It is not, however, the same thing as the Engine's
*predictions*, and two corrections stand between them. Both live here, in the
pipeline layer: `benchmark/scoring/` implements the scoring semantics, and this
module decides what it is handed.

**1. Phantom trace ids.** An occurrence naming a trace that is not in the
dataset is not a false positive *about that trace* — there is no trace to be
wrong about. Scoring it charges the Engine's precision for a run nobody made,
and it is the kind of thing a model does when it paraphrases an id. Such
occurrences are dropped before scoring, counted, and reported everywhere the
run is described.

**2. The seed board is not a prediction.** The assignment's input includes an
issueboard, so the returned board contains issues the Engine did not author.
Scoring them as predictions grades the benchmark's own text. What the Engine
actually contributed is the delta:

* **An issue the Engine authored** — scored fully: occurrences, category,
  severity, description.
* **An occurrence the Engine ADDED to a seed issue** — scored. It is a claim
  about where that failure happens, and the exact key resolves it like any
  other prediction.
* **A seed issue that gained occurrences** — kept as a *carrier* for them, and
  nothing more. Its severity and description were written by the benchmark, so
  it is excluded from Layer-2 severity/description pairing and is never
  reported as an E_h candidate.
* **A seed issue that gained nothing** — dropped. The Engine said nothing
  about it.
* **An occurrence pair already on the seed board** — dropped. The Engine was
  handed it.

The carrier exclusion is applied by re-running scorers 3 and 4 over the matches
with carriers removed, rather than by reaching into `benchmark/scoring/`: those
scorers are public functions over an explicit match list, which is exactly the
seam this needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.schemas import (
    BenchmarkReport,
    EngineConfig,
    Issueboard,
    ScoringConfig,
)
from benchmark.schemas.io import stamp_dataset_id
from benchmark.scoring import score, score_descriptions, score_severity
from benchmark.scoring.scorer_description import DescriptionJudge


@dataclass
class ScoredBoard:
    """The board handed to `score()`, and what was removed to get there."""

    board: Issueboard
    #: Seed issues kept only because the Engine added occurrences to them.
    carrier_error_ids: list[str] = field(default_factory=list)
    #: Seed issues the Engine said nothing about.
    dropped_seed_issues: list[str] = field(default_factory=list)
    #: Occurrence pairs that were already on the seed board.
    dropped_seed_occurrences: int = 0
    #: (error_id, trace_id) predicted against a trace that does not exist.
    phantom_occurrences: list[tuple[str, str]] = field(default_factory=list)

    @property
    def phantom_trace_ids(self) -> list[str]:
        return sorted({trace_id for _, trace_id in self.phantom_occurrences})

    def as_dict(self) -> dict[str, Any]:
        """The delta as it appears in `base_rates` and the manifest."""
        return {
            "carrier_error_ids": list(self.carrier_error_ids),
            "dropped_seed_issues": list(self.dropped_seed_issues),
            "dropped_seed_occurrences": self.dropped_seed_occurrences,
            "phantom_occurrences": len(self.phantom_occurrences),
            "phantom_trace_ids": self.phantom_trace_ids,
        }

    def warnings(self) -> list[str]:
        out: list[str] = []
        if self.phantom_occurrences:
            out.append(
                f"the Engine predicted {len(self.phantom_occurrences)} occurrence(s) against "
                f"{len(self.phantom_trace_ids)} trace id(s) that are not in the dataset "
                f"({self.phantom_trace_ids}); they were dropped before scoring rather than "
                f"counted as false positives"
            )
        return out


def prepare_scored_board(
    predicted: Issueboard, seed: Issueboard, trace_ids: list[str]
) -> ScoredBoard:
    """Turn the Engine's updated board into the Engine's predictions.

    Phantoms are removed first: an occurrence against a non-existent trace is
    invalid regardless of which issue carries it, and removing it can be what
    turns a would-be carrier back into an untouched seed issue.
    """
    universe = set(trace_ids)
    phantoms = [
        (o.error_id, o.trace_id) for o in predicted.occurrences if o.trace_id not in universe
    ]
    real = [o for o in predicted.occurrences if o.trace_id in universe]

    seed_issue_ids = {i.error_id for i in seed.issues}
    seed_pairs = {(o.error_id, o.trace_id) for o in seed.occurrences}

    delta_occurrences = [o for o in real if (o.error_id, o.trace_id) not in seed_pairs]
    dropped_seed_occurrences = len(real) - len(delta_occurrences)
    contributed = {o.error_id for o in delta_occurrences}

    kept_issues = []
    dropped_seed_issues = []
    carriers = []
    for item in predicted.issues:
        if item.error_id not in seed_issue_ids:
            kept_issues.append(item)
        elif item.error_id in contributed:
            kept_issues.append(item)
            carriers.append(item.error_id)
        else:
            dropped_seed_issues.append(item.error_id)

    board = stamp_dataset_id(
        Issueboard(
            source="engine_predicted", issues=kept_issues, occurrences=delta_occurrences
        )
    )
    return ScoredBoard(
        board=board,
        carrier_error_ids=carriers,
        dropped_seed_issues=dropped_seed_issues,
        dropped_seed_occurrences=dropped_seed_occurrences,
        phantom_occurrences=phantoms,
    )


def _macro_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_engine_delta(
    *,
    ground_truth: Issueboard,
    predicted: Issueboard,
    seed: Issueboard,
    trace_ids: list[str],
    cfg: ScoringConfig,
    base_rates: dict[str, Any],
    engine_config: EngineConfig,
    judge: DescriptionJudge | None = None,
) -> tuple[BenchmarkReport, ScoredBoard]:
    """Score what the Engine contributed. The single scoring entrypoint for a run.

    Both `run_pipeline` and the standalone `rescore_from_disk` go through here,
    so a report can always be reproduced from the artifacts that describe it —
    including the carrier adjustment, which would otherwise be an invisible
    step that only the original process knew about.
    """
    prepared = prepare_scored_board(predicted, seed, trace_ids)
    rates = {**base_rates, "engine_delta": prepared.as_dict()}

    report = score(
        ground_truth,
        prepared.board,
        cfg,
        rates,
        trace_ids=trace_ids,
        engine_config=engine_config,
        judge=judge,
    )
    if not prepared.carrier_error_ids:
        return report, prepared

    # Carriers keep their occurrences (scorers 1 and 2 are occurrence-driven)
    # but drop out of everything that reads their issue-level text: severity,
    # description, and the E_h pile. Their text is the benchmark's, not the
    # Engine's, and grading it either way is grading ourselves.
    carriers = set(prepared.carrier_error_ids)
    scored_matches = [m for m in report.matches if m.predicted_error_id not in carriers]
    severity_loss = score_severity(
        scored_matches, ground_truth, prepared.board, alpha=cfg.severity_alpha
    )
    description_scores = score_descriptions(
        scored_matches, ground_truth, prepared.board, cfg, judge=judge
    )
    adjusted = report.model_copy(
        update={
            "severity_loss": severity_loss,
            "description_scores": description_scores,
            "eh_candidates": [
                m.predicted_error_id for m in scored_matches if m.matched_error_id is None
            ],
            "headline": {
                **report.headline,
                "mean_severity_loss": severity_loss,
                "mean_description_score": _macro_mean(list(description_scores.values())),
            },
        }
    )
    return stamp_dataset_id(adjusted), prepared
