"""Report assembly — Stage V output (docs/architecture/06-scoring.md).

Wires the two-layer matcher and all four scorers into one BenchmarkReport.
Headline numbers are reported as separate keys — detection, localization,
severity calibration, and explanation quality fail independently, so no
single composite scalar collapses them.
"""

from __future__ import annotations

from benchmark.schemas import BenchmarkReport, EngineConfig, Issueboard, ScoringConfig
from benchmark.schemas.io import stamp_dataset_id
from benchmark.scoring.matcher import compute_fallback_rate, pair_issues, resolve_occurrences
from benchmark.scoring.scorer_categories import score_categories
from benchmark.scoring.scorer_description import DescriptionJudge, score_descriptions
from benchmark.scoring.scorer_per_error import score_per_error
from benchmark.scoring.scorer_severity import score_severity


def _all_trace_ids(gt: Issueboard, pred: Issueboard) -> list[str]:
    ids = {occ.trace_id for occ in gt.occurrences} | {occ.trace_id for occ in pred.occurrences}
    return sorted(ids)


def _macro_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score(
    ground_truth: Issueboard,
    predicted: Issueboard,
    cfg: ScoringConfig,
    base_rates: dict,
    engine_config: EngineConfig | None = None,
    judge: DescriptionJudge | None = None,
) -> BenchmarkReport:
    """Score `predicted` against `ground_truth` and assemble a BenchmarkReport.

    `engine_config` defaults to a placeholder identity when the caller doesn't
    have Engine-run provenance handy (e.g. scoring in isolation); real
    pipeline callers should always pass the actual run's EngineConfig.
    """
    engine_config = engine_config or EngineConfig(model="unknown")

    occ_matches = resolve_occurrences(ground_truth, predicted, cfg)
    matches = pair_issues(occ_matches, ground_truth, predicted)
    trace_ids = _all_trace_ids(ground_truth, predicted)

    category_scores = score_categories(ground_truth, predicted, trace_ids)
    per_error_scores = score_per_error(ground_truth, predicted, occ_matches, trace_ids)
    severity_loss = score_severity(matches, ground_truth, predicted, alpha=cfg.severity_alpha)
    description_scores = score_descriptions(matches, ground_truth, predicted, cfg, judge=judge)

    eh_candidates = [m.predicted_error_id for m in matches if m.matched_error_id is None]

    headline = {
        "category_precision_macro": _macro_mean([s.precision for s in category_scores]),
        "category_recall_macro": _macro_mean([s.recall for s in category_scores]),
        "category_f1_macro": _macro_mean([s.f1 for s in category_scores]),
        "matched_error_f1_macro": _macro_mean([s.f1 for s in per_error_scores]),
        "mean_severity_loss": severity_loss,
        "mean_description_score": _macro_mean(list(description_scores.values())),
    }

    report = BenchmarkReport(
        report_id="",
        engine_config=engine_config,
        base_rates=dict(base_rates),
        matcher_fallback_rate=compute_fallback_rate(occ_matches),
        occurrence_matches=occ_matches,
        matches=matches,
        category_scores=category_scores,
        per_error_scores=per_error_scores,
        severity_loss=severity_loss,
        description_scores=description_scores,
        eh_candidates=eh_candidates,
        headline=headline,
    )
    return stamp_dataset_id(report)
