"""`BenchmarkReport` -> a summary a person can argue with.

The JSON report is the machine artifact. This is the one that gets read, so it
carries the context that makes a number mean something:

* **base rates** — an F1 of 0.6 against a 3% injection rate and against a 40%
  one are not the same result;
* **matcher fallback rate** — how often the exact `(trace_id, category_id)` key
  missed and text similarity had to stand in. High fallback means the numbers
  below it are softer than they look;
* **severity confusion** — `severity_loss` is a scalar, and a scalar cannot
  tell you whether the Engine is systematically under- or over-calling;
* **the E_h appendix** — unmatched predictions are the pool where genuine
  hidden errors hide, and reading them is how the taxonomy grows;
* **runtime and warnings** — including, loudly, whether any stage was faked.
"""

from __future__ import annotations

from collections import Counter

from benchmark.pipeline.manifest import RunManifest
from benchmark.schemas import BenchmarkReport, ErrorMatch, Issueboard

SEVERITIES = ("low", "medium", "high")


def severity_confusion(
    matches: list[ErrorMatch], gt: Issueboard, pred: Issueboard
) -> Counter[tuple[str, str]]:
    """(ground-truth severity, predicted severity) counts over matched pairs."""
    gt_by_id = {i.error_id: i for i in gt.issues}
    pred_by_id = {i.error_id: i for i in pred.issues}
    counts: Counter[tuple[str, str]] = Counter()
    for match in matches:
        if match.matched_error_id is None:
            continue
        gt_issue = gt_by_id.get(match.matched_error_id)
        pred_issue = pred_by_id.get(match.predicted_error_id)
        if gt_issue is None or pred_issue is None:
            continue
        counts[(gt_issue.severity, pred_issue.severity)] += 1
    return counts


def _table(header: list[str], rows: list[list[str]], *, empty: str) -> list[str]:
    if not rows:
        return [f"_{empty}_", ""]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    out.append("")
    return out


def _num(value: float) -> str:
    # Round first: a kappa of -1e-17 is zero, and printing it as "-0" invites
    # exactly the wrong question.
    if abs(value) < 5e-4:
        return "0"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def render_markdown(
    report: BenchmarkReport,
    *,
    ground_truth: Issueboard,
    predicted: Issueboard,
    manifest: RunManifest | None = None,
) -> str:
    run_id = manifest.run_id if manifest else "benchmark run"
    model = report.engine_config.model
    lines: list[str] = [f"# Benchmark report — {run_id}", ""]

    lines.append(f"**Engine model**: `{model}`  ")
    lines.append(f"**Report id**: `{report.report_id or '(unstamped)'}`  ")
    if manifest:
        if manifest.created_at:
            lines.append(f"**Run started**: {manifest.created_at.isoformat()}  ")
        for stage, implementation in sorted(manifest.stages.items()):
            lines.append(f"**{stage} stage**: `{implementation}`  ")
    lines.append("")

    warnings = list(manifest.warnings) if manifest else []
    if warnings:
        lines.append("> **Warnings**")
        lines += [f"> - {w}" for w in warnings]
        lines.append("")

    # ------------------------------------------------------------- headline
    lines.append("## Headline")
    lines.append("")
    lines.append(
        "Reported as separate numbers on purpose: detection, localization, severity "
        "calibration and explanation quality fail independently, and one composite "
        "scalar would hide which of them broke."
    )
    lines.append("")
    lines += _table(
        ["metric", "value"],
        [[f"`{k}`", _num(v)] for k, v in sorted(report.headline.items())],
        empty="no headline metrics — the report is empty",
    )

    # ---------------------------------------------------------- reliability
    lines.append("## Matcher reliability")
    lines.append("")
    lines.append(
        f"Text-similarity **fallback rate**: **{report.matcher_fallback_rate:.2f}** — the "
        f"fraction of predicted occurrences whose exact `(trace_id, category_id)` key "
        f"missed and had to be resolved by TF-IDF against a different-category injection "
        f"on the same trace. The closer this is to zero, the more the numbers below are "
        f"exact bookkeeping rather than text matching."
    )
    lines.append("")
    method_counts = Counter(m.method for m in report.occurrence_matches)
    lines += _table(
        ["resolution", "occurrences"],
        [[f"`{k}`", str(v)] for k, v in sorted(method_counts.items())]
        + [
            [
                "`unresolved` (FP / E_h pool)",
                str(sum(1 for m in report.occurrence_matches if m.resolved_error_id is None)),
            ]
        ],
        empty="no predicted occurrences",
    )

    # ------------------------------------------------------------ base rates
    lines.append("## Base rates")
    lines.append("")
    lines.append(
        "What the scores are relative to. Precision and recall at a 3% injection rate "
        "are not comparable with the same numbers at 40%, and Cohen's kappa in the "
        "tables below exists precisely because prevalence is low."
    )
    lines.append("")
    lines += _table(
        ["quantity", "value"],
        [[f"`{k}`", f"`{v}`"] for k, v in sorted(report.base_rates.items())],
        empty="no base rates recorded — the report cannot be interpreted without them",
    )

    # --------------------------------------------------------- scorer 1 + 2
    lines.append("## Scorer 1 — category detection")
    lines.append("")
    lines += _table(
        ["category", "precision", "recall", "F1", "Cohen's kappa", "support"],
        [
            [
                f"{s.category_id}",
                _num(s.precision),
                _num(s.recall),
                _num(s.f1),
                _num(s.cohens_kappa),
                str(s.support),
            ]
            for s in report.category_scores
        ],
        empty="no categories appeared in either board",
    )

    lines.append("## Scorer 2 — per-error localization")
    lines.append("")
    gt_by_id = {i.error_id: i for i in ground_truth.issues}
    lines += _table(
        ["known error", "title", "precision", "recall", "F1", "Cohen's kappa", "support"],
        [
            [
                f"`{s.category_id}`",
                gt_by_id[s.category_id].title if s.category_id in gt_by_id else "—",
                _num(s.precision),
                _num(s.recall),
                _num(s.f1),
                _num(s.cohens_kappa),
                str(s.support),
            ]
            for s in report.per_error_scores
        ],
        empty="no known errors were matched",
    )

    # ------------------------------------------------------------- severity
    lines.append("## Scorer 3 — severity calibration")
    lines.append("")
    lines.append(
        f"Mean asymmetric severity loss: **{report.severity_loss:.3f}** "
        f"(under-calling is penalised quadratically, over-calling linearly — missing a "
        f"high-severity error costs more than crying wolf)."
    )
    lines.append("")
    lines.append("### Severity confusion (matched pairs)")
    lines.append("")
    confusion = severity_confusion(report.matches, ground_truth, predicted)
    if confusion:
        header = ["ground truth \\ predicted", *SEVERITIES]
        rows = [
            [f"**{gt_sev}**", *[str(confusion.get((gt_sev, p), 0)) for p in SEVERITIES]]
            for gt_sev in SEVERITIES
            if any(confusion.get((gt_sev, p), 0) for p in SEVERITIES)
        ]
        lines += _table(header, rows, empty="no matched pairs")
        off_diagonal = sum(v for (g, p), v in confusion.items() if g != p)
        lines.append(
            f"{off_diagonal} of {sum(confusion.values())} matched pairs disagree on severity."
        )
        lines.append("")
    else:
        lines.append("_no matched pairs to compare severities on_")
        lines.append("")

    # ---------------------------------------------------------- description
    lines.append("## Scorer 4 — description deviation")
    lines.append("")
    lines += _table(
        ["known error", "score"],
        [[f"`{k}`", _num(v)] for k, v in sorted(report.description_scores.items())],
        empty="no matched pairs to compare descriptions on",
    )

    # ------------------------------------------------------- runtime + counts
    lines.append("## Runtime")
    lines.append("")
    if manifest:
        lines += _table(
            ["stage", "seconds"],
            [[t.stage, f"{t.seconds:.1f}"] for t in manifest.timings]
            + [["**total**", f"{manifest.total_seconds:.1f}"]],
            empty="no timings recorded",
        )
        lines += _table(
            ["count", "value"],
            [[f"`{k}`", str(v)] for k, v in sorted(manifest.counts.items())],
            empty="no counts recorded",
        )
    else:
        lines.append("_no manifest supplied_")
        lines.append("")

    # ------------------------------------------------------ E_h appendix
    lines.append("## Appendix — E_h candidates")
    lines.append("")
    lines.append(
        "Predicted issues that resolved to no injected error. Some are false "
        "positives; some are real problems the ablation engine never planted "
        "(`E_h`). They are the read-me pile: a candidate confirmed by review is a "
        "new category or a new ablation, and until then every one of them is "
        "counted against precision."
    )
    lines.append("")
    pred_by_id = {i.error_id: i for i in predicted.issues}
    occurrences_by_error: Counter[str] = Counter(o.error_id for o in predicted.occurrences)
    lines += _table(
        ["predicted id", "title", "category", "severity", "occurrences", "description"],
        [
            [
                f"`{error_id}`",
                pred_by_id[error_id].title if error_id in pred_by_id else "—",
                pred_by_id[error_id].category_id if error_id in pred_by_id else "—",
                pred_by_id[error_id].severity if error_id in pred_by_id else "—",
                str(occurrences_by_error.get(error_id, 0)),
                (pred_by_id[error_id].description[:160] if error_id in pred_by_id else "—"),
            ]
            for error_id in report.eh_candidates
        ],
        empty="no unmatched predictions",
    )

    return "\n".join(lines) + "\n"
