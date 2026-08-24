"""The human-readable summary.

`report.json` is the machine artifact; `report.md` is the one a person reads
before deciding whether to believe it. So the markdown has to carry the things
that make a number interpretable — base rates, the matcher's fallback rate, the
E_h appendix, runtime — not just the headline.
"""

from __future__ import annotations

import pytest

from benchmark.pipeline.manifest import RunManifest, StageTiming
from benchmark.pipeline.render import render_markdown, severity_confusion
from benchmark.schemas import (
    BenchmarkReport,
    CategoryScore,
    EngineConfig,
    ErrorMatch,
    Issue,
    Issueboard,
    IssueOccurrence,
    OccurrenceMatch,
)


def issue(error_id, category, severity, title="t", description="d"):
    return Issue(
        error_id=error_id,
        title=title,
        description=description,
        category_id=category,
        severity=severity,
    )


@pytest.fixture
def ground_truth():
    return Issueboard(
        source="ground_truth",
        issues=[
            issue("K1", "hallucination", "high", title="planted hallucination"),
            issue("K2", "tool_misuse", "low", title="planted tool misuse"),
        ],
        occurrences=[
            IssueOccurrence(error_id="K1", trace_id="t1"),
            IssueOccurrence(error_id="K2", trace_id="t2"),
        ],
    )


@pytest.fixture
def predicted():
    return Issueboard(
        source="engine_predicted",
        issues=[
            issue("P1", "hallucination", "medium", title="invented refund policy"),
            issue("P2", "tool_misuse", "low", title="ignored a tool error"),
            issue("P9", "other", "high", title="a finding nothing explains"),
        ],
        occurrences=[
            IssueOccurrence(error_id="P1", trace_id="t1"),
            IssueOccurrence(error_id="P2", trace_id="t2"),
            IssueOccurrence(error_id="P9", trace_id="t3"),
        ],
    )


@pytest.fixture
def report():
    return BenchmarkReport(
        report_id="rep1",
        engine_config=EngineConfig(model="gpt-5.1-mini"),
        base_rates={
            "control_fraction": 0.3,
            "per_error_injection_counts": {"K1": 1, "K2": 1},
            "n_traces": 3,
        },
        matcher_fallback_rate=0.25,
        occurrence_matches=[
            OccurrenceMatch(trace_id="t1", predicted_error_id="P1", resolved_error_id="K1"),
            OccurrenceMatch(
                trace_id="t2",
                predicted_error_id="P2",
                resolved_error_id="K2",
                method="text_fallback",
            ),
            OccurrenceMatch(trace_id="t3", predicted_error_id="P9"),
        ],
        matches=[
            ErrorMatch(predicted_error_id="P1", matched_error_id="K1", overlap=1),
            ErrorMatch(predicted_error_id="P2", matched_error_id="K2", overlap=1),
            ErrorMatch(predicted_error_id="P9"),
        ],
        category_scores=[
            CategoryScore(
                category_id="hallucination",
                precision=1.0,
                recall=1.0,
                f1=1.0,
                cohens_kappa=1.0,
                support=1,
            )
        ],
        per_error_scores=[
            CategoryScore(
                category_id="K1", precision=1.0, recall=1.0, f1=1.0, cohens_kappa=1.0, support=1
            )
        ],
        severity_loss=0.375,
        description_scores={"K1": 0.42},
        eh_candidates=["P9"],
        headline={"category_f1_macro": 1.0, "mean_severity_loss": 0.375},
    )


@pytest.fixture
def manifest():
    return RunManifest(
        run_id="mini",
        models={"engine": "gpt-5.1-mini"},
        counts={"traces": 3, "engine_issues": 3, "engine_occurrences": 3},
        timings=[StageTiming(stage="generation", seconds=2.5)],
        stages={"ablation": "benchmark.pipeline.fakes.fake_run_ablation"},
        warnings=["the ablation stage was faked"],
    )


def test_the_severity_table_agrees_with_the_severity_loss_above_it(
    ground_truth, predicted, manifest
):
    """The reproduced contradiction: `severity_loss` excludes carriers (their
    severity is seed-authored) while the confusion table counted them, so the
    report said "loss 0.000" directly above "1 of 1 pairs disagree"."""
    carrier_report = BenchmarkReport(
        report_id="rep-carrier",
        engine_config=EngineConfig(model="m"),
        base_rates={"engine_delta": {"carrier_error_ids": ["P1"]}},
        matches=[
            # P1 is a seed carrier: high-vs-medium, but not the Engine's call.
            ErrorMatch(predicted_error_id="P1", matched_error_id="K1", overlap=1),
            ErrorMatch(predicted_error_id="P2", matched_error_id="K2", overlap=1),
        ],
        severity_loss=0.0,
        headline={"mean_severity_loss": 0.0},
    )
    md = render_markdown(
        carrier_report,
        ground_truth=ground_truth,
        scored_board=predicted,
        manifest=manifest,
    )
    scored_pairs = [m for m in carrier_report.matches if m.predicted_error_id != "P1"]
    assert f"of {len(scored_pairs)} matched pairs disagree" in md
    assert "0 of 1 matched pairs disagree" in md, (
        "the table still counted the carrier the loss excluded"
    )


def test_severity_confusion_can_exclude_carriers(ground_truth, predicted):
    matches = [
        ErrorMatch(predicted_error_id="P1", matched_error_id="K1"),
        ErrorMatch(predicted_error_id="P2", matched_error_id="K2"),
    ]
    both = severity_confusion(matches, ground_truth, predicted)
    without = severity_confusion(matches, ground_truth, predicted, exclude={"P1"})
    assert sum(both.values()) == 2
    assert sum(without.values()) == 1


def test_the_eh_appendix_counts_scored_occurrences(report, ground_truth, manifest):
    """Post-intersection counts: a phantom occurrence is not evidence about the
    prediction, and the appendix is a reading list, not a tally of what was
    returned."""
    scored_board = Issueboard(
        source="engine_predicted",
        issues=[issue("P9", "other", "high", title="a finding nothing explains")],
        # The verbatim board had this on t3 plus a phantom; only t3 survives.
        occurrences=[IssueOccurrence(error_id="P9", trace_id="t3")],
    )
    md = render_markdown(
        report, ground_truth=ground_truth, scored_board=scored_board, manifest=manifest
    )
    appendix = md.split("## Appendix")[1]
    assert "| `P9` |" in appendix
    row = next(line for line in appendix.splitlines() if "| `P9` |" in line)
    assert "| 1 |" in row, f"expected the post-intersection count of 1: {row}"


def test_severity_confusion_counts_matched_pairs(ground_truth, predicted, report):
    confusion = severity_confusion(report.matches, ground_truth, predicted)
    assert confusion[("high", "medium")] == 1
    assert confusion[("low", "low")] == 1


def test_severity_confusion_ignores_unmatched_predictions(ground_truth, predicted, report):
    assert sum(severity_confusion(report.matches, ground_truth, predicted).values()) == 2


def test_the_summary_leads_with_the_headline(report, ground_truth, predicted, manifest):
    md = render_markdown(
        report, ground_truth=ground_truth, scored_board=predicted, manifest=manifest
    )
    assert md.startswith("# ")
    assert "category_f1_macro" in md
    assert "gpt-5.1-mini" in md


def test_the_summary_carries_the_base_rates(report, ground_truth, predicted, manifest):
    md = render_markdown(
        report, ground_truth=ground_truth, scored_board=predicted, manifest=manifest
    )
    assert "control_fraction" in md
    assert "per_error_injection_counts" in md


def test_the_summary_reports_the_matcher_fallback_rate(
    report, ground_truth, predicted, manifest
):
    """How often the exact key missed is a reliability stat about the scores."""
    md = render_markdown(
        report, ground_truth=ground_truth, scored_board=predicted, manifest=manifest
    )
    assert "fallback" in md.lower() and "0.25" in md


def test_the_summary_has_a_per_category_table(report, ground_truth, predicted, manifest):
    md = render_markdown(
        report, ground_truth=ground_truth, scored_board=predicted, manifest=manifest
    )
    assert "| hallucination |" in md
    assert "Cohen" in md or "kappa" in md.lower()


def test_the_summary_shows_severity_confusion(report, ground_truth, predicted, manifest):
    md = render_markdown(
        report, ground_truth=ground_truth, scored_board=predicted, manifest=manifest
    )
    assert "Severity" in md
    assert "0.375" in md


def test_the_eh_appendix_names_the_unmatched_predictions(
    report, ground_truth, predicted, manifest
):
    md = render_markdown(
        report, ground_truth=ground_truth, scored_board=predicted, manifest=manifest
    )
    assert "P9" in md
    assert "a finding nothing explains" in md


def test_the_summary_reports_runtime(report, ground_truth, predicted, manifest):
    md = render_markdown(
        report, ground_truth=ground_truth, scored_board=predicted, manifest=manifest
    )
    assert "generation" in md and "2.5" in md


def test_the_summary_surfaces_warnings_prominently(report, ground_truth, predicted, manifest):
    """A faked ablation stage must never be a footnote."""
    md = render_markdown(
        report, ground_truth=ground_truth, scored_board=predicted, manifest=manifest
    )
    head = md.split("## ")[0] + md.split("## ")[1]
    assert "faked" in head


def test_the_summary_renders_without_a_manifest(report, ground_truth, predicted):
    md = render_markdown(report, ground_truth=ground_truth, scored_board=predicted)
    assert "# " in md


def test_an_empty_report_still_renders():
    md = render_markdown(
        BenchmarkReport(engine_config=EngineConfig(model="m")),
        ground_truth=Issueboard(source="ground_truth"),
        scored_board=Issueboard(source="engine_predicted"),
    )
    assert "no " in md.lower()
