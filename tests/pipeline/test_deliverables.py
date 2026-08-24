"""The assignment-deliverables checklist, run against a completed miniature run.

The `min_traces` gate is parametrized on purpose: the miniature exercises the
same check at N=6 that the full-scale run exercises at N>=300, so the >=300
requirement is not a number that only ever gets tested by hand once.
"""

from __future__ import annotations

import json

import pytest

from benchmark.pipeline.deliverables import check_deliverables, rescore_from_disk
from benchmark.schemas import BenchmarkReport, Issueboard


def by_name(checks):
    return {c.name: c for c in checks}


# --------------------------------------------------------- the run's own checks

def test_the_run_writes_its_deliverables_report(mini_run):
    payload = json.loads((mini_run.run_dir / "deliverables.json").read_text())
    assert {c["name"] for c in payload} == {c.name for c in mini_run.deliverables}


def test_every_deliverable_passes_on_the_miniature_run(mini_run):
    failed = [f"{c.name}: {c.detail}" for c in mini_run.deliverables if not c.ok]
    assert not failed, "failed deliverables:\n" + "\n".join(failed)


EXPECTED_CHECKS = [
    "traces_file_scale",
    "traces_file_schema_and_leak_free",
    "issueboard_in",
    "issueboard_out_updated",
    "standalone_scoring_entrypoint",
    "dataset_lineage",
]


@pytest.mark.parametrize("name", EXPECTED_CHECKS)
def test_each_assignment_deliverable_is_checked(mini_run, name):
    assert name in by_name(mini_run.deliverables)


# ----------------------------------------------------------- the scale gate

def test_the_scale_gate_passes_at_the_miniature_size(mini_run):
    checks = by_name(check_deliverables(mini_run.run_dir, min_traces=len(mini_run.ablated.traces)))
    assert checks["traces_file_scale"].ok


def test_the_scale_gate_bites_at_assignment_scale(mini_run):
    """Same code, the assignment's own number: a 7-trace run must not pass it."""
    checks = by_name(check_deliverables(mini_run.run_dir, min_traces=300))
    assert not checks["traces_file_scale"].ok
    assert "300" in checks["traces_file_scale"].detail


@pytest.mark.parametrize("min_traces", [1, 3, 6])
def test_the_scale_gate_is_the_same_check_at_every_size(mini_run, min_traces):
    checks = by_name(check_deliverables(mini_run.run_dir, min_traces=min_traces))
    assert checks["traces_file_scale"].ok


# ----------------------------------------------- standalone scoring entrypoint

def test_score_can_be_re_run_from_the_artifacts_alone(mini_run):
    rescored = rescore_from_disk(mini_run.run_dir)
    assert rescored.headline == mini_run.report.headline
    assert rescored.report_id == mini_run.report.report_id


def test_a_judge_scored_run_excludes_the_judge_headline_out_loud(mini_run):
    """An LLM judge cannot be reproduced offline; the check says so rather than
    silently re-scoring under a different rubric."""
    path = mini_run.run_dir / "pipeline_config.json"
    raw = json.loads(path.read_text())
    raw["scoring"]["description_mode"] = "judge"
    path.write_text(json.dumps(raw))
    check = by_name(check_deliverables(mini_run.run_dir, min_traces=1))[
        "standalone_scoring_entrypoint"
    ]
    assert check.ok
    assert "mean_description_score excluded" in check.detail


def test_the_scoring_check_notices_a_tampered_report(mini_run):
    path = mini_run.run_dir / "report.json"
    report = BenchmarkReport.model_validate_json(path.read_text())
    report.headline["category_f1_macro"] = 0.999
    path.write_text(report.model_dump_json(indent=2))
    checks = by_name(check_deliverables(mini_run.run_dir, min_traces=1))
    assert not checks["standalone_scoring_entrypoint"].ok


# ------------------------------------------------------ the two issueboards

def test_a_board_that_replaces_the_seed_rather_than_updating_it_fails(mini_run):
    seed = Issueboard(
        source="seed",
        issues=[
            {
                "error_id": "S-never-returned",
                "title": "t",
                "description": "d",
                "category_id": "other",
                "severity": "low",
            }
        ],
    )
    (mini_run.run_dir / "seed_issueboard.json").write_text(seed.model_dump_json())
    checks = by_name(check_deliverables(mini_run.run_dir, min_traces=1))
    assert not checks["issueboard_out_updated"].ok
    assert "S-never-returned" in checks["issueboard_out_updated"].detail


def test_a_leaked_export_fails_the_trace_file_check(mini_run):
    path = mini_run.run_dir / "traces.json"
    payload = json.loads(path.read_text())
    payload["traces"][0]["ablation_ids"] = ["abl-7"]
    path.write_text(json.dumps(payload))
    checks = by_name(check_deliverables(mini_run.run_dir, min_traces=1))
    assert not checks["traces_file_schema_and_leak_free"].ok


def test_broken_lineage_fails_the_lineage_check(mini_run):
    path = mini_run.run_dir / "ablated_traces.json"
    payload = json.loads(path.read_text())
    payload["parent_dataset_id"] = "not-the-traces-it-came-from"
    path.write_text(json.dumps(payload))
    checks = by_name(check_deliverables(mini_run.run_dir, min_traces=1))
    assert not checks["dataset_lineage"].ok


def test_a_missing_artifact_is_reported_not_raised(tmp_path):
    checks = check_deliverables(tmp_path, min_traces=1)
    assert all(not c.ok for c in checks)
    assert all(c.detail for c in checks)
