"""`uv run python -m benchmark.pipeline ...`.

The `run` subcommand is exercised with `run_pipeline` patched out — driving it
for real would need two live servers and a model. What IS tested for real is
everything a wrong invocation gets wrong: config overrides, the ablation-stage
choice, and the exit codes a CI job would branch on.
"""

from __future__ import annotations

import json

import pytest

from benchmark.pipeline import __main__ as cli
from benchmark.pipeline.contracts import AblationStageUnavailable
from tests.pipeline.conftest import MINI_PIPELINE_CONFIG


@pytest.fixture
def spy(monkeypatch):
    calls: list[dict] = []

    class FakeRun:
        markdown = "# report\n"
        deliverables: list = []

        def __init__(self, run_dir):
            self.run_dir = run_dir

    def fake_run_pipeline(cfg, **kwargs):
        calls.append({"cfg": cfg, **kwargs})
        return FakeRun(cfg.run_dir)

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    return calls


MINI = str(MINI_PIPELINE_CONFIG)


# ------------------------------------------------------------------- run

def test_run_needs_a_config():
    with pytest.raises(SystemExit):
        cli.main(["run"])


def test_run_without_phase_5_is_blocked_not_downgraded(spy, monkeypatch):
    """A missing ablation engine must never quietly become a fake one."""

    def unavailable():
        raise AblationStageUnavailable("benchmark.ablation is not available")

    monkeypatch.setattr(cli, "load_ablation_stage", unavailable)
    assert cli.main(["run", "--config", MINI]) == 3
    assert spy == [], "the pipeline ran without an ablation stage"


def test_run_with_fake_ablation_uses_the_stand_in(spy):
    assert cli.main(["run", "--config", MINI, "--fake-ablation"]) == 0
    assert spy[0]["ablation_stage"] is cli.fake_run_ablation


def test_run_manages_both_servers_by_default(spy):
    cli.main(["run", "--config", MINI, "--fake-ablation"])
    assert sorted(spy[0]["servers"].describe()) == ["engine", "target_app"]


def test_no_serve_disables_server_management(spy):
    cli.main(["run", "--config", MINI, "--fake-ablation", "--no-serve"])
    assert spy[0]["servers"].describe() == {}


def test_the_engine_model_can_be_overridden_for_the_comparison_axis(spy):
    cli.main(["run", "--config", MINI, "--fake-ablation", "--engine-model", "gpt-5.1"])
    assert spy[0]["cfg"].engine.model == "gpt-5.1"
    # Everything else about the run must be untouched — that is what makes the
    # two arms of a model comparison comparable.
    assert spy[0]["cfg"].engine.analysis_concurrency == 8


def test_the_run_id_and_artifacts_root_can_be_overridden(spy, tmp_path):
    cli.main(
        [
            "run",
            "--config",
            MINI,
            "--fake-ablation",
            "--run-id",
            "arm-a",
            "--artifacts-root",
            str(tmp_path),
        ]
    )
    assert spy[0]["cfg"].run_dir == tmp_path / "arm-a"


def test_overrides_keep_the_repo_root(spy):
    cli.main(["run", "--config", MINI, "--fake-ablation", "--run-id", "x"])
    assert (spy[0]["cfg"].root / "pyproject.toml").exists()


def test_a_failing_deliverable_makes_the_run_exit_nonzero(spy, monkeypatch):
    from benchmark.pipeline.deliverables import DeliverableCheck

    class FailingRun:
        markdown = "# r\n"
        deliverables = [DeliverableCheck(name="traces_file_scale", ok=False, detail="too few")]

        def __init__(self, run_dir):
            self.run_dir = run_dir

    def failing(cfg, **kwargs):
        spy.append({"cfg": cfg, **kwargs})
        return FailingRun(cfg.run_dir)

    monkeypatch.setattr(cli, "run_pipeline", failing)
    assert cli.main(["run", "--config", MINI, "--fake-ablation"]) == 1


# ----------------------------------------------------------- score / check

def test_score_reprints_the_headline_from_artifacts(mini_run, capsys):
    assert cli.main(["score", "--run", str(mini_run.run_dir)]) == 0
    assert json.loads(capsys.readouterr().out) == mini_run.report.headline


def test_check_passes_at_the_miniature_size(mini_run, capsys):
    n = len(mini_run.ablated.traces)
    assert cli.main(["check", "--run", str(mini_run.run_dir), "--min-traces", str(n)]) == 0
    assert "[ok] traces_file_scale" in capsys.readouterr().out


def test_check_fails_at_assignment_scale_on_a_miniature_run(mini_run, capsys):
    assert cli.main(["check", "--run", str(mini_run.run_dir), "--min-traces", "300"]) == 1
    assert "[FAIL] traces_file_scale" in capsys.readouterr().out


def test_an_unknown_command_is_refused():
    with pytest.raises(SystemExit):
        cli.main(["frobnicate"])
