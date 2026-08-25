"""`uv run python -m benchmark.pipeline ...`.

The `run` subcommand is exercised with `run_pipeline` patched out — driving it
for real would need two live servers and a model. What IS tested for real is
everything a wrong invocation gets wrong: config overrides, the ablation-stage
choice, and the exit codes a CI job would branch on.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from benchmark.pipeline import __main__ as cli
from benchmark.pipeline.contracts import AblationStageUnavailable
from tests.pipeline.conftest import MINI_PIPELINE_CONFIG, REPO_ROOT


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


def test_each_fake_flag_substitutes_its_own_seam(spy):
    cli.main(["run", "--config", MINI, "--fake-harness", "--fake-ablation", "--fake-engine"])
    call = spy[0]
    assert isinstance(call["harness_factory"], cli.FakeHarnessFactory)
    assert isinstance(call["engine_invoker"], cli.FakeEngineInvoker)
    assert isinstance(call["expander"], cli.FakeExpander)
    assert call["ablation_stage"] is cli.fake_run_ablation


def test_fake_engine_alone_leaves_the_other_two_seams_real(spy):
    """--fake-engine is independent: the harness and the ablation stay real."""
    cli.main(["run", "--config", MINI, "--fake-engine"])
    call = spy[0]
    assert isinstance(call["engine_invoker"], cli.FakeEngineInvoker)
    assert call["harness_factory"] is None
    assert call["expander"] is None
    assert call["ablation_stage"] is not cli.fake_run_ablation


def test_a_fake_harness_without_a_fake_ablation_is_refused(capsys):
    """The one combination that cannot work, refused before anything is paid for.

    `--fake-harness` alone leaves the REAL `benchmark.ablation.run_ablation`
    driving `FakeHarness`, which implements `run_batch` and nothing else — no
    `replay`, `run_with_faults`, `turn_boundaries`, `activation_evidence` or
    `locate_checkpoint`. The run would die on a raw `AttributeError` in the
    ablation stage, minutes in, with generation and the whole batch already
    spent.
    """
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--config", MINI, "--fake-harness"])
    assert exit_info.value.code != 0

    message = capsys.readouterr().err
    assert "--fake-harness requires --fake-ablation" in message
    # It has to say WHY, and name the way out — a bare "invalid combination"
    # sends the reader to the source to find out which flag to add.
    assert "run_with_faults" in message
    assert "--fake-ablation" in message


def test_a_fake_harness_without_a_fake_ablation_never_reaches_the_runner(spy):
    with pytest.raises(SystemExit):
        cli.main(["run", "--config", MINI, "--fake-harness", "--fake-engine"])
    assert spy == [], "the pipeline started despite an unusable flag combination"


def test_the_fake_harness_run_starts_no_target_app(spy):
    """Nothing in an offline run talks to the target app, so none is managed."""
    cli.main(["run", "--config", MINI, "--fake-harness", "--fake-ablation", "--fake-engine"])
    assert spy[0]["servers"].describe() == {}


def test_the_fake_expander_never_writes_to_the_shared_expansion_cache(spy):
    """The cache key does not name the expander, so the fake must not share it.

    `(config_hash, kind, dim_id, variation, persona_id, seed)` is the whole
    key. An offline run pointed at the shared cache would write
    "[topic/x] please help me with x" under exactly the keys the real expander
    uses, and the next real run — the timed one — would build its entire input
    corpus out of them without a word.
    """
    cli.main(["run", "--config", MINI, "--fake-harness", "--fake-ablation", "--fake-engine"])
    cfg = spy[0]["cfg"]
    assert cfg.resolve(cfg.expansion_cache) == cfg.run_dir / "fake_expansion_cache"


def test_a_real_run_keeps_the_shared_expansion_cache(spy):
    cli.main(["run", "--config", MINI, "--fake-ablation", "--fake-engine"])
    cfg = spy[0]["cfg"]
    assert cfg.resolve(cfg.expansion_cache) != cfg.run_dir / "fake_expansion_cache"


def test_the_default_run_fakes_nothing(spy):
    """Every seam is real unless a flag says otherwise."""
    cli.main(["run", "--config", MINI])
    call = spy[0]
    assert call["harness_factory"] is None
    assert call["engine_invoker"] is None
    assert call["expander"] is None


# ------------------------------------------------- the offline end-to-end run

def test_the_fake_flags_run_the_whole_pipeline_offline(tmp_path):
    """The exact command a developer types, with no servers, keys or network.

    A subprocess rather than `cli.main(...)`: the claim under test is about the
    process a terminal starts, and the environment it inherits is half of it.
    Both API keys are blanked — present-but-empty, so the repo-root `.env` that
    `load_dotenv` reads cannot put them back — which means any seam that had
    NOT been faked would fail loudly instead of quietly reaching the network.
    """
    env = {
        **os.environ,
        "OPENAI_API_KEY": "",
        "LANGSMITH_API_KEY": "",
        "LANGSMITH_TRACING": "false",
    }
    proc = subprocess.run(
        [
            sys.executable, "-m", "benchmark.pipeline", "run",
            "--config", MINI,
            "--fake-harness", "--fake-ablation", "--fake-engine",
            "--artifacts-root", str(tmp_path),
            "--run-id", "offline",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"

    run_dir = tmp_path / "offline"
    for name in ("inputs.json", "raw_traces.json", "ablated_traces.json", "traces.json",
                 "predicted_issueboard.json", "report.json", "report.md", "manifest.json"):
        assert (run_dir / name).exists(), f"{name} missing from {run_dir}"

    manifest = json.loads((run_dir / "manifest.json").read_text())
    faked = [w for w in manifest["warnings"] if "FAKED" in w]
    assert faked, f"no FAKED warning in {manifest['warnings']}"
    # All three, named — a run that faked the harness and said only "ablation"
    # would read as three-quarters real.
    for stage in ("ablation", "engine_invoker", "harness"):
        assert stage in faked[0], f"{stage} is faked but the warning does not say so"
    assert "FAKED" in (run_dir / "report.md").read_text()
    assert "FAKE" in proc.stderr


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
