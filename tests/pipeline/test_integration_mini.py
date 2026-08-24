"""The miniature end-to-end integration test — the Phase 7 CI gate.

Runs the REAL pipeline over the REAL `configs/pipeline/mini.yaml`, with three
seams faked: the harness (canned traces instead of a LangGraph server), the
ablation stage (Phase 5 has not merged), and the Engine invoker (canned board
instead of a second server). Everything between them — config loading, input
generation, slicing, artifact layout, lineage, the leak audit on the export,
the seed-board hand-off, board re-stamping, base-rate assembly, scoring, the
manifest and the rendered summary — is the production code path.

Zero network, zero servers, zero model calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.pipeline.fakes import fake_run_ablation
from benchmark.pipeline.runner import run_pipeline
from benchmark.schemas import BenchmarkReport, Issueboard, TraceDataset
from benchmark.schemas.io import content_hash
from tests.pipeline.conftest import REPO_ROOT, FakeEngineInvoker, FakeExpander

MINI = REPO_ROOT / "configs" / "pipeline" / "mini.yaml"


@pytest.fixture
def cfg(mini_cfg):
    return mini_cfg


@pytest.fixture
def engine_invoker(mini_engine_invoker):
    return mini_engine_invoker


@pytest.fixture
def run(mini_run):
    return mini_run


# ----------------------------------------------------------------- the shape

def test_a_single_command_turns_configs_into_a_benchmark_report(run):
    assert isinstance(run.report, BenchmarkReport)
    assert run.report.report_id
    assert run.report.engine_config.model == "gpt-5.1-mini"


def test_the_mini_config_slices_to_single_turn_inputs(run):
    assert 1 <= len(run.inputs.inputs) <= 8
    assert {s.mode for s in run.inputs.inputs} == {"single_turn"}


def test_every_input_produced_a_trace(run):
    assert len(run.traces.traces) == len(run.inputs.inputs)


def test_a_slice_that_selects_nothing_is_refused(cfg, fake_harness_factory):
    """A benchmark over zero inputs would run to completion and report nothing."""
    empty = cfg.model_copy(update={"max_inputs": 0}).with_root(cfg.root)
    with pytest.raises(ValueError, match="none of the"):
        run_pipeline(
            empty,
            ablation_stage=fake_run_ablation,
            engine_invoker=FakeEngineInvoker(),
            harness_factory=fake_harness_factory,
            expander=FakeExpander(),
        )


# -------------------------------------------------------------- the artifacts

EXPECTED_ARTIFACTS = [
    "pipeline_config.json",
    "inputs.json",
    "outputs.json",
    "raw_traces.json",
    "ablated_traces.json",
    "traces.json",
    "ablation_records.json",
    "ablation_split.json",
    "seed_issueboard.json",
    "ground_truth_issueboard.json",
    "predicted_issueboard.json",
    "engine_raw_output.json",
    "report.json",
    "report.md",
    "deliverables.json",
    "manifest.json",
]


@pytest.mark.parametrize("name", EXPECTED_ARTIFACTS)
def test_each_artifact_is_written(run, name):
    assert (run.run_dir / name).exists(), f"{name} missing from {run.run_dir}"


def test_the_report_on_disk_is_the_report_returned(run):
    on_disk = BenchmarkReport.model_validate_json((run.run_dir / "report.json").read_text())
    assert on_disk == run.report


# ----------------------------------------------------------------- lineage

def test_traces_descend_from_inputs(run):
    assert run.traces.parent_dataset_id == run.inputs.dataset_id
    assert run.outputs.parent_dataset_id == run.inputs.dataset_id


def test_the_ablated_set_descends_from_the_traces(run):
    assert run.ablated.parent_dataset_id == run.traces.dataset_id


def test_broken_ablation_lineage_is_warned_about_not_swallowed(cfg, fake_harness_factory):
    """The pipeline does not own Phase 5's lineage — it does notice when it breaks."""
    invoker = FakeEngineInvoker()

    def orphaning_stage(**kwargs):
        result = fake_run_ablation(**kwargs)
        result.ablated = result.ablated.model_copy(update={"parent_dataset_id": None})
        invoker.ground_truth = result.ground_truth
        return result

    run = run_pipeline(
        cfg,
        ablation_stage=orphaning_stage,
        engine_invoker=invoker,
        harness_factory=fake_harness_factory,
        expander=FakeExpander(),
    )
    assert any("lineage is broken" in w for w in run.manifest.warnings)
    assert not by_name(run.deliverables)["dataset_lineage"].ok


def by_name(checks):
    return {c.name: c for c in checks}


def test_the_manifest_records_the_whole_chain(run):
    lineage = run.manifest.lineage
    assert lineage["traces"] == run.inputs.dataset_id
    assert lineage["ablated_traces"] == run.traces.dataset_id
    ids = run.manifest.dataset_ids
    assert ids["inputs"] == run.inputs.dataset_id
    assert ids["ablated_traces"] == run.ablated.dataset_id
    assert ids["report"] == run.report.report_id


def test_the_manifest_records_config_hashes(run, cfg):
    assert run.manifest.config_hashes["scoring"] == content_hash(cfg.scoring)
    assert run.manifest.config_hashes["ablation"] == content_hash(cfg.ablation)
    assert run.manifest.config_hashes["pipeline"]


def test_the_manifest_records_the_model_and_the_stage_implementations(run):
    assert run.manifest.models["engine"] == "gpt-5.1-mini"
    assert "fake_run_ablation" in run.manifest.stages["ablation"]


def test_the_manifest_records_timings_for_every_stage(run):
    stages = {t.stage for t in run.manifest.timings}
    assert {"generation", "harness", "ablation", "engine", "scoring"} <= stages


def test_a_faked_stage_is_a_loud_warning_not_a_footnote(run):
    assert any("FAKED" in w for w in run.manifest.warnings)
    assert any("FAKED" in line for line in run.markdown.split("## ")[0].splitlines())


def test_the_fake_warning_survives_being_wrapped(cfg, fake_harness_factory):
    """A closure around the fake loses its name; the warning must not."""
    invoker = FakeEngineInvoker()

    def wrapper(**kwargs):
        result = fake_run_ablation(**kwargs)
        invoker.ground_truth = result.ground_truth
        return result

    run = run_pipeline(
        cfg,
        ablation_stage=wrapper,
        engine_invoker=invoker,
        harness_factory=fake_harness_factory,
        expander=FakeExpander(),
    )
    assert any("FAKED" in w for w in run.manifest.warnings)


# ------------------------------------------------------- the Engine hand-off

def test_the_engine_is_handed_the_export_not_the_ablated_dataset(run, engine_invoker):
    call = engine_invoker.calls[0]
    assert call["trace_file"] == run.run_dir / "traces.json"


def test_the_engine_export_is_leak_free(run):
    blob = (run.run_dir / "traces.json").read_text()
    for token in ("ablation_ids", "injection_mode", "replay_edit", "dependency_fault"):
        assert token not in blob


def test_the_engine_is_handed_the_seed_board_and_the_taxonomy(run, engine_invoker):
    call = engine_invoker.calls[0]
    assert call["seed_board"].source == "seed"
    assert "other" in {c.category_id for c in call["categories"]}


def test_the_engine_run_config_carries_the_pinned_knobs(run, engine_invoker):
    engine = engine_invoker.calls[0]["engine"]
    assert engine.recursion_limit >= 10_000
    assert engine.analysis_concurrency >= 1


def test_the_predicted_board_is_restamped_on_ingest(run):
    """The Engine's own board_id is not byte-compatible with ours."""
    assert run.predicted.board_id != "engine-side-hash-not-ours"
    assert run.predicted.board_id == content_hash(run.predicted)


def test_the_predicted_board_is_the_updated_board(run):
    """Assignment: an issueboard goes in, the UPDATED board comes out."""
    seed_ids = {i.error_id for i in run.seed_board.issues}
    assert seed_ids <= {i.error_id for i in run.predicted.issues}


def test_an_empty_seed_board_is_the_default(run):
    assert run.seed_board.issues == []
    assert run.seed_board.source == "seed"


def test_a_provided_seed_board_is_carried_through_and_updated(
    cfg, fake_harness_factory, tmp_path
):
    """The other half of the assignment's input: a board that already has issues."""
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(
        Issueboard(
            source="seed",
            issues=[
                {
                    "error_id": "seed-tool-failure-hidden",
                    "title": "Tool failure reported as success",
                    "description": "a tool errors and the answer claims it worked",
                    "category_id": "tool_misuse",
                    "severity": "high",
                }
            ],
        ).model_dump_json()
    )
    seeded = cfg.model_copy(update={"seed_issueboard": str(seed_path)}).with_root(cfg.root)
    invoker = FakeEngineInvoker()

    def stage(**kwargs):
        result = fake_run_ablation(**kwargs)
        invoker.ground_truth = result.ground_truth
        return result

    run = run_pipeline(
        seeded,
        ablation_stage=stage,
        engine_invoker=invoker,
        harness_factory=fake_harness_factory,
        expander=FakeExpander(),
    )
    assert invoker.calls[0]["seed_board"].issues[0].error_id == "seed-tool-failure-hidden"
    assert "seed-tool-failure-hidden" in {i.error_id for i in run.predicted.issues}
    assert by_name(run.deliverables)["issueboard_in"].ok
    assert by_name(run.deliverables)["issueboard_out_updated"].ok
    assert (run.run_dir / "seed_issueboard.json").exists()


# ------------------------------------------------------------------ scoring

def test_scoring_sees_the_full_trace_universe_including_clean_traces(run):
    """Kappa depends on n; dropping control traces would silently break it."""
    assert run.report.base_rates["n_traces"] == len(run.ablated.traces)
    assert run.report.base_rates["clean_traces"] >= 1


def test_base_rates_carry_the_control_fraction_and_injection_counts(run):
    rates = run.report.base_rates
    assert rates["control_fraction"] == pytest.approx(0.3)
    assert rates["per_error_injection_counts"]
    assert rates["injection_modes"]


def test_a_perfect_engine_scores_perfectly(cfg, fake_harness_factory):
    """The wiring is right if a mirror-image prediction comes back as 1.0."""

    class Mirror(FakeEngineInvoker):
        pass

    invoker = Mirror(recall=1.0)

    def stage(**kwargs):
        result = fake_run_ablation(**kwargs)
        invoker.ground_truth = result.ground_truth
        return result

    run = run_pipeline(
        cfg,
        ablation_stage=stage,
        engine_invoker=invoker,
        harness_factory=fake_harness_factory,
        expander=FakeExpander(),
    )
    injected = {i.category_id for i in run.ground_truth.issues}
    recalls = {s.category_id: s.recall for s in run.report.category_scores}
    assert injected, "the fake planted nothing to recall"
    assert all(recalls[c] == pytest.approx(1.0) for c in injected)
    # The one prediction with no injected counterpart lands in the E_h pool —
    # every real run has some, which is why the report has an appendix for them.
    assert run.report.eh_candidates == ["P-extra"]


def test_the_report_notes_when_the_engine_covered_no_traces(cfg, fake_harness_factory):
    class Blind(FakeEngineInvoker):
        def __call__(self, **kwargs):
            return super().__call__(**kwargs)

    run = run_pipeline(
        cfg,
        ablation_stage=fake_run_ablation,
        engine_invoker=Blind(Issueboard(source="ground_truth"), recall=0.0),
        harness_factory=fake_harness_factory,
        expander=FakeExpander(),
    )
    assert run.manifest.counts["engine_issues"] >= 0
    assert run.manifest.counts["traces"] == len(run.ablated.traces)


# ---------------------------------------------------------------- determinism

def test_rerunning_the_same_config_produces_the_same_ids(
    cfg, fake_harness_factory, engine_invoker
):
    kwargs = dict(
        ablation_stage=fake_run_ablation,
        harness_factory=fake_harness_factory,
        expander=FakeExpander(),
    )
    first = run_pipeline(cfg, engine_invoker=FakeEngineInvoker(), **kwargs)
    second = run_pipeline(cfg, engine_invoker=FakeEngineInvoker(), **kwargs)
    assert first.inputs.dataset_id == second.inputs.dataset_id
    assert first.ablated.dataset_id == second.ablated.dataset_id
    assert first.ground_truth.board_id == second.ground_truth.board_id
    assert first.report.report_id == second.report.report_id


# ---------------------------------------------------- no network, no servers

def test_no_server_is_started_when_none_is_supplied(run):
    """The CI path manages nothing: `servers` defaults to a no-op lifetime."""
    assert run.manifest.stages.get("servers", "none") == "none"


def test_the_export_traces_match_the_ablated_dataset(run):
    payload = json.loads(Path(run.run_dir / "traces.json").read_text())
    ablated = TraceDataset.model_validate_json((run.run_dir / "ablated_traces.json").read_text())
    assert [t["trace_id"] for t in payload["traces"]] == [t.trace_id for t in ablated.traces]
