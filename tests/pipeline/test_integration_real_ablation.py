"""The miniature run with the REAL Phase-5 ablation stage. The seam's CI gate.

`test_integration_mini.py` runs the pipeline against `fake_run_ablation`, which
was written to match the pinned contract — so it proves the pipeline works
against a shape somebody wrote down, not against the shape Phase 5 produces.
Everything that differs between those two is invisible to it. The first such
difference cost a real run: `benchmark.ablation.export` writes a BARE LIST of
stripped traces and the stand-in writes a `{traces: [...]}` envelope, and every
reader on the pipeline side assumed the envelope.

So this runs the genuine `benchmark.ablation` code path — the real split, the
real filters, the real injection, the real leak-stripped export, the real
`AblationResult` — with only the two things that cost money or a server
replaced:

* the LLM ablation agent, by `ScriptedAblationAgent` (the same proposals the
  ablation package's own tests use), and
* the LangGraph target app, by the harness stand-in from `tests/ablation` —
  which is the one written against the ablation engine's demands (`replay`,
  `run_with_faults`, `locate_checkpoint`, thread liveness), unlike the
  pipeline's own, which only knows how to run a batch.

Zero network, zero servers, zero model calls, and the ablation stage is real.
"""

from __future__ import annotations

import json

import pytest

from benchmark.ablation import AblationEngine
from benchmark.ablation.agent import ScriptedAblationAgent
from benchmark.pipeline.deliverables import check_deliverables
from benchmark.pipeline.export import assert_export_file_clean, export_traces
from benchmark.pipeline.fakes import FakeEngineInvoker, FakeExpander
from benchmark.pipeline.runner import run_pipeline
from benchmark.schemas import OutputDataset, OutputRecord
from benchmark.schemas.io import derive
from tests.ablation.conftest import FakeHarness as AblationFakeHarness
from tests.ablation.conftest import make_proposal, make_traces


class BatchableAblationHarness(AblationFakeHarness):
    """The ablation stand-in harness, plus the batch surface the pipeline needs.

    One object serving both stages is not a convenience here, it is the
    property under test: the runner builds ONE harness and hands the same
    instance to `run_batch` and to `run_ablation`, because Mode-A replay forks
    a thread the batch created. A test that used two would not be exercising
    the choreography the live run depends on.
    """

    is_pipeline_fake = True

    def run_batch(self, inputs):
        traces = make_traces(inputs)
        for trace in traces.traces:
            self.store.put(trace)
        self.stats = {"ran": len(traces.traces), "skipped": 0, "quarantined": 0, "app_error": 0}
        outputs = OutputDataset(
            outputs=[
                OutputRecord(
                    input_id=trace.input_id,
                    trace_id=trace.trace_id,
                    responses=[t.final_response for t in trace.turns],
                )
                for trace in traces.traces
            ]
        )
        return derive(outputs, inputs), derive(traces, inputs)


@pytest.fixture
def scripted_agent(taxonomy):
    """One proposal per taxonomy category, alternating the two injection modes.

    Alternating is deliberate: `replay_edit` and `dependency_fault` take
    completely different routes through the ablation engine (a forked thread
    versus a re-run under a fault shim), and a run that exercised one of them
    would leave the other's hand-off to the pipeline unproven.
    """
    modes = ("replay_edit", "dependency_fault")
    return ScriptedAblationAgent(
        {
            category.category_id: [
                make_proposal(
                    f"E-{category.category_id}",
                    category.category_id,
                    mode=modes[index % len(modes)],
                    target_count=2,
                )
            ]
            for index, category in enumerate(taxonomy)
        }
    )


@pytest.fixture
def real_ablation_run(mini_cfg, scripted_agent):
    invoker = FakeEngineInvoker()

    def harness_factory(cfg, store):
        return BatchableAblationHarness(cfg, store)

    def real_stage(*, traces, inputs, categories, cfg, harness, store, export_path):
        """The real `AblationEngine.run`, with the LLM agent injected.

        `run_ablation` itself takes no agent — it builds the OpenAI one — so
        the entrypoint is reached one level down. Everything the pipeline
        touches (the split, the injection, the export, the result object) is
        the production code.
        """
        result = AblationEngine(harness, store, cfg, agent=scripted_agent).run(
            traces, inputs, categories, export_path
        )
        invoker.ground_truth = result.ground_truth
        return result

    return run_pipeline(
        mini_cfg,
        ablation_stage=real_stage,
        engine_invoker=invoker,
        harness_factory=harness_factory,
        expander=FakeExpander(),
    )


# --------------------------------------------------------------- the hand-off

def test_the_real_stage_result_reaches_the_pipeline(real_ablation_run):
    run = real_ablation_run
    assert run.ground_truth.issues, "the real stage injected nothing"
    assert run.records, "no ablation records crossed the seam"
    assert run.ablated.parent_dataset_id == run.traces.dataset_id


def test_the_ablated_corpus_is_not_the_collected_one(real_ablation_run):
    """The property the pass-through stand-in can never demonstrate."""
    run = real_ablation_run
    source = {t.trace_id: t for t in run.traces.traces}
    changed = [
        t
        for t in run.ablated.traces
        if t.trace_id not in source
        or t.model_dump(mode="json", exclude={"ablation_ids"})
        != source[t.trace_id].model_dump(mode="json", exclude={"ablation_ids"})
    ]
    assert changed, "nothing was injected — the corpus came through untouched"


def test_the_split_is_provenance_based(real_ablation_run):
    split = real_ablation_run.split
    assert split.strata, "the split records no strata"
    assert split.control_input_ids and split.ablate_input_ids


def test_the_report_records_how_the_errors_were_planted(real_ablation_run):
    rates = real_ablation_run.report.base_rates
    assert rates["injection_modes"], "the report cannot say how errors were planted"
    assert set(rates["injection_modes"]) <= {"replay_edit", "dependency_fault"}
    assert rates["per_error_injection_counts"]


def test_the_ablation_stage_is_not_flagged_as_faked(real_ablation_run):
    """The harness and the invoker are stand-ins here; the ablation is not.

    The live smoke reads the ABSENCE of a FAKED warning as its proof that a run
    was real, so what gets counted as faked has to be per-stage and exact.
    """
    warnings = [w for w in real_ablation_run.manifest.warnings if "FAKED" in w]
    assert warnings, "the fake harness and invoker must still be declared"
    listed = warnings[0].split("FAKED stage(s): ", 1)[1].split(". This run", 1)[0]
    faked = {entry.split(" (")[0] for entry in listed.split(", ")}
    assert faked == {"engine_invoker", "harness"}, warnings[0]


# ------------------------------------------------------------- the export file

def test_the_pipeline_audits_the_export_the_real_stage_wrote(real_ablation_run):
    """A bare list, not an envelope — and every reader here copes with it."""
    run = real_ablation_run
    payload = json.loads(run.export_path.read_text())
    assert isinstance(payload, list)
    audited = assert_export_file_clean(run.export_path)
    assert len(export_traces(audited)) == len(run.ablated.traces)


def test_the_export_names_no_ground_truth(real_ablation_run):
    blob = real_ablation_run.export_path.read_text()
    for token in ("ablation_ids", "injection_mode", "replay_edit", "dependency_fault"):
        assert token not in blob, f"the Engine's trace file names {token!r}"


def test_the_export_carries_no_time_separator(real_ablation_run):
    """Ablation runs after collection; an un-normalized clock sorts the corpus."""
    from benchmark.ablation.export import EXPORT_EPOCH

    traces = export_traces(json.loads(real_ablation_run.export_path.read_text()))
    origins = {
        min(span["start_time"] for turn in trace["turns"] for span in turn["spans"])
        for trace in traces
        if any(turn["spans"] for turn in trace["turns"])
    }
    assert len(origins) == 1, f"exported traces start at {len(origins)} distinct clocks"
    assert origins.pop().startswith(EXPORT_EPOCH.date().isoformat())


def test_every_deliverable_passes_over_a_real_ablation_run(real_ablation_run):
    """The check the CLI's `check` subcommand runs, over a real export."""
    run = real_ablation_run
    checks = check_deliverables(run.run_dir, min_traces=len(run.ablated.traces))
    failed = [f"{c.name}: {c.detail}" for c in checks if not c.ok]
    assert not failed, failed
