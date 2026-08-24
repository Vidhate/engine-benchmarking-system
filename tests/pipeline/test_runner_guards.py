"""The guards that decide whether a run is allowed to be believed.

Each of these protects something that fails *quietly* otherwise: a model config
LangGraph declined to inject, an Engine response lost to a validation error
after half an hour of analysis, a replay forked against a server that was
already shut down, or a stand-in stage whose numbers get read as a result.
"""

from __future__ import annotations

import json

import pytest

from benchmark.pipeline.contracts import EngineInvocation
from benchmark.pipeline.engine import EngineModelMismatch, EngineRunFailed
from benchmark.pipeline.fakes import fake_run_ablation
from benchmark.pipeline.runner import run_pipeline
from benchmark.pipeline.servers import ServerLifetime
from benchmark.schemas import Issueboard
from tests.pipeline.conftest import FakeEngineInvoker, FakeExpander, FakeHarness


@pytest.fixture
def cfg(mini_cfg):
    return mini_cfg


@pytest.fixture
def run(mini_run):
    return mini_run


def go(cfg, fake_harness_factory, **overrides):
    invoker = overrides.pop("engine_invoker", None) or FakeEngineInvoker()

    def stage(**kwargs):
        result = fake_run_ablation(**kwargs)
        if hasattr(invoker, "ground_truth"):
            invoker.ground_truth = result.ground_truth
        return result

    kwargs = {
        "ablation_stage": stage,
        "engine_invoker": invoker,
        "harness_factory": fake_harness_factory,
        "expander": FakeExpander(),
    }
    kwargs.update(overrides)
    return run_pipeline(cfg, **kwargs)


# ------------------------------------------------- model readback (I4)

class WrongModelInvoker(FakeEngineInvoker):
    """The server ran something other than what the run asked for."""

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        result.recorded_models = ["some-other-model"]
        return result


class SilentInvoker(FakeEngineInvoker):
    """No readable run record — evidence unavailable, not evidence of a swap."""

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        result.recorded_models = []
        return result


def test_a_model_the_server_did_not_run_fails_the_run(cfg, fake_harness_factory):
    """Both arms of a comparison quietly running the same model is a result
    that looks like a finding and is not one."""
    with pytest.raises(EngineModelMismatch, match="some-other-model"):
        go(cfg, fake_harness_factory, engine_invoker=WrongModelInvoker())


def test_the_matching_model_passes(cfg, fake_harness_factory):
    run = go(cfg, fake_harness_factory)
    assert run.manifest.models["engine_recorded"] == cfg.engine.model


def test_an_unreadable_run_record_warns_rather_than_fails(cfg, fake_harness_factory):
    """A missing readback endpoint is a server capability gap, not a swapped
    model — but the report must not claim confirmation it does not have."""
    run = go(cfg, fake_harness_factory, engine_invoker=SilentInvoker())
    assert any("no readable record" in w for w in run.manifest.warnings)
    assert run.manifest.models["engine_recorded"] == ""


# ---------------------------------------- surviving a bad Engine reply (I3)

class BrokenInvoker:
    is_pipeline_fake = True

    def __init__(self):
        self.payload = {"issues": "not a list", "occurrences": []}

    def __call__(self, **kwargs):
        raise EngineRunFailed("engine output does not validate", raw_output=self.payload)


def test_a_failed_engine_run_still_persists_its_raw_output(cfg, fake_harness_factory):
    invoker = BrokenInvoker()
    with pytest.raises(EngineRunFailed):
        go(cfg, fake_harness_factory, engine_invoker=invoker)
    written = cfg.run_dir / "engine_raw_output.json"
    assert written.exists(), "hours of Engine time were lost with the response"
    assert json.loads(written.read_text()) == invoker.payload


class NoPayloadInvoker:
    is_pipeline_fake = True

    def __call__(self, **kwargs):
        raise EngineRunFailed("connection reset")


def test_a_failure_with_no_payload_still_propagates(cfg, fake_harness_factory):
    with pytest.raises(EngineRunFailed, match="connection reset"):
        go(cfg, fake_harness_factory, engine_invoker=NoPayloadInvoker())


# --------------------------------------- server choreography (I5)

class SpyLifetime(ServerLifetime):
    """Records enter/exit alongside whatever the stages record."""

    def __init__(self, events: list[str]):
        super().__init__(".", {}, enabled=True)
        self.events = events

    def start(self, name: str) -> None:
        self.events.append(f"start:{name}")

    def stop(self, name: str) -> None:
        self.events.append(f"stop:{name}")


class RecordingHarness(FakeHarness):
    def __init__(self, cfg, store, events):
        super().__init__(cfg, store)
        self.events = events

    def run_batch(self, inputs):
        self.events.append("harness.run_batch")
        return super().run_batch(inputs)


@pytest.fixture
def choreography(cfg, monkeypatch):
    events: list[str] = []
    invoker = FakeEngineInvoker()

    def factory(app_cfg, store):
        return RecordingHarness(app_cfg, store, events)

    def stage(**kwargs):
        events.append("ablation")
        result = fake_run_ablation(**kwargs)
        invoker.ground_truth = result.ground_truth
        return result

    # A wrapper function, not an assignment onto the instance: Python looks
    # dunder methods up on the class, so `invoker.__call__ = ...` is ignored.
    def recording_invoker(**kwargs):
        events.append("engine")
        return invoker(**kwargs)

    recording_invoker.is_pipeline_fake = True  # type: ignore[attr-defined]

    run_pipeline(
        cfg,
        ablation_stage=stage,
        engine_invoker=recording_invoker,
        harness_factory=factory,
        expander=FakeExpander(),
        servers=SpyLifetime(events),
    )
    return events


def test_the_harness_runs_inside_the_target_app_server_lifetime(choreography):
    assert choreography.index("start:target_app") < choreography.index("harness.run_batch")
    assert choreography.index("harness.run_batch") < choreography.index("stop:target_app")


def test_the_ablation_stage_runs_inside_the_SAME_target_app_lifetime(choreography):
    """Mode-A replay forks a thread the batch created; `langgraph dev` loses
    thread state on restart, so a stop between the two breaks every replay."""
    assert choreography.index("start:target_app") < choreography.index("ablation")
    assert choreography.index("ablation") < choreography.index("stop:target_app")
    between = choreography[
        choreography.index("start:target_app") : choreography.index("stop:target_app")
    ]
    assert "harness.run_batch" in between and "ablation" in between
    assert "stop:target_app" not in between[1:], "the server restarted mid-sequence"


def test_the_engine_runs_after_the_target_app_is_down(choreography):
    assert choreography.index("stop:target_app") < choreography.index("start:engine")
    assert choreography.index("start:engine") < choreography.index("engine")
    assert choreography.index("engine") < choreography.index("stop:engine")


def test_both_servers_are_stopped(choreography):
    assert choreography.count("stop:target_app") == 1
    assert choreography.count("stop:engine") == 1


# --------------------------------------------- faked-stage reporting (M3, M4)

def test_every_faked_stage_is_named_in_the_warning(run):
    faked = [w for w in run.manifest.warnings if "FAKED" in w]
    assert len(faked) == 1
    for stage in ("ablation", "engine_invoker", "harness"):
        assert stage in faked[0]


def test_the_faked_stages_reach_report_json_not_only_the_manifest(run):
    """report.json travels on its own; it must carry its own health warning."""
    assert run.report.base_rates["faked_stages"] == ["ablation", "engine_invoker", "harness"]
    on_disk = json.loads((run.run_dir / "report.json").read_text())
    assert on_disk["base_rates"]["faked_stages"]


def test_a_real_stage_is_not_flagged(cfg, fake_harness_factory):
    class RealEnoughInvoker(FakeEngineInvoker):
        is_pipeline_fake = False

    run = go(cfg, fake_harness_factory, engine_invoker=RealEnoughInvoker())
    assert "engine_invoker" not in run.report.base_rates["faked_stages"]
    assert "harness" in run.report.base_rates["faked_stages"]


def test_the_warning_says_what_the_numbers_are_worth(run):
    warning = next(w for w in run.manifest.warnings if "FAKED" in w)
    assert "wiring" in warning
    assert warning == run.manifest.warnings[0], "the caveat must lead, not trail"


# ------------------------------------------------ phantom occurrences (I2)

class PhantomInvoker(FakeEngineInvoker):
    """Predicts against a trace id that is not in the dataset."""

    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        board = result.board
        return EngineInvocation(
            board=Issueboard(
                board_id=board.board_id,
                source=board.source,
                issues=board.issues,
                occurrences=[
                    *board.occurrences,
                    {"error_id": board.issues[-1].error_id, "trace_id": "tr-unknown"},
                ],
            ),
            raw_output=result.raw_output,
            seconds=result.seconds,
            thread_id=result.thread_id,
            recorded_models=result.recorded_models,
            trace_count=result.trace_count,
        )


@pytest.fixture
def phantom_run(cfg, fake_harness_factory):
    return go(cfg, fake_harness_factory, engine_invoker=PhantomInvoker())


def test_a_prediction_against_a_nonexistent_trace_is_dropped(phantom_run):
    universe = {t.trace_id for t in phantom_run.ablated.traces}
    assert all(o.trace_id in universe for o in phantom_run.scored.board.occurrences)


def test_the_dropped_phantom_is_counted_in_the_manifest(phantom_run):
    assert phantom_run.manifest.counts["phantom_occurrences"] == 1


def test_the_dropped_phantom_is_counted_in_the_report(phantom_run):
    delta = phantom_run.report.base_rates["engine_delta"]
    assert delta["phantom_occurrences"] == 1
    assert delta["phantom_trace_ids"] == ["tr-unknown"]


def test_the_dropped_phantom_is_warned_about(phantom_run):
    assert any("not in the dataset" in w for w in phantom_run.manifest.warnings)


def test_the_dropped_phantom_appears_in_the_markdown(phantom_run):
    assert "tr-unknown" in phantom_run.markdown


def test_the_verbatim_board_still_carries_it(phantom_run):
    """The assignment deliverable is what the Engine returned, unedited."""
    assert any(o.trace_id == "tr-unknown" for o in phantom_run.predicted.occurrences)
    on_disk = json.loads((phantom_run.run_dir / "predicted_issueboard.json").read_text())
    assert any(o["trace_id"] == "tr-unknown" for o in on_disk["occurrences"])


def test_the_scored_board_is_persisted_next_to_it(phantom_run):
    on_disk = json.loads((phantom_run.run_dir / "scored_issueboard.json").read_text())
    assert all(o["trace_id"] != "tr-unknown" for o in on_disk["occurrences"])
