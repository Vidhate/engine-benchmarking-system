"""Stage III top level — the four-step loop, wired.

    [N, M, T]  ->  [N, M, T*], [N, E_K]
    traces         ablated traces  ground-truth issueboard

`run_ablation` is the entrypoint Phase 7 codes against. Its signature is fixed;
the swappable pieces (the LLM agent) are injected through `AblationEngine`,
which `run_ablation` is a thin default-wiring wrapper over. That is what keeps
the public signature stable while unit tests stay network-free.

Order of operations, and why:

1. **Split first, at the input level.** Before any proposal is drafted, so
   nothing downstream can accidentally consider a control input. Control
   inputs are never ablated and never re-run (`docs/architecture/04-ablation-engine.md`,
   "Prevalence control").
2. **Probe thread liveness.** Only if a `replay_edit` was actually proposed.
   Threads are server-lifetime state, and a dead-thread corpus should fail with
   an explanation, not with N mysterious checkpoint lookups.
3. **Validate, then apply.** Dry runs never persist; only step 4 writes.
4. **Export last, audited before it is written.**
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from benchmark.ablation.agent import AblationAgent, CorpusDigest, OpenAIAblationAgent
from benchmark.ablation.apply import apply_ablations
from benchmark.ablation.export import write_engine_export
from benchmark.ablation.inject import assert_threads_alive
from benchmark.ablation.propose import DEFAULT_MIN_PER_MODE, build_digest, propose_errors
from benchmark.ablation.split import make_split
from benchmark.ablation.validate import ValidationOutcome, validate_specs
from benchmark.schemas.ablation import AblationRecord, AblationSplit
from benchmark.schemas.configs import AblationConfig
from benchmark.schemas.inputs import InputDataset
from benchmark.schemas.io import stamp_dataset_id
from benchmark.schemas.issues import ErrorCategory, Issueboard
from benchmark.schemas.traces import Trace, TraceDataset
from benchmark.tracing.store import TraceStore

if TYPE_CHECKING:  # the annotation only — importing the LangGraph-aware package
    from benchmark.harness import Harness  # at runtime would drag its SDKs in

log = logging.getLogger("benchmark.ablation")


class AblationResult(BaseModel):
    """Everything Stage III produces. Only `export_path` crosses to Engine."""

    ablated: TraceDataset
    ground_truth: Issueboard
    records: list[AblationRecord] = Field(default_factory=list)
    split: AblationSplit
    export_path: str
    dropped_errors: list[str] = Field(default_factory=list)
    # Reported, not hidden: the design requires control fraction and per-error
    # injection counts to be recoverable so precision/recall are interpretable
    # against known base rates.
    injected_counts: dict[str, int] = Field(default_factory=dict)
    # Candidates burned by the not-retracted check, per error — the app
    # defending itself against an injection. Report-visible so the residual
    # false-negative surface documented in `inject.retraction_in` is bounded by
    # a number rather than by a guess.
    self_corrected_counts: dict[str, int] = Field(default_factory=dict)
    validation: ValidationOutcome = Field(default_factory=ValidationOutcome)
    digest: CorpusDigest = Field(default_factory=CorpusDigest)


class _StoreWithFallback:
    """Reads through the `TraceStore`, falling back to the in-memory dataset.

    The digest is built from store reads (the Phase-0 tracing boundary), but a
    caller may hand `run_ablation` a `TraceDataset` whose traces were never
    written to *this* store. Falling back keeps step 1 grounded in real traces
    instead of silently proposing against an empty corpus.
    """

    def __init__(self, store: TraceStore, traces: Sequence[Trace]):
        self._store = store
        self._memory = {t.trace_id: t for t in traces}

    def get(self, trace_id: str) -> Trace:
        try:
            return self._store.get(trace_id)
        except KeyError:
            return self._memory[trace_id]


def default_agent() -> AblationAgent:
    """The live agent. Patched out in unit tests; never called at import."""
    return OpenAIAblationAgent()


class AblationEngine:
    """The four-step loop with its dependencies made explicit."""

    def __init__(
        self,
        harness: Any,
        store: TraceStore,
        cfg: AblationConfig,
        *,
        agent: AblationAgent | None = None,
    ):
        self.harness = harness
        self.store = store
        self.cfg = cfg
        self._agent = agent

    @property
    def agent(self) -> AblationAgent:
        if self._agent is None:
            self._agent = default_agent()
        return self._agent

    def run(
        self,
        traces: TraceDataset,
        inputs: InputDataset,
        categories: Sequence[ErrorCategory],
        export_path: str | Path,
    ) -> AblationResult:
        if not traces.dataset_id:
            traces = stamp_dataset_id(traces)
        if not inputs.dataset_id:
            inputs = stamp_dataset_id(inputs)

        # ---------------------------------------------- step 0: split first
        split = make_split(inputs, self.cfg)
        ablate_ids = set(split.ablate_input_ids)
        ablate_traces = [t for t in traces.traces if t.input_id in ablate_ids]
        log.info(
            "split: %d control / %d ablate input(s) over %d strata; %d ablate-set trace(s)",
            len(split.control_input_ids),
            len(split.ablate_input_ids),
            len(split.strata),
            len(ablate_traces),
        )

        # ------------------------------------------------ step 1: propose
        digest = build_digest(
            _StoreWithFallback(self.store, traces.traces),
            [t.trace_id for t in ablate_traces],
            self.harness.cfg,
            app_context=inputs.generation_config.app_context,
        )
        proposals, digest, dropped_categories = propose_errors(
            _StoreWithFallback(self.store, traces.traces),
            [t.trace_id for t in ablate_traces],
            list(categories),
            self.cfg.n_per_category,
            self.harness.cfg,
            self.agent,
            digest=digest,
            # The engine owns the floor, not AblationConfig: one mechanism
            # error is what the report's content-vs-mechanism half needs, on
            # every run, and a knob would only invite it being turned off.
            min_per_mode=DEFAULT_MIN_PER_MODE,
        )
        log.info(
            "proposed %d error(s): %s",
            len(proposals),
            {p.issue.error_id: p.issue.injection_mode for p in proposals},
        )

        # ------------------------- thread liveness (only if Mode A is in play)
        # A Mode-C-only run never forks a thread, so probing is pure cost —
        # one `get_history` per distinct thread against a live server — and
        # `assert_threads_alive` would abort a perfectly valid dependency-fault
        # run over threads it was never going to use.
        replayable: set[str] | None = None
        if any(p.issue.injection_mode == "replay_edit" for p in proposals):
            replayable = assert_threads_alive(ablate_traces, self.harness)
        else:
            log.info("no replay_edit error proposed — skipping the thread-liveness probe")

        # ------------------------------------- steps 2 + 3: plan and validate
        inputs_by_id = {i.input_id: i for i in inputs.inputs}
        personas = {
            p.persona_id: p
            for p in (
                *inputs.generation_config.personas,
                *inputs.generation_config.adversarial_personas,
            )
        }
        outcome = validate_specs(
            proposals,
            traces.traces,
            ablate_ids,
            self.harness,
            inputs_by_id,
            agent=self.agent,
            digest=digest,
            min_eligible=self.cfg.min_eligible,
            dataset_id=traces.dataset_id,
            replayable_trace_ids=replayable,
            personas=personas,
            max_turns=inputs.generation_config.max_turns,
            max_replans=self.cfg.max_replans,
            target_count=self.cfg.target_count,
            seed=self.cfg.seed,
        )
        log.info(
            "validated %d/%d spec(s); dropped %s",
            len(outcome.specs),
            len(proposals),
            sorted(outcome.dropped),
        )

        # -------------------------------------------------- step 4: apply
        applied = apply_ablations(
            outcome.specs,
            outcome.proposals,
            traces,
            inputs,
            split,
            self.harness,
            seed=self.cfg.seed,
            dataset_id=traces.dataset_id,
            max_turns=inputs.generation_config.max_turns,
        )

        # ------------------------------------------ leak-stripped export
        behaviors = tuple(
            spec.fault_config.behavior
            for spec in outcome.specs
            if spec.fault_config is not None and "_" in (spec.fault_config.behavior or "")
        )
        path = write_engine_export(
            applied.ablated, export_path, self.harness.cfg, extra_tokens=behaviors
        )

        dropped = [
            *dropped_categories.values(),
            *outcome.dropped.values(),
            *applied.dropped.values(),
        ]
        return AblationResult(
            ablated=applied.ablated,
            ground_truth=applied.ground_truth,
            records=applied.records,
            split=split,
            export_path=str(path),
            dropped_errors=sorted(dropped),
            injected_counts=applied.injected,
            self_corrected_counts=applied.self_corrected,
            validation=outcome,
            digest=digest,
        )


def run_ablation(
    traces: TraceDataset,
    inputs: InputDataset,
    categories: list[ErrorCategory],
    cfg: AblationConfig,
    harness: Harness,
    store: TraceStore,
    export_path: Path,
) -> AblationResult:
    """Stage III entrypoint — the signature Phase 7 codes against."""
    return AblationEngine(harness, store, cfg).run(traces, inputs, categories, export_path)
