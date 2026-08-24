#!/usr/bin/env python3
"""Live end-to-end verification of the Phase 5 ablation engine. NOT run by scripts/ci.sh.

Manages the target app's `langgraph dev` server itself (via the app's own
`scripts/serve.sh` — a shell call, never an import), generates a small FRESH
corpus with the real Phase-4 harness, and then runs the full four-step ablation
loop against it **inside the same server lifetime**.

That last part is not a convenience. LangGraph threads are server-lifetime
state: Mode A forks a thread at a checkpoint, and a corpus collected under an
earlier `langgraph dev` process has dead thread refs. Generating and ablating in
one process is the only arrangement in which `replay_edit` can work at all.

Gates, in order:

  1. corpus   — a fresh ~8-input batch through the real harness, schema-valid,
                nothing quarantined, threads alive.
  2. propose  — the live ablation agent (ABLATION_AGENT_MODEL) drafts concrete
                errors per category from a digest of THESE traces.
  3. validate — mode-aware dry runs; a deliberately impossible spec is rejected
                with its reason surfaced.
  4. apply    — >= 2 replay_edit and >= 2 dependency_fault errors injected,
                ground-truth board + AblationRecords produced, same-category
                disjointness holds.
  5. control  — at least one control input's trace is byte-identical to the one
                the harness collected.
  6. export   — the Engine-facing file passes the no-leak audit and parses.

Usage:
    uv run python scripts/ablation_smoke.py            # manages the server
    uv run python scripts/ablation_smoke.py --no-serve # server already running
    uv run python scripts/ablation_smoke.py --keep     # leave the server up

Requires OPENAI_API_KEY and LANGSMITH_API_KEY (read from a repo-root .env).
Budget ~10-20 minutes: every Mode-C injection is a real app run plus LangSmith
child-span ingestion (which lags the root by up to ~30s).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SERVE = REPO_ROOT / "apps" / "target_app" / "scripts" / "serve.sh"
MINI_CONFIG = REPO_ROOT / "configs" / "generation" / "mini.yaml"
TAXONOMY = REPO_ROOT / "configs" / "taxonomy.yaml"
OUT_DIR = REPO_ROOT / "data" / "ablation_smoke"

# Which mode each category's error is drafted for. The agent still authors the
# error itself; pinning the mode is how this script guarantees the gate covers
# BOTH modes on a corpus this small, rather than hoping six independent LLM
# draws happen to split 2/2.
MODE_BY_CATEGORY = {
    "hallucination": ["replay_edit"],
    "instruction_violation": ["replay_edit"],
    "retrieval_failure": ["dependency_fault"],
    "tool_misuse": ["dependency_fault"],
}
TARGET_COUNT = 2  # injections per error — see "schema proposal" in the report


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip()


def banner(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}", flush=True)


def serve(action: str) -> None:
    print(f"[server] {action}…", flush=True)
    subprocess.run([str(SERVE), action], check=True, cwd=SERVE.parent.parent)


class ModePinnedAgent:
    """The live agent, with `allowed_modes` narrowed per category.

    A thin script-local wrapper: it changes nothing about how an error is
    drafted, only which modes the drafting prompt is allowed to choose from,
    and it pins `target_count` (which `AblationConfig` has no field for — see
    the report's schema proposals).
    """

    def __init__(self, inner, mode_by_category: dict[str, list[str]], target_count: int):
        self.inner = inner
        self.mode_by_category = mode_by_category
        self.target_count = target_count

    def propose(self, category, n, digest, allowed_modes):
        pinned = self.mode_by_category.get(category.category_id, list(allowed_modes))
        modes = [m for m in pinned if m in allowed_modes] or list(allowed_modes)
        drafts = self.inner.propose(category, n, digest, modes)
        return [d.model_copy(update={"target_count": self.target_count}) for d in drafts]

    def revise_corruption(self, proposal, digest, reasons):
        return self.inner.revise_corruption(proposal, digest, reasons)


def main() -> int:  # noqa: PLR0915 - a linear gate script reads better in one piece
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-serve", action="store_true", help="assume the server is up")
    parser.add_argument("--keep", action="store_true", help="leave the server running")
    parser.add_argument("--fresh", action="store_true", help="wipe the smoke artifacts first")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    for required in ("OPENAI_API_KEY", "LANGSMITH_API_KEY"):
        if not os.environ.get(required):
            print(f"BLOCKED: {required} is not set (copy .env to the repo root)")
            return 2
    os.environ.setdefault("LANGSMITH_TRACING", "true")

    import yaml

    from benchmark.ablation import AblationEngine, OpenAIAblationAgent
    from benchmark.ablation.export import audit_export
    from benchmark.ablation.inject import live_threads
    from benchmark.ablation.validate import validate_specs
    from benchmark.generation.config_loader import load_generation_config
    from benchmark.generation.expander import OpenAIPromptExpander
    from benchmark.generation.generators import generate_inputs
    from benchmark.harness import (
        Harness,
        LangGraphAppClient,
        LangSmithCollector,
        OpenAIUserSimulator,
        Quarantine,
        load_target_app_config,
    )
    from benchmark.schemas.ablation import FilterStep
    from benchmark.schemas.configs import AblationConfig
    from benchmark.schemas.inputs import InputDataset
    from benchmark.schemas.io import save, stamp_dataset_id
    from benchmark.schemas.issues import ErrorCategory, Issueboard
    from benchmark.schemas.traces import Trace
    from benchmark.tracing.store import LocalTraceStore

    if args.fresh and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_target_app_config(REPO_ROOT / "configs" / "target_app.yaml")
    categories = [
        ErrorCategory.model_validate(item)
        for item in yaml.safe_load(TAXONOMY.read_text())["categories"]
    ]
    print(f"target app contract : {cfg.model_dump_json()}")
    print(f"taxonomy C_E        : {[c.category_id for c in categories]}")

    if not args.no_serve:
        serve("start")
    try:
        # ------------------------------------------------------ GATE 1: corpus
        banner("GATE 1 — a FRESH corpus through the real harness, same server lifetime")
        gen_cfg = load_generation_config(MINI_CONFIG)
        full = generate_inputs(
            gen_cfg, OpenAIPromptExpander(), cache_dir=REPO_ROOT / "data" / "expansion_cache"
        )
        single = [i for i in full.inputs if i.mode == "single_turn"]
        multi = [i for i in full.inputs if i.mode == "multi_turn"]
        sliced = single[:6] + [multi[0], multi[-1]]  # 6 single-turn + 2 conversations
        inputs = stamp_dataset_id(
            InputDataset(
                created_at=full.created_at,
                generation_config=full.generation_config,
                inputs=sliced,
            )
        )
        save(inputs, OUT_DIR / "inputs.json")

        store = LocalTraceStore(OUT_DIR / "traces")
        quarantine = Quarantine(OUT_DIR / "quarantine")
        harness = Harness(
            cfg,
            store,
            client=LangGraphAppClient(cfg),
            # Gentler on LangSmith than the defaults. The collector polls
            # /runs/query once per poll_interval per in-flight trace, and a
            # measured 429 (Rate limit exceeded) killed a whole batch when two
            # smoke runs went back to back at concurrency 4 / 2.5s. The window
            # the collector waits for is (settle_polls-1) * poll_interval, so
            # settle_polls comes down as the interval goes up — same ~15s of
            # stability, a third of the requests.
            collector=LangSmithCollector(
                cfg.langsmith_project, cfg=cfg, poll_interval_s=6.0, settle_polls=4
            ),
            simulator=OpenAIUserSimulator(),
            quarantine=quarantine,
            concurrency=2,
        )
        started = time.time()
        outputs, traces = harness.run_batch(inputs)
        print(f"harness stats: {harness.stats}  ({time.time() - started:.1f}s)")
        save(traces, OUT_DIR / "traces.json")
        save(outputs, OUT_DIR / "outputs.json")

        assert not quarantine.list_ids(), f"quarantined: {quarantine.list_ids()}"
        assert len(traces.traces) == len(inputs.inputs), (
            f"{len(traces.traces)}/{len(inputs.inputs)} inputs produced a trace"
        )
        alive = live_threads(traces.traces, harness)
        print(f"live threads: {len(alive)}/{len(traces.traces)}")
        assert len(alive) == len(traces.traces), (
            "a thread died inside one server lifetime — that should be impossible"
        )
        pre_ablation = {t.input_id: t.model_dump_json() for t in traces.traces}
        conversations = [t for t in traces.traces if len(t.turns) > 1]
        print(f"corpus: {len(traces.traces)} traces, {len(conversations)} with M>1")
        print("PASS — fresh corpus, threads alive, nothing quarantined")

        # -------------------------------------------- GATE 3a: rejection proof
        banner("GATE 3a — validation rejects a deliberately broken spec, with the reason")
        from benchmark.ablation.agent import Corruption, ProposedError
        from benchmark.schemas.issues import Issue

        broken = ProposedError(
            issue=Issue(
                error_id="E-deliberately-broken",
                title="an error this corpus cannot express",
                description="filters on a tool span the app does not have",
                category_id="tool_misuse",
                severity="high",
                injection_mode="replay_edit",
            ),
            filter_steps=[
                FilterStep(field="span_names", op="eq", value="quantum_tunnelling_tool")
            ],
            corruption=Corruption(
                replacement="Ticket SMOKE-BROKEN-1 has been filed.", marker="SMOKE-BROKEN-1"
            ),
            target_count=2,
        )
        from benchmark.ablation.agent import CorpusDigest

        rejection = validate_specs(
            [broken],
            traces.traces,
            {i.input_id for i in inputs.inputs},
            harness,
            {i.input_id: i for i in inputs.inputs},
            agent=OpenAIAblationAgent(),
            digest=CorpusDigest(),
            min_eligible=99,
            dataset_id=traces.dataset_id,
            replayable_trace_ids=alive,
        )
        assert rejection.specs == [], "the broken spec must not validate"
        for failure in rejection.failures:
            print(f"  attempt {failure.attempt} [{failure.stage}]: {failure.reason}")
        assert "E-deliberately-broken" in rejection.dropped
        print(f"dropped: {rejection.dropped['E-deliberately-broken']}")
        print("PASS — rejected after the bounded re-plan loop, every reason surfaced")

        # -------------------------------------- GATES 2+3+4: the full live loop
        banner("GATES 2-4 — the live four-step loop (propose -> plan -> validate -> apply)")
        ablation_cfg = AblationConfig(
            seed=20260824,
            control_fraction=0.3,
            # 3, not the design default of 5: the ablate set of an 8-input smoke
            # corpus is ~6 traces, and a gate that cannot pass on its own fixture
            # is not a gate.
            min_eligible=3,
            n_per_category=1,
        )
        agent = ModePinnedAgent(OpenAIAblationAgent(), MODE_BY_CATEGORY, TARGET_COUNT)
        engine = AblationEngine(harness, store, ablation_cfg, agent=agent)
        started = time.time()
        result = engine.run(
            traces, inputs, categories, OUT_DIR / "engine_traces.json"
        )
        print(f"ablation took {time.time() - started:.1f}s")

        save(result.ablated, OUT_DIR / "ablated_traces.json")
        save(result.ground_truth, OUT_DIR / "ground_truth_issueboard.json")
        (OUT_DIR / "ablation_records.json").write_text(
            json.dumps([r.model_dump(mode="json") for r in result.records], indent=2) + "\n"
        )
        (OUT_DIR / "split.json").write_text(result.split.model_dump_json(indent=2) + "\n")

        print(f"split: {len(result.split.control_input_ids)} control / "
              f"{len(result.split.ablate_input_ids)} ablate over {len(result.split.strata)} strata")
        for issue in result.ground_truth.issues:
            count = result.injected_counts.get(issue.error_id, 0)
            print(f"  [{issue.injection_mode:17}] {issue.error_id:28} x{count}  {issue.title}")
        for reason in result.dropped_errors:
            print(f"  DROPPED: {reason}")

        by_mode: dict[str, int] = {}
        for issue in result.ground_truth.issues:
            if result.injected_counts.get(issue.error_id, 0) > 0:
                by_mode[issue.injection_mode] = by_mode.get(issue.injection_mode, 0) + 1
        print(f"errors injected by mode: {by_mode}")
        assert by_mode.get("replay_edit", 0) >= 2, f"need >= 2 replay_edit errors, got {by_mode}"
        assert by_mode.get("dependency_fault", 0) >= 2, (
            f"need >= 2 dependency_fault errors, got {by_mode}"
        )
        assert result.records, "no AblationRecords produced"
        Issueboard.model_validate_json(result.ground_truth.model_dump_json())

        category_of = {i.error_id: i.category_id for i in result.ground_truth.issues}
        keys = [(o.trace_id, category_of[o.error_id]) for o in result.ground_truth.occurrences]
        assert len(keys) == len(set(keys)), f"same-category disjointness broken: {keys}"
        print(f"occurrences: {len(keys)}, all (trace_id, category_id) keys unique")

        replay_records = [r for r in result.records if r.actions_applied]
        fault_records = [r for r in result.records if not r.actions_applied]
        print(f"records: {len(replay_records)} replay_edit, {len(fault_records)} dependency_fault")
        for record in replay_records[:2]:
            before, after = record.before_after[0]
            print(f"  {record.ablation_id}")
            print(f"    before: {before[:110]!r}")
            print(f"    after : {after[:110]!r}")
        for record in fault_records[:2]:
            print(f"  {record.ablation_id}\n    activation evidence: "
                  f"{record.before_after[0][1][:150]}")
        assert all(r.before_after[0][0] == "" for r in fault_records), (
            "dependency_fault records must carry ('', evidence)"
        )
        print("PASS — both modes injected, board + records produced, disjointness holds")

        # ----------------------------------------------------- GATE 5: control
        banner("GATE 5 — control inputs verifiably untouched")
        control = set(result.split.control_input_ids)
        assert control, "the split held nothing back"
        checked = 0
        for trace in result.ablated.traces:
            if trace.input_id in control:
                assert trace.model_dump_json() == pre_ablation[trace.input_id], (
                    f"control trace for {trace.input_id} was modified"
                )
                assert not trace.ablation_ids
                checked += 1
        injected_inputs = {t.input_id for t in result.ablated.traces if t.ablation_ids}
        assert not injected_inputs & control
        print(f"PASS — {checked} control trace(s) byte-identical to what the harness collected")

        # ------------------------------------------------------ GATE 6: export
        banner("GATE 6 — the Engine export passes the no-leak audit")
        payload = json.loads(Path(result.export_path).read_text())
        behaviors = tuple(
            spec.fault_config.behavior
            for spec in result.validation.specs
            if spec.fault_config is not None
        )
        audit_export(payload, cfg, extra_tokens=behaviors)
        assert len(payload) == len(result.ablated.traces)
        for item in payload:
            Trace.model_validate(item)
        blob = json.dumps(payload).lower()
        for tell in ("ablation", "injection_mode", "replay_edit", "dependency_fault",
                     "thread_id", "checkpoint", "session_id"):
            assert tell not in blob, f"the export names {tell!r}"
        print(f"exported {len(payload)} traces to {Path(result.export_path).name}, "
              f"scanned for {behaviors + ('ablation', 'thread_id', 'checkpoint')}")
        print("PASS — allowlist clean, no fault/ablation token, parses as Trace")

        banner("PHASE 5 SMOKE OK — every gate green")
        print(f"artifacts under {OUT_DIR.relative_to(REPO_ROOT)}")
        return 0
    finally:
        if not args.no_serve and not args.keep:
            serve("stop")


if __name__ == "__main__":
    raise SystemExit(main())
