#!/usr/bin/env python3
"""Live miniature end-to-end run of the Phase 7 pipeline. NOT run by scripts/ci.sh.

Drives the REAL path, with both real servers, the real ablation engine and the
real Engine on the mini model, over ~7 inputs:

    generate_inputs (real OpenAI expander)
      -> harness batch      (real target app on :2024, real LangSmith collection)
      -> run_ablation       (real LLM ablation agent, real injection, real replay)
      -> Engine             (real engine app on :2025, mini model)
      -> score()            -> BenchmarkReport + report.md + manifest.json

Every integration seam, none faked. That is what the gates below are checking:
the run's numbers are still miniature-scale and not worth much on their own,
but the *shape* of the run — a real injected ground truth, a leak-stripped
export the Engine can read, a score computed against errors somebody actually
planted — is the thing this script exists to prove.

**The absence of a FAKED warning in manifest.json is the signal that the run
was real.** Gate 4 asserts on it. Any stand-in anywhere in the assembly puts
that warning back, so it cannot be lost by accident.

Note the ordering the pipeline enforces and this script depends on: the
harness batch and the ablation stage run inside ONE target-app server
lifetime, because Mode-A replay forks a LangGraph thread created during the
batch and `langgraph dev` loses thread state on restart. A restart between
them shows up as `DeadThreadRefs` out of `assert_threads_alive`, so a run that
reaches gate 3 is evidence the choreography held. The Engine's server starts
only after the target app is down.

`benchmark/pipeline/fakes.py` stays where it is: CI cannot spend an LLM agent
and two servers per run, and `python -m benchmark.pipeline run --fake-harness
--fake-ablation --fake-engine` is the offline path. This script is the one
that pays for the real thing.

Usage:
    uv run python scripts/pipeline_smoke.py                 # manages both servers
    uv run python scripts/pipeline_smoke.py --no-serve      # servers already up
    uv run python scripts/pipeline_smoke.py --keep          # leave them running
    uv run python scripts/pipeline_smoke.py --fresh         # wipe prior artifacts
    uv run python scripts/pipeline_smoke.py --model gpt-5.1 # the other arm
    uv run python scripts/pipeline_smoke.py --target-port 2124 --engine-port 2125
    uv run python scripts/pipeline_smoke.py --engine-only  # skip the target app

`--engine-only` replays traces already in this run's TraceStore instead of
driving the target app, so the Engine / scoring / report / manifest seams can
be verified without LangSmith in the loop — useful when the trace-collection
backend is unavailable or its per-account rate limit is being shared with
another run. With no live target app there is nothing to inject against, so it
falls back to the stand-in ablation stage and the FAKED warning comes back. It
proves strictly less than a full pass and says so, twice.

`--target-port` / `--engine-port` exist because :2024 and :2025 are one machine's
worth of ports and more than one of these runs can want them — a parallel
worktree's dev server, or two arms of a model comparison side by side. They
move BOTH halves together: the env var each app's `serve.sh` reads, and a
port-swapped copy of that app's config written into the run directory, which
the pipeline is then pointed at. The checked-in configs are never edited.

Requires OPENAI_API_KEY and LANGSMITH_API_KEY (read from a repo-root .env).
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MINI_CONFIG = REPO_ROOT / "configs" / "pipeline" / "mini.yaml"
OUT_ROOT = REPO_ROOT / "data" / "pipeline_smoke"


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
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def port_of(base_url: str) -> int:
    return urlparse(base_url).port or 80


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    with socket.socket() as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


class StoredTraceHarness:
    """A `HarnessLike` that replays traces already in the TraceStore.

    `--engine-only`: the target app and LangSmith have already done their part
    in an earlier attempt, and this run only wants the stages after them. It is
    the same resumability the real harness has (an input with an ok trace is
    not re-run), reduced to its limit case — so it exercises the real stored
    artifacts, not fabricated ones.
    """

    def __init__(self, cfg, store):
        self.cfg = cfg
        self.store = store
        self.stats: dict[str, int] = {}

    def run_batch(self, inputs):
        from benchmark.harness.ids import session_id_for, trace_id_for  # noqa: PLC0415
        from benchmark.schemas.io import derive  # noqa: PLC0415
        from benchmark.schemas.traces import (  # noqa: PLC0415, E501
            OutputDataset,
            OutputRecord,
            TraceDataset,
        )

        traces, outputs, missing = [], [], []
        for spec in inputs.inputs:
            trace_id = trace_id_for(session_id_for(inputs.dataset_id, spec.input_id))
            if not self.store.exists(trace_id):
                missing.append(spec.input_id)
                continue
            trace = self.store.get(trace_id)
            traces.append(trace)
            outputs.append(
                OutputRecord(
                    input_id=spec.input_id,
                    trace_id=trace.trace_id,
                    responses=[t.final_response for t in trace.turns],
                )
            )
        if not traces:
            raise SystemExit(
                "BLOCKED: --engine-only found no stored traces for this config. Run the "
                "full smoke at least once so the TraceStore has something to replay."
            )
        self.stats = {"ran": 0, "skipped": len(traces), "quarantined": 0, "app_error": 0}
        if missing:
            print(f"  --engine-only: {len(missing)} input(s) have no stored trace: {missing}")
        return (
            derive(OutputDataset(outputs=outputs), inputs),
            derive(TraceDataset(traces=traces), inputs),
        )


def wait_for_port_closed(port: int, deadline: float = 15.0) -> bool:
    """A killed `langgraph dev` can hold its socket for a moment after exit."""
    until = time.time() + deadline
    while time.time() < until:
        if not port_open(port):
            return True
        time.sleep(0.5)
    return not port_open(port)


def app_config_on_port(source: Path, port: int, destination: Path) -> Path:
    """A copy of an app config with its `base_url` moved to `port`.

    Written into the run directory rather than over the checked-in config:
    `configs/target_app.yaml` and `configs/engine.yaml` are the shared black-box
    contracts, and a smoke run must not rewrite them to suit its own machine.
    """
    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(source.read_text()) or {}
    parsed = urlparse(raw["base_url"])
    raw["base_url"] = f"{parsed.scheme}://{parsed.hostname}:{port}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(raw, sort_keys=False))
    return destination


def main() -> int:  # noqa: PLR0915 - a linear gate script reads better in one piece
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-serve", action="store_true", help="assume both servers are up")
    parser.add_argument("--keep", action="store_true", help="leave the servers running")
    parser.add_argument("--fresh", action="store_true", help="wipe prior smoke artifacts")
    parser.add_argument("--model", default=None, help="Engine model (default: the config's)")
    parser.add_argument("--target-port", type=int, default=None, help="move the target app")
    parser.add_argument("--engine-port", type=int, default=None, help="move the Engine")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="harness concurrency (lower it when sharing a LangSmith quota)",
    )
    parser.add_argument(
        "--engine-only",
        action="store_true",
        help=(
            "replay traces already in this run's TraceStore instead of driving the "
            "target app — isolates the Engine/scoring/report seams from LangSmith"
        ),
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    for required in ("OPENAI_API_KEY", "LANGSMITH_API_KEY"):
        if not os.environ.get(required):
            print(f"BLOCKED: {required} is not set (copy .env to the repo root)")
            return 2
    os.environ.setdefault("LANGSMITH_TRACING", "true")

    from benchmark.generation.expander import OpenAIPromptExpander
    from benchmark.harness.config import load_target_app_config
    from benchmark.pipeline.config import load_engine_app_config, load_pipeline_config
    from benchmark.pipeline.contracts import assert_ablation_result, load_ablation_stage
    from benchmark.pipeline.export import export_traces
    from benchmark.pipeline.fakes import fake_run_ablation
    from benchmark.pipeline.runner import run_pipeline
    from benchmark.pipeline.servers import ServerLifetime
    from benchmark.schemas.io import content_hash

    if args.fresh and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    cfg = load_pipeline_config(MINI_CONFIG)
    overrides: dict = {"run_id": "smoke", "artifacts_root": str(OUT_ROOT)}
    if args.model:
        overrides["engine"] = cfg.engine.model_copy(update={"model": args.model})
    if args.concurrency:
        overrides["harness"] = cfg.harness.model_copy(update={"concurrency": args.concurrency})
    cfg = cfg.model_copy(update=overrides).with_root(cfg.root)

    # Port moves, if asked for: the env var each serve.sh reads, plus a
    # port-swapped copy of the app config the pipeline is pointed at.
    port_overrides: dict = {}
    if args.target_port:
        os.environ["TARGET_APP_PORT"] = str(args.target_port)
        port_overrides["target_app_config"] = str(
            app_config_on_port(
                cfg.resolve(cfg.target_app_config),
                args.target_port,
                cfg.run_dir / "target_app.port.yaml",
            )
        )
    if args.engine_port:
        os.environ["ENGINE_PORT"] = str(args.engine_port)
        port_overrides["engine_app_config"] = str(
            app_config_on_port(
                cfg.resolve(cfg.engine_app_config),
                args.engine_port,
                cfg.run_dir / "engine.port.yaml",
            )
        )
    if port_overrides:
        cfg = cfg.model_copy(update=port_overrides).with_root(cfg.root)

    target_app = load_target_app_config(cfg.resolve(cfg.target_app_config))
    engine_app = load_engine_app_config(cfg.resolve(cfg.engine_app_config))
    target_port, engine_port = port_of(target_app.base_url), port_of(engine_app.base_url)

    banner("PHASE 7 LIVE SMOKE — miniature end-to-end, both servers, real Engine")
    print(f"pipeline config : {MINI_CONFIG.relative_to(REPO_ROOT)}")
    print(f"target app      : {target_app.base_url} ({target_app.assistant_id})")
    print(f"engine          : {engine_app.base_url} ({engine_app.assistant_id})")
    print(f"engine model    : {cfg.engine.model}")
    print(f"engine knobs    : analysis_concurrency={cfg.engine.analysis_concurrency}, "
          f"recursion_limit={cfg.engine.recursion_limit}")
    print(f"artifacts       : {cfg.run_dir}")

    # --engine-only has no live target app, so there is nothing for the real
    # stage to replay or fault-inject against; it gets the stand-in, and the
    # FAKED warning that comes with it.
    if args.engine_only:
        real_stage = fake_run_ablation
        print("ablation stage  : FAKE (--engine-only: no target app to inject against)")
    else:
        real_stage = load_ablation_stage()
        print(f"ablation stage  : REAL ({real_stage.__module__}.{real_stage.__qualname__})")

    # The runner already runs `assert_ablation_result` at the seam, but it does
    # not hand the raw result back, and gate 2 wants to check the object Phase 5
    # returned rather than the fields the runner chose to keep. `functools.wraps`
    # keeps the wrapped function's `__module__`/`__qualname__`, so the manifest
    # still records `benchmark.ablation.engine.run_ablation` — which is the
    # truth: this closure adds an observation, not an implementation.
    captured: dict = {}

    @functools.wraps(real_stage)
    def observed_stage(**kwargs):
        result = real_stage(**kwargs)
        captured["result"] = result
        return result

    # --engine-only owns the Engine's server and nothing else: the target app
    # is not started because it is not driven.
    managed = (
        {k: v for k, v in cfg.servers.items() if k == "engine"}
        if args.engine_only
        else cfg.servers
    )
    servers = ServerLifetime(cfg.root, managed, enabled=not args.no_serve)
    if not args.no_serve:
        checked = [("engine", engine_port)]
        if not args.engine_only:
            checked.insert(0, ("target app", target_port))
        for name, port in checked:
            if port_open(port):
                print(f"\nNOTE: something is already listening on :{port} ({name}). "
                      f"Re-run with --no-serve, or stop it first.")
                return 2

    started = time.time()
    run = run_pipeline(
        cfg,
        ablation_stage=observed_stage,
        expander=OpenAIPromptExpander(),
        servers=servers,
        harness_factory=StoredTraceHarness if args.engine_only else None,
    )
    elapsed = time.time() - started

    # --------------------------------------------------------------- gate 1
    banner("GATE 1 — inputs -> traces through the real target app")
    print(f"inputs : {len(run.inputs.inputs)} ({run.inputs.dataset_id})")
    print(f"traces : {len(run.traces.traces)} ({run.traces.dataset_id})")
    print(f"harness stats: {run.manifest.harness_stats}")
    assert 1 <= len(run.inputs.inputs) <= 10, len(run.inputs.inputs)
    assert run.traces.parent_dataset_id == run.inputs.dataset_id, "lineage broken"
    if args.engine_only:
        print("  (--engine-only: traces replayed from the store, not freshly collected)")
    else:
        assert len(run.traces.traces) == len(run.inputs.inputs), "an input produced no trace"
    for trace in run.traces.traces:
        assert trace.turns, f"{trace.trace_id} has no turns"
        assert any(t.spans for t in trace.turns), f"{trace.trace_id} has no spans"
    print("PASS — every trace is schema-valid with spans, lineage intact")

    # --------------------------------------------------------------- gate 2
    banner("GATE 2 — real ablation -> injected ground truth + leak-stripped export")
    result = captured["result"]
    print(f"stage    : {type(result).__module__}.{type(result).__name__}")
    print(f"ablated  : {len(run.ablated.traces)} traces ({run.ablated.dataset_id}) "
          f"<- parent {run.ablated.parent_dataset_id}")
    print(f"split    : {len(run.split.control_input_ids)} control / "
          f"{len(run.split.ablate_input_ids)} ablate, seed={run.split.seed}, "
          f"strata={run.split.strata}")
    print(f"E_K      : {len(run.ground_truth.issues)} issue(s), "
          f"{len(run.ground_truth.occurrences)} occurrence(s)")
    for issue in run.ground_truth.issues:
        print(f"  {issue.error_id}: [{issue.injection_mode}] {issue.category_id} "
              f"/ {issue.severity} — {issue.title}")
    print(f"records  : {len(run.records)}")
    print(f"dropped  : {run.manifest.dropped_errors}")
    print(f"export   : {run.export_path}")

    # The pinned contract, re-checked on the object Phase 5 actually returned.
    assert_ablation_result(result)
    print(f"  assert_ablation_result: OK on {type(result).__name__}")

    # The path the stage CLAIMS it wrote is the path the Engine will be given.
    assert Path(result.export_path) == run.export_path, (
        f"the stage reported {result.export_path!r} but the run used {run.export_path}"
    )
    assert run.export_path.exists(), "the claimed export path is not on disk"
    assert run.ablated.parent_dataset_id == run.traces.dataset_id, "lineage broken"

    # Leak audit, run again here with the ablation package's OWN auditor (field
    # allowlist + token scan) rather than only the pipeline's token scan. Two
    # implementations of the same boundary, and the export has to survive both.
    from benchmark.ablation.export import EXPORT_EPOCH, audit_export  # noqa: PLC0415

    blob = run.export_path.read_text()
    exported = export_traces(json.loads(blob))
    audit_export(exported, target_app)
    for token in ("ablation_ids", "injection_mode", "replay_edit", "dependency_fault", "fault_"):
        assert token not in blob, f"the Engine's trace file names {token!r}"
    print(f"  leak audit: OK over {len(exported)} exported trace(s) "
          f"(allowlist + token scan, both auditors)")

    # No time separator: ablation runs after collection, so an un-normalized
    # export would let the Engine sort control from ablated by the clock alone
    # — no trace reading required. Every exported trace must start at the same
    # synthetic origin.
    # PARSED, not compared as strings: pydantic drops a zero microsecond field,
    # so the epoch serializes as "...T00:00:00Z" while its neighbours keep a
    # ".00xxxx" — and "." sorts before "Z", which makes a lexicographic min
    # pick the wrong span and report a violation that is not there.
    origins = {
        min(
            datetime.fromisoformat(span["start_time"].replace("Z", "+00:00"))
            for turn in trace["turns"]
            for span in turn["spans"]
        )
        for trace in exported
        if any(turn["spans"] for turn in trace["turns"])
    }
    assert len(origins) <= 1, f"exported traces start at {len(origins)} distinct clocks: {origins}"
    if origins:
        origin = origins.pop()
        assert origin == EXPORT_EPOCH, (
            f"the export's origin is {origin}, not the fixed EXPORT_EPOCH {EXPORT_EPOCH}"
        )
        print(f"  no time separator: all {len(exported)} trace(s) re-based to {origin}")

    # And the inverse of the assertion this gate used to make: a real ablation
    # CHANGES the corpus. A pass-through would leave the bytes identical.
    source_by_id = {t.trace_id: t for t in run.traces.traces}
    changed = [
        t.trace_id
        for t in run.ablated.traces
        if t.trace_id not in source_by_id
        or t.model_dump(mode="json", exclude={"ablation_ids"})
        != source_by_id[t.trace_id].model_dump(mode="json", exclude={"ablation_ids"})
    ]
    assert changed, (
        "the ablated corpus is byte-identical to the collected one — nothing was injected"
    )
    print(f"  {len(changed)} of {len(run.ablated.traces)} trace(s) differ from the collected "
          f"corpus (a pass-through stage would differ in none)")
    print(f"  injection modes in E_K: "
          f"{sorted({i.injection_mode for i in run.ground_truth.issues if i.injection_mode})}")
    print("PASS — real injection, contract honoured, export written and leak-free")

    # --------------------------------------------------------------- gate 3
    banner("GATE 3 — real Engine run over the export")
    print(f"engine seconds        : {run.engine.seconds:.1f}")
    print(f"server-side model     : {run.engine.recorded_models}")
    print(f"raw board_id (engine) : {run.engine.raw_output.get('board_id')}")
    print(f"restamped board_id    : {run.predicted.board_id}")
    print(f"updated board (as returned): {len(run.predicted.issues)} issue(s), "
          f"{len(run.predicted.occurrences)} occurrence(s)")
    print(f"scored delta          : {len(run.scored.board.issues)} issue(s), "
          f"{len(run.scored.board.occurrences)} occurrence(s) over "
          f"{len({o.trace_id for o in run.scored.board.occurrences})} of "
          f"{len(run.ablated.traces)} traces")
    print(f"  seed carriers={run.scored.carrier_error_ids} "
          f"dropped_seed_issues={run.scored.dropped_seed_issues} "
          f"dropped_seed_pairs={run.scored.dropped_seed_occurrences} "
          f"phantoms={run.scored.phantom_trace_ids}")
    assert run.predicted.source == "engine_predicted"
    assert run.predicted.board_id == content_hash(run.predicted), "board_id was not re-stamped"
    assert run.predicted.board_id != run.engine.raw_output.get("board_id"), (
        "the Engine's own board_id survived — it is not byte-compatible with ours"
    )
    seed_ids = {i.error_id for i in run.seed_board.issues}
    assert seed_ids <= {i.error_id for i in run.predicted.issues}, "seed issues were dropped"
    if run.engine.recorded_models:
        # run_pipeline already hard-fails on a mismatch; this reports the fact.
        assert set(run.engine.recorded_models) == {cfg.engine.model}, run.engine.recorded_models
        print(f"  server-side readback confirms model={cfg.engine.model!r}")
    else:
        print("  WARNING: the server kept no readable record of the run's model")
    if not run.scored.board.occurrences:
        print("  WARNING: the Engine returned no occurrences — check its stderr for "
              "per-trace analysis failures (they are stderr-only)")
    print("PASS — schema-valid updated issueboard, re-stamped, seed carried through")

    # --------------------------------------------------------------- gate 4
    banner("GATE 4 — report + manifest on disk")
    for name in (
        "report.json",
        "report.md",
        "manifest.json",
        "deliverables.json",
        "scored_issueboard.json",
    ):
        path = run.run_dir / name
        assert path.exists(), f"{name} was not written"
        print(f"  {name}: {path.stat().st_size} bytes")
    assert run.report.report_id
    assert run.report.base_rates["n_traces"] == len(run.ablated.traces)
    assert run.manifest.dataset_ids["report"] == run.report.report_id
    assert run.manifest.config_hashes["pipeline"]
    assert {t.stage for t in run.manifest.timings} >= {
        "generation", "harness", "ablation", "engine", "scoring"
    }
    print(f"\nmanifest stages: {run.manifest.stages}")
    print(f"manifest timings: "
          f"{ {t.stage: round(t.seconds, 1) for t in run.manifest.timings} }")
    print(f"manifest warnings: {run.manifest.warnings}")
    print(f"base_rates: control_fraction={run.report.base_rates['control_fraction']}, "
          f"injection_modes={run.report.base_rates['injection_modes']}, "
          f"per_error_injection_counts={run.report.base_rates['per_error_injection_counts']}")
    assert run.report.base_rates["injection_modes"], (
        "no injection mode was recorded — the report cannot say how errors were planted"
    )
    assert run.split.strata, "the split records no strata — it is not provenance-based"

    faked = [w for w in run.manifest.warnings if "FAKED" in w]
    if args.engine_only:
        assert faked, "--engine-only uses the stand-in ablation stage and must say so"
        print(f"  --engine-only: FAKED warning present as expected — {faked}")
    else:
        # THE signal that this run was real. Every stand-in in the assembly
        # sets it, so its absence is not something that can be arranged.
        assert not faked, f"a stage was faked: {faked}"
        assert run.manifest.stages["ablation"] == "benchmark.ablation.engine.run_ablation", (
            f"the manifest records {run.manifest.stages['ablation']!r} as the ablation stage"
        )
        print("  no FAKED warning — every stage in this run was the real one")
    print("PASS — BenchmarkReport, markdown summary and manifest written with lineage")

    # --------------------------------------------------------------- gate 5
    banner("GATE 5 — assignment deliverables")
    for check in run.deliverables:
        print(f"  [{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    failed = [c.name for c in run.deliverables if not c.ok]
    assert not failed, f"failed deliverables: {failed}"
    print("PASS — every deliverable check green at miniature scale")

    # --------------------------------------------------------------- gate 6
    banner("GATE 6 — both servers cleanly stopped")
    if args.no_serve or args.keep:
        print("  skipped (--no-serve/--keep: this run does not own the servers)")
    else:
        owned = [("engine", engine_port)]
        if not args.engine_only:
            owned.insert(0, ("target app", target_port))
        for name, port in owned:
            assert wait_for_port_closed(port), (
                f"the {name} server is still listening on :{port}"
            )
            print(f"  {name} (:{port}) is down")
        print(f"PASS — {len(owned)} server(s) started and stopped by the pipeline")

    banner("REPORT")
    print(run.markdown)

    banner(f"PHASE 7 SMOKE OK — every gate green ({elapsed:.1f}s wall clock)")
    print(f"artifacts under {run.run_dir}")
    print("\nper-stage seconds (the basis for the submission-run estimate):")
    for timing in run.manifest.timings:
        print(f"  {timing.stage:<12} {timing.seconds:8.1f}s")
    if args.engine_only:
        print(
            "\nREMINDER: --engine-only replayed stored traces and FAKED the ablation "
            "stage. This proves the Engine/scoring/report seams only."
        )
    else:
        print(
            "\nThis was a fully integrated run: real generation, real harness, real "
            "ablation, real Engine. The numbers are still miniature-scale — seven "
            "inputs is a wiring corpus, not a measurement — but nothing in the chain "
            "was a stand-in."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
