#!/usr/bin/env python3
"""Live miniature end-to-end run of the Phase 7 pipeline. NOT run by scripts/ci.sh.

Drives the REAL path with BOTH real servers and the real Engine on the mini
model, over ~7 inputs:

    generate_inputs (real OpenAI expander)
      -> harness batch      (real target app on :2024, real LangSmith collection)
      -> [FAKE ablation]    (Phase 5 has not merged — see below)
      -> Engine             (real engine app on :2025, mini model)
      -> score()            -> BenchmarkReport + report.md + manifest.json

It proves every integration seam EXCEPT ablation.

--------------------------------------------------------------------------
WHEN PHASE 5 MERGES: ONE LINE CHANGES.
--------------------------------------------------------------------------
Replace

    from benchmark.pipeline.fakes import fake_run_ablation
    ...
    ablation_stage=fake_run_ablation,

with

    from benchmark.pipeline.contracts import load_ablation_stage
    ...
    ablation_stage=load_ablation_stage(),

and drop `--fake-ablation` from the CLI invocation. Nothing else in this
script, and nothing in `benchmark/pipeline/`, needs to change: the pipeline
calls the stage through the pinned contract
(`run_ablation(traces, inputs, categories, cfg, harness, store, export_path)`)
and re-checks the returned shape at the seam. The only assertions below that
loosen are the ones marked FAKE-ONLY.

Note the ordering the pipeline enforces and this script depends on: the
harness batch and the ablation stage run inside ONE target-app server
lifetime, because Mode-A replay forks a LangGraph thread created during the
batch and `langgraph dev` loses thread state on restart. The Engine's server
starts only after the target app is down.

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
another run. It proves strictly less than a full pass and says so.

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
import json
import os
import shutil
import socket
import sys
import time
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

    # FAKE-ONLY IMPORT — see the header: swap for `load_ablation_stage()`.
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
    print("ablation stage  : FAKE (Phase 5 not merged) — see this script's header")

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
        # ---- THE ONE LINE THAT CHANGES WHEN PHASE 5 MERGES --------------
        ablation_stage=fake_run_ablation,
        # -----------------------------------------------------------------
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
    banner("GATE 2 — ablation stage (FAKE) -> ground truth + leak-stripped export")
    print(f"ablated  : {len(run.ablated.traces)} traces ({run.ablated.dataset_id}) "
          f"<- parent {run.ablated.parent_dataset_id}")
    print(f"split    : {len(run.split.control_input_ids)} control / "
          f"{len(run.split.ablate_input_ids)} ablate, seed={run.split.seed}")
    print(f"E_K      : {len(run.ground_truth.issues)} issue(s), "
          f"{len(run.ground_truth.occurrences)} occurrence(s)")
    print(f"export   : {run.export_path}")
    assert run.ablated.parent_dataset_id == run.traces.dataset_id
    assert run.export_path.exists()
    blob = run.export_path.read_text()
    for token in ("ablation_ids", "injection_mode", "replay_edit", "dependency_fault", "fault_"):
        assert token not in blob, f"the Engine's trace file names {token!r}"
    # FAKE-ONLY: a pass-through stage leaves the trace bytes untouched. With
    # the real Phase 5 stage this equality does NOT hold (that is the point of
    # ablation) — delete this assertion at integration time.
    assert [t["trace_id"] for t in json.loads(blob)["traces"]] == [
        t.trace_id for t in run.traces.traces
    ], "FAKE-ONLY invariant: the stand-in must pass traces through unchanged"
    print("PASS — export written, leak-free, ground truth + split recorded")

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
    print(f"\nmanifest timings: "
          f"{ {t.stage: round(t.seconds, 1) for t in run.manifest.timings} }")
    print(f"manifest warnings: {run.manifest.warnings}")
    assert any("FAKED" in w for w in run.manifest.warnings), (
        "a faked ablation stage must be recorded in the manifest"
    )
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
    print(
        "\nREMINDER: the ablation stage was FAKED. Scores here are evidence about "
        "wiring, not about the Engine. Swap in benchmark.ablation.run_ablation "
        "(one line, see this script's header) once Phase 5 merges."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
