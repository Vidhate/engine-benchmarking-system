"""CLI: one command turns configs into a `BenchmarkReport`.

    uv run python -m benchmark.pipeline run   --config configs/pipeline/mini.yaml
    uv run python -m benchmark.pipeline score --run  data/pipeline/mini
    uv run python -m benchmark.pipeline check --run  data/pipeline/mini

* `run`   — the whole pipeline. Manages both app servers by default (each via
            its own `scripts/serve.sh`); `--no-serve` assumes they are already
            up. `--fake-ablation` substitutes the Phase-5 stand-in and says so
            in the report; without it, Phase 5's real `run_ablation` is
            imported and a missing Phase 5 is a loud failure, never a silent
            downgrade.
* `score` — the standalone scoring entrypoint: re-runs `score()` over a
            finished run's artifacts, reading nothing but files.
* `check` — the assignment-deliverables checklist over a finished run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from benchmark.pipeline.config import find_root, load_pipeline_config
from benchmark.pipeline.contracts import AblationStageUnavailable, load_ablation_stage
from benchmark.pipeline.deliverables import check_deliverables, rescore_from_disk
from benchmark.pipeline.fakes import fake_run_ablation
from benchmark.pipeline.progress import Progress
from benchmark.pipeline.runner import run_pipeline
from benchmark.pipeline.servers import ServerLifetime


def load_dotenv(path: Path) -> None:
    """Read a repo-root .env into the environment without overwriting it."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark.pipeline", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="configs -> BenchmarkReport")
    run.add_argument("--config", required=True, help="path to a configs/pipeline/*.yaml")
    run.add_argument("--run-id", help="override the config's run_id (artifact directory name)")
    run.add_argument("--artifacts-root", help="override where artifacts are written")
    run.add_argument("--engine-model", help="override the Engine model (the comparison axis)")
    run.add_argument(
        "--no-serve",
        action="store_true",
        help="do not start/stop the app servers — they are already running",
    )
    run.add_argument(
        "--fake-ablation",
        action="store_true",
        help="use the Phase-5 stand-in (benchmark.pipeline.fakes.fake_run_ablation)",
    )
    run.add_argument(
        "--quiet",
        action="store_true",
        help="suppress stage-progress lines on stderr (banners, heartbeats, per-item counts)",
    )

    score = sub.add_parser("score", help="re-score a finished run from its artifacts")
    score.add_argument("--run", required=True, help="a run directory")

    check = sub.add_parser("check", help="assignment-deliverables checklist over a run")
    check.add_argument("--run", required=True, help="a run directory")
    check.add_argument("--min-traces", type=int, default=300)
    return parser


def _cmd_run(args) -> int:
    config_path = Path(args.config)
    root = find_root(config_path)
    load_dotenv(root / ".env")

    cfg = load_pipeline_config(config_path)
    overrides = {}
    if args.run_id:
        overrides["run_id"] = args.run_id
    if args.artifacts_root:
        overrides["artifacts_root"] = args.artifacts_root
    if args.engine_model:
        overrides["engine"] = cfg.engine.model_copy(update={"model": args.engine_model})
    if overrides:
        cfg = cfg.model_copy(update=overrides).with_root(cfg.root)

    if args.fake_ablation:
        stage = fake_run_ablation
        print(
            "WARNING: running with the FAKE ablation stage — traces are not modified and "
            "the ground truth is synthetic. Wiring evidence only.",
            file=sys.stderr,
        )
    else:
        try:
            stage = load_ablation_stage()
        except AblationStageUnavailable as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            return 3

    servers = ServerLifetime(cfg.root, cfg.servers, enabled=not args.no_serve)
    progress = Progress(quiet=args.quiet)
    run = run_pipeline(cfg, ablation_stage=stage, servers=servers, progress=progress)

    print(run.markdown)
    print(f"artifacts: {run.run_dir}")
    failed = [c for c in run.deliverables if not c.ok]
    for check in failed:
        print(f"DELIVERABLE FAILED — {check.name}: {check.detail}", file=sys.stderr)
    return 1 if failed else 0


def _cmd_score(args) -> int:
    report = rescore_from_disk(args.run)
    print(json.dumps(report.model_dump(mode="json")["headline"], indent=2))
    return 0


def _cmd_check(args) -> int:
    checks = check_deliverables(args.run, min_traces=args.min_traces)
    for check in checks:
        print(f"[{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    return 0 if all(c.ok for c in checks) else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return {"run": _cmd_run, "score": _cmd_score, "check": _cmd_check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
