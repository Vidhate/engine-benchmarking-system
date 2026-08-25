"""CLI: one command turns configs into a `BenchmarkReport`.

    uv run python -m benchmark.pipeline run   --config configs/pipeline/mini.yaml
    uv run python -m benchmark.pipeline score --run  data/pipeline/mini
    uv run python -m benchmark.pipeline check --run  data/pipeline/mini

* `run`   — the whole pipeline. Manages both app servers by default (each via
            its own `scripts/serve.sh`); `--no-serve` assumes they are already
            up. Every stage is real by default: a missing Phase 5 is a loud
            failure, never a silent downgrade.

            The three `--fake-*` flags substitute the CI doubles from
            `benchmark.pipeline.fakes`, one per external seam, and every one of
            them stamps a FAKED warning onto the manifest and the report:

                --fake-harness   no target app, no LangSmith, and no OpenAI
                                 call to expand prompts either (the canned
                                 traces never read the prompt text, so a real
                                 expansion would be a paid call thrown away).
                                 REQUIRES --fake-ablation: the real ablation
                                 stage replays and fault-injects through the
                                 harness, which the fake cannot do.
                --fake-ablation  no LLM ablation agent, no injection
                --fake-engine    no Engine server

            All three together is the offline end-to-end run — no servers, no
            keys, no network:

                python -m benchmark.pipeline run \\
                    --config configs/pipeline/mini.yaml \\
                    --fake-harness --fake-ablation --fake-engine

            `--resume <run_dir>` picks a killed run back up. Each of
            generation / harness / ablation / engine is skipped when that
            directory already holds its artifacts AND they are the ones this
            config would have produced; everything else runs, and scoring and
            rendering always do. Skipped stages print
            `↻ stage (resumed from disk)` and are named in the manifest and the
            report header. A config that does not match the directory's own
            record of itself is refused outright rather than mixed into it —
            see benchmark/pipeline/resume.py.

                python -m benchmark.pipeline run \\
                    --config configs/pipeline/submission.yaml \\
                    --resume data/pipeline/submission
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
from benchmark.pipeline.fakes import (
    FakeEngineInvoker,
    FakeExpander,
    FakeHarnessFactory,
    fake_run_ablation,
)
from benchmark.pipeline.progress import Progress
from benchmark.pipeline.resume import ResumeMismatch
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
        "--fake-harness",
        action="store_true",
        help=(
            "canned traces instead of the target app + LangSmith, and a network-free "
            "prompt expander with them (benchmark.pipeline.fakes). Requires "
            "--fake-ablation"
        ),
    )
    run.add_argument(
        "--fake-ablation",
        action="store_true",
        help="no injection: the pass-through stand-in (benchmark.pipeline.fakes)",
    )
    run.add_argument(
        "--fake-engine",
        action="store_true",
        help="a canned issueboard instead of the Engine server (benchmark.pipeline.fakes)",
    )
    run.add_argument(
        "--resume",
        metavar="RUN_DIR",
        help=(
            "reuse a previous run's artifacts. Each of generation/harness/ablation/engine "
            "is skipped when RUN_DIR already holds its artifacts AND they are the ones "
            "this config would have produced; everything else runs. Scoring and rendering "
            "always re-run. Hard-fails if the config does not match RUN_DIR's own record "
            "of itself"
        ),
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
    if args.resume:
        # `--resume` takes a run DIRECTORY, which is `<artifacts_root>/<run_id>`.
        # Splitting it back into those two fields is what makes the rest of the
        # pipeline — every `cfg.run_dir` in the runner — point at it, with no
        # second notion of "where the artifacts are" to keep in sync.
        # `main()` has already refused the combination with --run-id/--artifacts-root.
        resume_dir = Path(args.resume).resolve()
        if not resume_dir.is_dir():
            print(f"BLOCKED: --resume {args.resume} is not a directory", file=sys.stderr)
            return 3
        overrides["artifacts_root"] = str(resume_dir.parent)
        overrides["run_id"] = resume_dir.name
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
    else:
        try:
            stage = load_ablation_stage()
        except AblationStageUnavailable as exc:
            print(f"BLOCKED: {exc}", file=sys.stderr)
            return 3

    # `None` means "the runner builds the real one". The fakes are only ever
    # reached through an explicit flag, so no combination of defaults can
    # produce a faked run.
    harness_factory = FakeHarnessFactory() if args.fake_harness else None
    # The fake harness fabricates its traces and never reads the prompt text,
    # so expanding prompts for real would be a paid OpenAI call whose output is
    # discarded — and it would also stop this being an offline run.
    expander = FakeExpander() if args.fake_harness else None
    if expander is not None:
        # ...but the expansion cache is keyed on (config hash, dim, variation,
        # persona, seed) and NOT on which expander produced the text. Left
        # pointing at the shared cache, one offline run would write
        # "[topic/x] please help me with x" under exactly the keys the real
        # expander uses, and the next real run would silently generate its
        # entire corpus from those. Fake expansions get their own scratch
        # directory inside the run, so the shared cache is read-only here.
        cfg = cfg.model_copy(
            update={"expansion_cache": str(cfg.run_dir / "fake_expansion_cache")}
        ).with_root(cfg.root)
    engine_invoker = FakeEngineInvoker() if args.fake_engine else None

    faked = [
        name
        for name, on in (
            ("harness (+ prompt expansion)", args.fake_harness),
            ("ablation", args.fake_ablation),
            ("engine", args.fake_engine),
        )
        if on
    ]
    if faked:
        print(
            f"WARNING: running with FAKE stage(s): {', '.join(faked)}. Development "
            f"stand-ins produce evidence about wiring, never about the Engine — the "
            f"manifest and the report say so too.",
            file=sys.stderr,
        )

    # A faked seam has no server to talk to, so the run does not start one:
    # `--fake-harness --fake-ablation --fake-engine` starts nothing at all,
    # which is what makes the offline run offline rather than merely unused.
    unmanaged = set()
    if args.fake_harness:
        # Guarded in main(): --fake-harness implies --fake-ablation, so nothing
        # in this run talks to the target app.
        unmanaged.add("target_app")
    if args.fake_engine:
        unmanaged.add("engine")
    managed = {k: v for k, v in cfg.servers.items() if k not in unmanaged}

    servers = ServerLifetime(cfg.root, managed, enabled=not args.no_serve)
    progress = Progress(quiet=args.quiet)
    try:
        run = run_pipeline(
            cfg,
            ablation_stage=stage,
            engine_invoker=engine_invoker,
            harness_factory=harness_factory,
            expander=expander,
            servers=servers,
            progress=progress,
            resume=bool(args.resume),
        )
    except ResumeMismatch as exc:
        # Never a traceback: this is a user-facing "you pointed it at the wrong
        # directory", and the message already names the hashes that differ.
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3

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
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "resume", None) and (args.run_id or args.artifacts_root):
        # `--resume` sets both of those from the directory it is given. Letting
        # a second source of truth win would resume artifacts out of one
        # directory and write the report into another.
        parser.error(
            "--resume already determines the run directory (and with it --run-id and "
            "--artifacts-root); pass one or the other, not both"
        )
    if getattr(args, "fake_harness", False) and not args.fake_ablation:
        # The two fakes are not independent. `--fake-harness` alone leaves the
        # REAL benchmark.ablation.run_ablation driving the stand-in harness,
        # and the real stage needs a live target app: it forks threads
        # (locate_checkpoint, turn_boundaries), replays them (replay) and
        # re-runs traces under a fault shim (run_with_faults,
        # activation_evidence). FakeHarness implements none of those — it only
        # knows run_batch — so the run dies on a raw AttributeError several
        # minutes in, after generation and the batch have already been paid
        # for. Refused up front instead.
        parser.error(
            "--fake-harness requires --fake-ablation: the real ablation stage drives "
            "the harness through replay/run_with_faults/turn_boundaries, and the fake "
            "harness only implements run_batch. Add --fake-ablation for a fully "
            "offline run, or drop --fake-harness to ablate against the real target app."
        )
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return {"run": _cmd_run, "score": _cmd_score, "check": _cmd_check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
