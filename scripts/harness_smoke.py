#!/usr/bin/env python3
"""Live end-to-end verification of the Phase 4 harness. NOT run by scripts/ci.sh.

Manages the target app's `langgraph dev` server itself (via the app's own
`scripts/serve.sh` — a shell call, never an import), then walks every Phase 4
gate against the real app and the real LangSmith project:

  1. batch      — configs/generation/mini.yaml -> inputs -> traces in a
                  TraceStore: schema-valid, correct lineage, no leaks.
  2. multi-turn — a persona-simulator conversation collected as ONE trace with
                  M >= 2 turns, each carrying its own spans.
  3. rerun      — the same batch again skips every input that already has an
                  ok trace (idempotency).
  4. faults     — run_with_faults arms the declared retriever key and the
                  regenerated retrieval span is visibly corrupted vs baseline.
  5. replay     — fork a thread at an edited checkpoint and continue coherently.
  6. leak audit — no fault/shim token and no fault_* metadata key in any stored
                  trace, plus the live `serialized` manifest anti-vacuity check.

Usage:
    uv run python scripts/harness_smoke.py            # manages the server
    uv run python scripts/harness_smoke.py --no-serve # server already running
    uv run python scripts/harness_smoke.py --keep     # leave the server up

Requires OPENAI_API_KEY and LANGSMITH_API_KEY (read from a repo-root .env).
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
OUT_DIR = REPO_ROOT / "data" / "harness_smoke"

CASE_REFERENCE = "NBX-4471"
CORRUPTED_ANSWER = (
    "Annual plans are refundable in full within 365 days of the charge date, and refunds "
    "are paid out in store credit rather than to your card. I have logged this "
    f"conversation under case reference {CASE_REFERENCE}."
)
REPLAY_QUESTION = (
    "Sorry, I got distracted and did not read that. Without looking anything up again, "
    "remind me of the case reference you gave me and the refund window in days you quoted."
)


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


def main() -> int:  # noqa: PLR0915 - a linear gate script reads better in one piece
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-serve", action="store_true", help="assume the server is up")
    parser.add_argument("--keep", action="store_true", help="leave the server running")
    parser.add_argument("--fresh", action="store_true", help="wipe the smoke trace store first")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    for required in ("OPENAI_API_KEY", "LANGSMITH_API_KEY"):
        if not os.environ.get(required):
            print(f"BLOCKED: {required} is not set (copy .env to the repo root)")
            return 2
    os.environ.setdefault("LANGSMITH_TRACING", "true")

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
        session_id_for,
    )
    from benchmark.harness.scrub import find_leaked_keys, find_leaks, leak_tokens
    from benchmark.schemas.ablation import FaultConfig
    from benchmark.schemas.inputs import InputDataset
    from benchmark.schemas.io import save, stamp_dataset_id
    from benchmark.tracing.store import LocalTraceStore

    if args.fresh and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_target_app_config(REPO_ROOT / "configs" / "target_app.yaml")
    print(f"target app contract: {cfg.model_dump_json()}")

    if not args.no_serve:
        serve("start")
    try:
        # ------------------------------------------------------------ inputs
        banner("GATE 1a — mini generation config -> InputDataset")
        gen_cfg = load_generation_config(MINI_CONFIG)
        full = generate_inputs(
            gen_cfg, OpenAIPromptExpander(), cache_dir=REPO_ROOT / "data" / "expansion_cache"
        )
        single = [i for i in full.inputs if i.mode == "single_turn"]
        multi = [i for i in full.inputs if i.mode == "multi_turn"]
        # One target-persona and one adversarial-persona conversation.
        sliced = single + [multi[0], multi[-1]]
        inputs = stamp_dataset_id(
            InputDataset(
                created_at=full.created_at, generation_config=full.generation_config,
                inputs=sliced,
            )
        )
        save(inputs, OUT_DIR / "inputs.json")
        print(f"inputs: {len(single)} single-turn + 2 multi-turn, dataset_id={inputs.dataset_id}")
        print(f"  sample prompt: {single[0].prompt[:110]!r}")
        print(f"  sample scenario: {multi[0].scenario[:110]!r}")

        # ------------------------------------------------------------- batch
        banner("GATE 1b — batch run: inputs -> traces in the TraceStore")
        store = LocalTraceStore(OUT_DIR / "traces")
        quarantine = Quarantine(OUT_DIR / "quarantine")
        collector = LangSmithCollector(cfg.langsmith_project, cfg=cfg)
        harness = Harness(
            cfg,
            store,
            client=LangGraphAppClient(cfg),
            collector=collector,
            simulator=OpenAIUserSimulator(),
            quarantine=quarantine,
            concurrency=4,
        )
        started = time.time()
        outputs, traces = harness.run_batch(inputs)
        print(f"stats: {harness.stats}  ({time.time() - started:.1f}s)")
        save(traces, OUT_DIR / "traces.json")
        save(outputs, OUT_DIR / "outputs.json")

        assert not quarantine.list_ids(), f"quarantined: {quarantine.list_ids()}"
        assert len(traces.traces) == len(inputs.inputs), (
            f"{len(traces.traces)}/{len(inputs.inputs)} inputs produced a trace"
        )
        assert traces.parent_dataset_id == inputs.dataset_id
        assert outputs.parent_dataset_id == inputs.dataset_id
        for trace in traces.traces:
            assert store.exists(trace.trace_id)
            assert trace.turns and all(t.spans for t in trace.turns)
            expected = session_id_for(inputs.dataset_id, trace.input_id)
            assert trace.metadata["session_id"] == expected, "session id is not the lineage hash"
        span_types: dict[str, int] = {}
        for trace in traces.traces:
            for turn in trace.turns:
                for span in turn.spans:
                    span_types[span.span_type] = span_types.get(span.span_type, 0) + 1
        print(f"lineage: inputs={inputs.dataset_id} -> traces={traces.dataset_id}")
        print(f"span types across the batch: {dict(sorted(span_types.items()))}")
        assert {"agent", "llm", "tool", "retrieval"} <= set(span_types), span_types
        print("PASS — schema-valid traces with correct lineage, nothing quarantined")

        # --------------------------------------------------------- multi-turn
        banner("GATE 2 — persona-simulator conversation as one M>=2 trace")
        conversations = [t for t in traces.traces if t.mode == "multi_turn"]
        assert conversations, "no multi-turn traces collected"
        best = max(conversations, key=lambda t: len(t.turns))
        for turn in best.turns:
            print(f"  turn {turn.turn_index}: user={turn.user_message[:80]!r}")
            print(f"            app ={turn.final_response[:80]!r} ({len(turn.spans)} spans)")
        assert len(best.turns) >= 2, f"only {len(best.turns)} turn(s) collected"
        assert all(turn.spans for turn in best.turns), "a turn came back with no spans"
        print(f"PASS — {len(best.turns)} turns in one trace, per-turn spans present")

        # ------------------------------------------------------------- rerun
        banner("GATE 3 — rerun skips inputs that already have an ok trace")
        rerun_outputs, rerun_traces = harness.run_batch(inputs)
        print(f"stats: {harness.stats}")
        ok_count = sum(1 for t in traces.traces if t.status == "ok")
        assert harness.stats["skipped"] == ok_count, harness.stats
        assert harness.stats["ran"] == len(inputs.inputs) - ok_count
        assert rerun_traces.dataset_id == traces.dataset_id, "rerun changed the dataset id"
        assert rerun_outputs.dataset_id == outputs.dataset_id
        print(f"PASS — {ok_count} completed inputs skipped, dataset ids unchanged")

        # ------------------------------------------------------------ faults
        banner("GATE 4 — run_with_faults: retriever fault corrupts the retrieval span")
        # Pick an input whose clean trace actually exercised retrieval.
        def retrieval_spans(trace):
            return [s for t in trace.turns for s in t.spans if s.span_type == "retrieval"]

        candidates = [
            t for t in traces.traces if t.mode == "single_turn" and retrieval_spans(t)
        ]
        assert candidates, "no clean single-turn trace exercised the retriever"
        baseline = candidates[0]
        spec = next(i for i in inputs.inputs if i.input_id == baseline.input_id)
        print(f"input: {spec.input_id} — {spec.prompt[:100]!r}")
        print(f"baseline docs : {json.dumps(retrieval_spans(baseline)[-1].outputs)[:200]}")

        # `irrelevant_docs` is a *structural* behaviour name, so collection
        # also scans the armed trace for it — the extra_leak_tokens path gets
        # exercised live, not just in unit tests.
        fault = FaultConfig(shim="retriever", target="corpus_search", behavior="irrelevant_docs")
        armed = harness.run_with_faults(
            spec, fault, dataset_id=inputs.dataset_id, baseline=baseline
        )
        armed_docs = retrieval_spans(armed)
        assert armed_docs, "the armed run has no retrieval span"
        print(f"armed docs    : {json.dumps(armed_docs[-1].outputs)[:200]}")
        print(f"evidence      : {harness.activation_evidence[armed.trace_id][:160]}")
        assert armed.trace_id != baseline.trace_id
        assert json.dumps(armed_docs[-1].outputs) != json.dumps(
            retrieval_spans(baseline)[-1].outputs
        ), "the armed retrieval span is identical to the baseline"
        print("PASS — the declared fault visibly corrupted the retrieval span")

        # ------------------------------------------------------------ replay
        banner("GATE 5 — replay: fork an edited checkpoint and continue coherently")
        conversation = best
        thread_id = conversation.metadata["thread_id"]
        answer = conversation.turns[0].final_response
        # Text AND index: the index says which turn, the text double-checks it.
        checkpoint_id, message_id = harness.locate_checkpoint(thread_id, answer, turn_index=0)
        print(f"thread={thread_id} checkpoint={checkpoint_id} message={message_id}")
        print(f"original turn-0 answer: {answer[:160]!r}")
        assert CASE_REFERENCE not in answer, "the marker was not unique to the edit"

        replayed = harness.replay(
            thread_id,
            checkpoint_id,
            {"messages": [{"role": "ai", "id": message_id, "content": CORRUPTED_ANSWER}]},
            [REPLAY_QUESTION],
            input_id=conversation.input_id,
            dataset_id=inputs.dataset_id,
        )
        continuation = replayed.turns[-1].final_response
        print(f"continuation          : {continuation[:220]!r}")
        assert replayed.metadata["source_checkpoint_id"] == checkpoint_id
        assert replayed.metadata["fork_checkpoint_id"]
        assert replayed.metadata["thread_id"] == thread_id
        assert store.exists(replayed.trace_id)
        assert CASE_REFERENCE.lower() in continuation.lower(), (
            "the continuation did not build on the edited checkpoint"
        )
        print("PASS — the fork's continuation reproduces content only the edit contained")

        # --------------------------------------------------------- leak audit
        banner("GATE 6 — live leak audit over every stored trace")
        # The armed behaviour joins the scan for this pass, so the live audit
        # covers exactly what run_with_faults itself scanned for.
        tokens = leak_tokens(cfg, ("irrelevant_docs",))
        scanned = 0
        manifests = 0
        unaudited = []
        for trace_id in store.list_ids():
            trace = store.get(trace_id)
            payload = trace.model_dump(mode="json", exclude={"ablation_ids"})
            leaked = find_leaks(payload, tokens)
            keys = find_leaked_keys(payload)
            assert not leaked and not keys, f"{trace_id}: tokens={leaked} keys={keys}"
            audited = trace.metadata.get("llm_manifests_audited", 0)
            manifests += audited
            # Per trace, not summed: one trace with 40 manifests would otherwise
            # hide ten traces whose manifests were never fetched at all.
            if trace.status == "ok" and audited < 1:
                unaudited.append(trace_id)
            scanned += 1
        print(f"traces scanned: {scanned}; llm manifests fetched with serialized: {manifests}")
        assert scanned >= len(inputs.inputs)
        assert not unaudited, (
            f"{len(unaudited)} ok trace(s) had no serialized manifest fetched — the "
            f"audit was vacuous for them: {unaudited}"
        )
        print("PASS — no fault key, behaviour token, shim name, or fault_* metadata key")

        banner("PHASE 4 SMOKE OK — every gate green")
        print(f"artifacts under {OUT_DIR.relative_to(REPO_ROOT)}")
        return 0
    finally:
        if not args.no_serve and not args.keep:
            serve("stop")


if __name__ == "__main__":
    raise SystemExit(main())
