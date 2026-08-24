"""Gate 4 — each shim, armed through `config.configurable`, corrupts its span.

Runs the same prompts unarmed and armed, then reads the resulting spans back
out of LangSmith and diffs them. Evidence is taken from the trace, not from
the app, because that is what the ablation engine will validate against
("activation is visible in the regenerated spans").

The last section is the leak audit: an armed trace must be distinguishable
from an organic one only by what the dependency *did*, never by a fault name
sitting in span inputs, outputs, or metadata.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _contract import banner, load_contract  # noqa: E402
from gate2_trace_export import find_root  # noqa: E402
from langgraph_sdk import get_sync_client  # noqa: E402
from langsmith import Client  # noqa: E402

REFUND_Q = "What is your refund policy for an annual plan? Give me the exact number of days."
TICKET_Q = (
    "I was charged twice in June. Open a support ticket with billing for me now — "
    "do not ask me any follow-up questions first, just file it and tell me what happened."
)

# Fault values are mappings, never scalars: langchain promotes str/int/float/bool
# `configurable` entries into inheritable tracing metadata on every span.
CASES = [
    ("retriever/unarmed", REFUND_Q, {}),
    ("retriever/irrelevant_docs", REFUND_Q, {"fault_retriever": {"behavior": "irrelevant_docs"}}),
    ("retriever/empty", REFUND_Q, {"fault_retriever": {"behavior": "empty"}}),
    ("retriever/stale", REFUND_Q, {"fault_retriever": {"behavior": "stale"}}),
    ("tool/unarmed", TICKET_Q, {}),
    ("tool/error", TICKET_Q, {"fault_tool": {"behavior": "error"}}),
    ("tool/timeout", TICKET_Q, {"fault_tool": {"behavior": "timeout", "delay_seconds": 3}}),
    ("tool/corrupted_result", TICKET_Q, {"fault_tool": {"behavior": "corrupted_result"}}),
    ("llm/truncate_output", REFUND_Q, {"fault_llm": {"behavior": "truncate_output"}}),
]

# Structural giveaways. Plain behaviour words are excluded on purpose: corpus
# text legitimately contains e.g. "Emptying the Trash".
FORBIDDEN_TOKENS = (
    "fault_retriever",
    "fault_tool",
    "fault_llm",
    "irrelevant_docs",
    "corrupted_result",
    "truncate_output",
    "shim",
    "ablat",
    "supportchatmodel",
)


def text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


def retrieval_span(spans):
    """The retriever span, or None. Absence is a failure, never an empty result."""
    found = [run for run in spans if run.run_type == "retriever"]
    return found[-1] if found else None


def tool_span_output(spans, name: str) -> dict | None:
    for run in spans:
        if run.run_type == "tool" and run.name == name:
            raw = (run.outputs or {}).get("output")
            if isinstance(raw, dict):  # a serialized ToolMessage
                raw = raw.get("content")
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {"raw": raw}
            return raw
    return None


def final_llm_span(spans):
    """The model call that produced the final answer — by time, not list order."""
    calls = [run for run in spans if run.run_type == "llm"]
    if not calls:
        return None
    return max(calls, key=lambda run: (run.end_time or run.start_time, run.start_time))


def llm_span_text(run) -> str:
    generations = (run.outputs or {}).get("generations") or []
    flat = generations
    if generations and isinstance(generations[0], list):
        flat = generations[0]
    if not flat:
        return ""
    gen = flat[-1]
    return gen.get("text") or text_of(gen.get("message", {}).get("kwargs", {}))


def run_case(client, contract, label, prompt, configurable) -> dict:
    needed = "create_ticket" if label.startswith("tool/") else "rag_search"
    for attempt in range(3):
        session_id = f"gate4-{uuid.uuid4().hex[:12]}"
        thread_id = client.threads.create()["thread_id"]
        run = client.runs.wait(
            thread_id,
            contract["assistant_id"],
            input={"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": configurable},
            metadata={"session_id": session_id},
        )
        if "__error__" in run:
            raise SystemExit(f"{label}: run failed: {run['__error__']}")
        called = {
            call["name"]
            for message in run["messages"]
            for call in (message.get("tool_calls") or [])
        }
        if needed in called:
            break
        print(f"  {label}: model did not call {needed} (attempt {attempt + 1}/3), retrying")
    else:
        raise SystemExit(f"{label}: model never called {needed} in 3 attempts — cannot judge")
    print(f"ran {label:28s} configurable={configurable or '{}'} tools={sorted(called)}")
    return {
        "session_id": session_id,
        "final": text_of(run["messages"][-1]),
        "configurable": configurable,
    }


def fetch_spans(ls, project, label, payload):
    root = find_root(ls, project, payload["session_id"])
    want = "create_ticket" if label.startswith("tool/") else "corpus_search"
    for attempt in range(20):
        spans = list(ls.list_runs(project_name=project, trace_id=root.trace_id))
        if any(run.name == want for run in spans):
            return spans
        print(f"  waiting for child spans of {label} ({attempt + 1}/20)…")
        time.sleep(3)
    raise SystemExit(f"{label}: LangSmith never returned a {want} span for this trace")


def main() -> int:
    contract = load_contract()
    banner("GATE 4 — shims armed via config.configurable, evidence from the spans")
    print(f"declared fault keys: {json.dumps(contract['fault_configurable_keys'])}\n")

    client = get_sync_client(url=contract["base_url"])
    results = {
        label: run_case(client, contract, label, prompt, configurable)
        for label, prompt, configurable in CASES
    }

    ls = Client()
    project = contract["langsmith_project"]
    for label, payload in results.items():
        payload["spans"] = fetch_spans(ls, project, label, payload)

    def docs_of(label):
        span = retrieval_span(results[label]["spans"])
        assert span is not None, f"{label}: no retriever span in the trace at all"
        docs = (span.outputs or {}).get("output")
        assert isinstance(docs, list), f"{label}: retriever span has no document list"
        return docs

    # ---------------------------------------------------------------- retriever
    banner("shim 1/3 — retriever (key: fault_retriever)")
    base = docs_of("retriever/unarmed")
    print(f"unarmed          -> {[(d['doc_id'], d['updated']) for d in base]}")
    for behavior in ("irrelevant_docs", "empty", "stale"):
        got = docs_of(f"retriever/{behavior}")
        print(f"{behavior:16s} -> {[(d['doc_id'], d['updated']) for d in got]}")

    base_ids = {d["doc_id"] for d in base}
    assert "refund-policy" in base_ids, "unarmed retrieval was not relevant"
    irrelevant_ids = {d["doc_id"] for d in docs_of("retriever/irrelevant_docs")}
    assert irrelevant_ids and not (irrelevant_ids & base_ids), "irrelevant_docs did not swap docs"
    assert docs_of("retriever/empty") == [], "empty did not clear the retrieval span"
    stale = docs_of("retriever/stale")
    assert {d["doc_id"] for d in stale} == base_ids
    assert all(d["updated"].startswith("2019") for d in stale), "stale docs are not outdated"
    assert all("score" not in d for d in base + stale), "relevance scores leaked into the span"
    print("\nunarmed answer  :", results["retriever/unarmed"]["final"][:160].replace("\n", " "))
    print("stale answer    :", results["retriever/stale"]["final"][:160].replace("\n", " "))
    print("empty answer    :", results["retriever/empty"]["final"][:160].replace("\n", " "))
    print("PASS — retrieval span corrupted three ways, unarmed run unaffected")

    # --------------------------------------------------------------------- tool
    banner("shim 2/3 — action tool (key: fault_tool)")
    tool_labels = ("tool/unarmed", "tool/error", "tool/timeout", "tool/corrupted_result")
    outputs = {}
    for label in tool_labels:
        span = next(
            run
            for run in results[label]["spans"]
            if run.run_type == "tool" and run.name == "create_ticket"
        )
        outputs[label] = tool_span_output(results[label]["spans"], "create_ticket")
        seconds = (span.end_time - span.start_time).total_seconds()
        print(f"{label:24s} {seconds:5.2f}s -> {json.dumps(outputs[label])}")
        results[label]["tool_seconds"] = seconds

    assert outputs["tool/unarmed"]["status"] == "created"
    assert outputs["tool/unarmed"]["ticket_id"].startswith("NN-")
    assert outputs["tool/error"]["status"] == "error"
    assert "ticket_id" not in outputs["tool/error"]
    assert outputs["tool/timeout"]["error"] == "timeout"
    assert results["tool/timeout"]["tool_seconds"] >= 3, "timeout did not stall the span"
    assert results["tool/unarmed"]["tool_seconds"] < 3
    assert outputs["tool/corrupted_result"]["ticket_id"] == "NN-000000"
    assert outputs["tool/corrupted_result"]["eta_hours"] == -1
    print("\nunarmed answer  :", results["tool/unarmed"]["final"][:160].replace("\n", " "))
    print("error answer    :", results["tool/error"]["final"][:160].replace("\n", " "))
    print("timeout answer  :", results["tool/timeout"]["final"][:160].replace("\n", " "))
    print("PASS — create_ticket span shows the failure / stall / garbled payload")

    # ---------------------------------------------------------------------- llm
    banner("shim 3/3 — llm (key: fault_llm)")
    armed = results["llm/truncate_output"]
    span = final_llm_span(armed["spans"])
    assert span is not None, "no ChatOpenAI span in the armed trace"
    span_text = llm_span_text(span)
    unarmed_span_text = llm_span_text(final_llm_span(results["retriever/unarmed"]["spans"]))
    print(f"unarmed  : llm span {len(unarmed_span_text):4d} chars, "
          f"final {len(results['retriever/unarmed']['final']):4d} chars")
    print(f"armed    : llm span {len(span_text):4d} chars, final {len(armed['final']):4d} chars")
    print(f"armed final ends: …{armed['final'][-90:]!r}")
    assert span_text, "could not read the ChatOpenAI span output"
    # The shim runs inside the model call, so the span records the degraded text.
    # If it did not, "final != last llm span output" would be a harness tell.
    assert span_text == armed["final"], "llm span and final answer disagree — inconsistent trace"
    assert unarmed_span_text == results["retriever/unarmed"]["final"]
    assert len(armed["final"]) < len(unarmed_span_text), "armed answer was not shortened"
    assert not armed["final"].rstrip().endswith((".", "!", "?")), "does not look truncated"
    print("PASS — the llm span itself carries the truncated generation (no inconsistency)")

    # -------------------------------------------------------------- leak audit
    banner("leak audit — armed traces must not name their own fault")
    worst = 0
    for label, payload in results.items():
        blob = json.dumps(
            [
                {
                    "name": r.name,
                    "inputs": r.inputs,
                    "outputs": r.outputs,
                    "extra": r.extra,
                    # LangSmith retains `serialized` for llm and prompt runs, so
                    # the model manifest ships with every model call.
                    "serialized": getattr(r, "serialized", None),
                }
                for r in payload["spans"]
            ],
            default=str,
        ).lower()
        leaked = sorted({token for token in FORBIDDEN_TOKENS if token in blob})
        metadata_keys = sorted(
            {
                key
                for r in payload["spans"]
                for key in ((r.extra or {}).get("metadata") or {})
                if key.startswith("fault_")
            }
        )
        print(f"{label:28s} spans={len(payload['spans']):3d} leaked={leaked} "
              f"fault_metadata={metadata_keys}")
        worst += len(leaked) + len(metadata_keys)
    assert worst == 0, "a fault name reached the trace"
    print("PASS — no fault key, behaviour name, or shim token anywhere in any span")

    banner("GATE 4 OK — all three declared shims activate, visibly and without leaking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
