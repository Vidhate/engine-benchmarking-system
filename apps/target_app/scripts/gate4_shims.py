"""Gate 4 — each shim, armed through `config.configurable`, corrupts its span.

Runs the same prompts unarmed and armed, then reads the resulting spans back
out of LangSmith and diffs them. Evidence is taken from the trace, not from
the app, because that is what the ablation engine will validate against
("activation is visible in the regenerated spans").
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

CASES = [
    ("retriever/unarmed", REFUND_Q, {}),
    ("retriever/irrelevant_docs", REFUND_Q, {"fault_retriever": "irrelevant_docs"}),
    ("retriever/empty", REFUND_Q, {"fault_retriever": "empty"}),
    ("retriever/stale", REFUND_Q, {"fault_retriever": "stale"}),
    ("tool/unarmed", TICKET_Q, {}),
    ("tool/error", TICKET_Q, {"fault_tool": "error"}),
    ("tool/corrupted_result", TICKET_Q, {"fault_tool": "corrupted_result"}),
    ("llm/truncate_output", REFUND_Q, {"fault_llm": "truncate_output"}),
]


def text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content or [] if isinstance(b, dict))


def retrieval_docs(spans) -> list[dict]:
    for run in spans:
        if run.run_type == "retriever":
            outputs = run.outputs or {}
            docs = outputs.get("output") or outputs.get("documents") or []
            return docs if isinstance(docs, list) else []
    return []


def tool_output(spans, name: str) -> dict | None:
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


def llm_span_text(spans) -> str:
    for run in spans:
        if run.run_type == "llm" and run.name == "ChatOpenAI":
            generations = (run.outputs or {}).get("generations") or []
            flat = generations
            if generations and isinstance(generations[0], list):
                flat = generations[0]
            if flat:
                gen = flat[-1]
                return gen.get("text") or text_of(gen.get("message", {}).get("kwargs", {}))
    return ""


def main() -> int:
    contract = load_contract()
    banner("GATE 4 — shims armed via config.configurable, evidence from the spans")
    keys = contract["fault_configurable_keys"]
    print(f"declared fault keys: {json.dumps(keys)}\n")

    client = get_sync_client(url=contract["base_url"])
    results: dict[str, dict] = {}

    for label, prompt, configurable in CASES:
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
            called = {
                call["name"]
                for message in run["messages"]
                for call in (message.get("tool_calls") or [])
            }
            if needed in called:
                break
            print(f"  {label}: model did not call {needed} (attempt {attempt + 1}/3), retrying")
        results[label] = {
            "session_id": session_id,
            "final": text_of(run["messages"][-1]),
            "configurable": configurable,
        }
        print(f"ran {label:28s} configurable={configurable or '{}'} tools={sorted(called)}")

    ls = Client()
    project = contract["langsmith_project"]
    expected_tool = {"tool/unarmed", "tool/error", "tool/corrupted_result"}
    for label, payload in results.items():
        root = find_root(ls, project, payload["session_id"])
        want = "create_ticket" if label in expected_tool else "corpus_search"
        for attempt in range(20):
            spans = list(ls.list_runs(project_name=project, trace_id=root.trace_id))
            if any(run.name == want for run in spans):
                break
            print(f"  waiting for child spans of {label} ({attempt + 1}/20)…")
            time.sleep(3)
        payload["spans"] = spans

    def docs_of(label):
        return retrieval_docs(results[label]["spans"])

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
    print("\nunarmed answer  :", results["retriever/unarmed"]["final"][:160].replace("\n", " "))
    print("stale answer    :", results["retriever/stale"]["final"][:160].replace("\n", " "))
    print("empty answer    :", results["retriever/empty"]["final"][:160].replace("\n", " "))
    print("PASS — retrieval span corrupted three ways, unarmed run unaffected")

    # --------------------------------------------------------------------- tool
    banner("shim 2/3 — action tool (key: fault_tool)")
    for label in ("tool/unarmed", "tool/error", "tool/corrupted_result"):
        print(f"{label:24s} -> {json.dumps(tool_output(results[label]['spans'], 'create_ticket'))}")
    normal = tool_output(results["tool/unarmed"]["spans"], "create_ticket")
    errored = tool_output(results["tool/error"]["spans"], "create_ticket")
    corrupted = tool_output(results["tool/corrupted_result"]["spans"], "create_ticket")
    assert normal and normal["status"] == "created" and normal["ticket_id"].startswith("NN-")
    assert errored and errored["status"] == "error" and "ticket_id" not in errored
    assert corrupted and corrupted["ticket_id"] == "NN-000000" and corrupted["eta_hours"] == -1
    print("\nunarmed answer  :", results["tool/unarmed"]["final"][:160].replace("\n", " "))
    print("error answer    :", results["tool/error"]["final"][:160].replace("\n", " "))
    print("PASS — create_ticket span shows the injected failure / garbled payload")

    # ---------------------------------------------------------------------- llm
    banner("shim 3/3 — llm (key: fault_llm)")
    truncated_final = results["llm/truncate_output"]["final"]
    model_output = llm_span_text(results["llm/truncate_output"]["spans"])
    unarmed_final = results["retriever/unarmed"]["final"]
    print(f"unarmed final response      : {len(unarmed_final)} chars")
    print(f"armed model span output     : {len(model_output)} chars")
    print(f"armed final response        : {len(truncated_final)} chars")
    print(f"armed final response ends   : …{truncated_final[-90:]!r}")
    assert model_output, "could not read the ChatOpenAI span output"
    assert len(truncated_final) < len(model_output), "final response was not truncated"
    assert model_output.startswith(truncated_final), "truncation is not a prefix of the model span"
    print("PASS — the model span holds the full answer, the agent output is cut short")

    banner("GATE 4 OK — all three declared shims activate and are visible in the trace")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
