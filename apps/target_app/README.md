# target_app — the AI app under benchmark

A small product-support assistant for a fictional note-taking product ("Nimbus
Notes"). It is a **standalone uv project**: it has its own `pyproject.toml` and
is deliberately *not* a member of the root workspace, so `uv sync` at the repo
root never sees it.

The benchmark never imports anything from here. Everything it knows about this
app is in `configs/target_app.yaml`, and everything it does to this app goes
through the LangGraph Server API (`langgraph_sdk`).

## Shape

- `langgraph.prebuilt.create_react_agent` on a small OpenAI model
  (`TARGET_APP_MODEL`, default `gpt-5-mini`), served as assistant `target_app`
  via `langgraph.json`.
- **Exactly two tools**: `rag_search` (BM25 over `target_app/corpus/`, no
  embeddings, fully deterministic) and `create_ticket` (stub returning a fake
  ticket id). The prebuilt registers exactly the tools it is handed, so the
  app's *dispatchable* surface equals its *declared* surface — there is no
  second set of tools a fabricated tool call could reach.
- Retrieval runs inside its own LangSmith `retriever` span (`corpus_search`).
- The LLM shim lives in a thin `ChatOpenAI` subclass rather than in middleware,
  so the llm span records the degraded generation (see "Trace leak surface").
- No checkpointer is compiled into the graph — the LangGraph server owns
  persistence, which is what makes thread time-travel available.
- Tracing goes to the LangSmith project `engine-bench-target`.

An earlier revision used `deepagents`. It was dropped: a coding-agent harness
brings filesystem, shell, and subagent tools that stay registered on the
ToolNode even when hidden from the model, plus ~10 middleware spans of trace
noise per turn — all cost, no benefit, for a two-tool support assistant.

> **Known deprecation.** `langgraph.prebuilt.create_react_agent` emits
> `LangGraphDeprecatedSinceV10`: it moved to `langchain.agents.create_agent`
> and is slated for removal in LangGraph V2.0. Migrating is a one-line import
> change in `agent.py`, at the cost of re-adding the `langchain` dependency.
> Left as-is for now; worth doing before this app is depended on long-term.

## Fault shims (Mode C surface)

Each dependency reads its fault instruction from `config.configurable` on the
run. **No key present means completely normal behaviour.**

| configurable key | behaviours |
|---|---|
| `fault_retriever` | `irrelevant_docs`, `empty`, `stale` |
| `fault_tool` | `error`, `timeout`, `corrupted_result` |
| `fault_llm` | `truncate_output` |

A value **must be a mapping**: `{"behavior": "empty"}`, or with parameters,
`{"behavior": "timeout", "delay_seconds": 2}` (parameters may also be nested
under `"params"`). A bare string is refused — see below.

Activation is deliberately *organic*: no marker or flag announces the fault.
`stale` serves each document's checked-in archived revision (an outdated,
contradictory version of the same page) rather than a synthetic placeholder,
so an Engine cannot pattern-match its way to the answer.

## Trace leak surface — required reading for the Phase 4 collector

The Engine is supposed to find injected faults by *reading the trace*. Anything
that names the fault instead is a leak, and leaks are cheap to create by
accident. What this app does about it, and what it cannot do:

**Handled here**

1. **No scalar fault values.** `langchain_core.runnables.config` copies every
   str/int/float/bool `configurable` entry into LangSmith-inheritable metadata,
   so `{"fault_retriever": "stale"}` would tag *every span of the run*. The app
   therefore refuses scalars and requires a mapping, which is not promoted.
   Pinned by `tests/test_shims.py::test_the_mapping_rule_actually_defeats_langchain_metadata_promotion`.
2. **No fault in traced arguments.** The retrieval step is a `@traceable`
   function whose only parameters are `query` and `k`; the armed fault reaches
   it through a context var, so it can never be recorded as a span input.
3. **No fingerprints in payloads.** Retrieval results carry no relevance score
   (near-zero scores would separate `irrelevant_docs` from an organic miss
   statistically), and no document is labelled "archived" or "stale".
4. **No cross-span inconsistency.** The LLM shim degrades the completion
   *inside* the model call, so the llm span and the final answer agree. A
   post-hoc truncation would have left "final != last llm span output" in every
   armed trace — a tell an Engine could learn instead of reading the trace.
5. **No giveaway model identity.** Two separate places name the class, and
   both are overridden. `name="ChatOpenAI"` pins the *run name*; `_serialized`
   is overridden to rewrite `id[-1]`, because LangChain also ships `dumpd(self)`
   as the run's `serialized` field and LangSmith **retains it for `llm` and
   `prompt` runs**. Fixing only the run name would have left
   `serialized.id == [..., "SupportChatModel"]` sitting next to the name
   `ChatOpenAI` — and that mismatch is a sharper fingerprint than either half.
6. **No shim-bypassing code path.** `disable_streaming=True`, so every
   completion goes through `_generate`/`_agenerate`. `_stream`/`_astream`
   route through neither, so a streaming caller (LangGraph Studio,
   `stream_mode="messages"`) would have had an armed `fault_llm` silently
   no-op and produce a trace that looks organic.

Gate 4 (`scripts/gate4_shims.py`) ends with a live audit asserting that no
fault key, behaviour name, shim token, or subclass name appears in any span's
inputs, outputs, name, metadata, or serialized manifest, across all nine runs.

The two layers are deliberate and neither replaces the other: the **unit tests**
assert the manifest is clean on the local model object (they guard the fix),
while the **gate** asserts it is clean on the wire (it guards the live channel).
A local-object test cannot catch an API projection gap — `list_runs` does not
return `serialized` unless asked, so the gate fetches it explicitly and fails if
any llm run comes back without one, rather than silently scanning `None`.

**NOT handled here — blocking hand-off requirement for Phase 4**

> The trace collector **MUST strip run-level metadata and any `configurable`
> echo before traces are exposed to the Engine.** This app controls what it
> writes into spans; it does not control what the LangGraph server and
> LangSmith record *about* a run (run metadata, thread/checkpoint config, the
> `session_id` the harness itself sets). Normalizing a run tree into `Trace`
> must be an allowlist of fields, not a copy of everything LangSmith returns.

## Running

```bash
uv sync
uv run pytest                 # unit tests, no network
scripts/serve.sh start        # langgraph dev on :2024, waits for health
scripts/serve.sh stop
scripts/smoke.sh              # all four Phase-2 gates end to end
```

`scripts/` holds the server-dependent smoke checks. They read
`configs/target_app.yaml` and talk to the server — they never import
`target_app`.
