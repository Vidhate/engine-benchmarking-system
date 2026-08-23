# target_app — the AI app under benchmark

A small product-support assistant for a fictional note-taking product ("Nimbus
Notes"). It is a **standalone uv project**: it has its own `pyproject.toml` and
is deliberately *not* a member of the root workspace, so `uv sync` at the repo
root never sees it.

The benchmark never imports anything from here. Everything it knows about this
app is in `configs/target_app.yaml`, and everything it does to this app goes
through the LangGraph Server API (`langgraph_sdk`).

## Shape

- `deepagents` agent on a small OpenAI model (`TARGET_APP_MODEL`, default
  `gpt-5-mini`), served as assistant `target_app` via `langgraph.json`.
- **Exactly two tools**: `rag_search` (BM25 over `target_app/corpus/`, no
  embeddings, fully deterministic) and `create_ticket` (stub returning a fake
  ticket id). deepagents' built-in filesystem / shell / subagent tools are
  excluded through a `HarnessProfile`, so the model is bound with those two
  and nothing else.
- Retrieval runs inside its own LangSmith `retriever` span (`corpus_search`).
- No checkpointer is compiled into the graph — the LangGraph server owns
  persistence, which is what makes thread time-travel available.
- Tracing goes to the LangSmith project `engine-bench-target`.

## Fault shims (Mode C surface)

Each dependency reads its fault instruction from `config.configurable` on the
run. **No key present means completely normal behaviour.**

| configurable key | behaviours |
|---|---|
| `fault_retriever` | `irrelevant_docs`, `empty`, `stale` |
| `fault_tool` | `error`, `timeout`, `corrupted_result` |
| `fault_llm` | `truncate_output` |

A value is either a behaviour string (`"empty"`) or a dict with a `behavior`
key plus parameters (`{"behavior": "timeout", "delay_seconds": 2}`).

Activation is deliberately *organic*: no marker or flag announces the fault.
`stale` serves each document's checked-in archived revision (an outdated,
contradictory version of the same page) rather than a synthetic placeholder,
so an Engine cannot pattern-match its way to the answer.

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
