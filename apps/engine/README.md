# engine — the dummy Engine under benchmark

A stand-in for LangChain's Engine: it reads a trace file, finds the places where
the traced app misbehaved, and emits an **updated issueboard**. It is a
**standalone uv project** — its own `pyproject.toml`, deliberately *not* a member
of the root workspace, so `uv sync` at the repo root never sees it.

The benchmark never imports anything from here. Everything it knows about this
app is in `configs/engine.yaml`; everything it does to this app goes through the
LangGraph Server API (`langgraph_sdk`).

## Shape

A **deterministic LangGraph loop**, not an agent scaffold:

```
load ──▶ analyze (once per trace, sequential) ──▶ consolidate ──▶ END
            ▲            │
            └────────────┘
```

- **`analyze`** — one superstep per trace. The analysis LLM gets the four
  trace-inspection tools plus the running list of issue titles found so far in
  this run, and emits raw findings `{trace_id, title, description, category_id,
  severity, evidence}`. The tool loop is bounded
  (`ENGINE_MAX_TOOL_CALLS_PER_TRACE`, default 16); the structured emit step that
  follows is **tool-free**, so a reported finding is one the investigation log
  supports.
- **`consolidate`** — the meta pass. The LLM decides only the *clustering*:
  which findings are the same failure mode, and which of them the seed board
  already names. `engine/consolidate.py:assemble_board` then does the merge in
  pure code, so the invariants the benchmark depends on hold regardless of what
  the model returns.

Why not `deepagents`: Phase 2 dropped it after finding its scaffold kept
filesystem and shell tools registered on the ToolNode even when hidden from the
model. That matters more here — this agent reads attacker-influenced text out of
traces for a living, so a tool it can be talked into calling is a tool it must
not have. `tests/test_tool_registry.py` asserts the dispatchable surface is
exactly the four trace tools.

## Tools

| tool | returns |
|---|---|
| `get_trace(trace_id)` | turns, final responses, span table (no payloads) |
| `list_spans(trace_id, turn_index?)` | span ids/names/types, error flags, output previews |
| `read_span(trace_id, span_id)` | one span in full |
| `search_text(query, trace_id?)` | substring hits + snippets across trace text |

All read-only, all pure functions over a `TraceIndex` (`engine/traces.py`), all
unit-tested without a model. Results are fitted to a 6 000-character budget by
shedding whole items — never by chopping the serialized string, which would hand
the model an object it cannot parse.

## Invoking the Engine (what Phase 7 needs)

```python
client = get_sync_client(url="http://127.0.0.1:2025")   # configs/engine.yaml
thread = client.threads.create()
board = client.runs.wait(
    thread["thread_id"], "engine",
    input={
        "trace_file": "/abs/path/to/traces.json",   # path, not inline traces
        "seed_issueboard": {...},                   # Issueboard JSON, may be empty
        "categories": [{"category_id", "name", "description"}, ...],
    },
    config={
        "configurable": {"model": "gpt-5.1"},       # the comparison axis
        "recursion_limit": 2 * n_traces + 10,       # see below
    },
)
```

Four things worth knowing:

1. **The run output *is* the issueboard.** It comes back as
   `{board_id, source, issues, occurrences}` with `source="engine_predicted"` —
   `Issueboard.model_validate(output)` works with no unwrapping and no
   translation. `board_id` is a content hash over the board, computed the same
   way `benchmark.schemas.io.content_hash` does.
2. **`trace_file` is a path the server can read.** The corpus is not sent
   through the API and is never put into graph state — a 300-trace corpus in a
   checkpointed state would be re-serialized on every superstep.
3. **Set `recursion_limit`.** The loop runs `2 + n_traces` supersteps, and
   LangGraph's default of 25 caps a run at ~23 traces. The compiled graph raises
   its own default to 10 000, but pass it explicitly on the run for anything
   large.
4. **Partial failure is reported, not hidden.** A trace whose analysis throws is
   logged to stderr and skipped; the run continues. If *every* trace fails —
   what a bad key or an unknown model id looks like — the consolidate node
   raises rather than returning an empty board, because "no issues found" and
   "the Engine never ran" must not look alike downstream.

## Model selection — the comparison axis

`config.configurable["model"]` (the key declared in `configs/engine.yaml`) picks
the model per run. Falls back to `$ENGINE_MODEL`, then to `gpt-5-mini`. Nothing
else differs between arms: same prompts, same tools, same consolidation code.

> **Footgun, fixed here, worth knowing elsewhere.** LangGraph matches a node's
> `config` parameter *annotation* against a fixed list. Under
> `from __future__ import annotations` the annotation is a string, and only
> `"RunnableConfig"` and `"Optional[RunnableConfig]"` are accepted — the PEP-604
> form `RunnableConfig | None` is **silently rejected**, and the node is then
> never handed the run config at all. The symptom is invisible: the node just
> sees no override and uses its default, so both arms of a model comparison
> quietly run the same model. Guarded by
> `tests/test_graph.py::test_the_model_comes_from_the_run_configurable`.

## What the Engine is allowed to see

Only three things: the trace file, the seed issueboard, and the category
vocabulary (names + descriptions, including `other`). No ablation ids, no
injection modes, no ablation records, no ground truth, no pre-ablation traces.

Two guards, in `tests/test_no_leak.py`:

- **behavioural** — a trace file that *still carries* ablation fields loads
  fine, and none of those fields survive parsing or appear in any tool result.
  The loader tolerates them without ever reading them.
- **structural** — an AST scan asserts no identifier or string literal anywhere
  in `engine/` names the ablation surface. Prose is exempt (comments and
  docstrings never enter the AST), so the guard can be documented without
  tripping itself.

## Tests and gates

```bash
uv run pytest                       # 97 unit tests, no network
uv run ruff check engine tests scripts
scripts/smoke.sh                    # all three gates, ~10 min, needs a real key
```

- **Gate 1** (`scripts/gate1_contract.py`, offline) — `configs/engine.yaml`
  parses into `EngineAppConfig` and names the assistant `langgraph.json` serves;
  the vocabulary is names + descriptions only with `other` present; a board from
  the real consolidation code validates as `Issueboard`.
- **Gate 2** (`scripts/gate2_live_smoke.py`) — a live run over the six fixture
  traces; asserts schema validity, seed-board survival, no duplicate ids, and
  reports which planted errors were found.
- **Gate 3** (`scripts/gate3_model_swap.py`) — the same run twice, differing only
  in `configurable["model"]`.

Scripts may import `benchmark.schemas` — they stand in for the benchmark side of
the boundary, and validating output against the real `Issueboard` is the point.
They must **not** import `engine`; `scripts/_produce_board.py` exists precisely
so the two halves of Gate 1c stay in separate processes.

## Fixtures

`tests/fixtures/traces.json` — six hand-crafted Nimbus Notes traces in the
`Trace/Turn/Span` schema: three clean, three with human-obvious planted errors
spanning three categories (a hallucinated refund window contradicting its own
retrieval span; a `create_ticket` error ignored and a ticket id invented; a
response cut off mid-sentence). `tests/fixtures/seed_issueboard.json` seeds two
issues, one of which the planted ticket error should merge into rather than
duplicate. `tests/fixtures/planted_errors.json` is the answer key **for the
smoke scripts only** — it is never part of the Engine's run input.
