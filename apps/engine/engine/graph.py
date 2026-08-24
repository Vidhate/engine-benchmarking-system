"""The served Engine graph.

A deterministic LangGraph loop, not an agent scaffold: `load` -> `analyze`
-> `consolidate`. The orchestration is fixed code; the LLM is called inside the
nodes. Only the analysis pass has tools, and its registry is exactly the four
trace-inspection tools (`engine/tools.py`).

`analyze` runs once per *batch* of traces rather than once per trace: the
batch's traces are analysed concurrently (default 8, see `resolve_concurrency`),
while the batches themselves stay sequential so each one inherits the running
title list the previous ones built. Findings are assembled in input trace order,
never completion order, so the board does not depend on thread scheduling.
`analysis_concurrency=1` reproduces fully sequential analysis.

Run input  : {trace_file, seed_issueboard?, categories?}
Run output : an Issueboard-shaped object — {board_id, source, issues,
             occurrences} — with source="engine_predicted", ready to validate
             against `benchmark.schemas.issues.Issueboard` with no translation.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from engine.analysis import DEFAULT_MAX_TOOL_CALLS, analyze_trace, consolidate
from engine.llm import build_model, resolve_model_name
from engine.models import Category, RawFinding, SeedIssueboard
from engine.traces import TraceIndex, load_categories

MAX_TOOL_CALLS_ENV_VAR = "ENGINE_MAX_TOOL_CALLS_PER_TRACE"

# The node `config` parameter MUST be annotated exactly `RunnableConfig`.
# `_runnable.KWARGS_CONFIG_KEYS` matches the annotation against a fixed list,
# and because `from __future__ import annotations` turns annotations into
# strings, only the literal strings "RunnableConfig" and
# "Optional[RunnableConfig]" are accepted. The PEP-604 form
# `RunnableConfig | None` is not on that list, so LangGraph declines to inject
# the config at all. Written as `config: RunnableConfig | None = None` that is
# invisible — the node just sees no model override and uses the default, so
# BOTH arms of the Sol-vs-mini comparison would quietly run the same model. The
# bare annotation used below carries no default, so the same mistake raises
# TypeError on the first superstep instead of corrupting a comparison.
# Guarded by tests/test_graph.py::test_the_model_comes_from_the_run_configurable.

# Fraction of traces whose analysis may fail before the run is called off.
# Skip-and-continue below the line keeps a flaky trace from costing the other
# N-1; above it, the far likelier explanation is systemic (bad key, unknown
# model, rate limiting), and "the Engine barely ran" must never reach scoring
# wearing the shape of "the Engine found almost nothing".
MAX_TRACE_FAILURE_RATE = 0.2

# Traces analysed concurrently within one batch. Batches themselves stay
# sequential, because the running title list is what lets a later trace
# recognise a failure mode an earlier one already named.
ANALYSIS_CONCURRENCY_KEY = "analysis_concurrency"
ANALYSIS_CONCURRENCY_ENV_VAR = "ENGINE_ANALYSIS_CONCURRENCY"
DEFAULT_ANALYSIS_CONCURRENCY = 8
MIN_ANALYSIS_CONCURRENCY = 1
MAX_ANALYSIS_CONCURRENCY = 16


def resolve_concurrency(config: RunnableConfig | None = None) -> int:
    """Batch size for the analysis pass: run config, then env, then the default.

    Clamped rather than validated: an out-of-range value is a caller's typo, and
    refusing the whole run over it would be a worse outcome than running at a
    sane speed. 1 restores strictly sequential analysis.
    """
    configurable = (config or {}).get("configurable") or {}
    raw = configurable.get(ANALYSIS_CONCURRENCY_KEY)
    if raw is None:
        raw = os.environ.get(ANALYSIS_CONCURRENCY_ENV_VAR)
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_ANALYSIS_CONCURRENCY
    return max(MIN_ANALYSIS_CONCURRENCY, min(MAX_ANALYSIS_CONCURRENCY, value))

# Supersteps per run are 2 + ceil(n_traces / analysis_concurrency), so at the
# default batch size the LangGraph default limit of 25 caps a run at ~184
# traces — comfortable, but not comfortable enough to rely on when the batch
# size is a run-time knob. Raised on the compiled graph; callers driving large
# sets should still pass an explicit `recursion_limit` (see README).
RECURSION_LIMIT = 10_000


class EngineInput(TypedDict):
    """Everything the Engine is allowed to see. Nothing else crosses the
    boundary — no ablation records, no ground truth, no pre-ablation traces."""

    trace_file: str
    seed_issueboard: NotRequired[dict[str, Any]]
    categories: NotRequired[list[dict[str, Any]]]


class EngineOutput(TypedDict):
    board_id: str
    source: str
    issues: list[dict[str, Any]]
    occurrences: list[dict[str, Any]]


def _extend(left: list, right: list) -> list:
    return [*(left or []), *(right or [])]


class EngineState(TypedDict, total=False):
    trace_file: str
    seed_issueboard: dict[str, Any]
    categories: list[dict[str, Any]]
    # analysis loop
    trace_ids: list[str]
    cursor: int
    running_titles: Annotated[list[str], _extend]
    findings: Annotated[list[dict[str, Any]], _extend]
    errors: Annotated[list[str], _extend]
    # output
    board_id: str
    source: str
    issues: list[dict[str, Any]]
    occurrences: list[dict[str, Any]]


_INDEX_CACHE: dict[tuple[str, float], TraceIndex] = {}


def trace_index(trace_file: str) -> TraceIndex:
    """Load (and memoize per file mtime) the trace file for this run.

    The index is not graph state: it is a derived, read-only view of a file on
    disk, and putting an unbounded trace corpus into a checkpointed state would
    serialize the whole corpus on every superstep.
    """
    path = Path(trace_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"trace_file not found: {path}")
    key = (str(path.resolve()), path.stat().st_mtime)
    index = _INDEX_CACHE.get(key)
    if index is None:
        # Held in a local and returned from there: a concurrent run clearing the
        # cache between the insert and a re-read would otherwise KeyError.
        index = TraceIndex.from_file(path)
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = index
    return index


def _categories(state: EngineState) -> list[Category]:
    return load_categories(state.get("categories"))


def _seed(state: EngineState) -> SeedIssueboard:
    return SeedIssueboard.model_validate(state.get("seed_issueboard") or {})


def _max_tool_calls() -> int:
    raw = os.environ.get(MAX_TOOL_CALLS_ENV_VAR)
    return int(raw) if raw and raw.isdigit() else DEFAULT_MAX_TOOL_CALLS


def load_node(state: EngineState) -> dict[str, Any]:
    index = trace_index(state["trace_file"])
    return {"trace_ids": index.trace_ids, "cursor": 0}


def analyze_node(state: EngineState, config: RunnableConfig) -> dict[str, Any]:
    """One superstep = one batch of traces, analysed concurrently.

    Batching is what makes the assignment's scale affordable: a trace takes
    ~30s to analyse, so 300 strictly sequential traces is over two hours. The
    batch is the unit of *shared context* as well as of parallelism — every
    trace in a batch sees the same running title list, and the titles the batch
    discovers are merged in before the next one starts. Cross-trace context is
    therefore traded for throughput in exactly one place, and `N=1` gives back
    the fully sequential behaviour where every trace sees all its predecessors.
    """
    index = trace_index(state["trace_file"])
    cursor = state.get("cursor", 0)
    concurrency = resolve_concurrency(config)
    batch = state["trace_ids"][cursor : cursor + concurrency]

    # One ChatOpenAI shared by every worker in the batch, deliberately: the
    # client is stateless per call and its underlying httpx pool is thread-safe,
    # so sharing reuses connections instead of opening one pool per worker.
    # (Assumption, not just doctrine — confirmed by the live N=8 gate runs.)
    model = build_model(resolve_model_name(config))
    # Snapshotted before the batch runs: titles are shared BETWEEN batches, not
    # within one, or the analysis would depend on which worker finished first.
    running_titles = list(state.get("running_titles") or [])
    categories = _categories(state)
    max_tool_calls = _max_tool_calls()

    def analyse(trace_id: str) -> tuple[str, list, str | None]:
        try:
            findings = analyze_trace(
                model=model,
                index=index,
                trace_id=trace_id,
                running_titles=running_titles,
                categories=categories,
                max_tool_calls=max_tool_calls,
            )
            return trace_id, findings, None
        except Exception as exc:
            # Caught inside the worker so one unhappy trace costs neither its
            # batchmates nor the rest of the run. A systemic failure fails every
            # trace and is re-raised at consolidation by the rate check.
            message = f"{trace_id}: {type(exc).__name__}: {exc}"
            print(f"[engine] analysis failed for {message}", file=sys.stderr)
            return trace_id, [], message

    if concurrency == 1:
        results = [analyse(trace_id) for trace_id in batch]
    else:
        # Threads, not asyncio: the node stays a plain sync callable that
        # behaves the same whether LangGraph drives it sync or async, and
        # `map` yields in input order regardless of completion order — run
        # comparability depends on the findings list not being a race result.
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="engine") as pool:
            results = list(pool.map(analyse, batch))

    known = set(running_titles)
    new_titles: list[str] = []
    findings: list[dict[str, Any]] = []
    errors: list[str] = []
    for _, trace_findings, error in results:
        if error:
            errors.append(error)
            continue
        for finding in trace_findings:
            findings.append(finding.model_dump(mode="json"))
            if finding.title not in known:
                known.add(finding.title)
                new_titles.append(finding.title)

    return {
        "cursor": cursor + len(batch),
        "findings": findings,
        "running_titles": new_titles,
        "errors": errors,
    }


def more_traces(state: EngineState) -> str:
    done = state.get("cursor", 0) >= len(state.get("trace_ids") or [])
    return "consolidate" if done else "analyze"


def consolidate_node(state: EngineState, config: RunnableConfig) -> dict[str, Any]:
    trace_ids = state.get("trace_ids") or []
    errors = state.get("errors") or []
    if errors:
        print(
            f"[engine] {len(errors)}/{len(trace_ids)} traces failed analysis; "
            f"first: {errors[0]}",
            file=sys.stderr,
        )
    if trace_ids and len(errors) / len(trace_ids) > MAX_TRACE_FAILURE_RATE:
        raise RuntimeError(
            f"analysis failed on {len(errors)} of {len(trace_ids)} traces "
            f"(> {MAX_TRACE_FAILURE_RATE:.0%}); refusing to emit a board that would "
            f"read as 'few errors found'. First failure: {errors[0]}"
        )
    findings = [RawFinding.model_validate(f) for f in (state.get("findings") or [])]
    model = build_model(resolve_model_name(config))
    board = consolidate(model, findings, _seed(state), _categories(state))
    return {
        "board_id": board.board_id,
        "source": board.source,
        "issues": [i.model_dump(mode="json") for i in board.issues],
        "occurrences": [o.model_dump(mode="json") for o in board.occurrences],
    }


def build_graph():
    builder = StateGraph(EngineState, input_schema=EngineInput, output_schema=EngineOutput)
    builder.add_node("load", load_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("consolidate", consolidate_node)
    builder.add_edge(START, "load")
    builder.add_conditional_edges("load", more_traces, ["analyze", "consolidate"])
    builder.add_conditional_edges("analyze", more_traces, ["analyze", "consolidate"])
    builder.add_edge("consolidate", END)
    return builder.compile().with_config(recursion_limit=RECURSION_LIMIT)


graph = build_graph()
