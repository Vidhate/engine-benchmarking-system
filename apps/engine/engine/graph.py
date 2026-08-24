"""The served Engine graph.

A deterministic LangGraph loop, not an agent scaffold: `load` -> `analyze` (once
per trace, sequentially, carrying the running title list) -> `consolidate`. The
orchestration is fixed code; the LLM is called inside the nodes. Only the
per-trace analysis pass has tools, and its registry is exactly the four
trace-inspection tools (`engine/tools.py`).

Run input  : {trace_file, seed_issueboard?, categories?}
Run output : an Issueboard-shaped object — {board_id, source, issues,
             occurrences} — with source="engine_predicted", ready to validate
             against `benchmark.schemas.issues.Issueboard` with no translation.
"""

from __future__ import annotations

import os
import sys
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

# Supersteps per run are 2 + one per trace, so the LangGraph default recursion
# limit of 25 caps a run at ~23 traces. Raised on the compiled graph; callers
# driving large sets should still pass an explicit `recursion_limit` on the run
# config (see README, "Invoking the Engine").
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
    """One superstep = one trace, so a long run checkpoints as it goes."""
    index = trace_index(state["trace_file"])
    cursor = state.get("cursor", 0)
    trace_id = state["trace_ids"][cursor]
    model = build_model(resolve_model_name(config))

    try:
        findings = analyze_trace(
            model=model,
            index=index,
            trace_id=trace_id,
            running_titles=list(state.get("running_titles") or []),
            categories=_categories(state),
            max_tool_calls=_max_tool_calls(),
        )
    except Exception as exc:
        # One unhappy trace must not abandon the other N-1. A systemic failure
        # (bad key, unknown model) fails every trace and is re-raised at
        # consolidation rather than being reported as "no issues found".
        message = f"{trace_id}: {type(exc).__name__}: {exc}"
        print(f"[engine] analysis failed for {message}", file=sys.stderr)
        return {"cursor": cursor + 1, "errors": [message]}

    known = set(state.get("running_titles") or [])
    new_titles = []
    for finding in findings:
        if finding.title not in known:
            known.add(finding.title)
            new_titles.append(finding.title)
    return {
        "cursor": cursor + 1,
        "findings": [f.model_dump(mode="json") for f in findings],
        "running_titles": new_titles,
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
