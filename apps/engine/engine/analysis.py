"""The per-trace analysis pass and the meta consolidation pass.

Both are plain functions over an injected chat model, so unit tests drive them
with a fake model and never touch the network.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from engine import prompts
from engine.consolidate import (
    apply_merge,
    assemble_board,
    fallback_plan,
    fold_clusters,
    offset_plan,
)
from engine.models import (
    Category,
    ConsolidationPlan,
    Issueboard,
    RawFinding,
    RawFindingList,
    SeedIssueboard,
)
from engine.tools import build_trace_tools
from engine.traces import TraceIndex, truncate

# Tool-call budget per trace. A trace has a bounded number of spans; an agent
# still reading after this many calls is looping, not investigating.
DEFAULT_MAX_TOOL_CALLS = 16
TOOL_RESULT_LOG_CHARS = 1500

# Raw findings per consolidation call. At the assignment's scale (300+ traces)
# a single call would carry tens of thousands of tokens of findings; batching
# keeps each prompt bounded and each failure survivable.
CONSOLIDATION_CHUNK_SIZE = 80


def analyze_trace(
    model: BaseChatModel,
    index: TraceIndex,
    trace_id: str,
    running_titles: list[str],
    categories: list[Category],
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> list[RawFinding]:
    """One trace -> zero or more raw findings.

    Runs a bounded tool loop over the trace-inspection tools, then converts the
    investigation into structured findings in a separate, tool-free call. The
    two steps are separate because the emit step must not be able to keep
    reading: its input is the log of what was actually looked at, so a finding
    it reports is a finding something in the log supports.
    """
    tools = build_trace_tools(index)
    registry = {tool.name: tool for tool in tools}
    bound = model.bind_tools(tools)

    system = prompts.ANALYSIS_SYSTEM.format(
        categories=prompts.format_categories(categories),
        running_titles=prompts.format_running_titles(running_titles),
    )
    messages = [
        SystemMessage(system),
        HumanMessage(prompts.ANALYSIS_TASK.format(trace_id=trace_id)),
    ]
    log: list[str] = []
    budget = max_tool_calls

    while budget > 0:
        response = bound.invoke(messages)
        messages.append(response)
        calls = list(getattr(response, "tool_calls", None) or [])
        if not calls:
            notes = response.text if isinstance(response, AIMessage) else ""
            if isinstance(notes, str) and notes.strip():
                log.append(f"ANALYST NOTES:\n{notes}")
            break
        for call in calls:
            budget -= 1
            result = _dispatch(registry, call)
            log.append(
                f"TOOL {call.get('name')}({json.dumps(call.get('args') or {}, default=str)})"
                f" ->\n{truncate(result, TOOL_RESULT_LOG_CHARS)}"
            )
            messages.append(ToolMessage(content=result, tool_call_id=call.get("id") or ""))
    else:
        log.append("(tool-call budget exhausted)")

    emit = model.with_structured_output(RawFindingList)
    result = emit.invoke(
        [
            SystemMessage(prompts.EMIT_SYSTEM),
            HumanMessage(
                prompts.EMIT_TASK.format(
                    trace_id=trace_id,
                    transcript=(
                        f"Category vocabulary:\n{prompts.format_categories(categories)}\n\n"
                        + ("\n\n".join(log) if log else "(no tools were called)")
                    ),
                )
            ),
        ]
    )
    findings = _as_findings(result)
    # The trace under analysis is authoritative: a model-supplied trace_id would
    # let one trace's findings be filed against another's occurrences.
    return [f.model_copy(update={"trace_id": trace_id}) for f in findings]


def _dispatch(registry: dict, call: dict) -> str:
    """Execute one tool call. Only the four trace tools exist; anything else is
    reported back to the model as an error rather than being reached for."""
    tool = registry.get(call.get("name") or "")
    if tool is None:
        return json.dumps(
            {"error": f"no such tool {call.get('name')!r}", "available": sorted(registry)}
        )
    try:
        return str(tool.invoke(call.get("args") or {}))
    except Exception as exc:  # a bad argument is a model error, not a run failure
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _as_findings(result) -> list[RawFinding]:
    if isinstance(result, RawFindingList):
        return list(result.findings)
    if isinstance(result, dict):
        return list(RawFindingList.model_validate(result).findings)
    return []


def consolidate(
    model: BaseChatModel,
    findings: list[RawFinding],
    seed_board: SeedIssueboard | None,
    categories: list[Category],
    chunk_size: int = CONSOLIDATION_CHUNK_SIZE,
) -> Issueboard:
    """Cluster raw findings into canonical issues and merge over the seed board.

    The model decides only the clustering; `assemble_board` does the merge, so
    the board's invariants do not depend on the model behaving.
    """
    seed = seed_board or SeedIssueboard()
    if not findings:
        return assemble_board(ConsolidationPlan(), [], seed, categories)
    plan = cluster_findings(model, findings, seed, categories, chunk_size)
    return assemble_board(plan, findings, seed, categories)


def cluster_findings(
    model: BaseChatModel,
    findings: list[RawFinding],
    seed: SeedIssueboard,
    categories: list[Category],
    chunk_size: int = CONSOLIDATION_CHUNK_SIZE,
) -> ConsolidationPlan:
    """Cluster in batches, then merge across batches.

    A single call over every finding is what a 300-trace run cannot afford: the
    prompt grows without bound and the failure it produces (context overflow,
    truncated structured output) arrives at the one point in the run where
    everything is already paid for. Batching bounds each call; `fold_clusters`
    and the merge pass put the batches back together.
    """
    bounds = [(start, min(start + chunk_size, len(findings)))
              for start in range(0, len(findings), max(1, chunk_size))]
    if len(bounds) <= 1:
        return _cluster_chunk(model, findings, 0, seed, categories)

    print(
        f"[engine] consolidating {len(findings)} findings in {len(bounds)} batches",
        file=sys.stderr,
    )
    clusters = []
    for start, stop in bounds:
        batch = _cluster_chunk(model, findings[start:stop], start, seed, categories)
        clusters.extend(batch.clusters)
    folded = fold_clusters(ConsolidationPlan(clusters=clusters))
    return _merge_across_chunks(model, folded, seed, categories)


def _cluster_chunk(
    model: BaseChatModel,
    chunk: list[RawFinding],
    offset: int,
    seed: SeedIssueboard,
    categories: list[Category],
) -> ConsolidationPlan:
    system = prompts.CONSOLIDATION_SYSTEM.format(
        categories=prompts.format_categories(categories),
        seed_issues=prompts.format_seed_issues(seed.issues),
    )
    task = prompts.CONSOLIDATION_TASK.format(findings=prompts.format_findings(chunk))

    def call() -> ConsolidationPlan:
        return _as_plan(
            model.with_structured_output(ConsolidationPlan).invoke(
                [SystemMessage(system), HumanMessage(task)]
            )
        )

    plan = _guarded(call, lambda: fallback_plan(chunk), f"consolidation batch at {offset}")
    return offset_plan(plan, offset, len(chunk))


def _merge_across_chunks(
    model: BaseChatModel,
    plan: ConsolidationPlan,
    seed: SeedIssueboard,
    categories: list[Category],
) -> ConsolidationPlan:
    """Second stage: fold differently-worded duplicates from separate batches.

    Operates on cluster summaries, so its prompt is bounded by the number of
    distinct failure modes rather than by the number of findings.
    """
    if len(plan.clusters) <= 1:
        return plan
    system = prompts.MERGE_SYSTEM.format(
        categories=prompts.format_categories(categories),
        seed_issues=prompts.format_seed_issues(seed.issues),
    )
    task = prompts.MERGE_TASK.format(clusters=prompts.format_clusters(plan.clusters))

    def call() -> ConsolidationPlan:
        return _as_plan(
            model.with_structured_output(ConsolidationPlan).invoke(
                [SystemMessage(system), HumanMessage(task)]
            )
        )

    # Falling back to `plan` keeps every cluster: the batches simply stay
    # unmerged, which costs precision, not findings.
    merge = _guarded(call, lambda: None, "cross-batch merge")
    return plan if merge is None else apply_merge(merge, plan)


def _as_plan(result) -> ConsolidationPlan:
    if isinstance(result, dict):
        result = ConsolidationPlan.model_validate(result)
    if not isinstance(result, ConsolidationPlan) or not result.clusters:
        raise ValueError(f"clustering returned no usable plan: {type(result).__name__}")
    return result


def _guarded[T](call: Callable[[], T], fallback: Callable[[], T], what: str) -> T:
    """Run an LLM call, retry once, then fall back deterministically.

    Consolidation runs after every per-trace pass has been paid for, so an
    unhandled 429 or a malformed structured output there discards the whole run.
    Worse for the benchmark, it discards it *asymmetrically*: one arm of the
    model comparison losing consolidation while the other completes would look
    like a quality difference rather than an outage.
    """
    for attempt in (1, 2):
        try:
            return call()
        except Exception as exc:
            print(
                f"[engine] {what}: attempt {attempt} failed ({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
    print(f"[engine] {what}: falling back to deterministic clustering", file=sys.stderr)
    return fallback()
