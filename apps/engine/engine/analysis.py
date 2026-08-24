"""The per-trace analysis pass and the meta consolidation pass.

Both are plain functions over an injected chat model, so unit tests drive them
with a fake model and never touch the network.
"""

from __future__ import annotations

import json

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from engine import prompts
from engine.consolidate import assemble_board, fallback_plan
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
            if isinstance(response, AIMessage) and response.text():
                log.append(f"ANALYST NOTES:\n{response.text()}")
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
) -> Issueboard:
    """Cluster raw findings into canonical issues and merge over the seed board.

    The model decides only the clustering; `assemble_board` does the merge, so
    the board's invariants do not depend on the model behaving.
    """
    seed = seed_board or SeedIssueboard()
    if not findings:
        return assemble_board(ConsolidationPlan(), [], seed, categories)

    system = prompts.CONSOLIDATION_SYSTEM.format(
        categories=prompts.format_categories(categories),
        seed_issues=prompts.format_seed_issues(seed.issues),
    )
    task = prompts.CONSOLIDATION_TASK.format(findings=prompts.format_findings(findings))
    plan = model.with_structured_output(ConsolidationPlan).invoke(
        [SystemMessage(system), HumanMessage(task)]
    )
    if isinstance(plan, dict):
        plan = ConsolidationPlan.model_validate(plan)
    if not isinstance(plan, ConsolidationPlan) or not plan.clusters:
        plan = fallback_plan(findings)
    return assemble_board(plan, findings, seed, categories)
