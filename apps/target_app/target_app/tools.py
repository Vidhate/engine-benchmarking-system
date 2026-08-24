"""The target app's two tools: one retrieval tool and one stubbed action tool.

Both read their fault instruction from `config.configurable[<declared key>]`.
With no key armed they behave completely normally.
"""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langsmith import traceable

from target_app.corpus import load_corpus
from target_app.retriever import Retriever
from target_app.shims import (
    FAULT_RETRIEVER_KEY,
    FAULT_TOOL_KEY,
    Fault,
    apply_retriever_fault,
    apply_tool_fault,
    read_fault,
)

RETRIEVER = Retriever(load_corpus())
DEFAULT_K = 3
DEFAULT_ETA_HOURS = 24


# The armed fault is passed to the traced retrieval step out-of-band. If it
# were an argument, `@traceable` would record it verbatim in the retriever
# span's inputs — "Fault(behavior='stale')" sitting in the trace we are asking
# an Engine to analyse blind.
_ARMED_RETRIEVER_FAULT: ContextVar[Fault | None] = ContextVar(
    "armed_retriever_fault", default=None
)


@traceable(run_type="retriever", name="corpus_search")
def _retrieve(query: str, k: int) -> list[dict[str, Any]]:
    """The retrieval step itself — emitted as its own retriever span."""
    hits = apply_retriever_fault(
        _ARMED_RETRIEVER_FAULT.get(), query=query, retriever=RETRIEVER, k=k
    )
    return [hit.as_payload() for hit in hits]


@tool(parse_docstring=False)
def rag_search(query: str, config: RunnableConfig) -> str:
    """Search the Nimbus Notes support knowledge base.

    Use this before answering any question about Nimbus Notes features,
    pricing, policies, or troubleshooting steps. Returns the most relevant
    knowledge-base documents as JSON.

    Args:
        query: A natural-language description of what to look up.
    """
    token = _ARMED_RETRIEVER_FAULT.set(read_fault(config, FAULT_RETRIEVER_KEY))
    try:
        results = _retrieve(query, DEFAULT_K)
    finally:
        _ARMED_RETRIEVER_FAULT.reset(token)
    return json.dumps({"query": query, "results": results}, indent=2)


@tool(parse_docstring=False)
def create_ticket(
    subject: str,
    description: str,
    priority: Literal["low", "normal", "high"] = "normal",
    config: RunnableConfig = None,  # noqa: RUF013 - injected by LangChain
) -> str:
    """File a support ticket for a human agent to pick up.

    Use this only when the knowledge base cannot resolve the issue, or when
    the user explicitly asks for a human, a refund, or an account change that
    support has to perform.

    Args:
        subject: One-line summary of the request.
        description: Everything the human agent needs, including what was
            already tried.
        priority: "low", "normal", or "high".
    """
    digest = hashlib.sha1(f"{subject}|{description}".encode()).hexdigest()
    result: dict[str, Any] = {
        "status": "created",
        "ticket_id": f"NN-{digest[:6].upper()}",
        "subject": subject,
        "priority": priority,
        "eta_hours": DEFAULT_ETA_HOURS,
    }
    return json.dumps(apply_tool_fault(read_fault(config, FAULT_TOOL_KEY), result), indent=2)


TOOLS = [rag_search, create_ticket]
