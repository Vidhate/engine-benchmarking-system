"""Stage II — input orchestration harness (docs/architecture/03-trace-harness.md).

This package is the ONLY LangGraph/LangSmith-aware component in `benchmark/`.
It drives the target app exclusively through `langgraph_sdk` against
`configs/target_app.yaml`, normalizes the resulting LangSmith run trees into
our own `Trace` schema, and writes them into the Phase-0 `TraceStore`.
Everything downstream reads the store; no LangSmith type crosses that line.
"""

from benchmark.harness.config import load_target_app_config
from benchmark.harness.ids import session_id_for, trace_id_for

__all__ = [
    "load_target_app_config",
    "session_id_for",
    "trace_id_for",
]
