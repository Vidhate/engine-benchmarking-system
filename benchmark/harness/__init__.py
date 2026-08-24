"""Stage II — input orchestration harness (docs/architecture/03-trace-harness.md).

This package is the ONLY LangGraph/LangSmith-aware component in `benchmark/`.
It drives the target app exclusively through `langgraph_sdk` against
`configs/target_app.yaml`, normalizes the resulting LangSmith run trees into
our own `Trace` schema, and writes them into the Phase-0 `TraceStore`.
Everything downstream reads the store; no LangSmith type crosses that line.

Public surface:

* `run_harness(inputs, cfg, store, ...) -> (OutputDataset, TraceDataset)`
  — the batch entrypoint: single-turn prompts and multi-turn persona
  conversations, concurrent, resumable, leak-scrubbed.
* `Harness.replay(thread_ref, checkpoint_ref, corrupted_state,
  remaining_user_messages) -> Trace` — Mode A, via LangGraph time-travel.
* `Harness.run_with_faults(input_spec, fault_config) -> Trace` — Mode C, via
  the app's declared `configurable` fault keys.
"""

from benchmark.harness.client import AppResponse, LangGraphAppClient, TargetAppClient
from benchmark.harness.collector import (
    IngestionTimeout,
    LangSmithCollector,
    TraceCollector,
    TurnCoverageError,
    TurnHint,
    VacuousProjectionError,
)
from benchmark.harness.config import load_target_app_config
from benchmark.harness.faults import (
    FaultNotActivated,
    UndeclaredFault,
    activation_evidence,
    fault_configurable,
)
from benchmark.harness.ids import session_id_for, trace_id_for
from benchmark.harness.runner import (
    AmbiguousCheckpoint,
    Harness,
    Quarantine,
    replay,
    run_harness,
    run_with_faults,
)
from benchmark.harness.scrub import LeakDetected, assert_no_leak, leak_tokens
from benchmark.harness.simulator import (
    DONE_TOKEN,
    OpenAIUserSimulator,
    ScriptedUserSimulator,
    UserSimulator,
)

__all__ = [
    "DONE_TOKEN",
    "AmbiguousCheckpoint",
    "AppResponse",
    "FaultNotActivated",
    "Harness",
    "IngestionTimeout",
    "LangGraphAppClient",
    "LangSmithCollector",
    "LeakDetected",
    "OpenAIUserSimulator",
    "Quarantine",
    "ScriptedUserSimulator",
    "TargetAppClient",
    "TraceCollector",
    "TurnCoverageError",
    "TurnHint",
    "UndeclaredFault",
    "UserSimulator",
    "VacuousProjectionError",
    "activation_evidence",
    "assert_no_leak",
    "fault_configurable",
    "leak_tokens",
    "load_target_app_config",
    "replay",
    "run_harness",
    "run_with_faults",
    "session_id_for",
    "trace_id_for",
]
