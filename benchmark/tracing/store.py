"""TraceStore — the replaceable tracing boundary (docs/execution-plan.md).

v0: LangSmith is only the *collection* backend. The Phase 4 collector is the
sole LangSmith-aware component; it normalizes run trees into our Trace schema
and writes them here. Everything downstream (ablation, Engine input, scoring)
reads and writes traces exclusively through this interface, so dropping
LangSmith later replaces one collector implementation and nothing else.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from benchmark.schemas.traces import Trace


@runtime_checkable
class TraceStore(Protocol):
    def put(self, trace: Trace) -> None: ...

    def get(self, trace_id: str) -> Trace: ...

    def exists(self, trace_id: str) -> bool: ...

    def list_ids(self) -> list[str]: ...

    def __iter__(self) -> Iterator[Trace]: ...


class LocalTraceStore:
    """Filesystem store: one `<trace_id>.json` per trace, in our Trace schema."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, trace_id: str) -> Path:
        if "/" in trace_id or "\\" in trace_id or trace_id in ("", ".", ".."):
            raise ValueError(f"invalid trace_id: {trace_id!r}")
        return self.root / f"{trace_id}.json"

    def put(self, trace: Trace) -> None:
        self._path(trace.trace_id).write_text(trace.model_dump_json(indent=2) + "\n")

    def get(self, trace_id: str) -> Trace:
        path = self._path(trace_id)
        if not path.exists():
            raise KeyError(f"trace not found: {trace_id}")
        return Trace.model_validate_json(path.read_text())

    def exists(self, trace_id: str) -> bool:
        return self._path(trace_id).exists()

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def __iter__(self) -> Iterator[Trace]:
        for trace_id in self.list_ids():
            yield self.get(trace_id)
