"""Loading and inspecting the trace file.

Everything here is pure (no LLM, no network): the file is read once into a
`TraceIndex`, and the four trace-inspection operations are plain functions over
it. `engine/tools.py` is a thin LangChain wrapper around this module, which is
what lets the tools be unit-tested without a model.

Leak posture: the file is parsed through `models.Trace`, which declares only
the fields the Engine may read. Ablation bookkeeping (`ablation_ids`,
`injection_mode`, records, split labels) is discarded by pydantic before any
Engine code can touch it, and no symbol in this package refers to it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine.models import Category, Span, Trace

# Per-result character budget. Trace payloads (message lists, retrieved
# documents) are unbounded; an analysis agent that pulls one whole trace into
# context per tool call blows its window on the first big trace.
MAX_RESULT_CHARS = 6000
SNIPPET_CHARS = 240


def load_traces(path: str | Path) -> list[Trace]:
    """Read a trace file: either a `{traces: [...]}` dataset or a bare list."""
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        raw = payload.get("traces", [])
    elif isinstance(payload, list):
        raw = payload
    else:
        raise ValueError(f"unsupported trace file payload: {type(payload).__name__}")
    return [Trace.model_validate(item) for item in raw]


def load_categories(items: list[dict[str, Any]] | None) -> list[Category]:
    return [Category.model_validate(item) for item in (items or [])]


def truncate(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    """Shorten free text. NOT for tool results — see `fit_json`."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text) - limit} more characters]"


def _dumps(value: Any) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


def fit_json(payload: Any, items_key: str | None = None, limit: int = MAX_RESULT_CHARS) -> str:
    """Serialize a tool result within the character budget, keeping it valid JSON.

    Chopping the serialized string would hand the model a truncated object it
    cannot parse — the worst possible failure for a tool result, because the
    model cannot tell a malformed payload from an absent field. So oversized
    results shed whole items (and say how many), and the last resort is a valid
    envelope carrying the truncated text as a string value.
    """
    text = _dumps(payload)
    if len(text) <= limit:
        return text
    if items_key and isinstance(payload, dict) and isinstance(payload.get(items_key), list):
        items = payload[items_key]
        total = len(items)
        while len(items) > 1:
            items = items[: len(items) * 3 // 4]
            trimmed = {
                **payload,
                items_key: items,
                "truncated": f"showing {len(items)} of {total} {items_key}",
            }
            text = _dumps(trimmed)
            if len(text) <= limit:
                return text
    return _dumps({"truncated": True, "content": truncate(_dumps(payload), limit)})


class TraceIndex:
    """An in-memory, read-only view over one loaded trace file."""

    def __init__(self, traces: list[Trace]) -> None:
        self._traces = list(traces)
        self._by_id: dict[str, Trace] = {t.trace_id: t for t in self._traces}

    @classmethod
    def from_file(cls, path: str | Path) -> TraceIndex:
        return cls(load_traces(path))

    @property
    def traces(self) -> list[Trace]:
        return list(self._traces)

    @property
    def trace_ids(self) -> list[str]:
        return [t.trace_id for t in self._traces]

    def trace(self, trace_id: str) -> Trace | None:
        return self._by_id.get(trace_id)

    def spans(self, trace_id: str, turn_index: int | None = None) -> list[tuple[int, Span]]:
        """(turn_index, span) pairs, optionally narrowed to one turn."""
        trace = self._by_id.get(trace_id)
        if trace is None:
            return []
        return [
            (turn.turn_index, span)
            for turn in trace.turns
            if turn_index is None or turn.turn_index == turn_index
            for span in turn.spans
        ]

    def span(self, trace_id: str, span_id: str) -> tuple[int, Span] | None:
        for turn_index, span in self.spans(trace_id):
            if span.span_id == span_id:
                return turn_index, span
        return None

    # -- the four tool operations ------------------------------------------

    def get_trace(self, trace_id: str) -> str:
        """Trace overview: status, metadata, and per-turn conversation with a
        span table. Span payloads are NOT included — `read_span` fetches those."""
        trace = self._by_id.get(trace_id)
        if trace is None:
            return self._unknown_trace(trace_id)
        view = {
            "trace_id": trace.trace_id,
            "input_id": trace.input_id,
            "mode": trace.mode,
            "status": trace.status,
            "metadata": trace.metadata,
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "user_message": turn.user_message,
                    "final_response": turn.final_response,
                    "spans": [
                        {
                            "span_id": span.span_id,
                            "name": span.name,
                            "span_type": span.span_type,
                        }
                        for span in turn.spans
                    ],
                }
                for turn in trace.turns
            ],
        }
        return fit_json(view, "turns")

    def list_spans(self, trace_id: str, turn_index: int | None = None) -> str:
        """Span table for a trace: ids, names, types, parents, and a preview of
        each span's output — enough to choose which spans to `read_span`."""
        trace = self._by_id.get(trace_id)
        if trace is None:
            return self._unknown_trace(trace_id)
        rows = [
            {
                "turn_index": t_index,
                "span_id": span.span_id,
                "parent_span_id": span.parent_span_id,
                "name": span.name,
                "span_type": span.span_type,
                "has_error": bool(span.attributes.get("error")),
                "output_preview": truncate(_dumps(span.outputs), SNIPPET_CHARS),
            }
            for t_index, span in self.spans(trace_id, turn_index)
        ]
        if not rows and turn_index is not None:
            return _dumps({"error": f"no turn with turn_index={turn_index} in {trace_id!r}"})
        return fit_json({"trace_id": trace_id, "spans": rows}, "spans")

    def read_span(self, trace_id: str, span_id: str) -> str:
        """One span in full: inputs, outputs, attributes."""
        found = self.span(trace_id, span_id)
        if found is None:
            if trace_id not in self._by_id:
                return self._unknown_trace(trace_id)
            known = [span.span_id for _, span in self.spans(trace_id)]
            return _dumps({"error": f"no span {span_id!r} in {trace_id!r}", "known_span_ids": known})
        turn_index, span = found
        view = span.model_dump(mode="json")
        view["turn_index"] = turn_index
        return fit_json(view)

    def search_text(self, query: str, trace_id: str | None = None, limit: int = 25) -> str:
        """Case-insensitive substring search over trace text.

        Searches turn messages and span payloads, returning locations plus a
        snippet — the cheap way to check "does the retrieved document actually
        say what the answer claims?" without reading every span.
        """
        needle = query.strip().lower()
        if not needle:
            return _dumps({"error": "empty query"})
        targets = (
            [self._by_id[trace_id]]
            if trace_id and trace_id in self._by_id
            else ([] if trace_id else self._traces)
        )
        if trace_id and not targets:
            return self._unknown_trace(trace_id)

        hits: list[dict[str, Any]] = []
        for trace in targets:
            for turn in trace.turns:
                for field in ("user_message", "final_response"):
                    hits += self._hits(
                        needle,
                        getattr(turn, field),
                        {
                            "trace_id": trace.trace_id,
                            "turn_index": turn.turn_index,
                            "location": f"turn.{field}",
                        },
                    )
                for span in turn.spans:
                    for field in ("inputs", "outputs", "attributes"):
                        hits += self._hits(
                            needle,
                            _dumps(getattr(span, field)),
                            {
                                "trace_id": trace.trace_id,
                                "turn_index": turn.turn_index,
                                "span_id": span.span_id,
                                "span_name": span.name,
                                "location": f"span.{field}",
                            },
                        )
        result = {"query": query, "hit_count": len(hits), "hits": hits[:limit]}
        return fit_json(result, "hits")

    @staticmethod
    def _hits(needle: str, haystack: str, where: dict[str, Any]) -> list[dict[str, Any]]:
        position = haystack.lower().find(needle)
        if position < 0:
            return []
        start = max(0, position - SNIPPET_CHARS // 2)
        return [{**where, "snippet": haystack[start : start + SNIPPET_CHARS]}]

    def _unknown_trace(self, trace_id: str) -> str:
        return _dumps({"error": f"unknown trace_id {trace_id!r}", "known_trace_ids": self.trace_ids})
