"""The `TraceFilter` engine — declarative predicate steps over trace properties.

docs/architecture/04-ablation-engine.md, step 2: a filter selects the traces
where an error *can plausibly exist* ("has a tool span", "retrieval returned
docs"). It is data, not code, so a planner (or an LLM) can author one, step 3
can report why it matched too little, and a re-plan can relax it mechanically.

## Path grammar

Dotted segments into the `Trace` model, with `[*]` to fan out over a list and
`[i]` to take one position:

    status
    metadata.thread_id
    turns[*].final_response
    turns[0].spans[*].span_type
    turns[*].spans[*].outputs.documents

Resolution always returns a **list** of the values found (possibly empty). A
path that reaches nothing yields `[]` rather than raising — that is an ordinary
"this trace does not have one". A path whose *root* segment is not a Trace
field and not a derived field raises `UnknownFilterField`: that is a typo in a
plan, and silently matching nothing would turn it into "this error is not
expressible in the corpus", which is a completely different diagnosis.

## Derived fields

Convenience roots computed over the whole trace, because the useful predicates
("does this trace use retrieval at all") are otherwise three wildcards deep:

| field | value |
|---|---|
| `turn_count` | number of turns |
| `span_types` | every span type present, deduplicated |
| `span_names` | every span name present, deduplicated |
| `span_count` | total number of spans |
| `final_responses` | every turn's final response |
| `user_messages` | every turn's user message |
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from benchmark.schemas.ablation import FilterStep, TraceFilter
from benchmark.schemas.traces import Trace

_SEGMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)((?:\[(?:\*|\d+)])*)$")
_INDEX = re.compile(r"\[(\*|\d+)]")


class UnknownFilterField(KeyError):
    """A filter names a root field that does not exist — a plan bug, not a miss."""


def _dedup(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


DERIVED_FIELDS: dict[str, Any] = {
    "turn_count": lambda t: [len(t.turns)],
    "span_count": lambda t: [sum(len(turn.spans) for turn in t.turns)],
    "span_types": lambda t: _dedup(s.span_type for turn in t.turns for s in turn.spans),
    "span_names": lambda t: _dedup(s.name for turn in t.turns for s in turn.spans),
    "final_responses": lambda t: [turn.final_response for turn in t.turns],
    "user_messages": lambda t: [turn.user_message for turn in t.turns],
}


def _child(node: Any, name: str) -> Any:
    if isinstance(node, dict):
        return node.get(name, _MISSING)
    if hasattr(node, name):
        return getattr(node, name)
    return _MISSING


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()


def _descend(nodes: list[Any], name: str, indexes: list[str]) -> list[Any]:
    out: list[Any] = []
    for node in nodes:
        value = _child(node, name)
        if value is _MISSING or value is None:
            continue
        current = [value]
        for index in indexes:
            stepped: list[Any] = []
            for item in current:
                if not isinstance(item, (list, tuple)):
                    continue
                if index == "*":
                    stepped.extend(item)
                else:
                    position = int(index)
                    if 0 <= position < len(item):
                        stepped.append(item[position])
            current = stepped
        out.extend(current)
    return out


def resolve(trace: Trace, path: str) -> list[Any]:
    """Every value `path` reaches in `trace`, flattened. See the module docstring."""
    segments = path.split(".")
    head = _SEGMENT.match(segments[0])
    if head is None:
        raise UnknownFilterField(f"malformed filter field: {path!r}")
    root, root_indexes = head.group(1), _INDEX.findall(head.group(2))

    if root in DERIVED_FIELDS:
        if len(segments) > 1 or root_indexes:
            raise UnknownFilterField(
                f"derived field {root!r} is a leaf; {path!r} tries to descend into it"
            )
        return list(DERIVED_FIELDS[root](trace))

    if root not in type(trace).model_fields:
        raise UnknownFilterField(
            f"{root!r} is neither a Trace field nor a derived field "
            f"(known: {sorted(set(type(trace).model_fields) | set(DERIVED_FIELDS))})"
        )

    nodes = _descend([trace], root, root_indexes)
    for segment in segments[1:]:
        match = _SEGMENT.match(segment)
        if match is None:
            raise UnknownFilterField(f"malformed filter field: {path!r}")
        nodes = _descend(nodes, match.group(1), _INDEX.findall(match.group(2)))
    return nodes


# ----------------------------------------------------------------- operators

def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def _contains(value: Any, wanted: Any) -> bool:
    if isinstance(value, str) and isinstance(wanted, str):
        return wanted.lower() in value.lower()
    if isinstance(value, (list, tuple, set, dict)):
        return wanted in value
    return _as_text(wanted).lower() in _as_text(value).lower()


def _numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _step_matches(values: list[Any], step: FilterStep) -> bool:
    op = step.op
    if op == "exists":
        # `value=False` flips the test to "the trace must NOT have one" —
        # presence, not truthiness (an empty document list still exists).
        return bool(values) if step.value is not False else not values
    if op == "eq":
        return any(value == step.value for value in values)
    if op == "ne":
        # Deliberately "no resolved value equals": a field the trace does not
        # have vacuously satisfies !=, which is what "this trace is not an X"
        # means when X is expressed as a property.
        return all(value != step.value for value in values)
    if op == "contains":
        return any(_contains(value, step.value) for value in values)
    if op == "regex":
        pattern = re.compile(str(step.value), re.IGNORECASE)
        return any(pattern.search(_as_text(value)) for value in values)
    if op in ("gt", "lt"):
        bound = _numeric(step.value)
        if bound is None:
            raise ValueError(f"filter op {op!r} needs a numeric value, got {step.value!r}")
        numbers = [n for n in (_numeric(v) for v in values) if n is not None]
        return any(n > bound for n in numbers) if op == "gt" else any(n < bound for n in numbers)
    raise ValueError(f"unsupported filter op: {op!r}")


def matches(trace: Trace, trace_filter: TraceFilter) -> bool:
    """True when EVERY step holds. An empty filter matches everything."""
    return all(_step_matches(resolve(trace, step.field), step) for step in trace_filter.steps)


def eligible(
    traces: Sequence[Trace], trace_filter: TraceFilter, population_input_ids: Iterable[str]
) -> list[Trace]:
    """Traces matching `trace_filter` **within a population**, in stable order.

    The population is always the ablate set (docs/architecture/04-ablation-engine.md:
    "Filters and the min_eligible gate run within the ablate set only"), passed
    in rather than inferred so a caller cannot forget it. Sorted by trace_id so
    the downstream seeded sub-sample does not inherit the corpus's iteration
    order.
    """
    population = set(population_input_ids)
    return sorted(
        (t for t in traces if t.input_id in population and matches(t, trace_filter)),
        key=lambda t: t.trace_id,
    )
