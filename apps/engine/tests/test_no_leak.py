"""Anti-leak: the Engine sees only the traces, the seed board, and the vocabulary.

`docs/architecture/05-engine-simulation.md` requires that ablation bookkeeping
never reaches the Engine. Phase 5 strips it on export; this suite asserts the
*receiving* side independently, so a stripping bug upstream cannot silently turn
into an Engine that reads injection markers and scores suspiciously well.

Two complementary checks:
  * behavioural — a trace file that still carries ablation fields loads fine,
    and none of those fields survive into anything Engine code can reach;
  * structural  — no symbol anywhere in `engine/` names them, so the Engine
    cannot start reading them without this test being deleted first.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from engine.models import Trace
from engine.traces import TraceIndex, load_traces

ENGINE_PACKAGE = Path(__file__).resolve().parents[1] / "engine"

# The leak surface named in the architecture doc and the Phase 0 schemas.
FORBIDDEN_TOKENS = (
    "ablation_id",
    "ablation_ids",
    "ablation_record",
    "AblationRecord",
    "AblationSpec",
    "AblationSplit",
    "injection_mode",
    "replay_edit",
    "dependency_fault",
    "ground_truth",
    "fault_config",
    "FaultConfig",
)


def test_a_file_that_still_carries_ablation_fields_loads_fine(extra_fields_file):
    """Tolerance, not dependence: the loader neither requires nor rejects them."""
    traces = load_traces(extra_fields_file)
    assert [t.trace_id for t in traces] == ["trace-extra-1"]
    assert traces[0].turns[0].spans[0].span_id == "s-x-0"


def test_a_file_without_them_loads_identically(traces_file):
    assert len(load_traces(traces_file)) == 6


@pytest.mark.parametrize("token", ["ablation_ids", "injection_mode", "ablation_record", "split"])
def test_ablation_fields_are_dropped_at_parse_time(extra_fields_file, token):
    raw = extra_fields_file.read_text()
    assert token in raw, "fixture must actually contain the field being guarded against"
    trace = load_traces(extra_fields_file)[0]
    assert not hasattr(trace, token)
    assert token not in trace.model_dump_json()


def test_no_ablation_marker_survives_into_any_tool_result(extra_fields_file):
    """The tools are the Engine's whole window onto the file. If a marker is
    invisible here, no prompt can reach it."""
    index = TraceIndex.from_file(extra_fields_file)
    blob = json.dumps(
        [
            index.get_trace("trace-extra-1"),
            index.list_spans("trace-extra-1"),
            index.read_span("trace-extra-1", "s-x-0"),
        ]
    ).lower()
    for token in FORBIDDEN_TOKENS:
        assert token.lower() not in blob, f"{token} reachable through the tool surface"
    assert "abl-001" not in blob

    # search_text echoes its own query, so assert on the hits instead.
    for probe in ("abl-001", "injected", "replay_edit", "injection_mode"):
        assert json.loads(index.search_text(probe))["hit_count"] == 0


def test_span_level_markers_are_dropped_too(extra_fields_file):
    index = TraceIndex.from_file(extra_fields_file)
    span = json.loads(index.read_span("trace-extra-1", "s-x-0"))
    assert "ablation_id" not in span
    assert "injected" not in span
    assert set(span) == {
        "span_id", "parent_span_id", "name", "span_type",
        "start_time", "end_time", "inputs", "outputs", "attributes", "turn_index",
    }


def test_the_trace_model_declares_no_ablation_field():
    assert not any("ablation" in field for field in Trace.model_fields)
    assert "injection_mode" not in Trace.model_fields


def code_symbols(path: Path) -> set[str]:
    """Every identifier and literal string a module's *code* mentions.

    Parsed rather than grepped so that prose — comments and docstrings that
    explain what the Engine deliberately does not read — is not mistaken for a
    reference. Comments never enter the AST; docstrings are subtracted.
    """
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            first = body[0] if body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))

    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            symbols.add(node.id)
        elif isinstance(node, ast.Attribute):
            symbols.add(node.attr)
        elif isinstance(node, ast.arg):
            symbols.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            symbols.add(node.name)
        elif isinstance(node, ast.alias):
            symbols.update(filter(None, (node.name, node.asname)))
        elif isinstance(node, ast.keyword) and node.arg:
            symbols.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                symbols.add(node.value)
    return symbols


def test_engine_source_never_names_the_ablation_surface():
    """Structural guard: the Engine cannot *begin* reading these without an edit
    that trips this test."""
    offenders = [
        f"{path.name}: {token}"
        for path in sorted(ENGINE_PACKAGE.rglob("*.py"))
        for token in FORBIDDEN_TOKENS
        if any(token in symbol for symbol in code_symbols(path))
    ]
    assert offenders == []


def test_the_structural_guard_would_catch_a_real_reference(tmp_path):
    """Guards the guard: prose is exempt, code is not."""
    prose = tmp_path / "prose.py"
    prose.write_text('"""We never read injection_mode."""\n# nor ablation_ids\n')
    assert not any(t in s for s in code_symbols(prose) for t in FORBIDDEN_TOKENS)

    real = tmp_path / "real.py"
    real.write_text("def f(trace):\n    return trace.ablation_ids\n")
    assert any("ablation_ids" in symbol for symbol in code_symbols(real))

    lookup = tmp_path / "lookup.py"
    lookup.write_text('def f(d):\n    return d["injection_mode"]\n')
    assert any("injection_mode" in symbol for symbol in code_symbols(lookup))


def test_engine_app_code_never_imports_the_benchmark_package():
    """The black-box boundary, from this side: `apps/engine` is a self-contained
    codebase that happens to speak the benchmark's JSON."""
    offenders = [
        path.name
        for path in sorted(ENGINE_PACKAGE.rglob("*.py"))
        if "import benchmark" in path.read_text() or "from benchmark" in path.read_text()
    ]
    assert offenders == []
