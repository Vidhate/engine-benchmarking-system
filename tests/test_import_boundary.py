"""Gate: benchmark/ must never import from apps/* (the black-box boundary).

The target app and Engine are reached exclusively via config files + the
LangGraph Server API. Any `import apps...` under benchmark/ is a violation.
"""

import ast
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).parent.parent / "benchmark"


def iter_imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_benchmark_never_imports_apps():
    violations = [
        f"{path.relative_to(BENCHMARK_ROOT.parent)}: imports {module}"
        for path in sorted(BENCHMARK_ROOT.rglob("*.py"))
        for module in iter_imports(path)
        if module == "apps" or module.startswith("apps.")
    ]
    assert not violations, "black-box boundary violated:\n" + "\n".join(violations)


def test_boundary_scan_sees_benchmark_files():
    # Guard against the test silently passing because the scan found nothing.
    assert any(BENCHMARK_ROOT.rglob("*.py")), "benchmark/ package not found"
