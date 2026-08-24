"""Gate: `benchmark/harness/` is the ONLY LangSmith/LangGraph-aware package.

docs/execution-plan.md, "Tracing backend": LangSmith is v0's *collection*
backend only. Everything downstream of collection — ablation, Engine input,
scoring, pipeline — reads and writes traces exclusively through the
`TraceStore` interface. Swapping the collector later must replace one package
and nothing else, which only stays true if no other module reaches for the
SDKs directly.
"""

import ast
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).parent.parent / "benchmark"
COLLECTION_PACKAGE = BENCHMARK_ROOT / "harness"
TRACING_SDKS = ("langsmith", "langgraph_sdk", "langgraph", "langchain", "langchain_core")


def iter_imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def test_only_the_harness_imports_the_tracing_sdks():
    violations = [
        f"{path.relative_to(BENCHMARK_ROOT.parent)}: imports {module}"
        for path in sorted(BENCHMARK_ROOT.rglob("*.py"))
        if COLLECTION_PACKAGE not in path.parents
        for module in iter_imports(path)
        if module.split(".")[0] in TRACING_SDKS
    ]
    assert not violations, "tracing boundary violated:\n" + "\n".join(violations)


def test_the_scan_actually_covers_modules_outside_the_harness():
    # Guard against the test passing because it scanned nothing.
    outside = [
        p
        for p in BENCHMARK_ROOT.rglob("*.py")
        if COLLECTION_PACKAGE not in p.parents
    ]
    assert len(outside) > 5, "the boundary scan found almost no modules to check"
