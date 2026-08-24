"""Gate: `benchmark/harness/` is the ONLY LangSmith/LangChain-aware package.

docs/execution-plan.md, "Tracing backend": LangSmith is v0's *collection*
backend only. Everything downstream of collection — ablation, Engine input,
scoring, pipeline — reads and writes traces exclusively through the
`TraceStore` interface. Swapping the collector later must replace one package
and nothing else, which only stays true if no other module reaches for the
SDKs directly.

**One narrow exception, by design.** The same execution plan maps *Engine
invocation* onto `langgraph_sdk` against `configs/engine.yaml` — driving a
LangGraph Server is how the black-box contract is honoured, and there is no
other door. So `langgraph_sdk` (the *server API client*, which knows nothing
about traces) is allowed in exactly one declared file outside the harness, and
that file is separately asserted to touch no tracing SDK at all. Everything
else — `langsmith`, `langchain`, `langchain_core`, in-process `langgraph` —
stays inside the collector, everywhere, with no exceptions.
"""

import ast
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).parent.parent / "benchmark"
COLLECTION_PACKAGE = BENCHMARK_ROOT / "harness"

#: Never importable outside the collector. These are the ones that carry trace
#: types and would re-couple the system to a tracing vendor.
TRACING_SDKS = ("langsmith", "langgraph", "langchain", "langchain_core")

#: The LangGraph Server API client. Not a tracing SDK — it is how the benchmark
#: talks to a black-box app over HTTP.
SERVER_API_SDK = "langgraph_sdk"

#: The complete list of files outside the harness that may import it.
SERVER_API_ALLOWLIST = (BENCHMARK_ROOT / "pipeline" / "engine.py",)


def iter_imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def _outside_the_harness():
    return [p for p in sorted(BENCHMARK_ROOT.rglob("*.py")) if COLLECTION_PACKAGE not in p.parents]


def test_only_the_harness_imports_the_tracing_sdks():
    violations = [
        f"{path.relative_to(BENCHMARK_ROOT.parent)}: imports {module}"
        for path in _outside_the_harness()
        for module in iter_imports(path)
        if module.split(".")[0] in TRACING_SDKS
    ]
    assert not violations, "tracing boundary violated:\n" + "\n".join(violations)


def test_the_server_api_client_is_confined_to_the_allowlist():
    violations = [
        f"{path.relative_to(BENCHMARK_ROOT.parent)}: imports {SERVER_API_SDK}"
        for path in _outside_the_harness()
        if path not in SERVER_API_ALLOWLIST
        for module in iter_imports(path)
        if module.split(".")[0] == SERVER_API_SDK
    ]
    assert not violations, (
        "langgraph_sdk is allowed outside the harness only in "
        f"{[str(p) for p in SERVER_API_ALLOWLIST]}:\n" + "\n".join(violations)
    )


def test_the_allowlisted_engine_client_is_still_tracing_free():
    """The exception buys one HTTP client, not a re-opened tracing coupling."""
    for path in SERVER_API_ALLOWLIST:
        assert path.exists(), f"the allowlist names a file that does not exist: {path}"
        leaked = [m for m in iter_imports(path) if m.split(".")[0] in TRACING_SDKS]
        assert not leaked, f"{path} imports tracing SDKs: {leaked}"


def test_the_scan_actually_covers_modules_outside_the_harness():
    # Guard against the test passing because it scanned nothing.
    assert len(_outside_the_harness()) > 5, "the boundary scan found almost no modules to check"
