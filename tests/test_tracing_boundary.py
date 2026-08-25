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

**A second narrow exception, by design.** Two benchmark-side components call an
LLM of their own: the ablation agent that drafts errors, and the prompt
expander that writes generation inputs. Both used to hand-roll `urllib` against
the chat-completions endpoint, and both wanted the two things that transport
never gave them — a retry budget with 429 backoff, and a timeout that is not
the SDK's 600 s default. `langchain_openai` is those two things, so it is
allowed in exactly two declared files.

That exception is conditional, and the condition is asserted here: **those
calls must emit no LangSmith runs.** `LANGSMITH_TRACING` is set during a real
run — the harness needs it to collect the target app's traces — and LangChain
traces by default, so an unsuppressed benchmark-side call would land its run in
the collector's own project. That is two problems at once: the collector's
trace corpus gets polluted with runs that are not traces of the app under test,
and the ablation agent's proposals — which errors are about to be injected —
become readable by anyone with access to the project. So each allowlisted file
is separately asserted to wrap its invocations in `tracing_context(enabled=
False)` and to pass empty callbacks.
"""

import ast
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).parent.parent / "benchmark"
COLLECTION_PACKAGE = BENCHMARK_ROOT / "harness"

#: Never importable outside the collector (bar the allowlist below). These are
#: the ones that carry trace types, emit runs, or would otherwise re-couple the
#: system to a tracing vendor. `langchain_openai` is on the list because it is
#: an LLM client that traces by DEFAULT — the very thing this gate exists to
#: keep out — not because it carries trace types.
TRACING_SDKS = (
    "langsmith",
    "langgraph",
    "langchain",
    "langchain_core",
    "langchain_openai",
)

#: The LangGraph Server API client. Not a tracing SDK — it is how the benchmark
#: talks to a black-box app over HTTP.
SERVER_API_SDK = "langgraph_sdk"

#: The complete list of files outside the harness that may import it.
SERVER_API_ALLOWLIST = (BENCHMARK_ROOT / "pipeline" / "engine.py",)

#: The complete list of files outside the harness that may import an LLM client
#: from the LangChain family. EXACTLY these two, and each one is separately
#: asserted (below) to suppress tracing on every call it makes.
LLM_TRANSPORT_ALLOWLIST = (
    BENCHMARK_ROOT / "ablation" / "agent.py",
    BENCHMARK_ROOT / "generation" / "expander.py",
)

#: What an allowlisted file must import from `langsmith`, and nothing more: the
#: suppression context manager itself.
SUPPRESSION_SYMBOL = "tracing_context"


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
        if path not in LLM_TRANSPORT_ALLOWLIST
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


# ------------------------------------------- the LLM-transport allowlist

def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _suppression_blocks(tree: ast.Module) -> list[ast.With]:
    """Every `with tracing_context(enabled=False): ...` block in a module."""
    blocks = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
            if name != SUPPRESSION_SYMBOL:
                continue
            disabled = any(
                kw.arg == "enabled" and isinstance(kw.value, ast.Constant) and kw.value.value
                is False
                for kw in call.keywords
            )
            if disabled:
                blocks.append(node)
    return blocks


def _invocations(node: ast.AST) -> list[ast.Call]:
    """Every `<something>.invoke(...)` / `.ainvoke(...)` call under a node."""
    return [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and getattr(child.func, "attr", None) in ("invoke", "ainvoke", "stream", "batch")
    ]


def test_the_llm_transport_allowlist_names_only_real_files():
    for path in LLM_TRANSPORT_ALLOWLIST:
        assert path.exists(), f"the allowlist names a file that does not exist: {path}"


def test_the_allowlisted_llm_transports_import_only_the_client_and_the_suppression():
    """The exception buys an LLM client and the switch that mutes it — not the
    rest of the LangChain family, and above all not a trace-carrying SDK."""
    permitted = {"langchain_openai", "langsmith"}
    for path in LLM_TRANSPORT_ALLOWLIST:
        overreach = [
            module
            for module in iter_imports(path)
            if module.split(".")[0] in TRACING_SDKS and module.split(".")[0] not in permitted
        ]
        assert not overreach, (
            f"{path.name} imports {overreach} — the LLM-transport exception covers "
            f"{sorted(permitted)} only"
        )


def test_every_allowlisted_llm_call_is_wrapped_in_tracing_suppression():
    """The condition the exception was granted on, asserted per file.

    An unsuppressed benchmark-side call lands a run in the collector's own
    LangSmith project: it pollutes the trace corpus, and for the ablation agent
    it publishes which errors are about to be injected. So every invocation in
    these files must sit inside `tracing_context(enabled=False)`.
    """
    for path in LLM_TRANSPORT_ALLOWLIST:
        tree = _tree(path)
        blocks = _suppression_blocks(tree)
        assert blocks, (
            f"{path.name} is allowlisted for an LLM client but never enters "
            f"`with {SUPPRESSION_SYMBOL}(enabled=False):` — the allowlist entry is "
            f"conditional on that suppression"
        )
        all_calls = _invocations(tree)
        assert all_calls, f"{path.name} never invokes its model — has the transport moved?"
        suppressed = {id(call) for block in blocks for call in _invocations(block)}
        leaked = [
            f"{path.name}:{call.lineno}" for call in all_calls if id(call) not in suppressed
        ]
        assert not leaked, (
            f"model invocation(s) outside the tracing suppression: {leaked}. Every call "
            f"must go through the module's `_invoke_untraced` helper."
        )


def test_every_allowlisted_llm_call_also_passes_empty_callbacks():
    """Belt and braces, and both halves are needed.

    `tracing_context(enabled=False)` governs LangChain's own ambient tracer;
    an explicit empty `callbacks` list stops a handler installed by a caller
    from re-attaching one. Asserted on the live module object, not on source
    text, so a constant renamed to something harmless still fails here.
    """
    import importlib  # noqa: PLC0415

    for module_name in (
        "benchmark.ablation.agent",
        "benchmark.generation.expander",
    ):
        module = importlib.import_module(module_name)
        config = getattr(module, "NO_TRACING_CONFIG", None)
        assert isinstance(config, dict), f"{module_name} declares no NO_TRACING_CONFIG"
        assert config.get("callbacks") == [], (
            f"{module_name}.NO_TRACING_CONFIG must pass an empty callback list, got {config}"
        )


def test_the_suppression_check_can_actually_fail():
    """Guard against the AST walk passing because it matched nothing.

    A bare `.invoke()` outside the wrapper must be detected; if this stops
    holding, the two tests above are decorative.
    """
    tree = ast.parse("model.invoke(x)\nwith tracing_context(enabled=False):\n    pass\n")
    assert _suppression_blocks(tree), "the with-block matcher stopped matching"
    assert len(_invocations(tree)) == 1
    assert not {id(c) for b in _suppression_blocks(tree) for c in _invocations(b)}


def test_the_scan_actually_covers_modules_outside_the_harness():
    # Guard against the test passing because it scanned nothing.
    assert len(_outside_the_harness()) > 5, "the boundary scan found almost no modules to check"
