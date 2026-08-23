"""Engine benchmarking system.

Benchmark-side code only. Must never import from apps/* — the target app and
the Engine are black boxes reached via config + the LangGraph Server API.
"""
