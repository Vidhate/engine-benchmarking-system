#!/usr/bin/env bash
# All four Phase-2 gates, end to end, against a freshly started server.
# Server-dependent checks live here rather than in pytest (they need a live
# LangGraph server, OpenAI, and LangSmith).
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"

cleanup() { scripts/serve.sh stop >/dev/null 2>&1 || true; }
trap cleanup EXIT

scripts/serve.sh start
uv run python scripts/gate1_invocations.py
uv run python scripts/gate2_trace_export.py
uv run python scripts/gate3_time_travel.py
uv run python scripts/gate4_shims.py
echo
echo "ALL FOUR GATES PASSED"
