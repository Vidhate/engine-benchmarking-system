#!/usr/bin/env bash
# All three Phase-6 gates, end to end, against a freshly started server.
# Server-dependent checks live here rather than in pytest (they need a live
# LangGraph server and a real OpenAI key). Expect ~10 minutes for gates 2+3.
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$APP_DIR"

cleanup() { scripts/serve.sh stop >/dev/null 2>&1 || true; }
trap cleanup EXIT

uv run python scripts/gate1_contract.py   # offline: config + schema contract

scripts/serve.sh start
uv run python scripts/gate2_live_smoke.py # live: default model
uv run python scripts/gate3_model_swap.py # live: both arms of the comparison
echo
echo "ALL THREE GATES PASSED"
