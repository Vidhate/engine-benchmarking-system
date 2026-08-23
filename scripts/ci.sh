#!/usr/bin/env bash
# CI gate: lint + tests. Run from repo root.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run ruff check benchmark tests
uv run pytest
