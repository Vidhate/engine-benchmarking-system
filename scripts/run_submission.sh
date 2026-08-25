#!/usr/bin/env bash
# Submission benchmark run: loads .env, pre-warms the generation cache, then
# runs the full pipeline. Extra args are passed through to the pipeline CLI,
# so a died run resumes with:
#
#   ./scripts/run_submission.sh --resume data/pipeline/submission
#
set -euo pipefail
cd "$(dirname "$0")/.."

# --- dotenv ---------------------------------------------------------------
if [[ ! -f .env ]]; then
  echo "ERROR: .env not found in repo root (needs OPENAI_API_KEY, LANGSMITH_API_KEY)" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
: "${OPENAI_API_KEY:?OPENAI_API_KEY is empty after loading .env}"
: "${LANGSMITH_API_KEY:?LANGSMITH_API_KEY is empty after loading .env}"

# --- 1. pre-warm the expansion cache (idempotent; ~free when already warm) --
echo "[run_submission] pre-warming generation cache..."
uv run python - <<'PY'
from benchmark.generation.config_loader import load_generation_config
from benchmark.generation.generators import generate_inputs

cfg = load_generation_config("configs/generation/submission.yaml")
d = generate_inputs(cfg, cache_dir="data/expansion_cache")
print(f"[run_submission] {len(d.inputs)} inputs cached")
PY

# --- 2. the timed run -------------------------------------------------------
echo "[run_submission] starting pipeline (expect ~4-5.5h on a first run)..."
exec uv run python -m benchmark.pipeline run \
  --config configs/pipeline/submission.yaml "$@"
