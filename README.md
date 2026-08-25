# Engine Benchmarking System

An end-to-end benchmark for **Engine**-style agents — systems that read production traces of an AI application and report the errors hiding in them.

**Assignment Submissions Notes**
1. Start with high level architecture diagrams here : `docs/handwritten/`   
2. Responses to questions sent in assignment instruction questions : `docs/adi-notes.md`  
3. Some known issues I'm aware of in this build : `docs/known-issues.md`  
4. Final reports on gpt-5.1-mini dummy Engine benchmarking run - `docs/benchmarking_runs/report.md`  
5. A more readable capture of the benchmark report in a LaTeX report - `docs/paper/main.pdf`  

## The idea

Evaluating a trace error-analysis agent has a bootstrapping problem: real traces come without ground-truth labels. This benchmark manufactures its own ground truth by **ablation**:

1. **Generate** a stratified synthetic input corpus (dimension grid × personas, including adversarial ones).
2. **Trace** each input against a real target app through a black-box harness.
3. **Ablate**: inject known errors into a subset of the traces — either by corrupting recorded content (`replay_edit`) or by re-running the input with a faulty dependency (`dependency_fault`) — keeping full bookkeeping of what went where. A control split stays byte-identical.
4. **Analyze**: the Engine under test sees only a leak-stripped trace export plus the category taxonomy, and returns a predicted issueboard.
5. **Score** predictions against the injection record with an exact-key matcher, across four independent axes: category detection, per-error localization, severity calibration, and description fidelity. Unmatched predictions surface as auditable hidden-error candidates, so reported precision is an honest lower bound.

The benchmark is fully **app-agnostic**: the target app is reached only through a config file and standard LangGraph server surfaces.

Full design: [`docs/architecture/`](docs/architecture/00-overview.md) (one doc per stage) · plan: [`docs/execution-plan.md`](docs/execution-plan.md) · results write-up: [`docs/paper/main.pdf`](docs/paper/main.pdf)

## Tech stack

- **Python 3.12** managed with `uv`; contracts throughout are **Pydantic** schemas (`benchmark/schemas/`)
- **LangGraph** for both apps in `apps/` — the target app (`create_react_agent` support agent with RAG + ticket tools) and the Engine under test — each served via `langgraph` server
- **LangSmith** for trace collection, isolated behind a swappable `TraceStore` boundary (`benchmark/tracing/`)
- **OpenAI** models for input generation, the ablation agent, and the Engine

## Running it

```bash
cp .env.example .env       # fill in OPENAI_API_KEY, LANGSMITH_API_KEY
./scripts/run_submission.sh            # pre-warms the generation cache, then runs the full pipeline
./scripts/run_submission.sh --resume data/pipeline/submission   # pick a killed run back up
```

Or invoke the pipeline directly:

```bash
uv run python -m benchmark.pipeline run --config configs/pipeline/submission.yaml
```

Useful flags: `--resume <run_dir>` (stage-checkpointed resume), `--engine-model <id>` (the model-comparison axis), `--run-id` / `--artifacts-root`. Configs live in `configs/` — `pipeline/mini.yaml` for a small run, `pipeline/submission.yaml` for the full 400-trace run; `taxonomy.yaml` is the shared error-category vocabulary.

Artifacts land in `data/pipeline/<run_id>/`: every stage checkpoints its outputs (`inputs.json`, `traces.json`, `ground_truth_issueboard.json`, `predicted_issueboard.json`, …) and the run ends with `report.md` / `report.json`.

## Development

```bash
./scripts/ci.sh            # ruff + full test suite (network-free)
```

Per-stage live smoke scripts (`scripts/*_smoke.py`) exercise each stage against real services.
