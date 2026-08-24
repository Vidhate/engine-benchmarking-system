"""Single place where OpenAI model ids are pinned (see docs/execution-plan.md).

The Engine-under-test model is NOT pinned here — it is the comparison axis,
passed at invocation via the engine app's `model_configurable_key`.
"""

# NOTE: the execution plan names "gpt-5.1-mini"; this account's API returns 404
# for that id, and both apps already default to "gpt-5-mini" for the same reason
# (apps/target_app/target_app/agent.py, apps/engine/engine/llm.py).

# Target app: small/fast — high-volume I/O, low latency, <=2 tool calls.
TARGET_APP_MODEL = "gpt-5-mini"

# Benchmark-side LLM roles: mid-tier.
GENERATION_MODEL = "gpt-5.1"
PERSONA_SIM_MODEL = "gpt-5.1"
ABLATION_AGENT_MODEL = "gpt-5.1"
DESCRIPTION_JUDGE_MODEL = "gpt-5.1"

# Engine comparison axis defaults (Phase 8): large ("Sol"-class) vs mini.
ENGINE_MODEL_LARGE = "gpt-5.1"
ENGINE_MODEL_MINI = "gpt-5-mini"
