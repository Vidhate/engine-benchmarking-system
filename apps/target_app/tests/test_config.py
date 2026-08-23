"""configs/target_app.yaml is the black-box contract — keep it in sync with the code.

The app publishes this file, so the app tests it: it must validate against the
Phase-0 schema *and* describe what the shims actually implement.

Importing `benchmark.schemas` here does not cross the black-box boundary — the
rule is that `benchmark/` must never import `apps/`, not the reverse.
"""

import json
import sys
from pathlib import Path

import yaml

from target_app.agent import LANGSMITH_PROJECT
from target_app.shims import (
    FAULT_BEHAVIORS,
    FAULT_LLM_KEY,
    FAULT_RETRIEVER_KEY,
    FAULT_TOOL_KEY,
)
from target_app.tools import TOOLS

APP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_DIR.parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "target_app.yaml"
LANGGRAPH_JSON = APP_DIR / "langgraph.json"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def test_the_published_config_validates_against_the_phase_0_schema():
    sys.path.insert(0, str(REPO_ROOT))
    from benchmark.schemas.configs import TargetAppConfig

    config = TargetAppConfig(**load_config())
    assert config.base_url == "http://127.0.0.1:2024"
    assert config.assistant_id == "target_app"
    assert config.langsmith_project == LANGSMITH_PROJECT
    assert config.max_turns_supported == 8
    assert set(config.fault_configurable_keys) == {"retriever", "tool", "llm"}


def test_declared_fault_keys_are_the_ones_the_shims_read():
    keys = load_config()["fault_configurable_keys"]
    assert keys == {
        "retriever": FAULT_RETRIEVER_KEY,
        "tool": FAULT_TOOL_KEY,
        "llm": FAULT_LLM_KEY,
    }
    assert set(keys.values()) == set(FAULT_BEHAVIORS)


def test_config_matches_the_served_endpoint():
    config = load_config()
    served = json.loads(LANGGRAPH_JSON.read_text())
    assert config["assistant_id"] in served["graphs"]
    assert config["base_url"] == "http://127.0.0.1:2024"  # langgraph dev default port
    assert config["langsmith_project"] == LANGSMITH_PROJECT
    assert config["max_turns_supported"] == 8


def test_langgraph_entrypoint_resolves_to_a_real_graph():
    served = json.loads(LANGGRAPH_JSON.read_text())
    module_path, _, attribute = served["graphs"]["target_app"].partition(":")
    assert (APP_DIR / module_path).exists()
    assert attribute == "graph"


def test_config_declares_no_more_than_the_two_tools_exist():
    """Sanity: the app is a two-tool app, as the plan requires."""
    assert len(TOOLS) == 2
