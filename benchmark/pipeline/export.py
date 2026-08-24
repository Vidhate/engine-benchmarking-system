"""The one file that crosses into the Engine — and the audit that guards it.

Phase 5 owns writing the leak-stripped export; the pipeline still audits it
before handing the path to the Engine. That is not distrust of Phase 5, it is
where the cost lands: a leaked `ablation_ids` or `injection_mode` does not
crash anything, it silently turns the benchmark into a lookup exercise and the
resulting numbers look *better*, not worse. A guard whose failure mode is
invisible has to be checked at the boundary that consumes it.

`write_leak_stripped_export` exists for the same reason a fake ablation stage
does: the miniature run needs an export before Phase 5 lands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.harness.scrub import STRUCTURAL_TOKENS, find_leaked_keys, find_leaks
from benchmark.schemas.traces import Trace, TraceDataset

#: Ground-truth-side vocabulary. None of it may appear in a file the Engine
#: reads: each token names either the fact that a trace was manufactured or
#: which manufacturing recipe produced it.
GROUND_TRUTH_TOKENS: tuple[str, ...] = (
    "injection_mode",
    "replay_edit",
    "dependency_fault",
    "ablation_id",
    "ablation_ids",
    "ground_truth",
    "control_input_ids",
    "ablate_input_ids",
    "before_after",
)

EXPORT_LEAK_TOKENS: tuple[str, ...] = tuple(
    dict.fromkeys((*STRUCTURAL_TOKENS, *GROUND_TRUTH_TOKENS))
)


class ExportLeak(Exception):
    """The file bound for the Engine names the ground truth."""


def strip_trace(trace: Trace) -> dict[str, Any]:
    """One trace as the Engine may see it: everything except `ablation_ids`."""
    return trace.model_dump(mode="json", exclude={"ablation_ids"})


def export_payload(dataset: TraceDataset) -> dict[str, Any]:
    return {
        "dataset_id": dataset.dataset_id,
        "parent_dataset_id": dataset.parent_dataset_id,
        "traces": [strip_trace(t) for t in dataset.traces],
    }


def write_leak_stripped_export(dataset: TraceDataset, path: str | Path) -> Path:
    """Write the Engine's `trace_file`, with the internal fields removed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = export_payload(dataset)
    assert_export_clean(payload, where=str(path))
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def audit_export(payload: Any) -> dict[str, list[str]]:
    """Ground-truth fingerprints found in a payload bound for the Engine."""
    return {
        "tokens": find_leaks(payload, EXPORT_LEAK_TOKENS),
        "keys": find_leaked_keys(payload),
    }


def assert_export_clean(payload: Any, *, where: str) -> None:
    found = audit_export(payload)
    if found["tokens"] or found["keys"]:
        raise ExportLeak(
            f"the trace file bound for the Engine names the ground truth ({where}): "
            f"tokens={found['tokens']} keys={found['keys']}"
        )


def assert_export_file_clean(path: str | Path) -> dict[str, Any]:
    """Audit an export file on disk and return its parsed payload.

    Also validates that every trace still parses as a `Trace` — an export the
    Engine cannot load is a failed run that would otherwise be discovered
    twenty minutes into a 300-trace pass.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"engine export not found: {path}")
    payload = json.loads(path.read_text())
    assert_export_clean(payload, where=str(path))
    for raw in payload.get("traces", []):
        Trace.model_validate(raw)
    return payload
