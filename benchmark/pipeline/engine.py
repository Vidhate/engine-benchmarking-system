"""Invoking the Engine — the black-box boundary, from the benchmark's side.

`langgraph_sdk` against `configs/engine.yaml`. Nothing else. The Engine sees a
path to the leak-stripped trace file, the seed issueboard, and the category
vocabulary; the run output IS the updated issueboard.

Three facts are load-bearing and easy to lose (apps/engine/README.md, and the
Phase 6 hand-off notes):

1. **`recursion_limit` must be passed.** The LangGraph server's default is 25
   and the loop costs `2 + ceil(n / N)` supersteps, so the default caps a run
   at roughly 23 traces — and the failure arrives mid-corpus, after the money
   is spent.
2. **`analysis_concurrency` must be passed.** The default of 8 projects to
   ~35 min over 300 traces; 16 (the clamp ceiling) to ~21 min.
3. **`board_id` must be re-stamped.** The Engine computes a hash of the same
   *shape* as `benchmark.schemas.io.content_hash`, but over its own `Issue`
   model, which has no `injection_mode` field — so the canonical JSON differs
   and the values do not match on identical content. It is the Engine's label,
   not a benchmark dataset id. Never assert equality between the two.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from benchmark.pipeline.config import EngineStageConfig
from benchmark.pipeline.contracts import EngineInvocation
from benchmark.schemas import EngineAppConfig, ErrorCategory, Issueboard
from benchmark.schemas.io import stamp_dataset_id

log = logging.getLogger("benchmark.pipeline.engine")


class EngineRunFailed(RuntimeError):
    """The Engine run errored, or returned something that is not an issueboard."""


def _seed_payload(board: Issueboard) -> dict[str, Any]:
    """The seed board as the Engine may see it.

    `injection_mode` is dropped rather than serialized as null: it is
    ground-truth-side vocabulary, and the Engine's own loader tolerating it is
    not a reason to send it.
    """
    payload = board.model_dump(mode="json")
    for issue in payload.get("issues", []):
        issue.pop("injection_mode", None)
    return payload


class LangGraphEngineInvoker:
    """One Engine run, driven exclusively through the LangGraph Server API."""

    def __init__(self, app: EngineAppConfig, *, client: Any = None):
        self.app = app
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            from langgraph_sdk import get_sync_client  # noqa: PLC0415

            self._client = get_sync_client(url=self.app.base_url)
        return self._client

    def recorded_models(self, thread_id: str) -> list[str]:
        """Which model the SERVER recorded for each run on this thread.

        Read back rather than assumed: LangGraph silently declines to inject a
        run config whose node annotation it does not recognise, and the symptom
        is both arms of a model comparison quietly running the same model.
        Best-effort — a missing readback endpoint does not fail the run.
        """
        models: list[str] = []
        try:
            runs = list(self.client.runs.list(thread_id))
        except Exception as exc:  # noqa: BLE001 - provenance is best-effort
            log.warning("could not read the run config back: %s: %s", type(exc).__name__, exc)
            return []
        for run in runs:
            record = run if isinstance(run, dict) else getattr(run, "__dict__", {})
            config = (record.get("kwargs") or {}).get("config") or record.get("config") or {}
            configurable = config.get("configurable") or {}
            if self.app.model_configurable_key in configurable:
                models.append(configurable[self.app.model_configurable_key])
        return models

    def __call__(
        self,
        *,
        trace_file: str | Path,
        seed_board: Issueboard,
        categories: list[ErrorCategory],
        engine: EngineStageConfig,
    ) -> EngineInvocation:
        trace_file = Path(trace_file).resolve()
        if not trace_file.exists():
            raise FileNotFoundError(f"engine trace file not found: {trace_file}")
        trace_count = len(json.loads(trace_file.read_text()).get("traces", []))

        config = {
            "configurable": {
                self.app.model_configurable_key: engine.model,
                "analysis_concurrency": engine.analysis_concurrency,
            },
            "recursion_limit": engine.recursion_limit,
        }
        log.info(
            "engine run: %s traces, model=%s, analysis_concurrency=%s, recursion_limit=%s",
            trace_count,
            engine.model,
            engine.analysis_concurrency,
            engine.recursion_limit,
        )

        thread = self.client.threads.create()
        thread_id = thread["thread_id"] if isinstance(thread, dict) else thread.thread_id
        started = time.time()
        result = self.client.runs.wait(
            thread_id,
            self.app.assistant_id,
            input={
                "trace_file": str(trace_file),
                "seed_issueboard": _seed_payload(seed_board),
                "categories": [c.model_dump(mode="json") for c in categories],
            },
            config=config,
        )
        seconds = time.time() - started

        if not isinstance(result, dict):
            raise EngineRunFailed(f"engine returned {type(result).__name__}, not a mapping")
        if result.get("__error__"):
            raise EngineRunFailed(f"engine run failed: {result['__error__']}")
        if "issues" not in result or "occurrences" not in result:
            raise EngineRunFailed(
                f"engine did not return an issueboard — keys were {sorted(result)}"
            )

        # Re-stamp: the Engine's board_id is its own label, not our dataset id.
        board = stamp_dataset_id(Issueboard.model_validate(result))
        return EngineInvocation(
            board=board,
            raw_output=result,
            seconds=seconds,
            thread_id=thread_id,
            recorded_models=self.recorded_models(thread_id),
            trace_count=trace_count,
        )
