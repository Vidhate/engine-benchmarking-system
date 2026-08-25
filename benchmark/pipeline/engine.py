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
4. **The HTTP client needs an explicit read timeout.** `runs.wait` holds ONE
   blocking request open for the entire Engine pass, and `get_sync_client`
   defaults to `read=300s`. A full-scale run is ~25 minutes, so the default
   aborts it five minutes in — after every per-trace analysis has been paid
   for and with nothing to show. The timeout is a config field
   (`engine.timeout_s`), not a constant, because it is a property of the corpus
   size rather than of this module.
5. **A response that will not parse must not be unrecoverable.** Every failure
   raised here carries the raw payload on the exception, so the caller can
   persist it before re-raising. Hours of Engine time must never be lost to a
   schema surprise.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

from benchmark.pipeline.config import EngineStageConfig
from benchmark.pipeline.contracts import EngineInvocation
from benchmark.pipeline.export import export_traces
from benchmark.schemas import EngineAppConfig, ErrorCategory, Issueboard
from benchmark.schemas.io import stamp_dataset_id

log = logging.getLogger("benchmark.pipeline.engine")

#: Only the read leg is long. A server that will not accept a connection in
#: five seconds is down, and waiting three hours to learn that helps nobody.
CONNECT_TIMEOUT_S = 5.0
WRITE_TIMEOUT_S = 60.0
POOL_TIMEOUT_S = 5.0


class EngineRunFailed(RuntimeError):
    """The Engine run errored, or returned something that is not an issueboard.

    Carries `raw_output`: whatever the server actually returned. The caller is
    expected to persist it — a 25-minute run that ends in a validation error is
    exactly the run whose response you need to read.
    """

    def __init__(self, message: str, raw_output: Any = None):
        super().__init__(message)
        self.raw_output = raw_output


class EngineModelMismatch(RuntimeError):
    """The server ran a different model from the one the run asked for.

    A hard failure, not a note. LangGraph silently declines to inject a run
    config whose node annotation it does not recognise, and the symptom is both
    arms of a model comparison quietly running the same model — a result that
    looks like a finding ("the two models score the same") and is not one.
    """


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

    def __init__(self, app: EngineAppConfig, *, client: Any = None, client_factory: Any = None):
        self.app = app
        self._client = client
        self._client_factory = client_factory

    def _build_client(self, engine: EngineStageConfig) -> Any:
        factory = self._client_factory
        if factory is None:
            from langgraph_sdk import get_sync_client  # noqa: PLC0415

            factory = get_sync_client
        return factory(
            url=self.app.base_url,
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT_S,
                read=engine.timeout_s,
                write=WRITE_TIMEOUT_S,
                pool=POOL_TIMEOUT_S,
            ),
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            raise RuntimeError(
                "the client is built on the first call, from the run's engine config "
                "(it carries the read timeout) — inject one to use this before then"
            )
        return self._client

    def recorded_models(self, thread_id: str) -> list[str] | None:
        """Which model the SERVER recorded for each run on this thread.

        Read back rather than assumed: LangGraph silently declines to inject a
        run config whose node annotation it does not recognise, and the symptom
        is both arms of a model comparison quietly running the same model.

        Three outcomes, and the caller acts differently on each, so they must
        not share a value:

        * `None` — the run records could not be read at all. Absent evidence:
          a server or SDK capability gap, and no reason to doubt the run.
        * `[]` — the records ARE readable and none of them carries the model
          key. That is not absent evidence, it is evidence of absence: the
          exact signature of a `configurable` entry the server declined to
          accept. The caller treats it as a failure.
        * a non-empty list — what actually ran.
        """
        models: list[str] = []
        try:
            runs = list(self.client.runs.list(thread_id))
        except Exception as exc:  # noqa: BLE001 - unreadable is a capability gap
            log.warning("could not read the run config back: %s: %s", type(exc).__name__, exc)
            return None
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
        trace_count = len(export_traces(json.loads(trace_file.read_text())))

        if self._client is None:
            self._client = self._build_client(engine)

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

        # Every raise below carries `result`: after a run this long, a response
        # that will not parse must still be readable afterwards.
        if not isinstance(result, dict):
            raise EngineRunFailed(
                f"engine returned {type(result).__name__}, not a mapping", raw_output=result
            )
        if result.get("__error__"):
            raise EngineRunFailed(f"engine run failed: {result['__error__']}", raw_output=result)
        if "issues" not in result or "occurrences" not in result:
            raise EngineRunFailed(
                f"engine did not return an issueboard — keys were {sorted(result)}",
                raw_output=result,
            )

        try:
            parsed = Issueboard.model_validate(result)
        except Exception as exc:
            raise EngineRunFailed(
                f"engine output does not validate as an Issueboard: {exc}", raw_output=result
            ) from exc
        # Checked here rather than only as an exit deliverable: a board claiming
        # to be the seed or the ground truth must not reach scoring wearing a
        # prediction's face, and the failure should land next to the run that
        # produced it.
        if parsed.source != "engine_predicted":
            raise EngineRunFailed(
                f"engine returned a board with source={parsed.source!r}, expected "
                f"'engine_predicted'",
                raw_output=result,
            )
        # `error_id` is a key, not a label. The seed-delta transform groups by
        # it, the exact-key matcher resolves through it, and occurrences point
        # at it — two issues sharing one would be silently conflated at every
        # one of those points, most visibly by appearing twice in the carrier
        # list. Checked at ingest, where the raw payload is still to hand.
        # (`benchmark/schemas` deliberately stays permissive: it models the
        # shape, and this is a property of one producer's output.)
        counts = Counter(i.error_id for i in parsed.issues)
        duplicates = sorted(error_id for error_id, n in counts.items() if n > 1)
        if duplicates:
            raise EngineRunFailed(
                f"engine returned duplicate error_id(s) {duplicates} on one board — "
                f"error_id is the key occurrences and the matcher resolve through, so "
                f"two issues sharing one cannot be told apart downstream",
                raw_output=result,
            )

        # Re-stamp: the Engine's board_id is its own label, not our dataset id.
        board = stamp_dataset_id(parsed)
        return EngineInvocation(
            board=board,
            raw_output=result,
            seconds=seconds,
            thread_id=thread_id,
            recorded_models=self.recorded_models(thread_id),
            trace_count=trace_count,
        )
