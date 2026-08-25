"""Stage-level progress reporting for a long pipeline run.

A full-scale run (`configs/pipeline/full.yaml`, `configs/pipeline/submission.yaml`)
is 1.5-2.5 hours of wall clock, most of it inside a single blocking call (a
harness batch, an Engine `runs.wait`). Without something printing along the
way, that looks identical to a hang. This module is the "something": stdlib
only (no tqdm/rich — a benchmarking harness has no business vendoring a
progress-bar dependency), timestamped, `flush=True` lines written to stderr so
they interleave sanely with whatever else is running and are visible even when
stdout is piped or redirected.

**Never touches an artifact.** `report.json`, `manifest.json`, and friends are
built and written by `benchmark.pipeline.runner` from objects in memory; this
module only ever writes to a stream (stderr by default), so there is no path
by which a progress line could reach a file that downstream tooling parses.

**Quiet is the same code path, not a different one.** `Progress(quiet=True)`
runs every method, computes every string — it just never prints it. That is
what lets `benchmark.pipeline.runner` call `progress.stage(...)` unconditionally
rather than guarding every call site with `if progress is not None`.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import TextIO

ENTER = "▶"  # ▶
DONE = "✓"  # ✓
FAILED = "✗"  # ✗
RESUMED = "↻"  # ↻ — loaded from disk, never executed (see pipeline/resume.py)


def _clock() -> str:
    return datetime.now().strftime("%H:%M:%S")


class StageHandle:
    """Yielded by `Progress.stage()`. The body reports its own completion detail.

    Kept separate from `Progress` itself so a nested call (`progress.item(...)`
    inside a stage) never has to decide which stage it belongs to — the handle
    only ever carries the one string the exit banner prints alongside the
    elapsed time.
    """

    def __init__(self) -> None:
        self.detail: str = ""

    def set_detail(self, detail: str) -> None:
        self.detail = detail


class Progress:
    """Prints timestamped stage banners and progress lines to a stream.

    Every public method funnels through `_emit`, so a single `quiet` flag
    mutes the whole reporter without any call site needing to know that.
    """

    def __init__(self, *, stream: TextIO | None = None, quiet: bool = False) -> None:
        self.stream: TextIO = stream if stream is not None else sys.stderr
        self.quiet = quiet
        self._lock = threading.Lock()

    def _emit(self, line: str) -> None:
        if self.quiet:
            return
        with self._lock:
            print(f"[{_clock()}] {line}", file=self.stream, flush=True)

    def note(self, message: str) -> None:
        """A standalone, timestamped line outside any stage banner."""
        self._emit(message)

    def resumed(self, name: str, detail: str = "") -> None:
        """`↻ name (resumed from disk)` — a stage that was skipped, not run.

        Deliberately NOT a `stage()` banner with a zero elapsed time: a reader
        skimming a resumed run's output has to be able to tell "this took no
        time" from "this did not happen", and a `✓ harness (elapsed 0s)` says
        the first when the truth is the second.
        """
        suffix = f" — {detail}" if detail else ""
        self._emit(f"{RESUMED} {name} (resumed from disk){suffix}")

    @contextmanager
    def stage(self, name: str) -> Iterator[StageHandle]:
        """`▶ name` on entry, `✓ name (elapsed Xs[, detail])` on a clean exit.

        A stage that raises still reports — `✗ name (elapsed Xs) — FAILED` —
        and the exception propagates unchanged; a 2-hour run that dies inside
        the Engine pass is exactly the run whose "where did it get to" matters
        most.
        """
        started = time.time()
        self._emit(f"{ENTER} {name}")
        handle = StageHandle()
        try:
            yield handle
        except BaseException:
            elapsed = time.time() - started
            self._emit(f"{FAILED} {name} (elapsed {elapsed:.0f}s) — FAILED")
            raise
        else:
            elapsed = time.time() - started
            suffix = f", {handle.detail}" if handle.detail else ""
            self._emit(f"{DONE} {name} (elapsed {elapsed:.0f}s{suffix})")

    def item(self, label: str, current: int, total: int) -> None:
        """A one-off `label: current/total` line."""
        self._emit(f"    {label}: {current}/{total}")

    @contextmanager
    def poll(
        self,
        label: str,
        snapshot: Callable[[], int],
        total: int,
        *,
        interval: float = 1.0,
    ) -> Iterator[None]:
        """Print `label: n/total` on a background thread while the body runs.

        For stages the runner has no per-item completion signal from — a
        harness batch runs its inputs inside one `ThreadPoolExecutor.map` call
        that does not return until every input is done. `snapshot` is polled
        instead; it must be safe to call concurrently with whatever the body
        is doing (`Harness.stats` is, written under the harness's own lock).
        A snapshot that raises is skipped rather than propagated — a stalled
        counter must not take the run down with it.
        """
        stop = threading.Event()
        last = [-1]

        def loop() -> None:
            while not stop.wait(interval):
                try:
                    done = snapshot()
                except Exception:  # noqa: BLE001 - progress must not crash the run
                    continue
                if done != last[0]:
                    last[0] = done
                    self.item(label, done, total)

        thread = threading.Thread(target=loop, name=f"progress-poll-{label}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=interval + 1)
            try:
                done = snapshot()
            except Exception:  # noqa: BLE001
                done = None
            if done is not None and done != last[0]:
                self.item(label, done, total)

    @contextmanager
    def heartbeat(self, label: str, *, interval: float = 30.0) -> Iterator[None]:
        """Print an elapsed-time line on a background thread while the body runs.

        For a single blocking call the caller has no visibility inside of —
        the Engine's `runs.wait` holds one HTTP request open for the whole
        analysis pass. A call shorter than `interval` prints nothing, which is
        the point: this is a "still alive" signal, not a stopwatch.
        """
        stop = threading.Event()
        started = time.time()

        def loop() -> None:
            while not stop.wait(interval):
                elapsed = time.time() - started
                self._emit(f"    ... {label}: still running (elapsed {elapsed:.0f}s)")

        thread = threading.Thread(target=loop, name=f"progress-heartbeat-{label}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=interval + 1)
