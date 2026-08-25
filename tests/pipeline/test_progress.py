"""Phase 8 — stage-level progress reporting.

Stdlib only, stderr only. A 2-3h run has to be followable from a terminal
without corrupting `report.json` / `manifest.json` (which never touch this
stream), so every assertion here reads the reporter's own sink, never stdout
or an artifact file.
"""

from __future__ import annotations

import io
import re
import time

from benchmark.pipeline.progress import Progress

TIMESTAMP = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] ")


def lines(stream: io.StringIO) -> list[str]:
    return [line for line in stream.getvalue().splitlines() if line]


def test_a_stage_prints_an_entry_and_an_exit_banner():
    stream = io.StringIO()
    progress = Progress(stream=stream)
    with progress.stage("generation"):
        pass
    out = lines(stream)
    assert len(out) == 2
    assert TIMESTAMP.match(out[0])
    assert "▶ generation" in out[0]
    assert TIMESTAMP.match(out[1])
    assert "✓ generation" in out[1]
    assert "elapsed" in out[1]


def test_the_exit_banner_carries_the_stage_supplied_detail():
    stream = io.StringIO()
    progress = Progress(stream=stream)
    with progress.stage("harness") as stage:
        stage.set_detail("8 traces collected")
    out = lines(stream)
    assert "8 traces collected" in out[1]


def test_a_stage_that_raises_still_reports_and_reraises():
    stream = io.StringIO()
    progress = Progress(stream=stream)
    try:
        with progress.stage("engine"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    out = lines(stream)
    assert "FAILED" in out[1]
    assert "engine" in out[1]


def test_quiet_suppresses_every_line():
    stream = io.StringIO()
    progress = Progress(stream=stream, quiet=True)
    with progress.stage("scoring") as stage:
        stage.set_detail("should not appear")
        progress.item("x", 1, 2)
        progress.note("a note")
    assert stream.getvalue() == ""


def test_item_prints_a_labelled_fraction():
    stream = io.StringIO()
    progress = Progress(stream=stream)
    progress.item("harness batch", 3, 8)
    out = lines(stream)
    assert "harness batch: 3/8" in out[0]


def test_poll_reports_a_changing_snapshot_and_a_final_value():
    stream = io.StringIO()
    progress = Progress(stream=stream)
    counter = {"n": 0}

    def snapshot() -> int:
        return counter["n"]

    with progress.poll("harness batch", snapshot, 5, interval=0.02):
        for i in range(1, 6):
            counter["n"] = i
            time.sleep(0.03)

    out = "\n".join(lines(stream))
    assert "harness batch: 5/5" in out, out


def test_poll_survives_a_snapshot_that_raises():
    stream = io.StringIO()
    progress = Progress(stream=stream)

    def snapshot() -> int:
        raise ValueError("not ready yet")

    with progress.poll("ablation", snapshot, 3, interval=0.02):
        time.sleep(0.05)
    # No crash, and no garbage line either — the failing snapshot never
    # produced a value to print.
    assert lines(stream) == []


def test_heartbeat_fires_at_the_given_interval():
    stream = io.StringIO()
    progress = Progress(stream=stream)
    with progress.heartbeat("engine run", interval=0.03):
        time.sleep(0.11)
    out = lines(stream)
    assert len(out) >= 2, out
    assert all("still running" in line for line in out)
    assert all("engine run" in line for line in out)


def test_heartbeat_prints_nothing_for_a_call_shorter_than_the_interval():
    stream = io.StringIO()
    progress = Progress(stream=stream)
    with progress.heartbeat("engine run", interval=5.0):
        pass
    assert lines(stream) == []


def test_note_is_a_standalone_timestamped_line():
    stream = io.StringIO()
    progress = Progress(stream=stream)
    progress.note("engine: 42 traces, ~3 batch(es) projected")
    out = lines(stream)
    assert len(out) == 1
    assert TIMESTAMP.match(out[0])
    assert "42 traces" in out[0]


def test_default_stream_is_stderr():
    progress = Progress()
    assert progress.stream is __import__("sys").stderr
