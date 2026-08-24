"""Server-lifetime choreography.

The pipeline owns both LangGraph servers, through each app's own
`scripts/serve.sh` run as a subprocess — never an import. The ordering
constraint is the interesting part: the harness batch and the ablation stage
must share ONE target-app server lifetime, because Mode-A replay forks a
LangGraph thread created during the batch and that thread dies with the server.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.pipeline.config import ServerSpec
from benchmark.pipeline.servers import ServerLifetime, ServerStartFailed

SPECS = {
    "target_app": ServerSpec(script="apps/target_app/scripts/serve.sh", label="target app"),
    "engine": ServerSpec(script="apps/engine/scripts/serve.sh", label="engine"),
}


class FakeRunner:
    def __init__(self, fail_on: tuple[str, ...] = ()):
        self.calls: list[tuple[str, str, Path]] = []
        self.fail_on = fail_on

    def __call__(self, script: Path, action: str, cwd: Path) -> None:
        self.calls.append((script.name, action, cwd))
        if action in self.fail_on:
            raise ServerStartFailed(f"{script} {action} failed")

    @property
    def actions(self) -> list[str]:
        return [f"{c[2].name}:{c[1]}" for c in self.calls]


def lifetime(runner, root=Path("/repo"), specs=None):
    return ServerLifetime(root, specs if specs is not None else SPECS, runner=runner)


def test_starting_runs_the_apps_own_script(tmp_path):
    runner = FakeRunner()
    lifetime(runner, root=tmp_path).start("target_app")
    script, action, cwd = runner.calls[0]
    assert (script, action) == ("serve.sh", "start")
    assert cwd == tmp_path / "apps" / "target_app"


def test_the_context_manager_stops_what_it_started():
    runner = FakeRunner()
    with lifetime(runner).running("target_app"):
        pass
    assert runner.actions == ["target_app:start", "target_app:stop"]


def test_a_raising_body_still_stops_the_server():
    runner = FakeRunner()
    with pytest.raises(RuntimeError, match="boom"), lifetime(runner).running("target_app"):
        raise RuntimeError("boom")
    assert runner.actions == ["target_app:start", "target_app:stop"]


def test_the_two_servers_take_turns():
    """Target app up for harness+ablation, then down; engine up, then down."""
    runner = FakeRunner()
    manager = lifetime(runner)
    with manager.running("target_app"):
        pass
    with manager.running("engine"):
        pass
    assert runner.actions == [
        "target_app:start",
        "target_app:stop",
        "engine:start",
        "engine:stop",
    ]


def test_an_undeclared_server_is_refused():
    with pytest.raises(KeyError, match="ollama"):
        lifetime(FakeRunner()).start("ollama")


def test_a_server_that_will_not_start_fails_loudly():
    with pytest.raises(ServerStartFailed):
        with lifetime(FakeRunner(fail_on=("start",))).running("engine"):
            pytest.fail("the body must not run when the server never came up")


def test_a_failed_start_does_not_leave_a_stop_pending():
    runner = FakeRunner(fail_on=("start",))
    with pytest.raises(ServerStartFailed), lifetime(runner).running("engine"):
        pass
    assert runner.actions == ["engine:start"]


def test_an_unmanaged_lifetime_is_a_no_op():
    """`--no-serve`: the operator already has both servers up."""
    runner = FakeRunner()
    manager = ServerLifetime(Path("/repo"), SPECS, runner=runner, enabled=False)
    with manager.running("target_app"):
        pass
    assert runner.calls == []


def test_a_server_with_no_declared_spec_is_a_no_op():
    """A config that declares no servers runs against whatever is already up."""
    runner = FakeRunner()
    manager = ServerLifetime(Path("/repo"), {}, runner=runner)
    with manager.running("target_app"):
        pass
    assert runner.calls == []


def test_managed_servers_are_reported_for_the_manifest():
    manager = lifetime(FakeRunner())
    assert manager.describe() == {
        "target_app": "apps/target_app/scripts/serve.sh",
        "engine": "apps/engine/scripts/serve.sh",
    }
