"""Starting and stopping the two black-box apps' LangGraph servers.

Each app ships its own `scripts/serve.sh start|stop`, which backgrounds
`langgraph dev` and blocks until the health endpoint answers. The pipeline runs
that script as a subprocess. It never imports from `apps/`, and it does not
reimplement the health-check dance — the app owns how the app starts.

**Ordering is a correctness constraint, not a convenience.** The harness batch
and the ablation stage must run inside ONE target-app server lifetime: Mode-A
replay forks a LangGraph thread created during the batch, and `langgraph dev`
holds thread/checkpoint state in a process-local store that does not survive a
restart. Stopping the target app between the two stages turns every replay into
a missing-thread error. The Engine's server, by contrast, is only needed for
its own single run and can be started after the target app is down — which is
also cheaper, since two `langgraph dev` processes on one machine compete for
the same CPU during the target-app batch.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from benchmark.pipeline.config import ServerSpec

log = logging.getLogger("benchmark.pipeline.servers")


class ServerStartFailed(RuntimeError):
    """An app's serve script did not bring its server up."""


def subprocess_runner(script: Path, action: str, cwd: Path) -> None:
    """Run `<app>/scripts/serve.sh <action>` from the app's own directory."""
    if not script.exists():
        raise ServerStartFailed(f"serve script not found: {script}")
    try:
        subprocess.run([str(script), action], check=True, cwd=cwd)
    except subprocess.CalledProcessError as exc:
        raise ServerStartFailed(f"{script} {action} exited {exc.returncode}") from exc


class ServerLifetime:
    """Owns the start/stop of the declared app servers."""

    def __init__(
        self,
        root: str | Path,
        specs: dict[str, ServerSpec] | None = None,
        *,
        runner=subprocess_runner,
        enabled: bool = True,
    ):
        self.root = Path(root)
        self.specs = dict(specs or {})
        self.runner = runner
        self.enabled = enabled

    def describe(self) -> dict[str, str]:
        """What this run manages, for the manifest."""
        return {name: spec.script for name, spec in self.specs.items()} if self.enabled else {}

    def _spec(self, name: str) -> ServerSpec | None:
        if not self.enabled:
            return None
        if not self.specs:
            # No servers declared: run against whatever the operator has up.
            return None
        if name not in self.specs:
            raise KeyError(f"no server named {name!r} in this pipeline config")
        return self.specs[name]

    def _run(self, name: str, action: str) -> None:
        spec = self._spec(name)
        if spec is None:
            return
        script = self.root / spec.script
        log.info("[server] %s %s", spec.label or name, action)
        self.runner(script, action, script.parent.parent)

    def start(self, name: str) -> None:
        self._run(name, "start")

    def stop(self, name: str) -> None:
        self._run(name, "stop")

    @contextmanager
    def running(self, name: str) -> Iterator[None]:
        """Bracket a stage with one server lifetime.

        A failed start propagates *without* a matching stop: there is nothing
        to stop, and running a stop against a server someone else owns is how a
        smoke run kills a colleague's process.
        """
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)
