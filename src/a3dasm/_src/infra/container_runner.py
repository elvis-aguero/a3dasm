"""ContainerRunner — run AgenticRun inside a Docker/Colima container."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any

__all__ = ["ContainerRunner"]

# Mount point inside the container for the Claude CLI config/credentials dir.
_CONTAINER_CLAUDE_CONFIG = "/claude-config"

# Resolved at container start time; hardcoded mount point inside the container.
_CONTAINER_STUDY_DIR = "/study"


class ContainerRunner:
    """Run AgenticRun inside a Docker/Colima container.

    Claude backend (default) — subscription auth is PREFERRED, API key is a
    warned fallback (see ``_claude_auth_args``):
        docker run --rm
            -v <study_dir>:/study
            (subscription) -e CLAUDE_CODE_OAUTH_TOKEN   (from `claude setup-token`)
                  or  -v ~/.claude:/claude-config -e CLAUDE_CONFIG_DIR=/claude-config
            (fallback) -e ANTHROPIC_API_KEY             (warns; never set when
                  a subscription credential is present — the API key would
                  otherwise take CLI auth precedence and clobber it)
            <image> /study [--model X] [--budget N]

    Ollama backend — host Ollama (default):
        same + -e OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
             + --add-host host.docker.internal:host-gateway  (Linux)

    Ollama backend — sidecar (ollama_sidecar=True):
        docker compose -f <compose> -f <compose_ollama> up --abort-on-container-exit
    """

    def __init__(
        self,
        study_dir: Path,
        *,
        model: str | None = None,
        budget: float | None = None,
        backend: str = "claude",
        image: str = "f3dasm-agentic:latest",
        ollama_sidecar: bool = False,
        _docker_dir: Path | None = None,
    ) -> None:
        self.study_dir = Path(study_dir).resolve()
        self.model = model
        self.budget = budget
        self.backend = backend
        self.image = image
        self.ollama_sidecar = ollama_sidecar
        # Where docker/ lives — defaults to repo root / docker/
        self._docker_dir = _docker_dir or (
            Path(__file__).parent.parent.parent.parent.parent / "docker"
        )

    def build_image(self, dockerfile: str | None = None) -> None:
        """Build the container image via docker build."""
        df = dockerfile or str(self._docker_dir / "Dockerfile")
        context = str(self._docker_dir.parent)
        cmd = ["docker", "build", "-f", df, "-t", self.image, context]
        subprocess.run(cmd, check=True)

    def run(self) -> int:
        """Start container, stream run.log to stdout, return exit code."""
        if self.ollama_sidecar and self.backend == "ollama":
            return self._run_compose()
        return self._run_docker()

    def _run_docker(self) -> int:
        cmd = ["docker", "run", "--rm",
               "-v", f"{self.study_dir}:{_CONTAINER_STUDY_DIR}"]

        # Environment
        if self.backend == "claude":
            cmd += self._claude_auth_args()
        elif self.backend == "ollama":
            ollama_url = (
                os.environ.get("OLLAMA_BASE_URL")
                or "http://host.docker.internal:11434/v1"
            )
            cmd += ["-e", f"OLLAMA_BASE_URL={ollama_url}"]
            # On Linux, host.docker.internal doesn't resolve by default.
            if sys.platform.startswith("linux"):
                cmd += ["--add-host", "host.docker.internal:host-gateway"]

        # If the study ships its own run.py, execute it directly — it defines
        # the graph topology.  Otherwise fall back to the default entrypoint.
        run_py = self.study_dir / "run.py"
        if run_py.exists():
            cmd += ["-e", "PYTHONUNBUFFERED=1",
                    "--entrypoint", "python",
                    self.image,
                    f"{_CONTAINER_STUDY_DIR}/run.py"]
        else:
            cmd.append(self.image)
            cmd.append(_CONTAINER_STUDY_DIR)
            if self.model:
                cmd += ["--model", self.model]
            if self.budget is not None:
                cmd += ["--budget", str(self.budget)]

        return self._popen_and_stream(cmd)

    def _claude_auth_args(self) -> list[str]:
        """Docker args that authenticate the in-container Claude CLI.

        Precedence (subscription PREFERRED; API key is a warned fallback):
          1. CLAUDE_CODE_OAUTH_TOKEN — long-lived subscription token from
             ``claude setup-token``; forwarded as an env var (cluster-friendly,
             no host mount needed).
          2. host Claude config dir — if ``~/.claude/.credentials.json`` (or
             ``$CLAUDE_CONFIG_DIR``) exists, mount it so the in-container CLI
             reuses the host's logged-in session and refreshes it in place.
          3. ANTHROPIC_API_KEY — fallback only; emits a warning.
          4. nothing — hard fail with an actionable message.

        When a subscription credential is used the API key is deliberately NOT
        forwarded: the CLI ranks ANTHROPIC_API_KEY above the subscription
        token, so setting it would silently override the subscription.
        ``-e VAR`` (no value) passes the host value through without exposing
        the secret in the process arg list.
        """
        if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            return ["-e", "CLAUDE_CODE_OAUTH_TOKEN"]

        cfg_dir = os.environ.get("CLAUDE_CONFIG_DIR") or str(
            Path.home() / ".claude")
        if (Path(cfg_dir) / ".credentials.json").is_file():
            return [
                "-v", f"{cfg_dir}:{_CONTAINER_CLAUDE_CONFIG}",
                "-e", f"CLAUDE_CONFIG_DIR={_CONTAINER_CLAUDE_CONFIG}",
            ]

        if os.environ.get("ANTHROPIC_API_KEY"):
            warnings.warn(
                "No subscription credential found (CLAUDE_CODE_OAUTH_TOKEN "
                f"unset and no {cfg_dir}/.credentials.json); falling back to "
                "ANTHROPIC_API_KEY. Subscription auth is preferred — generate "
                "a token with `claude setup-token`.",
                stacklevel=2,
            )
            return ["-e", "ANTHROPIC_API_KEY"]

        raise RuntimeError(
            "No Claude credentials available for the container. Provide ONE "
            "of: CLAUDE_CODE_OAUTH_TOKEN (run `claude setup-token`), a host "
            "login at ~/.claude/.credentials.json (or $CLAUDE_CONFIG_DIR), or "
            "ANTHROPIC_API_KEY as a fallback."
        )

    def _run_compose(self) -> int:
        compose = str(self._docker_dir / "docker-compose.yml")
        compose_ollama = str(self._docker_dir / "docker-compose.ollama.yml")
        cmd = [
            "docker", "compose",
            "-f", compose, "-f", compose_ollama,
            "up", "--abort-on-container-exit", "--exit-code-from", "agentic",
        ]
        env = {**os.environ, "STUDY_DIR": str(self.study_dir)}
        proc = subprocess.Popen(cmd, env=env)
        proc.wait()
        return proc.returncode

    def _popen_and_stream(self, cmd: list[str]) -> int:
        """Run cmd, tail run.log from the study dir, return exit code."""
        proc = subprocess.Popen(cmd)
        log_thread = threading.Thread(
            target=self._tail_log, args=(proc,), daemon=True
        )
        log_thread.start()
        proc.wait()
        log_thread.join(timeout=2)
        return proc.returncode

    def _tail_log(self, proc: Any) -> None:
        """Poll for run.log and stream new lines until the container exits."""
        deadline = time.time() + 10  # wait up to 10s for log file to appear
        log_path = None
        while proc.poll() is None and time.time() < deadline:
            candidates = sorted(
                (self.study_dir / "runs").glob("*/debug/run.log")
                if (self.study_dir / "runs").exists() else [],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                log_path = candidates[0]
                break
            time.sleep(0.5)

        if log_path is None:
            return

        with log_path.open(encoding="utf-8", errors="replace") as f:
            while proc.poll() is None:
                line = f.readline()
                if line:
                    print(line, end="", flush=True)
                else:
                    time.sleep(0.1)
            # Drain any remaining lines after process exits
            for line in f:
                print(line, end="", flush=True)

    def _latest_solution(self) -> str:
        """Return the deliverable notebook's markdown (the writeup) from the
        study dir, or empty string. The notebook IS the writeup — there is no
        solution.md."""
        nb_path = self.study_dir / "pipeline.ipynb"
        if not nb_path.exists():
            return ""
        try:
            import nbformat
            nb = nbformat.read(str(nb_path), as_version=4)
        except Exception:  # noqa: BLE001
            return ""
        return "\n\n".join(
            c.source for c in nb.cells if c.get("cell_type") == "markdown")
