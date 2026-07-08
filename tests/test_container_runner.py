"""Tests for ContainerRunner — no live Docker required (subprocess mocked)."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proc_mock(returncode: int = 0):
    """Return a MagicMock that looks like a finished subprocess.Popen."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.poll.return_value = returncode   # looks like process already done
    proc.wait.return_value = returncode
    return proc


# ---------------------------------------------------------------------------
# 1. build_image
# ---------------------------------------------------------------------------

def test_build_image_calls_docker_build(tmp_path):
    """build_image() must invoke 'docker build' with -t and the image name."""
    from a3dasm._src.container_runner import ContainerRunner

    runner = ContainerRunner(tmp_path, image="myimg:test",
                             _docker_dir=tmp_path / "docker")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        runner.build_image()

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "docker" in cmd
    assert "build" in cmd
    assert "-t" in cmd
    assert "myimg:test" in cmd


# ---------------------------------------------------------------------------
# 2. run — claude backend
# ---------------------------------------------------------------------------

def test_run_claude_assembles_correct_docker_run_args(tmp_path, monkeypatch):
    """run() with backend='claude' mounts study_dir and forwards a credential.

    With only ANTHROPIC_API_KEY set (no subscription token/creds), the runner
    falls back to the key — passed as `-e ANTHROPIC_API_KEY` *passthrough* (the
    value is read from the env by Docker, NOT embedded in argv) — and warns that
    subscription auth is preferred. (See test_container_runner_auth.py for the
    full precedence.)"""
    # Deterministic: no subscription credential available → fallback path.
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no_creds_here"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

    from a3dasm._src.container_runner import ContainerRunner

    runner = ContainerRunner(tmp_path, backend="claude", image="f3dasm:test",
                             _docker_dir=tmp_path / "docker")
    proc_mock = _make_proc_mock(0)

    with patch("subprocess.Popen", return_value=proc_mock) as mock_popen:
        with pytest.warns(UserWarning, match="Subscription auth is preferred"):
            exit_code = runner.run()

    assert exit_code == 0
    mock_popen.assert_called_once()
    cmd = mock_popen.call_args[0][0]

    assert "docker" in cmd
    assert "run" in cmd
    # Volume mount
    assert "-v" in cmd
    mount_arg = cmd[cmd.index("-v") + 1]
    assert str(tmp_path) in mount_arg
    assert ":/study" in mount_arg
    # API key forwarded as passthrough: `-e ANTHROPIC_API_KEY` (no =value in argv)
    env_args = [cmd[i + 1] for i, c in enumerate(cmd) if c == "-e"]
    assert "ANTHROPIC_API_KEY" in env_args
    assert not any("=sk-test-key" in a for a in env_args), "secret leaked into argv"
    # Image and container study dir
    assert "f3dasm:test" in cmd
    assert "/study" in cmd


# ---------------------------------------------------------------------------
# 3. run — ollama backend, default URL
# ---------------------------------------------------------------------------

def test_run_ollama_host_sets_ollama_base_url(tmp_path, monkeypatch):
    """run() with backend='ollama' must set OLLAMA_BASE_URL to the default host URL."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    from a3dasm._src.container_runner import ContainerRunner

    runner = ContainerRunner(tmp_path, backend="ollama", ollama_sidecar=False,
                             image="f3dasm:test",
                             _docker_dir=tmp_path / "docker")
    proc_mock = _make_proc_mock(0)

    with patch("subprocess.Popen", return_value=proc_mock):
        exit_code = runner.run()

    assert exit_code == 0
    cmd = proc_mock.call_args  # not used directly — check via Popen mock
    # Re-capture
    with patch("subprocess.Popen", return_value=proc_mock) as mock_popen:
        runner.run()
    cmd = mock_popen.call_args[0][0]

    env_args = [cmd[i + 1] for i, c in enumerate(cmd) if c == "-e"]
    assert any("OLLAMA_BASE_URL=http://host.docker.internal:11434/v1" == a
               for a in env_args)


# ---------------------------------------------------------------------------
# 4. run — ollama backend, env var override
# ---------------------------------------------------------------------------

def test_run_ollama_respects_env_var_override(tmp_path, monkeypatch):
    """run() with backend='ollama' must honour a custom OLLAMA_BASE_URL."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://myhost:9999/v1")

    from a3dasm._src.container_runner import ContainerRunner

    runner = ContainerRunner(tmp_path, backend="ollama", ollama_sidecar=False,
                             image="f3dasm:test",
                             _docker_dir=tmp_path / "docker")
    proc_mock = _make_proc_mock(0)

    with patch("subprocess.Popen", return_value=proc_mock) as mock_popen:
        runner.run()

    cmd = mock_popen.call_args[0][0]
    env_args = [cmd[i + 1] for i, c in enumerate(cmd) if c == "-e"]
    assert any("OLLAMA_BASE_URL=http://myhost:9999/v1" == a for a in env_args)
    # Must NOT use the default
    assert not any("11434" in a for a in env_args)


# ---------------------------------------------------------------------------
# 5. Linux host-gateway flag
# ---------------------------------------------------------------------------

def test_run_adds_host_gateway_on_linux(tmp_path, monkeypatch):
    """On Linux, run() must add --add-host host.docker.internal:host-gateway."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    from a3dasm._src.container_runner import ContainerRunner

    runner = ContainerRunner(tmp_path, backend="ollama", ollama_sidecar=False,
                             image="f3dasm:test",
                             _docker_dir=tmp_path / "docker")
    proc_mock = _make_proc_mock(0)

    with patch("subprocess.Popen", return_value=proc_mock) as mock_popen:
        runner.run()

    cmd = mock_popen.call_args[0][0]
    assert "--add-host" in cmd
    idx = cmd.index("--add-host")
    assert cmd[idx + 1] == "host.docker.internal:host-gateway"


# ---------------------------------------------------------------------------
# 6. _latest_solution
# ---------------------------------------------------------------------------

def test_latest_solution_reads_notebook_markdown(tmp_path):
    """_latest_solution() returns the markdown (the writeup) of the deliverable
    notebook — there is no solution.md."""
    import nbformat

    from a3dasm._src.container_runner import ContainerRunner
    from a3dasm._src.notebook_exec import build_notebook

    nb = build_notebook([
        {"type": "markdown", "source": "result"},
        {"type": "code", "name": "analysis", "source": "print(1)"},
    ])
    nbformat.write(nb, str(tmp_path / "pipeline.ipynb"))

    runner = ContainerRunner(tmp_path)
    assert runner._latest_solution() == "result"


# ---------------------------------------------------------------------------
# 7. AgenticRun container=True delegates to ContainerRunner
# ---------------------------------------------------------------------------

def test_agentic_run_container_flag_delegates_to_container_runner(tmp_path, monkeypatch):
    """AgenticRun(container=True).execute() must call ContainerRunner.run, not the in-process graph."""
    # Need a PROBLEM_STATEMENT.md so AgenticRun can find the study
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("Solve X.", encoding="utf-8")

    # Patch ContainerRunner inside agent_runtime (where it's imported)
    mock_runner = MagicMock()
    mock_runner.run.return_value = 0
    mock_runner._latest_solution.return_value = "ok"

    with patch(
        "a3dasm._src.agent_runtime.ContainerRunner",
        return_value=mock_runner,
    ):
        from a3dasm._src.agent_runtime import AgenticRun

        result = AgenticRun(tmp_path, container=True).execute()

    assert result == "ok"
    mock_runner.run.assert_called_once()
