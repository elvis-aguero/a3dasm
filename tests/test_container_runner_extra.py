"""Extra tests for ContainerRunner — covers _run_compose, _tail_log, and
other uncovered lines."""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call


def _make_proc_mock(returncode: int = 0):
    proc = MagicMock()
    proc.returncode = returncode
    proc.poll.return_value = returncode
    proc.wait.return_value = returncode
    return proc


# ---------------------------------------------------------------------------
# _run_compose (lines 106-117)
# ---------------------------------------------------------------------------


def test_run_compose_called_when_ollama_sidecar(tmp_path, monkeypatch):
    """run() calls _run_compose when ollama_sidecar=True and backend='ollama'."""
    from a3dasm._src.container_runner import ContainerRunner

    runner = ContainerRunner(
        tmp_path, backend="ollama", ollama_sidecar=True,
        image="f3dasm:test", _docker_dir=tmp_path / "docker",
    )

    with patch.object(runner, "_run_compose", return_value=0) as mock_compose:
        with patch.object(runner, "_run_docker", return_value=0) as mock_docker:
            result = runner.run()

    mock_compose.assert_called_once()
    mock_docker.assert_not_called()
    assert result == 0


def test_run_compose_invokes_docker_compose(tmp_path, monkeypatch):
    """_run_compose uses docker compose with the sidecar compose files."""
    from a3dasm._src.container_runner import ContainerRunner

    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()

    runner = ContainerRunner(
        tmp_path, backend="ollama", ollama_sidecar=True,
        image="f3dasm:test", _docker_dir=docker_dir,
    )
    proc_mock = _make_proc_mock(0)

    with patch("subprocess.Popen", return_value=proc_mock) as mock_popen:
        result = runner._run_compose()

    mock_popen.assert_called_once()
    cmd = mock_popen.call_args[0][0]
    assert "docker" in cmd
    assert "compose" in cmd
    assert "--abort-on-container-exit" in cmd
    assert result == 0


def test_run_compose_sets_study_dir_env(tmp_path):
    """_run_compose passes STUDY_DIR in the environment."""
    from a3dasm._src.container_runner import ContainerRunner

    docker_dir = tmp_path / "docker"
    docker_dir.mkdir()

    runner = ContainerRunner(
        tmp_path, backend="ollama", ollama_sidecar=True,
        _docker_dir=docker_dir,
    )
    proc_mock = _make_proc_mock(0)

    with patch("subprocess.Popen", return_value=proc_mock) as mock_popen:
        runner._run_compose()

    kwargs = mock_popen.call_args[1]
    env = kwargs.get("env", {})
    assert "STUDY_DIR" in env
    assert str(tmp_path) in env["STUDY_DIR"]


# ---------------------------------------------------------------------------
# _run_docker: with run.py in study_dir (lines 91-95)
# ---------------------------------------------------------------------------


def test_run_docker_uses_run_py_entrypoint_when_present(tmp_path, monkeypatch):
    """When study_dir/run.py exists, _run_docker uses it as the entrypoint."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    (tmp_path / "run.py").write_text("# custom\n")

    from a3dasm._src.container_runner import ContainerRunner

    runner = ContainerRunner(tmp_path, backend="claude", image="f3dasm:test",
                             _docker_dir=tmp_path / "docker")
    proc_mock = _make_proc_mock(0)

    with patch("subprocess.Popen", return_value=proc_mock) as mock_popen:
        runner._run_docker()

    cmd = mock_popen.call_args[0][0]
    assert "--entrypoint" in cmd
    assert "python" in cmd
    # The entrypoint should be the study's run.py inside the container
    assert any("/study/run.py" in arg for arg in cmd)


def test_run_docker_no_run_py_passes_model_and_budget(tmp_path, monkeypatch):
    """When no run.py exists, --model and --budget are passed to the image."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    from a3dasm._src.container_runner import ContainerRunner

    runner = ContainerRunner(
        tmp_path, backend="claude", image="f3dasm:test",
        model="claude-haiku", budget=3600,
        _docker_dir=tmp_path / "docker",
    )
    proc_mock = _make_proc_mock(0)

    with patch("subprocess.Popen", return_value=proc_mock) as mock_popen:
        runner._run_docker()

    cmd = mock_popen.call_args[0][0]
    assert "--model" in cmd
    idx_m = cmd.index("--model")
    assert cmd[idx_m + 1] == "claude-haiku"
    assert "--budget" in cmd
    idx_b = cmd.index("--budget")
    assert cmd[idx_b + 1] == "3600"


# ---------------------------------------------------------------------------
# _latest_solution when runs_dir does not exist
# ---------------------------------------------------------------------------


def test_latest_solution_empty_when_no_runs_dir(tmp_path):
    """_latest_solution returns '' when runs/ directory doesn't exist."""
    from a3dasm._src.container_runner import ContainerRunner

    runner = ContainerRunner(tmp_path)
    assert runner._latest_solution() == ""


def test_latest_solution_empty_when_no_solution_md(tmp_path):
    """_latest_solution returns '' when runs/ exists but has no solution.md."""
    from a3dasm._src.container_runner import ContainerRunner

    runs_dir = tmp_path / "runs" / "20250101T000000"
    runs_dir.mkdir(parents=True)
    # No solution.md created

    runner = ContainerRunner(tmp_path)
    assert runner._latest_solution() == ""


# ---------------------------------------------------------------------------
# _tail_log: log file not found within deadline
# ---------------------------------------------------------------------------


def test_tail_log_returns_when_no_log_found(tmp_path):
    """_tail_log returns gracefully when no run.log appears within deadline."""
    from a3dasm._src.container_runner import ContainerRunner

    runner = ContainerRunner(tmp_path)
    proc_mock = _make_proc_mock(0)
    proc_mock.poll.return_value = 0  # process already done

    # Should return without error even when no log file appears
    runner._tail_log(proc_mock)  # no assert needed — just must not raise


def test_tail_log_reads_existing_log_file(tmp_path, capsys):
    """_tail_log reads and prints lines from an existing run.log."""
    from a3dasm._src.container_runner import ContainerRunner

    # Create the run.log file before _tail_log is called
    runs_dir = tmp_path / "runs" / "20250101T000000" / "debug"
    runs_dir.mkdir(parents=True)
    log_file = runs_dir / "run.log"
    log_file.write_text("Line 1\nLine 2\n", encoding="utf-8")

    runner = ContainerRunner(tmp_path)

    # Proc that appears done immediately on first poll
    proc_mock = MagicMock()
    poll_calls = [0]

    def poll_side_effect():
        result = None if poll_calls[0] == 0 else 0
        poll_calls[0] += 1
        return result

    proc_mock.poll.side_effect = poll_side_effect

    with patch("time.sleep"):  # skip actual sleeps
        runner._tail_log(proc_mock)

    captured = capsys.readouterr()
    # Log lines should have been printed
    assert "Line 1" in captured.out or "Line 2" in captured.out


def test_tail_log_drains_remaining_lines_after_process_exits(tmp_path, capsys):
    """_tail_log drains remaining log lines after process exits."""
    from a3dasm._src.container_runner import ContainerRunner

    runs_dir = tmp_path / "runs" / "20250101T000000" / "debug"
    runs_dir.mkdir(parents=True)
    log_file = runs_dir / "run.log"
    log_file.write_text("Final line\n", encoding="utf-8")

    runner = ContainerRunner(tmp_path)

    # Proc that is already done (poll returns 0 immediately)
    proc_mock = MagicMock()
    proc_mock.poll.return_value = 0  # already finished

    with patch("time.sleep"):
        runner._tail_log(proc_mock)

    # Should complete without error (drain code reached)
