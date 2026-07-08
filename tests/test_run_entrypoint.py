"""Tests for a3dasm._src.run — the CLI entrypoint."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# main() passes study_dir to AgenticRun
# ---------------------------------------------------------------------------


def test_main_calls_agentic_run_with_study_dir(tmp_path):
    """main() constructs AgenticRun(Path(study_dir)) with the positional arg."""
    from a3dasm._src import run as run_module

    mock_run_instance = MagicMock()
    mock_run_instance.execute.return_value = "done"
    mock_run_class = MagicMock(return_value=mock_run_instance)

    with patch.object(sys, "argv", ["run.py", str(tmp_path)]):
        with patch("a3dasm._src.run.AgenticRun", mock_run_class):
            run_module.main()

    mock_run_class.assert_called_once()
    call_args = mock_run_class.call_args
    # First positional arg should be a Path pointing to tmp_path
    assert call_args[0][0] == Path(str(tmp_path))


# ---------------------------------------------------------------------------
# main() exits 1 on AgenticRunError
# ---------------------------------------------------------------------------


def test_main_exits_1_on_agentic_run_error(tmp_path):
    """main() exits with code 1 when AgenticRunError is raised."""
    from a3dasm._src import run as run_module
    from a3dasm._src.agent_runtime import AgenticRunError

    mock_run_instance = MagicMock()
    mock_run_instance.execute.side_effect = AgenticRunError("fail")
    mock_run_class = MagicMock(return_value=mock_run_instance)

    with patch.object(sys, "argv", ["run.py", str(tmp_path)]):
        with patch("a3dasm._src.run.AgenticRun", mock_run_class):
            with patch("a3dasm._src.run.AgenticRunError", AgenticRunError):
                with pytest.raises(SystemExit) as exc_info:
                    run_module.main()

    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# main() exits 1 on generic Exception
# ---------------------------------------------------------------------------


def test_main_exits_1_on_generic_exception(tmp_path):
    """main() exits with code 1 on any unhandled exception."""
    from a3dasm._src import run as run_module

    mock_run_instance = MagicMock()
    mock_run_instance.execute.side_effect = RuntimeError("unexpected")
    mock_run_class = MagicMock(return_value=mock_run_instance)

    with patch.object(sys, "argv", ["run.py", str(tmp_path)]):
        with patch("a3dasm._src.run.AgenticRun", mock_run_class):
            with pytest.raises(SystemExit) as exc_info:
                run_module.main()

    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# main() uses default /study when no study_dir given
# ---------------------------------------------------------------------------


def test_main_uses_default_study_dir_when_not_provided():
    """main() passes Path('/study') when no study_dir arg is given."""
    from a3dasm._src import run as run_module

    mock_run_instance = MagicMock()
    mock_run_instance.execute.return_value = "done"
    mock_run_class = MagicMock(return_value=mock_run_instance)

    with patch.object(sys, "argv", ["run.py"]):
        with patch("a3dasm._src.run.AgenticRun", mock_run_class):
            run_module.main()

    call_args = mock_run_class.call_args
    assert call_args[0][0] == Path("/study")
