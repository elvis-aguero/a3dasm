"""Tests for the viewer CLI entry point and AgenticRun.serve_viewer()."""
from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

from a3dasm._src.agent_runtime import AgenticRun


def _make_study(tmp_path: Path) -> Path:
    study = tmp_path / "study"
    study.mkdir()
    (study / "PROBLEM_STATEMENT.md").write_text("# trivial\n")
    return study


def test_cli_help_exits_zero(capsys):
    from a3dasm._src.viewer.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "study-dir" in out


def test_cli_help_mentions_port_option(capsys):
    from a3dasm._src.viewer.__main__ import main

    with pytest.raises(SystemExit):
        main(["--help"])
    assert "--port" in capsys.readouterr().out


def test_cli_missing_viewer_extra_gives_clear_message(monkeypatch, tmp_path, capsys):
    """If Starlette (the viewer extra) isn't importable, the CLI must say
    so plainly — not surface a raw traceback."""
    from a3dasm._src.viewer import __main__ as viewer_main

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "starlette" or name.startswith("starlette."):
            raise ImportError("No module named 'starlette'")
        return real_import(name, *args, **kwargs)

    # Force a fresh import of .app on next `from .app import run_viewer`.
    monkeypatch.delitem(sys.modules, "a3dasm._src.viewer.app", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    rc = viewer_main.main([str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "viewer" in err.lower()
    assert "pip install a3dasm[viewer]" in err


def test_agentic_run_construction_does_not_require_starlette(monkeypatch, tmp_path):
    """Constructing AgenticRun must not need Starlette at all — only
    serve_viewer() itself does (a lazy import, exactly like
    render_architecture()'s own run_diagram import)."""
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "starlette" or name.startswith("starlette."):
            raise ImportError("No module named 'starlette'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    study = _make_study(tmp_path)
    run = AgenticRun(study_dir=study, interactive=False)  # must not raise
    assert hasattr(run, "serve_viewer")


def test_serve_viewer_raises_clearly_when_starlette_missing(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "starlette" or name.startswith("starlette."):
            raise ImportError("No module named 'starlette'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "a3dasm._src.viewer.app", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    study = _make_study(tmp_path)
    run = AgenticRun(study_dir=study, interactive=False)
    with pytest.raises(ImportError):
        run.serve_viewer()
