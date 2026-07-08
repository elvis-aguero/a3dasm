"""Tests for AgenticOptimizerAdapter."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_agentic_optimizer_init(tmp_path):
    """AgenticOptimizerAdapter initialises with a study_dir."""
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")
    from a3dasm._src.optimizer import AgenticOptimizerAdapter

    opt = AgenticOptimizerAdapter(tmp_path)
    assert opt._run is not None


def test_agentic_optimizer_forward_raises_not_implemented(tmp_path):
    """AgenticOptimizerAdapter.forward() raises NotImplementedError."""
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")
    from a3dasm._src.optimizer import AgenticOptimizerAdapter

    opt = AgenticOptimizerAdapter(tmp_path)
    with pytest.raises(NotImplementedError, match="serialisation"):
        opt.forward(MagicMock())


def test_agentic_optimizer_passes_kwargs_to_run(tmp_path):
    """AgenticOptimizerAdapter forwards kwargs like model= to AgenticRun."""
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")
    from a3dasm._src.optimizer import AgenticOptimizerAdapter

    opt = AgenticOptimizerAdapter(tmp_path, model="claude-haiku-4-5-20251001")
    assert opt._run._model == "claude-haiku-4-5-20251001"
