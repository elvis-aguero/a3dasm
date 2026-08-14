"""Tests for a3dasm._src.constraint_snapshot — the single source of truth
for a run's budget/time-remaining state, shared by every delegation
boundary (human->strategizer, strategizer->worker, strategizer->critic in
both GATE and FEEDBACK mode)."""
from __future__ import annotations

from a3dasm._src.constraint_snapshot import (
    ConstraintSnapshot,
    compute_constraint_snapshot,
)


def test_no_budgets_set_renders_as_unspecified():
    s = compute_constraint_snapshot(
        eval_budget=None, budget_seconds=None, run_start=None,
        experiment_data_dir=None,
    )
    assert s.evals_remaining is None
    assert s.wall_remaining_s is None
    assert not s.eval_exhausted
    assert not s.wall_exhausted
    text = s.as_text()
    assert "unspecified" in text
    assert "<constraints>" in text and "</constraints>" in text


def test_eval_budget_remaining_and_exhausted():
    s = ConstraintSnapshot(
        eval_budget=100, evals_used=80,
        wall_budget_s=None, wall_elapsed_s=None,
    )
    assert s.evals_remaining == 20
    assert not s.eval_exhausted
    exhausted = ConstraintSnapshot(
        eval_budget=100, evals_used=140,
        wall_budget_s=None, wall_elapsed_s=None,
    )
    assert exhausted.evals_remaining == 0, "remaining floors at 0, never negative"
    assert exhausted.eval_exhausted


def test_wall_budget_remaining_and_exhausted():
    s = ConstraintSnapshot(
        eval_budget=None, evals_used=0,
        wall_budget_s=3600.0, wall_elapsed_s=900.0,
    )
    assert s.wall_remaining_s == 2700.0
    assert not s.wall_exhausted
    exhausted = ConstraintSnapshot(
        eval_budget=None, evals_used=0,
        wall_budget_s=3600.0, wall_elapsed_s=5000.0,
    )
    assert exhausted.wall_remaining_s == 0.0
    assert exhausted.wall_exhausted


def test_as_text_reports_both_budgets_together():
    s = ConstraintSnapshot(
        eval_budget=200, evals_used=150,
        wall_budget_s=3600.0, wall_elapsed_s=1800.0,
    )
    text = s.as_text()
    assert "150/200 evals used" in text
    assert "30.0min/60.0min used (50%)" in text
    assert "EXHAUSTED" not in text


def test_as_text_flags_exhaustion_explicitly():
    s = ConstraintSnapshot(
        eval_budget=200, evals_used=200,
        wall_budget_s=None, wall_elapsed_s=None,
    )
    assert "(EXHAUSTED)" in s.as_text()


def test_as_dict_matches_as_text_numbers():
    """Single source of truth: the structured fields persisted to
    DelegationLog and the text injected into an agent's context must come
    from the same computation, never drift independently."""
    s = ConstraintSnapshot(
        eval_budget=50, evals_used=10,
        wall_budget_s=100.0, wall_elapsed_s=25.0,
    )
    d = s.as_dict()
    assert d["eval_budget"] == 50
    assert d["evals_used"] == 10
    assert d["evals_remaining"] == 40
    assert d["wall_budget_s"] == 100.0
    assert d["wall_elapsed_s"] == 25.0
    assert d["wall_remaining_s"] == 75.0


def test_compute_constraint_snapshot_falls_back_gracefully_on_bad_path(tmp_path):
    """A nonexistent/empty experiment_data_dir must never raise — evals_used
    falls back to 0, matching a fresh run before any evaluations exist."""
    s = compute_constraint_snapshot(
        eval_budget=10, budget_seconds=None, run_start=None,
        experiment_data_dir=tmp_path / "does_not_exist",
    )
    assert s.evals_used == 0
