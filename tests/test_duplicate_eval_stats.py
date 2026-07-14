"""duplicate_eval_stats: per-delegation duplicate-design-point counts.

Regression for run example_study/20260713T221841: D003 spent 76% of its 160
evals (122 rows) re-evaluating 38 unique points (one point evaluated 10x) —
three of its own scripts independently resampled an identical seed=42 LHS
design without checking the ledger first. Backlog #24 / spec 07.
"""
from __future__ import annotations

from pathlib import Path

from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus

from a3dasm._src.instrumented import duplicate_eval_stats


def _build_store(store_dir: Path, rows) -> None:
    """rows: list of (x1, x2, y, delegation_id)."""
    domain = Domain()
    domain.add_float("x1", -5.0, 5.0)
    domain.add_float("x2", -5.0, 5.0)
    for k in ("y", "_delegation_id", "_source", "_ts"):
        domain.add_output(k, exist_ok=True)
    samples = {}
    for i, (x1, x2, y, did) in enumerate(rows):
        samples[i] = ExperimentSample(
            _input_data={"x1": x1, "x2": x2},
            _output_data={"y": y, "_delegation_id": did, "_source": "test",
                          "_ts": "2026-01-01T00:00:00+00:00"},
            job_status=JobStatus.FINISHED,
        )
    ExperimentData.from_data(data=samples, domain=domain).store(
        project_dir=store_dir)


def test_no_duplicates_reports_zero(tmp_path):
    _build_store(tmp_path / "experiment_data", [
        (0.1, 0.2, 1.0, "D001"),
        (0.3, 0.4, 2.0, "D001"),
    ])
    stats = duplicate_eval_stats(tmp_path / "experiment_data")
    assert stats["D001"]["total_rows"] == 2
    assert stats["D001"]["unique_points"] == 2
    assert stats["D001"]["duplicate_rows"] == 0


def test_counts_duplicate_rows_per_delegation(tmp_path):
    _build_store(tmp_path / "experiment_data", [
        (0.1, 0.2, 1.0, "D003"),
        (0.1, 0.2, 1.0, "D003"),  # exact repeat
        (0.1, 0.2, 1.0, "D003"),  # exact repeat again
        (0.3, 0.4, 2.0, "D003"),  # unique
    ])
    stats = duplicate_eval_stats(tmp_path / "experiment_data")
    assert stats["D003"]["total_rows"] == 4
    assert stats["D003"]["unique_points"] == 2
    assert stats["D003"]["duplicate_rows"] == 2  # 3 copies - 1 = 2 wasted
    worst_coords, worst_count = stats["D003"]["worst"]
    assert worst_count == 3
    assert worst_coords["x1"] == 0.1 and worst_coords["x2"] == 0.2


def test_separates_delegations(tmp_path):
    _build_store(tmp_path / "experiment_data", [
        (0.1, 0.2, 1.0, "D001"),
        (0.1, 0.2, 1.0, "D002"),  # same point, DIFFERENT delegation
    ])
    stats = duplicate_eval_stats(tmp_path / "experiment_data")
    assert stats["D001"]["duplicate_rows"] == 0
    assert stats["D002"]["duplicate_rows"] == 0


def test_namespace_aware(tmp_path):
    """Rows in a design-namespace sibling store must be included too — the
    exact class of blind spot fixed in RecallStore/QueryStore (backlog #21)."""
    _build_store(tmp_path / "experiment_data", [
        (0.1, 0.2, 1.0, "D003"),
    ])
    _build_store(tmp_path / "experiment_data" / "polar", [
        (0.1, 0.2, 1.0, "D003"),  # same coords, namespace store
    ])
    stats = duplicate_eval_stats(tmp_path / "experiment_data")
    assert stats["D003"]["total_rows"] == 2
    assert stats["D003"]["duplicate_rows"] == 1


def test_regression_fixture_matches_the_run_shape(tmp_path):
    """Replays this run's exact shape (160 rows / 38 unique under one
    delegation) — ties the test to the actual evidence, not just a synthetic
    case."""
    rows = []
    # Build exactly 38 unique points; distribute 160 total rows across them.
    n_unique = 38
    total = 160
    base = total // n_unique
    remainder = total - base * n_unique
    per_point_counts = [base + 1] * remainder + [base] * (n_unique - remainder)
    assert sum(per_point_counts) == total
    for i, count in enumerate(per_point_counts):
        x1, x2 = float(i), float(-i)
        for _ in range(count):
            rows.append((x1, x2, 0.0, "D003"))
    _build_store(tmp_path / "experiment_data", rows)
    stats = duplicate_eval_stats(tmp_path / "experiment_data")
    assert stats["D003"]["total_rows"] == 160
    assert stats["D003"]["unique_points"] == 38
    assert stats["D003"]["duplicate_rows"] == 122
