"""ROOT 1 fix: the canonical agentic ledger must be un-clobberable.

A protected store (marked with PROTECTED_STORE_SENTINEL by the agentic runtime)
must refuse any ExperimentData.store() that would SHRINK it — so a stray agent
`.store(canonical_dir)` on a partial table cannot destroy the metered rows that
get_evaluator() accumulated (the v9 360→100 loss). Unprotected stores keep their
normal overwrite behavior (no blast radius for non-agentic users).
"""
from __future__ import annotations

import pytest

from a3dasm._src._f3dasm_compat import PROTECTED_STORE_SENTINEL
from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus


def _ed(n: int) -> ExperimentData:
    domain = Domain()
    domain.add_float("a", 0.0, 100.0)
    domain.add_output("z", exist_ok=True)
    rows = {
        i: ExperimentSample(
            _input_data={"a": float(i)},
            _output_data={"z": float(-i)},
            job_status=JobStatus.FINISHED,
        )
        for i in range(n)
    }
    return ExperimentData.from_data(data=rows, domain=domain)


def test_protected_store_refuses_shrinking_write(tmp_path):
    store = tmp_path / "canonical"
    store.mkdir()
    (store / PROTECTED_STORE_SENTINEL).touch()
    _ed(5).store(project_dir=store)                       # 0 → 5 (growth) OK
    with pytest.raises(RuntimeError, match="PROTECTED"):  # 5 → 2 would clobber
        _ed(2).store(project_dir=store)
    # the ledger was NOT truncated
    assert len(ExperimentData.from_file(project_dir=store)) == 5


def test_protected_store_allows_equal_and_growth(tmp_path):
    store = tmp_path / "canonical"
    store.mkdir()
    (store / PROTECTED_STORE_SENTINEL).touch()
    _ed(3).store(project_dir=store)
    _ed(3).store(project_dir=store)   # equal count (e.g. a full re-store) → OK
    _ed(6).store(project_dir=store)   # superset (instrumented merge) → OK
    assert len(ExperimentData.from_file(project_dir=store)) == 6


def _ed_multiline(n: int) -> ExperimentData:
    """Rows whose output value contains embedded newlines (like a numpy-array
    repr), so output.csv has FAR more physical lines than logical CSV rows."""
    domain = Domain()
    domain.add_float("a", 0.0, 100.0)
    domain.add_output("z", exist_ok=True)
    rows = {
        i: ExperimentSample(
            _input_data={"a": float(i)},
            _output_data={"z": f"row{i}\nline2\nline3\nline4\nline5"},
            job_status=JobStatus.FINISHED,
        )
        for i in range(n)
    }
    return ExperimentData.from_data(data=rows, domain=domain)


def test_protected_guard_counts_csv_rows_not_physical_lines(tmp_path):
    """Bug 3: multi-line output values must not inflate the row count and
    falsely block a legitimate superset write (the D003/D004/D005 false
    'shrink' rejections). 3 rows × 5 physical lines = ~15 physical lines, but
    growing 3 → 6 logical rows must be ALLOWED."""
    store = tmp_path / "canonical"
    store.mkdir()
    (store / PROTECTED_STORE_SENTINEL).touch()
    _ed_multiline(3).store(project_dir=store)
    _ed_multiline(6).store(project_dir=store)   # superset → must pass
    assert len(ExperimentData.from_file(project_dir=store)) == 6


def _ed_status(n: int, status: JobStatus) -> ExperimentData:
    domain = Domain()
    domain.add_float("a", 0.0, 100.0)
    domain.add_output("z", exist_ok=True)
    rows = {
        i: ExperimentSample(
            _input_data={"a": float(i)},
            _output_data={"z": float(-i)},
            job_status=status,
        )
        for i in range(n)
    }
    return ExperimentData.from_data(data=rows, domain=domain)


def test_protected_store_refuses_finished_status_regression(tmp_path):
    """Bug 6: a worker re-storing the canonical dir with IN_PROGRESS rows must
    not reset rows the oracle already marked FINISHED (the FINISHED→IN_PROGRESS
    corruption that forced a manual repair delegation)."""
    store = tmp_path / "canonical"
    store.mkdir()
    (store / PROTECTED_STORE_SENTINEL).touch()
    _ed_status(5, JobStatus.FINISHED).store(project_dir=store)   # 5 FINISHED
    with pytest.raises(RuntimeError, match="FINISHED"):
        _ed_status(5, JobStatus.IN_PROGRESS).store(project_dir=store)
    # statuses survived the refused write
    assert ExperimentData.from_file(project_dir=store).is_all_finished()


def test_unprotected_store_truncates_normally(tmp_path):
    """No sentinel → core behavior unchanged (overwrite, no guard)."""
    store = tmp_path / "scratch"
    store.mkdir()
    _ed(5).store(project_dir=store)
    _ed(2).store(project_dir=store)   # would-shrink, but unprotected → allowed
    assert len(ExperimentData.from_file(project_dir=store)) == 2
