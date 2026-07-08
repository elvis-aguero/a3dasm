"""Tests for InstrumentedDataGenerator and get_evaluator.

TDD: these tests were written BEFORE the implementation.
Run with:
    uv run pytest tests/agentic/test_instrumented.py -v --no-cov
"""
from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from pathlib import Path

import pytest

from f3dasm._src.core import DataGenerator, datagenerator
from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_domain() -> Domain:
    d = Domain()
    d.add_float("x0", 0.0, 1.0)
    return d


def _make_sample(x0: float = 0.5) -> ExperimentSample:
    return ExperimentSample(
        _input_data={"x0": x0},
        _output_data={},
        job_status=JobStatus.OPEN,
    )


class _SumGenerator(DataGenerator):
    """Trivial DataGenerator: f = sum of inputs."""

    def execute(
        self, experiment_sample: ExperimentSample, **kwargs
    ) -> ExperimentSample:
        val = sum(experiment_sample._input_data.values())
        experiment_sample._output_data["f"] = val
        experiment_sample.job_status = JobStatus.FINISHED
        return experiment_sample


# ---------------------------------------------------------------------------
# 1. Provenance stamping
# ---------------------------------------------------------------------------


def test_execute_stamps_provenance(tmp_path):
    from a3dasm._src.instrumented import InstrumentedDataGenerator

    gen = InstrumentedDataGenerator(
        inner=_SumGenerator(),
        store_dir=tmp_path,
        delegation_id="D001",
        source="test_source",
        flush_every=1,
    )
    sample = _make_sample(0.3)
    gen.execute(sample)

    data = ExperimentData.from_file(project_dir=tmp_path)
    df_in, df_out = data.to_pandas()

    assert "_delegation_id" in df_out.columns, df_out.columns.tolist()
    assert "_source" in df_out.columns, df_out.columns.tolist()
    assert "_ts" in df_out.columns, df_out.columns.tolist()
    assert "f" in df_out.columns, df_out.columns.tolist()

    row = df_out.iloc[0]
    assert row["_delegation_id"] == "D001"
    assert row["_source"] == "test_source"
    # _ts should be a non-empty string
    assert isinstance(row["_ts"], str) and len(row["_ts"]) > 0


def test_execute_marks_finished_in_canonical_store(tmp_path):
    """B (run 20260629T191754): a completed eval must persist as FINISHED in the
    canonical store. A real oracle does NOT self-mark its sample finished —
    f3dasm's _run_sample marks the agent's *working* copy, not the deepcopy the
    instrumented wrapper buffers. So InstrumentedDataGenerator must stamp it, or
    finished rows persist as IN_PROGRESS (defeating is_all_finished(), the
    FINISHED-regression store guard, and resumption). The stub here deliberately
    leaves job_status untouched, mimicking the real oracle."""
    from a3dasm._src.instrumented import InstrumentedDataGenerator

    class _NoMarkGenerator(DataGenerator):
        def execute(self, experiment_sample, **kwargs):
            experiment_sample._output_data["f"] = 1.0  # no job_status change
            return experiment_sample

    gen = InstrumentedDataGenerator(
        inner=_NoMarkGenerator(), store_dir=tmp_path,
        delegation_id="D001", source="s", flush_every=1,
    )
    # Mirrors the live dispatch: get_open_job() marks a row IN_PROGRESS before
    # handing it to the evaluator — the exact state that leaked to disk.
    gen.execute(ExperimentSample(
        _input_data={"x0": 0.5}, _output_data={},
        job_status=JobStatus.IN_PROGRESS))

    data = ExperimentData.from_file(project_dir=tmp_path)
    assert data.is_all_finished(), (
        f"completed eval persisted as non-FINISHED: {data.jobs.tolist()}")


def test_execute_stamps_wall_ms(tmp_path):
    """Spec A: each eval carries its own wall-time (_wall_ms), generically."""
    import time as _time

    from a3dasm._src.instrumented import (
        _PROVENANCE_COLS,
        InstrumentedDataGenerator,
    )

    class _Slow(DataGenerator):
        def execute(self, experiment_sample, **kwargs):
            _time.sleep(0.03)
            experiment_sample._output_data["f"] = 1.0
            experiment_sample.job_status = JobStatus.FINISHED
            return experiment_sample

    gen = InstrumentedDataGenerator(
        inner=_Slow(), store_dir=tmp_path, delegation_id="D001",
        source="t", flush_every=1)
    out = gen.execute(_make_sample(0.3))

    assert "_wall_ms" in out._output_data
    assert isinstance(out._output_data["_wall_ms"], float)
    assert out._output_data["_wall_ms"] >= 20.0  # slept ~30ms, allow slack
    # provenance convention: counted as metadata, excluded from value stats
    assert "_wall_ms" in _PROVENANCE_COLS


def test_to_numpy_excludes_underscore_provenance(tmp_path):
    """All provenance columns are now underscore-prefixed (_delegation_id,
    _source, _ts), so core to_numpy() drops them and returns a clean numeric
    array instead of an object-dtype array contaminated by the metadata."""
    import numpy as np
    from a3dasm._src.instrumented import InstrumentedDataGenerator

    gen = InstrumentedDataGenerator(
        inner=_SumGenerator(),
        store_dir=tmp_path,
        delegation_id="D001",
        source="test_source",
        flush_every=1,
    )
    gen.execute(_make_sample(0.3))

    data = ExperimentData.from_file(project_dir=tmp_path)
    _, out_arr = data.to_numpy()

    # only the real output "f" survives → numeric, not object dtype
    assert out_arr.shape[1] == 1, out_arr
    assert np.issubdtype(out_arr.dtype, np.floating), out_arr.dtype


# ---------------------------------------------------------------------------
# 2. Concurrent appends — no rows lost
# ---------------------------------------------------------------------------


def _worker(store_dir, delegation_id, n_samples, lock_path):
    """Run in a thread; each gets its own InstrumentedDataGenerator."""
    from a3dasm._src.instrumented import InstrumentedDataGenerator

    gen = InstrumentedDataGenerator(
        inner=_SumGenerator(),
        store_dir=store_dir,
        delegation_id=delegation_id,
        source="concurrent_test",
        lock_path=lock_path,
        flush_every=1,
    )
    for i in range(n_samples):
        gen.execute(_make_sample(float(i) * 0.05))
    gen.flush()


def test_concurrent_appends_no_loss(tmp_path):
    lock_path = tmp_path / "experiment_data" / ".lock"
    delegation_ids = ["D001", "D002", "D003"]
    K = 10  # samples per delegation

    threads = [
        threading.Thread(
            target=_worker,
            args=(tmp_path, did, K, lock_path),
        )
        for did in delegation_ids
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = ExperimentData.from_file(project_dir=tmp_path)
    df_in, df_out = data.to_pandas()

    total = len(df_out)
    assert total == len(delegation_ids) * K, (
        f"Expected {len(delegation_ids) * K} rows, got {total}"
    )

    for did in delegation_ids:
        count = (df_out["_delegation_id"] == did).sum()
        assert count == K, (
            f"Expected {K} rows for {did}, got {count}"
        )


# ---------------------------------------------------------------------------
# 4. Provenance survives + reindex
# ---------------------------------------------------------------------------


def test_provenance_survives_plus_reindex(tmp_path):
    from a3dasm._src.instrumented import InstrumentedDataGenerator

    lock_path = tmp_path / "experiment_data" / ".lock"

    gen1 = InstrumentedDataGenerator(
        inner=_SumGenerator(),
        store_dir=tmp_path,
        delegation_id="D001",
        source="s1",
        lock_path=lock_path,
        flush_every=2,
    )
    gen2 = InstrumentedDataGenerator(
        inner=_SumGenerator(),
        store_dir=tmp_path,
        delegation_id="D002",
        source="s2",
        lock_path=lock_path,
        flush_every=2,
    )

    gen1.execute(_make_sample(0.1))
    gen1.execute(_make_sample(0.2))  # triggers flush

    gen2.execute(_make_sample(0.3))
    gen2.execute(_make_sample(0.4))  # triggers flush

    data = ExperimentData.from_file(project_dir=tmp_path)
    _, df_out = data.to_pandas()

    assert len(df_out) == 4
    d1_rows = df_out[df_out["_delegation_id"] == "D001"]
    d2_rows = df_out[df_out["_delegation_id"] == "D002"]
    assert len(d1_rows) == 2
    assert len(d2_rows) == 2


# ---------------------------------------------------------------------------
# 5. get_evaluator binds delegation_id from cwd
# ---------------------------------------------------------------------------


def test_get_evaluator_binds_delegation_id_from_cwd(
    tmp_path, monkeypatch
):
    from a3dasm._src.instrumented import get_evaluator

    # Build the workspace: .../runs/ts/debug/delegations/D007
    debug_dir = tmp_path / "runs" / "ts" / "debug"
    delegation_dir = debug_dir / "delegations" / "D007"
    delegation_dir.mkdir(parents=True)

    store_dir = tmp_path / "store"
    store_dir.mkdir()

    # Register a tiny entrypoint so get_evaluator() (no inner) resolves the
    # source and we can still assert delegation_id derivation from cwd.
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "sum_eval.py").write_text(
        "def evaluate_kw(**kwargs):\n"
        "    return float(sum(kwargs.values()))\n"
    )

    run_config = {
        "store_dir": str(store_dir),
        "lock_path": str(store_dir / "experiment_data" / ".lock"),
        "source": "test_eval",
        "evaluator_name": "test_eval",
        "study_dir": str(study_dir),
        "fidelity_column": None,
        "evaluator_entrypoint": "sum_eval.py:evaluate_kw",
        "evaluator_output_names": ["f"],
    }
    run_config_path = debug_dir / "run_config.json"
    run_config_path.write_text(json.dumps(run_config))

    monkeypatch.chdir(delegation_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)

    gen = get_evaluator()
    assert gen.delegation_id == "D007"


# ---------------------------------------------------------------------------
# 6. get_evaluator raises outside delegation workspace
# ---------------------------------------------------------------------------


def test_get_evaluator_raises_outside_delegation(
    tmp_path, monkeypatch
):
    from a3dasm._src.instrumented import get_evaluator

    # cwd is not a D### directory
    bad_dir = tmp_path / "not_a_delegation"
    bad_dir.mkdir()
    monkeypatch.chdir(bad_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)

    with pytest.raises(ValueError, match="get_evaluator"):
        get_evaluator()


# ---------------------------------------------------------------------------
# 7. flush_every batching
# ---------------------------------------------------------------------------


def test_flush_every_batches(tmp_path):
    from a3dasm._src.instrumented import InstrumentedDataGenerator

    gen = InstrumentedDataGenerator(
        inner=_SumGenerator(),
        store_dir=tmp_path,
        delegation_id="D001",
        source="batch_test",
        flush_every=5,
    )

    for i in range(4):
        gen.execute(_make_sample(float(i) * 0.1))

    # Store should not exist yet (or be empty if dir exists)
    exp_data_dir = tmp_path / "experiment_data"
    if exp_data_dir.exists():
        # If store was written it would have CSV files
        input_csv = exp_data_dir / "input_data.csv"
        output_csv = exp_data_dir / "output_data.csv"
        if output_csv.exists():
            import pandas as pd
            df = pd.read_csv(output_csv, index_col=0)
            assert len(df) == 0, (
                f"Expected 0 rows before flush_every=5 triggered, "
                f"got {len(df)}"
            )

    # 5th execute triggers flush
    gen.execute(_make_sample(0.4))

    data = ExperimentData.from_file(project_dir=tmp_path)
    _, df_out = data.to_pandas()
    assert len(df_out) == 5, f"Expected 5 rows after flush, got {len(df_out)}"


# ---------------------------------------------------------------------------
# 8. Public API: importable from a3dasm
# ---------------------------------------------------------------------------


def test_public_api_importable():
    # get_evaluator() is the ONE agent-facing door.
    from a3dasm import get_evaluator  # noqa: F401
    import a3dasm as _agentic

    # InstrumentedDataGenerator is deliberately NOT public — it stays internal
    # so agents cannot construct a store-redirected evaluator (§1 seal).
    assert "InstrumentedDataGenerator" not in _agentic.__all__
    assert not hasattr(_agentic, "InstrumentedDataGenerator")
    # ...but it remains importable internally for the runtime and tests.
    from a3dasm._src.instrumented import (  # noqa: F401
        InstrumentedDataGenerator,
    )


def test_store_rows_accumulate_across_generator_instances(tmp_path):
    """Two generators in the SAME delegation accumulate rows in store.

    Observed live: a worker built one generator per phase; the store
    accumulated rows (600) correctly while a counter undercounted (300).
    The store is now the single source of truth for eval counts.
    """
    from a3dasm._src.instrumented import (
        InstrumentedDataGenerator,
        RunStateSummary,
    )

    store_dir = tmp_path / "store"
    store_dir.mkdir()

    def make_gen():
        @datagenerator(output_names=["f"])
        def inner(**kw):
            return float(sum(kw.values()))

        return InstrumentedDataGenerator(
            inner, store_dir, "D001",
            source="s", flush_every=1,
        )

    g1 = make_gen()
    g1.execute(_make_sample(0.1))
    g1.execute(_make_sample(0.2))

    g2 = make_gen()  # new instance, same delegation
    g2.execute(_make_sample(0.3))

    summary = RunStateSummary.from_store(store_dir)
    assert summary is not None
    # Store accumulates across both generator instances
    assert summary.n_per_delegation.get("D001", 0) == 3


def test_wall_per_delegation_and_footer(tmp_path):
    """from_store groups _wall_ms by delegation; delegation_footer renders it.

    Auto-appended to each delegation report so the strategizer plans its budget
    on measured sim cost (the 36.5x cost-prior miss in run 20260625T014520).
    """
    from a3dasm._src.instrumented import (
        InstrumentedDataGenerator,
        RunStateSummary,
    )

    store_dir = tmp_path / "store"
    store_dir.mkdir()

    def make_gen(did):
        @datagenerator(output_names=["f"])
        def inner(**kw):
            return float(sum(kw.values()))

        return InstrumentedDataGenerator(
            inner, store_dir, did, source="s", flush_every=1,
        )

    make_gen("D001").execute(_make_sample(0.1))
    make_gen("D001").execute(_make_sample(0.2))
    make_gen("D002").execute(_make_sample(0.3))

    summary = RunStateSummary.from_store(store_dir)
    assert summary is not None

    wpd = summary.wall_per_delegation
    assert wpd["D001"]["n"] == 2
    assert wpd["D002"]["n"] == 1
    # measured wall-times are real positive floats; max ≥ median; total ≈ sum
    d1 = wpd["D001"]
    assert d1["max_ms"] >= d1["median_ms"] > 0
    assert d1["total_ms"] >= d1["max_ms"]

    footer = summary.delegation_footer("D001")
    assert footer is not None
    assert "D001" in footer
    assert "per-eval wall-time" in footer
    assert "ledger total so far: 3" in footer  # 2 + 1 rows
    # No wall budget passed → no budget line.
    assert "wall budget remaining" not in footer

    # With a wall budget, the remaining line appears (telemetry, not a stop).
    footer_b = summary.delegation_footer(
        "D001", wall_remaining_s=1800.0, wall_budget_s=3600.0)
    assert "wall budget remaining" in footer_b

    # Peak-RAM telemetry: rendered when given, against the hard cap; absent otherwise.
    footer_mem = summary.delegation_footer(
        "D001", peak_rss_bytes=2 * 1024**3, ram_cap_bytes=4 * 1024**3)
    assert "peak RAM (this delegation): 2.00 GB of 4.0 GB hard cap" in footer_mem
    assert "peak RAM" not in summary.delegation_footer("D001")  # none → no line

    # A delegation that wrote no rows gets no footer (not a fabricated zero).
    assert summary.delegation_footer("D999") is None


def test_flush_merges_into_typed_canonical_domain(tmp_path):
    """Regression (run 20260624T021359): flushing a batch into a canonical store
    whose domain has a TYPED (add_int) parameter must not raise. The batch domain
    declares inputs as untyped base Parameter(); the merge previously failed with
    'Cannot add non-continuous parameter to continuous!'."""
    import pandas as pd
    from a3dasm._src.instrumented import InstrumentedDataGenerator

    # Pre-seed a canonical store with a TYPED domain (int + float).
    domain = Domain()
    domain.add_int("n", 1, 3)
    domain.add_float("x0", 0.0, 1.0)
    seed = ExperimentData(
        domain=domain, input_data=pd.DataFrame([{"n": 2, "x0": 0.5}]))
    seed.store(project_dir=tmp_path)

    # Flush a new eval through the instrumented generator (untyped batch domain).
    gen = InstrumentedDataGenerator(
        inner=_SumGenerator(), store_dir=tmp_path,
        delegation_id="D002", flush_every=1)
    gen.execute(ExperimentSample(
        _input_data={"n": 3, "x0": 0.25}, _output_data={},
        job_status=JobStatus.OPEN))  # must NOT raise the typed/untyped ValueError

    _, df_out = ExperimentData.from_file(project_dir=tmp_path).to_pandas()
    assert len(df_out) == 2
