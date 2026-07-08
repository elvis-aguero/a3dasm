"""End-to-end seam test: real _init_canonical_store consumed by real
get_evaluator from a simulated worker cwd, through append and
the runtime's eval-resolution helper. No LLM, deterministic.
"""

from __future__ import annotations

from pathlib import Path

from f3dasm import ExperimentData
from f3dasm._src.samplers import RandomUniform
from f3dasm.design import Domain
from a3dasm._src.agent_runtime import _init_canonical_store
from a3dasm._src.instrumented import RunStateSummary, get_evaluator
from a3dasm._src.nodes import _resolve_delegation_evals


def test_runtime_config_to_store_roundtrip(tmp_path, monkeypatch):
    # 1. runtime sets up the canonical store + run_config sidecar
    run_dir = tmp_path / "runs" / "T0"
    (run_dir / "debug" / "delegations" / "D001").mkdir(parents=True)
    study_dir = tmp_path / "agentic_demo_study"
    study_dir.mkdir()
    # Register the oracle as a study entrypoint (the runtime's job); the worker
    # then reaches it through the ONE door, get_evaluator(), with no overrides.
    (study_dir / "bb_eval.py").write_text(
        "def evaluate_kw(**kwargs):\n"
        "    return float(sum(kwargs.values()))\n"
    )
    cfg = _init_canonical_store(
        run_dir, study_dir,
        evaluator_config={
            "entrypoint": "bb_eval.py:evaluate_kw",
            "output_names": ["f"],
        },
    )

    # 2. a worker, cwd == its delegation folder, obtains the evaluator
    monkeypatch.chdir(run_dir / "debug" / "delegations" / "D001")

    gen = get_evaluator()
    assert gen.delegation_id == "D001"
    assert gen.source == "agentic_demo_study"  # evaluator_name → source

    # 3. evaluate through the instrumented path
    d = Domain()
    d.add_float("x", low=0.0, high=1.0)
    d.add_float("y", low=0.0, high=1.0)
    data = ExperimentData(domain=d)
    data = RandomUniform(seed=0).call(data, n_samples=4)
    data = gen.call(data, mode="sequential")
    gen.flush()

    # 4. canonical store has the rows with provenance
    store_dir = Path(cfg["store_dir"])
    canon = ExperimentData.from_file(project_dir=store_dir)
    _, out = canon.to_pandas()
    assert len(out) == 4
    assert (out["_delegation_id"] == "D001").all()
    assert (out["_source"] == "agentic_demo_study").all()
    assert out["_ts"].notna().all()
    assert "f" in out.columns

    # 5. store is the single source of truth — RunStateSummary.from_store
    # reads the same store_dir that InstrumentedDataGenerator wrote to.
    # This is the regression guard for the write-path/read-path agreement.
    summary = RunStateSummary.from_store(cfg["store_dir"])
    assert summary is not None, (
        "RunStateSummary.from_store(run_config['store_dir']) returned None; "
        "write-path and read-path store_dir disagree."
    )
    assert summary.n_per_delegation.get("D001", 0) == 4, (
        f"Expected 4 rows for D001, got {summary.n_per_delegation}"
    )

    # 6. _resolve_delegation_evals counts store rows, not a counter file
    evals = _resolve_delegation_evals(store_dir, "D001", reported=999)
    assert evals == 4, (
        f"Expected 4 (from store), got {evals}; reported=999 should be "
        "overridden by the ledger."
    )
