"""Namespace-aware eval accounting (fixes the run 20260626T231202 cascade).

Two namespace-blind defects, one root: per-namespace stores are isolated, but
the run-level eval accounting and the unledgered-evals guard both keyed on the
single canonical store.

1. `total_ledgered_evals` must sum rows across the canonical store AND every
   namespace store (so `evals_used` / the soft budget see all real evals).
2. `delegation_evals` must count a delegation's rows across EVERY store by its
   stamp, so the unledgered-evals guard finds the rows wherever the worker wrote
   them (not just the canonical store, which falsely reads 0 → false off-ledger
   bounce → re-run thrash → duplicate rows). Provenance-based, so it is immune to
   how the experiment was selected (Delegate arg or get_evaluator call site).
"""
from __future__ import annotations

from pathlib import Path

from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus


def _seed_store(store_dir: Path, n: int, delegation_id: str = "D001") -> None:
    """Write `n` finished rows to the ExperimentData project at store_dir."""
    dom = Domain()
    dom.add_float("x", 0.0, 1.0)
    dom.add_output("score", exist_ok=True)
    dom.add_output("_delegation_id", exist_ok=True)
    rows = {
        i: ExperimentSample(
            _input_data={"x": i / max(n, 1)},
            _output_data={"score": 0.5, "_delegation_id": delegation_id},
            job_status=JobStatus.FINISHED,
        )
        for i in range(n)
    }
    ExperimentData.from_data(data=rows, domain=dom).store(project_dir=store_dir)


def test_total_ledgered_evals_sums_across_namespaces(tmp_path):
    from a3dasm._src.instrumented import total_ledgered_evals

    # canonical store at <store>/experiment_data; namespace 'polar' at
    # <store>/polar/experiment_data — the on-disk layout get_evaluator uses.
    store = tmp_path / "experiment_data"
    _seed_store(store, 100)                 # canonical (cartesian)
    _seed_store(store / "polar", 100)       # namespace
    # Canonical-only count is the OLD (buggy) view.
    from a3dasm._src.instrumented import RunStateSummary
    assert RunStateSummary.from_store(store).n_rows == 100
    # Aggregate count sees both.
    assert total_ledgered_evals(store) == 200


def test_total_ledgered_evals_canonical_only_is_unchanged(tmp_path):
    """No namespaces → identical to the canonical row count (back-compat)."""
    from a3dasm._src.instrumented import total_ledgered_evals

    store = tmp_path / "experiment_data"
    _seed_store(store, 42)
    assert total_ledgered_evals(store) == 42


def test_delegation_evals_counts_a_delegation_across_stores(tmp_path):
    """Provenance-based: a delegation's rows are counted by its stamp wherever
    they landed — canonical OR any experiment store — never by guessing which."""
    from a3dasm._src.instrumented import delegation_evals

    store = tmp_path / "experiment_data"
    _seed_store(store, 100, "D001")                 # canonical: D001
    _seed_store(store / "ring", 80, "D002")         # experiment 'ring': D002
    assert delegation_evals(store, "D001") == 100
    assert delegation_evals(store, "D002") == 80    # found in the 'ring' store
    assert delegation_evals(store, "D999") == 0


def test_reconcile_not_fooled_by_call_site_experiment(tmp_path):
    """Run 20260627T045747 regression. A delegation that evaluated correctly via
    get_evaluator(namespace='ring') at the CALL SITE wrote its rows to the 'ring'
    store, not canonical. The reconciliation must find them by provenance and NOT
    falsely flag it off-ledger (which forced the wasteful re-run that blew the
    watchdog). The run root is passed — no namespace is guessed."""
    from a3dasm._src.nodes.parsing import _reconcile_delegation_evals

    store = tmp_path / "experiment_data"
    _seed_store(store, 100, "D001")                 # someone else, canonical
    _seed_store(store / "ring", 80, "D004")         # D004 wrote to 'ring'
    evals, off_ledger, stamped = _reconcile_delegation_evals(
        store, "D004", claimed=80, source_registered=True)
    assert evals == 80 and off_ledger is False and stamped == 80


def test_run_ledger_counts_rows_across_namespaces(tmp_path):
    """The KPI ledger's ledger_rows must sum the canonical store AND every
    namespace store — they sit at DIFFERENT depths under <run>/experiment_data
    (canonical: experiment_data/experiment_data; namespace: <ns>/experiment_data),
    so a single glob misses one (regression: the first fix attempt did)."""
    import sys
    sys.path.insert(0, "studies")
    import run_ledger

    run_dir = tmp_path / "runs" / "T"
    (run_dir / "debug").mkdir(parents=True)
    # canonical store at <run>/experiment_data (its data → experiment_data/...)
    _seed_store(run_dir / "experiment_data", 100, "D001")
    # namespace 'polar' store at <run>/experiment_data/polar
    _seed_store(run_dir / "experiment_data" / "polar", 100, "D002")

    row = run_ledger.extract(run_dir)
    assert row["ledger_rows"] == 200


def test_ledger_breakdown_is_per_experiment_per_delegation(tmp_path):
    """Report-time provenance: ledger_breakdown surfaces exactly what the
    accounting counts, so a writeup derives counts instead of hardcoding them
    (run 20260628T001710 hardcoded 70 polar evals; the ledger held 90 → UNGATED).
    """
    from a3dasm._src.instrumented import ledger_breakdown

    store = tmp_path / "experiment_data"
    _seed_store(store, 30, delegation_id="D004")          # default / cartesian
    _seed_store(store / "polar", 50, delegation_id="D006")
    _seed_store(store / "radial_focused", 15, delegation_id="D004")

    rows = ledger_breakdown(store)
    by_name = {r["experiment"]: r for r in rows}
    assert by_name["default"]["total"] == 30
    assert by_name["default"]["per_delegation"] == {"D004": 30}
    assert by_name["polar"]["total"] == 50
    assert by_name["polar"]["per_delegation"] == {"D006": 50}
    assert by_name["radial_focused"]["per_delegation"] == {"D004": 15}
    # default first, then experiments alphabetically.
    assert [r["experiment"] for r in rows] == ["default", "polar", "radial_focused"]


def test_ledger_breakdown_empty_store_is_empty_not_error(tmp_path):
    from a3dasm._src.instrumented import ledger_breakdown
    assert ledger_breakdown(tmp_path / "nope") == []


def test_load_experiments_returns_every_store_keyed_by_name(tmp_path):
    """The multi-namespace load idiom: one call returns all experiment stores
    (default + each design experiment), so a notebook does not silently miss
    polar by loading only the default store (run 20260628T130525 CRITICAL)."""
    from a3dasm._src.instrumented import load_experiments

    store = tmp_path / "experiment_data"
    _seed_store(store, 30, delegation_id="D004")          # default
    _seed_store(store / "polar", 50, delegation_id="D006")

    exps = load_experiments(store)
    assert set(exps) == {"default", "polar"}
    assert len(exps["default"]) == 30
    assert len(exps["polar"]) == 50


def test_load_experiments_single_study_is_just_default(tmp_path):
    from a3dasm._src.instrumented import load_experiments
    store = tmp_path / "experiment_data"
    _seed_store(store, 12)
    exps = load_experiments(store)
    assert set(exps) == {"default"} and len(exps["default"]) == 12


def test_load_experiments_empty_run_is_empty_dict(tmp_path):
    from a3dasm._src.instrumented import load_experiments
    assert load_experiments(tmp_path / "nope") == {}


def test_load_experiments_is_exported_from_agentic():
    from a3dasm import load_experiments  # noqa: F401
