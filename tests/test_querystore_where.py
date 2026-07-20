"""QueryStore compound-predicate (`where`) + `limit` — spec 09.

An agent must be able to ask the ledger a compound feasibility question in one
call ("coilable and compresses and low-strain, ranked by objective") instead of
falling back to ExperimentData.from_file + pandas by hand (the friction hit in
run 20260715T002538 by D005 and critic-1). `where` is a pandas query() over the
joined inputs+outputs frame; `limit` lifts the 20-row default cap.
"""
from __future__ import annotations

from pathlib import Path

from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def _build_store(store_dir: Path, rows) -> None:
    """rows: (x0, f, coilable, mcs, delegation_id)."""
    domain = Domain()
    domain.add_float("x0", 0.0, 10.0)
    for k in ("f", "coilable", "mcs", "_delegation_id", "_source", "_ts"):
        domain.add_output(k, exist_ok=True)
    samples = {}
    for i, (x0, f, coil, mcs, did) in enumerate(rows):
        samples[i] = ExperimentSample(
            _input_data={"x0": x0},
            _output_data={"f": f, "coilable": coil, "mcs": mcs,
                          "_delegation_id": did, "_source": "test",
                          "_ts": "2026-01-01T00:00:00+00:00"},
            job_status=JobStatus.FINISHED,
        )
    ExperimentData.from_data(data=samples, domain=domain).store(
        project_dir=store_dir)


def _querystore(tmp_path):
    run_dir = tmp_path / "runs" / "T1"
    (run_dir / "debug").mkdir(parents=True)
    _build_store(run_dir / "experiment_data", [
        (0.1, 1.0, 1, 0.95, "D001"),   # feasible
        (0.2, 2.0, 1, 0.50, "D001"),   # coilable but mcs<0.9 -> infeasible
        (0.3, 3.0, 0, 0.99, "D002"),   # not coilable
        (0.4, 2.5, 1, 0.92, "D002"),   # feasible, highest feasible f
        (0.5, 0.5, 1, 0.91, "D001"),   # feasible, low f
    ])

    class S(Agent):
        role = "strategizer"
        tools = frozenset({"Done"})
        description = "s"

    class Impl(Agent):
        role = "implementer"
        description = "i"
        tools = frozenset({"QueryStore"})

    class Lit(Agent):
        role = "literature_reviewer"
        description = "l"

    spec = Graph(
        nodes={"strategizer": S(), "implementer": Impl(),
               "literature_reviewer": Lit()},
        edges=(Edge("strategizer", "implementer"),
               Edge("implementer", "literature_reviewer")),
        entry="strategizer")
    dlog = DelegationLog(run_dir / "debug" / "delegation_log.jsonl")
    n = StrategizerNode(
        _Stub(), name="implementer", outgoing=["literature_reviewer"],
        spec=spec, worker_adapters={"literature_reviewer": _Stub()},
        notes_dir=None, delegation_log=dlog)
    return n._build_routing_closures()["QueryStore"]


def test_where_compound_feasibility_predicate(tmp_path):
    q = _querystore(tmp_path)
    out = q(where="coilable==1 and mcs>=0.90")
    assert "3 rows match" in out           # rows 0, 3, 4
    assert "2.0" not in out or "mcs" in out  # row1 (mcs 0.5) excluded from data


def test_where_plus_n_best_ranks_feasible_only(tmp_path):
    q = _querystore(tmp_path)
    # best FEASIBLE by f, maximizing -> row3 (f=2.5), not row2 (f=3.0, infeasible)
    out = q(where="coilable==1 and mcs>=0.90", n_best=1,
            output_name="f", minimize=False)
    assert "2.5" in out
    assert "3.0" not in out                # infeasible top-f excluded by where


def test_n_best_output_includes_input_columns(tmp_path):
    # Regression: a stray post-filter (`c in best_rows.columns`, the OUTPUTS
    # frame) silently dropped every input column just added for this branch,
    # so n_best regressed to output_name + _delegation_id only — the exact
    # friction two independent agents (critic + strategizer) hit in run
    # 20260717T014507, needing several manual where= calls to reconstruct a
    # multi-column table that n_best was already supposed to return in one.
    q = _querystore(tmp_path)
    out = q(where="coilable==1 and mcs>=0.90", n_best=1,
            output_name="f", minimize=False)
    assert "x0" in out
    assert "0.4" in out


def test_where_arithmetic_on_input_columns(tmp_path):
    q = _querystore(tmp_path)
    out = q(where="x0*10 >= 3")             # x0 in {0.3,0.4,0.5} -> 3 rows
    assert "3 rows match" in out


def test_bad_where_returns_error_not_raises(tmp_path):
    q = _querystore(tmp_path)
    out = q(where="no_such_column == 1")
    assert out.startswith("ERROR:")
    assert "Available columns" in out       # self-documenting


def test_limit_caps_and_reports_remainder(tmp_path):
    q = _querystore(tmp_path)
    out = q(limit=2)
    assert "Showing 2" in out
    assert "more not shown" in out          # 3 remaining flagged


def test_no_args_lists_all_under_cap(tmp_path):
    q = _querystore(tmp_path)
    out = q()
    assert "5 rows match" in out
    assert "more not shown" not in out      # 5 < default 20


def test_default_list_view_includes_input_columns(tmp_path):
    """Observability (#1): the list/where view must surface INPUT columns so a
    design's coordinates are directly verifiable — all 3 critics hit this."""
    q = _querystore(tmp_path)
    out = q()
    assert "x0" in out, f"input column not surfaced in list view: {out!r}"
    # a specific coordinate value is visible (x0=0.1, the first row)
    assert "0.1" in out


def test_where_input_columns_also_surface(tmp_path):
    q = _querystore(tmp_path)
    out = q(where="coilable==1 and mcs>=0.90")
    assert "x0" in out          # inputs shown alongside the where-filtered rows


def test_zero_match_reports_scanned_total_unambiguously(tmp_path):
    """Observability (#1): a true zero must state how many rows were scanned, so
    it can't be confused with a missing column (which returns an ERROR)."""
    q = _querystore(tmp_path)
    out = q(where="coilable==1 and mcs>=9.0")   # nothing matches
    assert "0 of 5 scanned" in out
    assert "TRUE zero" in out


def _querystore_with_namespace(tmp_path):
    """Same 5-row default store as _querystore, PLUS a 'meander' namespace
    store with its own rows sharing _source='test' with the default store —
    matching real runs, where _source is always the study name and can never
    disambiguate a namespace (run 20260718T031519: 4 agents independently hit
    this, one got a false-positive baseline row back believing it belonged to
    a design family)."""
    run_dir = tmp_path / "runs" / "T1"
    (run_dir / "debug").mkdir(parents=True)
    _build_store(run_dir / "experiment_data", [
        (0.1, 1.0, 1, 0.95, "D001"),
        (0.4, 2.5, 1, 0.92, "D002"),
    ])
    _build_store(run_dir / "experiment_data" / "meander", [
        (0.4, 2.5, 1, 0.92, "D003"),   # deliberately same f/coilable/mcs as
        (0.9, 9.0, 1, 0.99, "D004"),   # the default store's D002 row above
    ])

    class S(Agent):
        role = "strategizer"
        tools = frozenset({"Done"})
        description = "s"

    class Impl(Agent):
        role = "implementer"
        description = "i"
        tools = frozenset({"QueryStore"})

    class Lit(Agent):
        role = "literature_reviewer"
        description = "l"

    spec = Graph(
        nodes={"strategizer": S(), "implementer": Impl(),
               "literature_reviewer": Lit()},
        edges=(Edge("strategizer", "implementer"),
               Edge("implementer", "literature_reviewer")),
        entry="strategizer")
    dlog = DelegationLog(run_dir / "debug" / "delegation_log.jsonl")
    n = StrategizerNode(
        _Stub(), name="implementer", outgoing=["literature_reviewer"],
        spec=spec, worker_adapters={"literature_reviewer": _Stub()},
        notes_dir=None, delegation_log=dlog)
    return n._build_routing_closures()["QueryStore"]


def test_namespace_column_always_shown(tmp_path):
    """_namespace must be visible even with NO namespace= filter — a
    namespace row must never be silently indistinguishable from a baseline
    row just because nobody asked to filter by namespace yet."""
    q = _querystore_with_namespace(tmp_path)
    out = q()
    assert "4 rows match" in out
    assert "_namespace" in out
    assert "default" in out
    assert "meander" in out


def test_namespace_filters_to_one_store(tmp_path):
    q = _querystore_with_namespace(tmp_path)
    out = q(namespace="meander")
    assert "2 rows match" in out
    assert "D003" in out and "D004" in out
    assert "D001" not in out and "D002" not in out


def test_source_cannot_disambiguate_namespace(tmp_path):
    """source= filters _source (the study name — identical across every
    namespace); it must NOT be mistaken for a namespace filter."""
    q = _querystore_with_namespace(tmp_path)
    out = q(source="test")
    assert "4 rows match" in out, (
        "source= matched all 4 rows across both stores, as expected — "
        f"it cannot isolate a namespace: {out!r}"
    )


def test_namespace_surfaces_in_n_best_view(tmp_path):
    q = _querystore_with_namespace(tmp_path)
    out = q(n_best=1, output_name="f", minimize=False)
    assert "9.0" in out          # meander's D004 row is the global best
    assert "meander" in out


def test_zero_match_reports_namespace_in_applied_filters(tmp_path):
    q = _querystore_with_namespace(tmp_path)
    out = q(namespace="elliptical_rings")   # never opened in this test
    assert "0 of 4 scanned" in out
    assert "namespace='elliptical_rings'" in out
    # and a genuinely bad column still ERRORs (not a silent 0)
    bad = q(where="nonexistent_col==1")
    assert bad.startswith("ERROR:")
