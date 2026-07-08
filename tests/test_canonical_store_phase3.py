"""Phase 3 canonical-store tests: RunStateSummary, RecallStore/QueryStore,
UNLEDGERED_EVALS drift nudge, DelegationLog.evals field, nodes.py wiring.

TDD: all tests written before implementation.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from a3dasm._src.delegation_log import DelegationLog
from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_domain_with_output(output_keys=("f",), input_keys=("x0",)) -> Domain:
    d = Domain()
    for k in input_keys:
        d.add_float(k, 0.0, 1.0)
    for k in output_keys:
        d.add_output(k, exist_ok=True)
    for k in ("_delegation_id", "_source", "_ts"):
        d.add_output(k, exist_ok=True)
    return d


def _make_sample(
    x0: float,
    f: float,
    delegation_id: str,
    source: str = "test",
    fidelity: float | None = None,
    fidelity_col: str = "fidelity",
) -> ExperimentSample:
    inp = {"x0": x0}
    if fidelity is not None:
        inp[fidelity_col] = fidelity
    out = {
        "f": f,
        "_delegation_id": delegation_id,
        "_source": source,
        "_ts": "2026-01-01T00:00:00+00:00",
    }
    return ExperimentSample(
        _input_data=inp,
        _output_data=out,
        job_status=JobStatus.FINISHED,
    )


def _build_store(
    store_dir: Path,
    rows: list[tuple],  # (x0, f, delegation_id)
    fidelity_col: str | None = None,
) -> None:
    """Write rows to the canonical store under store_dir."""
    output_keys = ["f", "_delegation_id", "_source", "_ts"]
    if fidelity_col:
        input_keys = ["x0", fidelity_col]
    else:
        input_keys = ["x0"]
    domain = Domain()
    for k in input_keys:
        domain.add_float(k, 0.0, 10.0)
    for k in output_keys:
        domain.add_output(k, exist_ok=True)

    samples = {}
    for i, row in enumerate(rows):
        if len(row) == 3:
            x0, f, did = row
            fidelity = None
        else:
            x0, f, did, fidelity = row
        inp = {"x0": x0}
        if fidelity_col and fidelity is not None:
            inp[fidelity_col] = fidelity
        out = {
            "f": f,
            "_delegation_id": did,
            "_source": "test",
            "_ts": "2026-01-01T00:00:00+00:00",
        }
        samples[i] = ExperimentSample(
            _input_data=inp,
            _output_data=out,
            job_status=JobStatus.FINISHED,
        )
    data = ExperimentData.from_data(data=samples, domain=domain)
    data.store(project_dir=store_dir)


def _make_dlog_record(
    log: DelegationLog,
    did: str = "D001",
    status: str = "DONE",
    evals: int = 0,
) -> None:
    log.record(
        id=did,
        from_node="strategizer",
        to_node="implementer",
        task="task",
        deliverable="## Report\n\n### Actions taken\n- ran\n"
        "### Files touched\n- none\n### Conclusions\nok\n### Numbers\nbest: 1.0\n",
        hypothesis_ids=["H1"],
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T01:00:00+00:00",
        status=status,
        evals=evals,
    )


# ===========================================================================
# Part A — RunStateSummary
# ===========================================================================


class TestRunStateSummaryEmpty:
    def test_from_store_returns_none_when_no_store(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        result = RunStateSummary.from_store(tmp_path)
        assert result is None

    def test_from_store_returns_none_when_only_dir_exists(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        (tmp_path / "experiment_data").mkdir()
        result = RunStateSummary.from_store(tmp_path)
        assert result is None


class TestRunStateSummaryPopulated:
    def test_n_rows(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        _build_store(tmp_path, [
            (0.1, 1.0, "D001"),
            (0.2, 2.0, "D001"),
            (0.3, 3.0, "D001"),
            (0.4, 4.0, "D002"),
            (0.5, 5.0, "D002"),
        ])
        s = RunStateSummary.from_store(tmp_path)
        assert s is not None
        assert s.n_rows == 5

    def test_n_per_delegation(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        _build_store(tmp_path, [
            (0.1, 1.0, "D001"),
            (0.2, 2.0, "D001"),
            (0.3, 3.0, "D002"),
        ])
        s = RunStateSummary.from_store(tmp_path)
        assert s.n_per_delegation == {"D001": 2, "D002": 1}

    def test_n_per_source(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        # source column is always "test" in _build_store
        _build_store(tmp_path, [(0.1, 1.0, "D001"), (0.2, 2.0, "D001")])
        s = RunStateSummary.from_store(tmp_path)
        assert "test" in s.n_per_source
        assert s.n_per_source["test"] == 2

    def test_output_stats_excludes_provenance(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        _build_store(tmp_path, [
            (0.1, 1.0, "D001"),
            (0.2, 2.0, "D001"),
            (0.3, 3.0, "D002"),
        ])
        s = RunStateSummary.from_store(tmp_path)
        # "f" should be in stats; provenance cols should not
        assert "f" in s.output_stats
        assert "_delegation_id" not in s.output_stats
        assert "_source" not in s.output_stats
        assert "_ts" not in s.output_stats
        stats = s.output_stats["f"]
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(3.0)
        assert stats["mean"] == pytest.approx(2.0)

    def test_n_per_fidelity_none_when_no_column(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        _build_store(tmp_path, [(0.1, 1.0, "D001")])
        s = RunStateSummary.from_store(tmp_path, fidelity_column="fidelity")
        assert s.n_per_fidelity is None

    def test_n_per_fidelity_populated_when_column_present(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        _build_store(tmp_path, [
            (0.1, 1.0, "D001", 1.0),
            (0.2, 2.0, "D001", 2.0),
            (0.3, 3.0, "D002", 1.0),
        ], fidelity_col="fidelity")
        s = RunStateSummary.from_store(tmp_path, fidelity_column="fidelity")
        assert s.n_per_fidelity is not None
        assert s.n_per_fidelity[1.0] == 2
        assert s.n_per_fidelity[2.0] == 1

    def test_mean_eval_wall_ms_excludes_pool_and_nan(self, tmp_path):
        """Spec A: overall mean per-eval wall-time, dropping D000/pool rows."""
        from a3dasm._src.instrumented import RunStateSummary
        dom = Domain()
        dom.add_float("x0", 0.0, 1.0)
        for k in ("f", "_delegation_id", "_source", "_ts", "_wall_ms"):
            dom.add_output(k, exist_ok=True)
        rows = [
            (1.0, "D001", "eval", 10.0),
            (2.0, "D001", "eval", 30.0),
            (9.0, "D000", "precomputed_pool", 0.0),  # pool → excluded
        ]
        samples = {}
        for i, (f, did, src, wall) in enumerate(rows):
            samples[i] = ExperimentSample(
                _input_data={"x0": 0.1 * i},
                _output_data={
                    "f": f, "_delegation_id": did, "_source": src,
                    "_ts": "2026-01-01T00:00:00+00:00", "_wall_ms": wall,
                },
                job_status=JobStatus.FINISHED,
            )
        data = ExperimentData.from_data(data=samples, domain=dom)
        data.store(project_dir=tmp_path)

        s = RunStateSummary.from_store(tmp_path)
        assert s.mean_eval_wall_ms == pytest.approx(20.0)  # (10+30)/2; pool out
        assert "mean eval wall-time" in s.format()

    def test_mean_eval_wall_ms_none_without_column(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        _build_store(tmp_path, [(0.1, 1.0, "D001")])  # no _wall_ms column
        s = RunStateSummary.from_store(tmp_path)
        assert s.mean_eval_wall_ms is None
        assert "mean eval wall-time" not in s.format()

    def test_fidelity_ignored_when_not_in_input_columns(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        # Store has no fidelity column in inputs
        _build_store(tmp_path, [(0.1, 1.0, "D001"), (0.2, 2.0, "D002")])
        s = RunStateSummary.from_store(tmp_path, fidelity_column="fidelity")
        assert s.n_per_fidelity is None


class TestRunStateSummaryFormat:
    def test_format_returns_string(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        _build_store(tmp_path, [(0.1, i * 0.1, "D001") for i in range(5)])
        s = RunStateSummary.from_store(tmp_path)
        text = s.format()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_format_within_25_lines(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        _build_store(tmp_path, [(i * 0.1, i * 0.5, f"D{(i % 3) + 1:03d}") for i in range(10)])
        s = RunStateSummary.from_store(tmp_path)
        lines = s.format().splitlines()
        assert len(lines) <= 25, f"format() returned {len(lines)} lines (max 25)"

    def test_format_mentions_row_count(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        _build_store(tmp_path, [(0.1, 1.0, "D001")] * 7)
        s = RunStateSummary.from_store(tmp_path)
        assert "7" in s.format()


class TestRunStateSummaryMtimeCache:
    def test_same_object_returned_on_repeated_calls(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        _build_store(tmp_path, [(0.1, 1.0, "D001")])
        s1 = RunStateSummary.from_store(tmp_path)
        s2 = RunStateSummary.from_store(tmp_path)
        assert s1 is s2

    def test_new_object_when_file_changes(self, tmp_path):
        from a3dasm._src.instrumented import RunStateSummary
        _build_store(tmp_path, [(0.1, 1.0, "D001")])
        s1 = RunStateSummary.from_store(tmp_path)
        # Touch output.csv to change mtime
        time.sleep(0.01)
        csv_path = tmp_path / "experiment_data" / "output.csv"
        csv_path.write_text(csv_path.read_text())  # rewrite same content
        s2 = RunStateSummary.from_store(tmp_path)
        assert s1 is not s2


# ===========================================================================
# Part B — RecallStore and QueryStore closures
# ===========================================================================


def _build_run_layout(tmp_path: Path) -> tuple[Path, Path]:
    """Create run_dir/debug/strategizer_notes/ and run_dir/experiment_data/."""
    run_dir = tmp_path / "run"
    notes_dir = run_dir / "debug" / "strategizer_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = run_dir / "experiment_data"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, notes_dir


def _make_strategizer_with_notes(tmp_path: Path, notes_dir: Path):
    """Build a minimal StrategizerNode with _current_notes_dir set."""
    from a3dasm._src.backends.base import Agent, Edge, Graph
    from a3dasm._src.nodes import StrategizerNode

    class StubAdapter:
        def __init__(self):
            self.closure_tools: dict = {}
            self.route_watcher = None

        def invoke(self, messages):
            return "## Report\n## Done\n"

    class StratAgent(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "WriteNote", "ReadNote", "RecallStore", "QueryStore"})
        description = "test strategizer"

    class WorkAgent(Agent):
        description = "test worker"

    spec = Graph(
        nodes={"strat": StratAgent(), "worker": WorkAgent()},
        edges=(Edge("strat", "worker"),),
        entry="strat",
    )

    adapter = StubAdapter()
    node = StrategizerNode(
        adapter=adapter,
        name="strat",
        outgoing=["worker"],
        spec=spec,
        study_dir=str(tmp_path),
    )
    node._current_notes_dir = notes_dir
    # Rebuild closures now that _current_notes_dir is set
    node.adapter.closure_tools.update(node._build_routing_closures())
    return node


class TestRecallStoreClosure:
    def test_recall_store_present_in_closures(self, tmp_path):
        run_dir, notes_dir = _build_run_layout(tmp_path)
        node = _make_strategizer_with_notes(tmp_path, notes_dir)
        assert "RecallStore" in node.adapter.closure_tools

    def test_recall_store_empty_message_when_no_store(self, tmp_path):
        run_dir, notes_dir = _build_run_layout(tmp_path)
        node = _make_strategizer_with_notes(tmp_path, notes_dir)
        result = node.adapter.closure_tools["RecallStore"]()
        assert "empty" in result.lower() or "no" in result.lower()

    def test_recall_store_returns_summary_when_store_exists(self, tmp_path):
        run_dir, notes_dir = _build_run_layout(tmp_path)
        # _derive_store_dir() returns run_dir/experiment_data.
        # RunStateSummary.from_store(run_dir/experiment_data) reads
        # run_dir/experiment_data/experiment_data/output.csv, so
        # _build_store must use run_dir/experiment_data as project_dir.
        store_dir = run_dir / "experiment_data"
        _build_store(store_dir, [
            (0.1, 1.0, "D001"),
            (0.2, 2.0, "D001"),
            (0.3, 3.0, "D002"),
        ])
        node = _make_strategizer_with_notes(tmp_path, notes_dir)
        result = node.adapter.closure_tools["RecallStore"]()
        assert isinstance(result, str)
        assert len(result) > 20
        # Should mention the total row count or delegation IDs
        assert "3" in result or "D001" in result or "D002" in result

    def test_recall_store_injected_without_delegation_log(self, tmp_path):
        """RecallStore must be injected even when delegation_log is None."""
        run_dir, notes_dir = _build_run_layout(tmp_path)
        from a3dasm._src.backends.base import Agent, Edge, Graph
        from a3dasm._src.nodes import StrategizerNode

        class StubAdapter:
            def __init__(self):
                self.closure_tools: dict = {}
                self.route_watcher = None

            def invoke(self, messages):
                return "done"

        class SA(Agent):
            role = "strategizer"
            tools = frozenset({"Done", "RecallStore", "QueryStore"})
            description = "s"

        class WA(Agent):
            description = "w"

        spec = Graph(
            nodes={"s": SA(), "w": WA()},
            edges=(Edge("s", "w"),),
            entry="s",
        )
        node = StrategizerNode(
            adapter=StubAdapter(),
            name="s",
            outgoing=["w"],
            spec=spec,
            delegation_log=None,  # explicitly None
        )
        node._current_notes_dir = notes_dir
        node.adapter.closure_tools.update(node._build_routing_closures())
        assert "RecallStore" in node.adapter.closure_tools


class TestQueryStoreClosure:
    def test_query_store_present_in_closures(self, tmp_path):
        run_dir, notes_dir = _build_run_layout(tmp_path)
        node = _make_strategizer_with_notes(tmp_path, notes_dir)
        assert "QueryStore" in node.adapter.closure_tools

    def test_query_store_no_store_returns_message(self, tmp_path):
        run_dir, notes_dir = _build_run_layout(tmp_path)
        node = _make_strategizer_with_notes(tmp_path, notes_dir)
        result = node.adapter.closure_tools["QueryStore"]()
        assert isinstance(result, str)

    def _setup_store_and_node(self, tmp_path):
        run_dir, notes_dir = _build_run_layout(tmp_path)
        # _derive_store_dir() returns run_dir/experiment_data.
        # Use run_dir/experiment_data as project_dir so data lands at
        # run_dir/experiment_data/experiment_data/output.csv ✓
        store_dir = run_dir / "experiment_data"
        _build_store(store_dir, [
            (0.1, 5.0, "D001"),
            (0.2, 3.0, "D001"),
            (0.3, 7.0, "D002"),
            (0.4, 2.0, "D002"),
            (0.5, 9.0, "D003"),
        ])
        node = _make_strategizer_with_notes(tmp_path, notes_dir)
        return node, store_dir

    def test_filter_by_bare_delegation_id(self, tmp_path):
        node, _ = self._setup_store_and_node(tmp_path)
        result = node.adapter.closure_tools["QueryStore"](
            delegation_ids="D001"
        )
        assert "D001" in result
        assert "D002" not in result
        assert "D003" not in result

    def test_filter_by_comma_string(self, tmp_path):
        node, _ = self._setup_store_and_node(tmp_path)
        result = node.adapter.closure_tools["QueryStore"](
            delegation_ids="D001,D002"
        )
        assert "D001" in result
        assert "D002" in result
        assert "D003" not in result

    def test_filter_by_json_string(self, tmp_path):
        node, _ = self._setup_store_and_node(tmp_path)
        result = node.adapter.closure_tools["QueryStore"](
            delegation_ids='["D001"]'
        )
        assert "D001" in result
        assert "D002" not in result

    def test_filter_by_list(self, tmp_path):
        node, _ = self._setup_store_and_node(tmp_path)
        result = node.adapter.closure_tools["QueryStore"](
            delegation_ids=["D001", "D003"]
        )
        assert "D001" in result
        assert "D002" not in result
        assert "D003" in result

    def test_n_best_returns_n_smallest(self, tmp_path):
        node, _ = self._setup_store_and_node(tmp_path)
        result = node.adapter.closure_tools["QueryStore"](
            n_best=2, output_name="f"
        )
        # Best 2 by "f" (ascending): f=2.0 (D002), f=3.0 (D001)
        assert "2.0" in result or "2" in result
        assert "3.0" in result or "3" in result
        # Worst f=9.0 should NOT appear
        assert "9.0" not in result and "9" not in result

    def test_n_best_accepts_string_arg(self, tmp_path):
        """MCP string-in tools pass n_best as a string ('2'); QueryStore must
        coerce it (pandas nsmallest does `if n <= 0`, which TypeErrors on str)."""
        node, _ = self._setup_store_and_node(tmp_path)
        result = node.adapter.closure_tools["QueryStore"](
            n_best="2", output_name="f"
        )
        assert "2.0" in result or "2" in result
        assert "9.0" not in result and "9" not in result

    def test_query_store_is_read_only(self, tmp_path):
        node, store_dir = self._setup_store_and_node(tmp_path)
        csv_path = store_dir / "experiment_data" / "output.csv"
        mtime_before = csv_path.stat().st_mtime
        node.adapter.closure_tools["QueryStore"](delegation_ids="D001")
        mtime_after = csv_path.stat().st_mtime
        assert mtime_before == mtime_after

    def test_filter_by_source(self, tmp_path):
        node, _ = self._setup_store_and_node(tmp_path)
        result = node.adapter.closure_tools["QueryStore"](source="test")
        # All rows have source="test", so we should get a non-empty result
        assert isinstance(result, str)
        assert "test" in result or "5" in result  # 5 rows


# ===========================================================================
# Part C — DelegationLog.evals field
# ===========================================================================


class TestDelegationLogEvalsField:
    def test_evals_field_stored_in_record(self, tmp_path):
        log = DelegationLog(tmp_path / "log.jsonl")
        _make_dlog_record(log, did="D001", evals=5)
        rec = log.query_all()[0]
        assert rec["evals"] == 5

    def test_evals_defaults_to_zero(self, tmp_path):
        log = DelegationLog(tmp_path / "log.jsonl")
        # Use record() without evals
        log.record(
            id="D001",
            from_node="s",
            to_node="i",
            task="t",
            deliverable="d",
            hypothesis_ids=["H1"],
            started_at="x",
            completed_at="y",
            status="DONE",
        )
        rec = log.query_all()[0]
        assert rec["evals"] == 0

    def test_evals_round_trip(self, tmp_path):
        log = DelegationLog(tmp_path / "log.jsonl")
        _make_dlog_record(log, did="D001", evals=42)
        path = tmp_path / "log.jsonl"
        raw = json.loads(path.read_text().strip())
        assert raw["evals"] == 42


# ===========================================================================
# Part C — ScienceMonitor UNLEDGERED_EVALS rule
# ===========================================================================


def _make_monitor(tmp_path, store_dir=None):
    from a3dasm._src.hypothesis_ledger import HypothesisLedger
    from a3dasm._src.science_monitor import ScienceMonitor
    ledger = HypothesisLedger(tmp_path)
    dlog = DelegationLog(tmp_path / "log.jsonl")
    mon = ScienceMonitor(ledger, dlog, store_dir=store_dir)
    return ledger, dlog, mon


def _propose(ledger) -> str:
    return ledger.propose(
        statement="Test claim",
        falsification_criterion="any counterexample",
        prediction="none found",
        prior=0.5,
        proposed_by="strategizer",
    )


class TestUnledgeredEvalsRule:
    def test_fires_when_evals_positive_and_store_empty(self, tmp_path):
        from a3dasm._src.science_monitor import ScienceMonitor
        store_dir = tmp_path / "run"
        store_dir.mkdir()
        ledger, dlog, mon = _make_monitor(tmp_path, store_dir=store_dir)
        _propose(ledger)
        _make_dlog_record(dlog, did="D001", status="DONE", evals=5)
        violations = mon.evaluate()
        rules = {v.rule for v in violations}
        assert "UNLEDGERED_EVALS" in rules

    def test_message_mentions_delegation_id(self, tmp_path):
        from a3dasm._src.science_monitor import ScienceMonitor
        store_dir = tmp_path / "run"
        store_dir.mkdir()
        ledger, dlog, mon = _make_monitor(tmp_path, store_dir=store_dir)
        _propose(ledger)
        _make_dlog_record(dlog, did="D001", status="DONE", evals=3)
        violations = mon.evaluate()
        msgs = [v.message for v in violations if v.rule == "UNLEDGERED_EVALS"]
        assert msgs
        assert "D001" in msgs[0]

    def test_silent_when_store_has_rows_for_delegation(self, tmp_path):
        from a3dasm._src.science_monitor import ScienceMonitor
        store_dir = tmp_path / "run"
        store_dir.mkdir()
        # Populate store with D001 rows
        _build_store(store_dir, [(0.1, 1.0, "D001"), (0.2, 2.0, "D001")])
        ledger, dlog, mon = _make_monitor(tmp_path, store_dir=store_dir)
        _propose(ledger)
        _make_dlog_record(dlog, did="D001", status="DONE", evals=2)
        violations = mon.evaluate()
        rules = {v.rule for v in violations}
        assert "UNLEDGERED_EVALS" not in rules

    def test_silent_when_store_dir_is_none(self, tmp_path):
        ledger, dlog, mon = _make_monitor(tmp_path, store_dir=None)
        _propose(ledger)
        _make_dlog_record(dlog, did="D001", status="DONE", evals=10)
        violations = mon.evaluate()
        rules = {v.rule for v in violations}
        assert "UNLEDGERED_EVALS" not in rules

    def test_silent_when_evals_zero(self, tmp_path):
        """No UNLEDGERED_EVALS when delegation reports 0 evals, even with no store."""
        store_dir = tmp_path / "run"
        store_dir.mkdir()
        ledger, dlog, mon = _make_monitor(tmp_path, store_dir=store_dir)
        _propose(ledger)
        _make_dlog_record(dlog, did="D001", status="DONE", evals=0)
        violations = mon.evaluate()
        rules = {v.rule for v in violations}
        assert "UNLEDGERED_EVALS" not in rules

    def test_silent_for_non_done_delegation(self, tmp_path):
        """FAILED delegations should not trigger UNLEDGERED_EVALS."""
        store_dir = tmp_path / "run"
        store_dir.mkdir()
        ledger, dlog, mon = _make_monitor(tmp_path, store_dir=store_dir)
        _propose(ledger)
        _make_dlog_record(dlog, did="D001", status="FAILED", evals=5)
        violations = mon.evaluate()
        rules = {v.rule for v in violations}
        assert "UNLEDGERED_EVALS" not in rules

    def test_rule_severity_is_warn(self, tmp_path):
        store_dir = tmp_path / "run"
        store_dir.mkdir()
        ledger, dlog, mon = _make_monitor(tmp_path, store_dir=store_dir)
        _propose(ledger)
        _make_dlog_record(dlog, did="D001", status="DONE", evals=5)
        violations = mon.evaluate()
        for v in violations:
            if v.rule == "UNLEDGERED_EVALS":
                assert v.severity == "warn"
                return
        pytest.fail("UNLEDGERED_EVALS violation not found")


# ===========================================================================
# Part C — nodes.py _run epilogue passes evals to delegation_log
# ===========================================================================


class TestNodesDelegationLogEvalsWiring:
    """Test that _run epilogue passes resolved evals into delegation_log.record."""

    def test_delegation_log_record_has_evals(self, tmp_path):
        """After a delegation completes, the log record should have evals set."""
        from a3dasm._src.backends.base import Agent, Edge, Graph
        from a3dasm._src.delegation_log import DelegationLog
        from a3dasm._src.nodes import StrategizerNode

        class CountingAdapter:
            """Calls ReportEvals(7) and then returns a report."""
            def __init__(self):
                self.closure_tools: dict = {}
                self.route_watcher = None
                self.last_usage = {}

            def copy(self):
                a = CountingAdapter()
                a.closure_tools = dict(self.closure_tools)
                return a

            def invoke(self, messages):
                # Call ReportEvals via the closure if available
                if "ReportEvals" in self.closure_tools:
                    self.closure_tools["ReportEvals"](7)
                return (
                    "## Report\n\n### Actions taken\n- ran\n\n"
                    "### Files touched\n- none\n\n"
                    "### Conclusions\nok\n\n"
                    "### Numbers\nbest: 1.0\n"
                )

        class SA(Agent):
            role = "strategizer"
            tools = frozenset({"Done", "RecallStore", "QueryStore"})
            description = "s"

        class WA(Agent):
            description = "w"

        dlog_path = tmp_path / "log.jsonl"
        dlog = DelegationLog(dlog_path)
        spec = Graph(
            nodes={"s": SA(), "w": WA()},
            edges=(Edge("s", "w"),),
            entry="s",
        )
        worker_adapter = CountingAdapter()
        main_adapter = CountingAdapter()

        node = StrategizerNode(
            adapter=main_adapter,
            name="s",
            outgoing=["w"],
            spec=spec,
            delegation_log=dlog,
            worker_adapters={"w": worker_adapter},
        )
        # Manually trigger a delegation and wait
        # Build the Delegate closure tool
        closures = node._build_routing_closures()
        # Call Delegate(wait=True)
        result = closures["Delegate"](
            target="w",
            intent="do something",
            expected_report="a report",
            hypothesis_ids=None,
            wait=True,
        )
        # Now check the delegation log
        records = dlog.query_all()
        done_records = [r for r in records if r.get("status") == "DONE"]
        assert done_records, f"No DONE records in log. All: {records}"
        rec = done_records[0]
        assert "evals" in rec
        assert rec["evals"] == 7


# ===========================================================================
# Part C — store_dir wired in StrategizerNode.__call__
# ===========================================================================


class TestScienceMonitorStoreDirWiring:
    def test_store_dir_set_on_science_monitor_after_call(self, tmp_path):
        """After __call__ with run_dir set, science_monitor.store_dir is a Path."""
        import tempfile

        from langchain_core.messages import HumanMessage
        from langgraph.types import Command

        from a3dasm._src.backends.base import Agent, Edge, Graph
        from a3dasm._src.delegation_log import DelegationLog
        from a3dasm._src.graph_state import AgenticState
        from a3dasm._src.hypothesis_ledger import HypothesisLedger
        from a3dasm._src.nodes import StrategizerNode

        run_dir = tmp_path / "run"
        (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
        study_dir = tmp_path / "study"
        study_dir.mkdir()
        (study_dir / "pipeline.py").write_text("# test\n")

        dlog = DelegationLog(run_dir / "debug" / "delegation_log.jsonl")

        class DoneAdapter:
            def __init__(self):
                self.closure_tools: dict = {}
                self.route_watcher = None
                self.last_usage = {}

            def invoke(self, messages):
                # Call Done twice to close the run
                self.closure_tools["Done"]("summary")
                return self.closure_tools["Done"]("summary")

        class SA(Agent):
            role = "strategizer"
            tools = frozenset({"Done", "RecallStore", "QueryStore"})
            description = "s"

        class WA(Agent):
            description = "w"

        spec = Graph(
            nodes={"s": SA(), "w": WA()},
            edges=(Edge("s", "w"),),
            entry="s",
        )

        ledger = HypothesisLedger(run_dir / "debug" / "strategizer_notes")
        node = StrategizerNode(
            adapter=DoneAdapter(),
            name="s",
            outgoing=["w"],
            spec=spec,
            delegation_log=dlog,
            notes_dir=run_dir / "debug" / "strategizer_notes",
            study_dir=str(study_dir),
        )

        state = AgenticState(
            messages=[HumanMessage(content="test")],
            study_dir=str(study_dir),
            done=False,
            last_report=None,
            total_delegations=0,
            run_dir=str(run_dir),
        )

        node(state)

        # After __call__, science_monitor should have store_dir set
        assert node._science_monitor is not None
        assert node._science_monitor.store_dir is not None
        assert isinstance(node._science_monitor.store_dir, Path)


# ===========================================================================
# Narrow-metering invariant: surrogates/agent-built generators are NOT metered
# ===========================================================================


def test_surrogate_generator_not_metered(tmp_path):
    """A plain DataGenerator (surrogate stand-in) run WITHOUT get_evaluator
    writes nothing to the canonical store — only InstrumentedDataGenerator
    stamps _delegation_id rows. Documents the identity-metering invariant
    that keeps exploration free."""
    from f3dasm import datagenerator
    from a3dasm._src.instrumented import RunStateSummary

    store_dir = tmp_path / "store"
    _build_store(store_dir, [(0.1, 1.0, "D001"), (0.2, 2.0, "D001")])
    before = dict(RunStateSummary.from_store(store_dir).n_per_delegation)

    @datagenerator(output_names=["y"])
    def surrogate(x0: float) -> float:  # agent's OWN model, not the oracle
        return x0 * 10.0

    d = Domain()
    d.add_float("x0", 0.0, 1.0)
    d.add_output("y", exist_ok=True)
    own_samples = {
        i: ExperimentSample(
            _input_data={"x0": 0.1 * i},
            _output_data={},
            job_status=JobStatus.OPEN,
        )
        for i in range(5)
    }
    own = ExperimentData.from_data(data=own_samples, domain=d)
    surrogate.call(own, mode="sequential")  # free; not via get_evaluator()

    after = dict(RunStateSummary.from_store(store_dir).n_per_delegation)
    assert after == before  # canonical store untouched by the surrogate


# ---------------------------------------------------------------------------
# _stamped_eval_count + unledgered-evals bounce retry prompt
# ---------------------------------------------------------------------------

def test_stamped_eval_count_counts_only_provenance_rows(tmp_path):
    """_stamped_eval_count reports ONLY provenance-stamped rows, so a
    delegation that evaluated off-ledger reads as 0 (flagged off-ledger)."""
    from a3dasm._src.nodes import _stamped_eval_count
    _build_store(tmp_path, [
        (1.0, 0.5, "D001"),
        (2.0, 0.6, "D001"),
        (3.0, 0.7, "D002"),
    ])
    assert _stamped_eval_count(tmp_path, "D001") == 2
    assert _stamped_eval_count(tmp_path, "D002") == 1
    # A delegation that wrote no stamped rows → 0 (flagged off-ledger, not re-run).
    assert _stamped_eval_count(tmp_path, "D003") == 0
    # No store at all → 0, never raises.
    assert _stamped_eval_count(None, "D001") == 0


def test_ledger_breakdown_tool_renders_per_experiment_split(tmp_path):
    """LedgerBreakdown() reads the live stores under run_dir/experiment_data and
    renders a per-experiment / per-delegation split — the report-time provenance
    that prevents hardcoding stale counts (run 20260628T001710 UNGATED)."""
    from a3dasm._src.backends.base import Agent, Edge, Graph
    from a3dasm._src.nodes import StrategizerNode
    from f3dasm._src.design.domain import Domain
    from f3dasm._src.experimentdata import ExperimentData
    from f3dasm._src.experimentsample import ExperimentSample, JobStatus

    def seed(store_dir, n, did):
        dom = Domain()
        dom.add_float("x", 0.0, 1.0)
        dom.add_output("score", exist_ok=True)
        dom.add_output("_delegation_id", exist_ok=True)
        rows = {i: ExperimentSample(
            _input_data={"x": i / max(n, 1)},
            _output_data={"score": 0.5, "_delegation_id": did},
            job_status=JobStatus.FINISHED) for i in range(n)}
        ExperimentData.from_data(data=rows, domain=dom).store(project_dir=store_dir)

    import json as _json
    run_dir, notes_dir = _build_run_layout(tmp_path)
    store_root = run_dir / "experiment_data"
    seed(store_root, 30, "D004")
    seed(store_root / "polar", 50, "D006")
    # eval_budget grounds the spent/remaining line so the agent reads it, not
    # hand-computes it (run 20260628T130525 flipped spent<->remaining → UNGATED).
    (run_dir / "debug" / "run_config.json").write_text(
        _json.dumps({"eval_budget": 300}))

    class StubAdapter:
        def __init__(self):
            self.closure_tools: dict = {}
            self.route_watcher = None

        def invoke(self, messages):
            return ""

    class StratAgent(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "LedgerBreakdown"})
        description = "test strategizer"

    class WorkAgent(Agent):
        description = "test worker"

    spec = Graph(
        nodes={"strat": StratAgent(), "worker": WorkAgent()},
        edges=(Edge("strat", "worker"),), entry="strat")
    node = StrategizerNode(
        adapter=StubAdapter(), name="strat", outgoing=["worker"],
        spec=spec, study_dir=str(tmp_path))
    node._current_notes_dir = notes_dir
    node.adapter.closure_tools.update(node._build_routing_closures())

    assert "LedgerBreakdown" in node.adapter.closure_tools
    out = node.adapter.closure_tools["LedgerBreakdown"]()
    assert "default: 30 total" in out and "D004: 30" in out
    assert "polar: 50 total" in out and "D006: 50" in out
    # grounded against budget: 80 spent of 300 → 220 remaining (read, not computed)
    assert "80 of 300 eval budget spent" in out and "220 remaining" in out
