"""Tests for ScienceMonitor.

Only UNLEDGERED_EVALS remains as a runtime rule. Hard invariants
(EVIDENCE_DELEGATION_EXISTS, SUPPORTED_WITHOUT_ATTACK) have moved to
HypothesisUpdate in strategizer.py. Soft rules (STALE_OPEN,
UNANCHORED_DELEGATION, POSTERIOR_INERTIA, EVIDENCE_NUMBERS_MATCH) removed.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.hypothesis_ledger import HypothesisLedger
from a3dasm._src.instrumented import RunStateSummary
from a3dasm._src.science_monitor import ScienceMonitor


def make_world(tmp_path):
    ledger = HypothesisLedger(tmp_path)
    dlog = DelegationLog(tmp_path / "delegation_log.jsonl")
    drift_records = []
    mon = ScienceMonitor(
        ledger, dlog, diagnostics_writer=drift_records.append)
    return ledger, dlog, mon, drift_records


REPORT = (
    "## Report\n\n### Actions taken\n- ran\n\n### Conclusions\nok\n\n"
    "### Numbers\nbest_y: 1.47\n"
)


# ---------------------------------------------------------------------------
# UNLEDGERED_EVALS — must NOT fire when store has rows
# ---------------------------------------------------------------------------

def test_unledgered_does_not_fire_when_store_has_rows(tmp_path):
    """False-positive guard: rule is silent when rows exist for delegation."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    ledger.propose("claim", "criterion", "pred", 0.5, proposed_by="s")
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE", evals=5)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    stub = RunStateSummary(
        n_rows=5, n_per_delegation={"D001": 5},
        n_per_source={}, n_per_fidelity=None, output_stats={})
    with patch.object(RunStateSummary, "from_store", return_value=stub):
        violations = mon.evaluate()
    assert "UNLEDGERED_EVALS" not in {v.rule for v in violations}


def test_unledgered_silent_when_rows_in_an_experiment_store(tmp_path):
    """Provenance-based: a delegation that wrote to an EXPERIMENT (namespace)
    store, not the default one, is NOT false-flagged (run 20260627T203327 D003
    wrote to 'polar' and the old canonical-only check wrongly warned)."""
    from f3dasm._src.design.domain import Domain
    from f3dasm._src.experimentdata import ExperimentData
    from f3dasm._src.experimentsample import ExperimentSample, JobStatus

    ledger, dlog, mon, _ = make_world(tmp_path)
    ledger.propose("claim", "criterion", "pred", 0.5, proposed_by="s")
    dlog.record(
        id="D003", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE", evals=100)
    store_dir = tmp_path / "experiment_data"
    # D003 wrote to the 'polar' experiment store, NOT the default store.
    dom = Domain()
    dom.add_float("r", 0.0, 1.0)
    dom.add_output("score", exist_ok=True)
    dom.add_output("_delegation_id", exist_ok=True)
    ExperimentData.from_data(
        data={0: ExperimentSample(
            _input_data={"r": 0.75},
            _output_data={"score": 1.0, "_delegation_id": "D003"},
            job_status=JobStatus.FINISHED)},
        domain=dom,
    ).store(project_dir=store_dir / "polar")
    mon.store_dir = str(store_dir)
    violations = mon.evaluate()  # real stores, no patch
    assert "UNLEDGERED_EVALS" not in {v.rule for v in violations}


def test_unledgered_fires_when_store_has_no_rows(tmp_path):
    """UNLEDGERED_EVALS fires when delegation reported evals but wrote nothing."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    ledger.propose("claim", "criterion", "pred", 0.5, proposed_by="s")
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE", evals=10)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    with patch.object(RunStateSummary, "from_store", return_value=None):
        violations = mon.evaluate()
    assert "UNLEDGERED_EVALS" in {v.rule for v in violations}


def test_unledgered_does_not_fire_when_no_evals(tmp_path):
    """UNLEDGERED_EVALS must NOT fire for delegations that reported 0 evals."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE", evals=0)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    with patch.object(RunStateSummary, "from_store", return_value=None):
        violations = mon.evaluate()
    assert "UNLEDGERED_EVALS" not in {v.rule for v in violations}


# ---------------------------------------------------------------------------
# drain() basics
# ---------------------------------------------------------------------------

def test_drain_returns_empty_when_healthy(tmp_path):
    """drain() returns empty string when no violations fire."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    ledger.propose("claim", "criterion", "pred", 0.5, proposed_by="s")
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE")
    assert mon.drain() == ""


def test_drain_formats_unledgered_violation(tmp_path):
    """drain() formats the UNLEDGERED_EVALS violation into an injected message."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE", evals=5)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    with patch.object(RunStateSummary, "from_store", return_value=None):
        result = mon.drain()
    assert "UNLEDGERED_EVALS" in result
    assert "D001" in result


def test_drain_caps_output_to_max_inject(tmp_path):
    """drain() caps at max_inject=2 full lines + a digest line for the rest."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    for i in range(1, 4):
        dlog.record(
            id=f"D00{i}", from_node="strategizer", to_node="implementer",
            task="t", deliverable=REPORT, hypothesis_ids=[],
            started_at="x", completed_at="y", status="DONE", evals=5)
    # 3 unledgered delegations → 3 distinct violation keys
    with patch.object(RunStateSummary, "from_store", return_value=None):
        result = mon.drain()
    full_lines = [
        ln for ln in result.splitlines()
        if ln.startswith("[SCIENCE MONITOR — ")]
    assert len(full_lines) == mon._max_inject
    digest_lines = [
        ln for ln in result.splitlines()
        if ln.startswith("[SCIENCE MONITOR] +")]
    assert len(digest_lines) == 1


def test_diagnostics_written_once_per_live_violation(tmp_path):
    """diagnostics_writer called once per new violation, not on re-drain."""
    ledger, dlog, mon, drift_records = make_world(tmp_path)
    dlog.record(
        id="D001", from_node="strategizer", to_node="implementer",
        task="t", deliverable=REPORT, hypothesis_ids=[],
        started_at="x", completed_at="y", status="DONE", evals=5)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    with patch.object(RunStateSummary, "from_store", return_value=None):
        mon.drain()
        count1 = sum(1 for r in drift_records if r["rule"] == "UNLEDGERED_EVALS")
        assert count1 == 1
        mon.drain()
        count2 = sum(1 for r in drift_records if r["rule"] == "UNLEDGERED_EVALS")
        assert count2 == 1, "should not re-log a still-live violation"


# ---------------------------------------------------------------------------
# Removed rules must NOT fire (regression guard)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", [
    "EVIDENCE_NUMBERS_MATCH",
    "STALE_OPEN",
    "UNANCHORED_DELEGATION",
    "POSTERIOR_INERTIA",
])
def test_removed_rule_does_not_fire(tmp_path, rule):
    """Removed rules must never appear in evaluate()."""
    ledger, dlog, mon, _ = make_world(tmp_path)
    h = ledger.propose("claim", "criterion", "pred", 0.5, proposed_by="s")
    bare = "## Report\n\n### Conclusions\nno numbers.\n"
    for i in range(1, 5):
        dlog.record(
            id=f"D00{i}", from_node="strategizer", to_node="implementer",
            task="t", deliverable=bare, hypothesis_ids=[h],
            started_at="x", completed_at="y", status="DONE")
    violations = mon.evaluate()
    fired = {v.rule for v in violations}
    assert rule not in fired, f"{rule} was expected removed but fired: {fired}"


# ---------------------------------------------------------------------------
# UNLEDGERED_EVALS — datagenerator role is exempt (pre-registration by spec)
# ---------------------------------------------------------------------------

def test_unledgered_exempts_datagenerator_role(tmp_path):
    """The datagenerator validates ONE sample via gen.call() — its spec forbids
    get_evaluator() pre-registration — so its reported eval can NEVER be
    ledgered. UNLEDGERED_EVALS must not false-flag the datagenerator role,
    while an implementer with the same unledgered condition IS still flagged.
    Regression: run 20260705T181941 (D001/D005 datagenerator false positives).
    """
    ledger = HypothesisLedger(tmp_path)
    dlog = DelegationLog(tmp_path / "delegation_log.jsonl")

    def role_of(name):
        return {"datagenerator": "datagenerator",
                "implementer": "implementer"}.get(name, name)

    mon = ScienceMonitor(ledger, dlog, role_of=role_of)
    for did, to in (("D001", "datagenerator"), ("D002", "implementer")):
        dlog.record(
            id=did, from_node="strategizer", to_node=to, task="t",
            deliverable=REPORT, hypothesis_ids=[],
            started_at="2026-01-01T00:00:00+00:00",
            completed_at="2026-01-01T01:00:00+00:00", status="DONE", evals=3)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)

    flagged = {v.h_id for v in mon.evaluate() if v.rule == "UNLEDGERED_EVALS"}
    assert "D001" not in flagged, "datagenerator validation must be exempt"
    assert "D002" in flagged, "implementer bypass must still be flagged"


# ---------------------------------------------------------------------------
# UNSTAMPED_ROWS — the reverse-direction detector: rows in the store with no
# provenance owner (the unstamped-write door). Warn-only, principled nudge.
# ---------------------------------------------------------------------------

def test_unstamped_rows_fires_on_ownerless_rows(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    # 6 physical rows, 5 attributed to D001 -> 1 ownerless row.
    stub = RunStateSummary(
        n_rows=6, n_per_delegation={"D001": 5},
        n_per_source={}, n_per_fidelity=None, output_stats={})
    with patch.object(RunStateSummary, "from_store", return_value=stub):
        violations = mon.evaluate()
    unstamped = [v for v in violations if v.rule == "UNSTAMPED_ROWS"]
    assert len(unstamped) == 1
    assert unstamped[0].severity == "warn"          # never blocks
    assert "1 row" in unstamped[0].message


def test_unstamped_rows_silent_when_every_row_attributed(tmp_path):
    ledger, dlog, mon, _ = make_world(tmp_path)
    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()
    mon.store_dir = str(store_dir)
    stub = RunStateSummary(
        n_rows=5, n_per_delegation={"D001": 5},
        n_per_source={}, n_per_fidelity=None, output_stats={})
    with patch.object(RunStateSummary, "from_store", return_value=stub):
        violations = mon.evaluate()
    assert "UNSTAMPED_ROWS" not in {v.rule for v in violations}


# ---------------------------------------------------------------------------
# DUPLICATE_EVALUATION — nudge on repeated re-evaluation of an already-known
# design point. Counter-based (fires on 3 NEW duplicate rows since the last
# check, then resets), rate-limited (>=60s between nudges), capped (max 2
# nudges per delegation). Regression for run example_study/20260713T221841
# (backlog #24 / spec 07): D003 re-evaluated 38 unique points up to 10x each,
# 122 of 160 rows pure waste.
# ---------------------------------------------------------------------------

def _store_rows(store_dir, rows):
    """rows: list of (x1, x2, delegation_id). Overwrites store_dir wholesale —
    tests call this repeatedly with a growing row list to simulate a live
    campaign accumulating duplicate evaluations over time."""
    from f3dasm._src.design.domain import Domain
    from f3dasm._src.experimentdata import ExperimentData
    from f3dasm._src.experimentsample import ExperimentSample, JobStatus

    dom = Domain()
    dom.add_float("x1", -5.0, 5.0)
    dom.add_float("x2", -5.0, 5.0)
    dom.add_output("y", exist_ok=True)
    dom.add_output("_delegation_id", exist_ok=True)
    samples = {
        i: ExperimentSample(
            _input_data={"x1": x1, "x2": x2},
            _output_data={"y": 0.0, "_delegation_id": did},
            job_status=JobStatus.FINISHED)
        for i, (x1, x2, did) in enumerate(rows)
    }
    ExperimentData.from_data(data=samples, domain=dom).store(
        project_dir=store_dir)


def _make_monitor_with_clock(tmp_path):
    clock = {"t": 1000.0}
    ledger, dlog, mon, drift = make_world(tmp_path)
    store_dir = tmp_path / "experiment_data"
    mon.store_dir = str(store_dir)
    mon._now = lambda: clock["t"]
    return mon, store_dir, clock


def test_duplicate_eval_does_not_fire_below_threshold(tmp_path):
    mon, store_dir, _ = _make_monitor_with_clock(tmp_path)
    # 1 unique point evaluated 3x = 2 duplicate rows -> below the 3-dup bar.
    _store_rows(store_dir, [(0.1, 0.2, "D003")] * 3)
    violations = mon.evaluate()
    assert "DUPLICATE_EVALUATION" not in {v.rule for v in violations}


def test_duplicate_eval_fires_after_three_new_duplicates(tmp_path):
    mon, store_dir, _ = _make_monitor_with_clock(tmp_path)
    # 1 unique point evaluated 4x = 3 duplicate rows -> hits the bar.
    _store_rows(store_dir, [(0.1, 0.2, "D003")] * 4)
    violations = mon.evaluate()
    dup = [v for v in violations if v.rule == "DUPLICATE_EVALUATION"]
    assert len(dup) == 1
    assert dup[0].severity == "warn"
    assert dup[0].h_id == "D003"


def test_duplicate_eval_counter_resets_after_nudge(tmp_path):
    mon, store_dir, _ = _make_monitor_with_clock(tmp_path)
    _store_rows(store_dir, [(0.1, 0.2, "D003")] * 4)
    first = mon.evaluate()
    assert "DUPLICATE_EVALUATION" in {v.rule for v in first}
    # No NEW rows added -> counter is 0 again, must not re-fire immediately.
    second = mon.evaluate()
    assert "DUPLICATE_EVALUATION" not in {v.rule for v in second}


def test_duplicate_eval_respects_cooldown(tmp_path):
    mon, store_dir, clock = _make_monitor_with_clock(tmp_path)
    _store_rows(store_dir, [(0.1, 0.2, "D003")] * 4)
    assert "DUPLICATE_EVALUATION" in {v.rule for v in mon.evaluate()}
    # 3 more new duplicates, but only 10s later -> still within the 60s cooldown.
    clock["t"] += 10
    _store_rows(store_dir, [(0.1, 0.2, "D003")] * 4 + [(0.3, 0.4, "D003")] * 4)
    assert "DUPLICATE_EVALUATION" not in {
        v.rule for v in mon.evaluate()}


def test_duplicate_eval_fires_again_after_cooldown(tmp_path):
    mon, store_dir, clock = _make_monitor_with_clock(tmp_path)
    _store_rows(store_dir, [(0.1, 0.2, "D003")] * 4)
    assert "DUPLICATE_EVALUATION" in {v.rule for v in mon.evaluate()}
    clock["t"] += 61  # past the 60s cooldown
    _store_rows(store_dir, [(0.1, 0.2, "D003")] * 4 + [(0.3, 0.4, "D003")] * 4)
    assert "DUPLICATE_EVALUATION" in {v.rule for v in mon.evaluate()}


def test_duplicate_eval_max_two_nudges_per_delegation(tmp_path):
    mon, store_dir, clock = _make_monitor_with_clock(tmp_path)
    _store_rows(store_dir, [(0.1, 0.2, "D003")] * 4)
    assert "DUPLICATE_EVALUATION" in {v.rule for v in mon.evaluate()}  # nudge 1
    clock["t"] += 61
    _store_rows(store_dir, [(0.1, 0.2, "D003")] * 4 + [(0.3, 0.4, "D003")] * 4)
    assert "DUPLICATE_EVALUATION" in {v.rule for v in mon.evaluate()}  # nudge 2
    clock["t"] += 61
    _store_rows(
        store_dir,
        [(0.1, 0.2, "D003")] * 4 + [(0.3, 0.4, "D003")] * 4
        + [(0.5, 0.6, "D003")] * 4)
    # 3rd batch of new duplicates, cooldown clear -> capped at 2, must NOT fire.
    assert "DUPLICATE_EVALUATION" not in {v.rule for v in mon.evaluate()}
