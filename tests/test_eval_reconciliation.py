"""Unit tests for _reconcile_delegation_evals (§3 honesty check).

The ledger (summed across every experiment store) is the source of truth for the
eval count. When a source is registered and a worker claimed evals but none are
provenance-stamped in ANY store, the delegation evaluated off-ledger: the
truthful count is 0 and the condition is flagged. These tests pin the branching
in isolation by stubbing the two count helpers.
"""
from __future__ import annotations

import a3dasm._src.nodes.parsing as parsing
from a3dasm._src.nodes import _reconcile_delegation_evals


def _stub_store(monkeypatch, *, stamped: int, resolved: int) -> None:
    monkeypatch.setattr(parsing, "_stamped_eval_count", lambda *a, **k: stamped)
    monkeypatch.setattr(
        parsing, "_resolve_delegation_evals", lambda *a, **k: resolved
    )


def test_off_ledger_when_registered_claimed_but_unstamped(monkeypatch):
    # Worker claims 200 evals; the store has 0 stamped rows; a source IS
    # registered → off-ledger, counted as 0 (the D004 scenario).
    _stub_store(monkeypatch, stamped=0, resolved=200)
    evals, off_ledger, stamped = _reconcile_delegation_evals(
        run_exp_dir="/tmp/store", delegation_id="D004",
        claimed=200, source_registered=True,
    )
    assert (evals, off_ledger, stamped) == (0, True, 0)


def test_ledgered_evals_are_trusted(monkeypatch):
    # Store has 200 stamped rows → not off-ledger; trust the resolver.
    _stub_store(monkeypatch, stamped=200, resolved=200)
    evals, off_ledger, stamped = _reconcile_delegation_evals(
        run_exp_dir="/tmp/store", delegation_id="D004",
        claimed=200, source_registered=True,
    )
    assert (evals, off_ledger, stamped) == (200, False, 200)


def test_no_source_registered_keeps_honour_system(monkeypatch):
    # No source registered → cannot be ledgered; honour-system fallback stands
    # (e.g. lookup-direct studies). Not flagged off-ledger.
    _stub_store(monkeypatch, stamped=0, resolved=50)
    evals, off_ledger, stamped = _reconcile_delegation_evals(
        run_exp_dir="/tmp/store", delegation_id="D004",
        claimed=50, source_registered=False,
    )
    assert (evals, off_ledger, stamped) == (50, False, 0)


def test_zero_claimed_is_never_off_ledger(monkeypatch):
    # A delegation that claimed nothing (e.g. a setup/lit-review step) is never
    # flagged, even with a registered source and an empty store.
    _stub_store(monkeypatch, stamped=0, resolved=0)
    evals, off_ledger, stamped = _reconcile_delegation_evals(
        run_exp_dir="/tmp/store", delegation_id="D002",
        claimed=0, source_registered=True,
    )
    assert (evals, off_ledger, stamped) == (0, False, 0)
