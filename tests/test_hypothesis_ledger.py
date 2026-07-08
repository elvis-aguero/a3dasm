"""Tests for HypothesisLedger — Popperian schema."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from a3dasm._src.hypothesis_ledger import (
    HypothesisEntry,
    HypothesisLedger,
)


def fresh_ledger(tmp_path: Path) -> HypothesisLedger:
    return HypothesisLedger(tmp_path)


def propose_ok(ledger, statement="Optimal t/L lies in [0.07, 0.10]",
               prior=0.55):
    return ledger.propose(
        statement=statement,
        falsification_criterion=(
            "Any feasible design outside [0.07, 0.10] with load "
            "exceeding the best in-range value"
        ),
        prediction="Dense in-range sweep beats load 1.47",
        prior=prior,
        proposed_by="strategizer",
    )


EVIDENCE = {"delegation": "D001", "numbers": {"best_y": 1.47}}


# ------------------------- propose -------------------------

def test_propose_creates_open_entry_with_schema(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h_id = propose_ok(ledger)
    entry = ledger.get(h_id)
    assert entry["statement"].startswith("Optimal t/L")
    assert entry["falsification_criterion"]
    assert entry["prediction"]
    assert entry["prior"] == 0.55
    assert entry["status_log"][-1]["status"] == "OPEN"


def test_propose_rejects_prior_zero_one_and_nonfloat(tmp_path):
    ledger = fresh_ledger(tmp_path)
    assert propose_ok(ledger, prior=0.0).startswith("ERROR:")
    assert propose_ok(ledger, prior=1.0).startswith("ERROR:")
    assert propose_ok(ledger, prior="high").startswith("ERROR:")
    assert propose_ok(ledger, prior=1.3).startswith("ERROR:")


def test_propose_rejects_empty_but_nudges_oversized(tmp_path):
    """Empty is a hard error (breaks the schema); over-length is now a NUDGE —
    the hypothesis is created with advice appended, never refused."""
    ledger = fresh_ledger(tmp_path)
    r = ledger.propose(statement="", falsification_criterion="c",
                       prediction="p", prior=0.5,
                       proposed_by="strategizer")
    assert r.startswith("ERROR:")
    r = ledger.propose(statement="x" * 501, falsification_criterion="c",
                       prediction="p", prior=0.5,
                       proposed_by="strategizer")
    assert r.startswith("H")          # created, not refused
    assert "[NUDGE]" in r             # with the tighten-it advice


def test_propose_nudges_compound_statement(tmp_path):
    """A compound statement is now a NUDGE, not a rejection: the hypothesis is
    registered and advised to split — a blocked propose only burns a turn."""
    ledger = fresh_ledger(tmp_path)
    r = propose_ok(ledger, statement=(
        "Optimum is near t/L=0.08 and the surrogate search will "
        "find a design exceeding 95 kPa"
    ))
    assert r.startswith("H")
    assert "[NUDGE]" in r
    assert "split" in r.lower()


def test_propose_allows_borderline_non_compound(tmp_path):
    # ' and ' without quantities on both sides must pass.
    ledger = fresh_ledger(tmp_path)
    r = propose_ok(ledger, statement=(
        "The landscape is rugged and multimodal near the origin "
        "region below radius 2.0"
    ))
    assert r == "H1"


def test_propose_nudges_numbered_subclaims(tmp_path):
    ledger = fresh_ledger(tmp_path)
    r = propose_ok(ledger, statement=(
        "(1) the optimum is at high ratio_d (2) the search will "
        "exceed 95 kPa"
    ))
    assert r.startswith("H")
    assert "[NUDGE]" in r


def test_propose_max_open_is_proceed_with_tip_not_block(tmp_path):
    """Past the open ceiling, a new proposal is REGISTERED with a tip — not
    refused, not two-shot. Opening a hypothesis is fully reversible (close it),
    and exploring several designs at once legitimately wants more than MAX_OPEN
    open. The tip advises closing settled ones."""
    ledger = fresh_ledger(tmp_path)
    for i in range(3):
        assert propose_ok(
            ledger, statement=f"Claim {i} below threshold 1.{i}"
        ).startswith("H")
    # Over-cap → created (starts with "H") with a [NUDGE] tip; never refused.
    r = propose_ok(ledger, statement="Fourth claim below 9.9")
    assert r.startswith("H")
    assert "[NUDGE]" in r and not r.startswith("ERROR:")
    assert ledger.get(r.splitlines()[0])["status_log"][-1]["status"] == "OPEN"


# ------------------------- update -------------------------

def test_update_closing_requires_evidence_delegation(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h = propose_ok(ledger)
    r = ledger.update(h, "SUPPORTED", "looks good",
                      evidence=None, posterior=0.9, triggered_by="D001")
    assert r.startswith("ERROR:")
    r = ledger.update(h, "SUPPORTED", "looks good",
                      evidence={"numbers": {}}, posterior=0.9,
                      triggered_by="D001")
    assert r.startswith("ERROR:")
    r = ledger.update(h, "SUPPORTED", "looks good",
                      evidence=EVIDENCE, posterior=0.9,
                      triggered_by="D001")
    assert not r.startswith("ERROR:")


def test_update_requires_posterior_in_bounds(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h = propose_ok(ledger)
    for bad in (None, "high", -0.1, 1.1):
        r = ledger.update(h, "SUPPORTED", "c", evidence=EVIDENCE,
                          posterior=bad, triggered_by="D001")
        assert r.startswith("ERROR:"), bad


def test_update_identical_resubmit_is_settled_not_loop(tmp_path):
    """A LITERALLY identical re-submit (same status, delegation, numbers,
    posterior) is 'settled — move on', not a retryable ERROR. This is what
    breaks the no-op loop the wet runs hit."""
    ledger = fresh_ledger(tmp_path)
    h = propose_ok(ledger)
    ledger.update(h, "SUPPORTED", "first", evidence=EVIDENCE,
                  posterior=0.8, triggered_by="D001")
    r = ledger.update(h, "SUPPORTED", "again", evidence=EVIDENCE,
                      posterior=0.8, triggered_by="D001")
    assert "SETTLED" in r and not r.startswith("ERROR:")


def test_update_allows_corrected_numbers_on_settled_hypothesis(tmp_path):
    """Correcting the cited numbers IS a real update (resolves the
    NUMBERS_MATCH vs no-op deadlock) — must be allowed, not refused."""
    ledger = fresh_ledger(tmp_path)
    h = propose_ok(ledger)
    ledger.update(h, "SUPPORTED", "first", evidence=EVIDENCE,
                  posterior=0.8, triggered_by="D001")
    r = ledger.update(
        h, "SUPPORTED", "corrected number",
        evidence={"delegation": "D001", "numbers": {"best_y": 1.62}},
        posterior=0.8, triggered_by="D001")
    assert "SETTLED" not in r and not r.startswith("ERROR:")


def test_update_allows_posterior_move_and_new_delegation(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h = propose_ok(ledger)
    ledger.update(h, "SUPPORTED", "first", evidence=EVIDENCE,
                  posterior=0.8, triggered_by="D001")
    # belief moved with same evidence → real update, allowed
    r = ledger.update(h, "SUPPORTED", "more confident", evidence=EVIDENCE,
                      posterior=0.9, triggered_by="D001")
    assert "SETTLED" not in r and not r.startswith("ERROR:")
    # new delegation → allowed
    r = ledger.update(h, "SUPPORTED", "more evidence",
                      evidence={"delegation": "D002"},
                      posterior=0.92, triggered_by="D002")
    assert not r.startswith("ERROR:")


def test_update_reopen_requires_evidence(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h = propose_ok(ledger)
    ledger.update(h, "FALSIFIED", "dead", evidence=EVIDENCE,
                  posterior=0.05, triggered_by="D001")
    r = ledger.update(h, "OPEN", "second thoughts", evidence=None,
                      posterior=0.4, triggered_by=None)
    assert r.startswith("ERROR:")
    r = ledger.update(h, "OPEN", "new data contradicts",
                      evidence={"delegation": "D003"},
                      posterior=0.4, triggered_by="D003")
    assert not r.startswith("ERROR:")


def test_update_rejects_unknown_id_and_status(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h = propose_ok(ledger)
    assert ledger.update("H99", "SUPPORTED", "c", evidence=EVIDENCE,
                         posterior=0.9,
                         triggered_by=None).startswith("ERROR:")
    assert ledger.update(h, "MAYBE", "c", evidence=EVIDENCE,
                         posterior=0.9,
                         triggered_by=None).startswith("ERROR:")


# ------------------------- list / belief -------------------------

def test_list_all_includes_belief_trajectory(tmp_path):
    ledger = fresh_ledger(tmp_path)
    h = propose_ok(ledger)
    items = ledger.list_all()
    assert items[0]["prior"] == 0.55
    assert items[0]["belief"] == 0.55          # no update yet → prior
    ledger.update(h, "SUPPORTED", "c", evidence=EVIDENCE,
                  posterior=0.9, triggered_by="D001")
    items = ledger.list_all()
    assert items[0]["belief"] == 0.9
    assert items[0]["current_status"] == "SUPPORTED"


# ------------------------- strict from_dict -------------------------

def test_from_dict_raises_on_missing_fields():
    with pytest.raises((KeyError, ValueError, TypeError)):
        HypothesisEntry.from_dict({
            "id": "H1", "statement": "s",
            "proposed_by": "x", "proposed_at": "t",
        })


# ------------------------- schema validation at load boundary -----

def test_load_raises_on_old_schema_file(tmp_path):
    (tmp_path / "hypotheses.json").write_text(json.dumps({
        "H1": {"id": "H1", "statement": "s",
               "proposed_by": "x", "proposed_at": "t",
               "status_log": []}
    }))
    ledger = HypothesisLedger(tmp_path)
    with pytest.raises((KeyError, ValueError, TypeError)):
        ledger.list_all()


# ------------------------- thread safety -------------------------

def test_concurrent_propose_is_thread_safe(tmp_path):
    """Concurrent proposes don't corrupt the ledger: every distinct claim is
    registered exactly once with a unique id (the open ceiling is a tip, not a
    cap, so all register; thread-safety is about no lost/duplicated ids)."""
    ledger = fresh_ledger(tmp_path)
    results = []
    lock = threading.Lock()

    def propose(i):
        r = ledger.propose(
            statement=f"Claim {i} below threshold 1.{i}",
            falsification_criterion="Any design exceeding the bound",
            prediction="Dense sweep finds better",
            prior=0.5,
            proposed_by="strategizer",
        )
        with lock:
            results.append(r)

    threads = [
        threading.Thread(target=propose, args=(i,)) for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All 6 distinct claims register (the ceiling is a tip); ids are unique and
    # contiguous — no lost or duplicated registrations under concurrency.
    created = [r.splitlines()[0] for r in results if r.startswith("H")]
    assert len(created) == 6
    assert len(set(created)) == 6
    assert len(ledger.list_all()) == 6
