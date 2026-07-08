"""Run-2 back door: the agent could not RETRACT its own premature verdict.

The strategizer marked H1/H3 SUPPORTED, then correctly realised (Charter §2) they
should be OPEN — no falsification ATTEMPT was made. HypothesisUpdate refused the
downgrade ("reopening requires new evidence with a 'delegation' key"), but at
close time there is no new delegation to cite → deadlock → a narrative-vs-ledger
contradiction the critic rejects on (run-2 UNGATED, CONSISTENCY_FLAG).

Fix: retracting a verdict to OPEN is WITHDRAWING a claim, not asserting one — it
carries forward the evidence the verdict was based on, no new delegation needed.
"""
from __future__ import annotations

from a3dasm._src.hypothesis_ledger import HypothesisLedger


def _ledger(tmp_path):
    led = HypothesisLedger(tmp_path)
    hid = led.propose(
        statement="coverage-first beats BO",
        falsification_criterion="a BO run beats the sweep best",
        prediction="sweep best <= BO best",
        prior=0.5,
        proposed_by="strategizer",
    )
    return led, hid


def test_retract_to_open_without_new_evidence_is_allowed(tmp_path):
    led, hid = _ledger(tmp_path)
    # close it (cites D004), then realise it was premature and retract to OPEN
    led.update(hid, "SUPPORTED", "positive evidence",
               evidence={"delegation": "D004", "numbers": {"best": -0.79}},
               posterior=0.8, triggered_by="D004")
    res = led.update(
        hid, "OPEN",
        "no falsification ATTEMPT was made; retracting per Charter §2",
        evidence=None, posterior=0.5, triggered_by=None)
    assert not res.startswith("ERROR"), res
    last = led.get(hid)["status_log"][-1]
    assert last["status"] == "OPEN"
    # the evidence the verdict was based on is carried forward (audit trail)
    assert (last["evidence"] or {}).get("delegation") == "D004"


def test_closing_status_still_requires_evidence(tmp_path):
    """The fix must NOT loosen CLOSING: asserting SUPPORTED/FALSIFIED still
    requires a delegation citation (no rubber-stamping)."""
    led, hid = _ledger(tmp_path)
    res = led.update(hid, "SUPPORTED", "claim", evidence=None,
                     posterior=0.8, triggered_by=None)
    assert res.startswith("ERROR") and "delegation" in res
