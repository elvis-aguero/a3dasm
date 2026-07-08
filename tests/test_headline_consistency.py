"""The reproduction gate cross-checks the notebook's STATED headline against
its COMPUTED one (lenient: only when both markers are printed).

Regression for run 20260705T181941: the write-up stated 0.3644 while an idxmax
cell printed a 0.3648 noise row as REPRODUCED, and the gate passed it for 4
rounds because it never compared the two.
"""
from __future__ import annotations

from a3dasm._src.nodes.strategizer import _headline_consistency


def test_mismatch_is_flagged():
    out = "some logs\nCLAIMED_HEADLINE: 0.3644\nREPRODUCED: 0.3648\n"
    msg = _headline_consistency(out)
    assert msg is not None and "inconsistency" in msg.lower()


def test_agreement_within_tolerance_passes():
    assert _headline_consistency("CLAIMED_HEADLINE: 0.3644\nREPRODUCED: 0.36440001") is None


def test_lenient_when_claim_absent():
    # only REPRODUCED printed (the convention not adopted) -> no new failure
    assert _headline_consistency("REPRODUCED: 0.3648") is None


def test_lenient_when_both_absent():
    assert _headline_consistency("no markers here") is None
