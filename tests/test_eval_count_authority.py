"""Friction #3 fix: the ledger is the authoritative evaluation count, and the
agent is told to cite it rather than hand-compute (which produced the stale
'3710 vs 4313' headline errors that triggered critic REVISEs)."""
from __future__ import annotations


def test_recallstore_summary_labels_count_authoritative():
    from a3dasm._src.instrumented import RunStateSummary

    s = RunStateSummary(
        n_rows=4313, n_per_delegation={"D001": 4313}, n_per_source={},
        n_per_fidelity=None, output_stats={},
    )
    out = s.format()
    assert "4313 total ledgered evaluations" in out
    assert "AUTHORITATIVE" in out
    assert "never hand-compute" in out.lower()


def test_strategizer_prompt_directs_eval_count_to_ledger():
    from a3dasm._src.agents.strategizer import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    p = STRATEGIZER_SYSTEM_PROMPT
    assert "TOTAL EVALUATION COUNT" in p
    assert "RecallStore" in p
    # it must forbid hand-computing / worker-self-reported counts
    assert "never" in p.lower() and "self-reported" in p.lower()
