"""run_ledger.analysis_brief: mechanical post-run digest.

Asserts it (a) offloads the MECHANICAL facts (KPI row, diagnostics tally,
verbatim ERROR_RETURN) and (b) RELOCATES the prose (retrospectives verbatim)
rather than classifying it — the line we must not cross is turning a
judgement (reading prose for varied failure-mode wording) into a proxy grep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_STUDIES = Path(__file__).resolve().parents[1] / "studies"
if str(_STUDIES) not in sys.path:
    sys.path.insert(0, str(_STUDIES))

import run_ledger  # noqa: E402


def _make_run(tmp_path: Path) -> Path:
    # studies/<study>/runs/<id>/debug/...
    run_dir = tmp_path / "studyX" / "runs" / "20260101T000000"
    debug = run_dir / "debug"
    debug.mkdir(parents=True)
    (debug / "run_status.json").write_text(json.dumps({"status": "GATED"}))
    (debug / "diagnostics.jsonl").write_text(
        json.dumps({"type": "ERROR_RETURN", "source_id": "D002",
                    "detail": "tool raised"}) + "\n"
        + json.dumps({"type": "RAW_ORACLE_NUDGE", "source_id": "D003"}) + "\n"
    )
    (debug / "retrospectives.jsonl").write_text(
        json.dumps({"source_id": "D001", "role": "implementer",
                    "flagged": True,
                    "text": "FRICTION: the catalog named a tool I could not "
                            "call. BLOCKED: none."}) + "\n"
    )
    cr = debug / "critic_reviews"
    cr.mkdir()
    (cr / "call_001.md").write_text("### Verdict\n**REJECT**\nverdict: REJECT")
    (cr / "call_002.md").write_text("### Verdict\n**PASS**\nverdict: PASS")
    # headline store: an index column ('') + objective 'f' + provenance
    store = run_dir / "experiment_data" / "experiment_data"
    store.mkdir(parents=True)
    (store / "output.csv").write_text(
        ",f,_delegation_id\n0,-0.71,D006\n1,-0.20,D005\n")
    # hypothesis ledger: H1 closed, H2 left OPEN at close
    notes = debug / "strategizer_notes"
    notes.mkdir()
    (notes / "hypotheses.json").write_text(json.dumps({
        "H1": {"id": "H1", "status_log": [{"status": "OPEN"},
                                          {"status": "FALSIFIED"}]},
        "H2": {"id": "H2", "status_log": [{"status": "OPEN"}]},
    }))
    # per-delegation transcripts: D004 error-heavy + a stall, D005 clean
    tdir = debug / "transcripts"
    tdir.mkdir()
    (tdir / "D004.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"type": "assistant", "gap_s": 2.0},
        {"type": "tool_result", "gap_s": 115.2, "results": [
            {"tool_use_id": "t1",
             "content": "<tool_use_error>Error: No such tool available: "
                        "arxiv_search_papers</tool_use_error>"}]},
        {"type": "tool_result", "gap_s": 1.0, "results": [
            {"tool_use_id": "t2",
             "content": "<tool_use_error>Error: No such tool available: "
                        "search_openalex</tool_use_error>"}]},
    ]))
    (tdir / "D005.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"type": "assistant", "gap_s": 3.0},
        {"type": "tool_result", "gap_s": 4.0, "results": [
            {"tool_use_id": "t3", "content": "185 evaluations OK"}]},
    ]))
    return run_dir


def test_brief_offloads_mechanical_kpis_and_diagnostics(tmp_path):
    brief = run_ledger.analysis_brief(_make_run(tmp_path))
    # Step-5 KPI baseline + Step-2 tally are computed for us
    assert "## KPIs — this run vs previous (Step 5)" in brief
    assert "outcome:" in brief
    assert "critic_consults: 2" in brief          # two critic calls = gate attempts
    assert "## ScienceMonitor diagnostics tally (Step 2)" in brief
    assert "ERROR_RETURN: 1" in brief and "RAW_ORACLE_NUDGE: 1" in brief
    # ERROR_RETURN (KPI target 0) surfaced verbatim, structured — not a prose grep
    assert "ERROR_RETURN events verbatim (1; target 0)" in brief
    assert "tool raised" in brief


def test_brief_surfaces_science_result_and_gate_sequence(tmp_path):
    """The catchable false negatives (probed on a real run): the headline
    objective, hypotheses left OPEN at close, and the critic verdict sequence —
    all structured 'where to look', none classifying prose."""
    brief = run_ledger.analysis_brief(_make_run(tmp_path))
    # headline: the objective column 'f', not the empty index column; min/max
    # reported without asserting which end is 'good'
    assert "objective 'f': min=-0.71" in brief
    assert "yours to judge" in brief
    # hypothesis ledger: H1 closed, H2 flagged as still open at close
    assert "H1: FALSIFIED" in brief
    assert "H2: OPEN" in brief and "still open at close" in brief
    # critic gate sequence shows the bounce, not just the count
    assert "call_001=REJECT, call_002=PASS" in brief


def test_brief_ranks_delegation_transcript_proxies(tmp_path):
    """Transcripts proxy POSSIBLE failure modes: the error-heavy delegation is
    ranked first with its tool-error count, stall, and a verbatim example —
    'look here', not a classification of what failed."""
    brief = run_ledger.analysis_brief(_make_run(tmp_path))
    assert "## Delegation transcript proxies" in brief
    di = brief.index("## Delegation transcript proxies")
    panel = brief[di:]
    # D004 (2 tool-errors, 115s stall) ranks ABOVE D005 (0 errors)
    assert panel.index("D004:") < panel.index("D005:")
    assert "D004: 2 tool-errors, max stall 115.2s" in panel
    assert "D005: 0 tool-errors" in panel
    # the structured error text is surfaced verbatim (what tripped, not why)
    assert "No such tool available: arxiv_search_papers" in panel
    # still a pointer, not a verdict
    assert "not verdicts" in panel
    assert "is a bug" not in panel.lower()


def test_brief_relocates_prose_but_does_not_classify(tmp_path):
    brief = run_ledger.analysis_brief(_make_run(tmp_path))
    # prose artifacts are pointed to with counts...
    assert "Step 1 retrospectives: 1 entries" in brief
    assert "Step 3 critic_reviews: 2 calls" in brief
    assert "read FIRST" in brief and "judge, don't grep" in brief
    # ...and the retrospective is surfaced VERBATIM (relocation, not a verdict):
    # the brief must contain the exact prose, leaving the judgement to the reader.
    assert ("FRICTION: the catalog named a tool I could not call. BLOCKED: none."
            in brief)
    # and it must NOT invent a classification the agent never wrote
    assert "is a bug" not in brief.lower()
    assert "root cause" not in brief.lower()
