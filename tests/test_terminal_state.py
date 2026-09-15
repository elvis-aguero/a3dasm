"""The run's terminal state is recorded, not re-derived from its own prose.

Two bugs are pinned here, both of which logged a non-success as a validated
success because the outcome was recovered by grepping the final report's
banner and defaulting to GATED when nothing matched:

1. A backstop/repeated-errors halt. ``_halt_resumable`` wrote
   ``status="halted"``; the close path then overwrote it with the grep result,
   and the HALTED banner matches none of the patterns → GATED.
2. A critic-less graph. ``_close`` routes straight to done and emits no banner
   at all → GATED, i.e. removing the adversarial critic made every run a
   perfect success.
"""
from __future__ import annotations

import json

from langgraph.graph import END

from a3dasm._src.runtime import terminal

from .test_route_aware_termination import (
    StubAdapter,
    _make_state,
    _minimal_spec,
)


# --- the vocabulary itself -------------------------------------------------

def test_an_unrecorded_outcome_is_never_a_success():
    """The old default was GATED, which is exactly how a halt became a pass.

    A path that ended the run without declaring an outcome validated nothing.
    """
    outcome, termination, reviewed = terminal.resolve(None, None, None)
    assert outcome == terminal.UNGATED
    assert termination == terminal.NO_CLOSE
    assert reviewed is False


def test_a_halted_run_cannot_be_gated():
    """Nothing reviewed it — the run never reached its gate."""
    for term in terminal.HALT_TERMINATIONS:
        outcome, _, _ = terminal.resolve(terminal.GATED, term, True)
        assert outcome == terminal.UNGATED, term


def test_an_unreviewed_run_cannot_be_gated():
    """GATED means a critic gate passed it. No critic, no gate, no GATED."""
    outcome, _, _ = terminal.resolve(
        terminal.GATED, terminal.DONE, reviewed=False)
    assert outcome == terminal.UNGATED


def test_a_reviewed_deliberate_close_keeps_its_outcome():
    assert terminal.resolve(terminal.GATED, terminal.DONE, True) == (
        terminal.GATED, terminal.DONE, True)


def test_an_unknown_value_does_not_leak_through():
    outcome, term, _ = terminal.resolve("EXCELLENT", "vibes", True)
    assert outcome == terminal.UNGATED
    assert term == terminal.NO_CLOSE


# --- bug 1: a halt is not a success ---------------------------------------

def _halted_run_dir(tmp_path, monkeypatch):
    """Drive a node into the repeated-errors halt and return its run dir."""
    from a3dasm._src.nodes import Node

    monkeypatch.setenv("F3DASM_MAX_CONSECUTIVE_ERRORS", "3")
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "pipeline.ipynb").write_text("# test\n")
    run_dir = study_dir / "runs" / "T"
    (run_dir / "debug").mkdir(parents=True)
    (run_dir / "debug" / "thread_id").write_text("tid-halt")

    node = Node(
        StubAdapter(response="Should not be called."),
        name="strategizer", outgoing=["implementer"], spec=_minimal_spec(),
    )
    node._consecutive_errors["implementer"] = 3

    state = _make_state(study_dir=study_dir)
    state["run_dir"] = str(run_dir)
    return run_dir, node(state)


def test_a_halted_run_carries_its_termination_on_the_state(
        tmp_path, monkeypatch):
    """The reason it stopped travels as data, not only as banner prose.

    Without this the close path had nothing to read and fell back to the
    grep, which is what turned every backstop kill into a GATED row.
    """
    _, cmd = _halted_run_dir(tmp_path, monkeypatch)
    assert cmd.goto == END
    assert cmd.update["termination"] == terminal.REPEATED_ERRORS
    assert cmd.update["outcome"] == terminal.UNGATED
    assert cmd.update["reviewed"] is False


def test_a_halted_run_status_file_says_it_halted(tmp_path, monkeypatch):
    run_dir, _ = _halted_run_dir(tmp_path, monkeypatch)
    status = json.loads(
        (run_dir / "debug" / "run_status.json").read_text(encoding="utf-8"))
    assert status["status"] == "halted"
    assert status["termination"] == terminal.REPEATED_ERRORS
    assert status["outcome"] != terminal.GATED
    assert status["resumable"] is True


def test_the_close_path_does_not_relabel_a_halt_as_a_success(
        tmp_path, monkeypatch):
    """The regression itself: finalisation used to overwrite the halt status.

    Whatever the close path computes from the halt's own state must not be
    GATED — that is the row that used to read as a clean validated run.
    """
    _, cmd = _halted_run_dir(tmp_path, monkeypatch)
    outcome, _, _ = terminal.resolve(
        cmd.update.get("outcome"),
        cmd.update.get("termination"),
        cmd.update.get("reviewed"),
    )
    assert outcome != terminal.GATED


# --- bug 2: no critic, no gate --------------------------------------------

def test_a_critic_less_close_is_not_gated(tmp_path):
    """Closing a graph with no critic records UNGATED and reviewed=False.

    It read GATED before, so an ablation that removed the adversarial critic
    would have reported a 100% success rate by construction.
    """
    from a3dasm._src.nodes import Node
    from a3dasm._src.nodes.tools.routing.feedback import FeedbackTools

    study_dir = tmp_path / "study"
    (study_dir / "runs" / "T" / "debug").mkdir(parents=True)

    node = Node(
        StubAdapter(), name="strategizer", outgoing=["implementer"],
        spec=_minimal_spec(),
    )
    assert node._find_critic_name() is None, "fixture must have no critic"

    FeedbackTools(node)._close("the conclusion", prefix="")

    assert node._route["kind"] == "done"
    assert node._route["outcome"] == terminal.UNGATED
    assert node._route["termination"] == terminal.DONE
    assert node._route["reviewed"] is False
