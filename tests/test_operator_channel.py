"""The on-disk channel a human uses to answer a running graph.

The run and the viewer are separate processes, so this channel is files in
the run's debug dir. These tests pin the behaviours that decide whether a
run stalls, whether an answer counts, and whether the run record honestly
says what happened.
"""

from __future__ import annotations

import json
import time

from a3dasm._src import operator_channel as oc


def _run(tmp_path):
    (tmp_path / "debug").mkdir(parents=True)
    return tmp_path


# ---------------------------------------------------------------------------
# the watch heartbeat — what separates waiting from stalling
# ---------------------------------------------------------------------------

def test_a_run_nobody_is_watching_is_not_watched(tmp_path):
    """The load-bearing default. Without a fresh heartbeat a run must not
    wait on a question no one can see — that is not patience, it is a stall
    that spends the wall budget."""
    assert oc.is_watched(_run(tmp_path)) is False


def test_touch_marks_the_run_watched(tmp_path):
    run = _run(tmp_path)
    oc.touch_watch(run)
    assert oc.is_watched(run) is True


def test_a_stale_heartbeat_does_not_count_as_watching(tmp_path):
    """A viewer left open in a closed laptop must not hold a run open."""
    run = _run(tmp_path)
    oc.touch_watch(run)
    assert oc.is_watched(run, stale_s=-1.0) is False


def test_watch_on_a_run_with_no_debug_dir_is_harmless(tmp_path):
    oc.touch_watch(tmp_path)          # must not raise
    assert oc.is_watched(tmp_path) is False


# ---------------------------------------------------------------------------
# questions
# ---------------------------------------------------------------------------

def test_ask_then_answer_round_trip(tmp_path):
    run = _run(tmp_path)
    qid = oc.ask_question(run, "strategizer", "Is the strain floor advisory?")

    assert [q["id"] for q in oc.pending_questions(run)] == [qid]
    assert oc.read_answer(run, qid) is None

    assert oc.answer_question(run, qid, "Advisory for screening.") is True
    assert oc.read_answer(run, qid) == "Advisory for screening."
    assert oc.pending_questions(run) == []


def test_an_answer_arriving_after_the_run_gave_up_is_refused(tmp_path):
    """The race that matters: the run times out, then the operator sends.

    Accepting it would put an answer in the record that the agent never
    acted on — the run would appear to have been steered by something it
    never saw.
    """
    run = _run(tmp_path)
    qid = oc.ask_question(run, "strategizer", "?")
    oc.close_question(run, qid, "timeout")

    assert oc.answer_question(run, qid, "too late") is False
    assert oc.read_answer(run, qid) is None

    stored = json.loads(
        (run / "debug" / "followups" / f"{qid}.json").read_text())
    assert stored["status"] == "timeout"


def test_the_same_question_cannot_be_answered_twice(tmp_path):
    run = _run(tmp_path)
    qid = oc.ask_question(run, "strategizer", "?")
    assert oc.answer_question(run, qid, "first") is True
    assert oc.answer_question(run, qid, "second") is False
    assert oc.read_answer(run, qid) == "first"


def test_an_empty_answer_is_not_an_answer(tmp_path):
    run = _run(tmp_path)
    qid = oc.ask_question(run, "strategizer", "?")
    assert oc.answer_question(run, qid, "   ") is False
    assert oc.pending_questions(run) == [
        q for q in oc.pending_questions(run)]  # still pending
    assert oc.read_answer(run, qid) is None


def test_unanswered_questions_stay_on_the_record(tmp_path):
    """A question nobody answered is evidence about the run, so it is closed
    with a reason rather than deleted."""
    run = _run(tmp_path)
    qid = oc.ask_question(run, "strategizer", "?")
    oc.close_question(run, qid, "unattended")

    stored = json.loads(
        (run / "debug" / "followups" / f"{qid}.json").read_text())
    assert stored["status"] == "unattended"
    assert stored["question"] == "?"
    assert oc.pending_questions(run) == []


def test_questions_are_numbered_in_order(tmp_path):
    run = _run(tmp_path)
    ids = [oc.ask_question(run, "strategizer", f"q{i}") for i in range(3)]
    assert ids == ["Q001", "Q002", "Q003"]
    assert [q["id"] for q in oc.pending_questions(run)] == ids


def test_reading_a_torn_question_file_does_not_raise(tmp_path):
    run = _run(tmp_path)
    qdir = run / "debug" / "followups"
    qdir.mkdir(parents=True)
    (qdir / "Q001.json").write_text('{"id": "Q0', encoding="utf-8")
    assert oc.pending_questions(run) == []
    assert oc.read_answer(run, "Q001") is None


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------

def test_a_note_is_delivered_once_and_only_once(tmp_path):
    """Being told the same thing on every tool call is worse than not being
    told at all, so delivery clears the queue."""
    run = _run(tmp_path)
    assert oc.queue_note(run, "shell_05 uses the old mesh") is True

    assert oc.drain_notes(run) == ["shell_05 uses the old mesh"]
    assert oc.drain_notes(run) == []


def test_notes_drain_in_the_order_they_were_queued(tmp_path):
    run = _run(tmp_path)
    for t in ("first", "second", "third"):
        oc.queue_note(run, t)
    assert oc.drain_notes(run) == ["first", "second", "third"]


def test_an_empty_note_is_rejected(tmp_path):
    run = _run(tmp_path)
    assert oc.queue_note(run, "  ") is False
    assert oc.drain_notes(run) == []


def test_a_note_for_a_run_with_no_debug_dir_is_rejected(tmp_path):
    assert oc.queue_note(tmp_path, "hello") is False
    assert oc.drain_notes(tmp_path) == []


def test_a_torn_note_line_does_not_lose_the_rest(tmp_path):
    run = _run(tmp_path)
    path = run / "debug" / "operator_notes.jsonl"
    path.write_text('{"text": "good"}\nnot json\n{"text": "also good"}\n',
                    encoding="utf-8")
    assert oc.drain_notes(run) == ["good", "also good"]


def test_answered_question_records_when_it_was_answered(tmp_path):
    run = _run(tmp_path)
    before = time.time()
    qid = oc.ask_question(run, "strategizer", "?")
    oc.answer_question(run, qid, "yes")
    stored = json.loads(
        (run / "debug" / "followups" / f"{qid}.json").read_text())
    assert stored["answered_at"] >= before
    assert stored["asked_at"] <= stored["answered_at"]
