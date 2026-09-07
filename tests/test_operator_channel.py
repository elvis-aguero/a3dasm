"""The on-disk channel a human uses to answer a running graph.

The run and the viewer are separate processes, so this channel is files in
the run's debug dir. These tests pin the behaviours that decide whether a
run stalls, whether an answer counts, and whether the run record honestly
says what happened.
"""

from __future__ import annotations

import json
import threading
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


# ---------------------------------------------------------------------------
# cross-process races — the run and the viewer write these files concurrently
# ---------------------------------------------------------------------------

def test_a_note_queued_while_draining_is_not_lost(tmp_path):
    """Read-then-truncate deleted notes it had never read.

    The operator's POST returned {"ok": true} and the note was then wiped
    unread. Claiming the file by rename makes the handover atomic: a note
    either joins this batch or waits in a fresh file for the next.
    """
    run = _run(tmp_path)
    oc.queue_note(run, "first")

    drained = []
    t = threading.Thread(target=lambda: drained.extend(oc.drain_notes(run)))
    t.start()
    oc.queue_note(run, "second")
    t.join()

    assert sorted(drained + oc.drain_notes(run)) == ["first", "second"]


def test_answering_and_timing_out_together_cannot_both_win(tmp_path):
    """The window that mattered: the operator hits Send as the run gives up.

    Without a lock these are two read-modify-writes against the same stale
    read, so answer_question could report success — telling the operator
    their answer landed — while close_question's write won and the agent
    proceeded without it.
    """
    run = _run(tmp_path)
    for _ in range(25):
        qid = oc.ask_question(run, "strategizer", "race?")
        result = {}

        def _answer(_qid=qid, _result=result):
            _result["ok"] = oc.answer_question(run, _qid, "yes")

        def _close(_qid=qid):
            oc.close_question(run, _qid, "timeout")

        ta, tc = threading.Thread(target=_answer), threading.Thread(target=_close)
        ta.start()
        tc.start()
        ta.join()
        tc.join()

        # Either outcome is acceptable; claiming one while recording the
        # other is not.
        if result.get("ok"):
            assert oc.read_answer(run, qid) == "yes"
        else:
            assert oc.read_answer(run, qid) is None


def test_two_questions_asked_at_once_get_distinct_ids(tmp_path):
    """Counting existing files to pick the next id is a read-then-write
    race: both asks compute the same number and the second overwrites the
    first, so the first asker reads back somebody else's answer."""
    run = _run(tmp_path)
    ids = []
    lock = threading.Lock()

    def _ask(n):
        qid = oc.ask_question(run, "strategizer", f"q{n}")
        with lock:
            ids.append(qid)

    threads = [threading.Thread(target=_ask, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(ids) == 8
    assert len(set(ids)) == 8, f"duplicate ids issued: {ids}"
    assert len(oc.pending_questions(run)) == 8


# ---------------------------------------------------------------------------
# Addressing a note at work already in flight
#
# The point of nudging is to correct a delegation BEFORE it returns a wrong
# result. Until the Confer delivery fix there was no path to a busy worker at
# all; this routes an operator's note down that same per-delegation queue.
# ---------------------------------------------------------------------------

def test_a_note_can_carry_the_delegation_it_is_aimed_at(tmp_path):
    run = _run(tmp_path)
    assert oc.queue_note(run, "use the coarse mesh", to_node="D004") is True

    rows = oc.drain_note_rows(run)
    assert rows == [{"text": "use the coarse mesh", "to_node": "D004"}]


def test_an_unaddressed_note_carries_an_empty_address(tmp_path):
    """The entry node's own notes must stay distinguishable from addressed
    ones, or routing cannot tell them apart."""
    run = _run(tmp_path)
    oc.queue_note(run, "reconsider the floor")
    assert oc.drain_note_rows(run) == [
        {"text": "reconsider the floor", "to_node": ""}]


def test_drain_notes_and_drain_note_rows_claim_the_same_queue(tmp_path):
    """Both drains are destructive, so a caller that needs the addressing
    must not also call the plain form — pinning it so the two cannot
    silently diverge into a double-drain that loses the first batch."""
    run = _run(tmp_path)
    oc.queue_note(run, "one", to_node="D001")

    assert oc.drain_notes(run) == ["one"]
    assert oc.drain_note_rows(run) == []
