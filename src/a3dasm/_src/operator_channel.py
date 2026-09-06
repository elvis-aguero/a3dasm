"""The on-disk channel between a running graph and a human operator.

The run and the viewer are separate processes — the viewer is normally
started independently of the run it is watching, and may not exist at all —
so the channel between them is the run's own ``debug/`` directory rather
than a socket or a shared object. Everything here is small JSON on disk,
which also means the whole exchange is auditable after the fact: what was
asked, what was answered, and what went unanswered are all part of the run
record instead of scrolling past in a terminal.

Three things move across it:

* **questions** — ``FollowUp`` asks; the operator answers, from a terminal
  or from the viewer, whichever gets there first;
* **notes** — the operator queues a message; the entry node picks it up on
  its next tool call;
* **a watch heartbeat** — the viewer says a human is actually looking,
  which is what lets a run decide whether waiting for an answer is
  reasonable or merely a stall.

Nothing here raises on a missing directory or a malformed file: a run whose
debug dir was never created must behave exactly as one nobody is watching.
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = [
    "ask_question",
    "pending_questions",
    "answer_question",
    "read_answer",
    "close_question",
    "queue_note",
    "drain_notes",
    "touch_watch",
    "is_watched",
    "WATCH_STALE_S",
]

_QUESTIONS = "followups"
_NOTES = "operator_notes.jsonl"
_WATCH = "viewer_watching"

# Question ids are used to build a path, and the id arrives from an
# HTTP request body, so it is matched rather than trusted.
_QID_RE = re.compile(r"^Q\d{1,9}$")


class _ChannelUnavailable(Exception):
    """The channel could not be written (disk full, read-only)."""

# A heartbeat older than this means nobody is looking. It has to clear
# browser background-tab throttling, not just the viewer's nominal poll
# interval: Chrome clamps timers in a hidden tab to >=60s (and to once a
# minute under intensive throttling). At 45s, an operator switching tabs
# to look something up read as absent, the run gave up, and their answer
# was then refused.
WATCH_STALE_S = 180.0


@contextmanager
def _locked(run_dir: Path | str):
    """Hold the channel's cross-process lock, or proceed without it.

    The run and the viewer race on the same files from different
    processes, so read-modify-write on a question needs more than an atomic
    rename: the rename makes each WRITE indivisible, it does not stop two
    writers from deciding what to write against the same stale read. The
    concrete loss was an answer POSTed in the same instant the run gave up
    being recorded as accepted, telling the operator their answer landed
    while the agent had already moved on.

    filelock is already a project dependency (literature_corpus uses it for
    the same cross-process reason). If the lock cannot be taken at all —
    read-only dir, an exotic filesystem — the body still runs: an
    unsynchronised write is worse than a lost question, but only slightly,
    and refusing to function is worse than both.
    """
    debug = Path(run_dir) / "debug"
    try:
        from filelock import FileLock, Timeout
        lock = FileLock(str(debug / ".operator.lock"), timeout=5)
    except Exception:  # noqa: BLE001 — filelock missing or unconstructable
        yield
        return
    try:
        with lock:
            yield
    except Timeout:
        # Five seconds means the holder is wedged, not busy; the callers
        # here all tolerate a lost update better than a stalled run.
        yield
    except OSError:
        yield


def _dir(run_dir: Path | str) -> Path:
    return Path(run_dir) / "debug"


def _q_dir(run_dir: Path | str) -> Path:
    return _dir(run_dir) / _QUESTIONS


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write atomically.

    The reader is a different process polling in a loop, so a torn file is
    not a theoretical concern — it is what a plain write produces every time
    the poll lands mid-write. Rename on the same filesystem is atomic, so a
    reader sees either the old file or the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # The temp name must be unique PER WRITER. Two processes sharing one
    # "<name>.tmp" can interleave their writes into it and then rename
    # the mixture into place — which defeats the very guarantee this
    # function exists to provide.
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        # A full or read-only disk must not turn a clarifying question
        # into an exception escaping the tool.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise _ChannelUnavailable from None


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------

def ask_question(run_dir: Path | str, node: str, question: str) -> str | None:
    """Record a pending question and return its id, or ``None`` if it could
    not be recorded.

    The id is claimed by exclusive create rather than by counting existing
    files. Counting is a read-then-write race: two questions asked at once
    both compute the same next number and the second overwrites the first,
    so the first asker's read_answer() would return the answer to somebody
    else's question — worse than returning nothing at all.
    """
    qdir = _q_dir(run_dir)
    try:
        qdir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    seq = len(list(qdir.glob("*.json")))
    qid = None
    for _ in range(200):
        seq += 1
        candidate = qdir / f"Q{seq:03d}.json"
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        except OSError:
            return None
        os.close(fd)
        qid = f"Q{seq:03d}"
        break
    if qid is None:
        return None
    try:
        _write_json(qdir / f"{qid}.json", {
        "id": qid,
        "node": node,
        "question": question,
        "asked_at": time.time(),
        "status": "pending",
        "answer": None,
        "answered_at": None,
        })
    except _ChannelUnavailable:
        return None
    return qid


def pending_questions(run_dir: Path | str) -> list[dict[str, Any]]:
    """Every question still awaiting an answer, oldest first."""
    qdir = _q_dir(run_dir)
    if not qdir.is_dir():
        return []
    out = []
    for path in qdir.glob("*.json"):
        data = _read_json(path)
        if data and data.get("status") == "pending":
            out.append(data)
    # By ask time, not filename: zero-padding stops sorting correctly once
    # the sequence passes its width, and the caller wants oldest-first.
    out.sort(key=lambda d: d.get("asked_at") or 0)
    return out


def read_answer(run_dir: Path | str, qid: str) -> str | None:
    """The answer to *qid*, or ``None`` while it is still unanswered."""
    if not _QID_RE.match(qid or ""):
        return None
    data = _read_json(_q_dir(run_dir) / f"{qid}.json")
    if not data or data.get("status") != "answered":
        return None
    answer = data.get("answer")
    return answer if isinstance(answer, str) and answer.strip() else None


def answer_question(run_dir: Path | str, qid: str, answer: str) -> bool:
    """Answer *qid*. False if it is unknown or already resolved.

    Refusing to overwrite a resolved question is deliberate: an answer that
    arrives after the run gave up must not look like one the agent acted
    on, or the record would claim an influence the run never had.
    """
    if not _QID_RE.match(qid or "") or not answer.strip():
        return False
    path = _q_dir(run_dir) / f"{qid}.json"
    with _locked(run_dir):
        # Re-read INSIDE the lock: the status may have changed to timeout
        # between the caller deciding to answer and this write.
        data = _read_json(path)
        if not data or data.get("status") != "pending":
            return False
        data["answer"] = answer
        data["status"] = "answered"
        data["answered_at"] = time.time()
        try:
            _write_json(path, data)
        except _ChannelUnavailable:
            return False
    return True


def close_question(run_dir: Path | str, qid: str, status: str) -> None:
    """Close *qid* without an answer (``timeout`` or ``unattended``).

    Recorded rather than deleted: a question nobody answered, and how long
    the run waited before proceeding, is exactly the sort of thing a later
    reader of the run needs in order to judge the work.
    """
    if not _QID_RE.match(qid or ""):
        return
    path = _q_dir(run_dir) / f"{qid}.json"
    with _locked(run_dir):
        # Same lock as answer_question, so a timeout and an answer landing
        # together resolve one way or the other rather than both "winning".
        data = _read_json(path)
        if not data or data.get("status") != "pending":
            return
        data["status"] = status
        data["answered_at"] = time.time()
        try:
            _write_json(path, data)
        except _ChannelUnavailable:
            pass


# --------------------------------------------------------------------------
# notes
# --------------------------------------------------------------------------

def queue_note(run_dir: Path | str, text: str, to_node: str = "") -> bool:
    """Queue an operator note for delivery at the node's next tool call."""
    if not text.strip():
        return False
    debug = _dir(run_dir)
    if not debug.is_dir():
        return False
    row = {"ts": time.time(), "to_node": to_node, "text": text}
    with (debug / _NOTES).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return True


def drain_notes(run_dir: Path | str) -> list[str]:
    """Return undelivered note texts and mark them delivered.

    Delivery is recorded by CLAIMING the file — renaming it aside and
    reading the claimed copy — not by reading and then truncating. Reading
    first loses any note appended in between: the operator's POST returned
    {"ok": true} for a note that was then deleted unread. Rename is the
    atomic step, so a note either makes it into this batch or stays in a
    fresh file for the next one.

    A note is never handed over twice, because the file it came from no
    longer exists by the time it is returned — being told the same thing on
    every tool call is worse than not being told at all.
    """
    path = _dir(run_dir) / _NOTES
    if not path.is_file():
        return []
    claimed = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex[:8]}.claim")
    try:
        path.replace(claimed)
    except OSError:
        return []
    try:
        # errors="replace": a note appended concurrently can split a
        # multibyte character, and this must not raise into a tool call.
        raw = claimed.read_text(encoding="utf-8", errors="replace")
    except OSError:
        raw = ""
    finally:
        try:
            claimed.unlink()
        except OSError:
            pass
    if not raw.strip():
        return []
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            out.append(text)
    return out


# --------------------------------------------------------------------------
# watch heartbeat
# --------------------------------------------------------------------------

def touch_watch(run_dir: Path | str) -> None:
    """Record that a human currently has this run open."""
    debug = _dir(run_dir)
    if not debug.is_dir():
        return
    try:
        (debug / _WATCH).write_text(str(time.time()), encoding="utf-8")
    except OSError:
        pass


def is_watched(run_dir: Path | str, stale_s: float = WATCH_STALE_S) -> bool:
    """Whether someone is watching this run right now.

    This is what separates "wait for an answer" from "stall". A run with no
    terminal and no viewer must not pause on a question nobody can see, so
    the absence of a fresh heartbeat restores the old behaviour exactly:
    ask, get no operator, carry on.
    """
    path = _dir(run_dir) / _WATCH
    try:
        return (time.time() - path.stat().st_mtime) <= stale_s
    except OSError:
        return False
