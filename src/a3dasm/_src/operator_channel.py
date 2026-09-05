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
import time
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

# A heartbeat older than this means nobody is looking. Comfortably longer
# than the viewer's own refresh interval so a slow poll never reads as an
# absent operator.
WATCH_STALE_S = 45.0


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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------

def ask_question(run_dir: Path | str, node: str, question: str) -> str:
    """Record a pending question and return its id."""
    qdir = _q_dir(run_dir)
    qdir.mkdir(parents=True, exist_ok=True)
    seq = len(list(qdir.glob("*.json"))) + 1
    qid = f"Q{seq:03d}"
    _write_json(qdir / f"{qid}.json", {
        "id": qid,
        "node": node,
        "question": question,
        "asked_at": time.time(),
        "status": "pending",
        "answer": None,
        "answered_at": None,
    })
    return qid


def pending_questions(run_dir: Path | str) -> list[dict[str, Any]]:
    """Every question still awaiting an answer, oldest first."""
    qdir = _q_dir(run_dir)
    if not qdir.is_dir():
        return []
    out = []
    for path in sorted(qdir.glob("*.json")):
        data = _read_json(path)
        if data and data.get("status") == "pending":
            out.append(data)
    return out


def read_answer(run_dir: Path | str, qid: str) -> str | None:
    """The answer to *qid*, or ``None`` while it is still unanswered."""
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
    path = _q_dir(run_dir) / f"{qid}.json"
    data = _read_json(path)
    if not data or data.get("status") != "pending":
        return False
    if not answer.strip():
        return False
    data["answer"] = answer
    data["status"] = "answered"
    data["answered_at"] = time.time()
    _write_json(path, data)
    return True


def close_question(run_dir: Path | str, qid: str, status: str) -> None:
    """Close *qid* without an answer (``timeout`` or ``unattended``).

    Recorded rather than deleted: a question nobody answered, and how long
    the run waited before proceeding, is exactly the sort of thing a later
    reader of the run needs in order to judge the work.
    """
    path = _q_dir(run_dir) / f"{qid}.json"
    data = _read_json(path)
    if not data or data.get("status") != "pending":
        return
    data["status"] = status
    data["answered_at"] = time.time()
    _write_json(path, data)


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

    Delivery is recorded by truncating the file rather than by a per-row
    flag, so a note cannot be handed to the agent twice — being told the
    same thing repeatedly is worse than not being told at all.
    """
    path = _dir(run_dir) / _NOTES
    if not path.is_file():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
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
    try:
        path.write_text("", encoding="utf-8")
    except OSError:
        pass
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
