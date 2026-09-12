"""A bare Wait() collects whichever delegation finishes first.

Fan-out was already cheap to DISPATCH — Delegate(wait=False) is used on 85% of
real Delegate calls — but expensive to COLLECT: Wait() bound to a single
delegation id, so the only way to harvest several in flight was GetStatus
polling, which the poll-count nudges actively discourage. Measured consequence
across 39 cluster runs: reliance on Wait predicted SERIAL execution
(wait-share-of-harvest vs mean concurrent delegations r=-0.54 controlling for
delegation duration), and mean concurrency sat at 1.21 against a median 15
delegations created per run.

These cover the bare form: it harvests each finished delegation exactly once,
blocks only while nothing is ready, and refuses rather than hanging when
waiting cannot make progress.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.infra.delegation_log import DelegationLog
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def _node(run_dir: Path) -> StrategizerNode:
    class S(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "Wait"})
        description = "s"

    class W(Agent):
        description = "w"

    spec = Graph(
        nodes={"strategizer": S(), "worker": W()},
        edges=(Edge("strategizer", "worker"),),
        entry="strategizer",
    )
    dlog = DelegationLog(run_dir / "debug" / "delegation_log.jsonl")
    notes_dir = run_dir / "debug" / "strategizer_notes"
    notes_dir.mkdir(parents=True)
    return StrategizerNode(
        _Stub(), name="strategizer", outgoing=["worker"], spec=spec,
        worker_adapters={"worker": _Stub()}, notes_dir=notes_dir,
        delegation_log=dlog,
    )


def _wait(tmp_path, registry):
    node = _node(tmp_path / "runs" / "T1")
    with node._registry_lock:
        node._registry.update(registry)
    return node, node._build_routing_closures()["Wait"]


def test_bare_wait_returns_a_finished_delegation_labelled_with_its_id(tmp_path):
    _, Wait = _wait(tmp_path, {
        "D001": {"status": "Working"},
        "D002": {"status": "Done", "result": "the report"},
    })
    out = Wait()
    # The agent did not name a target, so it must be told which one this is.
    assert "D002" in out, out
    assert "the report" in out, out


def test_bare_wait_harvests_each_delegation_exactly_once(tmp_path):
    """N in flight are drained by N bare Wait() calls — the whole point."""
    _, Wait = _wait(tmp_path, {
        "D001": {"status": "Done", "result": "first"},
        "D002": {"status": "Errored", "result": "second"},
    })
    first, second = Wait(), Wait()
    assert {"first" in first, "second" in first} == {True, False}
    assert {"first" in second, "second" in second} == {True, False}
    # Not the same report twice.
    assert ("first" in first) != ("first" in second), (first, second)
    # Both drained: a third call has nothing left and must refuse.
    assert "ERROR" in Wait()


def test_bare_wait_blocks_until_a_worker_finishes(tmp_path):
    node, Wait = _wait(tmp_path, {"D001": {"status": "Working"}})

    def finish():
        time.sleep(1.5)  # longer than one poll tick
        with node._registry_lock:
            node._registry["D001"].update(
                {"status": "Done", "result": "late report"})

    t = threading.Thread(target=finish, daemon=True)
    t.start()
    out = Wait()
    t.join(timeout=5)
    assert "late report" in out, out
    assert "D001" in out, out


def test_bare_wait_refuses_when_nothing_is_in_flight(tmp_path):
    _, Wait = _wait(tmp_path, {})
    out = Wait()
    assert "ERROR" in out and "nothing to wait for" in out, out


def test_bare_wait_refuses_instead_of_hanging_on_followup_only(tmp_path):
    """Every open worker parked on a FollowUp: waiting cannot make progress,
    so refuse and name the unblock path rather than blocking forever."""
    _, Wait = _wait(tmp_path, {"D001": {"status": "FollowUp"}})
    started = time.monotonic()
    out = Wait()
    assert time.monotonic() - started < 5, "bare Wait() hung on a FollowUp"
    assert "ERROR" in out and "FollowUp" in out, out
    assert "Reply(" in out, out
    assert "D001" in out, out


class _DeadThread:
    """A worker thread that has exited without recording a terminal status."""

    def is_alive(self) -> bool:
        return False


def test_bare_wait_refuses_instead_of_hanging_on_a_crashed_worker(tmp_path):
    """A blocking tool call ends no turn, so the run's time backstop cannot
    fire while we are inside Wait(). Waiting on a delegation whose thread is
    already gone would therefore hang the whole run, not just the call."""
    node, Wait = _wait(tmp_path, {"D001": {"status": "Working"}})
    node._threads["D001"] = _DeadThread()
    started = time.monotonic()
    out = Wait()
    assert time.monotonic() - started < 5, "bare Wait() hung on a dead worker"
    assert "ERROR" in out and "D001" in out, out
    assert "never reported" in out, out


def test_bare_wait_keeps_waiting_while_any_worker_is_still_live(tmp_path):
    """One crashed worker must not cancel the wait for a healthy sibling."""
    node, Wait = _wait(tmp_path, {
        "D001": {"status": "Working"},   # crashed
        "D002": {"status": "Working"},   # healthy, no thread registered
    })
    node._threads["D001"] = _DeadThread()

    def finish():
        time.sleep(1.5)
        with node._registry_lock:
            node._registry["D002"].update(
                {"status": "Done", "result": "sibling report"})

    t = threading.Thread(target=finish, daemon=True)
    t.start()
    out = Wait()
    t.join(timeout=5)
    assert "sibling report" in out, out


def test_bare_wait_never_harvests_a_cancelled_delegation(tmp_path):
    """Cancelled is terminal but its result is explicitly excluded from the
    run, so it is not a harvestable completion."""
    _, Wait = _wait(tmp_path, {
        "D001": {"status": "Cancelled", "result": "discarded"},
    })
    out = Wait()
    assert "ERROR" in out, out
    assert "discarded" not in out, out


def test_named_wait_still_returns_its_report_unlabelled(tmp_path):
    """The single-id form is unchanged: the agent named the target, so the
    report needs no ID prefix."""
    _, Wait = _wait(tmp_path, {"D001": {"status": "Done", "result": "r1"}})
    out = Wait("D001")
    assert "r1" in out, out
    assert not out.startswith("[D001]"), out


def test_named_wait_marks_the_report_read_for_a_later_bare_wait(tmp_path):
    """Otherwise a bare Wait() would hand back a report already collected."""
    _, Wait = _wait(tmp_path, {"D001": {"status": "Done", "result": "r1"}})
    assert "r1" in Wait("D001")
    assert "ERROR" in Wait()


def test_named_wait_still_reports_an_unknown_delegation(tmp_path):
    _, Wait = _wait(tmp_path, {})
    assert "unknown delegation" in Wait("D404")
