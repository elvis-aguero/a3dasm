"""Wait() must wake a strategizer blocked on a live delegation when the
science monitor has something to say — not just report it after the fact
on the next tool call, by which point the delegation (and its eval budget)
has already finished.

Regression for the DUPLICATE_EVALUATION nudge (backlog #24 / spec 07):
without this, a delegation duplicating design points for its whole run only
gets flagged once Wait() returns, i.e. never in time to matter.
"""
from __future__ import annotations

from pathlib import Path

from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


class _FakeThread:
    """Duck-types threading.Thread: alive for exactly one poll tick. Writes
    the duplicate rows from inside join() — i.e. AFTER Wait()'s initial
    `_drain_notifications()` call already ran and found an empty store, and
    only visible to a check that happens INSIDE the poll loop. This is what
    actually distinguishes "picked up mid-wait" from "already there before
    Wait() was even called" — a store seeded up-front would pass through
    Wait()'s existing top-of-function drain regardless of any mid-loop fix."""

    def __init__(self, on_join) -> None:
        self._alive = True
        self._on_join = on_join

    def is_alive(self) -> bool:
        was_alive = self._alive
        self._alive = False
        return was_alive

    def join(self, timeout=None) -> None:
        self._on_join()


def _store_dup_rows(store_dir: Path, delegation_id: str) -> None:
    dom = Domain()
    dom.add_float("x1", -5.0, 5.0)
    dom.add_float("x2", -5.0, 5.0)
    dom.add_output("y", exist_ok=True)
    dom.add_output("_delegation_id", exist_ok=True)
    samples = {
        i: ExperimentSample(
            _input_data={"x1": 0.1, "x2": 0.2},
            _output_data={"y": 0.0, "_delegation_id": delegation_id},
            job_status=JobStatus.FINISHED)
        for i in range(4)  # 1 unique point x4 = 3 duplicate rows -> fires
    }
    ExperimentData.from_data(data=samples, domain=dom).store(
        project_dir=store_dir)


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


def test_wait_drains_science_monitor_mid_poll(tmp_path):
    run_dir = tmp_path / "runs" / "T1"
    store_dir = run_dir / "experiment_data"
    store_dir.mkdir(parents=True)  # empty at Wait()-call time

    node = _node(run_dir)
    node._science_monitor.store_dir = str(store_dir)
    with node._registry_lock:
        node._registry["D003"] = {"status": "Working"}
    node._threads["D003"] = _FakeThread(
        on_join=lambda: _store_dup_rows(store_dir, "D003"))

    routing = node._build_routing_closures()
    result = routing["Wait"]("D003")

    assert "DUPLICATE_EVALUATION" in result, (
        f"Wait() did not surface the science-monitor nudge mid-poll: {result!r}"
    )
