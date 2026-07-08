"""Friction #5 fix: premature-Done becomes a 3-option soft nudge, and a new
CancelDelegation tool lets the strategizer detach a still-running delegation so
it stops blocking Done() (instead of bouncing on "wait for all delegations")."""
from __future__ import annotations

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def _node():
    class A(Agent):
        role = "strategizer"
        # GetStatus/CancelDelegation are opt-in (plug-and-play) post-audit; the
        # test strategizer opts in so these behaviour tests still exercise them.
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote",
                           "WriteDeliverable", "CheckDeliverable",
                           "GetStatus", "CancelDelegation"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),), entry="strategizer",
    )
    n = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()},
    )
    return n


def test_cancel_delegation_detaches_and_unblocks():
    n = _node()
    n._registry["D001"] = {"status": "Working", "result": None, "evals": 0}
    out = n.adapter.closure_tools["CancelDelegation"]("D001")
    assert "cancelled" in out.lower() and "detached" in out.lower()
    assert n._registry["D001"]["status"] == "Cancelled"
    # premature-Done guard counts only Working → cancelled one no longer blocks
    pending = [d for d, e in n._registry.items() if e["status"] == "Working"]
    assert pending == []


def test_cancel_unknown_and_already_settled():
    n = _node()
    assert "no delegation" in n.adapter.closure_tools["CancelDelegation"](
        "D999").lower()
    n._registry["D002"] = {"status": "Done"}
    out = n.adapter.closure_tools["CancelDelegation"]("D002")
    assert "not" in out.lower() and "running" in out.lower()
    assert n._registry["D002"]["status"] == "Done"  # untouched


def test_premature_done_is_a_soft_two_option_nudge():
    n = _node()
    n._registry["D001"] = {"status": "Working"}
    out = n.adapter.closure_tools["Done"](summary="all done")
    # the nudge, not a hard error
    assert not out.lstrip().startswith("ERROR:")
    assert "two options" in out.lower()
    # CancelDelegation is dropped from production; the nudge no longer offers it.
    assert "GetStatus" in out
    assert "CancelDelegation" not in out
    # soft return → NOT counted as a tool error
    assert n._error_counts.get("strategizer", 0) == 0


def test_cancel_delegation_tool_is_registered():
    n = _node()
    assert "CancelDelegation" in n.adapter.closure_tools


def test_poll_escalation_offers_options():
    """A repeatedly-polled delegation gets real options (do other work / just
    wait), not a 'poll less' nag — so the agent never grinds out 30 status
    checks. CancelDelegation was dropped from production, so the stuck remedy
    is now the run watchdog, not agent-driven cancel."""
    import time as _t
    n = _node()
    n._registry["D001"] = {
        "status": "Working", "result": None,
        "start_time": _t.monotonic(), "getstatus_count": 0}
    out = ""
    for _ in range(6):  # cross the >=5 escalation
        out = n.adapter.closure_tools["GetStatus"]("D001")
    assert out.lstrip().startswith("Working")
    assert "do other work" in out.lower()          # (a) do something else
    assert "just wait" in out.lower()              # (b) wait it out
    assert "watchdog" in out.lower()               # stuck remedy (cancel dropped)
    assert "wait=True" in out                       # future-proofing tip
    assert "CancelDelegation" not in out


def _seed_store(store_dir, delegation_id):
    from a3dasm._src.instrumented import InstrumentedDataGenerator
    from f3dasm._src.core import DataGenerator
    from f3dasm._src.experimentsample import ExperimentSample, JobStatus

    class _Sum(DataGenerator):
        def execute(self, s, **k):
            s._output_data["f"] = sum(s._input_data.values())
            s.job_status = JobStatus.FINISHED
            return s

    gen = InstrumentedDataGenerator(
        inner=_Sum(), store_dir=store_dir, delegation_id=delegation_id,
        flush_every=1)
    gen.execute(ExperimentSample(
        _input_data={"x0": 1.0}, _output_data={}, job_status=JobStatus.OPEN))
    gen.flush()


def test_cancel_is_two_shot_for_progressing_delegation(tmp_path):
    """A delegation already writing ledgered evals is progressing, not stuck:
    the first cancel is HELD, a deliberate second call confirms."""
    run_dir = tmp_path / "runs" / "T0"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    _seed_store(run_dir / "experiment_data", "D004")
    n = _node()
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    n._registry["D004"] = {"status": "Working", "result": None, "evals": 0}

    out1 = n.adapter.closure_tools["CancelDelegation"]("D004")
    assert "HOLD" in out1
    assert n._registry["D004"]["status"] == "Working"   # NOT cancelled yet
    out2 = n.adapter.closure_tools["CancelDelegation"]("D004")
    assert "cancelled" in out2.lower()
    assert n._registry["D004"]["status"] == "Cancelled"


def test_followup_headless_proceeds_without_eoferror():
    """FollowUp must never block on input() in a non-interactive run (no TTY).

    Regression: a background run hit EOFError at `return input()`. The strategizer
    should instead be told no operator is present and proceed autonomously.
    """
    n = _node()
    # Default node is non-interactive → autonomous notice, no crash.
    out = n.adapter.closure_tools["FollowUp"]("Which units?")
    assert "no operator" in out.lower()
    # Even if flagged interactive, a non-TTY stdin (as under pytest) must NOT
    # crash — the isatty guard falls through to the same autonomous notice.
    n._interactive = True
    n._ask_count = 0
    out2 = n.adapter.closure_tools["FollowUp"]("Again?")
    assert "no operator" in out2.lower()


def test_done_bounces_on_broken_pipeline_before_critic(tmp_path):
    """A pipeline.py that fails to run bounces back in Done() WITHOUT spending a
    critic consult — the pre-critic reproduction gate (timeout-bounded)."""
    run_dir = tmp_path / "runs" / "T2"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    (run_dir / "experiment_data").mkdir()
    n = _node()  # graph has no critic → the gate still runs before close
    n._study_dir = tmp_path
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    (tmp_path / "pipeline.py").write_text("import sys\nsys.exit(1)\n")

    done = n.adapter.closure_tools["Done"]
    done(summary="x")                       # first call → two-shot warn
    out = done(summary="x")                 # second call → repro gate bounce
    assert "reproduction gate" in out.lower()
    assert n._route.get("kind") != "done"   # run did NOT close
    assert n._repro_attempts == 1


def _working_entry():
    import time as _t
    return {"status": "Working", "result": None,
            "start_time": _t.monotonic(), "getstatus_count": 0}


def test_getstatus_reports_ledger_progress_when_evals_stamped(tmp_path):
    """Feature (c): GetStatus surfaces stamped-eval progress so the delegator
    sees a delegation is progressing (and won't cancel it out of blindness)."""
    run_dir = tmp_path / "runs" / "T3"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    _seed_store(run_dir / "experiment_data", "D007")  # 1 stamped eval
    n = _node()
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    n._registry["D007"] = _working_entry()
    out = n.adapter.closure_tools["GetStatus"]("D007")
    assert out.startswith("Working")
    assert "evals stamped" in out and "progressing" in out


def test_getstatus_flags_zero_progress_as_possible_stuck(tmp_path):
    """Backlog #6 signal: zero stamped evals is surfaced distinctly (the only
    case where cancelling is framed as reasonable)."""
    run_dir = tmp_path / "runs" / "T4"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    (run_dir / "experiment_data").mkdir()
    n = _node()
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    n._registry["D008"] = _working_entry()
    out = n.adapter.closure_tools["GetStatus"]("D008")
    assert "0 evals stamped" in out


def test_getstatus_surfaces_worker_progress_note(tmp_path):
    """Thin (a): a non-blocking worker note shows up on the delegator's poll."""
    import time as _t
    run_dir = tmp_path / "runs" / "T5"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    (run_dir / "experiment_data").mkdir()
    n = _node()
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    entry = _working_entry()
    entry["progress_note"] = ("LHS done, 250 evals; fitting GP", _t.monotonic())
    n._registry["D009"] = entry
    out = n.adapter.closure_tools["GetStatus"]("D009")
    assert "worker note:" in out and "LHS done" in out


def _study_with_store(tmp_path, name):
    run_dir = tmp_path / "runs" / name
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    _seed_store(run_dir / "experiment_data", "D004")  # 1 row, objective f=1.0
    n = _node()
    n._study_dir = tmp_path
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    return n


def _write_nb(study_dir, source):
    """Write study_dir/pipeline.ipynb with a single code cell running `source`."""
    import nbformat

    from a3dasm._src.notebook_exec import build_notebook
    nb = build_notebook([{"type": "code", "name": "analysis", "source": source}])
    nbformat.write(nb, str(study_dir / "pipeline.ipynb"))


def test_check_deliverable_reports_pass_and_failure(tmp_path):
    """CheckDeliverable runs pipeline.ipynb through the gate WITHOUT closing: PASS
    for a grounded headline, full error for a broken notebook. Gives the agent
    sight to debug its own deliverable (the missing capability)."""
    n = _study_with_store(tmp_path, "C0")
    check = n.adapter.closure_tools["CheckDeliverable"]
    # no notebook yet
    assert "no pipeline.ipynb" in check().lower()
    # broken notebook → full error, NOT YET
    _write_nb(tmp_path, "import sys\nsys.exit(1)")
    out = check()
    assert "not yet" in out.lower() and "failed" in out.lower()
    # grounded headline (ledger max objective is 1.0) → PASS
    _write_nb(tmp_path, "print('REPRODUCED: 1.0')")
    assert check().lstrip().startswith("PASS")


def test_readnote_lists_a_directory_so_delegation_code_is_discoverable(tmp_path):
    """ReadNote on a directory returns a file LISTING (so the agent can discover
    and reuse the implementers' delegation code) instead of 'pass a file path'."""
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "ReadNote"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),), entry="strategizer",
    )
    d = tmp_path / "debug" / "delegations" / "D005"
    d.mkdir(parents=True)
    (d / "local_search.py").write_text("# the local search that found the optimum\n")
    n = StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()}, study_dir=str(tmp_path),
    )
    listing = n.adapter.closure_tools["ReadNote"]("debug/delegations/D005")
    assert "local_search.py" in listing and "directory" in listing.lower()
    body = n.adapter.closure_tools["ReadNote"](
        "debug/delegations/D005/local_search.py")
    assert "found the optimum" in body


def test_check_deliverable_shows_countdown_and_bounds_iteration(tmp_path):
    """CheckDeliverable has a visible 10-call budget: each call reports how many
    remain (so the agent never hits an unseen wall), and the 11th refuses —
    converting an endless check↔write grind into a fast, bounded close."""
    n = _study_with_store(tmp_path, "C2")
    _write_nb(tmp_path, "import sys\nsys.exit(1)")  # never passes
    check = n.adapter.closure_tools["CheckDeliverable"]
    first = check()
    assert "1/10 used" in first and "9 checks left" in first
    outs = [check() for _ in range(9)]   # calls 2..10
    assert "10/10 used" in outs[-1] and "0 checks left" in outs[-1]
    # 11th call refuses without running the gate again.
    exhausted = check()
    assert "budget exhausted" in exhausted.lower() and "Done()" in exhausted


def test_repro_failure_closes_FAILED_not_ungated(tmp_path):
    """After the bounded sighted attempts, a non-reproducing pipeline closes in a
    distinct FAILED state (loud ⛔ banner) via the retrospective round — NOT a
    quiet GATED/UNGATED, and the critic is never spent on it."""
    n = _study_with_store(tmp_path, "C1")
    (tmp_path / "pipeline.py").write_text("import sys\nsys.exit(1)\n")
    done = n.adapter.closure_tools["Done"]
    out = ""
    for _ in range(12):  # two-shot warn + 6 bounces + giveup
        out = done(summary="best result")
        if "Retrospective" in out:
            break
    assert "Retrospective" in out          # routed into the failed-run interview
    assert n._route.get("kind") != "done"  # not closed until the retro Done()
    # the capability-gap probe must be in that interview
    assert "BLOCKED" in out
    # the retrospective Done() then closes with the loud FAILED banner
    done(summary="### Retrospective\n- BLOCKED: no way to run pipeline.py")
    assert n._route.get("kind") == "done"
    assert "FAILED RUN" in n._route.get("summary", "")


def test_retrospectives_probe_capability_gaps():
    """Both interview prompts ask the BLOCKED capability-gap question (so 'I
    couldn't run my own deliverable' can surface)."""
    from a3dasm._src.nodes.tools.routing import (
        _EXIT_INTERVIEW, _FAILED_RETROSPECTIVE)
    for txt in (_EXIT_INTERVIEW, _FAILED_RETROSPECTIVE):
        assert "BLOCKED" in txt
        assert "couldn't" in txt.lower() or "could not" in txt.lower()


def test_cancel_single_shot_when_no_ledgered_evals(tmp_path):
    """No ledgered evals → cancel is immediate (the impatience guard only
    trips for delegations actually producing evaluations)."""
    run_dir = tmp_path / "runs" / "T1"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    (run_dir / "experiment_data").mkdir()
    n = _node()
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    n._registry["D003"] = {"status": "Working", "result": None, "evals": 0}
    out = n.adapter.closure_tools["CancelDelegation"]("D003")
    assert n._registry["D003"]["status"] == "Cancelled"
    assert "HOLD" not in out


def test_ghost_delegation_flushed_interrupted_at_run_close(tmp_path):
    """Working delegation at run-close gets an INTERRUPTED record in the log.

    Root cause: daemon threads are killed at process exit before they write
    their DONE entry, leaving a ghost RUNNING entry. The close path must flush
    an INTERRUPTED terminal record for every still-Working entry so
    query_all() (last-wins) collapses to a closed state rather than RUNNING.

    The adapter bypasses Done() and sets _route["kind"] = "done" directly —
    reproducing the pre-fix scenario where Done() accepted without first
    checking for still-running delegations.
    """
    import time as _t

    from a3dasm._src.backends.base import Agent, Edge, Graph
    from a3dasm._src.delegation_log import DelegationLog

    run_dir = tmp_path / "runs" / "T0"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    # pipeline.ipynb must exist so _missing_deliverables() doesn't block close
    _write_nb(tmp_path, "print('REPRODUCED: 1.0')")

    dlog_path = run_dir / "debug" / "delegation_log.jsonl"
    dlog = DelegationLog(dlog_path)

    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "WriteDeliverable", "CheckDeliverable"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),), entry="strategizer",
    )

    class _BypassDoneAdapter(_Stub):
        """Simulates the pre-fix scenario: closes without checking _working_now."""
        _node_ref = None

        def invoke(self, messages):
            # Set route directly — bypasses Done()'s soft nudge for Working
            # delegations (which didn't exist at the time of the original bug).
            self._node_ref._route["kind"] = "done"
            self._node_ref._route["summary"] = "test close"
            return "Done."

    adapter = _BypassDoneAdapter()
    n = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()},
        delegation_log=dlog, study_dir=str(tmp_path),
    )
    adapter._node_ref = n
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"

    # Seed the ghost: Working entry in the registry with no matching thread.
    n._registry["D004"] = {
        "status": "Working", "result": None, "evals": 0,
        "target": "implementer", "hypothesis_ids": [],
        "is_falsification_attempt": False, "phase": None,
        "started_at": "2026-01-01T00:00:00+00:00",
        "start_time": _t.monotonic(),
    }

    n({
        "messages": [],
        "study_dir": str(tmp_path),
        "done": False,
        "last_report": None,
        "total_delegations": 0,
        "budget_seconds": None,
        "budget_usd": None,
        "run_dir": str(run_dir),
        "eval_budget": None,
        "evals_used": 0,
        "start_time": _t.time(),
        "return_to": None,
        "required_deliverables": None,
        "experiment_data_dir": str(run_dir / "experiment_data"),
        "token_totals": None,
        "error_counts": None,
    })

    entries = {e["id"]: e for e in dlog.query_all()}
    assert "D004" in entries, "D004 missing from delegation log"
    assert entries["D004"]["status"] == "INTERRUPTED", (
        f"expected INTERRUPTED, got {entries['D004']['status']}"
    )
