"""Tests for route-aware run termination.

Covers:
- Bounded re-prompt on unaccepted termination (max 3 attempts)
- UNGATED banner on exhausted attempts
- Accepted Done() → clean END with no banner
- Run-level cost backstop (elapsed > RUN_BACKSTOP_MULTIPLE * budget)
- Working delegations survive loopbacks
- Ledger duplicate-statement guard
- ReadNote directory guard
"""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from a3dasm._src.backends.base import Agent, Edge, Graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class StubAdapter:
    """Minimal adapter stub."""

    def __init__(self, response: str = "Final analysis complete.") -> None:
        self._response = response
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.invoke_count = 0

    def invoke(self, messages: list[dict]) -> str:
        self.invoke_count += 1
        return self._response


def _minimal_spec(name: str = "strategizer", target: str = "implementer") -> Graph:
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote", "WriteDeliverable"})
        description = "Test strategizer."

    class B(Agent):
        description = "Test implementer."

    return Graph(
        nodes={name: A(), target: B()},
        edges=(Edge(name, target),),
        entry=name,
    )


def _make_state(study_dir=None, messages=None, **kwargs):
    from a3dasm._src.graph_state import AgenticState

    if study_dir is None:
        d = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
        (d / "pipeline.py").write_text("# test pipeline\n")
        study_dir = d
    return AgenticState(
        messages=messages or [HumanMessage(content="Test problem")],
        study_dir=str(study_dir),
        done=False,
        last_report=None,
        total_delegations=0,
        budget_seconds=kwargs.pop("budget_seconds", None),
        return_to=kwargs.pop("return_to", None),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Bounded re-prompt on unaccepted termination
# ---------------------------------------------------------------------------


def test_unaccepted_termination_reprompts():
    """When adapter doesn't call Done(), node loops back with diagnostic message."""
    from a3dasm._src.nodes import StrategizerNode

    # Adapter never calls Done
    adapter = StubAdapter(response="Final analysis complete.")
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )

    # Create study_dir with pipeline.py so missing-deliverables is NOT the issue
    study_dir = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
    (study_dir / "pipeline.ipynb").write_text("# test\n")
    state = _make_state(study_dir=study_dir)

    cmd = node(state)

    # Should loop back, not terminate
    assert cmd.goto == "strategizer", f"Expected loopback, got goto={cmd.goto!r}"
    assert node._finish_attempts == 1

    # Injected message should explain the issue
    messages = cmd.update.get("messages", [])
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    assert human_msgs, "No HumanMessage injected in loopback update"
    assert any(
        "without an accepted Done" in m.content for m in human_msgs
    ), f"Expected 'without an accepted Done' in messages; got: {[m.content for m in human_msgs]}"


def test_ungated_finish_after_three_attempts():
    """After 3 loopbacks, 4th call terminates with UNGATED banner."""
    from a3dasm._src.nodes import StrategizerNode

    # Adapter never calls Done; pipeline.py present so only Done is missing
    study_dir = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
    (study_dir / "pipeline.ipynb").write_text("# test\n")

    adapter = StubAdapter(response="Final analysis complete.")
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )

    # Simulate 3 loopbacks by calling __call__ and re-feeding messages
    state = _make_state(study_dir=study_dir)
    for attempt in range(1, 4):
        cmd = node(state)
        assert cmd.goto == "strategizer", (
            f"Attempt {attempt}: expected loopback, got goto={cmd.goto!r}"
        )
        assert node._finish_attempts == attempt
        # Feed the injected messages back into state for next call
        state = _make_state(
            study_dir=study_dir,
            messages=list(state["messages"]) + list(cmd.update.get("messages", [])),
        )

    # 4th call — attempts exhausted — must terminate
    cmd = node(state)
    assert cmd.goto == END, f"Expected END after 3 attempts, got goto={cmd.goto!r}"
    assert cmd.update.get("done") is True
    last_report = cmd.update.get("last_report", "")
    assert "UNGATED RUN" in last_report, (
        f"Expected 'UNGATED RUN' in last_report; got: {last_report!r}"
    )


def test_accepted_done_no_banner():
    """Full Done() dance → END with no UNGATED banner; _finish_attempts stays 0."""
    from a3dasm._src.nodes import StrategizerNode

    study_dir = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
    (study_dir / "pipeline.ipynb").write_text("# test\n")

    class DoneCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.invoke_count += 1
            self.closure_tools["Done"](summary="All done.")  # first: warning
            self.closure_tools["Done"](summary="All done.")  # second: accepted
            return "Run complete."

    adapter = DoneCallingAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )
    state = _make_state(study_dir=study_dir)
    cmd = node(state)

    assert cmd.goto == END, f"Expected END, got {cmd.goto!r}"
    assert cmd.update.get("done") is True
    last_report = cmd.update.get("last_report", "")
    assert "UNGATED" not in last_report, (
        f"Expected no UNGATED banner in accepted run; got: {last_report!r}"
    )
    assert node._finish_attempts == 0


# ---------------------------------------------------------------------------
# 2. Hard budget stop
# ---------------------------------------------------------------------------


def test_run_backstop_halts_resumable_past_multiple(tmp_path):
    """Past RUN_BACKSTOP_MULTIPLE x budget: invoke skipped, the run HALTS
    cleanly and resumably — a HALTED banner is prefixed (conclusion kept
    below it) and debug/run_status.json marks it resumable."""
    from a3dasm._src.nodes import StrategizerNode

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "pipeline.ipynb").write_text("# test\n")
    run_dir = study_dir / "runs" / "T"
    (run_dir / "debug").mkdir(parents=True)
    (run_dir / "debug" / "thread_id").write_text("tid-xyz")

    adapter = StubAdapter(response="Should not be called.")
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )

    # budget=10s, started 100s ago → elapsed = 10x budget >> 2x backstop.
    # Provide a prior AI conclusion so we can assert it is preserved.
    state = _make_state(
        study_dir=study_dir,
        messages=[HumanMessage(content="p"), AIMessage(content="CONCLUSION X")],
    )
    state["budget_seconds"] = 10
    state["start_time"] = time.time() - 100
    state["run_dir"] = str(run_dir)

    cmd = node(state)

    assert adapter.invoke_count == 0, "invoke must be skipped past the backstop"
    assert cmd.goto == END
    assert cmd.update.get("done") is True
    report = cmd.update.get("last_report", "")
    assert "HALTED (resumable)" in report and "time backstop" in report
    assert "CONCLUSION X" in report, "the prior conclusion must be preserved"

    import json
    status = json.loads(
        (run_dir / "debug" / "run_status.json").read_text()
    )
    assert status["status"] == "halted"
    assert status["resumable"] is True
    assert status["thread_id"] == "tid-xyz"


def test_usd_budget_exhausted_halts_resumable(tmp_path):
    """When accrued cost reaches budget_usd, the run halts resumably."""
    from a3dasm._src.nodes import StrategizerNode

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "pipeline.ipynb").write_text("# test\n")
    run_dir = study_dir / "runs" / "T"
    (run_dir / "debug").mkdir(parents=True)
    (run_dir / "debug" / "thread_id").write_text("tid-usd")

    adapter = StubAdapter(response="Should not be called.")
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"],
        spec=_minimal_spec(),
    )
    # Simulate cost already accrued past the ceiling (claude reports cost).
    node._accumulate_usage({"input_tokens": 1, "total_cost_usd": 0.60})
    assert node._cost_observed is True

    state = _make_state(study_dir=study_dir)
    state["budget_usd"] = 0.50
    state["run_dir"] = str(run_dir)

    cmd = node(state)

    assert adapter.invoke_count == 0
    assert cmd.goto == END and cmd.update.get("done") is True
    assert "USD budget exhausted" in cmd.update.get("last_report", "")
    import json
    status = json.loads((run_dir / "debug" / "run_status.json").read_text())
    assert status["status"] == "halted" and status["resumable"] is True


def test_usd_budget_inactive_under_ollama_does_not_halt(tmp_path):
    """No per-call cost (ollama) → the USD ceiling is inactive: the run is
    NOT halted even with a budget_usd set, and the strategizer runs."""
    from a3dasm._src.nodes import StrategizerNode

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "pipeline.ipynb").write_text("# test\n")

    adapter = StubAdapter(response="ollama answer")
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"],
        spec=_minimal_spec(),
    )
    # Ollama: cost is None → never observed.
    node._accumulate_usage({"input_tokens": 100, "total_cost_usd": None})
    assert node._cost_observed is False
    # Pretend a prior turn happened so the inactive-warning path is reachable.
    node._turn_count = 1

    state = _make_state(study_dir=study_dir)
    state["budget_usd"] = 0.01

    node(state)
    assert adapter.invoke_count == 1, "ollama run must continue (USD inactive)"
    assert node._usd_inactive_warned is True


def test_repeated_errors_halt_resumable(tmp_path, monkeypatch):
    """N consecutive Errored delegations from one target → resumable halt."""
    from a3dasm._src.nodes import StrategizerNode

    monkeypatch.setenv("F3DASM_MAX_CONSECUTIVE_ERRORS", "3")
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "pipeline.ipynb").write_text("# test\n")
    run_dir = study_dir / "runs" / "T"
    (run_dir / "debug").mkdir(parents=True)
    (run_dir / "debug" / "thread_id").write_text("tid-err")

    adapter = StubAdapter(response="Should not be called.")
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"],
        spec=_minimal_spec(),
    )
    node._consecutive_errors["implementer"] = 3  # at threshold

    state = _make_state(study_dir=study_dir)
    state["run_dir"] = str(run_dir)

    cmd = node(state)
    assert adapter.invoke_count == 0
    assert cmd.goto == END and cmd.update.get("done") is True
    assert "repeated errors: implementer" in cmd.update.get("last_report", "")


def test_soft_budget_does_not_terminate_below_backstop():
    """Time budget is SOFT: past 100% but below the backstop, the run
    CONTINUES (adapter.invoke is called) — warning only, no force-end."""
    from a3dasm._src.nodes import StrategizerNode

    study_dir = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
    (study_dir / "pipeline.ipynb").write_text("# test\n")

    adapter = StubAdapter(response="Continuing despite soft warning.")
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )

    # budget=10s, started 15s ago → 1.5x budget: over 100%, under 2x
    state = _make_state(study_dir=study_dir)
    state["budget_seconds"] = 10
    state["start_time"] = time.time() - 15

    node(state)

    assert adapter.invoke_count == 1, (
        "soft budget must NOT force-terminate below the backstop"
    )


# ---------------------------------------------------------------------------
# 3. Working delegations survive loopback
# ---------------------------------------------------------------------------


def test_working_delegations_survive_loopback():
    """A Working registry entry is not cleared by the A1/A2 reset on loopback."""
    from a3dasm._src.nodes import StrategizerNode

    study_dir = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
    (study_dir / "pipeline.ipynb").write_text("# test\n")

    # Adapter does NOT call Done → triggers loopback
    adapter = StubAdapter(response="Still thinking.")
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )

    # Seed the registry with a Working entry BEFORE __call__
    fake_event = threading.Event()
    with node._registry_lock:
        node._registry["D001"] = {
            "status": "Working",
            "result": None,
            "evals": 0,
            "hypothesis_ids": [],
            "started_at": "2026-01-01T00:00:00+00:00",
            "start_time": time.monotonic(),
            "followup_event": fake_event,
            "followup_question": None,
            "followup_answer": None,
            "followup_count": 0,
        }
        node._threads["D001"] = threading.Thread(target=lambda: None)

    state = _make_state(study_dir=study_dir)
    cmd = node(state)

    # Should be a loopback
    assert cmd.goto == "strategizer", f"Expected loopback, got {cmd.goto!r}"

    # The Working entry must still be in the registry
    with node._registry_lock:
        assert "D001" in node._registry, (
            f"D001 was cleared from registry during loopback; registry={node._registry}"
        )
        assert node._registry["D001"]["status"] == "Working"


def test_running_delegation_does_not_burn_finish_attempts():
    """Run-4 back door: a healthy delegation still in flight must NOT consume the
    bounded finish-attempt budget. Waiting it out is work, not a failed finish —
    otherwise a slow-but-healthy delegation force-terminates the run UNGATED with
    wall budget to spare. The run's time backstop bounds a true hang instead."""
    from a3dasm._src.nodes import StrategizerNode

    study_dir = Path(tempfile.mkdtemp(prefix="f3dasm_rat_"))
    (study_dir / "pipeline.ipynb").write_text("# test\n")
    adapter = StubAdapter(response="Polling D004.")
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"],
        spec=_minimal_spec())

    fake_event = threading.Event()
    with node._registry_lock:
        node._registry["D004"] = {
            "status": "Working", "result": None, "evals": 0,
            "hypothesis_ids": [], "started_at": "2026-01-01T00:00:00+00:00",
            "start_time": time.monotonic(), "followup_event": fake_event,
            "followup_question": None, "followup_answer": None,
            "followup_count": 0,
        }
        node._threads["D004"] = threading.Thread(target=lambda: None)

    # Many turns while D004 keeps running → always loops back, NEVER terminates,
    # NEVER increments _finish_attempts (the old behavior died after 3).
    state = _make_state(study_dir=study_dir)
    for _ in range(5):
        cmd = node(state)
        assert cmd.goto == "strategizer", f"expected poll loopback, got {cmd.goto!r}"
        assert node._finish_attempts == 0, (
            f"a running delegation burned a finish attempt: {node._finish_attempts}"
        )
        msgs = [m.content for m in cmd.update.get("messages", [])
                if isinstance(m, HumanMessage)]
        assert any("still running" in m.lower() for m in msgs), msgs
        state = _make_state(
            study_dir=study_dir,
            messages=list(state["messages"]) + list(cmd.update.get("messages", [])))


# ---------------------------------------------------------------------------
# 4. Ledger duplicate-statement guard
# ---------------------------------------------------------------------------


def test_propose_rejects_duplicate_statement(tmp_path):
    """Proposing an identical statement twice returns ERROR citing the first H-id."""
    from a3dasm._src.hypothesis_ledger import HypothesisLedger

    ledger = HypothesisLedger(tmp_path)

    stmt = "Optimal t/L lies near 0.08 for re-entrant geometry"
    h1 = ledger.propose(
        statement=stmt,
        falsification_criterion="A design outside [0.07,0.09] beats buckling_load 1.47",
        prediction="Best design found in range 0.07-0.09",
        prior=0.6,
        proposed_by="strategizer",
    )
    assert h1 == "H1", f"First propose should return H1, got {h1!r}"

    # Exact duplicate
    result = ledger.propose(
        statement=stmt,
        falsification_criterion="Different criterion",
        prediction="Different prediction",
        prior=0.4,
        proposed_by="strategizer",
    )
    assert result.startswith("ERROR"), f"Expected ERROR for duplicate, got {result!r}"
    assert "H1" in result, f"Expected H1 in error message, got {result!r}"
    assert "duplicate" in result.lower(), f"Expected 'duplicate' in error, got {result!r}"


def test_propose_rejects_duplicate_case_whitespace(tmp_path):
    """Duplicate detection is case- and whitespace-insensitive."""
    from a3dasm._src.hypothesis_ledger import HypothesisLedger

    ledger = HypothesisLedger(tmp_path)

    stmt = "Optimal t/L lies near 0.08 for re-entrant geometry"
    h1 = ledger.propose(
        statement=stmt,
        falsification_criterion="Counter-example outside range",
        prediction="Best in-range",
        prior=0.6,
        proposed_by="strategizer",
    )
    assert h1 == "H1"

    # Variant with different case and extra whitespace
    variant = "  OPTIMAL  T/L  LIES  NEAR  0.08  FOR  RE-ENTRANT  GEOMETRY  "
    result = ledger.propose(
        statement=variant,
        falsification_criterion="Counter",
        prediction="Best",
        prior=0.4,
        proposed_by="strategizer",
    )
    assert result.startswith("ERROR"), f"Expected ERROR for case/whitespace variant, got {result!r}"
    assert "H1" in result


# ---------------------------------------------------------------------------
# 5. ReadNote directory guard
# ---------------------------------------------------------------------------


def test_readnote_directory_returns_listing(tmp_path):
    """ReadNote on a directory returns a file LISTING (not an error) so the agent
    can discover and reuse the implementers' delegation code."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "pipeline.py").write_text("# test\n")
    # Create a subdirectory with a file to pass as the ReadNote path
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "worker_code.py").write_text("# reusable\n")

    results: list[str] = []

    class ReadNoteAdapter(StubAdapter):
        def invoke(self, messages):
            self.invoke_count += 1
            result = self.closure_tools["ReadNote"]("subdir")  # points to a directory
            results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = ReadNoteAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        study_dir=str(tmp_path),
    )
    state = _make_state(study_dir=tmp_path)
    node(state)

    assert results, "ReadNote was never called"
    assert "directory" in results[0].lower(), (
        f"Expected a directory listing, got: {results[0]!r}"
    )
    assert "worker_code.py" in results[0], (
        f"Expected the directory's files listed, got: {results[0]!r}"
    )


def test_readnote_rejects_paths_escaping_study_dir(tmp_path):
    """Health-run-1 hang: ReadNote('/') resolved to the filesystem root
    (Path(study)/'/' == Path('/')) and rglob('*') walked the WHOLE filesystem,
    hanging the run. ReadNote must contain to the study dir and reject escapes
    fast — never walk outside it."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "pipeline.py").write_text("# test\n")
    results: list[str] = []

    class EscapeAdapter(StubAdapter):
        def invoke(self, messages):
            self.invoke_count += 1
            # the exact trigger + a ..-escape; both must return fast, no hang
            results.append(self.closure_tools["ReadNote"]("/"))
            results.append(self.closure_tools["ReadNote"]("../../../../etc"))
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    node = StrategizerNode(
        EscapeAdapter(), name="strategizer", outgoing=["implementer"],
        spec=_minimal_spec(), study_dir=str(tmp_path))
    node(_make_state(study_dir=tmp_path))

    assert len(results) == 2
    for r in results:
        assert "outside the study directory" in r, (
            f"escape not contained, got: {r!r}")
