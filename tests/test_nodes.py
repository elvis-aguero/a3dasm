"""Tests for AgentNode, StrategizerNode, ImplementerNode."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END
from langgraph.types import Command

from a3dasm._src.backends.base import Agent, Edge, Graph


@pytest.fixture(autouse=True)
def _isolate_from_milestone_gate():
    """These tests exercise delegation/hypothesis/Done MECHANICS, not the
    process backlog (which has its own coverage in test_milestones.py). Disable
    the backlog so the milestone gate doesn't block implementer delegations
    under test. Reset after each test."""
    from a3dasm._src import settings
    settings.configure({"milestones_enabled": False})
    yield
    settings.configure({})


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------

class StubAdapter:
    """Minimal adapter stub for node tests."""

    def __init__(self, response: str = "## Done\nAll done.") -> None:
        self._response = response
        self.closure_tools: dict = {}

    def invoke(self, messages: list[dict]) -> str:
        return self._response


def _minimal_spec(name: str = "strategizer", target: str = "implementer") -> Graph:
    """Return a minimal two-node Graph for StrategizerNode tests."""
    class A(Agent):
        role = "strategizer"
        # GetStatus/CancelDelegation are opt-in (plug-and-play) post-audit; the
        # test strategizer opts in so behaviour tests still exercise them even
        # though production agents no longer grant them.
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote",
                           "GetStatus", "CancelDelegation", "HypothesisPropose", "HypothesisUpdate", "HypothesisList", "HypothesisGet", "LinkFalsificationAttempt", "MilestoneList", "MilestonePropose", "MilestoneComplete", "MilestoneSkip", "RecallStore", "QueryStore"})
        description = "Test strategizer."

    class B(Agent):
        description = "Test implementer."

    return Graph(
        nodes={name: A(), target: B()},
        edges=(Edge(name, target),),
        entry=name,
    )


_DEFAULT_STUDY_DIR: Path | None = None


def _default_study_dir() -> Path:
    """Return a shared temp study dir with pipeline.ipynb pre-written.

    Created once per test session; all make_state() calls that don't supply
    an explicit study_dir share this directory so Done() always has the
    required deliverable in place.
    """
    global _DEFAULT_STUDY_DIR
    if _DEFAULT_STUDY_DIR is None:
        import tempfile

        import nbformat
        from a3dasm._src.notebook_exec import build_notebook
        d = Path(tempfile.mkdtemp(prefix="f3dasm_test_"))
        # Must satisfy the controlled reproduction gate (a code cell printing a
        # verifiable REPRODUCED sentinel) so Done() reaches the critic.
        nb = build_notebook([
            {"type": "code", "name": "analysis",
             "source": "print('REPRODUCED: 0.0')"}])
        nbformat.write(nb, str(d / "pipeline.ipynb"))
        _DEFAULT_STUDY_DIR = d
    return _DEFAULT_STUDY_DIR


def make_state(
    messages=None,
    study_dir=None,
    done=False,
    last_report=None,
    total_delegations=0,
    budget_seconds=None,
    return_to=None,
):
    from a3dasm._src.graph_state import AgenticState

    return AgenticState(
        messages=messages or [HumanMessage(content="Test problem")],
        study_dir=str(study_dir or _default_study_dir()),
        done=done,
        last_report=last_report,
        total_delegations=total_delegations,
        budget_seconds=budget_seconds,
        return_to=return_to,
    )


# ---------------------------------------------------------------------------
# StrategizerNode tests
# ---------------------------------------------------------------------------


def test_strategizer_routes_done_when_done_called():
    """StrategizerNode returns Command(goto=END) when Done closure is called twice.

    Done() is two-shot: first call issues a WARNING, second call closes the run.
    """
    from a3dasm._src.nodes import StrategizerNode

    class DoneCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Done"](summary="Finished successfully.")  # WARNING
            self.closure_tools["Done"](summary="Finished successfully.")  # close
            return "Run complete."

    adapter = DoneCallingAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    cmd = node(make_state())

    assert cmd.goto == END
    assert cmd.update["done"] is True
    assert "Finished successfully." in (cmd.update.get("last_report") or "")


def test_strategizer_delegate_returns_task_id():
    """Delegate() returns a TASK-xxxxxxxx ID and starts a background thread."""
    import threading
    import time
    from a3dasm._src.nodes import StrategizerNode

    received_ids: list[str] = []
    worker_started = threading.Event()
    worker_finished = threading.Event()

    class SlowWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            worker_started.set()
            time.sleep(0.05)
            try:
                return (
                    "## Report\n### Actions taken\nDone.\n"
                    "### Files touched\n(none)\n"
                    "### Conclusions\nExperiment A completed successfully.\n"
                    "### Numbers\nn: 1, mean: 0.42, std: 0.01"
                )
            finally:
                worker_finished.set()

    class DelegateCallingAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Delegate"](
                target="implementer",
                intent="Run experiment A.",
                expected_report="Report results.",
            )
            received_ids.append(result)
            # Wait for the worker to actually finish, not a fixed guess (a
            # loaded CI runner blew past a bare 0.1s sleep, so Done() hit a
            # still-pending delegation and routed back instead of to END). A
            # short settle after the finish event lets the node record
            # completion before Done is called.
            worker_started.wait(timeout=5)
            worker_finished.wait(timeout=5)
            time.sleep(0.5)
            self.closure_tools["Done"](summary="All done.")  # first: warning
            self.closure_tools["Done"](summary="All done.")  # second: accepted
            return "Done."

    adapter = DelegateCallingAdapter()
    spec = _minimal_spec()
    worker = SlowWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    cmd = node(make_state())

    assert cmd.goto == END
    assert received_ids and re.search(r"D[0-9]{3}", received_ids[0])


def test_done_blocked_while_delegation_pending():
    """Done returns a soft 2-option nudge (not a hard error) when called while
    a delegation is still Working — it still refuses to close, but offers the
    keep-working / wait (GetStatus) paths (cancel dropped from production)."""
    from a3dasm._src.nodes import StrategizerNode

    results: list[str] = []

    class SlowWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            import time
            time.sleep(10)  # never finishes in test timeframe
            return "## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"

    class DelegateThenDoneAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](
                target="implementer",
                intent="Run experiment.",
                expected_report="Report back.",
            )
            results.append(self.closure_tools["Done"](summary="All done."))
            # Fall through to Done without waiting → should default to END
            self.closure_tools["Done"](summary="forced")
            return "Done."

    adapter = DelegateThenDoneAdapter()
    spec = _minimal_spec()
    worker = SlowWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(make_state())

    # Done should have refused with the soft 2-option nudge (not a hard error)
    assert results and "still running" in results[0]
    assert "GetStatus" in results[0]  # offers the wait + GetStatus path
    assert "CancelDelegation" not in results[0]  # cancel dropped from production
    assert not results[0].lstrip().startswith("ERROR:")  # soft, not an error


def test_strategizer_increments_delegation_count():
    """total_delegations reflects all fired delegations after the run completes."""
    import time
    from a3dasm._src.nodes import StrategizerNode

    class FastWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            return "## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"

    class DelegateCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](target="implementer", intent="task", expected_report="report")
            time.sleep(0.1)  # let worker finish
            self.closure_tools["Done"](summary="done")  # first: warning
            self.closure_tools["Done"](summary="done")  # second: accepted
            return "Done."

    adapter = DelegateCallingAdapter()
    spec = _minimal_spec()
    worker = FastWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    state = make_state(total_delegations=3)
    cmd = node(state)

    # 3 existing + 1 new delegation from registry
    assert cmd.update["total_delegations"] == 4


def test_strategizer_no_routing_tool_defaults_to_done():
    """StrategizerNode loops back with a re-prompt when no routing tool is called.

    Adapted: the old behaviour unconditionally ended the run when no Done() was
    called.  The new route-aware behaviour re-prompts the LLM (up to 3 times)
    so that an LLM that self-acquits in prose is caught before it writes
    solution.md.  The test now asserts the loopback on the first attempt.
    """
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter(response="Final analysis complete.")
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    cmd = node(make_state())

    # First attempt — loops back with a diagnostic message
    assert cmd.goto == "strategizer"
    assert node._finish_attempts == 1
    messages = cmd.update.get("messages", [])
    human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    assert any("without an accepted Done" in m.content for m in human_msgs)


def test_strategizer_delegate_invalid_target_returns_error():
    """Delegate returns an error string for unknown targets."""
    from a3dasm._src.nodes import StrategizerNode

    errors = []

    class BadTargetAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Delegate"](
                target="nonexistent",
                intent="task",
                expected_report="report",
            )
            errors.append(result)
            return "done"

    adapter = BadTargetAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    node(make_state())

    assert errors and "ERROR" in errors[0]


# ---------------------------------------------------------------------------
# ImplementerNode tests
# ---------------------------------------------------------------------------


def test_implementer_returns_to_caller():
    """ImplementerNode routes back to return_to from state."""
    from a3dasm._src.nodes import ImplementerNode

    adapter = StubAdapter(response="## Report\nDone. Results: 42.")
    node = ImplementerNode(adapter)
    cmd = node(make_state(return_to="strategizer"))

    assert cmd.goto == "strategizer"
    assert "## Report" in cmd.update.get("last_report", "")


def test_implementer_stores_response_as_last_report():
    """ImplementerNode stores full response text in last_report."""
    from a3dasm._src.nodes import ImplementerNode

    response = "## Report\n### Actions taken\nRan code.\n### Conclusions\nResult: 3.14"
    adapter = StubAdapter(response=response)
    node = ImplementerNode(adapter)
    cmd = node(make_state(return_to="strategizer"))

    assert cmd.update["last_report"] == response


def test_implementer_accumulates_evals_when_report_evals_called():
    """ImplementerNode adds ReportEvals count to state evals_used."""
    from a3dasm._src.nodes import ImplementerNode

    class ReportEvalsCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["ReportEvals"](count=1500)
            return "## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 1500"

    adapter = ReportEvalsCallingAdapter()
    node = ImplementerNode(adapter)
    state = make_state(return_to="strategizer")
    state["evals_used"] = 100
    cmd = node(state)

    assert cmd.update["evals_used"] == 1600


def test_implementer_evals_zero_when_report_evals_not_called():
    """ImplementerNode adds 0 to evals_used when ReportEvals is not called."""
    from a3dasm._src.nodes import ImplementerNode

    adapter = StubAdapter(response="## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0")
    node = ImplementerNode(adapter)
    state = make_state(return_to="strategizer")
    state["evals_used"] = 42
    cmd = node(state)

    assert cmd.update["evals_used"] == 42


def test_strategizer_delegate_includes_expected_report_in_message():
    """Delegate expected_report appears in the task message sent to the worker."""
    import time
    from a3dasm._src.nodes import StrategizerNode

    received_messages: list[list] = []

    class CapturingWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            received_messages.append(messages)
            return "## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"

    class DelegateCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](
                target="implementer",
                intent="Run the experiment.",
                expected_report="Must produce workspace/pipeline.py",
            )
            time.sleep(0.1)  # let worker finish
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = DelegateCallingAdapter()
    spec = _minimal_spec()
    worker = CapturingWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(make_state())

    assert received_messages, "Worker was never called"
    task_content = received_messages[0][0]["content"]
    assert "workspace/pipeline.py" in task_content
    assert "Required deliverables" in task_content


def test_strategizer_delegate_prepends_edge_preamble():
    """Edge preamble is prepended to the task message when the edge has one."""
    import time
    from a3dasm._src.nodes import StrategizerNode

    class A(Agent):
        role = "strategizer"
        # GetStatus/CancelDelegation are opt-in (plug-and-play) post-audit; the
        # test strategizer opts in so behaviour tests still exercise them even
        # though production agents no longer grant them.
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote",
                           "GetStatus", "CancelDelegation", "HypothesisPropose", "HypothesisUpdate", "HypothesisList", "HypothesisGet", "LinkFalsificationAttempt", "MilestoneList", "MilestonePropose", "MilestoneComplete", "MilestoneSkip", "RecallStore", "QueryStore"})
        description = "Test strategizer."

    class B(Agent):
        description = "Test implementer."

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer", preamble="PREAMBLE TEXT"),),
        entry="strategizer",
    )

    received_messages: list[list] = []

    class CapturingWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            received_messages.append(messages)
            return "## Report\n### Actions taken\nDone.\n### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"

    class DelegateCallingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](
                target="implementer",
                intent="Do work.",
                expected_report="",
            )
            time.sleep(0.1)  # let worker finish
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = DelegateCallingAdapter()
    worker = CapturingWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(make_state())

    assert received_messages, "Worker was never called"
    task_content = received_messages[0][0]["content"]
    assert "PREAMBLE TEXT" in task_content


# ---------------------------------------------------------------------------
# Parallel fan-out tests
# ---------------------------------------------------------------------------


def test_parallel_two_delegations_both_complete():
    """Strategizer can fire two delegations concurrently; both complete and are counted."""
    import time
    from a3dasm._src.nodes import StrategizerNode

    call_log: list[str] = []

    class LoggingWorkerAdapter(StubAdapter):
        def __init__(self, name: str) -> None:
            super().__init__()
            self._name = name

        def invoke(self, messages: list) -> str:
            call_log.append(self._name)
            time.sleep(0.05)
            return (
                f"## Report\n### Actions taken\nDone {self._name}.\n"
                f"### Files touched\n(none)\n### Conclusions\n{self._name} completed successfully.\n### Numbers\nn: 1"
            )

    class TwoDelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](target="worker_a", intent="Task A", expected_report="")
            self.closure_tools["Delegate"](target="worker_b", intent="Task B", expected_report="")
            # Let both workers finish. Each sleeps 0.05s, so this margin is a
            # 3x safety factor under light load — not enough on a contended CI
            # runner (observed flake: macos-latest/3.13, 2026-07-13, "assert
            # cmd.goto == END" got 'strategizer' instead — a delegation was
            # still pending when Done() fired). Bumped to a much wider margin;
            # this polls a fixed sleep rather than a completion signal, so it
            # is still probabilistic, just far less likely to flake.
            time.sleep(1.0)
            result = self.closure_tools["Done"](summary="Both done.")  # first: warning
            assert "ERROR" not in result, f"Done() failed: {result}"
            self.closure_tools["Done"](summary="Both done.")  # second: accepted
            return "Done."

    class A(Agent):
        role = "strategizer"
        # GetStatus/CancelDelegation are opt-in (plug-and-play) post-audit; the
        # test strategizer opts in so behaviour tests still exercise them even
        # though production agents no longer grant them.
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote",
                           "GetStatus", "CancelDelegation", "HypothesisPropose", "HypothesisUpdate", "HypothesisList", "HypothesisGet", "LinkFalsificationAttempt", "MilestoneList", "MilestonePropose", "MilestoneComplete", "MilestoneSkip", "RecallStore", "QueryStore"})
        description = "Test strategizer."

    class B(Agent):
        description = "Test worker."

    spec = Graph(
        nodes={"strategizer": A(), "worker_a": B(), "worker_b": B()},
        edges=(Edge("strategizer", "worker_a"), Edge("strategizer", "worker_b")),
        entry="strategizer",
    )

    adapter = TwoDelegateAdapter()
    node = StrategizerNode(
        adapter, name="strategizer",
        outgoing=["worker_a", "worker_b"], spec=spec,
        worker_adapters={
            "worker_a": LoggingWorkerAdapter("worker_a"),
            "worker_b": LoggingWorkerAdapter("worker_b"),
        },
    )
    cmd = node(make_state())

    assert cmd.goto == END
    assert cmd.update["done"] is True
    assert cmd.update["total_delegations"] == 2  # 0 existing + 2 from registry
    assert set(call_log) == {"worker_a", "worker_b"}


def test_get_status_returns_working_then_done():
    """GetStatus returns 'Working' while the delegation is running, then 'Done'."""
    import time
    from a3dasm._src.nodes import StrategizerNode

    status_snapshots: list[str] = []

    class SlowWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            time.sleep(0.1)
            return (
                "## Report\n### Actions taken\nDone.\n"
                "### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"
            )

    class PollAdapter(StubAdapter):
        def invoke(self, messages):
            task_id_msg = self.closure_tools["Delegate"](
                target="implementer", intent="Work.", expected_report=""
            )
            # Extract D### delegation ID from the return string
            import re
            m = re.search(r"D[0-9]{3}", task_id_msg)
            assert m, f"No task ID found in: {task_id_msg!r}"
            task_id = m.group()

            # Poll immediately — should be Working
            status_snapshots.append(self.closure_tools["GetStatus"](task_id))
            # Wait and poll again — should be Done
            time.sleep(0.5)
            status_snapshots.append(self.closure_tools["GetStatus"](task_id))
            self.closure_tools["Done"](summary="polled")
            return "Done."

    adapter = PollAdapter()
    spec = _minimal_spec()
    worker = SlowWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(make_state())

    assert status_snapshots[0].startswith("Working")  # no notification yet
    assert "Done" in status_snapshots[1]               # notification prefix + Done report


def test_registry_cleared_between_runs():
    """Registry and ask_count are reset on each __call__, preventing cross-run pollution."""
    import time
    from a3dasm._src.nodes import StrategizerNode

    call_count = [0]

    class FastWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            return (
                "## Report\n### Actions taken\nTask completed.\n### Files touched\n(none)\n"
                "### Conclusions\nAll experiments ran successfully.\n### Numbers\nn: 1, result: 0.42"
            )

    class OneDelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](target="implementer", intent="task", expected_report="")
            time.sleep(0.1)
            self.closure_tools["Done"](summary="done")  # first: warning
            self.closure_tools["Done"](summary="done")  # second: accepted
            call_count[0] += 1
            return "Done."

    adapter = OneDelegateAdapter()
    spec = _minimal_spec()
    worker = FastWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )

    # First run
    cmd1 = node(make_state(total_delegations=0))
    assert cmd1.update["total_delegations"] == 1

    # Second run on the same node — registry must be fresh
    cmd2 = node(make_state(total_delegations=0))
    assert cmd2.update["total_delegations"] == 1  # not 2


def test_get_status_unknown_id_returns_error():
    """GetStatus on an unknown ID returns a clear ERROR string."""
    from a3dasm._src.nodes import StrategizerNode

    errors: list[str] = []

    class PollUnknownAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["GetStatus"]("D999")
            errors.append(result)
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = PollUnknownAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    node(make_state())

    assert errors and errors[0].startswith("ERROR")
    assert "D999" in errors[0]


def test_errored_status_contains_traceback():
    """When a worker raises, GetStatus returns 'Errored:\\n<traceback>' with enough info."""
    import re
    import time
    from a3dasm._src.nodes import StrategizerNode

    status_seen: list[str] = []
    worker_done = threading.Event()

    class CrashingWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            raise ValueError("pool.csv not found at workspace/pool.csv")

    class PollErrorAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Delegate"](
                target="implementer", intent="task", expected_report=""
            )
            import re as _re
            m = _re.search(r"D[0-9]{3}", result); task_id = m.group() if m else None; assert task_id, f"No D### ID in: {result!r}"
            # Poll until resolved
            for _ in range(50):
                status = self.closure_tools["GetStatus"](task_id)
                if not status.startswith("Working"):
                    status_seen.append(status)
                    break
                time.sleep(0.02)
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = PollErrorAdapter()
    spec = _minimal_spec()
    worker = CrashingWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(make_state())

    assert status_seen, "Never got a non-Working status"
    msg = status_seen[0]
    assert "Errored:" in msg  # may have notification prefix
    # Must contain the exception type and message for diagnostics (B1)
    assert "ValueError" in msg
    assert "pool.csv" in msg


def test_delegation_still_working_returns_working_status():
    """GetStatus returns 'Working' for a running delegation regardless of elapsed time."""
    import time
    from a3dasm._src.nodes import StrategizerNode

    status_seen: list[str] = []
    worker_started = threading.Event()
    allow_finish = threading.Event()

    class HeldWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            worker_started.set()
            allow_finish.wait(timeout=5)
            return (
                "## Report\n### Actions taken\nDone.\n"
                "### Files touched\n(none)\n### Conclusions\nTask completed successfully.\n### Numbers\nn: 1"
            )

    class PollThenReleaseAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Delegate"](
                target="implementer", intent="task", expected_report=""
            )
            import re as _re
            m = _re.search(r"D[0-9]{3}", result)
            task_id = m.group() if m else None
            assert task_id, f"No D### ID in: {result!r}"
            worker_started.wait(timeout=2)
            # Poll while worker is still held — must return Working, not Timeout
            status = self.closure_tools["GetStatus"](task_id)
            status_seen.append(status)
            # Release the worker and wait for it to finish
            allow_finish.set()
            time.sleep(0.15)
            self.closure_tools["Done"](summary="done")  # first: warning
            self.closure_tools["Done"](summary="done")  # second: accepted
            return "Done."

    adapter = PollThenReleaseAdapter()
    spec = _minimal_spec()
    worker = HeldWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    # Even with a tiny budget, GetStatus must return Working (not Timeout)
    state = make_state()
    state["budget_seconds"] = 0.013
    state["start_time"] = time.time()  # realistic — just over budget by a hair

    cmd = node(state)
    assert cmd.goto == END
    assert status_seen and status_seen[0].startswith("Working"), (
        f"Expected 'Working...', got: {status_seen[0]!r}"
    )


# ---------------------------------------------------------------------------
# _to_adapter_messages tests
# ---------------------------------------------------------------------------


def test_to_adapter_messages_converts_human_and_ai():
    """_to_adapter_messages converts HumanMessage and AIMessage correctly."""
    from a3dasm._src.nodes import _to_adapter_messages

    msgs = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
        HumanMessage(content="Bye"),
    ]
    result = _to_adapter_messages(msgs)

    assert len(result) == 3
    assert result[0] == {"role": "user", "content": "Hello"}
    assert result[1] == {"role": "ai", "content": "Hi there"}
    assert result[2] == {"role": "user", "content": "Bye"}


# ---------------------------------------------------------------------------
# Hypothesis ledger integration tests
# ---------------------------------------------------------------------------

import re
import time as _time
import json as _json

from a3dasm._src.hypothesis_ledger import HypothesisLedger


def _ledger_spec():
    return _minimal_spec()


def test_delegate_id_is_sequential(tmp_path):
    """Delegation IDs are D001, D002, D003 within a run."""
    received_ids = []

    class CaptureAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Hypothesis for sequential ID test",
                falsification_criterion="any counter",
                prediction="none found",
                prior=0.5,
            )
            r1 = self.closure_tools["Delegate"](
                target="implementer", intent="task A", expected_report="",
                hypothesis_ids=["H1"],
            )
            r2 = self.closure_tools["Delegate"](
                target="implementer", intent="task B", expected_report="",
                hypothesis_ids=["H1"],
            )
            received_ids.extend([r1, r2])
            _time.sleep(0.3)
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    adapter = CaptureAdapter()
    spec = _ledger_spec()
    worker = StubAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
        notes_dir=tmp_path,
    )
    node(make_state())
    assert received_ids
    assert re.search(r"D\d{3}", received_ids[0]), f"Expected D### in {received_ids[0]!r}"
    assert re.search(r"D\d{3}", received_ids[1]), f"Expected D### in {received_ids[1]!r}"


def test_delegate_requires_hypothesis_ids_when_ledger_present(tmp_path):
    """Delegate() with empty hypothesis_ids returns ERROR when ledger is active."""
    error_result = []

    class EmptyHypoAdapter(StubAdapter):
        def invoke(self, messages):
            r = self.closure_tools["Delegate"](
                target="implementer", intent="task", expected_report="",
                hypothesis_ids=[],
            )
            error_result.append(r)
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    adapter = EmptyHypoAdapter()
    spec = _ledger_spec()
    worker = StubAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
        notes_dir=tmp_path,
    )
    node(make_state())
    assert error_result and error_result[0].startswith("ERROR:")


def test_delegate_wraps_bare_string_hypothesis_id(tmp_path):
    """hypothesis_ids='H1' must become ['H1'], never ['H','1']."""
    from a3dasm._src.nodes import StrategizerNode

    captured = []

    class ProposeAndDelegate(StubAdapter):
        def invoke(self, messages):
            # Propose H1 first so the ledger knows about it
            self.closure_tools["HypothesisPropose"](
                statement="Black-box is unimodal",
                falsification_criterion="Any y<-1",
                prediction="y>=-1 everywhere",
                prior=0.5,
            )
            # Pass hypothesis_ids as a bare string (not a list)
            r = self.closure_tools["Delegate"](
                target="implementer",
                intent="test string wrapping",
                expected_report="",
                hypothesis_ids="H1",
            )
            captured.append(r)
            _time.sleep(0.3)
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = ProposeAndDelegate()
    spec = _ledger_spec()
    worker = StubAdapter()
    node = StrategizerNode(
        adapter,
        name="strategizer",
        outgoing=["implementer"],
        spec=spec,
        worker_adapters={"implementer": worker},
        notes_dir=tmp_path,
    )
    node(make_state())
    assert captured, "No result captured from Delegate"
    assert not captured[0].startswith("ERROR:"), (
        f"Unexpected ERROR: {captured[0]!r}"
    )
    # The registry entry must have hypothesis_ids == ["H1"]
    reg_entry = list(node._registry.values())[0]
    assert reg_entry["hypothesis_ids"] == ["H1"], (
        f"hypothesis_ids shredded: {reg_entry['hypothesis_ids']!r}"
    )


def test_delegate_rejects_unknown_hypothesis_id(tmp_path):
    """Delegate with an unknown ID returns ERROR naming valid IDs."""
    from a3dasm._src.nodes import StrategizerNode

    captured = []

    class ProposeAndBadDelegate(StubAdapter):
        def invoke(self, messages):
            # Propose only H1
            self.closure_tools["HypothesisPropose"](
                statement="Black-box is unimodal",
                falsification_criterion="Any y<-1",
                prediction="y>=-1 everywhere",
                prior=0.5,
            )
            # Delegate with an unknown ID H7
            r = self.closure_tools["Delegate"](
                target="implementer",
                intent="task",
                expected_report="",
                hypothesis_ids=["H7"],
            )
            captured.append(r)
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = ProposeAndBadDelegate()
    spec = _ledger_spec()
    worker = StubAdapter()
    node = StrategizerNode(
        adapter,
        name="strategizer",
        outgoing=["implementer"],
        spec=spec,
        worker_adapters={"implementer": worker},
        notes_dir=tmp_path,
    )
    node(make_state())
    assert captured, "No result captured"
    assert captured[0].startswith("ERROR:"), (
        f"Expected ERROR, got: {captured[0]!r}"
    )
    # Error message must mention valid IDs (H1)
    assert "H1" in captured[0], (
        f"Valid IDs not in error message: {captured[0]!r}"
    )


def test_delegate_records_falsification_flag(tmp_path):
    """Delegate with is_falsification_attempt=True records it in DelegationLog."""
    from a3dasm._src.nodes import StrategizerNode
    from a3dasm._src.delegation_log import DelegationLog

    class ProposeAndFalsifyDelegate(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Black-box is unimodal",
                falsification_criterion="Any y<-1",
                prediction="y>=-1 everywhere",
                prior=0.5,
            )
            self.closure_tools["Delegate"](
                target="implementer",
                intent="falsification task",
                expected_report="",
                hypothesis_ids=["H1"],
                is_falsification_attempt=True,
                wait=True,
            )
            self.closure_tools["Done"](summary="done")
            return "Done."

    jsonl_path = tmp_path / "delegation_log.jsonl"
    delegation_log = DelegationLog(jsonl_path)
    adapter = ProposeAndFalsifyDelegate()
    spec = _ledger_spec()
    worker = StubAdapter()
    node = StrategizerNode(
        adapter,
        name="strategizer",
        outgoing=["implementer"],
        spec=spec,
        worker_adapters={"implementer": worker},
        notes_dir=tmp_path,
        delegation_log=delegation_log,
    )
    node(make_state())
    records = delegation_log.query_all()
    assert records, "No records in delegation log"
    assert records[-1]["is_falsification_attempt"] is True, (
        f"Expected is_falsification_attempt=True, got: {records[-1]!r}"
    )


def test_delegate_injects_workspace_subfolder_in_task(tmp_path):
    """Task message contains <workspace_subfolder>D001/</workspace_subfolder>."""
    task_messages = []

    class CapturingWorker(StubAdapter):
        def invoke(self, messages):
            task_messages.extend(messages)
            return "## Report\n\n### Actions taken\n- done\n\n### Files touched\n- none\n\n### Conclusions\nok\n\n### Numbers\nevals: 0"

    class DelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Workspace subfolder test hypothesis",
                falsification_criterion="any counter",
                prediction="none found",
                prior=0.5,
            )
            self.closure_tools["Delegate"](
                target="implementer", intent="test", expected_report="",
                hypothesis_ids=["H1"],
            )
            _time.sleep(0.3)
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    adapter = DelegateAdapter()
    spec = _ledger_spec()
    worker = CapturingWorker()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
        notes_dir=tmp_path,
    )
    node(make_state())
    assert task_messages
    full_task = " ".join(str(m) for m in task_messages)
    assert "workspace_subfolder" in full_task
    assert "D001/" in full_task


def test_delegate_writes_delegation_jsonl_on_done(tmp_path):
    """delegation_log.jsonl is written when a delegation completes."""
    from a3dasm._src.delegation_log import DelegationLog

    class DelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="First hypothesis for jsonl test",
                falsification_criterion="any counter",
                prediction="none found",
                prior=0.5,
            )
            self.closure_tools["HypothesisPropose"](
                statement="Second hypothesis for jsonl test",
                falsification_criterion="any counter 2",
                prediction="none found 2",
                prior=0.5,
            )
            self.closure_tools["Delegate"](
                target="implementer", intent="do analysis", expected_report="",
                hypothesis_ids=["H1", "H2"],
            )
            _time.sleep(0.3)
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    jsonl_path = tmp_path / "delegation_log.jsonl"
    delegation_log = DelegationLog(jsonl_path)
    adapter = DelegateAdapter()
    spec = _ledger_spec()
    worker = StubAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
        notes_dir=tmp_path,
        delegation_log=delegation_log,
    )
    node(make_state())
    assert jsonl_path.exists(), "delegation_log.jsonl was not created"
    records = delegation_log.query_all()   # collapsed last-wins (one per id)
    assert len(records) == 1
    rec = records[0]
    assert rec["from_node"] == "strategizer"
    assert rec["to_node"] == "implementer"
    assert rec["hypothesis_ids"] == ["H1", "H2"]
    assert rec["status"] in ("DONE", "FAILED")


def test_hypothesis_propose_via_strategizer_closure(tmp_path):
    """HypothesisPropose() closure creates an entry in hypotheses.json."""
    proposed_ids = []

    class ProposeAdapter(StubAdapter):
        def invoke(self, messages):
            h_id = self.closure_tools["HypothesisPropose"](
                statement="Thin longerons buckle first",
                falsification_criterion="any test showing they do not",
                prediction="sigma_crit drops below threshold",
                prior=0.6,
            )
            proposed_ids.append(h_id)
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    adapter = ProposeAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": StubAdapter()},
        notes_dir=tmp_path,
    )
    node(make_state())
    assert proposed_ids and not proposed_ids[0].startswith("ERROR:")
    hyp_path = tmp_path / "hypotheses.json"
    assert hyp_path.exists()
    data = _json.loads(hyp_path.read_text())
    assert proposed_ids[0] in data


def test_hypothesis_update_injects_triggered_by(tmp_path):
    """HypothesisUpdate injects the last completed delegation ID as triggered_by."""
    from a3dasm._src.nodes import StrategizerNode

    class UpdateAdapter(StubAdapter):
        def invoke(self, messages):
            h_id = self.closure_tools["HypothesisPropose"](
                statement="Test hyp claim",
                falsification_criterion="any counter-example",
                prediction="none found in sweep",
                prior=0.6,
            )
            result = self.closure_tools["Delegate"](
                target="implementer", intent="test", expected_report="",
                hypothesis_ids=[h_id],
            )
            import re as _re
            d_id = _re.search(r"D\d{3}", result).group()
            _time.sleep(0.3)
            self.closure_tools["HypothesisUpdate"](
                hypothesis_id=h_id,
                status="FALSIFIED",
                comment="disproved",
                evidence={"delegation": d_id},
                posterior=0.1,
            )
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = UpdateAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": StubAdapter()},
        notes_dir=tmp_path,
    )
    node(make_state())
    data = _json.loads((tmp_path / "hypotheses.json").read_text())
    h_id = list(data.keys())[0]
    last_entry = data[h_id]["status_log"][-1]
    assert last_entry["status"] == "FALSIFIED"
    assert last_entry["triggered_by"] is not None
    assert re.match(r"D\d{3}", last_entry["triggered_by"])


def test_max_three_open_hypothesis_guard(tmp_path):
    """HypothesisPropose returns ERROR when 3 OPEN hypotheses already exist."""
    error_seen = []

    from a3dasm._src.nodes import StrategizerNode

    class MaxAdapter(StubAdapter):
        def invoke(self, messages):
            _kw = dict(
                falsification_criterion="fc",
                prediction="pred",
                prior=0.5,
            )
            self.closure_tools["HypothesisPropose"](statement="A", **_kw)
            self.closure_tools["HypothesisPropose"](statement="B", **_kw)
            self.closure_tools["HypothesisPropose"](statement="C", **_kw)
            r = self.closure_tools["HypothesisPropose"](statement="D", **_kw)
            error_seen.append(r)
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = MaxAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": StubAdapter()},
        notes_dir=tmp_path,
    )
    node(make_state())
    # Past the open ceiling the 4th proposal is REGISTERED with a tip, not
    # blocked or two-shot — tracking several at once (e.g. one per design) is
    # allowed; the tip just advises closing settled ones.
    assert error_seen and error_seen[0].startswith("H")
    assert "[NUDGE]" in error_seen[0]


def test_worker_write_rejected_outside_delegation_subfolder(tmp_path):
    """Worker Write is rejected for paths outside workspace/{delegation_id}/."""
    write_results = []

    class WritingWorker(StubAdapter):
        def invoke(self, messages):
            # Path traversal attack: ../D000/ escapes the delegation subfolder
            r = self.closure_tools["Write"]("../D000/cross.txt", "bad data")
            write_results.append(r)
            return "## Report\n\n### Actions taken\n- tried write\n\n### Files touched\n- none\n\n### Conclusions\ntested\n\n### Numbers\nevals: 0"

    class DelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Write sandbox rejection test",
                falsification_criterion="any counter",
                prediction="none found",
                prior=0.5,
            )
            self.closure_tools["Delegate"](
                target="implementer", intent="test write", expected_report="",
                hypothesis_ids=["H1"],
            )
            _time.sleep(0.3)
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    study_tmp = tmp_path / "study"
    study_tmp.mkdir()
    adapter = DelegateAdapter()
    spec = _ledger_spec()
    worker = WritingWorker()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
        notes_dir=tmp_path,
        study_dir=study_tmp,
    )
    node(make_state(study_dir=str(study_tmp)))
    assert write_results and "ERROR" in write_results[0]


def test_worker_write_allowed_inside_delegation_subfolder(tmp_path):
    """Worker Write succeeds for paths inside workspace/{delegation_id}/."""
    write_results = []

    class WritingWorker(StubAdapter):
        def invoke(self, messages):
            import re as _re
            content = " ".join(str(m) for m in messages)
            m = _re.search(r"delegations/([^/]+)/", content)
            subfolder = m.group(1) if m else "D001"
            r = self.closure_tools["Write"](f"{subfolder}/result.csv", "col,val\n1,2")
            write_results.append(r)
            return "## Report\n\n### Actions taken\n- wrote result\n\n### Files touched\n- result.csv\n\n### Conclusions\ndone\n\n### Numbers\nevals: 0"

    class DelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Write sandbox allow test",
                falsification_criterion="any counter",
                prediction="none found",
                prior=0.5,
            )
            self.closure_tools["Delegate"](
                target="implementer", intent="test write", expected_report="",
                hypothesis_ids=["H1"],
            )
            _time.sleep(0.3)
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    study_tmp = tmp_path / "study"
    study_tmp.mkdir()
    adapter = DelegateAdapter()
    spec = _ledger_spec()
    worker = WritingWorker()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
        notes_dir=tmp_path,
        study_dir=study_tmp,
    )
    node(make_state(study_dir=str(study_tmp)))
    assert write_results and "ERROR" not in write_results[0]


def test_worker_write_strips_redundant_delegation_prefix(tmp_path):
    """A path that already includes the delegation id must NOT nest D###/D###/.

    The prompt calls the sandbox 'your D### subfolder', so agents prefix paths
    with it (run-3 friction, flagged by 3 agents). The sandbox is already rooted
    at {delegation_id}/, so the prefix is absorbed — both 'D001/a.csv' and
    'b.csv' land directly under .../delegations/D001/."""
    written = []

    class WritingWorker(StubAdapter):
        def invoke(self, messages):
            import re as _re
            did = (_re.search(r"delegations/([^/]+)/",
                              " ".join(str(m) for m in messages))
                   or _re.search(r"<workspace_subfolder>([^/]+)/",
                                 " ".join(str(m) for m in messages)))
            sub = did.group(1) if did else "D001"
            written.append(self.closure_tools["Write"](f"{sub}/a.csv", "x"))
            written.append(self.closure_tools["Write"]("b.csv", "y"))
            return ("## Report\n\n### Actions taken\n- wrote\n\n"
                    "### Files touched\n- a.csv\n\n### Conclusions\nok\n\n"
                    "### Numbers\nevals: 0")

    class DelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Write prefix test", falsification_criterion="c",
                prediction="p", prior=0.5)
            self.closure_tools["Delegate"](
                target="implementer", intent="test", expected_report="",
                hypothesis_ids=["H1"])
            _time.sleep(0.3)
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    study_tmp = tmp_path / "study"
    study_tmp.mkdir()
    node = StrategizerNode(
        DelegateAdapter(), name="strategizer", outgoing=["implementer"],
        spec=_ledger_spec(), worker_adapters={"implementer": WritingWorker()},
        notes_dir=tmp_path, study_dir=study_tmp)
    node(make_state(study_dir=str(study_tmp)))

    assert len(written) == 2 and all("ERROR" not in w for w in written)
    # neither write doubled the delegation id
    for w in written:
        assert "/D001/D001/" not in w, f"double-nested: {w}"
    assert written[0].replace("\\", "/").endswith("/D001/a.csv")
    assert written[1].replace("\\", "/").endswith("/D001/b.csv")


# ---------------------------------------------------------------------------
# Blindspot 1: _accumulate_usage() unit tests
# ---------------------------------------------------------------------------


def test_accumulate_usage_sums_correctly(tmp_path):
    """_accumulate_usage correctly sums token counts and ignores None cost."""
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=tmp_path,
    )

    node._accumulate_usage({"input_tokens": 100, "output_tokens": 50, "total_cost_usd": 0.01})
    node._accumulate_usage({"input_tokens": 200, "output_tokens": 30, "total_cost_usd": None})

    assert node._token_totals["input_tokens"] == 300
    assert node._token_totals["output_tokens"] == 80
    assert node._token_totals["total_cost_usd"] == 0.01


def test_accumulate_usage_thread_safe(tmp_path):
    """_accumulate_usage is thread-safe under concurrent calls."""
    import threading as _threading
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=tmp_path,
    )

    threads = [
        _threading.Thread(target=node._accumulate_usage, args=({"input_tokens": 10},))
        for _ in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert node._token_totals["input_tokens"] == 100


# ---------------------------------------------------------------------------
# Blindspot 4: Write sandbox edge cases
# ---------------------------------------------------------------------------


def _make_sandboxed_write_node(tmp_path):
    """Return a StrategizerNode + worker where the worker's Write is sandboxed."""
    write_results = []

    class CapturingWorker(StubAdapter):
        """Worker that delegates the actual write call via closure and captures result."""
        _pending_write = None

        def invoke(self, messages):
            if CapturingWorker._pending_write is not None:
                path, body = CapturingWorker._pending_write
                r = self.closure_tools["Write"](path, body)
                write_results.append(r)
            return (
                "## Report\n\n### Actions taken\n- tried\n\n"
                "### Files touched\n- none\n\n"
                "### Conclusions\ntested\n\n### Numbers\nevals: 0"
            )

    return CapturingWorker, write_results


def test_worker_write_rejects_absolute_path(tmp_path):
    """Write with an absolute path outside the sandbox returns ERROR."""
    write_results = []

    class AbsPathWorker(StubAdapter):
        def invoke(self, messages):
            r = self.closure_tools["Write"]("/etc/passwd", "bad")
            write_results.append(r)
            return (
                "## Report\n\n### Actions taken\n- tried\n\n"
                "### Files touched\n- none\n\n"
                "### Conclusions\ntested\n\n### Numbers\nevals: 0"
            )

    class DelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Abs path sandbox test",
                falsification_criterion="any counter",
                prediction="none found",
                prior=0.5,
            )
            self.closure_tools["Delegate"](
                target="implementer", intent="test", expected_report="",
                hypothesis_ids=["H1"],
            )
            _time.sleep(0.3)
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    study_tmp = tmp_path / "study"
    study_tmp.mkdir()
    adapter = DelegateAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": AbsPathWorker()},
        notes_dir=tmp_path,
        study_dir=study_tmp,
    )
    node(make_state(study_dir=str(study_tmp)))
    assert write_results, "Write was never called"
    assert write_results[0].startswith("ERROR")


def test_worker_write_rejects_empty_path(tmp_path):
    """Write with an empty path string must not raise an exception."""
    write_results = []

    class EmptyPathWorker(StubAdapter):
        def invoke(self, messages):
            try:
                r = self.closure_tools["Write"]("", "data")
                write_results.append(r)
            except Exception as exc:
                write_results.append(f"RAISED: {exc}")
            return (
                "## Report\n\n### Actions taken\n- tried\n\n"
                "### Files touched\n- none\n\n"
                "### Conclusions\ntested\n\n### Numbers\nevals: 0"
            )

    class DelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Empty path sandbox test",
                falsification_criterion="any counter",
                prediction="none found",
                prior=0.5,
            )
            self.closure_tools["Delegate"](
                target="implementer", intent="test", expected_report="",
                hypothesis_ids=["H1"],
            )
            _time.sleep(0.3)
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    study_tmp = tmp_path / "study"
    study_tmp.mkdir()
    adapter = DelegateAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": EmptyPathWorker()},
        notes_dir=tmp_path,
        study_dir=study_tmp,
    )
    node(make_state(study_dir=str(study_tmp)))
    # Must have a result (no uncaught exception)
    assert write_results, "Write was never called or raised before appending"
    # It must not have raised
    assert not write_results[0].startswith("RAISED"), (
        f"Write raised an exception: {write_results[0]}"
    )


def test_worker_write_allows_nested_subdirectory(tmp_path):
    """Write to subdir/output.json inside delegations/D001/ succeeds."""
    write_results = []

    class NestedWriteWorker(StubAdapter):
        def invoke(self, messages):
            import re as _re
            content = " ".join(str(m) for m in messages)
            m = _re.search(r"delegations/([^/]+)/", content)
            subfolder = m.group(1) if m else "D001"
            r = self.closure_tools["Write"](f"{subfolder}/subdir/output.json", '{"result": 1}')
            write_results.append(r)
            return (
                "## Report\n\n### Actions taken\n- wrote nested\n\n"
                "### Files touched\n- output.json\n\n"
                "### Conclusions\ndone\n\n### Numbers\nevals: 0"
            )

    class DelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Nested write sandbox test",
                falsification_criterion="any counter",
                prediction="none found",
                prior=0.5,
            )
            self.closure_tools["Delegate"](
                target="implementer", intent="test", expected_report="",
                hypothesis_ids=["H1"],
            )
            _time.sleep(0.3)
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    study_tmp = tmp_path / "study"
    study_tmp.mkdir()
    adapter = DelegateAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": NestedWriteWorker()},
        notes_dir=tmp_path,
        study_dir=study_tmp,
    )
    node(make_state(study_dir=str(study_tmp)))
    assert write_results, "Write was never called"
    assert "Written:" in write_results[0], f"Expected success, got: {write_results[0]}"
    # The file should physically exist
    written_path = write_results[0].replace("Written: ", "").strip()
    assert _json.loads(open(written_path).read())["result"] == 1


# ---------------------------------------------------------------------------
# Blindspot 5: WriteNote and ReadNote direct tests
# ---------------------------------------------------------------------------


def test_write_note_creates_file_in_notes_dir(tmp_path):
    """WriteNote closure writes .md file to _current_notes_dir."""
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=tmp_path,
    )
    node._current_notes_dir = tmp_path

    result = node.adapter.closure_tools["WriteNote"]("hypotheses.md", "# H1\ncontent")
    assert (tmp_path / "hypotheses.md").exists(), f"File not created; result={result}"
    assert "H1" in (tmp_path / "hypotheses.md").read_text()


def test_write_note_rejects_non_md_extension(tmp_path):
    """WriteNote auto-appends .md when extension is not .md."""
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=tmp_path,
    )
    node._current_notes_dir = tmp_path

    node.adapter.closure_tools["WriteNote"]("script.py", "import os")
    # .py file must NOT exist
    assert not (tmp_path / "script.py").exists(), ".py file should not be created"
    # .md file should exist (auto-append)
    assert (tmp_path / "script.py.md").exists() or (tmp_path / "script.md").exists(), (
        "Expected a .md file to be created with appended extension"
    )


def test_read_note_returns_content(tmp_path):
    """ReadNote closure returns file content from study_dir."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "data.txt").write_text("test content")

    adapter = StubAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=tmp_path,
        study_dir=tmp_path,
    )
    node._current_notes_dir = tmp_path

    result = node.adapter.closure_tools["ReadNote"]("data.txt")
    assert "test content" in result


def test_read_note_returns_not_found_for_missing(tmp_path):
    """ReadNote returns NOT FOUND or ERROR when file does not exist."""
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=tmp_path,
        study_dir=tmp_path,
    )
    node._current_notes_dir = tmp_path

    result = node.adapter.closure_tools["ReadNote"]("does_not_exist.txt")
    assert "NOT FOUND" in result or result.startswith("ERROR")


# ---------------------------------------------------------------------------
# Blindspot 6: delegations.jsonl token fields
# ---------------------------------------------------------------------------


def test_delegation_jsonl_contains_token_fields(tmp_path):
    """delegation_log.jsonl records tokens_in, tokens_out, cost_usd from worker usage."""
    from a3dasm._src.delegation_log import DelegationLog

    class MockWorkerAdapter(StubAdapter):
        last_usage = {"input_tokens": 77, "output_tokens": 33, "total_cost_usd": 0.005}

        def invoke(self, messages):
            return (
                "## Report\n\n### Actions taken\n- done\n\n"
                "### Files touched\n- none\n\n"
                "### Conclusions\nok\n\n### Numbers\nevals: 0"
            )

    class DelegateAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Token fields test hypothesis",
                falsification_criterion="any counter",
                prediction="none found",
                prior=0.5,
            )
            self.closure_tools["Delegate"](
                target="implementer", intent="test usage", expected_report="",
                hypothesis_ids=["H1"],
            )
            _time.sleep(0.3)
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    jsonl_path = tmp_path / "delegation_log.jsonl"
    delegation_log = DelegationLog(jsonl_path)
    adapter = DelegateAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": MockWorkerAdapter()},
        notes_dir=tmp_path,
        delegation_log=delegation_log,
    )
    node(make_state())

    assert jsonl_path.exists(), "delegation_log.jsonl was not created"
    rec = delegation_log.query_all()[0]   # collapsed terminal record (one per id)
    assert rec["tokens_in"] == 77
    assert rec["tokens_out"] == 33
    assert rec["cost_usd"] == 0.005


# ---------------------------------------------------------------------------
# Helpers for new feature tests
# ---------------------------------------------------------------------------


def _spec_with_critic():
    """Three-node graph: strategizer → implementer + strategizer → critic."""

    class S(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote", "HypothesisPropose", "HypothesisUpdate", "HypothesisList", "HypothesisGet", "LinkFalsificationAttempt", "MilestoneList", "MilestonePropose", "MilestoneComplete", "MilestoneSkip", "RecallStore", "QueryStore"})
        description = "Test strategizer."

    class W(Agent):
        description = "Test implementer."

    class C(Agent):
        role = "critic"
        description = "Test adversarial critic."

    return Graph(
        nodes={"strategizer": S(), "implementer": W(), "critic": C()},
        edges=(Edge("strategizer", "implementer"), Edge("strategizer", "critic")),
        entry="strategizer",
    )


class MockCriticAdapter(StubAdapter):
    """Stub adapter that returns a full critique report with a configurable verdict."""

    def __init__(self, verdict: str = "PASS") -> None:
        super().__init__()
        self._verdict = verdict

    def invoke(self, messages: list) -> str:
        return (
            f"## Report\n\n"
            f"### Actions taken\n- Read files\n\n"
            f"### Findings\nNo issues.\n\n"
            f"### Verdict\n{self._verdict}\n\n"
            f"### Numbers\n"
            f"findings_critical: 0\n"
            f"findings_major: 0\n"
            f"findings_minor: 0\n"
            f"verdict: {self._verdict}\n"
        )

    def copy(self):
        fresh = MockCriticAdapter(self._verdict)
        fresh.closure_tools = dict(self.closure_tools)
        return fresh


# ---------------------------------------------------------------------------
# _classify_response tests
# ---------------------------------------------------------------------------


def test_classify_response_uses_custom_sections():
    """_classify_response with required_sections returns None when all present."""
    from a3dasm._src.nodes import _classify_response

    text = "## Report\n\n### Alpha\nsome content\n\n### Beta\nmore content\n" * 5
    result = _classify_response(text, required_sections=["### Alpha", "### Beta"])
    assert result is None


def test_classify_response_missing_custom_section():
    """_classify_response returns diagnosis when a required section is absent."""
    from a3dasm._src.nodes import _classify_response

    text = "## Report\n\n### Alpha\nsome content\n" * 5
    result = _classify_response(text, required_sections=["### Alpha", "### Beta"])
    assert result is not None
    assert isinstance(result, str)


def test_classify_response_falls_back_to_default():
    """_classify_response(text, required_sections=None) uses the 4 default sections."""
    from a3dasm._src.nodes import _classify_response

    text = (
        "## Report\n\n"
        "### Actions taken\n- Did stuff\n\n"
        "### Files touched\n- file.py\n\n"
        "### Conclusions\nAll good.\n\n"
        "### Numbers\nn: 0\n"
    ) * 3  # make it long enough to pass the length check
    result = _classify_response(text, required_sections=None)
    assert result is None


def test_classify_response_allows_extra_content_after_sections():
    """Extra sections beyond the required ones do not cause a failure."""
    from a3dasm._src.nodes import _classify_response

    text = (
        "## Report\n\n"
        "### Alpha\ncontent here\n\n"
        "### Beta\nmore content\n\n"
        "### Extra stuff\nfree form content that is irrelevant\n"
    ) * 4
    result = _classify_response(text, required_sections=["### Alpha", "### Beta"])
    assert result is None


# ---------------------------------------------------------------------------
# Two-shot Done() tests
# ---------------------------------------------------------------------------


def test_done_first_call_returns_warning():
    """Done() first call (no pending delegations) returns a WARNING string."""
    from a3dasm._src.nodes import StrategizerNode

    results: list[str] = []

    class FirstDoneAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Done"](summary="first attempt")
            results.append(result)
            # Call Done() a second time so the run actually closes
            self.closure_tools["Done"](summary="second attempt closes")
            return "Done."

    adapter = FirstDoneAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    node(make_state())

    assert results, "Done() was never called"
    assert results[0].startswith("WARNING"), (
        f"Expected first Done() to return WARNING, got: {results[0]!r}"
    )


def test_done_second_call_no_critic_closes():
    """Done() second call (no critic) sets route kind=done and returns goto=END.

    The first call must return WARNING (not close immediately).  Only the
    second call should actually close.  If the node currently closes on the
    first call, _done_warned will never be set and the first result will NOT
    start with WARNING — that assertion catches the current (pre-feature) state.
    """
    from a3dasm._src.nodes import StrategizerNode
    from langgraph.graph import END

    first_results: list[str] = []

    class TwoDoneAdapter(StubAdapter):
        def invoke(self, messages):
            r = self.closure_tools["Done"](summary="first – warning")
            first_results.append(r)
            self.closure_tools["Done"](summary="second – close")
            return "Done."

    adapter = TwoDoneAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    cmd = node(make_state())

    # First call must warn, not close
    assert first_results and first_results[0].startswith("WARNING"), (
        f"Expected first Done() to WARNING, got: {first_results[0]!r}"
    )
    # Second call closes
    assert cmd.goto == END
    assert node._route.get("kind") == "done"


def test_done_resets_warning_after_new_delegate():
    """_done_warned resets to False when a new Delegate() fires."""
    import time as _t
    from a3dasm._src.nodes import StrategizerNode

    results: list[str] = []

    class FastWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            return (
                "## Report\n### Actions taken\nDone.\n"
                "### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"
            )

    class ResetWarningAdapter(StubAdapter):
        def invoke(self, messages):
            # First Done → WARNING
            r1 = self.closure_tools["Done"](summary="first")
            results.append(r1)
            # Delegate to reset the warning flag
            self.closure_tools["Delegate"](
                target="implementer", intent="reset task", expected_report=""
            )
            _t.sleep(0.15)  # let worker finish
            # Done again after Delegate → should warn again (not close directly)
            r2 = self.closure_tools["Done"](summary="third")
            results.append(r2)
            # Final Done to actually close
            self.closure_tools["Done"](summary="final close")
            return "Done."

    adapter = ResetWarningAdapter()
    spec = _minimal_spec()
    worker = FastWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(make_state())

    assert len(results) >= 2, f"Expected at least 2 Done() results, got {results}"
    assert results[0].startswith("WARNING"), f"First Done should be WARNING: {results[0]!r}"
    assert results[1].startswith("WARNING"), (
        f"Done() after Delegate should reset to WARNING again: {results[1]!r}"
    )


# ---------------------------------------------------------------------------
# AskForFeedback tests
# ---------------------------------------------------------------------------


def test_ask_for_feedback_absent_without_critic():
    """AskForFeedback closure is NOT registered when no critic node is in the graph.

    Currently AskForFeedback doesn't exist at all, so it's naturally absent —
    this test should still PASS after the feature lands (when the feature adds
    AskForFeedback only when a critic is in the graph).  We make it fail *now*
    by also asserting that AskForFeedback IS present in the critic-graph case,
    then confirm it's absent in the non-critic case.  Because the
    critic-in-graph variant is not yet implemented, the pairing test below will
    fail.  This test itself only checks the no-critic case, so it reflects
    correct future behaviour and should pass both before and after the feature.

    To make it a genuine TDD fail (verifying the feature boundary) we also
    check that the two-node spec node does NOT have _done_warned attribute
    (pre-feature) which is the new state flag introduced with two-shot Done.
    """
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _minimal_spec()  # only strategizer + implementer, no critic
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)

    assert "AskForFeedback" not in node.adapter.closure_tools
    # The two-shot Done feature introduces _done_warned; it must exist once the feature lands
    assert hasattr(node, "_done_warned"), (
        "_done_warned attribute not found — two-shot Done feature not yet implemented"
    )


def test_ask_for_feedback_present_with_critic():
    """AskForFeedback closure IS registered when a critic node is in the graph."""
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _spec_with_critic()
    node = StrategizerNode(
        adapter, name="strategizer",
        outgoing=["implementer", "critic"],
        spec=spec,
        worker_adapters={"implementer": StubAdapter(), "critic": MockCriticAdapter()},
    )

    assert "AskForFeedback" in node.adapter.closure_tools


def test_ask_for_feedback_synchronous_returns_string():
    """AskForFeedback() returns a non-empty string that doesn't start with ERROR."""
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _spec_with_critic()
    node = StrategizerNode(
        adapter, name="strategizer",
        outgoing=["implementer", "critic"],
        spec=spec,
        worker_adapters={"implementer": StubAdapter(), "critic": MockCriticAdapter("PASS")},
    )
    # Manually set a notes_dir so the closure can run
    node._current_notes_dir = None  # no ledger write needed for basic test

    result = node.adapter.closure_tools["AskForFeedback"]()
    assert isinstance(result, str)
    assert len(result) > 0
    assert not result.startswith("ERROR"), f"Got unexpected error: {result!r}"


def test_ask_for_feedback_logged_in_jsonl(tmp_path):
    """AskForFeedback() appends a record to delegation_log.jsonl."""
    from a3dasm._src.delegation_log import DelegationLog
    from a3dasm._src.nodes import StrategizerNode

    jsonl_path = tmp_path / "delegation_log.jsonl"
    delegation_log = DelegationLog(jsonl_path)
    adapter = StubAdapter()
    spec = _spec_with_critic()
    node = StrategizerNode(
        adapter, name="strategizer",
        outgoing=["implementer", "critic"],
        spec=spec,
        worker_adapters={"implementer": StubAdapter(), "critic": MockCriticAdapter()},
        notes_dir=tmp_path,
        delegation_log=delegation_log,
    )
    node._current_notes_dir = tmp_path

    node.adapter.closure_tools["AskForFeedback"]()

    assert jsonl_path.exists(), "delegation_log.jsonl not created by AskForFeedback"
    last_line = jsonl_path.read_text().strip().splitlines()[-1]
    rec = _json.loads(last_line)
    assert rec["to_node"] == "critic"


def test_ask_for_feedback_auto_injects_all_hypothesis_ids(tmp_path):
    """AskForFeedback() with no args injects all hypothesis IDs from the ledger."""
    from a3dasm._src.delegation_log import DelegationLog
    from a3dasm._src.nodes import StrategizerNode
    from a3dasm._src.hypothesis_ledger import HypothesisLedger

    # Pre-populate the ledger with H1 (OPEN) and H2 (FALSIFIED)
    ledger = HypothesisLedger(tmp_path)
    _kw = dict(
        falsification_criterion="fc",
        prediction="pred",
        prior=0.5,
        proposed_by="test",
    )
    ledger.propose(statement="Hypothesis one", **_kw)
    ledger.propose(statement="Hypothesis two", **_kw)
    ledger.update(
        "H2", "FALSIFIED", "disproved",
        evidence={"delegation": "D001"},
        posterior=0.1,
        triggered_by=None,
    )

    jsonl_path = tmp_path / "delegation_log.jsonl"
    delegation_log = DelegationLog(jsonl_path)
    adapter = StubAdapter()
    spec = _spec_with_critic()
    node = StrategizerNode(
        adapter, name="strategizer",
        outgoing=["implementer", "critic"],
        spec=spec,
        worker_adapters={"implementer": StubAdapter(), "critic": MockCriticAdapter()},
        notes_dir=tmp_path,
        delegation_log=delegation_log,
    )
    node._current_notes_dir = tmp_path

    node.adapter.closure_tools["AskForFeedback"]()

    rec = _json.loads(jsonl_path.read_text().strip().splitlines()[-1])
    assert "H1" in rec["hypothesis_ids"]
    assert "H2" in rec["hypothesis_ids"]


def test_ask_for_feedback_respects_explicit_ids(tmp_path):
    """AskForFeedback(hypothesis_ids=['H1']) only includes H1 in the record."""
    from a3dasm._src.delegation_log import DelegationLog
    from a3dasm._src.nodes import StrategizerNode
    from a3dasm._src.hypothesis_ledger import HypothesisLedger

    ledger = HypothesisLedger(tmp_path)
    _kw = dict(
        falsification_criterion="fc",
        prediction="pred",
        prior=0.5,
        proposed_by="test",
    )
    ledger.propose(statement="Hypothesis one", **_kw)
    ledger.propose(statement="Hypothesis two", **_kw)

    jsonl_path = tmp_path / "delegation_log.jsonl"
    delegation_log = DelegationLog(jsonl_path)
    adapter = StubAdapter()
    spec = _spec_with_critic()
    node = StrategizerNode(
        adapter, name="strategizer",
        outgoing=["implementer", "critic"],
        spec=spec,
        worker_adapters={"implementer": StubAdapter(), "critic": MockCriticAdapter()},
        notes_dir=tmp_path,
        delegation_log=delegation_log,
    )
    node._current_notes_dir = tmp_path

    node.adapter.closure_tools["AskForFeedback"](hypothesis_ids=["H1"])

    rec = _json.loads(jsonl_path.read_text().strip().splitlines()[-1])
    assert rec["hypothesis_ids"] == ["H1"]


def test_done_second_call_with_critic_pass():
    """Done() flow with critic PASS, including the post-Done exit interview.

    First Done() WARNs (two-shot gate). Second Done() runs the critic; on PASS
    the run does NOT close yet — it returns the exit interview (a question about
    the system, asked only after the critic accepted). The third Done(),
    carrying the retrospective, closes the run (goto=END).
    """
    from a3dasm._src.nodes import StrategizerNode
    from langgraph.graph import END

    results: list[str] = []

    class ThreeDoneCriticPassAdapter(StubAdapter):
        def invoke(self, messages):
            results.append(
                self.closure_tools["Done"](summary="first – warning"))
            results.append(
                self.closure_tools["Done"](summary="second – critic gate"))
            results.append(self.closure_tools["Done"](
                summary="### Retrospective\n- CONSISTENCY: ok\n"
                        "- DECISION: n/a\n- FRICTION: none"))
            return "Done."

    adapter = ThreeDoneCriticPassAdapter()
    spec = _spec_with_critic()
    node = StrategizerNode(
        adapter, name="strategizer",
        outgoing=["implementer", "critic"],
        spec=spec,
        worker_adapters={
            "implementer": StubAdapter(),
            "critic": MockCriticAdapter(verdict="PASS"),
        },
    )
    cmd = node(make_state())

    assert results[0].startswith("WARNING"), (
        f"Expected first Done() to WARNING, got: {results[0]!r}")
    assert "accepted by the critic" in results[1], (
        f"Expected exit interview after critic PASS, got: {results[1]!r}")
    assert cmd.goto == END


def test_done_second_call_with_critic_revise():
    """Done() second call with critic returning REVISE: does NOT close, returns ERROR."""
    from a3dasm._src.nodes import StrategizerNode
    from langgraph.graph import END

    second_results: list[str] = []

    class TwoDoneCriticReviseAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Done"](summary="first – warning")
            r = self.closure_tools["Done"](summary="second – critic gate")
            second_results.append(r)
            # Must call Done again to eventually close (avoid infinite loop)
            # Simulate strategizer giving up and forcing close after REVISE
            self.closure_tools["Done"](summary="force close after revise")
            self.closure_tools["Done"](summary="actually close")
            return "Done."

    adapter = TwoDoneCriticReviseAdapter()
    spec = _spec_with_critic()
    node = StrategizerNode(
        adapter, name="strategizer",
        outgoing=["implementer", "critic"],
        spec=spec,
        worker_adapters={
            "implementer": StubAdapter(),
            "critic": MockCriticAdapter(verdict="REVISE"),
        },
    )
    node(make_state())

    assert second_results, "Second Done() was never captured"
    assert "REVISE" in second_results[0], (
        f"Expected REVISE in second Done() result, got: {second_results[0]!r}"
    )
    # _done_warned should have been reset after REVISE response
    assert not node._done_warned, "_done_warned should reset to False after REVISE"


def test_done_closes_gracefully_ungated_after_three_revisions():
    """Bounded escape: after 3 unsatisfiable REVISE verdicts the gate closes
    UNGATED (records objections) instead of looping forever."""
    from a3dasm._src.nodes import StrategizerNode

    results: list[str] = []

    class PersistentReviseAdapter(StubAdapter):
        def invoke(self, messages):
            # Call Done() until it actually closes (or a safety cap).
            for _ in range(20):
                r = self.closure_tools["Done"](summary="please close")
                results.append(r)
                if "Run complete" in r:
                    break
            return "Done."

    node = StrategizerNode(
        PersistentReviseAdapter(), name="strategizer",
        outgoing=["implementer", "critic"],
        spec=_spec_with_critic(),
        worker_adapters={
            "implementer": StubAdapter(),
            "critic": MockCriticAdapter(verdict="REVISE"),
        },
    )
    node(make_state())

    closed = [r for r in results if "Run complete" in r]
    assert closed, "run never closed — escape did not fire"
    # The escape now routes through a retrospective round before closing; the
    # UNGATED banner lands in the recorded run summary, not the close string.
    assert any("Retrospective" in r for r in results), (
        "expected a retrospective prompt before the UNGATED close")
    assert "UNGATED" in node._route.get("summary", ""), (
        f"expected UNGATED banner in the recorded summary, got: "
        f"{node._route.get('summary', '')!r}")
    # Two "revision N/3" prompts (1/3, 2/3), then the 3rd verdict escapes.
    revise_msgs = [r for r in results if "Critic verdict:" in r]
    assert len(revise_msgs) == 2, (
        f"expected 2 revision prompts before close, got {len(revise_msgs)}")


# ---------------------------------------------------------------------------
# WriteDeliverable closure tests
# ---------------------------------------------------------------------------


def _spec_with_write_deliverable():
    """Minimal spec where the strategizer declares WriteDeliverable."""

    class A(Agent):
        role = "strategizer"
        tools = frozenset(
            {"Done", "FollowUp", "WriteNote", "ReadNote", "WriteDeliverable", "HypothesisPropose", "HypothesisUpdate", "HypothesisList", "HypothesisGet", "LinkFalsificationAttempt", "MilestoneList", "MilestonePropose", "MilestoneComplete", "MilestoneSkip", "RecallStore", "QueryStore"}
        )
        description = "Test strategizer."

    class B(Agent):
        description = "Test implementer."

    return Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )


def test_invoke_critic_persists_review_to_disk(tmp_path):
    """#7: the critic's verdict/review is always written to disk (the PASS
    branch never echoes it to the strategizer, so this is the audit trail)."""
    from a3dasm._src.nodes import StrategizerNode

    node = StrategizerNode(
        StubAdapter(), name="strategizer",
        outgoing=["implementer", "critic"], spec=_spec_with_critic(),
        worker_adapters={
            "implementer": StubAdapter(),
            "critic": MockCriticAdapter(verdict="REVISE"),
        },
    )
    node._current_notes_dir = tmp_path / "debug" / "strategizer_notes"
    node._current_notes_dir.mkdir(parents=True)

    out = node._invoke_critic("<mode>GATE</mode> review this")
    assert "REVISE" in out
    review = tmp_path / "debug" / "critic_reviews" / "call_001.md"
    assert review.exists(), "critic review not persisted to disk"
    assert "Verdict" in review.read_text()


def test_missing_deliverables_normalizes_workspace_prefix(tmp_path):
    """A config 'workspace/solution.md' must match the bare file WriteDeliverable
    actually writes at study_dir/ (audit Finding 1 — the resonance UNGATED bug)."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "pipeline.ipynb").write_text("x")
    (tmp_path / "notes.md").write_text("y")
    node = StrategizerNode(
        StubAdapter(), name="strategizer", outgoing=["implementer"],
        spec=_spec_with_write_deliverable(), notes_dir=tmp_path,
    )
    # Prefixed paths still resolve to the bare files at study root → none missing.
    state = {
        "study_dir": str(tmp_path),
        "required_deliverables": ["workspace/pipeline.ipynb", "workspace/notes.md"],
    }
    assert node._missing_deliverables(state) == []
    # A genuinely absent deliverable is still reported (as a bare name).
    state2 = {"study_dir": str(tmp_path), "required_deliverables": ["report.pdf"]}
    assert node._missing_deliverables(state2) == ["report.pdf"]


def test_write_deliverable_injected_when_in_tools(tmp_path):
    """WriteDeliverable closure is present when declared in agent tools."""
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _spec_with_write_deliverable()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=tmp_path,
    )
    assert "WriteDeliverable" in node.adapter.closure_tools


def test_write_deliverable_absent_when_not_in_tools():
    """WriteDeliverable closure is NOT registered when not in agent tools."""
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _minimal_spec()  # tools without WriteDeliverable
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )
    assert "WriteDeliverable" not in node.adapter.closure_tools


def test_write_deliverable_writes_notebook(tmp_path):
    """WriteDeliverable writes pipeline.ipynb directly to study_dir/."""
    import nbformat

    from a3dasm._src.nodes import StrategizerNode
    from a3dasm._src.notebook_exec import build_notebook

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    notes_dir = tmp_path / "run" / "debug" / "strategizer_notes"
    notes_dir.mkdir(parents=True)

    adapter = StubAdapter()
    spec = _spec_with_write_deliverable()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        study_dir=study_dir, notes_dir=notes_dir,
    )
    node._current_notes_dir = notes_dir

    nb = build_notebook([{"type": "code", "name": "analysis",
                          "source": "import f3dasm\nprint('hello')"}])
    result = node.adapter.closure_tools["WriteDeliverable"](
        "pipeline.ipynb", nbformat.writes(nb)
    )
    assert result.startswith("Written:"), f"Unexpected result: {result!r}"
    written = study_dir / "pipeline.ipynb"
    assert written.exists(), f"File not found at {written}"
    assert "import f3dasm" in written.read_text()


def test_write_deliverable_rejects_py_script(tmp_path):
    """WriteDeliverable REJECTS a .py script — the deliverable is a notebook."""
    from a3dasm._src.nodes import StrategizerNode

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    notes_dir = tmp_path / "run" / "debug" / "strategizer_notes"
    notes_dir.mkdir(parents=True)

    adapter = StubAdapter()
    spec = _spec_with_write_deliverable()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        study_dir=study_dir, notes_dir=notes_dir,
    )
    node._current_notes_dir = notes_dir

    result = node.adapter.closure_tools["WriteDeliverable"](
        "pipeline.py", "import f3dasm\nprint('hello')"
    )
    assert result.startswith("ERROR:"), f"Expected ERROR, got: {result!r}"
    assert "notebook" in result.lower()
    assert not (study_dir / "pipeline.py").exists()


def test_write_deliverable_rejects_unsupported_extension(tmp_path):
    """WriteDeliverable returns ERROR for any extension other than .ipynb."""
    from a3dasm._src.nodes import StrategizerNode

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    notes_dir = tmp_path / "run" / "debug" / "strategizer_notes"
    notes_dir.mkdir(parents=True)

    adapter = StubAdapter()
    spec = _spec_with_write_deliverable()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        study_dir=study_dir, notes_dir=notes_dir,
    )
    node._current_notes_dir = notes_dir

    result = node.adapter.closure_tools["WriteDeliverable"](
        "output.csv", "col,val\n1,2"
    )
    assert result.startswith("ERROR:"), f"Expected ERROR, got: {result!r}"
    assert ".csv" in result or "allowed" in result.lower() or "must end in" in result


def test_write_deliverable_rejects_path_separators(tmp_path):
    """WriteDeliverable returns ERROR when filename contains a path separator."""
    from a3dasm._src.nodes import StrategizerNode

    run_dir = tmp_path / "run_dir"
    notes_dir = run_dir / "debug" / "strategizer_notes"
    notes_dir.mkdir(parents=True)

    adapter = StubAdapter()
    spec = _spec_with_write_deliverable()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=notes_dir,
    )
    node._current_notes_dir = notes_dir

    result = node.adapter.closure_tools["WriteDeliverable"](
        "sub/pipeline.py", "code"
    )
    assert result.startswith("ERROR:"), f"Expected ERROR, got: {result!r}"


def test_write_deliverable_returns_error_without_notes_dir(tmp_path):
    """WriteDeliverable returns ERROR when _current_notes_dir is None."""
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _spec_with_write_deliverable()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=tmp_path,
    )
    node._current_notes_dir = None  # simulate pre-run state

    result = node.adapter.closure_tools["WriteDeliverable"](
        "pipeline.py", "code"
    )
    assert result.startswith("ERROR:"), f"Expected ERROR, got: {result!r}"


def test_strategizer_agent_tools_includes_write_deliverable():
    """StrategizerAgent.tools frozenset includes WriteDeliverable."""
    from a3dasm._src.agents.strategizer import StrategizerAgent

    assert "WriteDeliverable" in StrategizerAgent.tools, (
        "StrategizerAgent.tools must include 'WriteDeliverable'"
    )


# ---------------------------------------------------------------------------
# RecallHistory closure tests
# ---------------------------------------------------------------------------


def test_recall_history_tool_present_in_strategizer_closures(tmp_path):
    """RecallHistory closure is registered on StrategizerNode when delegation_log is set."""
    from a3dasm._src.delegation_log import DelegationLog
    from a3dasm._src.nodes import StrategizerNode

    delegation_log = DelegationLog(tmp_path / "delegation_log.jsonl")
    adapter = StubAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        delegation_log=delegation_log,
    )
    assert "RecallHistory" in node.adapter.closure_tools


def test_recall_history_returns_empty_when_no_prior(tmp_path):
    """RecallHistory returns 'No prior delegations found.' when log is empty
    for a node that CAN receive delegations (a non-entry, sub-delegating
    node — e.g. datagenerator/implementer, which have their own outgoing
    edges per agents/_graphs.py). The entry node has its own dedicated
    message (see test_recall_history_entry_node_gets_orchestrator_message)
    since its 'nothing found' is structural, not a transient empty log."""
    from a3dasm._src.delegation_log import DelegationLog
    from a3dasm._src.nodes import StrategizerNode

    delegation_log = DelegationLog(tmp_path / "delegation_log.jsonl")
    adapter = StubAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="implementer", outgoing=["literature_reviewer"],
        spec=spec, delegation_log=delegation_log,
    )
    result = node.adapter.closure_tools["RecallHistory"]()
    assert "No prior delegations found." in result


def test_recall_history_entry_node_gets_orchestrator_message(tmp_path):
    """Regression (run 20260718T132852): the entry/orchestrating node is
    ALWAYS the from_node, never the to_node, so RecallHistory is
    structurally, permanently empty for it — a generic 'No prior
    delegations found.' reads as amnesia (misdiagnosed as a context/turn-
    boundary memory bug in that run's DONE retrospective, despite 7
    delegations and 102 evals already existing). The entry node must get an
    explanatory message pointing at the right tools instead."""
    from a3dasm._src.delegation_log import DelegationLog
    from a3dasm._src.nodes import StrategizerNode

    delegation_log = DelegationLog(tmp_path / "delegation_log.jsonl")
    # Even with records that WOULD match a naive to_node="strategizer"
    # query, the entry node still gets the orchestrator-specific message —
    # this is a structural fact about the role, not about the log's state.
    delegation_log.record(
        id="D001", from_node="other", to_node="strategizer",
        task="t", deliverable="d", hypothesis_ids=["H1"],
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:01:00+00:00", status="DONE",
    )
    adapter = StubAdapter()
    spec = _minimal_spec()  # entry="strategizer"
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        delegation_log=delegation_log,
    )
    result = node.adapter.closure_tools["RecallHistory"]()
    assert "No prior delegations found." not in result
    assert "never receives one" in result
    assert "RecallStore" in result or "QueryStore" in result


def test_recall_history_returns_formatted_pairs(tmp_path):
    """RecallHistory returns formatted (task, deliverable) pairs from delegation log."""
    from a3dasm._src.delegation_log import DelegationLog
    from a3dasm._src.nodes import StrategizerNode

    jsonl_path = tmp_path / "delegation_log.jsonl"
    delegation_log = DelegationLog(jsonl_path)

    # Write two records directed to "implementer" (a non-entry, sub-
    # delegating node per agents/_graphs.py — the entry node can never be
    # a to_node, see test_recall_history_entry_node_gets_orchestrator_message)
    delegation_log.record(
        id="D001", from_node="other", to_node="implementer",
        task="First task", deliverable="First result",
        hypothesis_ids=["H1"],
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:01:00+00:00",
        status="DONE",
    )
    delegation_log.record(
        id="D002", from_node="other", to_node="implementer",
        task="Second task", deliverable="Second result",
        hypothesis_ids=["H2"],
        started_at="2026-01-01T00:02:00+00:00",
        completed_at="2026-01-01T00:03:00+00:00",
        status="DONE",
    )

    adapter = StubAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="implementer", outgoing=["literature_reviewer"],
        spec=spec, delegation_log=delegation_log,
    )
    result = node.adapter.closure_tools["RecallHistory"](n=5)
    assert "First task" in result
    assert "First result" in result
    assert "Second task" in result
    assert "Second result" in result
    assert "Prior delegation 1" in result
    assert "Prior delegation 2" in result
    # Regression: the model may pass n as a STRING ("5"); query_received does
    # matching[-n:] which raised "bad operand type for unary -: 'str'"
    # (observed run 20260627T233316). RecallHistory must coerce, not crash.
    result_str_n = node.adapter.closure_tools["RecallHistory"](n="5")
    assert "First task" in result_str_n and "Second task" in result_str_n
    # garbage falls back to the default rather than raising
    assert node.adapter.closure_tools["RecallHistory"](n="lots").startswith(
        "== Prior delegation")


def test_worker_node_has_recall_history_closure(tmp_path):
    """WorkerNode registers RecallHistory when delegation_log is passed."""
    from a3dasm._src.delegation_log import DelegationLog
    from a3dasm._src.nodes import WorkerNode

    delegation_log = DelegationLog(tmp_path / "delegation_log.jsonl")
    adapter = StubAdapter()
    node = WorkerNode(adapter, delegation_log=delegation_log, name="implementer")
    assert "RecallHistory" in node.adapter.closure_tools


def test_recall_history_gated_on_delegation_log_presence():
    """RecallHistory closure is NOT registered when delegation_log is None."""
    from a3dasm._src.nodes import StrategizerNode, WorkerNode

    # StrategizerNode without delegation_log
    adapter1 = StubAdapter()
    spec = _minimal_spec()
    node1 = StrategizerNode(
        adapter1, name="strategizer", outgoing=["implementer"], spec=spec,
        delegation_log=None,
    )
    assert "RecallHistory" not in node1.adapter.closure_tools

    # WorkerNode without delegation_log
    adapter2 = StubAdapter()
    node2 = WorkerNode(adapter2, delegation_log=None, name="implementer")
    assert "RecallHistory" not in node2.adapter.closure_tools


# ---------------------------------------------------------------------------
# _wrap_closure type-coercion shim tests
# ---------------------------------------------------------------------------


def test_wrap_closure_coerces_string_typed_args():
    """_wrap_closure coerces str→int/float/bool when annotations say so."""
    from a3dasm._src.nodes import StrategizerNode

    node = StrategizerNode(
        StubAdapter(), name="strategizer", outgoing=[],
        spec=_minimal_spec(),
    )

    def f(n: int = 5, x: float = 1.0, flag: bool = False) -> str:
        assert isinstance(n, int) and isinstance(x, float)
        assert isinstance(flag, bool)
        return f"{n}|{x}|{flag}"

    wrapped = node._wrap_closure(f, "strategizer")
    assert wrapped(n="3", x="0.5", flag="true") == "3|0.5|True"


def test_wrap_closure_leaves_uncoercible_strings():
    """Uncoercible string values pass through unchanged."""
    from a3dasm._src.nodes import StrategizerNode

    node = StrategizerNode(
        StubAdapter(), name="strategizer", outgoing=[],
        spec=_minimal_spec(),
    )

    def f(n: int = 5) -> str:
        return repr(n)

    wrapped = node._wrap_closure(f, "strategizer")
    # "five" can't be parsed as int → passes through; closure returns repr
    assert wrapped(n="five") == "'five'"


def test_wrap_closure_recall_history_regression():
    """RecallHistory-style: n='5' (str) must not TypeError on unary -."""
    from a3dasm._src.nodes import StrategizerNode

    node = StrategizerNode(
        StubAdapter(), name="strategizer", outgoing=[],
        spec=_minimal_spec(),
    )

    def g(n: int = 5) -> str:
        return str(-n)

    wrapped = node._wrap_closure(g, "strategizer")
    assert wrapped(n="5") == "-5"


# ---------------------------------------------------------------------------
# ScienceMonitor integration tests
# ---------------------------------------------------------------------------

_GOOD_WORKER_REPORT = (
    "## Report\n"
    "### Actions taken\n- ran sweep\n"
    "### Files touched\n- results.csv\n"
    "### Conclusions\nOK\n"
    "### Numbers\nbest_y: 1.47"
)


def test_supported_without_attack_is_two_shot_confirm_at_boundary(tmp_path):
    """SUPPORTED_WITHOUT_ATTACK is a TWO-SHOT CONFIRM at HypothesisUpdate (§4).

    Flow: propose H1, delegate with wait=True (not a falsification attempt),
    attempt to update H1 to SUPPORTED — HypothesisUpdate must return a
    "[CONFIRM]" string directly (not a silent accept, not a science monitor
    drain message). The short comment here is not a justification, so it stays
    a confirm.
    """
    from a3dasm._src.nodes import StrategizerNode
    from a3dasm._src.delegation_log import DelegationLog

    update_results: list[str] = []

    class MonitorAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Best_y < 2 on Ackley 8D",
                falsification_criterion="any run with best_y >= 2",
                prediction="best_y will be ~1.47",
                prior=0.6,
            )
            self.closure_tools["Delegate"](
                target="implementer",
                intent="run sweep",
                expected_report="",
                hypothesis_ids=["H1"],
                wait=True,
                is_falsification_attempt=False,
            )
            r = self.closure_tools["HypothesisUpdate"](
                hypothesis_id="H1",
                status="SUPPORTED",
                comment="numbers match",
                posterior=0.85,
                evidence={"delegation": "D001", "numbers": {"best_y": 1.47}},
            )
            update_results.append(r)
            self.closure_tools["Done"](summary="done")
            return "Done."

    jsonl_path = tmp_path / "delegation_log.jsonl"
    delegation_log = DelegationLog(jsonl_path)
    worker = StubAdapter(response=_GOOD_WORKER_REPORT)
    adapter = MonitorAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter,
        name="strategizer",
        outgoing=["implementer"],
        spec=spec,
        worker_adapters={"implementer": worker},
        notes_dir=tmp_path,
        delegation_log=delegation_log,
    )
    node._current_notes_dir = tmp_path
    node(make_state(study_dir=str(tmp_path)))

    assert update_results, "HypothesisUpdate was never called"
    assert any(r.startswith("[CONFIRM]") for r in update_results), (
        f"Expected HypothesisUpdate to return [CONFIRM] (two-shot) when no "
        f"falsification attempt exists, got: {update_results!r}"
    )


def test_phantom_delegation_blocked_inline_on_update(tmp_path):
    """HypothesisUpdate citing nonexistent D999 returns an inline ERROR.

    The check is now at the data boundary — the error is returned by
    HypothesisUpdate directly, not via the science monitor.
    """
    from a3dasm._src.nodes import StrategizerNode
    from a3dasm._src.delegation_log import DelegationLog

    update_results: list[str] = []

    class BadEvidenceAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Claim about results",
                falsification_criterion="any counter",
                prediction="none found",
                prior=0.5,
            )
            r = self.closure_tools["HypothesisUpdate"](
                hypothesis_id="H1",
                status="FALSIFIED",
                comment="trust me",
                posterior=0.1,
                evidence={"delegation": "D999"},
            )
            update_results.append(r)
            self.closure_tools["Done"](summary="done")
            return "Done."

    jsonl_path = tmp_path / "delegation_log.jsonl"
    delegation_log = DelegationLog(jsonl_path)
    adapter = BadEvidenceAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter,
        name="strategizer",
        outgoing=["implementer"],
        spec=spec,
        worker_adapters={"implementer": StubAdapter()},
        notes_dir=tmp_path,
        delegation_log=delegation_log,
    )
    node._current_notes_dir = tmp_path
    node(make_state(study_dir=str(tmp_path)))

    assert update_results, "HypothesisUpdate was never called"
    assert any(r.startswith("ERROR:") and "D999" in r for r in update_results), (
        f"Expected ERROR mentioning D999 from data-boundary check, got:"
        f" {update_results!r}"
    )


def test_boundary_errors_do_not_write_science_drift(tmp_path):
    """Data-boundary HypothesisUpdate errors must NOT write to diagnostics.jsonl.

    EVIDENCE_DELEGATION_EXISTS and SUPPORTED_WITHOUT_ATTACK are hard blocks
    at the tool level — they return an inline ERROR and do not go through the
    science monitor's diagnostics writer.
    """
    from a3dasm._src.nodes import StrategizerNode
    from a3dasm._src.delegation_log import DelegationLog

    class BadEvidenceAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Drift diagnostics claim",
                falsification_criterion="any counter",
                prediction="none found",
                prior=0.5,
            )
            # D999 does not exist — blocked at boundary, not via monitor
            self.closure_tools["HypothesisUpdate"](
                hypothesis_id="H1",
                status="FALSIFIED",
                comment="phantom delegation",
                posterior=0.1,
                evidence={"delegation": "D999"},
            )
            self.closure_tools["Done"](summary="done")
            return "Done."

    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    notes_dir = debug_dir / "strategizer_notes"
    notes_dir.mkdir()

    jsonl_path = tmp_path / "delegation_log.jsonl"
    delegation_log = DelegationLog(jsonl_path)
    adapter = BadEvidenceAdapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter,
        name="strategizer",
        outgoing=["implementer"],
        spec=spec,
        worker_adapters={"implementer": StubAdapter()},
        notes_dir=notes_dir,
        delegation_log=delegation_log,
    )
    node._current_notes_dir = notes_dir
    node(make_state(study_dir=str(tmp_path)))

    diag_path = debug_dir / "diagnostics.jsonl"
    # If diagnostics.jsonl doesn't exist, no drift records were written — pass.
    if not diag_path.exists():
        return
    drift_rules = set()
    for line in diag_path.read_text().strip().splitlines():
        r = _json.loads(line)
        if r.get("error_type") == "SCIENCE_DRIFT":
            drift_rules.add(r.get("rule"))
    assert "EVIDENCE_DELEGATION_EXISTS" not in drift_rules, (
        "EVIDENCE_DELEGATION_EXISTS should not reach diagnostics.jsonl — "
        "it is an inline boundary error, not a monitor violation."
    )


# ---------------------------------------------------------------------------
# Task 4.2: critic escalation on repeated science drift
# ---------------------------------------------------------------------------


def test_escalation_invokes_critic_and_injects_findings(tmp_path):
    """ScienceMonitor escalation invokes critic and injects findings.

    Setup: ledger + delegation_log + critic worker adapter.
    Monkeypatch node._science_monitor.escalation_due to return ["H1"]
    once (stateful fake, then returns None), and note_escalated to record
    it was called.
    The critic StubAdapter returns a canned report containing
    "### Verdict\nREVISE".
    Drive a draining closure (WriteNote) from the strategizer adapter and
    assert its result contains "[SCIENCE MONITOR — ESCALATION]" and
    "REVISE", and that note_escalated was called.
    """
    from a3dasm._src.delegation_log import DelegationLog
    from a3dasm._src.nodes import StrategizerNode

    drain_results: list[str] = []
    escalated_calls: list[bool] = []

    class ReviseCriticAdapter(StubAdapter):
        """Returns a critique report with REVISE verdict."""

        def invoke(self, messages: list) -> str:
            return (
                "## Report\n\n"
                "### Actions taken\n- Reviewed work\n\n"
                "### Findings\nSeveral issues noted.\n\n"
                "### Verdict\nREVISE\n\n"
                "### Numbers\nfindings_critical: 1\n"
            )

        def copy(self):
            fresh = ReviseCriticAdapter()
            fresh.closure_tools = dict(self.closure_tools)
            return fresh

    class WriteNoteAdapter(StubAdapter):
        """Calls WriteNote which drains notifications (including escalation)."""

        def invoke(self, messages):
            r = self.closure_tools["WriteNote"]("escalation_test.md", "x")
            drain_results.append(r)
            self.closure_tools["Done"](summary="done")
            return "Done."

    jsonl_path = tmp_path / "delegation_log.jsonl"
    delegation_log = DelegationLog(jsonl_path)

    debug_dir = tmp_path / "debug"
    debug_dir.mkdir()
    notes_dir = debug_dir / "strategizer_notes"
    notes_dir.mkdir()

    adapter = WriteNoteAdapter()
    spec = _spec_with_critic()
    node = StrategizerNode(
        adapter,
        name="strategizer",
        outgoing=["implementer", "critic"],
        spec=spec,
        worker_adapters={
            "implementer": StubAdapter(),
            "critic": ReviseCriticAdapter(),
        },
        notes_dir=notes_dir,
        delegation_log=delegation_log,
    )
    node._current_notes_dir = notes_dir

    # Stateful fake: escalation_due returns ["H1"] on first call, then None
    _escalation_calls = []

    def _fake_escalation_due():
        if not _escalation_calls:
            _escalation_calls.append(True)
            return ["H1"]
        return None

    def _fake_note_escalated():
        escalated_calls.append(True)

    node._science_monitor.escalation_due = _fake_escalation_due
    node._science_monitor.note_escalated = _fake_note_escalated

    # Monkeypatch drain() to return a sentinel — if escalation fails to
    # displace drift, this sentinel would appear alongside ESCALATION.
    _SENTINEL = "[SCIENCE MONITOR — FAKE_RULE] x\n"

    def _fake_drain():
        return _SENTINEL

    node._science_monitor.drain = _fake_drain

    node(make_state(study_dir=str(tmp_path)))

    assert drain_results, "WriteNote was never called"
    combined = "\n".join(drain_results)
    assert "[SCIENCE MONITOR — ESCALATION]" in combined, (
        f"Expected '[SCIENCE MONITOR — ESCALATION]' in drain, got: {combined!r}"
    )
    assert "REVISE" in combined, (
        f"Expected 'REVISE' in drain result, got: {combined!r}"
    )
    assert _SENTINEL.strip() not in combined, (
        "Escalation must displace regular drift injection — sentinel "
        f"should be absent when ESCALATION fires, got: {combined!r}"
    )
    assert escalated_calls, (
        "note_escalated() was never called — escalation not acknowledged"
    )


# ---------------------------------------------------------------------------
# Task 4.3: Done() critic gate embeds ledger + falsification flags
# ---------------------------------------------------------------------------


def _spec_with_critic_and_deliverable():
    """Three-node graph where strategizer has WriteDeliverable + Done."""

    class S(Agent):
        role = "strategizer"
        tools = frozenset(
            {"Done", "FollowUp", "WriteNote", "ReadNote", "WriteDeliverable", "HypothesisPropose", "HypothesisUpdate", "HypothesisList", "HypothesisGet", "LinkFalsificationAttempt", "MilestoneList", "MilestonePropose", "MilestoneComplete", "MilestoneSkip", "RecallStore", "QueryStore"}
        )
        description = "Test strategizer."

    class W(Agent):
        description = "Test implementer."

    class C(Agent):
        role = "critic"
        description = "Test adversarial critic."

    return Graph(
        nodes={"strategizer": S(), "implementer": W(), "critic": C()},
        edges=(
            Edge("strategizer", "implementer"),
            Edge("strategizer", "critic"),
        ),
        entry="strategizer",
    )


def test_done_critic_gate_embeds_ledger_and_falsification_flags(tmp_path):
    """Done() critic gate message embeds hypothesis ledger and delegation flags.

    Flow:
    1. Strategizer proposes H1 with statement "Claim below 1.0".
    2. Delegates with wait=True and is_falsification_attempt=True.
       Worker returns a report with best_y: 1.47.
    3. Updates H1 to SUPPORTED with evidence citing that delegation,
       posterior=0.9.
    4. Writes pipeline.py via WriteDeliverable (required deliverable).
    5. Calls Done() twice (two-shot).

    The critic StubAdapter captures the task_msg it receives.
    Assert captured message contains:
    (a) the literal statement "Claim below 1.0"
    (b) the substring "is_falsification_attempt"
    (c) the substring "falsification_criterion"
    """
    from a3dasm._src.delegation_log import DelegationLog
    from a3dasm._src.nodes import StrategizerNode

    captured_critic_messages: list[str] = []

    class CapturingCriticAdapter(StubAdapter):
        """Captures the task message and returns a PASS verdict."""

        def invoke(self, messages: list) -> str:
            for msg in messages:
                if isinstance(msg, dict):
                    captured_critic_messages.append(msg.get("content", ""))
                else:
                    captured_critic_messages.append(str(msg))
            return "### Verdict\nPASS"

        def copy(self):
            fresh = CapturingCriticAdapter()
            fresh.closure_tools = dict(self.closure_tools)
            return fresh

    _WORKER_REPORT = (
        "## Report\n"
        "### Actions taken\n- ran falsification sweep\n"
        "### Files touched\n- results.csv\n"
        "### Conclusions\nbest_y was 1.47, above 1.0 threshold\n"
        "### Numbers\nbest_y: 1.47"
    )

    class FalsificationWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            return _WORKER_REPORT

        def copy(self):
            fresh = FalsificationWorkerAdapter()
            fresh.closure_tools = dict(self.closure_tools)
            return fresh

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    notes_dir = tmp_path / "debug" / "strategizer_notes"
    notes_dir.mkdir(parents=True)

    jsonl_path = tmp_path / "delegation_log.jsonl"
    delegation_log = DelegationLog(jsonl_path)

    class FullFlowAdapter(StubAdapter):
        def invoke(self, messages):
            # 1. Propose H1
            self.closure_tools["HypothesisPropose"](
                statement="Claim below 1.0",
                falsification_criterion=(
                    "any run with best_y >= 1.0 falsifies this"
                ),
                prediction="best_y will stay below 1.0",
                prior=0.5,
            )
            # 2. Delegate with is_falsification_attempt=True, wait=True
            self.closure_tools["Delegate"](
                target="implementer",
                intent="Run falsification sweep on Ackley 8D.",
                expected_report="Report best_y.",
                hypothesis_ids=["H1"],
                is_falsification_attempt=True,
                wait=True,
            )
            # 3. Update H1 to SUPPORTED
            self.closure_tools["HypothesisUpdate"](
                hypothesis_id="H1",
                status="SUPPORTED",
                comment="best_y=1.47 consistent with claim",
                posterior=0.9,
                evidence={
                    "delegation": "D001",
                    "numbers": {"best_y": 1.47},
                },
            )
            # 4. Write required deliverable (must pass the reproduction gate to
            # reach the critic: print a verifiable REPRODUCED sentinel).
            self.closure_tools["WriteDeliverable"](
                "pipeline.py", "# pipeline\nprint('REPRODUCED: 0.0')"
            )
            # 4b. Resolve process milestones (orthogonal to this test) so the
            # Done() close-gate lets us reach the critic gate under test.
            import re as _re
            for _mid in _re.findall(
                    r"M\d{3}", self.closure_tools["MilestoneList"]()):
                self.closure_tools["MilestoneSkip"](_mid, "n/a for this test")
            # 5. Two-shot Done()
            self.closure_tools["Done"](summary="H1 is supported; below 1.0.")
            self.closure_tools["Done"](summary="H1 is supported; below 1.0.")
            return "Done."

    adapter = FullFlowAdapter()
    spec = _spec_with_critic_and_deliverable()
    node = StrategizerNode(
        adapter,
        name="strategizer",
        outgoing=["implementer", "critic"],
        spec=spec,
        worker_adapters={
            "implementer": FalsificationWorkerAdapter(),
            "critic": CapturingCriticAdapter(),
        },
        notes_dir=notes_dir,
        study_dir=str(study_dir),
        delegation_log=delegation_log,
    )
    node._current_notes_dir = notes_dir
    node(make_state(study_dir=str(study_dir)))

    assert captured_critic_messages, (
        "Critic adapter was never invoked — Done() gate did not reach critic"
    )
    full_msg = "\n".join(captured_critic_messages)

    # (a) Ledger dump embeds the hypothesis statement
    assert "Claim below 1.0" in full_msg, (
        f"Expected hypothesis statement in critic message; got:\n{full_msg}"
    )
    # (b) Delegation flags embed is_falsification_attempt
    assert "is_falsification_attempt" in full_msg, (
        f"Expected 'is_falsification_attempt' in critic message; got:\n{full_msg}"
    )
    # (c) Adequacy instruction references falsification_criterion
    assert "falsification_criterion" in full_msg, (
        f"Expected 'falsification_criterion' in critic message; got:\n{full_msg}"
    )


def test_delegate_decodes_json_and_comma_string_hypothesis_ids(tmp_path):
    """LLMs pass '["H1","H2"]' or 'H1, H2' — both decode to lists.

    Observed live: Haiku strategizer sent a JSON-encoded string and a
    comma-joined string before getting the list form right.
    """
    captured = []

    class Adapter(StubAdapter):
        def invoke(self, messages):
            for stmt in ("Claim alpha below 1.0", "Claim beta below 2.0"):
                self.closure_tools["HypothesisPropose"](
                    statement=stmt,
                    falsification_criterion="any counter",
                    prediction="none found",
                    prior=0.5,
                )
            captured.append(self.closure_tools["Delegate"](
                target="implementer", intent="t", expected_report="",
                hypothesis_ids='["H1", "H2"]',
            ))
            captured.append(self.closure_tools["Delegate"](
                target="implementer", intent="t", expected_report="",
                hypothesis_ids="H1, H2",
            ))
            self.closure_tools["Done"](summary="done")
            return "Done."

    from a3dasm._src.nodes import StrategizerNode
    adapter = Adapter()
    spec = _ledger_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"],
        spec=spec, worker_adapters={"implementer": StubAdapter()},
        notes_dir=tmp_path,
    )
    node(make_state())
    assert not captured[0].startswith("ERROR:"), captured[0]
    assert not captured[1].startswith("ERROR:"), captured[1]
    with node._registry_lock:
        entries = list(node._registry.values())
    assert entries[0]["hypothesis_ids"] == ["H1", "H2"]
    assert entries[1]["hypothesis_ids"] == ["H1", "H2"]


# ---------------------------------------------------------------------------
# _resolve_delegation_evals unit tests
# ---------------------------------------------------------------------------


def test_resolve_delegation_evals_returns_reported_when_no_store_dir():
    """Falls back to reported when store_dir is None."""
    from a3dasm._src.nodes import _resolve_delegation_evals

    assert _resolve_delegation_evals(None, "D001", 77) == 77


def test_resolve_delegation_evals_returns_reported_when_store_empty(
    tmp_path,
):
    """Falls back to reported when store has no rows for delegation."""
    from a3dasm._src.nodes import _resolve_delegation_evals
    from unittest.mock import patch
    from a3dasm._src.instrumented import RunStateSummary

    # Store exists but has no data for D001
    stub = RunStateSummary(
        n_rows=0,
        n_per_delegation={},
        n_per_source={},
        n_per_fidelity=None,
        output_stats={},
    )
    with patch.object(RunStateSummary, "from_store", return_value=stub):
        result = _resolve_delegation_evals(tmp_path, "D001", 55)
    assert result == 55


def test_resolve_delegation_evals_reads_store_rows(tmp_path):
    """Returns row count from store when delegation has rows."""
    from a3dasm._src.nodes import _resolve_delegation_evals
    from unittest.mock import patch
    from a3dasm._src.instrumented import RunStateSummary

    stub = RunStateSummary(
        n_rows=42,
        n_per_delegation={"D001": 42},
        n_per_source={},
        n_per_fidelity=None,
        output_stats={},
    )
    with patch.object(RunStateSummary, "from_store", return_value=stub):
        result = _resolve_delegation_evals(tmp_path, "D001", 7)
    assert result == 42


def test_resolve_delegation_evals_store_overrides_self_report(tmp_path):
    """Store row count overrides a different ReportEvals self-report."""
    from a3dasm._src.nodes import _resolve_delegation_evals
    from unittest.mock import patch
    from a3dasm._src.instrumented import RunStateSummary

    stub = RunStateSummary(
        n_rows=100,
        n_per_delegation={"D003": 100},
        n_per_source={},
        n_per_fidelity=None,
        output_stats={},
    )
    with patch.object(RunStateSummary, "from_store", return_value=stub):
        result = _resolve_delegation_evals(tmp_path, "D003", 5)
    assert result == 100


def test_resolve_delegation_evals_falls_back_when_store_none(tmp_path):
    """Falls back to reported when RunStateSummary returns None."""
    from a3dasm._src.nodes import _resolve_delegation_evals
    from unittest.mock import patch
    from a3dasm._src.instrumented import RunStateSummary

    with patch.object(RunStateSummary, "from_store", return_value=None):
        result = _resolve_delegation_evals(tmp_path, "D002", 33)
    assert result == 33


def test_resolve_delegation_evals_store_dir_path_resolution(tmp_path):
    """store_dir is derived as notes_dir.parent.parent/'experiment_data',
    and _resolve_delegation_evals correctly counts rows for a delegation.

    This is the deterministic unit-test of the store_dir derivation
    path used inside StrategizerNode._run().
    """
    from a3dasm._src.nodes import _resolve_delegation_evals
    from unittest.mock import patch
    from a3dasm._src.instrumented import RunStateSummary

    # Simulate path derivation:
    # notes_dir = run_dir/debug/strategizer_notes
    # store_dir = notes_dir.parent.parent / "experiment_data"
    debug_dir = tmp_path / "debug"
    notes_dir = debug_dir / "strategizer_notes"
    notes_dir.mkdir(parents=True)
    store_dir = notes_dir.parent.parent / "experiment_data"
    store_dir.mkdir()

    stub = RunStateSummary(
        n_rows=42,
        n_per_delegation={"D001": 42},
        n_per_source={},
        n_per_fidelity=None,
        output_stats={},
    )
    with patch.object(RunStateSummary, "from_store", return_value=stub):
        result = _resolve_delegation_evals(store_dir, "D001", 7)
    assert result == 42


def test_a6_critic_gate_allows_pass_feedback_does_not():
    """Regression for the A6 deadlock: the Done() acceptance gate must run the
    critic in a mode where PASS is available; AskForFeedback must not."""
    from a3dasm._src.agents.critic import (
        ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT,
    )
    p = ADVERSARIAL_CRITIQUE_SYSTEM_PROMPT
    # Two distinct verdict modes documented.
    assert "<mode>FEEDBACK</mode>" in p
    assert "<mode>GATE</mode>" in p
    # FEEDBACK forbids PASS; GATE allows it (PASS = accept / how a run closes).
    assert "PASS is NOT available" in p
    assert "PASS IS\n    available" in p or "PASS IS available" in p

    # The Done() gate task message must use GATE mode, not FEEDBACK.
    import inspect
    from a3dasm._src.nodes.tools import routing as _routing
    src = inspect.getsource(_routing)
    assert "<mode>GATE</mode>" in src, "Done() gate must invoke critic in GATE mode"


def test_gate_prompt_and_runscratch_point_at_namespace_aware_reads():
    """The critic GATE prompt and RunScratch's docstring must not teach the
    single-store idiom (ExperimentData.from_file(project_dir=<default store>))
    as THE way to audit the ledger — it silently misses design-namespace
    stores, the same gap fixed in RecallStore/QueryStore (backlog #21)."""
    import inspect

    from a3dasm._src.nodes.tools import routing as _routing
    src = inspect.getsource(_routing)
    assert "audit the ledger yourself: call RecallStore()" in src
    assert "misses any namespace store" in src
    assert "load_experiments()" in src
