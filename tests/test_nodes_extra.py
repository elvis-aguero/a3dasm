"""Additional tests for nodes.py — FollowUp/Reply, AskForFeedback, budget
broadcast, Delegate(wait=True), and other uncovered paths.

These tests extend coverage without touching the existing test_nodes.py file.
"""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import END

from a3dasm._src.backends.base import Agent, Edge, Graph


# ---------------------------------------------------------------------------
# Shared helpers (mirror test_nodes.py but local to this module)
# ---------------------------------------------------------------------------


class StubAdapter:
    """Minimal adapter stub that records closure tools."""

    def __init__(self, response: str = "## Done\nAll done.") -> None:
        self._response = response
        self.closure_tools: dict = {}

    def invoke(self, messages: list[dict]) -> str:
        return self._response


def _minimal_spec(name: str = "strategizer", target: str = "implementer") -> Graph:
    class A(Agent):
        role = "strategizer"
        # GetStatus/CancelDelegation are opt-in (plug-and-play) post-audit; opt
        # in so behaviour tests still exercise them.
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote",
                           "GetStatus", "CancelDelegation"})
        description = "Test strategizer."

    class B(Agent):
        description = "Test implementer."

    return Graph(
        nodes={name: A(), target: B()},
        edges=(Edge(name, target),),
        entry=name,
    )


def _make_state(study_dir=None, **kwargs):
    from a3dasm._src.graph_state import AgenticState
    import tempfile
    if study_dir is None:
        d = Path(tempfile.mkdtemp(prefix="f3dasm_nodes_extra_"))
        (d / "pipeline.py").write_text("# test\n")
        study_dir = d
    return AgenticState(
        messages=[HumanMessage(content="Test problem")],
        study_dir=str(study_dir),
        done=False,
        last_report=None,
        total_delegations=0,
        budget_seconds=kwargs.pop("budget_seconds", None),
        return_to=kwargs.pop("return_to", None),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Reply to unknown delegation
# ---------------------------------------------------------------------------


def test_reply_unknown_delegation_returns_error():
    """Reply() for an unknown delegation ID returns an ERROR string."""
    from a3dasm._src.nodes import StrategizerNode

    replies: list[str] = []

    class ReplyCallingAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Reply"]("NONEXISTENT_ID", "some answer")
            replies.append(result)
            self.closure_tools["Done"](summary="closing")
            self.closure_tools["Done"](summary="closing")
            return "done"

    adapter = ReplyCallingAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    node(_make_state())

    assert replies and "ERROR" in replies[0]
    assert "NONEXISTENT_ID" in replies[0]


# ---------------------------------------------------------------------------
# FollowUp + Reply round-trip
# ---------------------------------------------------------------------------


def test_followup_reply_roundtrip():
    """Worker FollowUp blocks until Reply is called; returns the answer."""
    from a3dasm._src.nodes import StrategizerNode

    followup_answers: list[str] = []
    delegation_id_box: list[str] = []
    worker_ready = threading.Event()

    class FollowUpWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            worker_ready.set()
            answer = self.closure_tools["FollowUp"]("What is the expected shape?")
            followup_answers.append(answer)
            return (
                "## Report\n### Actions taken\nDone.\n"
                "### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"
            )

    class OrchestratorAdapter(StubAdapter):
        def invoke(self, messages):
            did = self.closure_tools["Delegate"](
                target="implementer",
                intent="Analyse the data.",
                expected_report="Shapes.",
            )
            delegation_id_box.append(did)
            # Wait for worker to reach FollowUp before calling Reply
            worker_ready.wait(timeout=3)
            time.sleep(0.05)  # give worker time to register FollowUp status
            reply_result = self.closure_tools["Reply"](
                # Extract D### from the delegation id string returned by Delegate
                re.search(r"D\d{3}", did).group(0),
                "Shape is (100, 3).",
            )
            time.sleep(0.1)  # let worker resume and finish
            self.closure_tools["Done"](summary="completed")
            self.closure_tools["Done"](summary="completed")
            return "done"

    adapter = OrchestratorAdapter()
    spec = _minimal_spec()
    worker = FollowUpWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(_make_state())

    assert followup_answers, "Worker FollowUp was never resolved"
    assert "Shape is (100, 3)." in followup_answers[0]


# ---------------------------------------------------------------------------
# AskForFeedback when critic is present
# ---------------------------------------------------------------------------


def test_ask_for_feedback_calls_critic_synchronously():
    """AskForFeedback() invokes the critic adapter synchronously and returns its output."""
    from a3dasm._src.nodes import StrategizerNode

    feedback_calls: list[str] = []

    class CriticAdapter(StubAdapter):
        def invoke(self, messages):
            feedback_calls.append(messages[0]["content"])
            return "### Verdict\nREVISE\n\nYour approach has a flaw."

    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote", "AskForFeedback"})
        description = "Test strategizer."

    class C(Agent):
        role = "critic"
        description = "Test critic."

    spec = Graph(
        nodes={"strategizer": A(), "critic": C()},
        edges=(Edge("strategizer", "critic"),),
        entry="strategizer",
    )

    results: list[str] = []

    class FeedbackCallingAdapter(StubAdapter):
        def invoke(self, messages):
            # AskForFeedback should be in closure_tools when critic is wired
            if "AskForFeedback" in self.closure_tools:
                result = self.closure_tools["AskForFeedback"]()
                results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = FeedbackCallingAdapter()
    critic = CriticAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["critic"], spec=spec,
        worker_adapters={"critic": critic},
    )
    node(_make_state())

    assert results, "AskForFeedback was never present or never called"
    assert "REVISE" in results[0] or "flaw" in results[0]
    assert feedback_calls, "Critic adapter was never called"


# ---------------------------------------------------------------------------
# AskForFeedback absent when no critic
# ---------------------------------------------------------------------------


def test_ask_for_feedback_absent_without_critic():
    """AskForFeedback is NOT in closure_tools when the graph has no critic node."""
    from a3dasm._src.nodes import StrategizerNode

    tool_names: list[set] = []

    class InspectAdapter(StubAdapter):
        def invoke(self, messages):
            tool_names.append(set(self.closure_tools.keys()))
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = InspectAdapter()
    spec = _minimal_spec()  # no critic
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    node(_make_state())

    assert tool_names
    assert "AskForFeedback" not in tool_names[0]


# ---------------------------------------------------------------------------
# Budget broadcast: GetStatus includes BUDGET warning at 85%+ elapsed
# ---------------------------------------------------------------------------


def test_getstatus_includes_budget_warning_when_over_80_pct():
    """GetStatus() on a Working delegation surfaces BUDGET when >80% elapsed."""
    from a3dasm._src.nodes import StrategizerNode

    getstatus_results: list[str] = []
    delegation_started = threading.Event()

    class SlowWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            delegation_started.set()
            time.sleep(5)  # never finishes in test — we poll before it ends
            return (
                "## Report\n### Actions taken\nDone.\n"
                "### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"
            )

    class PollCallingAdapter(StubAdapter):
        def invoke(self, messages):
            did_str = self.closure_tools["Delegate"](
                target="implementer", intent="slow task", expected_report=""
            )
            did = re.search(r"D\d{3}", did_str).group(0)
            delegation_started.wait(timeout=2)
            result = self.closure_tools["GetStatus"](did)
            getstatus_results.append(result)
            self.closure_tools["Done"](summary="partial")
            self.closure_tools["Done"](summary="partial")
            return "done"

    adapter = PollCallingAdapter()
    spec = _minimal_spec()
    worker = SlowWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    # Pass budget via state so __call__ picks it up (budget=100s, started 85s ago)
    run_start = time.time() - 85.0
    state = _make_state(budget_seconds=100.0)
    state["start_time"] = run_start
    node(state)

    assert getstatus_results
    # At least one GetStatus result should mention BUDGET
    assert any("BUDGET" in r for r in getstatus_results), (
        f"Expected BUDGET in GetStatus results, got: {getstatus_results}"
    )


# ---------------------------------------------------------------------------
# Delegate(wait=True) returns result directly
# ---------------------------------------------------------------------------


def test_delegate_wait_true_returns_report_directly():
    """Delegate(wait=True) blocks and returns Done\\n\\n<report> without polling."""
    from a3dasm._src.nodes import StrategizerNode

    delegate_results: list[str] = []

    class FastWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            return (
                "## Report\n### Actions taken\nComputed result.\n"
                "### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 42"
            )

    class WaitDelegateAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Delegate"](
                target="implementer",
                intent="Fast synchronous task.",
                expected_report="A result.",
                wait=True,
            )
            delegate_results.append(result)
            self.closure_tools["Done"](summary="wrapped up")
            self.closure_tools["Done"](summary="wrapped up")
            return "done"

    adapter = WaitDelegateAdapter()
    spec = _minimal_spec()
    worker = FastWorkerAdapter()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(_make_state())

    assert delegate_results, "Delegate(wait=True) never returned"
    result = delegate_results[0]
    assert result.startswith("Done"), f"Expected result starting with 'Done', got: {result!r}"
    assert "Computed result." in result or "42" in result


# ---------------------------------------------------------------------------
# WriteNote and ReadNote basics
# ---------------------------------------------------------------------------


def test_writenote_and_readnote(tmp_path):
    """WriteNote writes a file to strategizer_notes/; the file should be created."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "pipeline.py").write_text("# r\n")
    # run_dir/debug/strategizer_notes is where WriteNote will write
    run_dir = tmp_path / "run"
    notes_dir = run_dir / "debug" / "strategizer_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    write_results: list[str] = []

    class NoteAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["WriteNote"]("my_note.md", "# Finding\nSome data.")
            write_results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = NoteAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        study_dir=str(tmp_path),
        notes_dir=notes_dir,
    )
    state = _make_state(study_dir=tmp_path)
    # Provide run_dir so __call__ sets _current_notes_dir correctly
    state["run_dir"] = str(run_dir)
    node(state)

    # WriteNote should have created the file inside strategizer_notes
    assert write_results, "WriteNote was never called"
    assert "ERROR" not in write_results[0], f"WriteNote returned error: {write_results[0]}"
    note_file = notes_dir / "my_note.md"
    assert note_file.exists(), f"WriteNote did not create the file; write result: {write_results}"
    assert "Finding" in note_file.read_text()


# ---------------------------------------------------------------------------
# GetStatus for unknown delegation
# ---------------------------------------------------------------------------


def test_getstatus_unknown_id_returns_error():
    """GetStatus for an unknown ID returns ERROR."""
    from a3dasm._src.nodes import StrategizerNode

    results: list[str] = []

    class GetStatusAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["GetStatus"]("D999")
            results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = GetStatusAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    node(_make_state())

    assert results and "ERROR" in results[0]


# ---------------------------------------------------------------------------
# Change 2a: time-budget banner injected into every worker task message
# ---------------------------------------------------------------------------


def test_delegate_task_msg_includes_budget_banner():
    """When _budget_seconds and _run_start are set, the task message
    passed to the worker begins with a time-budget banner.
    """
    import time as _time

    from a3dasm._src.nodes import StrategizerNode

    captured_msgs: list[list[dict]] = []

    class CapturingWorkerAdapter:
        """Captures the messages list passed to invoke()."""
        closure_tools: dict = {}
        last_usage: dict = {}

        def copy(self):
            a = CapturingWorkerAdapter()
            a.closure_tools = dict(self.closure_tools)
            return a

        def invoke(self, messages):
            captured_msgs.append(list(messages))
            # Minimal valid report.
            return (
                "## Report\n\n"
                "### Actions taken\n- did\n\n"
                "### Files touched\n- none\n\n"
                "### Conclusions\nok\n\n"
                "### Numbers\nbest: 1\n"
            )

    class DelegatingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](
                target="implementer",
                intent="run something",
                expected_report="a result",
                hypothesis_ids=None,
                wait=True,
            )
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    worker_adapter = CapturingWorkerAdapter()
    main_adapter = DelegatingAdapter()
    spec = _minimal_spec()

    node = StrategizerNode(
        adapter=main_adapter,
        name="strategizer",
        outgoing=["implementer"],
        spec=spec,
        worker_adapters={"implementer": worker_adapter},
    )
    # Pass budget via state so __call__ sets _budget_seconds/_run_start.
    _start = _time.time() - 60.0  # 60s elapsed
    node(_make_state(
        budget_seconds=600.0,
        start_time=_start,
    ))

    assert captured_msgs, "Worker was never invoked"
    first_msg_content = captured_msgs[0][0]["content"]
    assert "Time budget" in first_msg_content, (
        f"Expected 'Time budget' in first worker message; "
        f"got: {first_msg_content[:200]!r}"
    )


def test_delegate_no_budget_banner_when_budget_unset():
    """When _budget_seconds is None, no time-budget banner is prepended."""
    import time as _time

    from a3dasm._src.nodes import StrategizerNode

    captured_msgs: list[list[dict]] = []

    class CapturingWorkerAdapter:
        closure_tools: dict = {}
        last_usage: dict = {}

        def copy(self):
            a = CapturingWorkerAdapter()
            a.closure_tools = dict(self.closure_tools)
            return a

        def invoke(self, messages):
            captured_msgs.append(list(messages))
            return (
                "## Report\n\n"
                "### Actions taken\n- did\n\n"
                "### Files touched\n- none\n\n"
                "### Conclusions\nok\n\n"
                "### Numbers\nbest: 1\n"
            )

    class DelegatingAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["Delegate"](
                target="implementer",
                intent="run something",
                expected_report="a result",
                hypothesis_ids=None,
                wait=True,
            )
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    worker_adapter = CapturingWorkerAdapter()
    main_adapter = DelegatingAdapter()
    spec = _minimal_spec()

    node = StrategizerNode(
        adapter=main_adapter,
        name="strategizer",
        outgoing=["implementer"],
        spec=spec,
        worker_adapters={"implementer": worker_adapter},
    )
    # No budget in state — banner must be absent.
    node(_make_state())

    assert captured_msgs, "Worker was never invoked"
    first_msg_content = captured_msgs[0][0]["content"]
    assert "Time budget" not in first_msg_content, (
        f"Unexpected 'Time budget' in worker message with no budget: "
        f"{first_msg_content[:200]!r}"
    )
