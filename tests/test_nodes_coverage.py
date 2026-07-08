"""Coverage tests for nodes.py — WorkerNode, WriteDeliverable, RecallHistory,
hypothesis closures, budget warnings, and other uncovered lines."""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph import END

from a3dasm._src.backends.base import Agent, Edge, Graph


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class StubAdapter:
    """Minimal adapter stub."""

    def __init__(self, response: str = "## Report\n### Actions taken\nDone.\n"
                 "### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0") -> None:
        self._response = response
        self.closure_tools: dict = {}
        self.last_usage: dict = {}

    def invoke(self, messages: list[dict]) -> str:
        return self._response


def _make_state(study_dir=None, **kwargs):
    from a3dasm._src.graph_state import AgenticState
    if study_dir is None:
        d = Path(tempfile.mkdtemp(prefix="f3dasm_nodes_cov_"))
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


def _minimal_spec(name: str = "strategizer", target: str = "implementer") -> Graph:
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote",
                           "WriteDeliverable", "HypothesisPropose", "HypothesisUpdate", "HypothesisList", "HypothesisGet", "LinkFalsificationAttempt", "MilestoneList", "MilestonePropose", "MilestoneComplete", "MilestoneSkip", "RecallStore", "QueryStore"})
        description = "Test strategizer."

    class B(Agent):
        description = "Test implementer."

    return Graph(
        nodes={name: A(), target: B()},
        edges=(Edge(name, target),),
        entry=name,
    )


# ---------------------------------------------------------------------------
# WorkerNode basic invocation
# ---------------------------------------------------------------------------


def test_worker_node_returns_command(tmp_path):
    """WorkerNode.__call__ returns a Command with last_report set."""
    from a3dasm._src.nodes import WorkerNode
    from langgraph.types import Command

    (tmp_path / "pipeline.py").write_text("# r\n")
    report = (
        "## Report\n### Actions taken\nComputed stuff.\n"
        "### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 5"
    )
    adapter = StubAdapter(response=report)
    node = WorkerNode(adapter, name="implementer")
    state = _make_state(study_dir=tmp_path, return_to=END)

    result = node(state)

    assert isinstance(result, Command)
    assert result.update["last_report"] == report


def test_worker_node_retry_on_malformed_response(tmp_path):
    """WorkerNode retries once when response is malformed."""
    from a3dasm._src.nodes import WorkerNode

    (tmp_path / "pipeline.py").write_text("# r\n")
    good_report = (
        "## Report\n### Actions taken\nDone.\n"
        "### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 0"
    )
    call_count = [0]

    class RetryAdapter(StubAdapter):
        def invoke(self, messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return "too short"
            return good_report

    adapter = RetryAdapter()
    node = WorkerNode(adapter, name="implementer")
    state = _make_state(study_dir=tmp_path, return_to=END)

    result = node(state)

    assert call_count[0] == 2
    assert result.update["last_report"] == good_report


def test_worker_node_reports_evals(tmp_path):
    """WorkerNode.ReportEvals closure records evaluation count."""
    from a3dasm._src.nodes import WorkerNode

    (tmp_path / "pipeline.py").write_text("# r\n")
    good_report = (
        "## Report\n### Actions taken\nDone.\n"
        "### Files touched\n(none)\n### Conclusions\nOK\n### Numbers\nn: 42"
    )

    class EvalAdapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["ReportEvals"](42)
            return good_report

    adapter = EvalAdapter()
    node = WorkerNode(adapter, name="implementer")
    state = _make_state(study_dir=tmp_path, return_to=END)

    result = node(state)

    assert result.update["evals_used"] == 42


# ---------------------------------------------------------------------------
# WorkerNode sandboxed Write
# ---------------------------------------------------------------------------


def test_worker_node_sandboxed_write_allows_workspace(tmp_path):
    """WorkerNode Write closure allows writes inside workspace_dir."""
    from a3dasm._src.nodes import WorkerNode

    workspace = tmp_path / "ws"
    workspace.mkdir()

    write_results = []

    class WriteAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Write"]("output.txt", "hello")
            write_results.append(result)
            return (
                "## Report\n### Actions taken\nWrote.\n"
                "### Files touched\noutput.txt\n### Conclusions\nOK\n### Numbers\nn: 0"
            )

    adapter = WriteAdapter()
    node = WorkerNode(adapter, name="implementer", workspace_dir=workspace)
    state = _make_state(study_dir=tmp_path, return_to=END)
    node(state)

    assert write_results
    assert "ERROR" not in write_results[0]
    assert (workspace / "output.txt").exists()


def test_worker_node_sandboxed_write_rejects_escape(tmp_path):
    """WorkerNode Write closure rejects paths outside workspace_dir."""
    from a3dasm._src.nodes import WorkerNode

    workspace = tmp_path / "ws"
    workspace.mkdir()

    write_results = []

    class EscapeAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Write"]("../../etc/passwd", "hack")
            write_results.append(result)
            return (
                "## Report\n### Actions taken\nTried to escape.\n"
                "### Files touched\nnone\n### Conclusions\nOK\n### Numbers\nn: 0"
            )

    adapter = EscapeAdapter()
    node = WorkerNode(adapter, name="implementer", workspace_dir=workspace)
    state = _make_state(study_dir=tmp_path, return_to=END)
    node(state)

    assert write_results
    assert "ERROR" in write_results[0]
    assert "outside" in write_results[0].lower() or "rejected" in write_results[0].lower()


# ---------------------------------------------------------------------------
# WorkerNode RecallHistory
# ---------------------------------------------------------------------------


def test_worker_node_recall_history_with_delegation_log(tmp_path):
    """WorkerNode.RecallHistory returns prior delegations from the log."""
    from a3dasm._src.nodes import WorkerNode
    from a3dasm._src.delegation_log import DelegationLog

    log_path = tmp_path / "delegation_log.jsonl"
    log = DelegationLog(log_path)
    log.record(
        id="D001",
        from_node="strategizer",
        to_node="implementer",
        task="First task",
        deliverable="First deliverable.",
        hypothesis_ids=[],
        started_at="2024-01-01T12:00:00+00:00",
        completed_at="2024-01-01T12:05:00+00:00",
        status="DONE",
    )

    recall_results = []

    class RecallAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["RecallHistory"](5)
            recall_results.append(result)
            return (
                "## Report\n### Actions taken\nRecalled.\n"
                "### Files touched\nnone\n### Conclusions\nOK\n### Numbers\nn: 0"
            )

    adapter = RecallAdapter()
    node = WorkerNode(adapter, name="implementer", delegation_log=log)
    state = _make_state(study_dir=tmp_path, return_to=END)
    node(state)

    assert recall_results
    assert "First task" in recall_results[0]


def test_worker_node_recall_history_empty(tmp_path):
    """WorkerNode.RecallHistory returns no-records message when log is empty."""
    from a3dasm._src.nodes import WorkerNode
    from a3dasm._src.delegation_log import DelegationLog

    log_path = tmp_path / "delegation_log.jsonl"
    log = DelegationLog(log_path)

    recall_results = []

    class RecallAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["RecallHistory"](5)
            recall_results.append(result)
            return (
                "## Report\n### Actions taken\nDone.\n"
                "### Files touched\nnone\n### Conclusions\nOK\n### Numbers\nn: 0"
            )

    adapter = RecallAdapter()
    node = WorkerNode(adapter, name="implementer", delegation_log=log)
    state = _make_state(study_dir=tmp_path, return_to=END)
    node(state)

    assert recall_results
    assert "No prior" in recall_results[0]


# ---------------------------------------------------------------------------
# StrategizerNode: WriteDeliverable
# ---------------------------------------------------------------------------


def test_write_deliverable_creates_file(tmp_path):
    """WriteDeliverable writes pipeline.ipynb to study_dir."""
    import nbformat

    from a3dasm._src.nodes import StrategizerNode
    from a3dasm._src.notebook_exec import build_notebook

    nb_json = nbformat.writes(build_notebook(
        [{"type": "code", "name": "analysis", "source": "x = 42"}]))
    write_results = []

    class DeliverableAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["WriteDeliverable"](
                "pipeline.ipynb", nb_json
            )
            write_results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = DeliverableAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        study_dir=str(tmp_path),
    )
    state = _make_state(study_dir=tmp_path)
    node(state)

    assert write_results
    assert "ERROR" not in write_results[0]
    assert (tmp_path / "pipeline.ipynb").exists()
    assert "42" in (tmp_path / "pipeline.ipynb").read_text()


def test_write_deliverable_rejects_bad_extension(tmp_path):
    """WriteDeliverable rejects any file that doesn't end in .ipynb."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "pipeline.ipynb").write_text("# r\n")

    results = []

    class DeliverableAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["WriteDeliverable"](
                "output.csv", "col1,col2\n1,2\n"
            )
            results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = DeliverableAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        study_dir=str(tmp_path),
    )
    state = _make_state(study_dir=tmp_path)
    node(state)

    assert results
    assert "ERROR" in results[0]


def test_write_deliverable_rejects_path_separators(tmp_path):
    """WriteDeliverable rejects filenames with path separators."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "pipeline.py").write_text("# r\n")

    results = []

    class DeliverableAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["WriteDeliverable"](
                "subdir/output.py", "x = 1\n"
            )
            results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = DeliverableAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        study_dir=str(tmp_path),
    )
    state = _make_state(study_dir=tmp_path)
    node(state)

    assert results
    assert "ERROR" in results[0]


# ---------------------------------------------------------------------------
# StrategizerNode: RecallHistory
# ---------------------------------------------------------------------------


def test_strategizer_recall_history_with_log(tmp_path):
    """StrategizerNode.RecallHistory returns entries received by strategizer."""
    from a3dasm._src.nodes import StrategizerNode
    from a3dasm._src.delegation_log import DelegationLog

    (tmp_path / "pipeline.py").write_text("# r\n")
    log_path = tmp_path / "delegation_log.jsonl"
    log = DelegationLog(log_path)
    # RecallHistory for strategizer queries records where to_node == "strategizer"
    log.record(
        id="D001",
        from_node="supervisor",
        to_node="strategizer",  # records received BY strategizer
        task="Run experiment",
        deliverable="Result: 42",
        hypothesis_ids=[],
        started_at="2024-01-01T12:00:00+00:00",
        completed_at="2024-01-01T12:05:00+00:00",
        status="DONE",
    )

    recall_results = []

    class RecallAdapter(StubAdapter):
        def invoke(self, messages):
            if "RecallHistory" in self.closure_tools:
                result = self.closure_tools["RecallHistory"](5)
                recall_results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = RecallAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        study_dir=str(tmp_path),
        delegation_log=log,
    )
    state = _make_state(study_dir=tmp_path)
    node(state)

    assert recall_results
    assert "Run experiment" in recall_results[0]


# ---------------------------------------------------------------------------
# StrategizerNode: HypothesisPropose/Update/List/Get closures
# ---------------------------------------------------------------------------


def test_hypothesis_propose_without_ledger():
    """HypothesisPropose returns ERROR when no ledger is set."""
    from a3dasm._src.nodes import StrategizerNode

    results = []

    class HypAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["HypothesisPropose"](
                statement="Test hypothesis",
                falsification_criterion="any counter-example",
                prediction="none found",
                prior=0.5,
            )
            results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = HypAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    node(_make_state())

    assert results
    assert "ERROR" in results[0]


def test_hypothesis_list_without_ledger():
    """HypothesisList returns ERROR when no ledger is set."""
    from a3dasm._src.nodes import StrategizerNode

    results = []

    class HypAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["HypothesisList"]()
            results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = HypAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    node(_make_state())

    assert results
    assert "ERROR" in results[0]


def test_hypothesis_propose_with_ledger(tmp_path):
    """HypothesisPropose returns H-id when ledger is active."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "pipeline.py").write_text("# r\n")
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    results = []

    class HypAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["HypothesisPropose"](
                statement="The sky is blue.",
                falsification_criterion="any night-time observation",
                prediction="daytime sky appears blue",
                prior=0.9,
            )
            results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = HypAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=notes_dir,
    )
    state = _make_state(study_dir=tmp_path)
    node(state)

    assert results
    # Should return an H-id like H1
    assert results[0].startswith("H") or "ERROR" not in results[0]


def test_hypothesis_list_with_entries(tmp_path):
    """HypothesisList returns hypothesis entries when ledger has items."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "pipeline.py").write_text("# r\n")
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    list_results = []
    call_phase = [0]

    class HypListAdapter(StubAdapter):
        def invoke(self, messages):
            phase = call_phase[0]
            call_phase[0] += 1
            if phase == 0:
                self.closure_tools["HypothesisPropose"](
                    statement="Hypothesis A is correct.",
                    falsification_criterion="counter-example exists",
                    prediction="no counter-example found",
                    prior=0.55,
                )
                result = self.closure_tools["HypothesisList"]()
                list_results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = HypListAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=notes_dir,
    )
    state = _make_state(study_dir=tmp_path)
    node(state)

    assert list_results
    # Should contain hypothesis A statement or an H-id
    combined = list_results[0]
    assert (
        "Hypothesis A" in combined
        or "H1" in combined
        or "ERROR" not in combined
    )
    # New format must show belief
    assert "(belief 0.55)" in combined


def test_hypothesis_update_coerces_json_string_evidence(tmp_path):
    """Backends may pass evidence as a JSON string; closure coerces."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "pipeline.py").write_text("# r\n")
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    results = []

    class Adapter(StubAdapter):
        def invoke(self, messages):
            self.closure_tools["HypothesisPropose"](
                statement="Claim below 1.0",
                falsification_criterion="any point below 0.5",
                prediction="sweep finds nothing below 0.5",
                prior=0.5,
            )
            results.append(self.closure_tools["HypothesisUpdate"](
                hypothesis_id="H1",
                status="SUPPORTED",
                comment="c",
                evidence='{"delegation": "D001"}',
                posterior=0.8,
            ))
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "Done."

    adapter = Adapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=notes_dir,
    )
    state = _make_state(study_dir=tmp_path)
    node(state)

    assert results
    assert "Updated H1" in results[0]


def test_hypothesis_get_not_found(tmp_path):
    """HypothesisGet returns ERROR for unknown hypothesis id."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "pipeline.py").write_text("# r\n")
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    results = []

    class HypGetAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["HypothesisGet"]("H999")
            results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = HypGetAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        notes_dir=notes_dir,
    )
    state = _make_state(study_dir=tmp_path)
    node(state)

    assert results
    assert "ERROR" in results[0]


# ---------------------------------------------------------------------------
# StrategizerNode: budget warnings in __call__
# ---------------------------------------------------------------------------


def test_budget_95_percent_warning_in_context():
    """At 95% budget, a budget warning is injected into the context."""
    from a3dasm._src.nodes import StrategizerNode

    received_messages = []

    class BudgetAdapter(StubAdapter):
        def invoke(self, messages):
            received_messages.extend(messages)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = BudgetAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)

    state = _make_state()
    # 95% elapsed of a 100s budget
    state["budget_seconds"] = 100.0
    state["start_time"] = time.time() - 96.0
    node(state)

    # At least one message should contain budget warning
    budget_msgs = [m for m in received_messages if "budget" in str(m.get("content", "")).lower()]
    assert budget_msgs, f"Expected budget warning in messages, got: {received_messages}"


def test_eval_budget_exceeded_warning():
    """When eval_budget is exceeded, a warning is injected into context."""
    from a3dasm._src.nodes import StrategizerNode

    received_messages = []

    class EvalBudgetAdapter(StubAdapter):
        def invoke(self, messages):
            received_messages.extend(messages)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = EvalBudgetAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)

    state = _make_state()
    state["eval_budget"] = 10
    state["evals_used"] = 15
    node(state)

    eval_msgs = [
        m for m in received_messages
        if "eval" in str(m.get("content", "")).lower()
    ]
    assert eval_msgs, f"Expected eval budget warning in messages, got: {received_messages}"


# ---------------------------------------------------------------------------
# StrategizerNode: _record_tool_error / _wrap_closure (lines 981-1059)
# ---------------------------------------------------------------------------


def test_wrap_closure_counts_error_returns(tmp_path):
    """_wrap_closure increments error_counts when closure returns ERROR:."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "pipeline.py").write_text("# r\n")

    class ErrorClosureAdapter(StubAdapter):
        def invoke(self, messages):
            # WriteNote with no notes_dir set returns ERROR
            result = self.closure_tools["WriteNote"]("test.md", "body")
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = ErrorClosureAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
    )
    state = _make_state(study_dir=tmp_path)
    node(state)

    # error_counts should have been incremented for strategizer
    assert node._error_counts.get("strategizer", 0) >= 1


# ---------------------------------------------------------------------------
# StrategizerNode: Delegate to unknown target returns ERROR
# ---------------------------------------------------------------------------


def test_delegate_unknown_target_returns_error():
    """Delegate to a non-existent target returns ERROR."""
    from a3dasm._src.nodes import StrategizerNode

    results = []

    class BadDelegateAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Delegate"](
                target="nonexistent_agent",
                intent="Do something",
                expected_report="",
            )
            results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = BadDelegateAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)
    node(_make_state())

    assert results
    assert "ERROR" in results[0]


# ---------------------------------------------------------------------------
# StrategizerNode: Delegate with no worker adapter returns ERROR
# ---------------------------------------------------------------------------


def test_delegate_no_worker_adapter_returns_error():
    """Delegate to a valid target with no worker_adapters returns ERROR."""
    from a3dasm._src.nodes import StrategizerNode

    results = []

    class NoWorkerAdapter(StubAdapter):
        def invoke(self, messages):
            result = self.closure_tools["Delegate"](
                target="implementer",
                intent="Do something",
                expected_report="",
            )
            results.append(result)
            self.closure_tools["Done"](summary="done")
            self.closure_tools["Done"](summary="done")
            return "done"

    adapter = NoWorkerAdapter()
    spec = _minimal_spec()
    # No worker_adapters provided — "implementer" has no adapter
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={},
    )
    node(_make_state())

    assert results
    assert "ERROR" in results[0]


# ---------------------------------------------------------------------------
# WorkerNode: _make_recall_history with None delegation_log
# ---------------------------------------------------------------------------


def test_worker_node_recall_history_none_log(tmp_path):
    """WorkerNode RecallHistory when delegation_log is None is not added."""
    from a3dasm._src.nodes import WorkerNode

    adapter = StubAdapter()
    node = WorkerNode(adapter, name="implementer", delegation_log=None)

    # RecallHistory should NOT be injected when delegation_log is None
    assert "RecallHistory" not in adapter.closure_tools


# ---------------------------------------------------------------------------
# StrategizerNode: Done() with pending delegations returns ERROR
# ---------------------------------------------------------------------------


def test_done_with_pending_delegations_returns_error(tmp_path):
    """Done() is refused when delegations are still Working."""
    from a3dasm._src.nodes import StrategizerNode

    (tmp_path / "pipeline.py").write_text("# r\n")

    done_results = []
    delegation_started = threading.Event()

    class SlowWorker(StubAdapter):
        def invoke(self, messages):
            delegation_started.set()
            time.sleep(3)
            return (
                "## Report\n### Actions taken\nDone.\n"
                "### Files touched\nnone\n### Conclusions\nOK\n### Numbers\nn: 0"
            )

    class EagerDoneAdapter(StubAdapter):
        def invoke(self, messages):
            # Delegate then immediately call Done
            self.closure_tools["Delegate"](
                target="implementer",
                intent="slow task",
                expected_report="",
            )
            delegation_started.wait(timeout=2)
            result = self.closure_tools["Done"](summary="trying to close")
            done_results.append(result)
            # Now wait and close properly
            time.sleep(0.1)
            self.closure_tools["Done"](summary="wait to close")
            self.closure_tools["Done"](summary="close for real")
            return "done"

    adapter = EagerDoneAdapter()
    spec = _minimal_spec()
    worker = SlowWorker()
    node = StrategizerNode(
        adapter, name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": worker},
    )
    node(_make_state(study_dir=tmp_path))

    assert done_results
    # Soft 2-option nudge (not a hard error): still refuses to close, but
    # offers keep-working / wait (GetStatus). Cancel was dropped from production.
    assert "still running" in done_results[0].lower()
    assert "GetStatus" in done_results[0]
    assert "CancelDelegation" not in done_results[0]
    assert not done_results[0].lstrip().startswith("ERROR:")


# ---------------------------------------------------------------------------
# StrategizerNode: _accumulate_usage
# ---------------------------------------------------------------------------


def test_accumulate_usage_sums_token_counts():
    """_accumulate_usage correctly sums token counts across multiple calls."""
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)

    node._accumulate_usage({"input_tokens": 10, "output_tokens": 5})
    node._accumulate_usage({"input_tokens": 20, "output_tokens": 15})

    assert node._token_totals["input_tokens"] == 30
    assert node._token_totals["output_tokens"] == 20


def test_accumulate_usage_handles_none_values():
    """_accumulate_usage treats None values as 0."""
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)

    node._accumulate_usage({"input_tokens": None, "output_tokens": None})

    assert node._token_totals["input_tokens"] == 0
    assert node._token_totals["output_tokens"] == 0


def test_accumulate_usage_adds_cost():
    """_accumulate_usage sums total_cost_usd."""
    from a3dasm._src.nodes import StrategizerNode

    adapter = StubAdapter()
    spec = _minimal_spec()
    node = StrategizerNode(adapter, name="strategizer", outgoing=["implementer"], spec=spec)

    node._accumulate_usage({"total_cost_usd": 0.01})
    node._accumulate_usage({"total_cost_usd": 0.02})

    assert abs(node._token_totals["total_cost_usd"] - 0.03) < 1e-9


# ---------------------------------------------------------------------------
# _classify_response function
# ---------------------------------------------------------------------------


def test_classify_response_short_text():
    """_classify_response returns REFLECT for text under 100 chars."""
    from a3dasm._src.nodes import _classify_response

    result = _classify_response("too short")
    assert result is not None


def test_classify_response_capability_phrase():
    """_classify_response returns REFLECT when capability-limit phrase present."""
    from a3dasm._src.nodes import _classify_response

    long_text = "A" * 200 + " I cannot do this task because it requires internet access."
    result = _classify_response(long_text)
    assert result is not None


def test_classify_response_no_report_heading():
    """_classify_response returns REFLECT when ## Report heading is missing."""
    from a3dasm._src.nodes import _classify_response

    text = "A" * 200 + "\nSome content without the required heading."
    result = _classify_response(text)
    assert result is not None


def test_classify_response_missing_subsections():
    """_classify_response returns REFLECT when required subsections are missing."""
    from a3dasm._src.nodes import _classify_response

    text = (
        "A" * 200 +
        "\n## Report\n### Actions taken\nDid stuff.\n"
        # Missing: Files touched, Conclusions, Numbers
    )
    result = _classify_response(text)
    assert result is not None


def test_classify_response_honors_per_agent_sections():
    """Audit Finding 4: validation uses the passed report_sections (DRY single
    source), so e.g. a missing ### Retrospective is caught when required."""
    from a3dasm._src.nodes import _classify_response

    body = (
        "A" * 200 +
        "\n## Report\n### Actions taken\nx\n### Conclusions\ny\n"
        # has Actions + Conclusions, but NO Retrospective
    )
    secs = ["### Actions taken", "### Conclusions", "### Retrospective"]
    assert _classify_response(body, secs) is not None  # Retrospective missing
    # Same body passes when Retrospective isn't in the required set.
    assert _classify_response(body, ["### Actions taken", "### Conclusions"]) is None


def test_classify_response_valid_report():
    """_classify_response returns None for a well-formed report."""
    from a3dasm._src.nodes import _classify_response

    text = (
        "A" * 200 +
        "\n## Report\n### Actions taken\nDid stuff.\n"
        "### Files touched\nfile.py\n"
        "### Conclusions\nSuccess.\n"
        "### Numbers\nn: 42\n"
    )
    result = _classify_response(text)
    assert result is None


# ---------------------------------------------------------------------------
# _to_adapter_messages handles list content
# ---------------------------------------------------------------------------


def test_to_adapter_messages_handles_list_content():
    """_to_adapter_messages concatenates list-typed content items."""
    from a3dasm._src.nodes import _to_adapter_messages
    from langchain_core.messages import HumanMessage

    msg = HumanMessage(content=[{"text": "Hello"}, {"text": "World"}])
    result = _to_adapter_messages([msg])

    assert len(result) == 1
    assert "Hello" in result[0]["content"]
    assert "World" in result[0]["content"]
