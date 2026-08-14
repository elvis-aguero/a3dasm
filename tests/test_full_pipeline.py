"""End-to-end integration test: AgenticRun.execute() with mock adapters."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from a3dasm._src.agent_runtime import AgenticRun
from a3dasm._src.backends.base import Agent, Edge, Graph

from .fixtures import MockWorkerAdapter, ScriptedStrategistAdapter


PROBLEM_MD = """\
# Test Problem

Maximise sigma_crit for coilable designs.

## Design space
ratio_d: [0.004, 0.073]

## Objective
Find highest sigma_crit with coilable == 1.
"""


def _make_study(tmp_path: Path) -> Path:
    study = tmp_path / "study"
    study.mkdir()
    (study / "PROBLEM_STATEMENT.md").write_text(PROBLEM_MD)
    return study


class _StrategistSpec(Agent):
    role = "strategizer"
    description = "Test strategizer."
    # GetStatus is opt-in (plug-and-play) since the Confer rework; this scripted
    # driver polls delegations deterministically, so it declares the opt-in.
    tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote",
                       "WriteDeliverable", "GetStatus", "HypothesisPropose", "HypothesisUpdate", "HypothesisList", "HypothesisGet", "LinkFalsificationAttempt", "MilestoneList", "MilestonePropose", "MilestoneComplete", "MilestoneSkip", "RecallStore", "QueryStore"})


class _WorkerSpec(Agent):
    description = "Test implementer."


def _graph_spec() -> Graph:
    return Graph(
        nodes={"strategizer": _StrategistSpec(), "implementer": _WorkerSpec()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )


@pytest.fixture
def pipeline_run(tmp_path):
    """AgenticRun with mock adapters; returns (run, study_dir) after execute()."""
    study = _make_study(tmp_path)
    run = AgenticRun(study_dir=study, graph=_graph_spec())
    strat = ScriptedStrategistAdapter(run=run)
    worker = MockWorkerAdapter()

    def _mock_make_adapter(name, agent):
        return strat if name == "strategizer" else worker

    run._make_adapter = _mock_make_adapter
    run.execute()
    return run, study


# ---------------------------------------------------------------------------
# Output directory structure
# ---------------------------------------------------------------------------


def test_run_dir_created(pipeline_run, tmp_path):
    """execute() creates runs/<timestamp>/ directory; solution.md at study root."""
    _, study = pipeline_run
    runs = list((study / "runs").iterdir())
    assert len(runs) == 1
    run_dir = runs[0]
    assert (run_dir / "debug" / "strategizer_notes").is_dir()
    assert (run_dir / "debug" / "run.log").exists()
    # the deliverable notebook lives at the study root (there is no solution.md)
    assert (study / "pipeline.ipynb").exists()
    assert not (study / "solution.md").exists()


def test_workspace_subfolders_created(pipeline_run, tmp_path):
    """Worker delegations create debug/delegations/D001/ and D002/."""
    _, study = pipeline_run
    run_dir = next((study / "runs").iterdir())
    workspace = run_dir / "debug" / "delegations"
    assert workspace.is_dir()
    # Two delegations → two subfolders
    subfolders = [p.name for p in workspace.iterdir() if p.is_dir()]
    assert "D001" in subfolders
    assert "D002" in subfolders


# ---------------------------------------------------------------------------
# Hypothesis ledger
# ---------------------------------------------------------------------------


def test_hypotheses_json_created(pipeline_run):
    """hypotheses.json exists under strategizer_notes/."""
    _, study = pipeline_run
    notes = next((study / "runs").iterdir()) / "debug" / "strategizer_notes"
    assert (notes / "hypotheses.json").exists()


def test_hypotheses_json_has_two_entries(pipeline_run):
    """H1 and H2 are recorded in hypotheses.json."""
    _, study = pipeline_run
    notes = next((study / "runs").iterdir()) / "debug" / "strategizer_notes"
    data = json.loads((notes / "hypotheses.json").read_text())
    assert "H1" in data
    assert "H2" in data


def test_h1_falsified_h2_supported(pipeline_run):
    """H1 ends FALSIFIED and H2 ends SUPPORTED."""
    _, study = pipeline_run
    notes = next((study / "runs").iterdir()) / "debug" / "strategizer_notes"
    data = json.loads((notes / "hypotheses.json").read_text())
    assert data["H1"]["status_log"][-1]["status"] == "FALSIFIED"
    assert data["H2"]["status_log"][-1]["status"] == "SUPPORTED"


def test_triggered_by_links_delegation(pipeline_run):
    """H1's FALSIFIED entry has triggered_by = D001."""
    _, study = pipeline_run
    notes = next((study / "runs").iterdir()) / "debug" / "strategizer_notes"
    data = json.loads((notes / "hypotheses.json").read_text())
    falsified_entry = data["H1"]["status_log"][-1]
    assert falsified_entry["triggered_by"] == "D001"


# ---------------------------------------------------------------------------
# Delegation log
# ---------------------------------------------------------------------------


def test_delegation_log_created(pipeline_run):
    """delegation_log.jsonl exists under debug/ with two logical records.

    (The raw file also carries dispatch-time RUNNING entries; query_all()
    collapses last-wins to one record per delegation — see provenance fix.)"""
    from a3dasm._src.delegation_log import DelegationLog
    _, study = pipeline_run
    debug = next((study / "runs").iterdir()) / "debug"
    records = DelegationLog(debug / "delegation_log.jsonl").query_all()
    assert len(records) == 2


def test_delegation_records_have_token_fields(pipeline_run):
    """Each COMPLETED delegation record has non-zero tokens_in and tokens_out.

    (Dispatch-time RUNNING entries legitimately carry 0 tokens; query_all()
    collapses to the terminal record per delegation.)"""
    from a3dasm._src.delegation_log import DelegationLog
    _, study = pipeline_run
    debug = next((study / "runs").iterdir()) / "debug"
    for rec in DelegationLog(debug / "delegation_log.jsonl").query_all():
        if rec["status"] != "DONE":
            continue
        assert rec["tokens_in"] > 0, f"tokens_in missing in {rec['id']}"
        assert rec["tokens_out"] > 0, f"tokens_out missing in {rec['id']}"
        assert rec["cost_usd"] is not None


def test_delegation_records_link_hypotheses(pipeline_run):
    """D001 links to H1 and D002 links to H2."""
    _, study = pipeline_run
    debug = next((study / "runs").iterdir()) / "debug"
    records = [
        json.loads(l)
        for l in (debug / "delegation_log.jsonl").read_text().strip().splitlines()
    ]
    by_id = {r["id"]: r for r in records}
    assert "H1" in by_id["D001"]["hypothesis_ids"]
    assert "H2" in by_id["D002"]["hypothesis_ids"]


# ---------------------------------------------------------------------------
# solution.md token table
# ---------------------------------------------------------------------------


def _notebook_markdown(study):
    """Concatenated markdown of the deliverable notebook (the writeup)."""
    import nbformat
    nb = nbformat.read(str(study / "pipeline.ipynb"), as_version=4)
    return "\n\n".join(
        c.source for c in nb.cells if c.get("cell_type") == "markdown")


def test_notebook_has_token_table(pipeline_run):
    """The deliverable notebook carries a ## Token usage section (stamped by the
    runtime as a trailing markdown cell) — there is no solution.md."""
    _, study = pipeline_run
    assert not (study / "solution.md").exists()
    text = _notebook_markdown(study)
    assert "## Token usage" in text
    assert "input_tokens" in text
    assert "output_tokens" in text
    assert "estimated_cost" in text


def test_notebook_nonzero_tokens(pipeline_run):
    """Token counts in the notebook writeup are non-zero (mock usage)."""
    _, study = pipeline_run
    text = _notebook_markdown(study)
    m = re.search(r"\| total_tokens \| ([\d,]+) \|", text)
    assert m, "total_tokens row not found in the notebook writeup"
    total = int(m.group(1).replace(",", ""))
    assert total > 0


# ---------------------------------------------------------------------------
# notebook provenance stamp
# ---------------------------------------------------------------------------


def test_notebook_has_provenance_metadata(pipeline_run):
    """The deliverable notebook carries run provenance in its metadata (model +
    run dir + timestamp) — the script provenance header is gone."""
    import nbformat
    _, study = pipeline_run
    nb = nbformat.read(str(study / "pipeline.ipynb"), as_version=4)
    agentic = nb.metadata.get("agentic", {})
    assert agentic.get("model")
    assert agentic.get("run")
    assert agentic.get("timestamp")


def test_notebook_carries_problem_statement_provenance(pipeline_run):
    """A study is meant to be run once per PROBLEM_STATEMENT.md; when it isn't
    (the file drifts and gets re-run against), a reader of a later
    pipeline.ipynb needs a way to check what statement THAT run actually
    answered. debug/PROBLEM_STATEMENT_snapshot.md is the verbatim, recoverable
    copy; the notebook only carries its hash (both the visible Run metadata
    cell and nb.metadata['agentic']) so a reader can confirm which snapshot
    matches without needing to inspect raw notebook JSON.
    """
    import hashlib

    import nbformat
    run, study = pipeline_run
    run_dir = next((study / "runs").iterdir())
    snapshot_path = run_dir / "debug" / "PROBLEM_STATEMENT_snapshot.md"
    assert snapshot_path.exists()
    assert snapshot_path.read_text(encoding="utf-8") == (
        study / "PROBLEM_STATEMENT.md"
    ).read_text(encoding="utf-8")

    expected_hash = hashlib.sha256(
        snapshot_path.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()

    nb = nbformat.read(str(study / "pipeline.ipynb"), as_version=4)
    assert nb.metadata["agentic"]["problem_statement_sha256"] == expected_hash

    stamp_cell = next(
        c for c in nb.cells
        if (getattr(c, "metadata", None) or {}).get("name") == "_run_provenance"
    )
    assert expected_hash in stamp_cell.source
    assert "PROBLEM_STATEMENT_snapshot.md" in stamp_cell.source


# ---------------------------------------------------------------------------
# run.log
# ---------------------------------------------------------------------------


def test_run_log_mentions_tokens(pipeline_run):
    """run.log final line includes token counts."""
    _, study = pipeline_run
    run_dir = next((study / "runs").iterdir())
    log_text = (run_dir / "debug" / "run.log").read_text()
    assert "Tokens" in log_text or "tokens" in log_text


# ---------------------------------------------------------------------------
# constraint snapshot on the very first strategizer turn (human -> strategizer
# is a delegation like any other; the budget/time state must be automatically
# in the literal first message text, not something the agent has to go query
# for — see constraint_snapshot.py)
# ---------------------------------------------------------------------------


def test_first_strategizer_message_carries_constraint_snapshot(tmp_path):
    """The FIRST message the strategizer's underlying adapter.invoke() ever
    receives — traced through the real AgenticRun.execute() path, not a
    smaller unit test of one function in isolation — already contains the
    <constraints> block, before the strategizer has taken any action at all.
    """
    study = _make_study(tmp_path)

    captured_first_messages: list = []

    class CapturingStrategist:
        """Records the messages of its FIRST invoke() call, then closes the
        run immediately (no critic in this graph, so Done() closes directly)."""

        def __init__(self):
            self.closure_tools: dict = {}
            self.route_watcher = None
            self.last_usage: dict = {}

        def invoke(self, messages):
            if not captured_first_messages:
                captured_first_messages.append(list(messages))
            tools = self.closure_tools
            if "MilestoneList" in tools:
                import re as _re
                for mid in _re.findall(r"M\d{3}", tools["MilestoneList"]()):
                    tools["MilestoneSkip"](mid, "n/a for this test")
            tools["WriteDeliverable"](
                "pipeline.py", "# pipeline\nprint('REPRODUCED: 0.0')"
            )
            tools["Done"](summary="closing immediately")
            tools["Done"](summary="closing immediately")
            return "done"

    class NoOpWorker:
        closure_tools: dict = {}
        last_usage: dict = {}

        def invoke(self, messages):
            return "## Report\n### Actions taken\n- n/a\n"

    run = AgenticRun(study_dir=study, graph=_graph_spec())
    strat = CapturingStrategist()
    worker = NoOpWorker()

    def _mock_make_adapter(name, agent):
        return strat if name == "strategizer" else worker

    run._make_adapter = _mock_make_adapter
    run.execute()

    assert captured_first_messages, "Strategizer adapter was never invoked"
    first_call_messages = captured_first_messages[0]
    assert first_call_messages, "First invoke() call had no messages"
    first_content = first_call_messages[0]["content"]
    assert "<constraints>" in first_content, (
        "Expected the constraint snapshot in the very first message the "
        f"strategizer receives; got:\n{first_content[:300]!r}"
    )
    assert "Evaluation budget" in first_content
    assert "Wall-clock budget" in first_content


def test_run_log_does_not_falsely_claim_the_notebook_was_stamped(tmp_path):
    """Regression: a live wet-test run (an under-capable model that closed
    the run without ever calling WriteDeliverable) showed run.log claiming
    "pipeline.ipynb stamped" even though pipeline.ipynb never existed — the
    log line was unconditional while the actual stamping code was gated on
    nb_path.exists(). If the notebook was never written, run.log must say so
    honestly instead of claiming success.
    """
    study = _make_study(tmp_path)

    class NoDeliverableStrategist:
        """Closes the run without ever calling WriteDeliverable()."""

        def __init__(self):
            self.closure_tools: dict = {}
            self.route_watcher = None
            self.last_usage: dict = {}

        def invoke(self, messages):
            tools = self.closure_tools
            if "MilestoneList" in tools:
                for mid in re.findall(r"M\d{3}", tools["MilestoneList"]()):
                    tools["MilestoneSkip"](mid, "n/a for this test")
            tools["Done"](summary="closing without a deliverable")
            tools["Done"](summary="closing without a deliverable")
            return "done"

    class NoOpWorker:
        closure_tools: dict = {}
        last_usage: dict = {}

        def invoke(self, messages):
            return "## Report\n### Actions taken\n- n/a\n"

    run = AgenticRun(study_dir=study, graph=_graph_spec())
    strat = NoDeliverableStrategist()
    worker = NoOpWorker()

    def _mock_make_adapter(name, agent):
        return strat if name == "strategizer" else worker

    run._make_adapter = _mock_make_adapter
    run.execute()

    assert not (study / "pipeline.ipynb").exists(), (
        "test setup invariant broken: the deliverable should not exist"
    )
    run_dir = next((study / "runs").iterdir())
    log_text = (run_dir / "debug" / "run.log").read_text()
    assert "pipeline.ipynb stamped" not in log_text, (
        f"run.log falsely claims the notebook was stamped; log:\n{log_text}"
    )
    assert "NEVER WRITTEN" in log_text, (
        f"Expected run.log to honestly report the missing deliverable; "
        f"log:\n{log_text}"
    )
