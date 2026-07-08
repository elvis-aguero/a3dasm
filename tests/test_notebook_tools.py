"""Headless tests for the gold-stated notebook tool surface: name-addressed
CRUD with optimistic concurrency (rev guard), surgical find/replace, a read
tool (ShowNotebook), and scratch execution (RunScratch). Plus regression that
the dead live-Jupyter-MCP path is gone.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""

    def copy(self):
        s = self.__class__.__new__(self.__class__)
        _Stub.__init__(s)
        s.closure_tools = dict(self.closure_tools)
        return s


_NB_TOOLS = frozenset({
    "AddPipelineMarkdownCell", "AddPipelineCell", "EditPipelineCell",
    "DeletePipelineCell", "ShowNotebook", "RunScratch",
})


def _node(tmp_path):
    class A(Agent):
        role = "strategizer"
        tools = _NB_TOOLS
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )
    return StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()}, study_dir=str(tmp_path),
    )


def _tool(n, name):
    return n.adapter.closure_tools[name]


def _read_nb(tmp_path):
    return nbformat.read(str(tmp_path / "pipeline.ipynb"), as_version=4)


def _cell(nb, name):
    for c in nb.cells:
        if (c.get("metadata", {}) or {}).get("name") == name:
            return c
    return None


def _rev(tmp_path, phase):
    src = _cell(_read_nb(tmp_path), phase)["source"]
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:8]


# ── AddPipelineCell — create-only ────────────────────────────────────────────

def test_add_creates_and_reports_rev(tmp_path):
    n = _node(tmp_path)
    out = _tool(n, "AddPipelineCell")("doe", why="w", code="x = 1")
    assert "Added doe" in out and "rev" in out
    assert _cell(_read_nb(tmp_path), "doe")["source"] == "x = 1"


def test_add_existing_phase_is_create_only_error(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="x = 1")
    out = _tool(n, "AddPipelineCell")("doe", why="w2", code="x = 2")
    assert "already exists" in out
    assert "EditPipelineCell" in out
    # unchanged
    assert _cell(_read_nb(tmp_path), "doe")["source"] == "x = 1"


# ── AddPipelineMarkdownCell + narrative-cell CRUD (replaces SetNotebookIntro) ──

def test_add_markdown_cell_creates_problem_and_hypotheses(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineMarkdownCell")("problem", "minimise f over the box")
    _tool(n, "AddPipelineMarkdownCell")("hypotheses", "H1: …")
    nb = _read_nb(tmp_path)
    assert "minimise f" in _cell(nb, "problem")["source"]
    assert "# Problem & objective" in _cell(nb, "problem")["source"]  # heading added
    assert "## Hypotheses" in _cell(nb, "hypotheses")["source"]


def test_add_markdown_cell_rejects_unknown_name(tmp_path):
    n = _node(tmp_path)
    out = _tool(n, "AddPipelineMarkdownCell")("intro", "x")
    assert out.startswith("ERROR:") and "problem" in out and "hypotheses" in out


def test_add_markdown_cell_is_create_only(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineMarkdownCell")("problem", "p1")
    out = _tool(n, "AddPipelineMarkdownCell")("problem", "p2")
    assert "already exists" in out
    assert "p1" in _cell(_read_nb(tmp_path), "problem")["source"]


def test_edit_narrative_cell_with_content_and_rev(tmp_path):
    # The clunk fix: update hypotheses ALONE (no resupplying problem), rev-guarded.
    n = _node(tmp_path)
    _tool(n, "AddPipelineMarkdownCell")("problem", "the problem")
    _tool(n, "AddPipelineMarkdownCell")("hypotheses", "old hyp")
    out = _tool(n, "EditPipelineCell")(
        "hypotheses", content="new hyp", expected_rev=_rev(tmp_path, "hypotheses"))
    assert "Edited hypotheses" in out
    nb = _read_nb(tmp_path)
    assert "new hyp" in _cell(nb, "hypotheses")["source"]
    assert "the problem" in _cell(nb, "problem")["source"]  # untouched


def test_edit_narrative_cell_stale_rev_rejected(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineMarkdownCell")("hypotheses", "h")
    out = _tool(n, "EditPipelineCell")("hypotheses", content="x", expected_rev="deadbeef")
    assert "changed since" in out


def test_edit_narrative_cell_rejects_code_param(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineMarkdownCell")("problem", "p")
    out = _tool(n, "EditPipelineCell")("problem", code="x=1", expected_rev=_rev(tmp_path, "problem"))
    assert "markdown cell" in out and "content" in out


def test_delete_narrative_cell(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineMarkdownCell")("problem", "p")
    _tool(n, "AddPipelineMarkdownCell")("hypotheses", "h")
    out = _tool(n, "DeletePipelineCell")("hypotheses", expected_rev=_rev(tmp_path, "hypotheses"))
    assert "Deleted hypotheses" in out
    nb = _read_nb(tmp_path)
    assert _cell(nb, "hypotheses") is None and _cell(nb, "problem") is not None


# ── ShowNotebook — read + list by name ───────────────────────────────────────

def test_show_brief_lists_all_cells_by_name(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="rationale", code="x = 1")
    _tool(n, "AddPipelineCell")("ml", why="fit it", code="fit()")
    out = _tool(n, "ShowNotebook")()
    for name in ("doe", "doe__why", "ml", "ml__why"):
        assert name in out
    assert "rev" in out
    assert "Missing:" in out  # not all pillars present


def test_show_detailed_returns_full_source_and_rev(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="x = 1\ny = 2")
    out = _tool(n, "ShowNotebook")("doe")
    assert "x = 1\ny = 2" in out
    assert _rev(tmp_path, "doe") in out


def test_show_missing_phase_lists_present(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="x=1")
    out = _tool(n, "ShowNotebook")("analysis")
    assert "no cell named" in out and "doe" in out


def test_show_absent_file_reports_without_creating(tmp_path):
    n = _node(tmp_path)
    out = _tool(n, "ShowNotebook")()
    assert "does not exist" in out
    assert not (tmp_path / "pipeline.ipynb").exists()


# ── EditPipelineCell — surgical (self-guarding) ──────────────────────────────

def test_surgical_edit_replaces_single_occurrence(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="a = 1\nb = 2")
    out = _tool(n, "EditPipelineCell")("doe", old="a = 1", new="a = 99")
    assert "Edited doe" in out
    assert _cell(_read_nb(tmp_path), "doe")["source"] == "a = 99\nb = 2"
    # why untouched, name preserved
    assert "w" in _cell(_read_nb(tmp_path), "doe__why")["source"]
    assert _cell(_read_nb(tmp_path), "doe")["metadata"]["name"] == "doe"


def test_surgical_old_not_found_errors(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="a = 1")
    out = _tool(n, "EditPipelineCell")("doe", old="zzz", new="q")
    assert "not found" in out


def test_surgical_ambiguous_old_names_count(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="p = 1\np = 1")
    out = _tool(n, "EditPipelineCell")("doe", old="p = 1", new="p = 2")
    assert "2" in out and ("occurs" in out or "×" in out)


def test_surgical_and_fullfield_mutually_exclusive(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="a = 1")
    out = _tool(n, "EditPipelineCell")("doe", old="a", new="b", code="z = 1")
    assert "EITHER" in out or "not both" in out


# ── EditPipelineCell — full-field, rev-guarded ───────────────────────────────

def test_fullfield_edit_requires_expected_rev(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="x = 1")
    out = _tool(n, "EditPipelineCell")("doe", code="x = 2")
    assert "expected_rev" in out


def test_fullfield_edit_with_current_rev_succeeds(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="x = 1")
    rev = _rev(tmp_path, "doe")
    out = _tool(n, "EditPipelineCell")("doe", code="x = 2", expected_rev=rev)
    assert "Edited doe" in out
    assert _cell(_read_nb(tmp_path), "doe")["source"] == "x = 2"


def test_fullfield_edit_with_stale_rev_rejected(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="x = 1")
    out = _tool(n, "EditPipelineCell")("doe", code="x = 2", expected_rev="deadbeef")
    assert "changed since" in out
    assert _cell(_read_nb(tmp_path), "doe")["source"] == "x = 1"  # unchanged


def test_edit_missing_phase_errors(tmp_path):
    n = _node(tmp_path)
    out = _tool(n, "EditPipelineCell")("analysis", code="y = 1", expected_rev="x")
    assert "not in pipeline.ipynb" in out


# ── DeletePipelineCell — rev-guarded ─────────────────────────────────────────

def test_delete_requires_expected_rev(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="c")
    out = _tool(n, "DeletePipelineCell")("doe")
    assert "expected_rev" in out
    assert _cell(_read_nb(tmp_path), "doe") is not None  # not deleted


def test_delete_with_current_rev_removes_cell_and_why(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="c")
    _tool(n, "AddPipelineCell")("ml", why="w2", code="c2")
    out = _tool(n, "DeletePipelineCell")("ml", expected_rev=_rev(tmp_path, "ml"))
    assert "Deleted ml" in out
    nb = _read_nb(tmp_path)
    assert _cell(nb, "ml") is None and _cell(nb, "ml__why") is None
    assert _cell(nb, "doe") is not None


def test_delete_stale_rev_rejected(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="c")
    out = _tool(n, "DeletePipelineCell")("doe", expected_rev="deadbeef")
    assert "changed since" in out
    assert _cell(_read_nb(tmp_path), "doe") is not None


def test_delete_absent_phase_is_noop(tmp_path):
    n = _node(tmp_path)
    _tool(n, "AddPipelineCell")("doe", why="w", code="c")
    out = _tool(n, "DeletePipelineCell")("analysis", expected_rev="x")
    assert "Nothing to delete" in out


# ── RunScratch ───────────────────────────────────────────────────────────────

def test_run_scratch_executes_and_returns_stdout(tmp_path):
    n = _node(tmp_path)
    notes = tmp_path / "runs" / "T1" / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    n._current_notes_dir = notes
    out = _tool(n, "RunScratch")("print(6 * 7)")
    assert "exit 0" in out and "42" in out


def test_run_scratch_surfaces_errors(tmp_path):
    n = _node(tmp_path)
    notes = tmp_path / "runs" / "T1" / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    n._current_notes_dir = notes
    out = _tool(n, "RunScratch")("raise ValueError('boom')")
    assert "exit 1" in out and "boom" in out


def test_run_scratch_empty_code_errors(tmp_path):
    n = _node(tmp_path)
    out = _tool(n, "RunScratch")("   ")
    assert "ERROR" in out


# ── Dead jupyter-MCP path removed ────────────────────────────────────────────

def test_strategizer_has_no_jupyter_config():
    from a3dasm._src.agents.strategizer import StrategizerAgent
    a = StrategizerAgent()
    assert not getattr(a, "needs_jupyter_server", False)
    assert getattr(a, "mcp_servers", {}) == {}
    assert not any("jupyter" in t for t in getattr(a, "extra_allowed_tools", []))


def test_agent_runtime_imports_without_notebook_server():
    import importlib
    import os
    importlib.import_module("a3dasm._src.agent_runtime")
    import a3dasm._src as ag
    assert not os.path.exists(ag.__path__[0] + "/notebook_server.py")
