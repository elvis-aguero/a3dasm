"""Structured notebook-authoring closures (SetNotebookIntro / AddPipelineCell).

These make the four-pillar structure + the WHY-explainer UNFORGEABLE: the pillar
name and the rationale are required arguments, so the agent cannot author a
structureless notebook. They replace the live Jupyter MCP server (ripped out).
"""
from __future__ import annotations

import nbformat
import pytest

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self):
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def _node(study_dir):
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "AddPipelineMarkdownCell", "AddPipelineCell",
                           "CheckDeliverable"})
        description = "strategizer"

    class B(Agent):
        role = "implementer"
        description = "implementer"

    spec = Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),), entry="strategizer",
    )
    return StrategizerNode(
        _Stub(), name="strategizer", outgoing=["implementer"], spec=spec,
        worker_adapters={"implementer": _Stub()}, study_dir=str(study_dir),
    )


def _read(study_dir):
    return nbformat.read(str(study_dir / "pipeline.ipynb"), as_version=4)


def _named(nb):
    return {c.metadata.get("name"): c for c in nb.cells if c.metadata.get("name")}


def test_closures_present_only_when_declared(tmp_path):
    n = _node(tmp_path)
    assert "AddPipelineMarkdownCell" in n.adapter.closure_tools
    assert "AddPipelineCell" in n.adapter.closure_tools
    assert "SetNotebookIntro" not in n.adapter.closure_tools  # retired


def test_set_intro_then_add_pillars_canonical_order(tmp_path):
    n = _node(tmp_path)
    tools = n.adapter.closure_tools
    # Add pillars OUT of order — the notebook must still come out canonical.
    tools["AddPipelineCell"]("analysis", "derive headline", "print('REPRODUCED: 1.0')")
    tools["AddPipelineCell"]("doe", "LHS over the box", "domain = ...; sampler = ...")
    tools["AddPipelineMarkdownCell"]("problem", "minimise f over the 3-box.")
    tools["AddPipelineMarkdownCell"]("hypotheses", "H1: ... H2: ...")
    tools["AddPipelineCell"]("data_generation", "evaluate via get_evaluator", "data = ...")

    nb = _read(tmp_path)
    names = [c.metadata.get("name") for c in nb.cells if c.metadata.get("name")]
    # canonical: problem, hypotheses, then doe(+why), data_generation(+why),
    # ... analysis(+why) — phases present appear in pillar order regardless of
    # the call order above.
    assert names == [
        "problem", "hypotheses",
        "doe__why", "doe",
        "data_generation__why", "data_generation",
        "analysis__why", "analysis",
    ]
    # code cell carries name + tag metadata (machine-checkable pillar presence)
    by = _named(nb)
    assert by["doe"].cell_type == "code"
    assert by["doe"].metadata.get("tags") == ["doe"]
    assert by["doe__why"].cell_type == "markdown"


def test_add_pillar_is_create_only_on_recall(tmp_path):
    # AddPipelineCell is create-only: re-calling an existing phase errors
    # (directing to EditPipelineCell) instead of blindly overwriting it.
    n = _node(tmp_path)
    tools = n.adapter.closure_tools
    tools["AddPipelineCell"]("doe", "first", "v = 1")
    out = tools["AddPipelineCell"]("doe", "second", "v = 2")
    assert "already exists" in out and "EditPipelineCell" in out
    nb = _read(tmp_path)
    does = [c for c in nb.cells if c.metadata.get("name") == "doe"]
    assert len(does) == 1 and "v = 1" in does[0].source  # unchanged


def test_unknown_phase_proceeds_with_tip_as_custom_cell(tmp_path):
    """A non-pillar phase PROCEEDS immediately (adding a cell is reversible) with
    a tip — no refusal, no two-shot. It is appended after the standard pillars.
    The deliverable's shape must not constrain what science can be expressed."""
    n = _node(tmp_path)
    tools = n.adapter.closure_tools
    tools["AddPipelineCell"]("doe", "LHS", "domain = ...")
    # Custom phase: added on the FIRST call (not refused, not two-shot), with a tip.
    out = tools["AddPipelineCell"]("ellipse_sweep", "explore ellipse phase",
                                   "data = ...")
    assert not out.startswith("ERROR:") and not out.startswith("[CONFIRM]")
    assert "custom" in out.lower()
    nb = _read(tmp_path)
    names = [c.metadata.get("name") for c in nb.cells if c.metadata.get("name")]
    # standard pillar first, custom section appended after (and it survived).
    assert "doe" in names and "ellipse_sweep" in names
    assert names.index("doe") < names.index("ellipse_sweep")


def test_add_pillar_requires_why_and_code(tmp_path):
    n = _node(tmp_path)
    tools = n.adapter.closure_tools
    assert tools["AddPipelineCell"]("doe", "  ", "code").startswith("ERROR:")
    assert tools["AddPipelineCell"]("doe", "why", "").startswith("ERROR:")


def test_markdown_cells_are_per_cell_and_create_only(tmp_path):
    # AddPipelineMarkdownCell authors problem/hypotheses INDEPENDENTLY (no
    # bundling) and is create-only — re-adding errors (use EditPipelineCell).
    n = _node(tmp_path)
    tools = n.adapter.closure_tools
    assert "Added problem" in tools["AddPipelineMarkdownCell"]("problem", "p1")
    assert "Added hypotheses" in tools["AddPipelineMarkdownCell"]("hypotheses", "h1")
    out = tools["AddPipelineMarkdownCell"]("problem", "p2")  # re-add
    assert "already exists" in out and "EditPipelineCell" in out
    nb = _read(tmp_path)
    by = _named(nb)
    assert "p1" in by["problem"].source  # unchanged by the failed re-add
    assert "h1" in by["hypotheses"].source
    # unknown name rejected
    assert tools["AddPipelineMarkdownCell"]("intro", "x").startswith("ERROR:")


def test_authored_notebook_passes_the_gate(tmp_path):
    """A notebook authored purely through the closures runs through the
    reproduction gate (a real, executable deliverable)."""
    from a3dasm._src.instrumented import InstrumentedDataGenerator
    from f3dasm._src.core import DataGenerator
    from f3dasm._src.experimentsample import ExperimentSample, JobStatus

    run_dir = tmp_path / "runs" / "A0"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    store_dir = run_dir / "experiment_data"
    store_dir.mkdir()

    class _Sum(DataGenerator):
        def execute(self, s, **k):
            s._output_data["f"] = sum(s._input_data.values())
            s.job_status = JobStatus.FINISHED
            return s

    gen = InstrumentedDataGenerator(
        inner=_Sum(), store_dir=store_dir, delegation_id="D001", flush_every=1)
    gen.execute(ExperimentSample(
        _input_data={"x0": 1.0}, _output_data={}, job_status=JobStatus.OPEN))
    gen.flush()

    n = _node(tmp_path)
    n._study_dir = tmp_path
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    tools = n.adapter.closure_tools
    tools["AddPipelineMarkdownCell"]("problem", "minimise f")
    tools["AddPipelineMarkdownCell"]("hypotheses", "H1")
    tools["AddPipelineCell"]("analysis", "derive", "print('REPRODUCED: 1.0')")
    assert n._reproduction_gate({"study_dir": str(tmp_path)}) is None


def test_check_deliverable_sees_evals_in_a_design_namespace_only(tmp_path):
    """CheckDeliverable's row-count fail-fast keyed off run_dir/experiment_data/
    experiment_data/output.csv only, so a run whose evals all landed in a
    design-namespace store (run_dir/experiment_data/<namespace>/) got a false
    'canonical store has no evaluations yet' block even though the ledger is
    populated — CheckDeliverable never even reached the reproduction gate."""
    from a3dasm._src.instrumented import InstrumentedDataGenerator
    from f3dasm._src.core import DataGenerator
    from f3dasm._src.experimentsample import ExperimentSample, JobStatus

    run_dir = tmp_path / "runs" / "A1"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    # Rows live ONLY in the "polar" namespace store. The default store's
    # output.csv EXISTS (header only, zero data rows) — the exact shape that
    # trips the old single-file row-count fail-fast (a namespace-only run
    # with an empty-but-present default CSV).
    default_csv = run_dir / "experiment_data" / "experiment_data" / "output.csv"
    default_csv.parent.mkdir(parents=True)
    default_csv.write_text("f\n")
    ns_store_dir = run_dir / "experiment_data" / "polar"
    ns_store_dir.mkdir(parents=True)

    class _Sum(DataGenerator):
        def execute(self, s, **k):
            s._output_data["f"] = sum(s._input_data.values())
            s.job_status = JobStatus.FINISHED
            return s

    gen = InstrumentedDataGenerator(
        inner=_Sum(), store_dir=ns_store_dir, delegation_id="D001",
        flush_every=1)
    gen.execute(ExperimentSample(
        _input_data={"x0": 1.0}, _output_data={}, job_status=JobStatus.OPEN))
    gen.flush()

    n = _node(tmp_path)
    n._study_dir = tmp_path
    n._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    tools = n.adapter.closure_tools
    tools["AddPipelineMarkdownCell"]("problem", "minimise f")
    tools["AddPipelineMarkdownCell"]("hypotheses", "H1")
    tools["AddPipelineCell"]("analysis", "derive", "print('REPRODUCED: 1.0')")

    result = tools["CheckDeliverable"]()
    assert "no evaluations yet" not in result, (
        f"CheckDeliverable falsely blocked a namespace-only run: {result!r}"
    )
