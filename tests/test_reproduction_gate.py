"""Unit tests for StrategizerNode._reproduction_gate (§B of the pipeline
deliverable redesign).

The gate EXECUTES pipeline.py lazily against the canonical ledger and PASSES
(returns None) iff the script exits 0 (its own derive-from-ledger assert held)
AND adds ZERO new oracle rows (lazy: a reproduction must skip finished evals).
These tests drive each control-flow branch with real subprocess execution.
"""
from __future__ import annotations

import os
from pathlib import Path

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.instrumented import InstrumentedDataGenerator
from a3dasm._src.nodes import StrategizerNode
from f3dasm._src.core import DataGenerator
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus


class _StubAdapter:
    def __init__(self) -> None:
        self.closure_tools: dict = {}

    def invoke(self, messages):  # pragma: no cover - not exercised
        return "ok"


class _Sum(DataGenerator):
    def execute(self, s, **k):
        s._output_data["f"] = sum(s._input_data.values())
        s.job_status = JobStatus.FINISHED
        return s


def _spec() -> Graph:
    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "WriteNote"})
        description = "s"

    class B(Agent):
        description = "i"

    return Graph(
        nodes={"strategizer": A(), "implementer": B()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )


def _seed_store(store_dir: Path, n: int = 2) -> None:
    """Write n provenance-stamped rows to store_dir (= the project_dir)."""
    gen = InstrumentedDataGenerator(
        inner=_Sum(), store_dir=store_dir, delegation_id="D001", flush_every=1)
    for i in range(n):
        gen.execute(ExperimentSample(
            _input_data={"x0": float(i)}, _output_data={},
            job_status=JobStatus.OPEN))
    gen.flush()


def _setup(tmp_path: Path):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    run_dir = tmp_path / "runs" / "T0"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    _seed_store(run_dir / "experiment_data", n=2)  # before == 2 rows
    node = StrategizerNode(
        _StubAdapter(), name="strategizer", outgoing=["implementer"],
        spec=_spec(), study_dir=study_dir)
    node._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    return node, study_dir


def test_gate_passes_for_clean_lazy_pipeline(tmp_path):
    node, study_dir = _setup(tmp_path)
    (study_dir / "pipeline.py").write_text("print('REPRODUCED: 1.0')\n")
    assert node._reproduction_gate({"study_dir": str(study_dir)}) is None


def test_gate_fails_for_erroring_pipeline(tmp_path):
    node, study_dir = _setup(tmp_path)
    (study_dir / "pipeline.py").write_text("import sys\nsys.exit(2)\n")
    problem = node._reproduction_gate({"study_dir": str(study_dir)})
    assert problem is not None and "failed" in problem.lower()


def test_gate_fails_when_pipeline_adds_evals(tmp_path):
    """A NON-lazy pipeline that re-evaluates stamps a new row → not lazy."""
    node, study_dir = _setup(tmp_path)
    (study_dir / "pipeline.py").write_text(
        "import os\n"
        "from a3dasm._src.instrumented import InstrumentedDataGenerator\n"
        "from f3dasm._src.core import DataGenerator\n"
        "from f3dasm._src.experimentsample import ExperimentSample, JobStatus\n"
        "class G(DataGenerator):\n"
        "    def execute(self, s, **k):\n"
        "        s._output_data['f'] = 1.0\n"
        "        s.job_status = JobStatus.FINISHED\n"
        "        return s\n"
        "store = os.environ['F3DASM_CANONICAL_STORE']\n"
        "g = InstrumentedDataGenerator(inner=G(), store_dir=store,\n"
        "                              delegation_id='D777', flush_every=1)\n"
        "g.execute(ExperimentSample(_input_data={'x0': 9.0}, _output_data={},\n"
        "                           job_status=JobStatus.OPEN))\n"
        "g.flush()\n"
    )
    problem = node._reproduction_gate({"study_dir": str(study_dir)})
    assert problem is not None and "lazy" in problem.lower()


def test_gate_fails_when_pipeline_adds_evals_to_a_namespace_store(tmp_path):
    """A NON-lazy pipeline that re-evaluates into a DESIGN-NAMESPACE store
    (run_dir/experiment_data/<namespace>/) instead of the default one must be
    caught too — _ledger_snapshot has to aggregate across every
    experiment_stores() store, the same primitive LedgerBreakdown/
    ScienceMonitor already use, not just read the default store."""
    node, study_dir = _setup(tmp_path)
    (study_dir / "pipeline.py").write_text(
        "import os\n"
        "from a3dasm._src.instrumented import InstrumentedDataGenerator\n"
        "from f3dasm._src.core import DataGenerator\n"
        "from f3dasm._src.experimentsample import ExperimentSample, JobStatus\n"
        "class G(DataGenerator):\n"
        "    def execute(self, s, **k):\n"
        "        s._output_data['f'] = 1.0\n"
        "        s.job_status = JobStatus.FINISHED\n"
        "        return s\n"
        "store = os.path.join(os.environ['F3DASM_CANONICAL_STORE'], 'polar')\n"
        "g = InstrumentedDataGenerator(inner=G(), store_dir=store,\n"
        "                              delegation_id='D777', flush_every=1)\n"
        "g.execute(ExperimentSample(_input_data={'x0': 9.0}, _output_data={},\n"
        "                           job_status=JobStatus.OPEN))\n"
        "g.flush()\n"
    )
    problem = node._reproduction_gate({"study_dir": str(study_dir)})
    assert problem is not None and "lazy" in problem.lower(), (
        f"gate should have caught the namespace-store write as non-lazy, "
        f"got: {problem!r}"
    )


def test_gate_skips_without_run_context(tmp_path):
    node, study_dir = _setup(tmp_path)
    (study_dir / "pipeline.py").write_text("print('ok')\n")
    node._current_notes_dir = None
    assert node._reproduction_gate({"study_dir": str(study_dir)}) is None


def test_empty_ledger_rejected_by_gate(tmp_path):
    """Gate rejects when the canonical store has no rows — the campaign hasn't
    run yet, so there is nothing to ground the headline against."""
    study_dir = tmp_path / "study"; study_dir.mkdir()
    run_dir = tmp_path / "runs" / "T0"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    # Deliberately do NOT seed the store — leave it absent/empty.
    node = StrategizerNode(
        _StubAdapter(), name="strategizer", outgoing=["implementer"],
        spec=_spec(), study_dir=study_dir)
    node._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    (study_dir / "pipeline.py").write_text("print('REPRODUCED: 0.0')\n")
    problem = node._reproduction_gate({"study_dir": str(study_dir)})
    assert problem is not None and "no rows" in problem.lower()


# ── headline grounding is the critic's job, not a runtime extremum match ──────
# The runtime gate enforces clean-exit + zero-new-evals + ledger-integrity. It
# does NOT machine-match the REPRODUCED value: that wrongly rejected legitimate
# CONSTRAINED optima (a constrained best is not an objective extremum), forcing
# studies to headline their infeasible unconstrained extremum (run
# 20260624T021359 shipped REPRODUCED: 0.907090, a non-coilable design). Grounding
# now lives in the critic's HEADLINE PROVENANCE check.

def test_gate_does_not_machine_check_headline_value(tmp_path):
    """A REPRODUCED value the runtime cannot tie to an extremum (e.g. 999.0) now
    PASSES the runtime gate — provenance is the critic's job, not the runtime's."""
    node, study_dir = _setup(tmp_path)
    (study_dir / "pipeline.py").write_text("print('REPRODUCED: 999.0')\n")
    assert node._reproduction_gate({"study_dir": str(study_dir)}) is None


def test_gate_passes_without_headline_line(tmp_path):
    """The REPRODUCED line is informational, not gate-binding: a pipeline that
    reproduces cleanly but prints no headline still PASSES the runtime gate."""
    node, study_dir = _setup(tmp_path)
    (study_dir / "pipeline.py").write_text("print('all done, trust me')\n")
    assert node._reproduction_gate({"study_dir": str(study_dir)}) is None


def test_gate_passes_constrained_nonextremum_headline(tmp_path):
    """Regression for run 20260624T021359: a CONSTRAINED optimum — a real
    evaluated value that is NOT the objective extremum — must PASS. Seed f∈{0,1,2}
    so REPRODUCED: 1.0 is a genuine mid-ledger value (extrema are 0.0 and 2.0).
    Pre-fix the runtime rejected it as 'not grounded'; post-fix it passes."""
    study_dir = tmp_path / "study"; study_dir.mkdir()
    run_dir = tmp_path / "runs" / "T0"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True)
    _seed_store(run_dir / "experiment_data", n=3)  # f = 0.0, 1.0, 2.0
    node = StrategizerNode(
        _StubAdapter(), name="strategizer", outgoing=["implementer"],
        spec=_spec(), study_dir=study_dir)
    node._current_notes_dir = run_dir / "debug" / "strategizer_notes"
    (study_dir / "pipeline.py").write_text("print('REPRODUCED: 1.0')\n")
    assert node._reproduction_gate({"study_dir": str(study_dir)}) is None


def test_gate_runs_in_sandbox_never_pollutes_real_ledger(tmp_path):
    """Regression: a NON-lazy pipeline (re-evaluating) must be caught as not-lazy
    WITHOUT adding its evals to the real canonical ledger. The gate runs against
    a throwaway copy, so repeated checks can never inflate the real store.
    """
    node, study_dir = _setup(tmp_path)  # real store seeded with 2 rows
    run_dir = node._current_notes_dir.parent.parent
    store_dir = run_dir / "experiment_data"
    real_before = len(ExperimentData.from_file(project_dir=store_dir).to_pandas()[1])
    # A non-lazy pipeline: it stamps a NEW eval into F3DASM_CANONICAL_STORE.
    (study_dir / "pipeline.py").write_text(
        "import os\n"
        "from a3dasm._src.instrumented import InstrumentedDataGenerator\n"
        "from f3dasm._src.core import DataGenerator\n"
        "from f3dasm._src.experimentsample import ExperimentSample, JobStatus\n"
        "class G(DataGenerator):\n"
        "    def execute(self, s, **k):\n"
        "        s._output_data['f'] = 1.0\n"
        "        s.job_status = JobStatus.FINISHED\n"
        "        return s\n"
        "store = os.environ['F3DASM_CANONICAL_STORE']\n"
        "g = InstrumentedDataGenerator(inner=G(), store_dir=store,\n"
        "                              delegation_id='D777', flush_every=1)\n"
        "g.execute(ExperimentSample(_input_data={'x0': 9.0}, _output_data={},\n"
        "                           job_status=JobStatus.OPEN))\n"
        "g.flush()\n"
        "print('REPRODUCED: 1.0')\n"
    )
    # Run the gate several times — each would have added a row pre-fix.
    for _ in range(3):
        problem = node._reproduction_gate({"study_dir": str(study_dir)})
        assert problem is not None and "lazy" in problem.lower()
    real_after = len(ExperimentData.from_file(project_dir=store_dir).to_pandas()[1])
    assert real_after == real_before, (
        f"real ledger was polluted: {real_before} → {real_after}")


def test_gate_rejects_ledger_tampering(tmp_path):
    """Rewriting an existing ledger row's value (to fake a zero-delta) is caught
    by the integrity check even though the row COUNT is unchanged."""
    node, study_dir = _setup(tmp_path)
    (study_dir / "pipeline.py").write_text(
        "import os\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "csv = Path(os.environ['F3DASM_CANONICAL_STORE'])"
        " / 'experiment_data' / 'output.csv'\n"
        "df = pd.read_csv(csv, index_col=0)\n"
        "df.iloc[0, df.columns.get_loc('f')] = 12345.0\n"
        "df.to_csv(csv)\n"
        "print('REPRODUCED: 12345.0')\n"
    )
    problem = node._reproduction_gate({"study_dir": str(study_dir)})
    assert problem is not None and "modified" in problem.lower()


# ── ROOT 2 proof: ONE load-or-create script satisfies BOTH checks ─────────────
# The deliverable spec is non-contradictory iff a single composable pipeline.py
# can BOTH (a) reproduce lazily on a shipped ledger (zero new evals, headline
# grounded) AND (b) regenerate the campaign from an empty store. We prove it by
# building exactly that script and exercising both paths — no assertion, the
# gate and the subprocess are the evidence.

_LOAD_OR_CREATE_PIPELINE = (
    "import os\n"
    "from f3dasm._src.experimentdata import ExperimentData\n"
    "store = os.environ['F3DASM_CANONICAL_STORE']\n"
    "def _n_finished(s):\n"
    "    try:\n"
    "        return len(ExperimentData.from_file(project_dir=s).to_pandas()[1])\n"
    "    except Exception:\n"
    "        return 0\n"
    "if _n_finished(store) > 0:\n"
    "    data = ExperimentData.from_file(project_dir=store)   # LOAD: lazy, 0 new\n"
    "else:\n"
    "    from a3dasm._src.instrumented import InstrumentedDataGenerator\n"
    "    from f3dasm._src.core import DataGenerator\n"
    "    from f3dasm._src.experimentsample import ExperimentSample, JobStatus\n"
    "    class _Sum(DataGenerator):\n"
    "        def execute(self, s, **k):\n"
    "            s._output_data['f'] = sum(s._input_data.values())\n"
    "            s.job_status = JobStatus.FINISHED\n"
    "            return s\n"
    "    g = InstrumentedDataGenerator(inner=_Sum(), store_dir=store,\n"
    "                                  delegation_id='D001', flush_every=1)\n"
    "    for i in range(2):\n"
    "        g.execute(ExperimentSample(_input_data={'x0': float(i)},\n"
    "                  _output_data={}, job_status=JobStatus.OPEN))\n"
    "    g.flush()\n"
    "    data = ExperimentData.from_file(project_dir=store)    # CREATE: regenerate\n"
    "_, out = data.to_pandas()\n"
    "print(f\"REPRODUCED: {float(out['f'].max())}\")\n"
)


def test_load_or_create_pipeline_passes_gate_lazily(tmp_path):
    """(a) On a SHIPPED ledger, the load-or-create script takes the LOAD branch:
    zero new evals + headline grounded → the real gate PASSES."""
    node, study_dir = _setup(tmp_path)                 # seeds 2 finished rows
    (study_dir / "pipeline.py").write_text(_LOAD_OR_CREATE_PIPELINE)
    assert node._reproduction_gate({"study_dir": str(study_dir)}) is None


def test_load_or_create_pipeline_regenerates_from_empty(tmp_path):
    """(b) The SAME script, pointed at an EMPTY store, takes the CREATE branch:
    it regenerates the campaign (2 finished rows) and prints the headline."""
    import os
    import subprocess
    import sys

    from f3dasm._src.experimentdata import ExperimentData

    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text(_LOAD_OR_CREATE_PIPELINE)
    empty_store = tmp_path / "fresh_store"               # does not exist yet

    env = dict(os.environ, F3DASM_CANONICAL_STORE=str(empty_store))
    proc = subprocess.run(
        [sys.executable, str(pipeline)], capture_output=True, text=True, env=env)

    assert proc.returncode == 0, proc.stderr
    assert "REPRODUCED: 1.0" in proc.stdout            # headline derived, not hardcoded
    # the empty store was populated from scratch — the script regenerates
    _, out = ExperimentData.from_file(project_dir=empty_store).to_pandas()
    assert len(out) == 2


# ── Spec-04: notebook deliverable (executor-agnostic gate) ────────────────────
import nbformat  # noqa: E402
from a3dasm._src import settings  # noqa: E402
from a3dasm._src.notebook_exec import (  # noqa: E402
    build_notebook, required_deliverable_name, run_deliverable,
)


def _write_nb(study_dir, cells):
    nbformat.write(build_notebook(cells), str(study_dir / "pipeline.ipynb"))


def test_required_deliverable_is_always_notebook():
    """The system is committed to the notebook: the single deliverable is
    pipeline.ipynb, unconditionally (no dual-mode flag)."""
    assert required_deliverable_name() == "pipeline.ipynb"
    # A stray config flag must not flip it back to a script.
    try:
        settings.configure({"debug": True})
        assert required_deliverable_name() == "pipeline.ipynb"
    finally:
        settings.configure({})


def test_repro_gate_executes_notebook_lazily(tmp_path):
    """A notebook that loads the ledger and self-asserts the headline → PASS,
    zero new rows (mirror of test_gate_passes_for_clean_lazy_pipeline)."""
    node, study_dir = _setup(tmp_path)
    _write_nb(study_dir, [
        {"type": "markdown", "source": "# Problem\nminimise f."},
        {"type": "code", "name": "analyze", "source": "print('REPRODUCED: 1.0')"},
    ])
    assert node._reproduction_gate({"study_dir": str(study_dir)}) is None


def test_repro_gate_fails_when_notebook_adds_evals(tmp_path):
    """A NON-lazy notebook that stamps a new eval → caught as not-lazy."""
    node, study_dir = _setup(tmp_path)
    _write_nb(study_dir, [{"type": "code", "name": "run", "source": (
        "import os\n"
        "from a3dasm._src.instrumented import InstrumentedDataGenerator\n"
        "from f3dasm._src.core import DataGenerator\n"
        "from f3dasm._src.experimentsample import ExperimentSample, JobStatus\n"
        "class G(DataGenerator):\n"
        "    def execute(self, s, **k):\n"
        "        s._output_data['f'] = 1.0; s.job_status = JobStatus.FINISHED; return s\n"
        "g = InstrumentedDataGenerator(inner=G(), store_dir=os.environ['F3DASM_CANONICAL_STORE'],\n"
        "                              delegation_id='D777', flush_every=1)\n"
        "g.execute(ExperimentSample(_input_data={'x0': 9.0}, _output_data={}, job_status=JobStatus.OPEN))\n"
        "g.flush()\n"
        "print('REPRODUCED: 1.0')\n")}])
    problem = node._reproduction_gate({"study_dir": str(study_dir)})
    assert problem is not None and "lazy" in problem.lower()


def test_repro_gate_env_vars_reach_kernel(tmp_path):
    """The notebook kernel must see F3DASM_CANONICAL_STORE (env propagation is a
    known footgun); a cell asserting it errors → gate FAIL if it didn't reach."""
    node, study_dir = _setup(tmp_path)
    _write_nb(study_dir, [{"type": "code", "name": "analyze", "source": (
        "import os\n"
        "assert os.environ['F3DASM_CANONICAL_STORE'], 'env not propagated'\n"
        "print('REPRODUCED: 1.0')\n")}])
    # PASS proves the env var was visible inside the kernel.
    assert node._reproduction_gate({"study_dir": str(study_dir)}) is None


def test_shared_assert_helper_is_executor_agnostic(tmp_path):
    """run_deliverable gives a CompletedProcess with the same shape + headline
    for an equivalent .py and .ipynb (DRY guard on the executor split)."""
    sandbox = tmp_path / "sb"; sandbox.mkdir()
    env = dict(os.environ, F3DASM_CANONICAL_STORE=str(sandbox))
    py = tmp_path / "d.py"; py.write_text("print('REPRODUCED: 1.0')\n")
    nbformat.write(build_notebook(
        [{"type": "code", "source": "print('REPRODUCED: 1.0')"}]), str(tmp_path / "d.ipynb"))
    r_py = run_deliverable(py, cwd=sandbox, env=env, timeout=120)
    r_nb = run_deliverable(tmp_path / "d.ipynb", cwd=sandbox, env=env, timeout=120)
    assert r_py.returncode == 0 == r_nb.returncode
    assert "REPRODUCED: 1.0" in r_py.stdout and "REPRODUCED: 1.0" in r_nb.stdout


def test_write_deliverable_accepts_ipynb(tmp_path):
    """WriteDeliverable now accepts .ipynb (valid nbformat JSON); malformed
    notebook JSON is rejected at write time, not deferred to the gate."""
    study_dir = tmp_path / "study"; study_dir.mkdir()

    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "WriteNote", "WriteDeliverable"})
        description = "s"

    class B(Agent):
        description = "i"

    spec = Graph(nodes={"strategizer": A(), "implementer": B()},
                 edges=(Edge("strategizer", "implementer"),), entry="strategizer")
    node = StrategizerNode(_StubAdapter(), name="strategizer",
                           outgoing=["implementer"], spec=spec, study_dir=study_dir)
    wd = node.adapter.closure_tools["WriteDeliverable"]

    good = nbformat.writes(build_notebook([{"type": "code", "source": "print(1)"}]))
    assert "Written" in wd("pipeline.ipynb", good)
    assert (study_dir / "pipeline.ipynb").exists()
    assert "ERROR" in wd("bad.ipynb", "{ this is not notebook json }")


def test_write_deliverable_accepts_declared_required_deliverable(tmp_path):
    """B4 regression (run 20260624T021359): the Done() gate REQUIRES files in
    config.yaml required_deliverables (e.g. replicate.py), so WriteDeliverable
    must accept them verbatim — otherwise gate-vs-tool deadlock. A non-.ipynb
    that is NOT declared is still rejected."""
    study_dir = tmp_path / "study"; study_dir.mkdir()

    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "WriteNote", "WriteDeliverable"})
        description = "s"

    class B(Agent):
        description = "i"

    spec = Graph(nodes={"strategizer": A(), "implementer": B()},
                 edges=(Edge("strategizer", "implementer"),), entry="strategizer")
    node = StrategizerNode(_StubAdapter(), name="strategizer",
                           outgoing=["implementer"], spec=spec, study_dir=study_dir)
    # Simulate __call__ having synced state["required_deliverables"] onto the node.
    node._required_deliverables = ["replicate.py"]
    wd = node.adapter.closure_tools["WriteDeliverable"]

    assert "Written" in wd("replicate.py", "print('reproduce me')\n")
    assert (study_dir / "replicate.py").exists()
    # An undeclared non-notebook file is still rejected.
    assert "ERROR" in wd("rogue.py", "print('nope')\n")
