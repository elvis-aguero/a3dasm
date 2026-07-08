"""Any node granted the read tools must resolve its run context so
RecallStore/QueryStore and HypothesisList/HypothesisGet actually work — on the
entry strategizer, a delegating worker, and a leaf worker (critic) alike. Tools
are declaration-gated (single source of truth = the Agent's `tools`).

Regression for runs 20260705T181941 and 20260706T204732: the implementer and
datagenerator have outgoing edges (to the literature_reviewer), so
graph_builder wraps them as StrategizerNodes and injects the read tools — but
passes notes_dir only to the entry node. That left worker `_current_notes_dir`
and `_ledger` both None, so RecallStore/QueryStore returned "Canonical store is
empty" and HypothesisList/HypothesisGet returned "hypothesis ledger not
available" against stores holding hundreds of rows. Six worker delegations
across the two runs hit this and fell back to hand-rolled
ExperimentData.from_file / reading hypotheses.json.
"""
from __future__ import annotations

from pathlib import Path

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.hypothesis_ledger import HypothesisLedger
from a3dasm._src.nodes import StrategizerNode
from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def _build_store(store_dir: Path, rows) -> None:
    domain = Domain()
    domain.add_float("x0", 0.0, 10.0)
    for k in ("f", "_delegation_id", "_source", "_ts"):
        domain.add_output(k, exist_ok=True)
    samples = {}
    for i, (x0, f, did) in enumerate(rows):
        samples[i] = ExperimentSample(
            _input_data={"x0": x0},
            _output_data={"f": f, "_delegation_id": did, "_source": "test",
                          "_ts": "2026-01-01T00:00:00+00:00"},
            job_status=JobStatus.FINISHED,
        )
    ExperimentData.from_data(data=samples, domain=domain).store(
        project_dir=store_dir)


def _worker_node(run_dir: Path) -> StrategizerNode:
    """A StrategizerNode built the way graph_builder builds a delegating
    worker: an orchestrating node (outgoing edge) with notes_dir=None."""
    class S(Agent):
        role = "strategizer"
        tools = frozenset({"Done"})
        description = "s"

    class Impl(Agent):
        role = "implementer"
        description = "i"
        # Declaration-driven: the worker must DECLARE the read tools to get them.
        tools = frozenset({"RecallStore", "QueryStore",
                           "HypothesisList", "HypothesisGet"})

    class Lit(Agent):
        role = "literature_reviewer"
        description = "l"

    spec = Graph(
        nodes={"strategizer": S(), "implementer": Impl(),
               "literature_reviewer": Lit()},
        edges=(Edge("strategizer", "implementer"),
               Edge("implementer", "literature_reviewer")),
        entry="strategizer",
    )
    dlog = DelegationLog(run_dir / "debug" / "delegation_log.jsonl")
    return StrategizerNode(
        _Stub(), name="implementer", outgoing=["literature_reviewer"],
        spec=spec, worker_adapters={"literature_reviewer": _Stub()},
        notes_dir=None, delegation_log=dlog,
    )


def _setup(tmp_path):
    run_dir = tmp_path / "runs" / "T1"
    (run_dir / "debug").mkdir(parents=True)
    _build_store(run_dir / "experiment_data",
                 [(0.1, 1.0, "D001"), (0.2, 2.0, "D001"), (0.3, 3.0, "D002")])
    notes = run_dir / "debug" / "strategizer_notes"
    notes.mkdir()
    led = HypothesisLedger(notes)
    hid = led.propose("thin walls buckle first", "a sweep finds f>=2.0",
                      "dense sweep finds nothing below 1.5", 0.5, "strategizer")
    return run_dir, hid


def test_worker_resolves_run_dir_from_delegation_log(tmp_path):
    run_dir, _ = _setup(tmp_path)
    n = _worker_node(run_dir)
    assert n._current_notes_dir is None          # the bug's precondition
    assert n._resolve_run_dir() == run_dir        # resolved via delegation log


def test_worker_recallstore_and_querystore_see_the_rows(tmp_path):
    run_dir, _ = _setup(tmp_path)
    n = _worker_node(run_dir)
    routing = n._build_routing_closures()
    rec = routing["RecallStore"]()
    assert "empty" not in rec.lower(), f"RecallStore falsely empty: {rec!r}"
    q = routing["QueryStore"]()
    assert "empty" not in q.lower(), f"QueryStore falsely empty: {q!r}"


def test_worker_hypothesislist_sees_the_ledger(tmp_path):
    run_dir, hid = _setup(tmp_path)
    n = _worker_node(run_dir)
    # HypothesisList is now a declaration-gated read tool merged into the
    # routing closures (same builder leaf workers use).
    lst = n._build_routing_closures()["HypothesisList"]()
    assert "not available" not in lst.lower(), lst
    assert hid in lst, f"expected {hid} in listing: {lst!r}"


# ---------------------------------------------------------------------------
# Leaf WorkerNode (e.g. the critic): declaration-gated read tools, working
# ---------------------------------------------------------------------------

def _leaf_worker(run_dir, agent_tools):
    """A leaf WorkerNode (no outgoing edges) with a stub adapter — the critic /
    lit-reviewer shape."""
    from a3dasm._src.nodes.worker import WorkerNode

    class _A:
        def __init__(self):
            self.closure_tools: dict = {}
            self.native_tools: list = []
            self.last_usage: dict = {}
            self.model = "m"

        def invoke(self, messages):
            return ""

    dlog = DelegationLog(run_dir / "debug" / "delegation_log.jsonl")
    return WorkerNode(_A(), name="critic", delegation_log=dlog,
                      agent_tools=frozenset(agent_tools))


def test_leaf_worker_gets_declared_read_tools_and_they_work(tmp_path):
    run_dir, hid = _setup(tmp_path)
    n = _leaf_worker(run_dir, {"RecallStore", "QueryStore",
                               "HypothesisList", "HypothesisGet"})
    ct = n.adapter.closure_tools
    assert {"RecallStore", "QueryStore", "HypothesisList",
            "HypothesisGet"} <= set(ct)
    assert "empty" not in ct["RecallStore"]().lower()
    assert hid in ct["HypothesisList"]()


def test_leaf_worker_without_declaration_has_no_read_tools(tmp_path):
    run_dir, _ = _setup(tmp_path)
    n = _leaf_worker(run_dir, {"Read", "Glob"})
    ct = n.adapter.closure_tools
    assert "RecallStore" not in ct
    assert "QueryStore" not in ct
    assert "HypothesisList" not in ct
