"""OracleStatus — on-demand read of the CURRENT canonical oracle registration.

register_evaluator_entrypoint() can repoint run_config.json's
evaluator_entrypoint BETWEEN delegations (whenever a datagenerator delegation
authors/extends the generator) — the only prior signal was a one-shot
notification a busy agent could fail to reconcile with its own stated
beliefs, which cost one wasted real evaluation job (run 20260717T014507:
D004's task brief asserted "entrypoint unchanged" when it had already been
repointed to D003's snapshot). OracleStatus lets any agent check the live
value instead of trusting memory.
"""
from __future__ import annotations

import json

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.delegation_log import DelegationLog
from a3dasm._src.nodes import StrategizerNode


class _Stub:
    def __init__(self) -> None:
        self.closure_tools: dict = {}
        self.last_usage: dict = {}
        self.model = "m"

    def invoke(self, messages):
        return ""


def _oracle_status(tmp_path, run_config: dict | None):
    run_dir = tmp_path / "runs" / "T1"
    (run_dir / "debug").mkdir(parents=True)
    if run_config is not None:
        (run_dir / "debug" / "run_config.json").write_text(
            json.dumps(run_config), encoding="utf-8")

    class S(Agent):
        role = "strategizer"
        tools = frozenset({"Done"})
        description = "s"

    class Impl(Agent):
        role = "implementer"
        description = "i"
        tools = frozenset({"OracleStatus"})

    class Lit(Agent):
        role = "literature_reviewer"
        description = "l"

    spec = Graph(
        nodes={"strategizer": S(), "implementer": Impl(),
               "literature_reviewer": Lit()},
        edges=(Edge("strategizer", "implementer"),
               Edge("implementer", "literature_reviewer")),
        entry="strategizer")
    dlog = DelegationLog(run_dir / "debug" / "delegation_log.jsonl")
    n = StrategizerNode(
        _Stub(), name="implementer", outgoing=["literature_reviewer"],
        spec=spec, worker_adapters={"literature_reviewer": _Stub()},
        notes_dir=None, delegation_log=dlog)
    return n._build_routing_closures()["OracleStatus"]


def test_no_run_config_reports_not_registered(tmp_path):
    status = _oracle_status(tmp_path, run_config=None)
    assert "No run_config.json yet" in status()


def test_reports_current_entrypoint(tmp_path):
    status = _oracle_status(tmp_path, {
        "evaluator_entrypoint": "runs/T1/debug/delegations/D003/"
                                 "generators/data_generator.py:Gen",
        "evaluator_output_names": ["sigma_crit", "coilable"],
        "evaluator_lookup": None,
    })
    out = status()
    assert "D003/generators/data_generator.py:Gen" in out
    assert "sigma_crit" in out


def test_reports_lookup_pool_when_no_entrypoint(tmp_path):
    status = _oracle_status(tmp_path, {
        "evaluator_entrypoint": None,
        "evaluator_lookup": {"path": "pool.csv"},
    })
    assert "lookup pool" in status()


def test_reports_not_yet_registered_when_neither_set(tmp_path):
    status = _oracle_status(tmp_path, {
        "evaluator_entrypoint": None, "evaluator_lookup": None,
    })
    assert "NOT YET REGISTERED" in status()


def test_reports_namespace_oracles(tmp_path):
    status = _oracle_status(tmp_path, {
        "evaluator_entrypoint": "workspace/data_generator.py:Gen",
        "oracles": {
            "elliptical_rings": {
                "evaluator_entrypoint": "workspace/rings_generator.py:Gen",
            },
        },
    })
    out = status()
    assert "elliptical_rings" in out
    assert "rings_generator.py:Gen" in out


def test_bad_json_returns_error_not_raises(tmp_path):
    run_dir = tmp_path / "runs" / "T1"
    (run_dir / "debug").mkdir(parents=True)
    (run_dir / "debug" / "run_config.json").write_text(
        "{not valid json", encoding="utf-8")

    class S(Agent):
        role = "strategizer"
        tools = frozenset({"Done"})
        description = "s"

    class Impl(Agent):
        role = "implementer"
        description = "i"
        tools = frozenset({"OracleStatus"})

    class Lit(Agent):
        role = "literature_reviewer"
        description = "l"

    spec = Graph(
        nodes={"strategizer": S(), "implementer": Impl(),
               "literature_reviewer": Lit()},
        edges=(Edge("strategizer", "implementer"),
               Edge("implementer", "literature_reviewer")),
        entry="strategizer")
    dlog = DelegationLog(run_dir / "debug" / "delegation_log.jsonl")
    n = StrategizerNode(
        _Stub(), name="implementer", outgoing=["literature_reviewer"],
        spec=spec, worker_adapters={"literature_reviewer": _Stub()},
        notes_dir=None, delegation_log=dlog)
    status = n._build_routing_closures()["OracleStatus"]
    assert status().startswith("ERROR")
