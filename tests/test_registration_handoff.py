"""Phase 2: the datagenerator -> oracle auto-registration hook.

When a delegation to a node with role=="datagenerator" completes and left a
registration.json manifest in its workspace, the runtime points the canonical
evaluator entrypoint at the authored generator (best-effort).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from a3dasm._src.backends.base import Agent, Edge, Graph
from a3dasm._src.nodes import StrategizerNode

from tests.test_evaluator_resolution import _write_run_config


class _Strat(Agent):
    role = "strategizer"
    tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote"})
    description = "Test strategizer."


class _DataGen(Agent):
    role = "datagenerator"
    description = "Test datagenerator."


class _Impl(Agent):
    role = "implementer"
    description = "Test implementer."


class _Worker:
    closure_tools: dict = {}

    def invoke(self, messages):
        return (
            "## Report\n\n### Actions taken\n- built\n\n"
            "### Files touched\n- gen\n\n### Conclusions\nok\n\n"
            "### Numbers\n- n: 0\n"
        )

    def copy(self):
        return self


def _build(tmp_path, target_name, target_agent):
    """Return (node, closures, run_dir, cfg_path) wired for one Delegate."""
    study_dir = tmp_path / "study"
    run_dir = study_dir / "runs" / "T"
    debug_dir = run_dir / "debug"
    store_dir = run_dir / "experiment_data"
    store_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(debug_dir, store_dir, study_dir)

    spec = Graph(
        nodes={"strat": _Strat(), target_name: target_agent},
        edges=(Edge("strat", target_name),),
        entry="strat",
    )
    strat_adapter = MagicMock()
    strat_adapter.closure_tools = {}
    strat_adapter.route_watcher = None
    worker = _Worker()
    node = StrategizerNode(
        adapter=strat_adapter,
        name="strat",
        outgoing=[target_name],
        spec=spec,
        study_dir=study_dir,
        worker_adapters={target_name: worker},
    )
    node._current_notes_dir = debug_dir / "strategizer_notes"
    return node, strat_adapter.closure_tools, run_dir, debug_dir / "run_config.json"


def _drop_manifest(run_dir, delegation_id="D001", attr="x_gen"):
    gens = run_dir / "debug" / "delegations" / delegation_id / "generators"
    gens.mkdir(parents=True, exist_ok=True)
    (gens / "x.py").write_text(
        f"def {attr}(**kw):\n    return float(sum(kw.values()))\n"
    )
    (gens / "registration.json").write_text(json.dumps({
        "generator_file": "x.py", "attr": attr, "output_names": ["f"],
    }))


def test_datagenerator_manifest_triggers_registration(tmp_path):
    node, closures, run_dir, cfg_path = _build(
        tmp_path, "datagen", _DataGen()
    )
    _drop_manifest(run_dir, "D001")  # first Delegate id is D001
    closures["Delegate"](
        target="datagen", intent="build oracle",
        expected_report="", wait=True,
    )
    cfg = json.loads(cfg_path.read_text())
    assert cfg["evaluator_entrypoint"].endswith(
        "runs/T/debug/delegations/D001/generators/x.py:x_gen"
    )
    assert cfg["evaluator_output_names"] == ["f"]
    assert cfg["evaluator_lookup"] is None


def test_non_datagenerator_target_does_not_register(tmp_path):
    node, closures, run_dir, cfg_path = _build(tmp_path, "impl", _Impl())
    _drop_manifest(run_dir, "D001")  # manifest present but target is implementer
    before = cfg_path.read_text()
    closures["Delegate"](
        target="impl", intent="run", expected_report="", wait=True,
    )
    assert cfg_path.read_text() == before  # unchanged


def test_missing_manifest_no_crash(tmp_path):
    node, closures, run_dir, cfg_path = _build(
        tmp_path, "datagen", _DataGen()
    )
    before = cfg_path.read_text()
    result = closures["Delegate"](
        target="datagen", intent="no manifest",
        expected_report="", wait=True,
    )
    assert "Done" in result  # delegation still completed
    assert cfg_path.read_text() == before  # config untouched
