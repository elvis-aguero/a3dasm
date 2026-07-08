"""Tests for Phase 2: evaluator entrypoint resolution.

Tests written BEFORE implementation (TDD).
Covers:
1. Bare-fn file-path entrypoint resolves and evaluates.
2. DataGenerator-class entrypoint resolves (no-args instantiation).
3. Lookup config resolves to LookupDataGenerator.
4. No evaluator config → clear error mentioning ReportEvals fallback.
5. Delegation-ID regression: distinct IDs across loop-back turns.
6. _init_canonical_store writes new keys.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from f3dasm._src.core import DataGenerator
from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_run_config(
    directory: Path,
    store_dir: Path,
    study_dir: Path,
    *,
    evaluator_entrypoint: str | None = None,
    evaluator_output_names: list | None = None,
    evaluator_lookup: dict | None = None,
    fidelity_column: str | None = None,
) -> None:
    cfg = {
        "store_dir": str(store_dir),
        "lock_path": str(store_dir / "experiment_data" / ".lock"),
        "evaluator_name": "test_study",
        "evaluator_entrypoint": evaluator_entrypoint,
        "evaluator_output_names": evaluator_output_names,
        "evaluator_lookup": evaluator_lookup,
        "fidelity_column": fidelity_column,
        "study_dir": str(study_dir),
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run_config.json").write_text(json.dumps(cfg))


def _make_delegation_dir(
    tmp_path: Path, delegation_id: str = "D001"
) -> Path:
    """Build .../debug/delegations/D001 and write run_config one level up."""
    debug_dir = tmp_path / "debug"
    delegation_dir = debug_dir / "delegations" / delegation_id
    delegation_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir, delegation_dir


# ---------------------------------------------------------------------------
# Test 1 — Bare-fn file-path entrypoint
# ---------------------------------------------------------------------------


def test_bare_fn_file_path_entrypoint_resolves(tmp_path, monkeypatch):
    """A file-path:attr entrypoint wraps a bare fn; evaluates one point."""
    from a3dasm._src.instrumented import get_evaluator

    # Write a tiny study evaluator
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    evaluator_src = textwrap.dedent("""\
        def evaluate_kw(**kwargs):
            return float(sum(kwargs.values()))
    """)
    (study_dir / "tiny_eval.py").write_text(evaluator_src)

    # Build delegation workspace
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir, delegation_dir = _make_delegation_dir(tmp_path)

    _write_run_config(
        debug_dir,
        store_dir,
        study_dir,
        evaluator_entrypoint="tiny_eval.py:evaluate_kw",
        evaluator_output_names=["f"],
    )

    monkeypatch.chdir(delegation_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)

    gen = get_evaluator()  # no inner — resolves from config

    # Execute a sample
    domain = Domain()
    domain.add_float("x0", 0.0, 1.0)
    domain.add_float("x1", 0.0, 1.0)
    sample = ExperimentSample(
        _input_data={"x0": 0.3, "x1": 0.2},
        _output_data={},
        job_status=JobStatus.OPEN,
    )
    out = gen.execute(sample)

    assert out._output_data["f"] == pytest.approx(0.5, abs=1e-9)
    assert out._output_data["_delegation_id"] == "D001"

    # Check the canonical store has a row
    data = ExperimentData.from_file(project_dir=store_dir)
    _, df_out = data.to_pandas()
    assert len(df_out) == 1
    assert df_out.iloc[0]["_delegation_id"] == "D001"


def test_run_config_resolves_via_env_var_independent_of_cwd(
    tmp_path, monkeypatch
):
    """F3DASM_RUN_CONFIG resolves get_evaluator() regardless of cwd.

    Regression: the SDK spawns the worker with cwd=study_dir, but run_config.json
    lives DOWN at runs/<id>/debug/ — a walk-UP never reaches it, so the worker
    had to cd into debug/ first. The env var points straight at the file.
    """
    from a3dasm._src.instrumented import get_evaluator

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "tiny_eval.py").write_text(
        "def evaluate_kw(**kwargs):\n    return float(sum(kwargs.values()))\n"
    )
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir, _delegation_dir = _make_delegation_dir(tmp_path)
    _write_run_config(
        debug_dir, store_dir, study_dir,
        evaluator_entrypoint="tiny_eval.py:evaluate_kw",
        evaluator_output_names=["f"],
    )

    # cwd = study_dir (as the SDK sets it): run_config.json is NOT on the
    # walk-up path from here. Only the env var can resolve it.
    monkeypatch.chdir(study_dir)
    monkeypatch.setenv("F3DASM_DELEGATION_ID", "D001")
    monkeypatch.setenv("F3DASM_RUN_CONFIG", str(debug_dir / "run_config.json"))

    gen = get_evaluator()
    sample = ExperimentSample(
        _input_data={"x0": 0.3, "x1": 0.2}, _output_data={},
        job_status=JobStatus.OPEN)
    out = gen.execute(sample)
    assert out._output_data["f"] == pytest.approx(0.5, abs=1e-9)
    assert out._output_data["_delegation_id"] == "D001"


# ---------------------------------------------------------------------------
# Test 2 — DataGenerator-class entrypoint
# ---------------------------------------------------------------------------


def test_datagenerator_class_entrypoint_resolves(tmp_path, monkeypatch):
    """A file-path:ClassAttr entrypoint; class is no-args-instantiated."""
    from a3dasm._src.instrumented import get_evaluator

    study_dir = tmp_path / "study"
    study_dir.mkdir()

    gen_src = textwrap.dedent("""\
        from f3dasm._src.core import DataGenerator
        from f3dasm._src.experimentsample import ExperimentSample, JobStatus

        class DoubleGen(DataGenerator):
            def execute(self, experiment_sample, **kwargs):
                v = experiment_sample._input_data.get("x0", 0.0)
                experiment_sample._output_data["y"] = v * 2
                experiment_sample.job_status = JobStatus.FINISHED
                return experiment_sample
    """)
    (study_dir / "mygen.py").write_text(gen_src)

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir, delegation_dir = _make_delegation_dir(tmp_path)

    _write_run_config(
        debug_dir,
        store_dir,
        study_dir,
        evaluator_entrypoint="mygen.py:DoubleGen",
    )

    monkeypatch.chdir(delegation_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)

    gen = get_evaluator()

    sample = ExperimentSample(
        _input_data={"x0": 3.0},
        _output_data={},
        job_status=JobStatus.OPEN,
    )
    out = gen.execute(sample)
    assert out._output_data["y"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Test 3 — Lookup config resolves to LookupDataGenerator
# ---------------------------------------------------------------------------


def _build_tiny_pool(project_dir: Path) -> None:
    """Write a 3-row ExperimentData pool to project_dir."""
    domain = Domain()
    domain.add_float("a", 0.0, 1.0)
    domain.add_float("b", 0.0, 1.0)
    domain.add_output("z", exist_ok=True)
    rows = {
        0: ExperimentSample(
            _input_data={"a": 0.0, "b": 0.0},
            _output_data={"z": 0.0},
            job_status=JobStatus.FINISHED,
        ),
        1: ExperimentSample(
            _input_data={"a": 1.0, "b": 0.0},
            _output_data={"z": 1.0},
            job_status=JobStatus.FINISHED,
        ),
        2: ExperimentSample(
            _input_data={"a": 0.0, "b": 1.0},
            _output_data={"z": 2.0},
            job_status=JobStatus.FINISHED,
        ),
    }
    pool = ExperimentData.from_data(data=rows, domain=domain)
    pool.store(project_dir=project_dir)


def test_lookup_config_resolves_to_lookup_data_generator(
    tmp_path, monkeypatch
):
    """evaluator_lookup config builds a LookupDataGenerator."""
    from a3dasm._src.instrumented import get_evaluator
    from a3dasm._src.lookup import LookupDataGenerator

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    pool_project = study_dir / "mypool"
    pool_project.mkdir()
    _build_tiny_pool(pool_project)

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir, delegation_dir = _make_delegation_dir(tmp_path)

    _write_run_config(
        debug_dir,
        store_dir,
        study_dir,
        evaluator_lookup={
            "pool": "mypool",
            "input_columns": ["a", "b"],
            "output_columns": ["z"],
        },
    )

    monkeypatch.chdir(delegation_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)

    gen = get_evaluator()

    # The inner generator should be a LookupDataGenerator
    assert isinstance(gen.inner, LookupDataGenerator)

    # Execute a point near (0, 0) — should return z=0
    sample = ExperimentSample(
        _input_data={"a": 0.05, "b": 0.05},
        _output_data={},
        job_status=JobStatus.OPEN,
    )
    out = gen.execute(sample)
    assert out._output_data["z"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 4 — No evaluator config → clear error mentioning ReportEvals
# ---------------------------------------------------------------------------


def test_no_evaluator_config_raises_clear_error(
    tmp_path, monkeypatch
):
    """get_evaluator() with no entrypoint raises ValueError mentioning
    ReportEvals fallback."""
    from a3dasm._src.instrumented import get_evaluator

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    debug_dir, delegation_dir = _make_delegation_dir(tmp_path)

    _write_run_config(
        debug_dir,
        store_dir,
        study_dir,
        evaluator_entrypoint=None,
        evaluator_lookup=None,
    )

    monkeypatch.chdir(delegation_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)

    with pytest.raises(ValueError, match="ReportEvals"):
        get_evaluator()


# ---------------------------------------------------------------------------
# Test 5 — Delegation-ID regression: distinct IDs across loop-back turns
# ---------------------------------------------------------------------------


def test_delegation_ids_distinct_across_loopback_turns(tmp_path):
    """Two Delegate() calls in separate turns get distinct IDs.

    Simulates: turn 1 → D001 fires; registry reset on loop-back →
    turn 2 → must NOT re-use D001.
    """
    from a3dasm._src.backends.base import Agent, Edge, Graph
    from a3dasm._src.nodes import StrategizerNode

    class A(Agent):
        role = "strategizer"
        tools = frozenset({"Done", "FollowUp", "WriteNote", "ReadNote"})
        description = "Test strategizer."

    class B(Agent):
        description = "Test implementer."

    spec = Graph(
        nodes={"strat": A(), "impl": B()},
        edges=(Edge("strat", "impl"),),
        entry="strat",
    )

    class ImmediateWorker:
        closure_tools: dict = {}

        def invoke(self, messages):
            return (
                "## Report\n\n"
                "### Actions taken\n- Did stuff\n\n"
                "### Files touched\n- none\n\n"
                "### Conclusions\nAll good\n\n"
                "### Numbers\n- n: 0\n"
            )

        def copy(self):
            return self

    worker = ImmediateWorker()

    from unittest.mock import MagicMock

    strat_adapter = MagicMock()
    strat_adapter.closure_tools = {}
    strat_adapter.route_watcher = None

    node = StrategizerNode(
        adapter=strat_adapter,
        name="strat",
        outgoing=["impl"],
        spec=spec,
        study_dir=tmp_path,
        worker_adapters={"impl": worker},
    )

    # Closures are stored on adapter.closure_tools by __init__
    closures = strat_adapter.closure_tools

    # Collect delegation IDs from two Delegate calls in different turns.
    ids: list[str] = []

    # Turn 1: fire delegation, collect ID
    closures["Delegate"](
        target="impl",
        intent="Task 1",
        expected_report="",
        wait=True,
    )
    # Introspect the registry to get the ID assigned.
    with node._registry_lock:
        ids.extend(list(node._registry.keys()))

    # Simulate loop-back: reset registry (as __call__ does) — only
    # Working/FollowUp survive, so done D001 is purged.
    with node._registry_lock:
        node._registry = {
            d: e
            for d, e in node._registry.items()
            if e["status"] in ("Working", "FollowUp")
        }

    # Turn 2: fire another delegation
    closures["Delegate"](
        target="impl",
        intent="Task 2",
        expected_report="",
        wait=True,
    )
    with node._registry_lock:
        ids.extend(list(node._registry.keys()))

    # IDs must be unique
    assert len(set(ids)) == len(ids), (
        f"Duplicate delegation IDs detected: {ids}"
    )
    # Specifically: second ID must NOT be D001
    assert ids[1] != ids[0], (
        f"Expected distinct IDs, got {ids}"
    )


# ---------------------------------------------------------------------------
# Test 6 — _init_canonical_store writes new keys
# ---------------------------------------------------------------------------


def test_init_canonical_store_writes_evaluator_keys(tmp_path):
    """_init_canonical_store with evaluator config writes all Phase 2 keys."""
    import json
    from a3dasm._src.agent_runtime import _init_canonical_store

    run_dir = tmp_path / "runs" / "ts"
    (run_dir / "debug").mkdir(parents=True, exist_ok=True)
    study_dir = tmp_path / "my_study"

    eval_cfg = {
        "entrypoint": "workspace/evaluator.py:evaluate_kw",
        "output_names": ["f"],
    }
    cfg = _init_canonical_store(
        run_dir, study_dir, evaluator_config=eval_cfg
    )

    loaded = json.loads(
        (run_dir / "debug" / "run_config.json").read_text()
    )

    assert loaded["evaluator_entrypoint"] == (
        "workspace/evaluator.py:evaluate_kw"
    )
    assert loaded["evaluator_output_names"] == ["f"]
    assert loaded["evaluator_lookup"] is None
    assert "study_dir" in loaded
    assert loaded["study_dir"] == str(study_dir)
    assert cfg == loaded


def test_init_canonical_store_lookup_cfg_writes_keys(tmp_path):
    """_init_canonical_store with lookup config writes evaluator_lookup."""
    import json
    from a3dasm._src.agent_runtime import _init_canonical_store

    run_dir = tmp_path / "runs" / "ts"
    (run_dir / "debug").mkdir(parents=True, exist_ok=True)
    study_dir = tmp_path / "my_study"

    eval_cfg = {
        "lookup": {
            "pool": "experiment_data",
            "input_columns": ["x0", "x1"],
            "output_columns": ["y"],
        }
    }
    cfg = _init_canonical_store(
        run_dir, study_dir, evaluator_config=eval_cfg
    )

    loaded = json.loads(
        (run_dir / "debug" / "run_config.json").read_text()
    )

    assert loaded["evaluator_lookup"] == eval_cfg["lookup"]
    assert loaded["evaluator_entrypoint"] is None
    assert loaded["evaluator_output_names"] is None


def test_init_canonical_store_no_eval_cfg_backward_compat(tmp_path):
    """_init_canonical_store with no evaluator_config: existing tests pass."""
    import json
    from a3dasm._src.agent_runtime import _init_canonical_store

    run_dir = tmp_path / "runs" / "ts"
    (run_dir / "debug").mkdir(parents=True, exist_ok=True)
    study_dir = tmp_path / "my_study"

    cfg = _init_canonical_store(run_dir, study_dir)

    loaded = json.loads(
        (run_dir / "debug" / "run_config.json").read_text()
    )

    # Old keys still present
    assert loaded["evaluator_entrypoint"] is None
    assert loaded["evaluator_name"] == "my_study"
    # New keys present with null defaults
    assert loaded["evaluator_lookup"] is None
    assert loaded["evaluator_output_names"] is None
    assert "study_dir" in loaded


# ---------------------------------------------------------------------------
# config.yaml output_names write-back guardrail (keeps the human-facing config
# in sync with the registered objective column — prevents the stale-config /
# split-objective-column failure where an agent builds a pipeline around the
# wrong column).
# ---------------------------------------------------------------------------

def test_sync_config_output_names_surgical_preserves_comments(tmp_path):
    from a3dasm._src.agent_runtime import _sync_config_output_names
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "model: claude-haiku-4-5\n"
        "# the evaluator block is the schema\n"
        "evaluator:\n"
        "  entrypoint: \"workspace/evaluator.py:evaluate_kw\"\n"
        "  output_names: [f]\n"
    )
    assert _sync_config_output_names(cfg, ["y"]) is True
    text = cfg.read_text()
    assert "output_names: [y]" in text
    assert "output_names: [f]" not in text
    # comments + other keys untouched
    assert "# the evaluator block is the schema" in text
    assert "model: claude-haiku-4-5" in text
    # idempotent: re-syncing the same names is a no-op
    assert _sync_config_output_names(cfg, ["y"]) is False


def test_sync_config_output_names_noop_when_key_absent(tmp_path):
    from a3dasm._src.agent_runtime import _sync_config_output_names
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: claude-haiku-4-5\nbudget: \"00:15:00\"\n")
    assert _sync_config_output_names(cfg, ["y"]) is False
    assert "output_names" not in cfg.read_text()  # not guessed/added


def test_register_writes_back_to_config_yaml(tmp_path):
    """register_evaluator_entrypoint syncs config.yaml's output_names to what
    was actually registered — the end-to-end guardrail."""
    from a3dasm._src.agent_runtime import register_evaluator_entrypoint
    study = tmp_path / "study"
    (study / "debug").mkdir(parents=True)
    (study / "config.yaml").write_text(
        "evaluator:\n  entrypoint: \"workspace/evaluator.py:evaluate_kw\"\n"
        "  output_names: [f]\n")
    rc = study / "debug" / "run_config.json"
    rc.write_text(json.dumps({"study_dir": str(study), "store_dir": str(study)}))
    (study / "gen.py").write_text("def g(**k):\n    return 0.0\n")
    register_evaluator_entrypoint(rc, "gen.py", "g", output_names=["y"])
    # run_config updated
    assert json.loads(rc.read_text())["evaluator_output_names"] == ["y"]
    # config.yaml written back (no longer stale)
    assert "output_names: [y]" in (study / "config.yaml").read_text()
