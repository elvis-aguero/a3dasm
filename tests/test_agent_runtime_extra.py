"""Extra tests for agent_runtime.py — config loading, budget parsing, and
_make_adapter edge cases."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _load_study_config
# ---------------------------------------------------------------------------


def test_load_study_config_returns_empty_dict_when_no_config(tmp_path):
    """_load_study_config returns {} when config.yaml does not exist."""
    from a3dasm._src.agent_runtime import _load_study_config

    result = _load_study_config(tmp_path)
    assert result == {}


def test_load_study_config_reads_yaml_when_present(tmp_path):
    """_load_study_config reads and parses config.yaml."""
    import yaml as _yaml
    from a3dasm._src.agent_runtime import _load_study_config

    config = {"model": "claude-haiku", "budget": 3600}
    (tmp_path / "config.yaml").write_text(_yaml.dump(config))

    result = _load_study_config(tmp_path)
    assert result["model"] == "claude-haiku"
    assert result["budget"] == 3600


def test_load_study_config_returns_empty_for_empty_yaml(tmp_path):
    """_load_study_config returns {} when config.yaml is empty."""
    from a3dasm._src.agent_runtime import _load_study_config

    (tmp_path / "config.yaml").write_text("")

    result = _load_study_config(tmp_path)
    assert result == {}


# ---------------------------------------------------------------------------
# _parse_budget_str
# ---------------------------------------------------------------------------


def test_parse_budget_str_none():
    """_parse_budget_str returns None for None input."""
    from a3dasm._src.agent_runtime import _parse_budget_str

    assert _parse_budget_str(None) is None


def test_parse_budget_str_float():
    """_parse_budget_str returns float for numeric input."""
    from a3dasm._src.agent_runtime import _parse_budget_str

    assert _parse_budget_str(3600.0) == 3600.0
    assert _parse_budget_str(1800) == 1800.0


def test_parse_budget_str_hhmmss():
    """_parse_budget_str parses HH:MM:SS string."""
    from a3dasm._src.agent_runtime import _parse_budget_str

    result = _parse_budget_str("01:30:00")
    assert result == 5400.0  # 1h30m = 5400s


def test_parse_budget_str_float_string():
    """_parse_budget_str parses a plain float string."""
    from a3dasm._src.agent_runtime import _parse_budget_str

    result = _parse_budget_str("7200")
    assert result == 7200.0


# ---------------------------------------------------------------------------
# AgenticRun.__init__ reads config.yaml
# ---------------------------------------------------------------------------


def test_agentic_run_reads_model_from_config(tmp_path):
    """AgenticRun reads model from config.yaml if not passed explicitly."""
    import yaml as _yaml
    from a3dasm._src.agent_runtime import AgenticRun

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")
    (tmp_path / "config.yaml").write_text(_yaml.dump({"model": "claude-opus"}))

    run = AgenticRun(tmp_path)
    assert run._model == "claude-opus"


def test_agentic_run_reads_budget_from_config(tmp_path):
    """AgenticRun reads budget from config.yaml if not passed explicitly."""
    import yaml as _yaml
    from a3dasm._src.agent_runtime import AgenticRun

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")
    (tmp_path / "config.yaml").write_text(_yaml.dump({"budget": "02:00:00"}))

    run = AgenticRun(tmp_path)
    assert run._budget == 7200.0


def test_agentic_run_explicit_budget_overrides_config(tmp_path):
    """AgenticRun explicit budget parameter overrides config.yaml value."""
    import yaml as _yaml
    from a3dasm._src.agent_runtime import AgenticRun

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")
    (tmp_path / "config.yaml").write_text(_yaml.dump({"budget": "02:00:00"}))

    run = AgenticRun(tmp_path, budget=1000)
    assert run._budget == 1000


# ---------------------------------------------------------------------------
# AgenticRun.execute raises AgenticRunError when PROBLEM_STATEMENT.md missing
# ---------------------------------------------------------------------------


def test_execute_raises_when_no_problem_statement(tmp_path):
    """AgenticRun.execute raises AgenticRunError when PROBLEM_STATEMENT.md is missing."""
    from a3dasm._src.agent_runtime import AgenticRun, AgenticRunError

    run = AgenticRun(tmp_path)
    with pytest.raises(AgenticRunError, match="PROBLEM_STATEMENT"):
        run.execute()


# ---------------------------------------------------------------------------
# _make_adapter for ollama backend calls build_closure_tools
# ---------------------------------------------------------------------------


def test_make_adapter_ollama_calls_build_closure_tools(tmp_path):
    """_make_adapter for ollama backend calls agent.build_closure_tools."""
    from a3dasm._src.agent_runtime import AgenticRun, _default_graph, DEFAULT_MODEL
    from a3dasm._src.backends.base import Agent

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")

    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._backend = "ollama"
    run._graph_spec = MagicMock()
    run._graph_spec.outgoing.return_value = []

    run_dir = tmp_path / "runs" / "test"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True, exist_ok=True)
    run._run_dir = run_dir

    class TestAgent(Agent):
        description = "Test."
        tools = frozenset({"Bash"})

    agent = TestAgent()

    mock_closures = {"Done": lambda s: "done"}
    with patch.object(agent, "build_closure_tools", return_value=mock_closures) as mock_build:
        with patch("a3dasm._src.backends.ollama.OllamaAdapter") as MockOllama:
            mock_instance = MagicMock()
            mock_instance.closure_tools = {}
            MockOllama.return_value = mock_instance
            import os
            with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434/v1"}):
                result = run._make_adapter("implementer", agent)

    mock_build.assert_called_once()
    assert "Done" in mock_instance.closure_tools


# ---------------------------------------------------------------------------
# AgenticRun._make_adapter with extra_closures updates adapter
# ---------------------------------------------------------------------------


def test_make_adapter_claude_extra_closures_injected(tmp_path):
    """_make_adapter injects extra_closures from agent.build_closure_tools into adapter."""
    from a3dasm._src.agent_runtime import AgenticRun, _default_graph, DEFAULT_MODEL
    from a3dasm._src.backends.base import Agent

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("test")

    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._backend = "claude"
    run._graph_spec = MagicMock()
    run._graph_spec.outgoing.return_value = []

    run_dir = tmp_path / "runs" / "test"
    (run_dir / "debug" / "strategizer_notes").mkdir(parents=True, exist_ok=True)
    run._run_dir = run_dir

    class TestAgent(Agent):
        description = "Test."
        tools = frozenset({"Bash"})

    agent = TestAgent()

    extra_closures = {"SpecialTool": lambda x: x}
    with patch.object(agent, "build_closure_tools", return_value=extra_closures):
        with patch("a3dasm._src.backends.claude.ClaudeAdapter") as MockClaude:
            mock_instance = MagicMock()
            mock_instance.closure_tools = {}
            MockClaude.return_value = mock_instance

            result = run._make_adapter("implementer", agent)

    assert "SpecialTool" in mock_instance.closure_tools


# ---------------------------------------------------------------------------
# _init_canonical_store helper
# ---------------------------------------------------------------------------


def test_init_canonical_store_creates_dirs(tmp_path):
    """_init_canonical_store creates experiment_data/."""
    from a3dasm._src.agent_runtime import _init_canonical_store

    run_dir = tmp_path / "runs" / "20260101T000000"
    (run_dir / "debug").mkdir(parents=True, exist_ok=True)
    study_dir = tmp_path / "my_study"

    _init_canonical_store(run_dir, study_dir)

    assert (run_dir / "experiment_data").is_dir()
    # eval_counter directory is no longer created (counter removed)
    assert not (run_dir / "debug" / "eval_counter").exists()


def test_init_canonical_store_writes_run_config_json(tmp_path):
    """_init_canonical_store writes run_config.json with correct keys."""
    import json
    from a3dasm._src.agent_runtime import _init_canonical_store

    run_dir = tmp_path / "runs" / "20260101T000000"
    (run_dir / "debug").mkdir(parents=True, exist_ok=True)
    study_dir = tmp_path / "my_study"

    cfg = _init_canonical_store(run_dir, study_dir)

    cfg_path = run_dir / "debug" / "run_config.json"
    assert cfg_path.exists()
    loaded = json.loads(cfg_path.read_text())
    assert loaded["store_dir"] == str(run_dir / "experiment_data")
    assert "counter_dir" not in loaded
    # Lock co-located with the data (store_dir/experiment_data/) so D000
    # ingestion and D001+ evaluations lock the same file (audit Finding 3).
    assert loaded["lock_path"] == str(
        run_dir / "experiment_data" / "experiment_data" / ".lock"
    )
    assert loaded["evaluator_name"] == "my_study"
    assert loaded["fidelity_column"] is None
    assert loaded["evaluator_entrypoint"] is None
    # return value matches written file
    assert cfg == loaded


def test_init_canonical_store_returns_config_dict(tmp_path):
    """_init_canonical_store return value is the config dict."""
    from a3dasm._src.agent_runtime import _init_canonical_store

    run_dir = tmp_path / "runs" / "ts"
    (run_dir / "debug").mkdir(parents=True, exist_ok=True)
    study_dir = tmp_path / "study_x"

    result = _init_canonical_store(run_dir, study_dir)

    assert isinstance(result, dict)
    assert "store_dir" in result
    assert "counter_dir" not in result


def test_execute_creates_canonical_store_dirs_and_state(tmp_path):
    """After execute(), canonical store dirs exist and state has
    experiment_data_dir.
    """
    from langgraph.checkpoint.memory import MemorySaver

    from a3dasm._src.agent_runtime import (
        DEFAULT_MODEL,
        AgenticRun,
        _default_graph,
    )
    from a3dasm._src.graph_builder import build_graph

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("Find min of f(x)=x^2")
    (tmp_path / "pipeline.py").write_text("# test\n")

    state_snapshots: list[dict] = []

    class CapturingStratAdapter:
        closure_tools: dict = {}

        def invoke(self, messages):
            state_snapshots.append(dict(self.closure_tools))
            self.closure_tools["Done"](summary="Done.")
            self.closure_tools["Done"](summary="Done.")
            return "Done."

    class StubImplAdapter:
        closure_tools: dict = {}

        def invoke(self, messages):
            return "## Report\nDone."

    graph_spec = _default_graph()
    compiled = build_graph(
        graph_spec,
        lambda n, a: (
            CapturingStratAdapter()
            if n == "strategizer"
            else StubImplAdapter()
        ),
        MemorySaver(),
        study_dir=tmp_path,
    )

    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._budget = None
    run._graph_spec = graph_spec
    run._graph = compiled

    run.execute()

    # run_dir was set
    assert run._run_dir is not None
    run_dir = run._run_dir

    # dirs exist
    assert (run_dir / "experiment_data").is_dir()
    # eval_counter directory is no longer created (counter removed)
    assert not (run_dir / "debug" / "eval_counter").exists()

    # run_config.json exists with correct keys
    import json

    cfg = json.loads(
        (run_dir / "debug" / "run_config.json").read_text()
    )
    assert cfg["store_dir"] == str(run_dir / "experiment_data")
    assert "counter_dir" not in cfg
    assert cfg["evaluator_name"] == tmp_path.name


# ---------------------------------------------------------------------------
# Change 1: _ingest_precomputed_pool
# ---------------------------------------------------------------------------


def _make_pool(pool_dir: Path, n: int = 5) -> None:
    """Write a tiny ExperimentData pool to pool_dir."""
    from f3dasm import ExperimentData, datagenerator
    from f3dasm.design import Domain
    from f3dasm._src.samplers import RandomUniform

    d = Domain()
    d.add_float("x0", low=0.0, high=1.0)
    d.add_float("x1", low=0.0, high=1.0)
    d.add_output("y")
    data = ExperimentData(domain=d)
    data = RandomUniform(seed=42).call(data, n_samples=n)

    @datagenerator(output_names=["y"])
    def _fn(**kw):
        return float(kw["x0"]) + float(kw["x1"])

    data = _fn.call(data, mode="sequential")
    data.store(project_dir=pool_dir)


def test_ingest_precomputed_pool_d000_rows_in_store(tmp_path):
    """_ingest_precomputed_pool writes D000 rows to the canonical store."""
    from f3dasm import ExperimentData
    from a3dasm._src.agent_runtime import _ingest_precomputed_pool

    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    _make_pool(pool_dir, n=5)

    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()

    lookup_cfg = {"pool": "pool"}
    n = _ingest_precomputed_pool(store_dir, tmp_path, lookup_cfg)

    assert n == 5, f"Expected 5 rows ingested, got {n}"

    canon = ExperimentData.from_file(project_dir=store_dir)
    _, df_out = canon.to_pandas()
    assert len(df_out) == 5
    assert (df_out["_delegation_id"] == "D000").all(), (
        f"Expected all D000, got: {df_out['_delegation_id'].unique()}"
    )
    assert (df_out["_source"] == "precomputed_pool").all()
    assert df_out["_ts"].notna().all()


def test_ingest_precomputed_pool_readable_by_runstatesummary(tmp_path):
    """D000 rows are visible in RunStateSummary.from_store."""
    from a3dasm._src.agent_runtime import _ingest_precomputed_pool
    from a3dasm._src.instrumented import RunStateSummary

    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    _make_pool(pool_dir, n=7)

    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()

    _ingest_precomputed_pool(store_dir, tmp_path, {"pool": "pool"})

    summary = RunStateSummary.from_store(store_dir)
    assert summary is not None, (
        "RunStateSummary.from_store returned None after D000 ingest"
    )
    assert summary.n_per_delegation.get("D000", 0) == 7, (
        f"Expected D000=7, got {summary.n_per_delegation}"
    )


def test_d000_not_counted_as_evals(tmp_path):
    """_resolve_delegation_evals for D000 returns 0 (no real eval rows)."""
    from a3dasm._src.agent_runtime import _ingest_precomputed_pool
    from a3dasm._src.nodes import _resolve_delegation_evals

    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    _make_pool(pool_dir, n=3)

    store_dir = tmp_path / "experiment_data"
    store_dir.mkdir()

    _ingest_precomputed_pool(store_dir, tmp_path, {"pool": "pool"})

    # D000 rows exist in the store, but we never call
    # _resolve_delegation_evals("D000", ...) in production.
    # Verify: if it were called, it would return the row count.
    # More importantly, for a real delegation that doesn't exist (D001),
    # the reported fallback is used — D000 doesn't pollute D001.
    evals_d001 = _resolve_delegation_evals(store_dir, "D001", reported=0)
    assert evals_d001 == 0, (
        f"D000 rows must not inflate D001 eval count; got {evals_d001}"
    )


def test_execute_with_lookup_config_ingests_d000(tmp_path):
    """execute() with evaluator.lookup ingests D000 rows before graph runs."""
    import yaml as _yaml
    from langgraph.checkpoint.memory import MemorySaver

    from a3dasm._src.agent_runtime import (
        DEFAULT_MODEL,
        AgenticRun,
        _default_graph,
    )
    from a3dasm._src.graph_builder import build_graph
    from a3dasm._src.instrumented import RunStateSummary

    # Build pool.
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    _make_pool(pool_dir, n=4)

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("Find min")
    (tmp_path / "pipeline.py").write_text("# test\n")
    (tmp_path / "config.yaml").write_text(_yaml.dump({
        "evaluator": {
            "lookup": {
                "pool": "pool",
                "input_columns": ["x0", "x1"],
                "output_columns": ["y"],
            }
        }
    }))

    class CapturingStratAdapter:
        closure_tools: dict = {}

        def invoke(self, messages):
            self.closure_tools["Done"](summary="Done.")
            self.closure_tools["Done"](summary="Done.")
            return "Done."

    class StubImplAdapter:
        closure_tools: dict = {}

        def invoke(self, messages):
            return "## Report\nDone."

    graph_spec = _default_graph()
    compiled = build_graph(
        graph_spec,
        lambda n, a: (
            CapturingStratAdapter()
            if n == "strategizer"
            else StubImplAdapter()
        ),
        MemorySaver(),
        study_dir=tmp_path,
    )

    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._budget = None
    run._graph_spec = graph_spec
    run._graph = compiled

    run.execute()

    run_dir = run._run_dir
    store_dir = run_dir / "experiment_data"
    summary = RunStateSummary.from_store(store_dir)
    assert summary is not None, (
        "No rows in canonical store after execute() with lookup config"
    )
    assert summary.n_per_delegation.get("D000", 0) == 4, (
        f"Expected D000=4, got {summary.n_per_delegation}"
    )


def test_execute_with_training_data_ingests_d000_no_oracle(tmp_path):
    """A top-level `training_data` pool is ingested as D000 WITHOUT declaring
    any evaluator — get_evaluator() then resolves nothing (surrogate-only
    study). This separates 'training data' from 'lookup oracle'."""
    import json as _json

    import yaml as _yaml
    from langgraph.checkpoint.memory import MemorySaver

    from a3dasm._src.agent_runtime import (
        DEFAULT_MODEL,
        AgenticRun,
        _default_graph,
    )
    from a3dasm._src.graph_builder import build_graph
    from a3dasm._src.instrumented import RunStateSummary

    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    _make_pool(pool_dir, n=4)

    (tmp_path / "PROBLEM_STATEMENT.md").write_text("Surrogate study")
    (tmp_path / "pipeline.py").write_text("# test\n")
    (tmp_path / "config.yaml").write_text(
        _yaml.dump({"training_data": "pool"})
    )

    class CapturingStratAdapter:
        closure_tools: dict = {}

        def invoke(self, messages):
            self.closure_tools["Done"](summary="Done.")
            self.closure_tools["Done"](summary="Done.")
            return "Done."

    class StubImplAdapter:
        closure_tools: dict = {}

        def invoke(self, messages):
            return "## Report\nDone."

    graph_spec = _default_graph()
    compiled = build_graph(
        graph_spec,
        lambda n, a: (
            CapturingStratAdapter()
            if n == "strategizer"
            else StubImplAdapter()
        ),
        MemorySaver(),
        study_dir=tmp_path,
    )
    run = AgenticRun.__new__(AgenticRun)
    run.study_dir = tmp_path
    run._model = DEFAULT_MODEL
    run._budget = None
    run._graph_spec = graph_spec
    run._graph = compiled
    run.execute()

    run_dir = run._run_dir
    summary = RunStateSummary.from_store(run_dir / "experiment_data")
    assert summary is not None and summary.n_per_delegation.get("D000", 0) == 4
    # No oracle declared → run_config carries no entrypoint/lookup.
    cfg = _json.loads((run_dir / "debug" / "run_config.json").read_text())
    assert cfg["evaluator_entrypoint"] is None
    assert cfg["evaluator_lookup"] is None
