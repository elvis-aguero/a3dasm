"""Item A — separable, analysis-ready telemetry subsystem.

Telemetry is *additive*: it records one JSONL row per LLM call (per-PID file,
so parallel worker threads/processes never corrupt each other) and merges them
into an analysis-ready ``summary.json`` broken down by role / phase / model.
It lives off the decision path — the run's behaviour must not depend on it.
"""
from __future__ import annotations

import json
import os

from a3dasm._src.telemetry import Telemetry


def _usage(i, o, cost=None, cr=0, cc=0):
    return {
        "input_tokens": i,
        "output_tokens": o,
        "cache_read_input_tokens": cr,
        "cache_creation_input_tokens": cc,
        "total_cost_usd": cost,
    }


def test_record_call_writes_parseable_jsonl_row(tmp_path):
    tel = Telemetry(tmp_path / "debug")
    tel.record_call(
        role="strategizer", model="claude-x", phase="strategizer_turn",
        delegation_id=None, usage=_usage(100, 50, cost=0.01),
    )
    files = list((tmp_path / "debug" / "telemetry").glob("calls.*.jsonl"))
    assert len(files) == 1
    # the file is named per-PID
    assert files[0].name == f"calls.{os.getpid()}.jsonl"
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert len(rows) == 1
    r = rows[0]
    assert r["role"] == "strategizer"
    assert r["model"] == "claude-x"
    assert r["phase"] == "strategizer_turn"
    assert r["delegation_id"] is None
    assert r["input_tokens"] == 100
    assert r["output_tokens"] == 50
    assert r["total_cost_usd"] == 0.01
    assert "ts" in r


def test_record_call_handles_none_cost(tmp_path):
    """Ollama returns no cost — record_call must not crash and must not
    fabricate a number (None stays None / 0, never an error)."""
    tel = Telemetry(tmp_path / "debug")
    tel.record_call(
        role="datagenerator", model="llama", phase="delegation",
        delegation_id="D001", usage=_usage(10, 5, cost=None),
    )
    f = next((tmp_path / "debug" / "telemetry").glob("calls.*.jsonl"))
    r = json.loads(f.read_text().splitlines()[0])
    assert r["total_cost_usd"] is None


def test_merge_aggregates_by_role_phase_model_and_totals(tmp_path):
    debug = tmp_path / "debug"
    tel = Telemetry(debug)
    tel.record_call(role="strategizer", model="claude-x",
                    phase="strategizer_turn", delegation_id=None,
                    usage=_usage(100, 50, cost=0.01))
    tel.record_call(role="critic", model="claude-x", phase="critic_review",
                    delegation_id="D002", usage=_usage(200, 20, cost=0.02))
    tel.record_call(role="datagenerator", model="claude-x",
                    phase="delegation", delegation_id="D001",
                    usage=_usage(300, 80, cost=0.03))

    summary = Telemetry.merge(debug)

    # summary.json written
    sj = json.loads((debug / "telemetry" / "summary.json").read_text())
    assert sj == summary

    # totals match the row-wise sums exactly
    assert summary["totals"]["calls"] == 3
    assert summary["totals"]["input_tokens"] == 600
    assert summary["totals"]["output_tokens"] == 150
    assert summary["totals"]["total_tokens"] == 750
    assert abs(summary["totals"]["total_cost_usd"] - 0.06) < 1e-9

    # breakdowns present and partition the totals
    assert set(summary["by_role"]) == {"strategizer", "critic", "datagenerator"}
    assert summary["by_role"]["critic"]["input_tokens"] == 200
    assert summary["by_phase"]["delegation"]["output_tokens"] == 80
    assert summary["by_model"]["claude-x"]["calls"] == 3
    # the per-role input-token sum equals the grand total (true partition)
    assert sum(v["input_tokens"] for v in summary["by_role"].values()) == 600


def test_merge_reads_multiple_pid_files(tmp_path):
    """Parallel processes write separate per-PID files; merge unions them."""
    tdir = tmp_path / "debug" / "telemetry"
    tdir.mkdir(parents=True)
    (tdir / "calls.111.jsonl").write_text(
        json.dumps({"role": "critic", "model": "m", "phase": "p",
                    "delegation_id": "D1", "input_tokens": 5,
                    "output_tokens": 1, "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "total_cost_usd": 0.1, "ts": 1.0}) + "\n")
    (tdir / "calls.222.jsonl").write_text(
        json.dumps({"role": "critic", "model": "m", "phase": "p",
                    "delegation_id": "D2", "input_tokens": 7,
                    "output_tokens": 2, "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "total_cost_usd": 0.2, "ts": 2.0}) + "\n")

    summary = Telemetry.merge(tmp_path / "debug")
    assert summary["totals"]["calls"] == 2
    assert summary["totals"]["input_tokens"] == 12
    assert abs(summary["totals"]["total_cost_usd"] - 0.3) < 1e-9


def test_merge_with_no_files_is_empty_not_crash(tmp_path):
    summary = Telemetry.merge(tmp_path / "debug")
    assert summary["totals"]["calls"] == 0
    assert summary["totals"]["total_cost_usd"] == 0.0
    assert summary["by_role"] == {}


def test_claude_allowed_tools_never_none_for_toolless():
    """Gap 1 regression: a tool-less agent (e.g. the problem-statement
    reviewer) must yield allowed_tools=[] — never None — or the SDK crashes on
    list(None) when building its command."""
    from a3dasm._src.backends.claude import ClaudeAdapter

    toolless = ClaudeAdapter(
        model="m", system_prompt="s", native_tools=[], extra_allowed_tools=[]
    )
    assert toolless._compute_allowed_tools([]) == []  # list, NOT None

    withtools = ClaudeAdapter(
        model="m", system_prompt="s", native_tools=["Read"],
        extra_allowed_tools=["mcp__x__y"],
    )
    assert withtools._compute_allowed_tools(["mcp__s__t"]) == [
        "mcp__s__t", "Read", "mcp__x__y",
    ]


def test_critic_consult_usage_recorded(tmp_path):
    """Gap 2 regression: a critic consultation is a real LLM call — its tokens
    and cost must land in BOTH token_totals and telemetry, under role
    'critic' / phase 'critic_review' (previously uncounted)."""
    import json

    from a3dasm._src.backends.base import Agent, Edge, Graph
    from a3dasm._src.nodes import StrategizerNode

    class _Strat(Agent):
        role = "strategizer"
        description = "s"

    class _Crit(Agent):
        role = "critic"
        description = "c"

    spec = Graph(
        nodes={"strategizer": _Strat(), "critic": _Crit()},
        edges=(Edge("strategizer", "critic"),),
        entry="strategizer",
    )

    class _StratStub:
        def __init__(self):
            self.closure_tools: dict = {}
            self.last_usage: dict = {}
            self.model = "m"

        def invoke(self, messages):
            return ""

    class _CriticStub:
        model = "claude-haiku-4-5-20251001"

        def __init__(self):
            self.closure_tools: dict = {}
            self.last_usage: dict = {}

        def copy(self):
            return self

        def invoke(self, messages):
            self.last_usage = {
                "input_tokens": 50, "output_tokens": 30,
                "total_cost_usd": 0.05,
            }
            return "### Verdict\n**PASS**\n### Retrospective\n- CONSISTENCY: ok\n"

    notes = tmp_path / "debug" / "strategizer_notes"
    notes.mkdir(parents=True)
    node = StrategizerNode(
        _StratStub(), name="strategizer", outgoing=["critic"], spec=spec,
        worker_adapters={"critic": _CriticStub()}, notes_dir=notes,
    )
    node._current_notes_dir = notes

    out = node._invoke_critic("FEEDBACK task")
    assert "PASS" in out

    # token_totals picked up the critic's cost
    assert abs(node._token_totals["total_cost_usd"] - 0.05) < 1e-9
    assert node._token_totals["input_tokens"] == 50

    # telemetry recorded exactly one critic_review row
    f = next((tmp_path / "debug" / "telemetry").glob("calls.*.jsonl"))
    rows = [json.loads(line) for line in f.read_text().splitlines()]
    critic = [r for r in rows if r["role"] == "critic"]
    assert len(critic) == 1
    assert critic[0]["phase"] == "critic_review"
    assert critic[0]["model"] == "claude-haiku-4-5-20251001"
    assert critic[0]["total_cost_usd"] == 0.05
    assert critic[0]["delegation_id"] == "critic-1"


def test_record_call_never_raises_into_caller(tmp_path, monkeypatch):
    """Telemetry is off the decision path: a write failure must be swallowed
    so it can never break a run."""
    tel = Telemetry(tmp_path / "debug")

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(tel, "_append_row", _boom)
    # must not raise
    tel.record_call(role="r", model="m", phase="p", delegation_id=None,
                    usage=_usage(1, 1))
