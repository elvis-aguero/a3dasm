"""Tests for the viewer's pure filesystem-reading functions — no HTTP."""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import pytest

from a3dasm._src.viewer.readers import (
    graph_spec_json,
    load_graph_for_study,
    read_delegations,
    read_diagnostics_tail,
    read_run_status,
    read_runs,
    read_transcript,
    tail_jsonl,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# read_runs
# ---------------------------------------------------------------------------

def test_read_runs_empty_when_no_runs_dir(tmp_path):
    assert read_runs(tmp_path) == []


def test_read_runs_excludes_non_run_siblings(tmp_path):
    """A cache dir like `lit_reviewer_notes` sits alongside real run dirs
    under runs/ but has no debug/ subdir — must not be listed as a run."""
    runs = tmp_path / "runs"
    (runs / "lit_reviewer_notes" / "papers").mkdir(parents=True)
    (runs / "20260904T120000" / "debug").mkdir(parents=True)
    (runs / "20260904T120000" / "debug" / "delegation_log.jsonl").write_text("")

    result = read_runs(tmp_path)
    assert [r["run_id"] for r in result] == ["20260904T120000"]


def test_read_runs_status_running_when_no_run_status_json(tmp_path):
    runs = tmp_path / "runs" / "20260904T120000" / "debug"
    runs.mkdir(parents=True)
    (runs / "delegation_log.jsonl").write_text("")

    result = read_runs(tmp_path)
    assert result[0]["status"] == "running"


def test_read_runs_status_from_run_status_json(tmp_path):
    runs = tmp_path / "runs" / "20260904T120000" / "debug"
    runs.mkdir(parents=True)
    (runs / "run_status.json").write_text(json.dumps({"status": "GATED"}))

    result = read_runs(tmp_path)
    assert result[0]["status"] == "GATED"


# ---------------------------------------------------------------------------
# read_delegations — the collapse-by-id contract
# ---------------------------------------------------------------------------

def test_read_delegations_collapses_running_then_done(tmp_path):
    run_dir = tmp_path / "run"
    log = run_dir / "debug" / "delegation_log.jsonl"
    _write_jsonl(log, [
        {"id": "D001", "status": "RUNNING", "from_node": "strategizer",
         "to_node": "critic"},
        {"id": "D001", "status": "DONE", "from_node": "strategizer",
         "to_node": "critic"},
    ])

    rows = read_delegations(run_dir)
    assert len(rows) == 1
    assert rows[0]["status"] == "DONE"


def test_read_delegations_empty_when_log_missing(tmp_path):
    assert read_delegations(tmp_path / "run") == []


# ---------------------------------------------------------------------------
# read_diagnostics_tail
# ---------------------------------------------------------------------------

def test_read_diagnostics_tail_empty_when_missing(tmp_path):
    assert read_diagnostics_tail(tmp_path / "run") == []


def test_read_diagnostics_tail_parses_rows(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "debug" / "diagnostics.jsonl", [
        {"ts": "t1", "node": "implementer", "error_type": "TimeoutError"},
    ])
    rows = read_diagnostics_tail(run_dir)
    assert rows == [{"ts": "t1", "node": "implementer", "error_type": "TimeoutError"}]


# ---------------------------------------------------------------------------
# read_run_status
# ---------------------------------------------------------------------------

def test_read_run_status_none_when_missing(tmp_path):
    assert read_run_status(tmp_path / "run") is None


def test_read_run_status_parses_normal_close(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "debug").mkdir(parents=True)
    (run_dir / "debug" / "run_status.json").write_text(
        json.dumps({"status": "GATED", "stop_reason": None}))
    assert read_run_status(run_dir) == {"status": "GATED", "stop_reason": None}


def test_read_run_status_parses_crash(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "debug").mkdir(parents=True)
    (run_dir / "debug" / "run_status.json").write_text(
        json.dumps({"status": "crashed", "resumable": True}))
    assert read_run_status(run_dir)["status"] == "crashed"


# ---------------------------------------------------------------------------
# read_transcript — the "debug flag off" vs "key not found" distinction
# ---------------------------------------------------------------------------

def test_read_transcript_none_when_transcripts_dir_absent(tmp_path):
    """transcripts/ entirely absent means the debug flag was off for this
    run — a distinct, honestly-different condition from an unknown key."""
    assert read_transcript(tmp_path / "run", "D007") is None


def test_read_transcript_empty_when_key_not_found(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "debug" / "transcripts").mkdir(parents=True)
    assert read_transcript(run_dir, "D999") == []


def test_read_transcript_parses_flat_delegation_file(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "debug" / "transcripts" / "D007.jsonl", [
        {"ts": "t1", "type": "assistant", "text": "hi", "tools": []},
    ])
    events = read_transcript(run_dir, "D007")
    assert events == [{"ts": "t1", "type": "assistant", "text": "hi", "tools": []}]


def test_read_transcript_parses_nested_strategizer_turn(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(
        run_dir / "debug" / "transcripts" / "strategizer" / "turn_003.jsonl",
        [{"ts": "t1", "type": "assistant", "text": "ok"}],
    )
    events = read_transcript(run_dir, "strategizer/turn_003")
    assert events == [{"ts": "t1", "type": "assistant", "text": "ok"}]


# ---------------------------------------------------------------------------
# graph_spec_json — reuses run_diagram's real layout/tool logic
# ---------------------------------------------------------------------------

def test_graph_spec_json_reuses_bfs_layers_and_node_tools():
    from a3dasm._src.backends.base import Agent, Edge, Graph
    from a3dasm._src.run_diagram import _bfs_layers, _node_tools

    class _Strategizer(Agent):
        role = "strategizer"
        description = "hub"
        tools = frozenset({"Delegate"})

    class _Critic(Agent):
        role = "critic"
        description = "gate"
        tools = frozenset({"Bash"})

    graph = Graph(
        nodes={"strategizer": _Strategizer(), "critic": _Critic()},
        edges=(Edge("strategizer", "critic"),),
        entry="strategizer",
    )

    spec = graph_spec_json(graph)

    expected_layers = _bfs_layers(graph)
    by_name = {n["name"]: n for n in spec["nodes"]}
    for name, node in by_name.items():
        assert node["layer"] == expected_layers[name]
        assert node["tools"] == _node_tools(name, graph.nodes[name], graph, None)

    assert spec["entry"] == "strategizer"
    assert spec["edges"] == [{"source": "strategizer", "target": "critic"}]
    assert by_name["strategizer"]["is_entry"] is True
    assert by_name["critic"]["is_entry"] is False


# ---------------------------------------------------------------------------
# load_graph_for_study — recovering a study's Graph with no in-memory object
# ---------------------------------------------------------------------------

def test_load_graph_for_study_uses_run_py_build_graph(tmp_path):
    (tmp_path / "run.py").write_text(
        "from a3dasm._src.backends.base import Agent, Edge, Graph\n"
        "class _S(Agent):\n"
        "    role = 'strategizer'\n"
        "    description = 'hub'\n"
        "    tools = frozenset()\n"
        "def build_graph():\n"
        "    return Graph(nodes={'strategizer': _S()}, edges=(), "
        "entry='strategizer')\n",
        encoding="utf-8",
    )
    graph = load_graph_for_study(tmp_path)
    assert graph is not None
    assert set(graph.nodes) == {"strategizer"}
    assert graph.entry == "strategizer"


def test_load_graph_for_study_falls_back_to_default_graph_when_no_run_py(tmp_path):
    from a3dasm._src.agents import _default_graph

    graph = load_graph_for_study(tmp_path)
    expected = _default_graph()
    assert set(graph.nodes) == set(expected.nodes)
    assert graph.entry == expected.entry


def test_load_graph_for_study_falls_back_when_run_py_has_no_build_graph(tmp_path):
    (tmp_path / "run.py").write_text("X = 1\n", encoding="utf-8")
    from a3dasm._src.agents import _default_graph

    graph = load_graph_for_study(tmp_path)
    expected = _default_graph()
    assert set(graph.nodes) == set(expected.nodes)


# ---------------------------------------------------------------------------
# tail_jsonl
# ---------------------------------------------------------------------------

def _try_next(gen, timeout):
    """Returns ("ok", value), ("err", exc), or ("timeout", None) — never
    raises itself, so callers can assert on EITHER outcome (a test that
    expects no value within the deadline is a legitimate assertion, not a
    failure of this helper)."""
    q: queue.Queue = queue.Queue()

    def _run():
        try:
            q.put(("ok", next(gen)))
        except Exception as exc:  # noqa: BLE001
            q.put(("err", exc))

    threading.Thread(target=_run, daemon=True).start()
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return ("timeout", None)


def _next_with_timeout(gen, timeout=3.0):
    kind, value = _try_next(gen, timeout)
    if kind == "timeout":
        pytest.fail(f"tail_jsonl produced nothing within {timeout}s")
    if kind == "err":
        raise value
    return value


def test_tail_jsonl_waits_for_file_creation(tmp_path):
    path = tmp_path / "log.jsonl"
    gen = tail_jsonl(path, poll_interval=0.05)

    def _produce():
        import time
        time.sleep(0.2)
        path.write_text('{"a": 1}\n', encoding="utf-8")

    threading.Thread(target=_produce, daemon=True).start()
    row = _next_with_timeout(gen)
    assert row == {"a": 1}


def test_tail_jsonl_starts_from_end_not_existing_content(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"old": true}\n', encoding="utf-8")
    gen = tail_jsonl(path, poll_interval=0.05)

    def _append():
        import time
        time.sleep(0.2)
        with path.open("a", encoding="utf-8") as f:
            f.write('{"new": true}\n')

    threading.Thread(target=_append, daemon=True).start()
    row = _next_with_timeout(gen)
    assert row == {"new": True}  # the pre-existing {"old": true} was NOT replayed


def test_tail_jsonl_withholds_partial_line(tmp_path):
    """A line split across two writes must reassemble correctly, not be
    parsed (and mis-yield or crash) from its incomplete first half.

    Both writes happen from a SINGLE background thread, and the main thread
    makes exactly one blocking next() call spanning the whole sequence —
    tail_jsonl's generator is not designed for concurrent multi-thread
    iteration (calling next() from two threads on the same live generator
    raises "generator already executing"), so this drives it the same way
    real usage does: one consumer, one at a time.
    """
    path = tmp_path / "log.jsonl"
    path.write_text("", encoding="utf-8")
    gen = tail_jsonl(path, poll_interval=0.05)

    def _write_in_two_parts():
        import time
        with path.open("a", encoding="utf-8") as f:
            f.write('{"partial": tr')  # no trailing newline, invalid JSON too
        time.sleep(0.3)
        with path.open("a", encoding="utf-8") as f:
            f.write("ue}\n")

    threading.Thread(target=_write_in_two_parts, daemon=True).start()
    row = _next_with_timeout(gen, timeout=3.0)
    assert row == {"partial": True}
