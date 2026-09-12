"""Tests for the viewer's pure filesystem-reading functions — no HTTP."""
from __future__ import annotations

import json
import queue
import time
import threading
from pathlib import Path

import pytest

from a3dasm._src.viewer.readers import (
    read_artifacts,
    read_oracle,
    read_vitals,
    graph_spec_json,
    list_node_transcripts,
    load_graph_for_study,
    read_delegations,
    read_diagnostics_tail,
    read_hypotheses,
    read_milestones,
    read_notebook,
    read_problem_statement,
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
# read_problem_statement
# ---------------------------------------------------------------------------

def test_read_problem_statement_none_when_missing(tmp_path):
    assert read_problem_statement(tmp_path / "run") is None


def test_read_problem_statement_reads_snapshot_verbatim(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "debug").mkdir(parents=True)
    (run_dir / "debug" / "PROBLEM_STATEMENT_snapshot.md").write_text(
        "# Trivial task\n\nDo the thing.\n", encoding="utf-8")
    assert read_problem_statement(run_dir) == "# Trivial task\n\nDo the thing.\n"


# ---------------------------------------------------------------------------
# list_node_transcripts — real, disk-verified keys, never guessed
# ---------------------------------------------------------------------------

def test_list_node_transcripts_worker_returns_delegation_ids_to_it(tmp_path):
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "debug" / "delegation_log.jsonl", [
        {"id": "D001", "status": "DONE", "from_node": "strategizer",
         "to_node": "implementer"},
        {"id": "D002", "status": "DONE", "from_node": "strategizer",
         "to_node": "critic"},
    ])
    for did in ("D001", "D002"):
        _write_jsonl(run_dir / "debug" / "transcripts" / f"{did}.jsonl",
                     [{"type": "assistant", "text": "x"}])

    assert list_node_transcripts(run_dir, "implementer") == ["D001"]
    assert list_node_transcripts(run_dir, "critic") == ["D002"]
    assert list_node_transcripts(run_dir, "nonexistent_node") == []


def test_list_node_transcripts_omits_ids_with_no_file_on_disk(tmp_path):
    """A key that resolves to nothing is worse than no key.

    The critic's gate delegation has an id (GATE...) but writes its
    transcript to the NESTED critic/call_NNN.jsonl instead, so returning
    the id offered the UI a transcript that did not exist — observed on run
    20260905T162758, where the critic tab was simply blank.
    """
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "debug" / "delegation_log.jsonl", [
        {"id": "GATE1200", "status": "GATE:PASS", "from_node": "strategizer",
         "to_node": "critic"},
    ])
    _write_jsonl(run_dir / "debug" / "transcripts" / "critic" / "call_001.jsonl",
                 [{"type": "assistant", "text": "verdict"}])

    # The nested file is offered; the id with no flat file is not.
    assert list_node_transcripts(run_dir, "critic") == ["critic/call_001"]


def test_list_node_transcripts_finds_entry_turns_under_its_own_name(tmp_path):
    """The nested directory is keyed on the node's real name.

    It used to be the literal "strategizer", so a graph whose entry node was
    called anything else silently had no transcripts at all.
    """
    run_dir = tmp_path / "run"
    _write_jsonl(run_dir / "debug" / "transcripts" / "planner" / "turn_001.jsonl",
                 [{"type": "assistant", "text": "plan"}])
    assert list_node_transcripts(run_dir, "planner", is_entry=True) == [
        "planner/turn_001"]


def test_list_node_transcripts_entry_lists_real_turn_files_only(tmp_path):
    """The entry node's keys are never a guessed count — only turn files
    that genuinely exist on disk, sorted."""
    run_dir = tmp_path / "run"
    st_dir = run_dir / "debug" / "transcripts" / "strategizer"
    st_dir.mkdir(parents=True)
    (st_dir / "turn_001.jsonl").write_text("")
    (st_dir / "turn_002.jsonl").write_text("")
    # A worker delegation exists too -- must NOT leak into the entry
    # node's own key list (it's a different node's transcript).
    _write_jsonl(run_dir / "debug" / "delegation_log.jsonl", [
        {"id": "D001", "status": "DONE", "from_node": "strategizer",
         "to_node": "implementer"},
    ])

    keys = list_node_transcripts(run_dir, "strategizer", is_entry=True)
    assert keys == ["strategizer/turn_001", "strategizer/turn_002"]


def test_list_node_transcripts_entry_empty_when_no_turn_files(tmp_path):
    assert list_node_transcripts(tmp_path / "run", "strategizer", is_entry=True) == []


# ---------------------------------------------------------------------------
# graph_spec_json — reuses run_diagram's real layout/tool logic
# ---------------------------------------------------------------------------

def test_graph_spec_json_reuses_bfs_layers_and_node_tools():
    from a3dasm._src.backends.base import Agent, Edge, Graph
    from a3dasm._src.runtime.run_diagram import _bfs_layers, _node_tools

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


def _three_node_graph():
    from a3dasm._src.backends.base import Agent, Edge, Graph

    class _Strategizer(Agent):
        role = "strategizer"
        description = "hub"
        tools = frozenset()

    class _Implementer(Agent):
        role = "implementer"
        description = "writes and runs code"
        tools = frozenset()

    class _Critic(Agent):
        role = "critic"
        description = "gate"
        tools = frozenset()

    return Graph(
        nodes={
            "strategizer": _Strategizer(),
            "implementer": _Implementer(),
            "critic": _Critic(),
        },
        edges=(
            Edge("strategizer", "implementer"),
            Edge("strategizer", "critic"),
        ),
        entry="strategizer",
    )


def test_graph_spec_json_emits_authoritative_node_coordinates():
    """Every node carries an (x, y) and the canvas that contains it.

    These exist so the client never measures the DOM to place a node or aim
    an edge — the regression this guards is the old ``drawEdges()``, which
    read ``getBoundingClientRect()`` after layout and so drew edges that
    disagreed with their boxes depending on load timing.
    """
    spec = graph_spec_json(_three_node_graph())

    by_name = {n["name"]: n for n in spec["nodes"]}
    for node in by_name.values():
        assert isinstance(node["x"], int)
        assert isinstance(node["y"], int)
        # Inside the advertised canvas, box included.
        assert 0 <= node["x"] <= spec["canvas_w"] - spec["node_w"]
        assert 0 <= node["y"] <= spec["canvas_h"] - spec["node_h"]

    # The entry node is a layer above its two workers, which share a row.
    assert by_name["strategizer"]["y"] < by_name["implementer"]["y"]
    assert by_name["implementer"]["y"] == by_name["critic"]["y"]
    # Same-row nodes do not overlap.
    xs = sorted((by_name["critic"]["x"], by_name["implementer"]["x"]))
    assert xs[1] - xs[0] >= spec["node_w"]


def test_graph_spec_json_layout_is_stable_under_node_reordering():
    """Registration order must not move the diagram.

    ``graph.nodes`` is insertion-ordered, so laying out in iteration order
    would silently reshuffle a study's diagram the day someone reorders its
    node registrations. Positions are keyed on (layer, name) instead.
    """
    from a3dasm._src.backends.base import Graph

    graph = _three_node_graph()
    shuffled = Graph(
        nodes={k: graph.nodes[k] for k in
               ("critic", "strategizer", "implementer")},
        edges=graph.edges,
        entry=graph.entry,
    )

    def positions(g):
        return {n["name"]: (n["x"], n["y"], n["identity"])
                for n in graph_spec_json(g)["nodes"]}

    assert positions(graph) == positions(shuffled)


def test_graph_spec_json_identity_index_excludes_the_entry_node():
    """The entry node is not one hue among peers.

    It is the only node that can reach the human operator (``FollowUp``
    routes to the operator only when the asker is the entry node), so it is
    deliberately outside the identity ramp and marked -1.
    """
    spec = graph_spec_json(_three_node_graph())
    by_name = {n["name"]: n for n in spec["nodes"]}

    assert by_name["strategizer"]["identity"] == -1
    others = sorted(by_name[n]["identity"] for n in ("critic", "implementer"))
    assert others == [0, 1]


def test_graph_spec_json_exposes_model_system_prompt_and_tool_docs(tmp_path):
    from a3dasm._src.backends.base import Agent, Graph

    class _Strategizer(Agent):
        role = "strategizer"
        description = "hub"
        system_prompt = "You are the hub."
        tools = frozenset({"Bash"})

    graph = Graph(
        nodes={"strategizer": _Strategizer()}, edges=(), entry="strategizer")

    # No config.yaml, no explicit model on the instance -> honest fallback.
    spec = graph_spec_json(graph, study_dir=tmp_path)
    node = spec["nodes"][0]
    assert node["model"] == "(backend default)"
    assert node["system_prompt"] == "You are the hub."
    assert "Executes a shell command" in spec["tool_docs"]["Bash"]

    # A study's config.yaml sets the run-wide model; the instance itself
    # never got one explicitly (the common case — see graph_spec_json's
    # docstring).
    (tmp_path / "config.yaml").write_text(
        "model: claude-haiku-4-5-20251001\n", encoding="utf-8")
    spec2 = graph_spec_json(graph, study_dir=tmp_path)
    assert spec2["nodes"][0]["model"] == "Claude Haiku 4.5"


def test_graph_spec_json_instance_model_overrides_study_config(tmp_path):
    from a3dasm._src.backends.base import Agent, Graph

    class _Strategizer(Agent):
        role = "strategizer"
        description = "hub"

    graph = Graph(
        nodes={"strategizer": _Strategizer(model="claude-opus-5")},
        edges=(), entry="strategizer")
    (tmp_path / "config.yaml").write_text(
        "model: claude-haiku-4-5-20251001\n", encoding="utf-8")

    spec = graph_spec_json(graph, study_dir=tmp_path)
    assert spec["nodes"][0]["model"] == "Claude Opus 5"


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


def _next_with_timeout(gen, timeout=10.0):
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
    row = _next_with_timeout(gen, timeout=10.0)
    assert row == {"partial": True}


def test_tail_jsonl_backs_up_to_last_complete_line_when_already_mid_line(tmp_path):
    """Deterministic version of the race the test above exercises via real
    thread timing (confirmed by direct reproduction to fail ~25% of the
    time before this fix, even with a 10s timeout margin — not just tight
    timing): tail_jsonl's generator body only starts running on its FIRST
    next() call, not when tail_jsonl() itself is invoked, so a caller that
    starts iterating even slightly late can observe the file already ending
    in an unterminated line at the moment the initial offset is captured —
    if a writer's append of the first half of a split line happened to land
    in that gap. Naively setting offset = current file size then silently
    swallows that in-flight line's prefix forever: once the remainder
    arrives, reading from `offset` onward yields only the fragment, never
    valid JSON on its own. This constructs that exact starting condition
    directly (no race needed): the file already ends mid-line before
    tail_jsonl is even called.
    """
    path = tmp_path / "log.jsonl"
    # A genuinely complete prior line (correctly skipped — it ended before
    # tailing started) followed by an unterminated one (must NOT be skipped).
    path.write_text('{"a": 1}\n{"partial": tr', encoding="utf-8")
    gen = tail_jsonl(path, poll_interval=0.05)

    def _complete_the_line():
        import time
        time.sleep(0.2)
        with path.open("a", encoding="utf-8") as f:
            f.write("ue}\n")

    threading.Thread(target=_complete_the_line, daemon=True).start()
    row = _next_with_timeout(gen, timeout=10.0)
    assert row == {"partial": True}  # not {"a": 1} — that one was already complete


# ---------------------------------------------------------------------------
# hypotheses / milestones — the epistemic ledger
# ---------------------------------------------------------------------------

def _notes(run_dir: Path) -> Path:
    d = run_dir / "debug" / "strategizer_notes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_read_hypotheses_resolves_status_from_the_last_log_entry(tmp_path):
    """hypotheses.json has no status field — the live status is the last
    status_log entry's, and resolving that is ledger semantics that belongs
    in one place rather than being re-derived by every caller."""
    run = tmp_path / "runs" / "r1"
    (_notes(run) / "hypotheses.json").write_text(json.dumps({
        "H1": {
            "id": "H1",
            "statement": "s",
            "prediction": "p",
            "prior": 0.4,
            "proposed_at": "2026-09-04T19:00:00+00:00",
            "status_log": [
                {"status": "OPEN", "ts": "2026-09-04T19:00:00+00:00",
                 "posterior": 0.4},
                {"status": "REFUTED", "ts": "2026-09-04T19:30:00+00:00",
                 "posterior": 0.02, "triggered_by": "D004",
                 "evidence": {"delegation": "D004"}},
            ],
        }
    }), encoding="utf-8")

    (h,) = read_hypotheses(run)
    assert h["status"] == "REFUTED"
    assert h["posterior"] == 0.02
    assert h["triggered_by"] == "D004"
    # Both ends of the log are reported, not just the latest.
    assert h["opened_at"] == "2026-09-04T19:00:00+00:00"
    assert h["updated_at"] == "2026-09-04T19:30:00+00:00"
    assert h["history"] == 2


def test_read_hypotheses_handles_an_empty_status_log(tmp_path):
    """A hypothesis registered but never updated still has a status."""
    run = tmp_path / "runs" / "r1"
    (_notes(run) / "hypotheses.json").write_text(
        json.dumps({"H1": {"statement": "s", "status_log": []}}),
        encoding="utf-8")

    (h,) = read_hypotheses(run)
    assert h["id"] == "H1"          # falls back to the dict key
    assert h["status"] == "OPEN"


def test_read_milestones_keeps_the_note_that_justifies_the_status(tmp_path):
    """A SKIPPED milestone with a defended reason is a judgment the run
    made; without the note it is indistinguishable from one never reached."""
    run = tmp_path / "runs" / "r1"
    (_notes(run) / "milestones.json").write_text(json.dumps({
        "M002": {"id": "M002", "key": "oracle_gold_state",
                 "status": "SKIPPED", "note": "no oracle in this run"},
        "M001": {"id": "M001", "key": "assess_literature_need",
                 "status": "DONE", "note": ""},
    }), encoding="utf-8")

    ms = read_milestones(run)
    assert [m["id"] for m in ms] == ["M001", "M002"]   # id order, not file order
    assert ms[1]["note"] == "no oracle in this run"


def test_ledger_readers_degrade_when_the_run_has_no_notes_dir(tmp_path):
    """The debug flag may have been off, or the run may not have got that
    far — both must read as "nothing yet", never as an error."""
    run = tmp_path / "runs" / "r1"
    (run / "debug").mkdir(parents=True)
    assert read_hypotheses(run) == []
    assert read_milestones(run) == []


def test_ledger_readers_survive_a_half_written_file(tmp_path):
    """Both ledgers are rewritten whole during a live run, so a poll can
    genuinely catch a partial write."""
    run = tmp_path / "runs" / "r1"
    (_notes(run) / "hypotheses.json").write_text('{"H1": {"stat',
                                                 encoding="utf-8")
    assert read_hypotheses(run) == []


# ---------------------------------------------------------------------------
# read_notebook — pipeline.ipynb is STUDY-scoped, so attribution matters
# ---------------------------------------------------------------------------

def _write_nb(path: Path, cells: list[dict], run: str | None = None) -> None:
    nb = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    if run is not None:
        nb["metadata"]["agentic"] = {"run": f"/some/study/runs/{run}"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb), encoding="utf-8")


def _code(src: str, outputs=None):
    return {"cell_type": "code", "source": src, "execution_count": 1,
            "metadata": {}, "outputs": outputs or []}


def test_read_notebook_uses_the_live_file_when_it_is_stamped_for_this_run(tmp_path):
    _write_nb(tmp_path / "pipeline.ipynb", [_code("x = 1")], run="R1")
    nb = read_notebook(tmp_path, "R1")
    assert nb["live"] is True
    assert nb["cells"][0]["source"] == "x = 1"


def test_read_notebook_refuses_a_live_file_stamped_for_another_run(tmp_path):
    """The regression that matters: pipeline.ipynb lives in the STUDY dir
    and belongs to whichever run last wrote it. Showing it on an older
    run's page would attribute one run's deliverable to another — silently,
    and plausibly enough that nobody would notice."""
    _write_nb(tmp_path / "pipeline.ipynb", [_code("newer = 1")], run="R2")
    assert read_notebook(tmp_path, "R1") is None


def test_read_notebook_falls_back_to_this_runs_archive(tmp_path):
    """A finished run's notebook is renamed pipeline_<run_id>.ipynb when the
    next run starts, so an older run's page must read the archive."""
    _write_nb(tmp_path / "pipeline.ipynb", [_code("newer = 1")], run="R2")
    _write_nb(tmp_path / "pipeline_R1.ipynb", [_code("older = 1")], run="R1")
    nb = read_notebook(tmp_path, "R1")
    assert nb["live"] is False
    assert nb["cells"][0]["source"] == "older = 1"


def test_read_notebook_none_when_the_agent_never_wrote_one(tmp_path):
    """Not an error state — "no deliverable" is itself a real finding."""
    assert read_notebook(tmp_path, "R1") is None


def test_read_notebook_normalizes_outputs(tmp_path):
    _write_nb(tmp_path / "pipeline.ipynb", [
        {"cell_type": "markdown", "source": "# Title", "metadata": {}},
        _code("print(1)", [
            {"output_type": "stream", "name": "stdout", "text": ["1\n"]},
            {"output_type": "execute_result", "data": {"text/plain": "42"},
             "metadata": {}, "execution_count": 1},
            {"output_type": "display_data",
             "data": {"image/png": "iVBORw0KGgo="}, "metadata": {}},
            {"output_type": "error", "ename": "ValueError",
             "evalue": "bad", "traceback": ["line1", "line2"]},
        ]),
    ], run="R1")

    cells = read_notebook(tmp_path, "R1")["cells"]
    assert cells[0]["type"] == "markdown"
    kinds = [o["kind"] for o in cells[1]["outputs"]]
    assert kinds == ["stream", "text", "image", "error"]
    assert cells[1]["outputs"][0]["text"] == "1\n"
    assert cells[1]["outputs"][2]["mime"] == "image/png"
    assert cells[1]["outputs"][3]["text"] == "line1\nline2"


def test_read_notebook_reports_a_corrupt_file_instead_of_hiding_it(tmp_path):
    """A notebook that exists but cannot be parsed is a different answer
    from no notebook at all, and the reader must not collapse the two."""
    (tmp_path / "pipeline_R1.ipynb").write_text("{not json", encoding="utf-8")
    nb = read_notebook(tmp_path, "R1")
    assert nb is not None
    assert nb["cells"] == []
    assert "could not read" in nb["error"]


# ---------------------------------------------------------------------------
# tool documentation — read from source, not hand-copied
# ---------------------------------------------------------------------------

def test_every_tool_an_agent_is_handed_has_a_description():
    """The agent profile exists to say what an agent can do, so a tool with
    no description is a hole in the only page that answers that.

    Most tools are nested closures built per-run, so their ``__doc__`` is
    unreachable by import; the docs are parsed out of the node modules'
    source instead. This guards the drift case: a tool added without a
    docstring, or moved to a module the parser does not scan, silently
    reappears here as "no description available".
    """
    from a3dasm._src.agents import _default_graph

    graph = _default_graph()
    spec = graph_spec_json(graph)
    docs = spec["tool_docs"]
    missing = sorted({
        t for node in spec["nodes"] for t in node["tools"] if t not in docs
    })
    assert not missing, f"tools with no description: {missing}"


def test_routing_tool_docs_ignores_non_tool_helpers():
    """Tools are PascalCase; a lowercase local helper sharing a tool's name
    must not be mistaken for one."""
    from a3dasm._src.viewer.readers import _routing_tool_docs

    docs = _routing_tool_docs()
    assert docs, "expected some tool docstrings to be parsed out of source"
    assert all(name[:1].isupper() for name in docs)
    # First paragraph only, matching what the agent's own tool catalog shows.
    assert all("\n\n" not in d for d in docs.values())


def test_read_notebook_does_not_prefix_match_another_runs_archive(tmp_path):
    """"pipeline_{run_id}*" let run R1 match pipeline_R10.ipynb.

    Timestamped ids happen to be fixed width, but nothing enforces that and
    run_id arrives from the URL — and showing another run's deliverable as
    this one's is the exact failure this reader exists to prevent.
    """
    _write_nb(tmp_path / "pipeline_R10.ipynb", [_code("other = 1")], run="R10")
    assert read_notebook(tmp_path, "R1") is None

    # The uuid suffix the archiver adds for a repeated id still resolves.
    _write_nb(tmp_path / "pipeline_R1_ab12cd34.ipynb", [_code("mine = 1")],
              run="R1")
    nb = read_notebook(tmp_path, "R1")
    assert nb["cells"][0]["source"] == "mine = 1"


def test_read_artifacts_does_not_call_a_started_run_a_shared_workspace(tmp_path):
    """A run that has not created debug/ yet is still a run.

    Treating it as a shared workspace labels its files "not attributable to
    any run", which is the opposite of the truth.
    """
    study = tmp_path
    run = study / "runs" / "20260904T120000"
    (run / "debug" / "delegations" / "D001").mkdir(parents=True)
    (run / "debug" / "delegations" / "D001" / "report.md").write_text("x")
    pending = study / "runs" / "20260905T090000"
    pending.mkdir(parents=True)
    (pending / "notes.md").write_text("not yours", encoding="utf-8")

    rows = read_artifacts(run, study)
    assert all("20260905T090000" not in r["path"] for r in rows)
    assert [r["scope"] for r in rows] == ["run"]


def test_tail_jsonl_stops_when_asked(tmp_path):
    """Without a stop signal the generator never returns, so the thread
    running it outlives the SSE connection that started it."""
    path = tmp_path / "log.jsonl"
    path.write_text('{"a": 1}\n', encoding="utf-8")
    stop = {"now": False}
    it = tail_jsonl(path, poll_interval=0.01, should_stop=lambda: stop["now"])
    stop["now"] = True
    assert list(it) == []


def test_tail_jsonl_waiting_for_a_file_still_honours_the_stop_signal(tmp_path):
    """The wait-for-creation loop needs the stop check too.

    A run that never writes diagnostics.jsonl leaves that tailer parked in
    this loop forever — it never even reaches the tail proper, so a stop
    check only in the main loop would not free the thread.
    """
    it = tail_jsonl(tmp_path / "never.jsonl", poll_interval=0.01,
                    should_stop=lambda: True)
    assert list(it) == []


# ---------------------------------------------------------------------------
# read_oracle — the input space, the namespaces, and every ledgered eval
# ---------------------------------------------------------------------------

def _store(root: Path, rows: list[tuple[str, str, str]]) -> None:
    """A minimal f3dasm store: domain + input/output/jobs ledgers."""
    data = root / "experiment_data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "domain.json").write_text(json.dumps(
        {"input_space": {"x": {}, "y": {}}}), encoding="utf-8")
    (data / "input.csv").write_text(
        ",x,y\n" + "".join(f"{i},{r[0]},1.0\n" for i, r in enumerate(rows)),
        encoding="utf-8")
    (data / "output.csv").write_text(
        ",score,_delegation_id,_ts,_wall_ms\n"
        + "".join(f"{i},{r[1]},{r[2]},2026-09-06T12:00:0{i}+00:00,1500\n"
                  for i, r in enumerate(rows)),
        encoding="utf-8")
    (data / "jobs.csv").write_text(
        ",0\n" + "".join(f"{i},FINISHED\n" for i in range(len(rows))),
        encoding="utf-8")


def test_read_oracle_reports_the_input_space_and_every_eval(tmp_path):
    run = tmp_path / "runs" / "R1"
    (run / "debug").mkdir(parents=True)
    store = run / "experiment_data"
    _store(store, [("0.1", "9.5", "D001"), ("0.2", "8.5", "D002")])
    (run / "debug" / "run_config.json").write_text(json.dumps({
        "store_dir": str(store), "evaluator_name": "demo",
        "evaluator_entrypoint": "w/gen.py:Gen", "eval_budget": 50,
    }), encoding="utf-8")

    o = read_oracle(run)
    assert o["evaluator_name"] == "demo"
    assert o["eval_budget"] == 50
    assert o["total_evals"] == 2
    (st,) = o["stores"]
    assert st["namespace"] is None
    assert st["input_space"] == ["x", "y"]
    # Newest first: on a live run the interesting rows are the latest.
    assert [e["index"] for e in st["evals"]] == [1, 0]
    assert st["evals"][0]["delegation_id"] == "D002"
    assert st["evals"][0]["outputs"] == {"score": "8.5"}


def test_read_oracle_keeps_provenance_out_of_the_results(tmp_path):
    """_delegation_id/_ts/_wall_ms answer "who ran this and when", not
    "what did the oracle return" — mixing them makes a result table
    unreadable and invites treating a timestamp as a measurement."""
    run = tmp_path / "runs" / "R1"
    (run / "debug").mkdir(parents=True)
    store = run / "experiment_data"
    _store(store, [("0.1", "9.5", "D001")])
    (run / "debug" / "run_config.json").write_text(
        json.dumps({"store_dir": str(store)}), encoding="utf-8")

    (ev,) = read_oracle(run)["stores"][0]["evals"]
    assert ev["outputs"] == {"score": "9.5"}
    assert not any(k.startswith("_") for k in ev["outputs"])
    assert ev["delegation_id"] == "D001" and ev["wall_ms"] == "1500"


def test_read_oracle_finds_namespaces_on_disk_not_only_in_the_config(tmp_path):
    """A namespace store exists the moment it is registered; run_config.json
    is rewritten separately. Trusting the config alone undercounts evals —
    the same defect the runtime's own accounting had to fix, where a run
    reported 100 evals against 200 real ones.
    """
    run = tmp_path / "runs" / "R1"
    (run / "debug").mkdir(parents=True)
    store = run / "experiment_data"
    _store(store, [("0.1", "1.0", "D001")])
    _store(store / "design_b", [("0.5", "2.0", "D002"), ("0.6", "3.0", "D002")])
    # Config mentions the canonical store only.
    (run / "debug" / "run_config.json").write_text(
        json.dumps({"store_dir": str(store)}), encoding="utf-8")

    o = read_oracle(run)
    assert [s["namespace"] for s in o["stores"]] == [None, "design_b"]
    assert o["total_evals"] == 3


def test_read_oracle_on_a_run_with_no_oracle(tmp_path):
    run = tmp_path / "runs" / "R1"
    (run / "debug").mkdir(parents=True)
    o = read_oracle(run)
    assert o["stores"] == [] and o["total_evals"] == 0


def test_read_vitals_start_survives_a_rewritten_run_config(tmp_path):
    """The wall clock anchors on the EARLIEST start-file mtime.

    run_config.json is rewritten mid-run (oracle registration), so
    anchoring on it alone made a run that had been going 1h58m report 67
    seconds — the clock visibly reset to zero on a live run.
    """
    import os
    run = tmp_path / "runs" / "R1"
    debug = run / "debug"
    debug.mkdir(parents=True)
    old = time.time() - 7080
    for name in ("thread_id", "PROBLEM_STATEMENT_snapshot.md"):
        (debug / name).write_text("x", encoding="utf-8")
        os.utime(debug / name, (old, old))
    (debug / "run_config.json").write_text("{}", encoding="utf-8")  # mtime now

    assert read_vitals(run)["elapsed_s"] > 7000


def test_vitals_prefer_the_runs_own_recorded_start(tmp_path):
    """The run records its start explicitly; the viewer must use that rather
    than infer one, so the operator's clock and the run's own wall budget
    cannot disagree. Before the anchor existed this was a min() over three
    file mtimes, and a mid-run rewrite of run_config.json made a run that had
    been going 1h58m report 67 seconds."""
    import time as _t

    from a3dasm._src.viewer.readers import read_vitals

    run = tmp_path / "20260101T000000"
    debug = run / "debug"
    debug.mkdir(parents=True)
    (debug / "thread_id").write_text("t")          # a much NEWER mtime
    anchor = _t.time() - 7200.0                     # run began two hours ago
    (debug / "run_started_at").write_text(repr(anchor))

    v = read_vitals(run)
    assert abs(v["started_at"] - anchor) < 1.0
    assert v["elapsed_s"] > 7000, (
        "the recorded anchor must win over the mtime heuristic"
    )


def test_vitals_fall_back_when_there_is_no_anchor(tmp_path):
    """Runs recorded before the anchor existed must still report a clock."""
    from a3dasm._src.viewer.readers import read_vitals

    run = tmp_path / "20260101T000000"
    debug = run / "debug"
    debug.mkdir(parents=True)
    (debug / "thread_id").write_text("t")

    v = read_vitals(run)
    assert v["started_at"] is not None
    assert v["elapsed_s"] is not None


def test_vitals_ignore_a_corrupt_anchor(tmp_path):
    from a3dasm._src.viewer.readers import read_vitals

    run = tmp_path / "20260101T000000"
    debug = run / "debug"
    debug.mkdir(parents=True)
    (debug / "thread_id").write_text("t")
    (debug / "run_started_at").write_text("not-a-float")

    v = read_vitals(run)          # must not raise
    assert v["started_at"] is not None


# ---------------------------------------------------------------------------
# A node's model is a property of the RUN, not of the reconstructed graph
# ---------------------------------------------------------------------------

def test_graph_spec_json_prefers_the_runs_own_model_record(tmp_path):
    """Regression: the viewer showed math_expert as "Claude Haiku 4.5" for a
    run that put it on a locally-served Qwen.

    The viewer reconstructs the graph by re-executing the study's build_graph()
    in ITS OWN process. A study whose composition depends on runtime state — an
    env var naming a local endpoint, say — rebuilds differently there, so the
    re-derived model is the study default rather than what the run used. The
    run writes down what it actually used; read that.
    """
    import json as _json

    from a3dasm._src.backends.base import Agent, Graph

    class _Hub(Agent):
        role = "strategizer"
        description = "hub"

    class _Math(Agent):
        role = "math_expert"
        description = "derivations"

    graph = Graph(
        nodes={"strategizer": _Hub(), "math_expert": _Math()},
        edges=(), entry="strategizer",
    )
    (tmp_path / "config.yaml").write_text(
        "model: claude-haiku-4-5-20251001\n", encoding="utf-8")

    run_dir = tmp_path / "runs" / "20260912T142229"
    (run_dir / "debug").mkdir(parents=True)
    (run_dir / "debug" / "node_models.json").write_text(_json.dumps({
        "strategizer": {"model": "claude-haiku-4-5-20251001", "backend": "claude"},
        "math_expert": {"model": "qwen3.8-27b-256k", "backend": "ollama"},
    }), encoding="utf-8")

    by_name = {n["name"]: n for n in
               graph_spec_json(graph, tmp_path, run_dir)["nodes"]}
    assert by_name["math_expert"]["model"] == "Qwen/Qwen3.8-27B (256k ctx)"
    assert by_name["math_expert"]["backend"] == "ollama"
    assert by_name["strategizer"]["model"] == "Claude Haiku 4.5"

    # Without the record (a run predating it) the old re-derivation still
    # applies — degraded, but never an error.
    stale = {n["name"]: n for n in graph_spec_json(graph, tmp_path)["nodes"]}
    assert stale["math_expert"]["model"] == "Claude Haiku 4.5"
    assert stale["math_expert"]["backend"] is None


def test_graph_spec_json_survives_a_corrupt_model_record(tmp_path):
    """A viewer must never 500 on a malformed artifact; it degrades."""
    from a3dasm._src.backends.base import Agent, Graph

    class _Hub(Agent):
        role = "strategizer"
        description = "hub"

    graph = Graph(nodes={"strategizer": _Hub()}, edges=(), entry="strategizer")
    run_dir = tmp_path / "runs" / "r1"
    (run_dir / "debug").mkdir(parents=True)
    (run_dir / "debug" / "node_models.json").write_text("{not json", encoding="utf-8")

    spec = graph_spec_json(graph, tmp_path, run_dir)
    assert spec["nodes"][0]["model"] == "(backend default)"
