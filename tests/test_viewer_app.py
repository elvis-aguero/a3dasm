"""Tests for the viewer's Starlette app — TestClient against fixture study
dirs, no real LLM/network involved.

One exception: the SSE `/stream` tests use a REAL live uvicorn server, not
TestClient. Verified directly (see the commit this file was added in) that
httpx's ASGI test transport — under both the sync `TestClient` wrapper and a
plain async `httpx.AsyncClient(transport=ASGITransport(...))` — buffers an
async generator's ENTIRE output until it fully completes before releasing
ANY of it to `iter_lines()`/`aiter_lines()`, even across an internal
`asyncio.sleep`. That makes it structurally incapable of testing a stream
that is deliberately infinite (real SSE never "completes"). A real uvicorn
server + a real socket-based `httpx.stream()` call delivers each chunk as
it's actually written, confirmed by the same minimal repro timing correctly
(0.03s / 0.33s) against the buffered version's (0.31s / 0.31s, i.e. never
incremental)."""
from __future__ import annotations

import json
import queue
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from starlette.testclient import TestClient

from a3dasm._src.viewer.app import create_app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _LiveServer:
    """A real uvicorn server on a real port, for tests that need genuine
    incremental streaming (see module docstring)."""

    def __init__(self, app):
        self.port = _free_port()
        config = uvicorn.Config(
            app, host="127.0.0.1", port=self.port, log_level="error")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(50):
            try:
                httpx.get(f"{self.url}/api/runs", timeout=0.2)
                return self
            except httpx.TransportError:
                time.sleep(0.1)
        raise RuntimeError("live server did not start in time")

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _make_study(tmp_path: Path) -> Path:
    study = tmp_path / "study"
    study.mkdir()
    (study / "run.py").write_text(
        "from a3dasm._src.backends.base import Agent, Edge, Graph\n"
        "class _S(Agent):\n"
        "    role = 'strategizer'\n"
        "    description = 'hub'\n"
        "    tools = frozenset()\n"
        "class _C(Agent):\n"
        "    role = 'critic'\n"
        "    description = 'gate'\n"
        "    tools = frozenset()\n"
        "def build_graph():\n"
        "    return Graph(nodes={'strategizer': _S(), 'critic': _C()}, "
        "edges=(Edge('strategizer', 'critic'),), entry='strategizer')\n",
        encoding="utf-8",
    )
    return study


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _make_run(study: Path, run_id: str) -> Path:
    run_dir = study / "runs" / run_id
    (run_dir / "debug").mkdir(parents=True)
    return run_dir


# ---------------------------------------------------------------------------
# /api/runs
# ---------------------------------------------------------------------------

def test_list_runs_empty(tmp_path):
    study = _make_study(tmp_path)
    client = TestClient(create_app(study))
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_runs_lists_fixture_run(tmp_path):
    study = _make_study(tmp_path)
    _make_run(study, "20260904T120000")
    client = TestClient(create_app(study))
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert [r["run_id"] for r in resp.json()] == ["20260904T120000"]


# ---------------------------------------------------------------------------
# /api/runs/{id}/graph
# ---------------------------------------------------------------------------

def test_get_graph_returns_fixture_topology(tmp_path):
    study = _make_study(tmp_path)
    _make_run(study, "20260904T120000")
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/20260904T120000/graph")
    assert resp.status_code == 200
    data = resp.json()
    names = {n["name"] for n in data["nodes"]}
    assert names == {"strategizer", "critic"}
    assert data["edges"] == [{"source": "strategizer", "target": "critic"}]
    assert data["entry"] == "strategizer"


def test_get_graph_404_for_missing_run(tmp_path):
    study = _make_study(tmp_path)
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/nonexistent/graph")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/runs/{id}/delegations
# ---------------------------------------------------------------------------

def test_get_delegations_matches_reader(tmp_path):
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    _write_jsonl(run_dir / "debug" / "delegation_log.jsonl", [
        {"id": "D001", "status": "RUNNING", "from_node": "strategizer",
         "to_node": "critic"},
        {"id": "D001", "status": "DONE", "from_node": "strategizer",
         "to_node": "critic"},
    ])
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/20260904T120000/delegations")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["status"] == "DONE"


def test_get_delegations_404_for_missing_run(tmp_path):
    study = _make_study(tmp_path)
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/nonexistent/delegations")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/runs/{id}/problem_statement
# ---------------------------------------------------------------------------

def test_get_problem_statement_returns_snapshot_text(tmp_path):
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    (run_dir / "debug" / "PROBLEM_STATEMENT_snapshot.md").write_text(
        "# Do the thing\n", encoding="utf-8")
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/20260904T120000/problem_statement")
    assert resp.status_code == 200
    assert resp.json() == {"text": "# Do the thing\n"}


def test_get_problem_statement_404_when_snapshot_missing(tmp_path):
    study = _make_study(tmp_path)
    _make_run(study, "20260904T120000")
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/20260904T120000/problem_statement")
    assert resp.status_code == 404


def test_get_problem_statement_404_for_missing_run(tmp_path):
    study = _make_study(tmp_path)
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/nonexistent/problem_statement")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/runs/{id}/node/{name}/transcripts — real, disk-verified keys
# ---------------------------------------------------------------------------

def test_get_node_transcripts_worker(tmp_path):
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    _write_jsonl(run_dir / "debug" / "delegation_log.jsonl", [
        {"id": "D001", "status": "DONE", "from_node": "strategizer",
         "to_node": "critic"},
    ])
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/20260904T120000/node/critic/transcripts")
    assert resp.status_code == 200
    assert resp.json() == ["D001"]


def test_get_node_transcripts_entry_lists_real_turn_files(tmp_path):
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    st_dir = run_dir / "debug" / "transcripts" / "strategizer"
    st_dir.mkdir(parents=True)
    (st_dir / "turn_001.jsonl").write_text("")
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/20260904T120000/node/strategizer/transcripts")
    assert resp.status_code == 200
    assert resp.json() == ["strategizer/turn_001"]


def test_get_node_transcripts_404_for_missing_run(tmp_path):
    study = _make_study(tmp_path)
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/nonexistent/node/critic/transcripts")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/runs/{id}/transcript/{key} — both 404 flavors
# ---------------------------------------------------------------------------

def test_get_transcript_200_with_events(tmp_path):
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    _write_jsonl(run_dir / "debug" / "transcripts" / "D007.jsonl", [
        {"ts": "t1", "type": "assistant", "text": "hi"},
    ])
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/20260904T120000/transcript/D007")
    assert resp.status_code == 200
    assert resp.json() == [{"ts": "t1", "type": "assistant", "text": "hi"}]


def test_get_transcript_404_debug_flag_off(tmp_path):
    study = _make_study(tmp_path)
    _make_run(study, "20260904T120000")  # no transcripts/ dir at all
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/20260904T120000/transcript/D007")
    assert resp.status_code == 404
    assert "debug flag" in resp.json()["error"]


def test_get_transcript_404_unknown_key(tmp_path):
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    (run_dir / "debug" / "transcripts").mkdir(parents=True)
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/20260904T120000/transcript/D999")
    assert resp.status_code == 404
    assert "no such transcript" in resp.json()["error"]


def test_get_transcript_404_for_missing_run(tmp_path):
    study = _make_study(tmp_path)
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/nonexistent/transcript/D007")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/runs/{id}/transcript/{key}/fragment — incremental append contract
# ---------------------------------------------------------------------------

def test_transcript_fragment_returns_only_events_after_index(tmp_path):
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    _write_jsonl(run_dir / "debug" / "transcripts" / "D007.jsonl", [
        {"ts": "t1", "type": "assistant", "text": "first"},
        {"ts": "t2", "type": "assistant", "text": "second"},
        {"ts": "t3", "type": "assistant", "text": "third"},
    ])
    client = TestClient(create_app(study))
    resp = client.get(
        "/api/runs/20260904T120000/transcript/D007/fragment?after=1")
    assert resp.status_code == 200
    assert "first" not in resp.text
    assert "second" in resp.text
    assert "third" in resp.text


def test_transcript_fragment_event_count_header_counts_raw_events_not_bubbles(
    tmp_path,
):
    """The client's next `after` cursor must count RAW events consumed,
    not rendered bubbles — most real events (stream_evt/partial/result)
    render to no bubble at all. A cursor counting bubbles falls behind the
    raw list and re-matches already-shown events on every subsequent poll,
    duplicating content forever even once the underlying file has stopped
    growing (confirmed for real: a genuine transcript had 527 raw events,
    only 47 rendered as bubbles — polling with a bubble-counted cursor grew
    the panel from 22KB to 74KB over three 2s polls on an already-finished
    run). X-Event-Count must reflect ALL events consumed, bubble or not."""
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    _write_jsonl(run_dir / "debug" / "transcripts" / "D007.jsonl", [
        {"ts": "t1", "type": "assistant", "text": "hello"},  # 1 bubble
        {"ts": "t2", "type": "stream_evt", "evt": "ping"},   # 0 bubbles
        {"ts": "t3", "type": "partial", "text": "..."},      # 0 bubbles
        {"ts": "t4", "type": "result", "usage": {}},         # 0 bubbles
    ])
    client = TestClient(create_app(study))
    resp = client.get(
        "/api/runs/20260904T120000/transcript/D007/fragment?after=0")
    assert resp.status_code == 200
    assert resp.headers["X-Event-Count"] == "4"  # not "1" (the bubble count)

    # A second poll using that cursor must return NOTHING new — the file
    # hasn't grown, so a correct cursor is already past every event.
    resp2 = client.get(
        "/api/runs/20260904T120000/transcript/D007/fragment?after=4")
    assert resp2.text == ""


def test_transcript_fragment_resolves_tool_result_name_by_position(tmp_path):
    """A tool_result event only carries `tool_use_id`, never the tool's
    name -- the fragment renderer must resolve it positionally against the
    full in-order tool-call queue (see `_render_fragment`'s docstring), not
    render a generic unlabeled "tool result" the reader can't attribute to
    anything (the exact defect the user's live screenshot showed)."""
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    _write_jsonl(run_dir / "debug" / "transcripts" / "D007.jsonl", [
        {"ts": "t1", "type": "assistant", "text": "",
         "tools": [{"name": "mcp__f3dasm_agent_tools__Delegate", "input": {}}]},
        {"ts": "t2", "type": "tool_result",
         "results": [{"tool_use_id": "x", "content": "ok"}]},
    ])
    client = TestClient(create_app(study))
    resp = client.get(
        "/api/runs/20260904T120000/transcript/D007/fragment?after=0")
    assert resp.status_code == 200
    # Displayed name strips the mcp__<server>__ prefix for readability...
    assert "Delegate result" in resp.text
    # ...but the raw registered name is still preserved for precision.
    assert "mcp__f3dasm_agent_tools__Delegate" in resp.text


def test_transcript_fragment_resolution_correct_when_after_splits_events(
    tmp_path,
):
    """The name-resolution queue is recomputed over the FULL event list
    every call, specifically so a tool_result landing in a LATER poll (its
    matching tool-call assistant event already past the `after` cursor)
    still resolves to the right name."""
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    _write_jsonl(run_dir / "debug" / "transcripts" / "D007.jsonl", [
        {"ts": "t1", "type": "assistant", "text": "",
         "tools": [{"name": "Read", "input": {}}]},
        {"ts": "t2", "type": "tool_result",
         "results": [{"tool_use_id": "x", "content": "file contents"}]},
    ])
    client = TestClient(create_app(study))
    # The client already consumed event 0 (the assistant/tool-call event)
    # on a prior poll; this call only asks for event 1 onward.
    resp = client.get(
        "/api/runs/20260904T120000/transcript/D007/fragment?after=1")
    assert resp.status_code == 200
    assert "Read result" in resp.text


def test_transcript_fragment_404_when_debug_off(tmp_path):
    """404 (not 200): the frontend's error path relies on !ok to stop
    polling and show the message — a 200 with an "empty-looking" body was
    indistinguishable from a legitimate empty poll, so the panel silently
    stayed blank with no explanation and kept polling forever."""
    study = _make_study(tmp_path)
    _make_run(study, "20260904T120000")
    client = TestClient(create_app(study))
    resp = client.get(
        "/api/runs/20260904T120000/transcript/D007/fragment?after=0")
    assert resp.status_code == 404
    assert "debug flag" in resp.text


def test_transcript_fragment_404_for_key_with_no_transcript_file(tmp_path):
    """A real delegation id with no captured conversation at all (e.g. a
    Done()-gate-check bookkeeping record like "GATE190739", confirmed for
    real to have a delegation_log.jsonl row but no transcripts/*.jsonl file)
    must 404 with an honest explanation, not silently return 200 with an
    empty body — the panel otherwise goes blank with zero explanation."""
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    (run_dir / "debug" / "transcripts").mkdir(parents=True)  # debug ON
    client = TestClient(create_app(study))
    resp = client.get(
        "/api/runs/20260904T120000/transcript/GATE190739/fragment?after=0")
    assert resp.status_code == 404
    assert "No transcript recorded" in resp.text


# ---------------------------------------------------------------------------
# /runs/{id} — the HTML page
# ---------------------------------------------------------------------------

def test_graph_page_renders(tmp_path):
    study = _make_study(tmp_path)
    _make_run(study, "20260904T120000")
    client = TestClient(create_app(study))
    resp = client.get("/runs/20260904T120000")
    assert resp.status_code == 200
    assert "20260904T120000" in resp.text


def test_graph_page_404_for_missing_run(tmp_path):
    study = _make_study(tmp_path)
    client = TestClient(create_app(study))
    resp = client.get("/runs/nonexistent")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/runs/{id}/stream — the append-while-streaming SSE test
# ---------------------------------------------------------------------------

def _read_sse_events(url, n, timeout=5.0):
    """Read *n* SSE "data:" lines off a REAL socket URL, bounded by
    *timeout* — run in a background thread so a stuck stream can't hang the
    test suite. Returns just the JSON payloads (event *type* discarded) —
    use `_read_typed_sse_events` when the test needs to disambiguate event
    types that share a JSON shape (e.g. `run_status` vs `delegation`, both
    of which carry a `status` key)."""
    return [data for _etype, data in _read_typed_sse_events(url, n, timeout)]


def _read_typed_sse_events(url, n, timeout=5.0):
    """Like `_read_sse_events`, but also captures each event's preceding
    "event: <type>" line, returning a list of (event_type, data) pairs.
    Needed because an SSE payload's JSON shape alone doesn't always say
    which event it is: a `run_status` event's `{"status": "GATED"}` is not
    structurally distinguishable from a `delegation` row's `status` field."""
    q: queue.Queue = queue.Queue()

    def _run():
        try:
            with httpx.stream("GET", url, timeout=timeout) as resp:
                collected = []
                current_event = "message"
                for line in resp.iter_lines():
                    if line.startswith("event:"):
                        current_event = line[len("event:"):].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line[len("data:"):].strip())
                        collected.append((current_event, data))
                        if len(collected) >= n:
                            break
                q.put(("ok", collected))
        except Exception as exc:  # noqa: BLE001
            q.put(("err", exc))

    threading.Thread(target=_run, daemon=True).start()
    try:
        kind, value = q.get(timeout=timeout)
    except queue.Empty:
        pytest.fail(f"did not receive {n} SSE events within {timeout}s")
    if kind == "err":
        raise value
    return value


@pytest.mark.smoke
def test_stream_replays_then_reflects_status_change(tmp_path):
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    log_path = run_dir / "debug" / "delegation_log.jsonl"
    _write_jsonl(log_path, [
        {"id": "D001", "status": "RUNNING", "from_node": "strategizer",
         "to_node": "critic"},
    ])

    def _append_done():
        time.sleep(0.5)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": "D001", "status": "DONE",
                "from_node": "strategizer", "to_node": "critic",
            }) + "\n")

    with _LiveServer(create_app(study)) as srv:
        threading.Thread(target=_append_done, daemon=True).start()
        events = _read_sse_events(
            f"{srv.url}/api/runs/20260904T120000/stream", n=2, timeout=8.0)

    assert events[0]["status"] == "RUNNING"
    assert events[1]["status"] == "DONE"


@pytest.mark.smoke
def test_stream_replays_run_status_when_already_present(tmp_path):
    """run_status.json written before the client ever connects (a run that
    already finished by the time someone opens the viewer) must replay
    immediately, not wait for a live filesystem event — it's write-once, so
    there is nothing to "tail"."""
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    (run_dir / "debug" / "run_status.json").write_text(
        json.dumps({"status": "GATED"}), encoding="utf-8")

    with _LiveServer(create_app(study)) as srv:
        events = _read_typed_sse_events(
            f"{srv.url}/api/runs/20260904T120000/stream", n=1, timeout=8.0)

    assert events[0] == ("run_status", {"status": "GATED"})


@pytest.mark.smoke
def test_stream_emits_run_status_once_it_appears_mid_stream(tmp_path):
    """A still-running run has no run_status.json yet at connect time; once
    the run closes and writes it, the already-open stream must emit it
    without the client needing to reconnect."""
    study = _make_study(tmp_path)
    run_dir = _make_run(study, "20260904T120000")
    status_path = run_dir / "debug" / "run_status.json"

    def _write_status():
        time.sleep(0.5)
        status_path.write_text(
            json.dumps({"status": "UNGATED"}), encoding="utf-8")

    with _LiveServer(create_app(study)) as srv:
        threading.Thread(target=_write_status, daemon=True).start()
        events = _read_typed_sse_events(
            f"{srv.url}/api/runs/20260904T120000/stream", n=1, timeout=8.0)

    assert events[0] == ("run_status", {"status": "UNGATED"})


def test_stream_404_for_missing_run(tmp_path):
    study = _make_study(tmp_path)
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/nonexistent/stream")
    assert resp.status_code == 404
