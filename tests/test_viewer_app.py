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


def test_transcript_fragment_empty_html_when_debug_off(tmp_path):
    study = _make_study(tmp_path)
    _make_run(study, "20260904T120000")
    client = TestClient(create_app(study))
    resp = client.get(
        "/api/runs/20260904T120000/transcript/D007/fragment?after=0")
    assert resp.status_code == 200
    assert "debug flag" in resp.text


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
    test suite."""
    q: queue.Queue = queue.Queue()

    def _run():
        try:
            with httpx.stream("GET", url, timeout=timeout) as resp:
                collected = []
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        collected.append(json.loads(line[len("data:"):].strip()))
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


def test_stream_404_for_missing_run(tmp_path):
    study = _make_study(tmp_path)
    client = TestClient(create_app(study))
    resp = client.get("/api/runs/nonexistent/stream")
    assert resp.status_code == 404
