"""Starlette app for the read-only live run viewer.

Requires the ``viewer`` extra (``pip install a3dasm[viewer]``) — imported
lazily by every caller (``agent_runtime.py``'s ``serve_viewer()``,
``viewer/__main__.py``) so the core install never depends on Starlette.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route
from starlette.templating import Jinja2Templates

from . import readers

__all__ = ["create_app", "run_viewer"]

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _run_dir(study_dir: Path, run_id: str) -> Path | None:
    run_dir = study_dir / "runs" / run_id
    if not (run_dir / "debug").is_dir():
        return None
    return run_dir


def _not_found(message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=404)


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _bubble_html(event: dict) -> str:
    """Render one transcript event as a chat-bubble HTML fragment.

    Only ``assistant``/``tool_result`` events render (Claude Code's own
    transcript-viewer convention: turn text as bubbles, tool calls/results
    collapsed by default); ``stream_evt``/``partial`` are intentionally not
    rendered in v1 — no live token-by-token typing effect yet.
    """
    etype = event.get("type")
    if etype == "assistant":
        text = event.get("text") or ""
        tools_html = ""
        for tool in event.get("tools") or []:
            tools_html += (
                "<details class='tool-call'><summary>&#9656; "
                f"{_esc(tool.get('name', 'tool'))}(...)</summary>"
                f"<pre>{_esc(json.dumps(tool.get('input', {}), indent=2))}</pre>"
                "</details>"
            )
        return (
            "<div class='bubble bubble-assistant'>"
            f"<div class='bubble-text'>{_esc(text)}</div>{tools_html}</div>"
        )
    if etype == "tool_result":
        results_html = ""
        for r in event.get("results") or []:
            content = r.get("content", "")
            results_html += (
                "<details class='tool-result'><summary>&#9656; tool result"
                f"</summary><pre>{_esc(json.dumps(content, indent=2))}</pre>"
                "</details>"
            )
        return f"<div class='bubble bubble-tool'>{results_html}</div>"
    return ""


def _esc(s: str) -> str:
    return (
        (s or "").replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )


def create_app(study_dir: Path | str, graph=None) -> Starlette:
    """*graph*: pass the study's real, already-in-memory ``Graph`` when one
    is available (``AgenticRun.serve_viewer()`` always has it — the same
    object ``self._graph_spec`` holds, no need to reconstruct anything).
    Left as ``None`` for the standalone CLI path (a separate process from
    the run itself, with no in-memory Graph to hand over), which falls back
    to ``readers.load_graph_for_study()`` instead.
    """
    study_dir = Path(study_dir)
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    # The Graph is the same for every run of one study — resolved once at
    # app construction, not per-request.
    if graph is None:
        graph = readers.load_graph_for_study(study_dir)

    async def list_runs(request):
        return JSONResponse(readers.read_runs(study_dir))

    async def get_graph(request):
        run_id = request.path_params["run_id"]
        if _run_dir(study_dir, run_id) is None:
            return _not_found(f"no such run {run_id!r}")
        if graph is None:
            return JSONResponse({"nodes": [], "edges": [], "entry": None})
        return JSONResponse(readers.graph_spec_json(graph, study_dir))

    async def get_delegations(request):
        run_id = request.path_params["run_id"]
        run_dir = _run_dir(study_dir, run_id)
        if run_dir is None:
            return _not_found(f"no such run {run_id!r}")
        return JSONResponse(readers.read_delegations(run_dir))

    async def get_transcript(request):
        run_id = request.path_params["run_id"]
        key = request.path_params["key"]
        run_dir = _run_dir(study_dir, run_id)
        if run_dir is None:
            return _not_found(f"no such run {run_id!r}")
        events = readers.read_transcript(run_dir, key)
        if events is None:
            return _not_found(
                "transcripts not recorded for this run (debug flag was off)")
        if not events:
            return _not_found(f"no such transcript {key!r}")
        return JSONResponse(events)

    async def get_transcript_fragment(request):
        run_id = request.path_params["run_id"]
        key = request.path_params["key"]
        after = int(request.query_params.get("after", 0))
        run_dir = _run_dir(study_dir, run_id)
        if run_dir is None:
            return _not_found(f"no such run {run_id!r}")
        events = readers.read_transcript(run_dir, key)
        if events is None:
            return HTMLResponse(
                "<p class='empty'>Transcripts not recorded for this run "
                "(debug flag was off).</p>")
        html = "".join(_bubble_html(e) for e in events[after:])
        return HTMLResponse(html)

    async def stream(request):
        run_id = request.path_params["run_id"]
        run_dir = _run_dir(study_dir, run_id)
        if run_dir is None:
            return _not_found(f"no such run {run_id!r}")

        async def event_gen():
            for row in readers.read_delegations(run_dir):
                yield _sse("delegation", row)
            for row in readers.read_diagnostics_tail(run_dir):
                yield _sse("diagnostic", row)

            q: queue.Queue = queue.Queue()

            def _tail_delegations():
                for _ in readers.tail_jsonl(
                    run_dir / "debug" / "delegation_log.jsonl"
                ):
                    q.put(("delegation_touched", None))

            def _tail_diagnostics():
                for row in readers.tail_jsonl(
                    run_dir / "debug" / "diagnostics.jsonl"
                ):
                    q.put(("diagnostic", row))

            threading.Thread(target=_tail_delegations, daemon=True).start()
            threading.Thread(target=_tail_diagnostics, daemon=True).start()

            last_status = {
                r["id"]: r["status"] for r in readers.read_delegations(run_dir)
            }
            while True:
                if await request.is_disconnected():
                    break
                try:
                    kind, payload = q.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.2)
                    continue
                if kind == "delegation_touched":
                    for row in readers.read_delegations(run_dir):
                        if last_status.get(row["id"]) != row["status"]:
                            last_status[row["id"]] = row["status"]
                            yield _sse("delegation", row)
                elif kind == "diagnostic":
                    yield _sse("diagnostic", payload)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    async def graph_page(request):
        run_id = request.path_params["run_id"]
        if _run_dir(study_dir, run_id) is None:
            return _not_found(f"no such run {run_id!r}")
        return templates.TemplateResponse(
            request, "graph.html", {"run_id": run_id})

    routes = [
        Route("/api/runs", list_runs),
        Route("/api/runs/{run_id}/graph", get_graph),
        Route("/api/runs/{run_id}/delegations", get_delegations),
        Route(
            "/api/runs/{run_id}/transcript/{key:path}/fragment",
            get_transcript_fragment,
        ),
        Route("/api/runs/{run_id}/transcript/{key:path}", get_transcript),
        Route("/api/runs/{run_id}/stream", stream),
        Route("/runs/{run_id}", graph_page),
    ]
    return Starlette(routes=routes)


def run_viewer(
    study_dir: Path | str, host: str = "127.0.0.1", port: int = 8765,
    graph=None,
) -> None:
    """Launch the viewer web server (blocking). Requires ``uvicorn``."""
    import uvicorn

    uvicorn.run(create_app(study_dir, graph=graph), host=host, port=port)
