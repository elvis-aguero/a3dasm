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


def _display_tool_name(name: str) -> str:
    """Strip a leading ``mcp__<server>__`` prefix for on-screen display —
    the raw registered name (e.g. ``mcp__f3dasm_agent_tools__Delegate``) is
    what the model actually calls, kept in the ``title`` attribute, but
    showing it verbatim in every bubble is unreadable noise the reader has
    to mentally strip on every single line."""
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3 and parts[2]:
            return parts[2]
    return name


def _bubble_html(event: dict) -> str:
    """Render one ``assistant`` event as a chat-turn HTML fragment.

    Matches the nested trace-tree convention real agent-trace viewers use
    (thinking, then text, then each tool call as a subordinate child of the
    SAME turn) rather than a flat list of same-weight bubbles — a tool
    call's matching result is attached separately, by
    ``get_transcript_fragment``, via ``_tool_result_html`` (results arrive
    as their own later event on disk, so it can't be inlined here).
    ``stream_evt``/``partial`` are intentionally not rendered — no live
    token-by-token typing effect yet.
    """
    if event.get("type") != "assistant":
        return ""
    text = event.get("text") or ""
    thinking = "".join(event.get("thinking") or [])
    thinking_html = (
        "<details class='thinking'><summary>&#9656; thinking</summary>"
        f"<div class='thinking-text'>{_esc(thinking)}</div></details>"
    ) if thinking.strip() else ""
    tools_html = ""
    for tool in event.get("tools") or []:
        raw_name = tool.get("name", "tool")
        tools_html += (
            "<div class='tool-call'>"
            "<span class='tool-icon'>&#8226;</span>"
            f"<span class='tool-name' title='{_esc(raw_name)}'>"
            f"{_esc(_display_tool_name(raw_name))}</span>"
            "<details><summary>args</summary>"
            f"<pre>{_esc(json.dumps(tool.get('input', {}), indent=2))}</pre>"
            "</details></div>"
        )
    if not text and not thinking_html and not tools_html:
        return ""
    return (
        "<div class='turn'>"
        "<span class='avatar avatar-assistant' title='assistant'>A</span>"
        "<div class='turn-body'>"
        f"{thinking_html}"
        f"<div class='bubble-text'>{_esc(text)}</div>"
        f"{tools_html}"
        "</div></div>"
    )


def _tool_result_html(event: dict, names: list[str]) -> str:
    """Render a ``tool_result`` event as a subordinate continuation row —
    visually attached under the tool call it answers (no independent bubble
    chrome), labelled with the REAL tool name it belongs to. *names* is this
    event's ``results`` list, positionally resolved against the transcript's
    full in-order tool-call queue by ``get_transcript_fragment`` (a
    tool_result event carries only ``tool_use_id``, not the name — see that
    function's docstring for why positional resolution is safe here)."""
    results = event.get("results") or []
    html = ""
    for i, r in enumerate(results):
        name = names[i] if i < len(names) else "tool"
        content = r.get("content", "")
        html += (
            "<div class='tool-result-row'>"
            "<span class='tool-icon result'>&#8618;</span>"
            f"<details><summary>{_esc(_display_tool_name(name))} result</summary>"
            f"<pre>{_esc(json.dumps(content, indent=2))}</pre>"
            "</details></div>"
        )
    return html


def _render_fragment(events: list[dict], after: int) -> str:
    """Render ``events[after:]`` to HTML, resolving each ``tool_result``
    event's real tool name(s) by position against the FULL event list.

    A ``tool_result`` event only carries ``tool_use_id`` (confirmed against
    a real transcript — the preceding ``assistant`` event's ``tools`` list
    carries no id to match against). Claude's own conversation structure
    strictly alternates assistant-tool-call -> tool_result before the next
    assistant turn, so the Nth item across every tool_result event, in file
    order, is always the Nth tool call across every assistant event, in file
    order — recomputed over the WHOLE file (not just this slice) so the
    mapping is correct regardless of where ``after`` falls, then only the
    slice past ``after`` is actually rendered.
    """
    all_names = [
        tool.get("name", "tool")
        for e in events if e.get("type") == "assistant"
        for tool in (e.get("tools") or [])
    ]
    consumed = sum(
        len(e.get("results") or [])
        for e in events[:after] if e.get("type") == "tool_result"
    )
    html_parts = []
    for e in events[after:]:
        if e.get("type") == "tool_result":
            results = e.get("results") or []
            names = all_names[consumed:consumed + len(results)]
            consumed += len(results)
            html_parts.append(_tool_result_html(e, names))
        else:
            html_parts.append(_bubble_html(e))
    return "".join(html_parts)


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

    async def get_problem_statement(request):
        run_id = request.path_params["run_id"]
        run_dir = _run_dir(study_dir, run_id)
        if run_dir is None:
            return _not_found(f"no such run {run_id!r}")
        text = readers.read_problem_statement(run_dir)
        if text is None:
            return _not_found("PROBLEM_STATEMENT_snapshot.md not found for this run")
        return JSONResponse({"text": text})

    async def get_node_transcripts(request):
        run_id = request.path_params["run_id"]
        name = request.path_params["name"]
        run_dir = _run_dir(study_dir, run_id)
        if run_dir is None:
            return _not_found(f"no such run {run_id!r}")
        is_entry = graph is not None and name == graph.entry
        return JSONResponse(
            readers.list_node_transcripts(run_dir, name, is_entry=is_entry))

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
                "(debug flag was off).</p>", status_code=404)
        if not events:
            # The FULL list (before slicing by `after`) being empty means
            # this key has no transcript at all — e.g. a Done()-gate-check
            # delegation_log entry (confirmed for real: "GATE190739",
            # status GATE:PASS, has no transcripts/GATE190739.jsonl at all,
            # since the gate check is recorded as a bookkeeping row, not a
            # real sub-delegation with its own captured conversation) — NOT
            # "an open, still-running transcript with nothing new since the
            # last poll" (there, `events` itself is non-empty; only the
            # `events[after:]` SLICE is). Without this check the endpoint
            # silently returned 200 with an empty body for a key that will
            # NEVER have content, and the panel just went blank with no
            # explanation at all.
            return HTMLResponse(
                f"<p class='empty'>No transcript recorded for {key!r} — "
                "likely a bookkeeping record (e.g. a gate-check verdict) "
                "rather than a captured conversation.</p>",
                status_code=404)
        html = _render_fragment(events, after)
        # `after`/the response cursor MUST count raw events, not rendered
        # bubbles — most events (stream_evt/partial/result) render to no
        # bubble at all (confirmed for real: a genuine transcript had 527
        # raw events, only 47 of them assistant/tool_result). A cursor
        # counting bubbles instead falls behind the raw list on every call,
        # so events already shown keep re-matching `events[after:]` and get
        # duplicated on every poll, without the underlying file ever
        # changing. X-Event-Count is the actual raw count consumed this
        # call — the client's next `after` value, not a bubble count.
        return HTMLResponse(
            html, headers={"X-Event-Count": str(len(events))})

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

            # run_status.json is written ONCE, at the very end (GATED/
            # UNGATED/FAILED/crashed) -- distinct, global "how did the
            # WHOLE run turn out" signal, separate from any per-node
            # delegation status (a node's own dot only ever reflects its
            # most recent incoming delegation, which the entry node never
            # has one of at all). Absence means "still running" -- there
            # is no separate "RUNNING" state written anywhere; it's the
            # client's own default until this event ever arrives.
            run_status_seen = False
            initial_status = readers.read_run_status(run_dir)
            if initial_status is not None:
                yield _sse("run_status", initial_status)
                run_status_seen = True

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
                if not run_status_seen:
                    # Written once, so a plain existence poll here (not the
                    # append-aware tail_jsonl, which is for growing files)
                    # is enough — cheap, and stops once found.
                    current = readers.read_run_status(run_dir)
                    if current is not None:
                        yield _sse("run_status", current)
                        run_status_seen = True
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
        Route("/api/runs/{run_id}/problem_statement", get_problem_statement),
        Route("/api/runs/{run_id}/node/{name}/transcripts", get_node_transcripts),
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
