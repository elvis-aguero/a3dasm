"""Pure filesystem-reading functions backing the viewer's HTTP routes.

No HTTP here — every function takes a path (or a live ``Graph``) and returns
plain data, so these are trivially unit-testable against ``tmp_path`` fixture
debug dirs with no server involved.

Graceful degradation is the load-bearing contract throughout: a run that
hasn't started yet, or was started without the debug flag, must produce an
empty/"not available" result from these functions, never an exception — the
frontend's "idle/no data yet" rendering path is the SAME code path as "file
missing", not a special case bolted on afterward.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..delegation_log import DelegationLog

__all__ = [
    "read_runs",
    "read_delegations",
    "read_diagnostics_tail",
    "read_run_status",
    "read_transcript",
    "read_problem_statement",
    "list_node_transcripts",
    "graph_spec_json",
    "load_graph_for_study",
    "tail_jsonl",
]


def read_runs(study_dir: Path | str) -> list[dict[str, Any]]:
    """List ``<study_dir>/runs/*/debug`` dirs, newest first.

    A "run" is any entry under ``runs/`` that has a ``debug/`` subdirectory —
    NOT every directory under ``runs/`` (e.g. a literature-corpus cache dir
    like ``lit_reviewer_notes`` lives alongside real run dirs at the same
    level and must not be misclassified as a run).
    """
    runs_dir = Path(study_dir) / "runs"
    if not runs_dir.is_dir():
        return []
    out = []
    for entry in sorted(runs_dir.iterdir(), reverse=True):
        debug_dir = entry / "debug"
        if not debug_dir.is_dir():
            continue
        status_data = read_run_status(entry)
        out.append({
            "run_id": entry.name,
            "path": str(entry),
            "has_debug": (debug_dir / "delegation_log.jsonl").exists()
            or any(debug_dir.iterdir()),
            "status": (status_data or {}).get("status", "running"),
        })
    return out


def read_delegations(run_dir: Path | str) -> list[dict[str, Any]]:
    """Collapsed, last-write-wins delegation rows for one run.

    Reuses ``DelegationLog.query_all()`` directly rather than
    re-implementing its collapsing logic — safe to construct read-side:
    ``DelegationLog.__init__``'s only side effect is an idempotent
    ``mkdir(parents=True, exist_ok=True)`` on a directory that already
    exists (the run's own ``debug/`` dir).
    """
    log_path = Path(run_dir) / "debug" / "delegation_log.jsonl"
    return DelegationLog(log_path).query_all()


def read_diagnostics_tail(run_dir: Path | str) -> list[dict[str, Any]]:
    """Every diagnostics.jsonl row seen so far (open vocabulary — no fixed
    schema beyond ``ts``/``node``/``tool``/``error_type``/``fault``/
    ``message``)."""
    path = Path(run_dir) / "debug" / "diagnostics.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_run_status(run_dir: Path | str) -> dict[str, Any] | None:
    """``run_status.json``'s contents, or ``None`` if the run hasn't closed
    yet (normal close and crash both write this file, but only once, at the
    very end — absence means "still running", not an error)."""
    path = Path(run_dir) / "debug" / "run_status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_transcript(run_dir: Path | str, key: str) -> list[dict[str, Any]] | None:
    """Parsed events for one transcript file.

    *key* is either ``"strategizer/turn_003"`` (the strategizer's own turns,
    nested under ``transcripts/strategizer/``) or a bare delegation id like
    ``"D007"`` (every worker delegation, flat under ``transcripts/``) —
    matching the two real on-disk layouts exactly.

    Returns ``None`` (not ``[]``) when ``transcripts/`` doesn't exist AT ALL
    for this run — the debug flag was off, a distinct condition from "this
    specific key wasn't found" (which returns ``[]``), so callers can render
    two different, honest messages instead of one generic "nothing here".
    """
    transcripts_dir = Path(run_dir) / "debug" / "transcripts"
    if not transcripts_dir.is_dir():
        return None
    path = transcripts_dir / f"{key}.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_problem_statement(run_dir: Path | str) -> str | None:
    """The exact PROBLEM_STATEMENT.md this run answered — the verbatim
    snapshot (``debug/PROBLEM_STATEMENT_snapshot.md``), never the live
    study_dir file, which may have since been edited for a later run.
    ``None`` if the snapshot doesn't exist (a run from before this file
    was introduced, or one that crashed before writing it)."""
    path = Path(run_dir) / "debug" / "PROBLEM_STATEMENT_snapshot.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def list_node_transcripts(
    run_dir: Path | str, node_name: str, is_entry: bool = False,
) -> list[str]:
    """Real, disk-verified transcript keys available for one node —
    NEVER computed/guessed from a formula (a prior version of this viewer
    guessed the entry node's turn number as
    ``f"strategizer/turn_{len(delegations_touching_it):03d}"``, which has
    no real relationship to actual turn-file numbering and would 404 or
    open the wrong turn silently — found and removed, not left as a
    latent bug).

    A worker node's keys are the ``id`` of every delegation where
    ``to_node == node_name`` (each maps 1:1 to a real
    ``transcripts/{id}.jsonl`` file). The entry node has no delegations
    TO it in a typical graph, so its keys are instead every real
    ``transcripts/strategizer/turn_*.jsonl`` file that actually exists,
    sorted — never a guessed count.
    """
    if is_entry:
        st_dir = Path(run_dir) / "debug" / "transcripts" / "strategizer"
        if not st_dir.is_dir():
            return []
        return sorted(
            f"strategizer/{p.stem}" for p in st_dir.glob("turn_*.jsonl"))
    return [
        d["id"] for d in read_delegations(run_dir)
        if d.get("to_node") == node_name
    ]


def graph_spec_json(graph, study_dir=None) -> dict[str, Any]:
    """Node/edge/role/tool/layer JSON for the live network diagram.

    Reuses ``run_diagram._bfs_layers()``/``_node_tools()`` directly rather
    than re-deriving topology/tool-surface logic — the viewer only needs a
    layer index per node (for a simple CSS-grid row placement) and the real
    tool list, not the static SVG's pixel-precise card layout (built for a
    richer, wrapped-text card the live viewer doesn't need).
    """
    from ..run_diagram import _bfs_layers, _node_tools

    layers = _bfs_layers(graph)
    nodes = []
    for name, agent in graph.nodes.items():
        nodes.append({
            "name": name,
            "role": agent.role,
            "description": agent.description or "",
            "is_entry": name == graph.entry,
            "layer": layers[name],
            "tools": _node_tools(name, agent, graph, study_dir),
        })
    edges = [{"source": e.source, "target": e.target} for e in graph.edges]
    return {"nodes": nodes, "edges": edges, "entry": graph.entry}


def load_graph_for_study(study_dir: Path | str):
    """Best-effort: recover the ``Graph`` a study's runs actually use, for
    the standalone CLI path (``python -m a3dasm.viewer <study-dir>``), which
    has no in-memory ``Graph`` object the way ``AgenticRun.serve_viewer()``
    does — nothing durable on disk captures graph topology
    (``run_config.json`` only holds canonical-store/evaluator config,
    confirmed by reading a real one), so this is the only way to recover it
    generically.

    Tries ``<study_dir>/run.py``'s ``build_graph()`` first — the convention
    every custom-graph study in this repo follows (dynamically imports and
    executes that module; module-level code in every study's ``run.py`` in
    this repo only defines functions/constants, guarded by
    ``if __name__ == "__main__"``, so this is safe here, but is inherently
    running study-specific code, not sandboxed). Falls back to the stock
    5-node ``_default_graph()`` when no custom ``run.py``/``build_graph`` is
    importable — that IS the real topology a study with no custom graph
    actually runs. Returns ``None`` only if even that fails (should not
    happen in practice); callers must still degrade gracefully.
    """
    import importlib.util

    study_dir = Path(study_dir)
    run_py = study_dir / "run.py"
    if run_py.exists():
        try:
            spec = importlib.util.spec_from_file_location(
                f"_a3dasm_viewer_study_{study_dir.name}", run_py)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            build_graph = getattr(module, "build_graph", None)
            if build_graph is not None:
                return build_graph()
        except Exception:  # noqa: BLE001
            pass
    try:
        from ..agents import _default_graph
        return _default_graph()
    except Exception:  # noqa: BLE001
        return None


def tail_jsonl(
    path: Path | str, poll_interval: float = 0.5,
) -> Iterator[dict[str, Any]]:
    """Yield parsed JSON objects appended to *path*, forever, starting from
    current end-of-file at call time.

    If *path* does not exist yet (a run hasn't started, or debug wasn't
    enabled), polls for its creation before tailing — never raises. Tolerant
    of a partial (no trailing newline) write: an incomplete trailing line is
    simply not yielded until a later poll sees it terminated.

    Deliberately polling, not an OS-level file-watcher (``watchdog``, already
    resolvable transitively in this venv): FSEvents (macOS) is known to
    coalesce rapid successive writes with platform-specific latency
    quirks for append-detection specifically, whereas polling is identical
    cross-platform and trivial to test deterministically — and at
    human-dashboard timescales (a handful of JSONL rows per second at
    most), the latency difference against a real watcher is imperceptible.
    """
    path = Path(path)
    # A pre-existing file: skip its current content, tail only new growth —
    # EXCEPT any trailing partial (unterminated) line, which is backed up to
    # instead of skipped. A generator's body only starts running on its
    # FIRST next() call, not when tail_jsonl() itself is invoked — so a
    # caller that starts iterating slightly late (any real consumer that
    # isn't a bare `for` loop starting instantly, e.g. one driven from a
    # separate thread) can genuinely observe the file already mid-line at
    # the moment this offset is captured, if a writer's append happened to
    # land in that gap. Naively setting offset = current size would then
    # silently swallow that in-flight line's prefix forever: once its
    # remainder arrives, reading from `offset` onward yields only the
    # fragment, which is never valid JSON on its own — the exact failure
    # this caused before backing up (confirmed by direct reproduction: ~25%
    # of runs of a test writing a line in two parts never yielded it at
    # all). Backing up to the position right after the LAST newline (or 0 if
    # the file has none at all) means that partial suffix is treated as new,
    # unconsumed content — the same as if tailing had started before it was
    # ever written.
    # A not-yet-created file: offset stays 0 once it appears — the whole
    # first write is "new" from this tailer's point of view, however much
    # content it happens to contain (a single write_text() can create the
    # file AND write its full content before we ever observe exists()==True,
    # so setting offset = size-at-creation-time would silently skip
    # whatever was written before we got around to checking).
    if path.exists():
        content = path.read_bytes()
        offset = content.rfind(b"\n") + 1  # 0 if no newline is present at all
    else:
        offset = 0
        while not path.exists():
            time.sleep(poll_interval)
    buffer = ""
    while True:
        size = path.stat().st_size
        if size > offset:
            with path.open("r", encoding="utf-8") as f:
                f.seek(offset)
                chunk = f.read()
            offset = size
            buffer += chunk
            *complete, buffer = buffer.split("\n")
            for line in complete:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        time.sleep(poll_interval)
