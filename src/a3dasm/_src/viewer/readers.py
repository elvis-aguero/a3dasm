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
import re
import time
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..epistemics.delegation_log import DelegationLog

__all__ = [
    "read_runs",
    "read_delegations",
    "read_diagnostics_tail",
    "read_hypotheses",
    "read_milestones",
    "read_notebook",
    "read_vitals",
    "read_oracle",
    "read_artifacts",
    "read_artifact_text",
    "read_run_status",
    "read_transcript",
    "read_problem_statement",
    "list_node_transcripts",
    "graph_spec_json",
    "load_graph_for_study",
    "tail_jsonl",
]

# Docs for the fixed, well-known tool names that are never real Python
# closures the agent's own `build_closure_tools()` returns (native backend
# tools, and the topology/protocol tools the runtime wires in from
# routing.py's nested closures — those docstrings live on functions built
# per-run inside a live registry, not importable statically). Copied
# verbatim (first paragraph) from routing.py/the Claude backend's own native
# tool set as of this writing, so the hover tooltip says the same thing the
# agent itself was told, not an invented paraphrase.
_KNOWN_TOOL_DOCS: dict[str, str] = {
    "Bash": "Executes a shell command in a persistent session.",
    "Read": "Reads a file from the local filesystem (text, image, or PDF).",
    "Write": "Writes a file to the local filesystem, overwriting it if it "
             "already exists.",
    "Edit": "Performs an exact string replacement in a file.",
    "Grep": "Searches file contents for a pattern (ripgrep-backed).",
    "BashOutput": "Retrieves output from a running or completed background "
                  "bash shell.",
    "KillShell": "Kills a running background bash shell by its id.",
    "Glob": "Finds files matching a glob pattern, sorted by modification "
            "time.",
    "Delegate": "Hand a task to another node in the graph; returns "
                "immediately (async) unless wait=True.",
    "Wait": "Block until a delegation finishes (Done or Errored), then "
            "return its result — use instead of polling GetStatus().",
    "Reply": "Answer a worker's FollowUp question and unblock it.",
    "FollowUp": "Ask the delegating party one clarifying question before "
                "proceeding.",
    "RecallHistory": "Return the last N delegations received by this node "
                      "as (task, deliverable) pairs.",
    "GetStatus": "Poll a background delegation; also delivers push "
                 "notifications.",
    "Done": "Signal end of run with a summary of findings (two-shot: first "
            "call warns, second call closes).",
    "WriteNote": "Write a Markdown note to strategizer_notes/ — free-form "
                 "reasoning, not code or hypothesis priors.",
    "ReadNote": "Read a file, or list a directory, from the study "
                "directory (e.g. PROBLEM_STATEMENT.md, prior notes, a "
                "delegation's own workspace).",
    "ReadProblemStatement": "The run's PROBLEM_STATEMENT.md verbatim: what "
                            "this run is actually trying to establish, and "
                            "its stated success/termination criteria.",
}

# Best-effort humanization of a model id into the name the model is
# actually known by — matches the naming this project's own agent (Claude
# Code) uses for itself, so the two stay consistent.
_MODEL_LABELS: dict[str, str] = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-opus-5": "Claude Opus 5",
    "claude-fable-5": "Claude Fable 5",
}


@lru_cache(maxsize=1)
def _routing_tool_docs() -> dict[str, str]:
    """Tool docstrings read straight out of ``nodes/tools/routing.py``'s source.

    Most tools an agent declares are implemented as functions nested inside
    per-run closure builders, so they exist only once a live registry has
    been constructed — ``build_closure_tools()`` returns a handful, and the
    rest cannot be imported to read ``__doc__`` from at all.

    The alternative was a hand-maintained copy of those docstrings, which
    is what ``_KNOWN_TOOL_DOCS`` is; the trouble with that is silent drift —
    the map keeps claiming what a tool did a year ago and nothing fails.
    Parsing the module's AST reads the SAME docstring the agent is given
    (``tool_catalog.render_tool_catalog`` uses ``__doc__``), stays correct
    as routing.py changes, and executes none of it.

    Only the first paragraph is kept, matching what the tool catalog shows.
    """
    import ast

    nodes_dir = Path(__file__).parent.parent / "nodes"
    out: dict[str, str] = {}
    # Tool definitions are split across the routing layer and the node
    # modules that inject their own; scanned in a fixed order so the same
    # name defined twice resolves the same way on every call.
    for rel in ("tools/routing.py", "strategizer.py", "worker.py"):
        path = nodes_dir / rel
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            # Tools are PascalCase; helpers and privates are not. This keeps
            # a local helper from being mistaken for a tool of the same name.
            if not node.name[:1].isupper():
                continue
            doc = (ast.get_docstring(node) or "").strip()
            if doc and node.name not in out:
                out[node.name] = doc.split("\n\n")[0].replace("\n", " ")
    return out


def _humanize_model(model_id: str | None) -> str:
    if not model_id:
        return "(backend default)"
    return _MODEL_LABELS.get(model_id, model_id)


def _load_study_config(study_dir: Path | None) -> dict[str, Any]:
    """Mirrors ``agent_runtime._load_study_config`` exactly (kept as its own
    copy rather than an import, to keep this module's only cross-package
    dependency the lightweight ``delegation_log`` — the viewer must stay
    importable without pulling in ``agent_runtime``'s much heavier stack)."""
    if study_dir is None:
        return {}
    cfg_path = Path(study_dir) / "config.yaml"
    if not cfg_path.exists():
        return {}
    import yaml

    with cfg_path.open(encoding="utf-8", errors="replace") as f:
        return yaml.safe_load(f) or {}


def _node_tools_and_docs(
    name: str, agent, graph, study_dir=None,
) -> tuple[list[str], dict[str, str]]:
    """Same tool-surface logic as ``run_diagram._node_tools`` (declared +
    topology + real runtime-injected closures), but ALSO returns each
    closure's real docstring — the same text the agent itself was given
    (``tool_catalog.render_tool_catalog`` reads this exact ``__doc__``).

    Kept as its own copy in the viewer rather than extending
    ``run_diagram._node_tools`` in place: that function is shared with the
    tested static SVG renderer, and its own docstring already warns that
    calling ``agent.build_closure_tools()`` twice per node is wasteful and
    can double any real side effect a closure builder has — a second,
    independent call from here would be exactly that mistake. This
    duplicates ~10 lines of topology-tool logic instead, at zero shared-code
    risk.
    """
    from ..runtime.run_diagram import (
        _TOPOLOGY_TOOLS_IF_OUTGOING,
        _topology_tools,
    )

    declared = sorted(set(agent.tools) - set(_TOPOLOGY_TOOLS_IF_OUTGOING))
    tools = _topology_tools(graph, name) + declared
    docs: dict[str, str] = {}
    if study_dir is not None:
        # Built against a THROWAWAY directory, not the study. Closure
        # builders are not side-effect free — LiteratureReviewAgent's
        # constructs a LiteratureCorpus, whose __init__ mkdir()s
        # runs/lit_reviewer_notes/ and papers/ — and a read-only viewer must
        # not create directories inside the study it is only reading. The
        # tool NAMES and docstrings are what is wanted here and neither
        # depends on the path, so a temp dir yields the same surface with
        # the writes landing somewhere disposable.
        import tempfile
        try:
            with tempfile.TemporaryDirectory(prefix="a3dasm-viewer-") as tmp:
                closures = agent.build_closure_tools(Path(tmp))
        except Exception:  # noqa: BLE001
            closures = {}
        extra = sorted(
            set(closures) - set(tools) - set(_TOPOLOGY_TOOLS_IF_OUTGOING))
        tools = tools + extra
        for tool_name, fn in closures.items():
            doc = (getattr(fn, "__doc__", None) or "").strip()
            if doc:
                docs[tool_name] = doc.split("\n\n")[0].replace("\n", " ")
    return tools, docs


# Node-box geometry for the viewer's own graph layout, in abstract CSS px at
# zoom 1. Deliberately NOT reused from run_diagram: that module's constants
# size a 400px-wide documentation card whose HEIGHT depends on how much tool
# text it wraps, which is the wrong shape for a compact live diagram.
_NODE_W = 220
_NODE_H = 74
_COL_GAP = 90
_ROW_GAP = 150
_MARGIN = 40
# Must match the client's same-row edge routing in graph.html.
_SAME_ROW_LANE = 60
_SAME_ROW_STEP = 18
# A layer wider than this wraps onto further rows. Every non-entry node
# shares one layer, so a six-node graph put five boxes in a single row
# 1550px wide — past the viewport, forcing a horizontal scroll to see the
# agents at all.
_MAX_ROW_W = 1180


def _layout_nodes(
    layers: dict[str, int], same_row_edges: int = 0,
) -> tuple[dict[str, tuple[int, int]], int, int]:
    """Deterministic (x, y) per node, plus the canvas size that contains them.

    Computed here, server-side, rather than measured from the rendered DOM.
    The previous client drew edges by reading ``getBoundingClientRect()``
    after layout, which made every edge depend on layout timing, webfont
    load and scroll position — the "flaky arrows" failure mode. Emitting
    authoritative coordinates makes node placement and edge endpoints the
    same numbers by construction, so an edge cannot disagree with the box it
    points at, and no resize handler is needed to keep them in sync.

    Within a layer, nodes are ordered by name so the layout is stable across
    calls (``graph.nodes`` is insertion-ordered, which would silently
    reshuffle the diagram if a study reordered its node registrations).
    """
    by_layer: dict[int, list[str]] = {}
    for name, layer in layers.items():
        by_layer.setdefault(layer, []).append(name)
    for names in by_layer.values():
        names.sort()

    def row_width(n: int) -> int:
        return n * _NODE_W + max(0, n - 1) * _COL_GAP

    per_row = max(1, (_MAX_ROW_W + _COL_GAP) // (_NODE_W + _COL_GAP))

    # Each layer becomes one or more visual rows, so a wide layer wraps
    # instead of running off the side of the screen.
    visual_rows: list[list[str]] = []
    for layer in sorted(by_layer):
        names = by_layer[layer]
        for i in range(0, len(names), per_row):
            visual_rows.append(names[i:i + per_row])

    widest = max((row_width(len(r)) for r in visual_rows), default=0)
    centre = _MARGIN + widest / 2

    pos: dict[str, tuple[int, int]] = {}
    for row_index, names in enumerate(visual_rows):
        left = centre - row_width(len(names)) / 2
        y = _MARGIN + row_index * (_NODE_H + _ROW_GAP)
        for i, name in enumerate(names):
            pos[name] = (round(left + i * (_NODE_W + _COL_GAP)), y)

    canvas_w = widest + 2 * _MARGIN
    depth = max(0, len(visual_rows) - 1)
    canvas_h = _MARGIN * 2 + (depth + 1) * _NODE_H + depth * _ROW_GAP
    # Same-row edges are routed BELOW the row they connect, one lane each.
    # Without reserving that space the SVG ends above them and the edges are
    # simply cut off mid-canvas — which is what a 6-node graph showed: the
    # lanes fell at y=384 and 402 in a 378px canvas.
    if same_row_edges:
        canvas_h += _SAME_ROW_LANE + (same_row_edges - 1) * _SAME_ROW_STEP
    return pos, canvas_w, canvas_h


def _identity_indices(layers: dict[str, int], entry: str) -> dict[str, int]:
    """Stable identity-colour index per node, or -1 for the entry node.

    The entry node is excluded from the hue ramp on purpose: it is the only
    node that can reach the human operator (``FollowUp`` routes to the
    operator only when the asker IS the entry node), so it reads as neutral
    rather than as one agent among peers.

    Ordered by (layer, name) so a node keeps its colour across runs of the
    same study, and so two studies sharing a topology colour it identically.
    """
    ordered = sorted(
        (n for n in layers if n != entry), key=lambda n: (layers[n], n))
    out = {name: i for i, name in enumerate(ordered)}
    out[entry] = -1
    return out


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
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


# The epistemic ledger lives under a FIXED directory name, not one derived
# from the entry node (``agent_runtime`` writes
# ``debug/strategizer_notes`` literally, even for a graph whose entry node
# is called something else).
_NOTES_DIR = "strategizer_notes"


def _read_json_object(path: Path) -> dict[str, Any]:
    """A JSON object from *path*, or ``{}`` for any absence or damage.

    Both ledgers are written repeatedly during a live run, so a read can
    genuinely catch a half-written file — that must degrade to "nothing yet"
    like every other missing input here, never to a 500.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def read_hypotheses(run_dir: Path | str) -> list[dict[str, Any]]:
    """Hypotheses with their CURRENT status already resolved.

    ``hypotheses.json`` stores an append-only ``status_log`` per hypothesis
    rather than a status field, so the live status is the last entry's. That
    is resolved here rather than in the client: it is ledger semantics, and
    two callers re-deriving "the last one wins" is how the two of them
    eventually disagree.

    ``opened_at`` comes from the FIRST log entry and ``updated_at`` from the
    last, so a hypothesis that was proposed and then closed reports both.
    """
    path = Path(run_dir) / "debug" / _NOTES_DIR / "hypotheses.json"
    out = []
    for hid, h in sorted(_read_json_object(path).items()):
        if not isinstance(h, dict):
            continue
        log = [e for e in h.get("status_log") or [] if isinstance(e, dict)]
        latest = log[-1] if log else {}
        out.append({
            "id": h.get("id", hid),
            "statement": h.get("statement", ""),
            "prediction": h.get("prediction", ""),
            "falsification_criterion": h.get("falsification_criterion", ""),
            "proposed_by": h.get("proposed_by", ""),
            "status": latest.get("status", "OPEN"),
            "comment": latest.get("comment", ""),
            "evidence": latest.get("evidence"),
            "posterior": latest.get("posterior", h.get("prior")),
            "validator_note": latest.get("validator_note"),
            "triggered_by": latest.get("triggered_by"),
            "opened_at": (log[0].get("ts") if log else h.get("proposed_at")),
            "updated_at": latest.get("ts"),
            "history": len(log),
        })
    return out


def read_milestones(run_dir: Path | str) -> list[dict[str, Any]]:
    """Milestones in id order, each with the note justifying its status.

    The note matters as much as the status: a SKIPPED milestone with a
    reason is a judgment the run made and defended, which reads very
    differently from one that was simply never reached.
    """
    path = Path(run_dir) / "debug" / _NOTES_DIR / "milestones.json"
    out = []
    for mid, m in sorted(_read_json_object(path).items()):
        if not isinstance(m, dict):
            continue
        out.append({
            "id": m.get("id", mid),
            "key": m.get("key", ""),
            "description": m.get("description", ""),
            "status": m.get("status", "OPEN"),
            "note": m.get("note") or "",
            "manual": bool(m.get("manual")),
        })
    return out


# Columns f3dasm's output ledger adds for provenance rather than science.
# Kept apart from real outputs: "which delegation produced this row, when,
# and how long it took" answers a different question from "what did the
# oracle return", and mixing them makes a result table unreadable.
_PROVENANCE_COLS = {
    "_delegation_id", "_source", "_ts", "_wall_ms",
}

# The default store's own data directory. The name is reserved by
# agent_runtime (a namespace may not be called this), and it is how every
# accounting helper tells the canonical store from a namespace store.
_DATA_DIR = "experiment_data"

# Cap on rows returned per store. A campaign can ledger thousands; the
# count is always reported in full so a truncated view never reads as the
# whole ledger.
_MAX_EVAL_ROWS = 400


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Header and rows of a ledger CSV, or ([], []) if unreadable."""
    import csv
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as fh:
            rows = list(csv.reader(fh))
    except OSError:
        return [], []
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _read_one_store(store: Path, namespace: str | None) -> dict[str, Any]:
    """One oracle store's input space and ledgered evaluations."""
    data = store / _DATA_DIR
    space: list[str] = []
    dom = data / "domain.json"
    if dom.exists():
        parsed = _read_json_object(dom)
        space = sorted((parsed.get("input_space") or {}).keys())

    in_head, in_rows = _read_csv_rows(data / "input.csv")
    out_head, out_rows = _read_csv_rows(data / "output.csv")
    _, job_rows = _read_csv_rows(data / "jobs.csv")
    statuses = [r[1] if len(r) > 1 else "" for r in job_rows]

    n = max(len(in_rows), len(out_rows))
    # Newest first: a live run's interesting rows are the ones just added.
    order = list(range(n))[::-1][:_MAX_EVAL_ROWS]
    evals = []
    for i in order:
        inputs, outputs, prov = {}, {}, {}
        if i < len(in_rows):
            for col, val in zip(in_head[1:], in_rows[i][1:], strict=False):
                if val != "":
                    inputs[col] = val
        if i < len(out_rows):
            for col, val in zip(out_head[1:], out_rows[i][1:], strict=False):
                if val == "":
                    continue
                (prov if col in _PROVENANCE_COLS else outputs)[col] = val
        evals.append({
            "index": i,
            "status": statuses[i] if i < len(statuses) else "",
            "inputs": inputs,
            "outputs": outputs,
            "delegation_id": prov.get("_delegation_id") or "",
            "ts": prov.get("_ts") or "",
            "wall_ms": prov.get("_wall_ms") or "",
        })
    return {
        "namespace": namespace,
        "path": str(store),
        "input_space": space,
        "n_evals": n,
        "truncated": n > len(order),
        "evals": evals,
    }


def read_oracle(run_dir: Path | str) -> dict[str, Any]:
    """What the run's oracle is, and every evaluation it has ledgered.

    Namespaces are enumerated from DISK as well as from run_config.json's
    ``oracles`` map. A namespace store is created the moment it is
    registered, and the config is rewritten separately, so trusting the
    config alone can miss one — and an eval count that misses a namespace
    is the bug agent_runtime's own eval accounting had to fix (a run
    reported 100 evals against 200 real ones because it counted the
    canonical store only).
    """
    run_dir = Path(run_dir)
    cfg = _read_json_object(run_dir / "debug" / "run_config.json")
    base = cfg.get("store_dir")
    base_path = Path(base) if base else run_dir / _DATA_DIR

    stores: list[dict[str, Any]] = []
    if (base_path / _DATA_DIR).is_dir():
        stores.append(_read_one_store(base_path, None))

    seen = {s["path"] for s in stores}
    names = set((cfg.get("oracles") or {}).keys())
    if base_path.is_dir():
        for entry in sorted(base_path.iterdir()):
            if entry.is_dir() and entry.name != _DATA_DIR:
                names.add(entry.name)
    for name in sorted(names):
        ns_store = base_path / name
        if str(ns_store) in seen or not (ns_store / _DATA_DIR).is_dir():
            continue
        stores.append(_read_one_store(ns_store, name))

    # "No oracle registered" and "an oracle is registered but was never
    # called" are different facts about a run, and only the second one was
    # true on run 20260906T202140 — where the datagenerator wrote an
    # entrypoint into run_config.json, then did all its work in workspace/
    # and ledgered nothing. Saying "not registered" there is false, and it
    # hides the more interesting finding: the oracle went unused.
    registered = bool(cfg.get("evaluator_entrypoint")
                      or cfg.get("evaluator_lookup")
                      or cfg.get("oracles"))
    return {
        "registered": registered,
        "evaluator_name": cfg.get("evaluator_name") or "",
        "entrypoint": cfg.get("evaluator_entrypoint") or "",
        "eval_budget": cfg.get("eval_budget"),
        "store_dir": str(base_path),
        "stores": stores,
        # Summed across canonical AND namespaces, for the same reason the
        # runtime's own accounting does.
        "total_evals": sum(s["n_evals"] for s in stores),
    }


def read_vitals(run_dir: Path | str) -> dict[str, Any]:
    """The run's REAL cost and wall clock, from telemetry and file times.

    Not from the delegation log. A delegation row is not a complete account
    of a run's spend: the critic's gate row is written with ``cost_usd``
    None and ``tokens_out`` 0, while its actual work is recorded in
    telemetry under internal ids the delegation log never sees
    (``critic-1`` for the review, ``verdict-validator`` for each validation
    pass). Summing ``cost_usd`` across delegations therefore under-reports
    by the critic's entire budget — measured at $0.697 of a real $1.078 on
    run 20260905T162758, a 35% undercount presented as a total.

    ``telemetry/calls*.jsonl`` is append-per-call, so this is correct
    mid-run as well as after close; ``summary.json`` is only written at the
    end and is used as nothing more than a cross-check.

    Wall comes from file mtimes (``run_config.json`` written at start,
    ``run_status.json`` at close), deliberately NOT from parsing the run
    directory name against ``run_status.timestamp`` — the directory name is
    local time and that field is UTC, so mixing them is wrong by the
    machine's UTC offset. mtimes are plain epochs and cannot disagree.
    """
    run_dir = Path(run_dir)
    debug = run_dir / "debug"

    cost = 0.0
    calls = 0
    out_tokens = 0
    by_role: dict[str, dict[str, Any]] = {}
    for path in sorted(debug.glob("telemetry/calls*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = row.get("total_cost_usd") or 0.0
            o = row.get("output_tokens") or 0
            # A hand-patched or backend-changed row can carry a string here;
            # one bad row must not take down the whole status bar.
            if not isinstance(c, (int, float)) or isinstance(c, bool):
                c = 0.0
            if not isinstance(o, int) or isinstance(o, bool):
                o = 0
            cost += c
            out_tokens += o
            calls += 1
            role = row.get("role") or "unknown"
            slot = by_role.setdefault(role, {"calls": 0, "cost_usd": 0.0})
            slot["calls"] += 1
            slot["cost_usd"] += c

    # The EARLIEST mtime among the files written once at run start, not any
    # single one of them. run_config.json alone was wrong: something
    # rewrites it mid-run (oracle registration), and anchoring on it made a
    # run that had been going 1h58m report 67 seconds — the wall clock
    # visibly reset to zero while the operator watched. Taking the minimum
    # is robust to any one of these being rewritten.
    started = ended = None
    # The run now records its own start explicitly, and that is authoritative:
    # it is the same anchor the run itself charges its wall budget against, so
    # the viewer and the critic cannot disagree about how long a run has been
    # going. The mtime heuristic below stays as a fallback for runs recorded
    # before the anchor existed.
    try:
        started = float((debug / "run_started_at").read_text().strip())
    except (OSError, ValueError):
        started = None
    if started is None:
        stamps = []
        for name in ("thread_id", "PROBLEM_STATEMENT_snapshot.md",
                     "run_config.json"):
            path = debug / name
            try:
                stamps.append(path.stat().st_mtime)
            except OSError:
                continue
        if stamps:
            started = min(stamps)
    status = debug / "run_status.json"
    if status.exists():
        ended = status.stat().st_mtime
    elapsed = None
    if started is not None:
        elapsed = (ended if ended is not None else time.time()) - started

    return {
        # started_at lets the client tick the clock itself. Serving only a
        # snapshot of elapsed_s made the wall time freeze between polls —
        # it looked stopped, because for five seconds at a time it was.
        "started_at": started,
        "cost_usd": round(cost, 6),
        "calls": calls,
        "output_tokens": out_tokens,
        "by_role": {r: {"calls": v["calls"],
                        "cost_usd": round(v["cost_usd"], 6)}
                    for r, v in sorted(by_role.items())},
        "elapsed_s": elapsed,
        "closed": ended is not None,
    }


# Text artifacts a read-only view can render. Anything else is listed but
# not offered for reading.
# Run directories are named <YYYYMMDD>T<HHMMSS>.
_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}")

_READABLE_SUFFIXES = {".md", ".tex", ".py", ".json", ".txt", ".csv", ".yaml", ".yml"}


def read_artifacts(
    run_dir: Path | str, study_dir: Path | str,
) -> list[dict[str, Any]]:
    """Files this run's work produced, each tagged with what it can be
    attributed to.

    Two scopes, kept apart on purpose. ``run`` files live under this run's
    own ``debug/delegations/<ID>/`` and are unambiguously its work.
    ``shared`` files live in a study-level workspace beside the run
    directories (``runs/math_workspace/``, and the like) — those persist
    across runs and are NOT attributable to any one of them, which is the
    same trap ``read_notebook`` guards against for pipeline.ipynb. They are
    listed because a symbolic study's real deliverable lives there, but
    they carry the scope and mtime so a reader can see for themselves
    whether this run wrote them.
    """
    run_dir, study_dir = Path(run_dir), Path(study_dir)
    out: list[dict[str, Any]] = []

    def add(path: Path, scope: str, owner: str) -> None:
        try:
            st = path.stat()
        except OSError:
            return
        out.append({
            "path": str(path.relative_to(study_dir)),
            "name": path.name,
            "scope": scope,
            "owner": owner,
            "size": st.st_size,
            "mtime": st.st_mtime,
            "readable": path.suffix.lower() in _READABLE_SUFFIXES,
        })

    deleg_root = run_dir / "debug" / "delegations"
    if deleg_root.is_dir():
        for d in sorted(deleg_root.iterdir()):
            if d.is_dir():
                for f in sorted(d.rglob("*")):
                    if f.is_file():
                        add(f, "run", d.name)

    runs_root = run_dir.parent
    if runs_root.is_dir():
        for entry in sorted(runs_root.iterdir()):
            # Skip run directories; a workspace is any other dir. A run is
            # identified by its name matching the run-id shape as well as by
            # a debug/ dir, because a just-started run has not created
            # debug/ yet and would otherwise have its files listed as
            # "shared" — i.e. not attributable to any run, the opposite of
            # the truth.
            if not entry.is_dir():
                continue
            if (entry / "debug").is_dir() or _RUN_ID_RE.match(entry.name):
                continue
            for f in sorted(entry.rglob("*")):
                if f.is_file():
                    add(f, "shared", entry.name)
    return out


def read_artifact_text(
    study_dir: Path | str, rel_path: str,
) -> str | None:
    """One artifact's text, or ``None`` if it is not a readable file inside
    the study.

    *rel_path* arrives from the client, so it is resolved and checked to be
    inside *study_dir* before anything is opened — a viewer that will read
    an arbitrary path is a file-disclosure hole, however local it is.
    """
    study_dir = Path(study_dir).resolve()
    try:
        target = (study_dir / rel_path).resolve()
    except (OSError, ValueError):
        return None
    if not target.is_relative_to(study_dir):
        return None
    if not target.is_file() or target.suffix.lower() not in _READABLE_SUFFIXES:
        return None
    try:
        return target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _normalize_output(out: dict[str, Any]) -> dict[str, Any] | None:
    """One notebook output reduced to what a read-only view can show."""
    kind = out.get("output_type")
    if kind == "stream":
        text = out.get("text")
        return {"kind": "stream", "name": out.get("name", "stdout"),
                "text": "".join(text) if isinstance(text, list) else (text or "")}
    if kind == "error":
        tb = out.get("traceback") or []
        return {"kind": "error", "ename": out.get("ename", ""),
                "evalue": out.get("evalue", ""), "text": "\n".join(tb)}
    if kind in ("execute_result", "display_data"):
        data = out.get("data") or {}
        for mime in ("image/png", "image/jpeg"):
            if mime in data:
                payload = data[mime]
                if isinstance(payload, list):
                    payload = "".join(payload)
                # An embedded figure is base64 already; oversized ones are
                # dropped rather than pushed through the JSON response,
                # since one plot must not stall the whole view.
                if len(payload) > 4_000_000:
                    return {"kind": "image_too_large", "mime": mime}
                return {"kind": "image", "mime": mime, "data": payload}
        text = data.get("text/plain", "")
        if isinstance(text, list):
            text = "".join(text)
        return {"kind": "text", "text": text}
    return None


def read_notebook(
    study_dir: Path | str, run_id: str,
) -> dict[str, Any] | None:
    """The deliverable notebook belonging to *run_id*, or ``None``.

    ``pipeline.ipynb`` is STUDY-scoped, not run-scoped: it lives at
    ``<study_dir>/pipeline.ipynb`` and a fresh run archives any prior one to
    ``pipeline_<that run's id>.ipynb``
    (``agent_runtime._archive_prior_pipeline_notebook``). So the live file
    belongs to whichever run last wrote it — showing it unconditionally on
    an older run's page would attribute one run's deliverable to another,
    silently and plausibly.

    The live file is therefore only used when its own provenance stamp
    (``metadata.agentic.run``) names this run; otherwise the archive for
    this run id is used, and if neither exists the answer is "no notebook",
    which is itself a real finding — the agent never wrote one.
    """
    study_dir = Path(study_dir)
    path = None
    live = study_dir / "pipeline.ipynb"
    if live.exists():
        try:
            import nbformat
            stamped = (nbformat.read(str(live), as_version=4)
                       .metadata.get("agentic") or {}).get("run")
            if stamped and Path(stamped).name == run_id:
                path = live
        except Exception:  # noqa: BLE001 — unreadable/unstamped: fall through
            pass
    if path is None:
        # Anchored, not a bare prefix glob: "pipeline_{run_id}*" lets run
        # "R1" match "pipeline_R10.ipynb" and show another run's deliverable
        # as its own — the precise misattribution this function exists to
        # prevent. The optional suffix is only the uuid the archiver appends
        # when the same id is archived twice.
        archived = sorted(
            p for p in study_dir.glob(f"pipeline_{run_id}*.ipynb")
            if p.stem == f"pipeline_{run_id}"
            or p.stem.startswith(f"pipeline_{run_id}_")
        )
        if archived:
            path = archived[0]
    if path is None:
        return None

    try:
        import nbformat
        nb = nbformat.read(str(path), as_version=4)
    except Exception as exc:  # noqa: BLE001
        return {"cells": [], "path": str(path),
                "error": f"could not read {path.name}: {exc}"}

    cells = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        if cell.get("cell_type") == "markdown":
            cells.append({"type": "markdown", "source": src})
        elif cell.get("cell_type") == "code":
            outs = [_normalize_output(o) for o in (cell.get("outputs") or [])]
            cells.append({
                "type": "code",
                "source": src,
                "execution_count": cell.get("execution_count"),
                "outputs": [o for o in outs if o is not None],
            })
    return {"cells": cells, "path": str(path),
            "live": path == live, "error": None}


def read_run_status(run_dir: Path | str) -> dict[str, Any] | None:
    """``run_status.json``'s contents, or ``None`` if the run hasn't closed
    yet (normal close and crash both write this file, but only once, at the
    very end — absence means "still running", not an error)."""
    path = Path(run_dir) / "debug" / "run_status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
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
    # *key* reaches here straight from a {key:path} URL segment, so it may
    # contain "/" and "..". Without containment this reads ANY .jsonl on the
    # host: confirmed against a live server, where
    # ../../../../../secret returned 200 with the file's contents — and the
    # viewer is routinely bound to a LAN/ZeroTier address, so that is
    # remotely reachable. Resolve first, then require the result to be
    # inside the run's own transcripts dir, exactly as read_artifact_text
    # already does for its client-supplied path.
    root = transcripts_dir.resolve()
    try:
        path = (transcripts_dir / f"{key}.jsonl").resolve()
    except (OSError, ValueError):
        return []
    if not path.is_relative_to(root):
        return []
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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
    return path.read_text(encoding="utf-8", errors="replace")


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

    Two on-disk layouts exist and a node can use EITHER, so both are
    checked and the results merged:

    * flat ``transcripts/<delegation id>.jsonl`` — a worker's per-delegation
      transcript, keyed by the id of a delegation TO that node;
    * nested ``transcripts/<node name>/<stem>.jsonl`` — used by the entry
      node (``turn_001``) and, discovered the hard way, by the critic
      (``call_001``). The critic has delegation rows (its gate check) whose
      ids do NOT correspond to any flat file, so returning ids alone
      offered a key that resolves to nothing: on run 20260905T162758 the
      critic's transcript sat at ``transcripts/critic/call_001.jsonl``
      while this function reported ``["GATE164710"]``.

    The nested directory is keyed on the node's OWN name rather than the
    literal "strategizer", so a graph whose entry node is called something
    else still finds its turns. *is_entry* is consequently no longer needed
    to pick a layout; it is kept so existing callers keep working.

    Keys are never computed from a formula. A prior version guessed the
    entry node's turn as
    ``f"strategizer/turn_{len(delegations_touching_it):03d}"``, which bore
    no relation to real turn numbering and would silently open the wrong
    turn.
    """
    root = Path(run_dir) / "debug" / "transcripts"
    keys: list[str] = []

    node_dir = root / node_name
    if node_dir.is_dir():
        keys += sorted(f"{node_name}/{p.stem}" for p in node_dir.glob("*.jsonl"))

    keys += [
        d["id"] for d in read_delegations(run_dir)
        if d.get("to_node") == node_name
        and (root / f"{d['id']}.jsonl").exists()
    ]
    return keys


def graph_spec_json(graph, study_dir=None) -> dict[str, Any]:
    """Node/edge/role/tool/layer JSON for the live network diagram, plus the
    per-node metadata the hover card needs (model, system prompt) and a
    global tool-name -> docstring map for the tool-badge tooltips.

    Reuses ``run_diagram._bfs_layers()`` directly rather than re-deriving
    layer logic — the viewer only needs a layer index per node (for a simple
    CSS-grid row placement), not the static SVG's pixel-precise card layout.

    ``model`` per node: an ``Agent`` instance's own ``.model`` if the study's
    ``run.py`` set one explicitly, else the study's ``config.yaml`` top-level
    ``model:`` (the actual, common case — a study normally sets the model
    once for the whole run, not per-agent), else "(backend default)".
    """
    from ..runtime.run_diagram import _bfs_layers

    layers = _bfs_layers(graph)
    config = _load_study_config(study_dir)
    run_model = config.get("model")
    same_row = sum(
        1 for e in graph.edges
        if e.source in layers and e.target in layers
        and layers[e.source] == layers[e.target]
    )
    pos, canvas_w, canvas_h = _layout_nodes(layers, same_row)
    identity = _identity_indices(layers, graph.entry)
    nodes = []
    # Precedence, weakest first: the hand-written map (which is the only
    # source for native backend tools), then routing.py's real
    # docstrings, then any live closure's own __doc__.
    tool_docs: dict[str, str] = dict(_KNOWN_TOOL_DOCS)
    tool_docs.update(_routing_tool_docs())
    for name, agent in graph.nodes.items():
        tools, docs = _node_tools_and_docs(name, agent, graph, study_dir)
        tool_docs.update(docs)
        x, y = pos[name]
        nodes.append({
            "name": name,
            "role": agent.role,
            "description": agent.description or "",
            "is_entry": name == graph.entry,
            "layer": layers[name],
            "tools": tools,
            "model": _humanize_model(agent.model or run_model),
            "system_prompt": agent.system_prompt or "",
            "x": x,
            "y": y,
            "identity": identity[name],
        })
    edges = [{"source": e.source, "target": e.target} for e in graph.edges]
    return {
        "nodes": nodes, "edges": edges, "entry": graph.entry,
        "tool_docs": tool_docs,
        # Run-level, not per-node: config.yaml sets one backend for the
        # whole run, so it is emitted once rather than repeated on every
        # node as if it could differ between them.
        "backend": config.get("backend") or "",
        "node_w": _NODE_W, "node_h": _NODE_H,
        "canvas_w": canvas_w, "canvas_h": canvas_h,
    }


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
            # Executing run.py otherwise leaves a __pycache__ inside the
            # study. Small, but the viewer's contract is that it reads a
            # study and changes nothing in it.
            import sys as _sys
            _prev = _sys.dont_write_bytecode
            _sys.dont_write_bytecode = True
            try:
                spec.loader.exec_module(module)
            finally:
                _sys.dont_write_bytecode = _prev
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
    path: Path | str, poll_interval: float = 0.5, should_stop=None,
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
    # *should_stop* lets a caller end the tail. Without it this generator
    # never returns, so the thread running it outlives the SSE connection it
    # was started for: refreshing the dashboard twenty times left forty
    # threads polling the disk for the life of the process, each feeding a
    # queue nobody reads.
    if should_stop is None:
        def should_stop() -> bool:
            return False

    if path.exists():
        content = path.read_bytes()
        offset = content.rfind(b"\n") + 1  # 0 if no newline is present at all
    else:
        offset = 0
        while not path.exists():
            if should_stop():
                return
            time.sleep(poll_interval)
    buffer = ""
    while not should_stop():
        try:
            size = path.stat().st_size
        except OSError:
            # The file was removed under us — run.py wipes runs/ before a new
            # run, and a viewer left open across a relaunch hits exactly
            # that. Ending the tail is right; dying with an unhandled
            # FileNotFoundError in a daemon thread silently froze the stream.
            return
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
