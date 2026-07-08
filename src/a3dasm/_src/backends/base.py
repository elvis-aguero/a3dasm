"""Agent, Edge, and Graph primitives for agentic-f3dasm backends."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Debug / transcript capture — master switch F3DASM_DEBUG, OFF by default.
# When on, agents stream their full reasoning + tool-calls + tool-results to
# per-delegation / per-strategizer-turn JSONL transcripts so we can see what
# the workers are actually thinking. Capture is thread-local: each worker
# delegation runs in its own thread and sets its own sink.
# ---------------------------------------------------------------------------

_DEBUG_TRUE = {"1", "true", "yes", "on"}
_transcript_tls = threading.local()


def debug_enabled() -> bool:
    """Master debug switch. Off unless the `debug` knob is truthy.

    Source of truth is config.yaml's runtime block; F3DASM_DEBUG overrides it."""
    from ..settings import get_bool
    return get_bool("debug", False)


def set_transcript_sink(path: str | None) -> None:
    """Point this thread's transcript at *path* (None clears it)."""
    _transcript_tls.path = str(path) if path else None


def get_transcript_sink() -> str | None:
    return getattr(_transcript_tls, "path", None)


def set_delegation_id(delegation_id: str | None) -> None:
    """Bind this thread's delegation id (None clears it).

    Thread-local so concurrent delegations don't clobber each other. The Claude
    backend injects it as a per-session ``F3DASM_DELEGATION_ID`` env var so
    ``get_evaluator()`` resolves without the worker having to ``cd`` into its
    ``D###`` directory first (audit Finding 2).
    """
    _transcript_tls.delegation_id = delegation_id


def get_delegation_id() -> str | None:
    return getattr(_transcript_tls, "delegation_id", None)


def set_run_config_path(path: str | None) -> None:
    """Bind this thread's run_config.json path (None clears it).

    Thread-local, mirroring ``set_delegation_id``. The Claude backend injects it
    as a per-session ``F3DASM_RUN_CONFIG`` env var so ``get_evaluator()`` finds
    the config by an explicit path — the SDK spawns the worker with
    ``cwd=study_dir``, but run_config.json lives *down* at
    ``runs/<id>/debug/``, so the old walk-up-from-cwd never reached it and the
    worker had to ``cd`` into ``debug/`` first.
    """
    _transcript_tls.run_config_path = str(path) if path else None


def get_run_config_path() -> str | None:
    return getattr(_transcript_tls, "run_config_path", None)


def set_namespace(namespace: str | None) -> None:
    """Bind this thread's design namespace (None clears it).

    Thread-local, mirroring ``set_delegation_id``. When a delegation is scoped to
    a design namespace, the Claude backend injects it as a per-session
    ``F3DASM_NAMESPACE`` env var so the worker's plain ``get_evaluator()`` resolves
    that namespace's oracle + ledger (Axis 3a) — the agent's call site stays
    ``get_evaluator()`` with no argument. None (the default) → no env var → the
    single-study canonical oracle, exactly as before.
    """
    _transcript_tls.namespace = namespace


def get_namespace() -> str | None:
    return getattr(_transcript_tls, "namespace", None)


def set_oracle_registered(registered: bool) -> None:
    """Bind whether a canonical oracle is registered for this thread's worker.

    Thread-local, mirroring ``set_delegation_id``. The raw-oracle nudge keys on
    this: until an oracle is registered, ``get_evaluator()`` resolves nothing,
    so "use get_evaluator() instead" is non-advice — the nudge stays silent.
    This exempts the datagenerator (which works BEFORE registration, by state
    not by role name) and re-arms automatically once the oracle is live.
    """
    _transcript_tls.oracle_registered = bool(registered)


def oracle_registered() -> bool:
    return getattr(_transcript_tls, "oracle_registered", False)


def append_transcript(record: dict) -> None:
    """Best-effort append one JSON record to the active transcript.

    No-op unless F3DASM_DEBUG is on and a sink is set on this thread.
    Never raises into the agent loop.
    """
    if not debug_enabled():
        return
    path = get_transcript_sink()
    if not path:
        return
    try:
        stamped = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **record,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(stamped, default=str) + "\n")
    except Exception:  # noqa: BLE001 — debug capture must never break a run
        pass


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------


class Agent:
    """Base class for all agentic nodes in a Graph.

    Subclasses override class-level attributes (``system_prompt``,
    ``tools``, etc.) to configure behaviour.  Behavioural differences
    belong in class attributes, not constructor arguments.

    **Tool system — three categories:**

    ``tools: frozenset[str]`` declares tools from two categories:

    1. **Native backend tools** — names from :data:`NATIVE_TOOL_NAMES`
       (``"Bash"``, ``"Read"``, ``"Write"``, etc.).  The runtime passes
       these to the backend session's native tool executor.

    2. **Protocol closure tools** — names from
       :data:`PROTOCOL_CLOSURE_NAMES` (``"Done"``, ``"WriteNote"``,
       ``"ReadNote"``).  The runtime builds Python callables for these and
       passes them as ``closure_tools`` to the session factory.

    3. **Topology-injected tools** — ``"Delegate"``, ``"Parallel"``,
       ``"Debate"``, ``"Retry"`` (outgoing edges), ``"Ask"`` (entry node),
       and ``"FollowUp"`` (incoming edges).  **Never declare these in**
       ``Agent.tools``.  The runtime injects them automatically from the
       graph topology; any declaration here is ignored.

    4. **External MCP server tools** — names declared in
       ``extra_allowed_tools`` (e.g.
       ``"arxiv_search_papers"``).  The runtime passes these to the
       backend together with ``mcp_servers``, a dict of
       ``{server_name: McpStdioServerConfig}`` that declares which external
       MCP servers to start.

    Default is ``frozenset()`` — no tools (opt-in, conservative).

    Parameters
    ----------
    model : str or None
        Model identifier.  ``None`` delegates to the backend default.
    """

    system_prompt: str = ""
    tools: frozenset[str] = frozenset()
    reset_on_checkpoint: bool = True
    description: str = ""
    # Neutral default: a subclass that forgets to declare its role must NOT
    # silently inherit "implementer" and pick up implementer-only behavior
    # (the milestone gate, the eval-parallelism nudge). Every shipped agent
    # declares its own role; this default only guards future ones.
    role: str = "worker"
    backend: str | None = None
    mcp_servers: dict = {}
    extra_allowed_tools: frozenset[str] = frozenset()
    inject_problem_statement: bool = False
    max_history_pairs: int = 5
    report_sections: tuple[str, ...] = (
        "### Actions taken",
        "### Files touched",
        "### Conclusions",
        "### Numbers",
    )

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def forward(self) -> None:
        """ADAS hook — override for inspectable Python orchestration."""

    def build_closure_tools(
        self,
        study_dir: Path,
        delegation_id: str | None = None,
        lit_reviewer_notes_dir: Path | None = None,
    ) -> dict:
        """Return runtime closure tools for this agent. Override in subclasses.

        Called by the runtime when constructing the worker adapter so agents can
        inject Python callables (e.g. corpus management tools) without declaring
        them in Agent.tools.
        """
        return {}


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """A directed delegation edge between two named agents.

    Parameters
    ----------
    source : str
        Name of the agent that is allowed to call ``Delegate``.
    target : str
        Name of the agent that receives the delegated task.
    preamble : str
        Text prepended to the task message when this edge is traversed.
    """

    source: str
    target: str
    preamble: str = ""


@dataclass
class Graph:
    """Agent graph passed to :class:`~agent_runtime.AgenticRun`.

    Declares which agents exist, which directed delegation edges connect
    them, and which agent starts the run.  Loops are permitted.

    Parameters
    ----------
    nodes : dict[str, Agent]
        Maps unique agent names to :class:`Agent` instances.
    edges : sequence of Edge
        Directed delegation edges.  An agent with no outgoing edges
        receives no ``Delegate`` tool.
    entry : str
        Name of the agent that receives the initial briefing.

    Raises
    ------
    ValueError
        If any edge endpoint names an undeclared node, or if *entry* is
        not declared.
    """

    nodes: dict  # dict[str, Agent]
    edges: tuple = ()
    entry: str = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.edges = tuple(self.edges)
        if self.entry is None:
            raise ValueError("Graph.entry is required and must not be None.")
        bad = [k for k, v in self.nodes.items() if not isinstance(v, Agent)]
        if bad:
            raise TypeError(
                f"Graph nodes values must be Agent instances; "
                f"got invalid values for keys: {sorted(bad)}"
            )
        names = set(self.nodes)
        for e in self.edges:
            if e.source not in names or e.target not in names:
                raise ValueError(
                    f"Edge {e!r} references undeclared node. "
                    f"Declared names: {sorted(names)}"
                )
        if self.entry not in names:
            raise ValueError(
                f"entry={self.entry!r} not in nodes (entry node undeclared). "
                f"Declared names: {sorted(names)}"
            )
        missing_desc = [n for n, a in self.nodes.items() if not a.description]
        if missing_desc:
            raise ValueError(
                f"All agents must define a non-empty description. "
                f"Missing in: {sorted(missing_desc)}"
            )

    def outgoing(self, name: str) -> list[str]:
        """Return target names for all edges out of *name*."""
        return [e.target for e in self.edges if e.source == name]

    def incoming(self, name: str) -> list[str]:
        """Return source names for all edges into *name*."""
        return [e.source for e in self.edges if e.target == name]

    def edge(self, source: str, target: str) -> Edge | None:
        """Return the Edge from *source* to *target*, or None if absent."""
        for e in self.edges:
            if e.source == source and e.target == target:
                return e
        return None

    # Curated semantic colours (fill, stroke) by agent class name;
    # unknown classes cycle a fallback palette so any graph stays legible.
    _MERMAID_COLOURS = {
        "StrategizerAgent": ("#1d4ed8", "#1e3a8a"),
        "LiteratureReviewAgent": ("#6d28d9", "#4c1d95"),
        "DataGeneratorAgent": ("#b45309", "#7c2d12"),
        "F3dasmImplementerAgent": ("#15803d", "#14532d"),
        "AdversarialCritiqueAgent": ("#b91c1c", "#7f1d1d"),
        "DebuggerAgent": ("#475569", "#1e293b"),
    }
    _MERMAID_FALLBACK = [
        ("#0f766e", "#134e4a"), ("#a16207", "#713f12"),
        ("#be185d", "#831843"), ("#4338ca", "#312e81"),
    ]

    def to_mermaid(self) -> str:
        """Return a styled Mermaid flowchart for this graph.

        Generated from the live spec: node labels carry each agent's
        class, role, and a short description; nodes are coloured by agent
        class; edges from the entry node render as solid delegation
        arrows and edges from worker nodes as dotted consultation arrows.
        Paste at https://mermaid.live or any Mermaid-aware renderer
        (GitHub markdown, Jupyter, VS Code).
        """
        def _clean(text: str, n: int = 46) -> str:
            t = " ".join(str(text).split()).replace('"', "'")
            return (t[: n - 1] + "…") if len(t) > n else t

        lines = ["flowchart TD"]
        colours: dict = {}
        members: dict = {}
        fb = 0
        for name in self.nodes:
            agent = self.nodes[name]
            cls = type(agent).__name__
            if cls not in colours:
                if cls in self._MERMAID_COLOURS:
                    colours[cls] = self._MERMAID_COLOURS[cls]
                else:
                    colours[cls] = self._MERMAID_FALLBACK[
                        fb % len(self._MERMAID_FALLBACK)]
                    fb += 1
            members.setdefault(cls, []).append(name)
            desc = _clean(
                getattr(agent, "description", "") or agent.role)
            label = f"<b>{name}</b><br/>{cls}<br/><i>{desc}</i>"
            if name == self.entry:
                lines.append(f'    {name}(["{label}"])')
            else:
                lines.append(f'    {name}["{label}"]')
        for e in self.edges:
            arrow = "-->" if e.source == self.entry else "-.->"
            if e.preamble:
                snippet = _clean(e.preamble, 35)
                lines.append(
                    f'    {e.source} {arrow}|"{snippet}"| {e.target}')
            else:
                lines.append(f'    {e.source} {arrow} {e.target}')
        for cls, (fill, stroke) in colours.items():
            lines.append(
                f"    classDef {cls} fill:{fill},stroke:{stroke},"
                "color:#fff,stroke-width:1px;")
        for cls, names in members.items():
            lines.append(f"    class {','.join(names)} {cls}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        out_map: dict[str, list] = {}
        for e in self.edges:
            out_map.setdefault(e.source, []).append(e)
        lines = [f"Graph(entry={self.entry!r})"]
        for name in self.nodes:
            tag = " [entry]" if name == self.entry else ""
            edges = out_map.get(name, [])
            if edges:
                for e in edges:
                    preamble = f"  # {e.preamble!r}" if e.preamble else ""
                    lines.append(f"  {name}{tag}  ──▶  {e.target}{preamble}")
                    tag = ""  # only label first edge row
            else:
                lines.append(f"  {name}{tag}  (leaf)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Transient-error retry (shared by all backend adapters)
# ---------------------------------------------------------------------------

# Substrings (case-insensitive) marking a retryable transient failure. The
# Claude path runs through claude-agent-sdk (not the anthropic SDK) and the
# Ollama path through langchain/httpx, so there is no shared exception class to
# catch — we classify heuristically on the message plus a few stdlib types.
_TRANSIENT_SUBSTRINGS = (
    "overloaded", "rate limit", "ratelimit", "429", "502", "503",
    "timeout", "timed out", "temporarily unavailable", "service unavailable",
    "connection error", "connection reset", "connection aborted",
    "remote disconnected", "econnreset",
)
_TRANSIENT_TYPES = (TimeoutError, ConnectionError)


def is_transient_error(exc: BaseException) -> bool:
    """True if *exc* looks like a retryable transient API/network failure.

    Conservative: anything not recognised as transient (auth, 400, tool/logic
    errors) returns False so we never silently retry a real bug.
    """
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(s in text for s in _TRANSIENT_SUBSTRINGS)


def retry_on_transient(
    fn,
    *,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float = 60.0,
):
    """Call ``fn()`` retrying transient failures with backoff + jitter.

    Non-transient exceptions propagate immediately. Defaults come from the
    ``llm_retry_max`` (5) and ``llm_retry_base`` (2.0s) knobs (config.yaml
    runtime block; F3DASM_LLM_RETRY_MAX / F3DASM_LLM_RETRY_BASE override).
    """
    from ..settings import get_float, get_int
    if max_attempts is None:
        max_attempts = get_int("llm_retry_max", 5)
    if base_delay is None:
        base_delay = get_float("llm_retry_base", 2.0)
    attempt = 0
    while True:
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised unless transient
            attempt += 1
            if attempt >= max_attempts or not is_transient_error(exc):
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay += random.uniform(0, delay * 0.5)
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Raw-oracle-access nudge (non-blocking, just-in-time)
# ---------------------------------------------------------------------------
# When a worker reaches the ground-truth oracle directly (e.g. `from evaluator
# import evaluate`, or loading evaluator.dylib) instead of via get_evaluator(),
# its evaluations are NOT stamped into the canonical ExperimentData ledger — so
# any headline they back cannot be reproduced by pipeline.ipynb. This is a
# best-effort regex NUDGE, not a sandbox: it exists to save a worker from
# burning a whole phase off-ledger, not to stop a determined bypass (the
# reproducibility gate in the critic is the real backstop). Deliberately does
# NOT fire on get_evaluator(), the correct path (which also contains the
# substring "evaluator").

ORACLE_NUDGE_CAP = 2

_RAW_ORACLE_PATTERNS = (
    re.compile(r"\bfrom\s+evaluator\s+import\b"),
    re.compile(r"\bimport\s+evaluator\b"),
    re.compile(r"\bevaluator\.(?:dylib|so)\b"),
)

_ORACLE_NUDGE_MESSAGE = (
    "[ORACLE ACCESS] This reaches the ground-truth evaluator directly — "
    "off-ledger, so it cannot anchor the headline and the critic will reject "
    "the run. Re-run true-oracle calls through get_evaluator() (the oracle-door "
    "block in your system prompt; full contract: handbook "
    "evaluate-through-get-evaluator). Surrogates/optimisers you build yourself "
    "stay off-ledger — that is fine."
)


def detect_raw_oracle_access(tool_name: str, tool_input: dict) -> str | None:
    """Return a nudge string if a Bash/Write call appears to reach the
    ground-truth oracle directly (bypassing get_evaluator), else None.

    Pure and best-effort. Only inspects Bash commands and Write content.
    """
    if tool_name not in ("Bash", "Write"):
        return None
    if not isinstance(tool_input, dict):
        return None
    parts = [
        v for k, v in tool_input.items()
        if k in ("command", "content", "file_text", "new_string")
        and isinstance(v, str)
    ]
    if not parts:
        return None
    text = "\n".join(parts)
    for pat in _RAW_ORACLE_PATTERNS:
        if pat.search(text):
            return _ORACLE_NUDGE_MESSAGE
    return None


class OracleNudgeBudget:
    """Per-delegation, non-blocking cap for the raw-oracle nudge.

    A worker run (one backend invoke) is one delegation; call ``reset`` at the
    start of each invoke and ``check`` per tool call. Emits the nudge at most
    ``cap`` times, then stays silent so it never becomes nagging.
    """

    def __init__(self, cap: int = ORACLE_NUDGE_CAP,
                 enabled: bool = True) -> None:
        self.cap = cap
        # When False, the nudge never fires: there is no registered oracle to
        # bypass, so reaching the raw source directly is the only option (this
        # is the datagenerator's situation during pre-registration validation).
        self.enabled = enabled
        self.used = 0
        # Firings since the last reset, so the runtime can LOG them as direct
        # evidence the nudge acted (not just infer it). Each: {"tool", "snip"}.
        self.events: list[dict] = []

    def reset(self) -> None:
        self.used = 0
        self.events = []

    def check(self, tool_name: str, tool_input: dict) -> str | None:
        if not self.enabled or self.used >= self.cap:
            return None
        msg = detect_raw_oracle_access(tool_name, tool_input)
        if msg is None:
            return None
        self.used += 1
        _snip = ""
        if isinstance(tool_input, dict):
            for _k in ("command", "content", "file_text", "new_string"):
                if isinstance(tool_input.get(_k), str):
                    _snip = tool_input[_k][:120]
                    break
        self.events.append({"tool": tool_name, "snip": _snip})
        return msg

