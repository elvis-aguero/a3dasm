"""Shared base for OpenAI-compatible chat backends (Ollama, OpenRouter, vLLM).

All three speak the OpenAI chat-completions API, so they share one
implementation: ChatOpenAI pointed at the backend's ``base_url`` plus
``create_react_agent`` for the tool-execution loop. A concrete backend is a
thin subclass that only declares its endpoint default + auth (see the class
attributes ``DEFAULT_BASE_URL`` / ``BASE_URL_ENV`` / ``API_KEY`` /
``API_KEY_ENV``) — the invoke/tool/usage machinery is inherited verbatim.

The adapter exposes the same public surface as ClaudeAdapter (a mutable
``closure_tools`` dict, ``invoke()``, ``last_usage``, ``copy()``,
``select_native_tools()``, …) — enforced by tests/agentic/test_backend_parity.py.
"""

from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any

__all__ = ["OpenAICompatibleAdapter"]

# Tool names that are injected as Python CLOSURES (not native CLI tools); the
# OpenAI-compatible backends pass everything ELSE through as a native tool.
_CLOSURE_TOOL_NAMES = frozenset(
    {"Done", "FollowUp", "WriteNote", "ReadNote", "ReportEvals"}
)


def _to_lc_messages(messages: list[dict]) -> list:
    from langchain_core.messages import AIMessage, HumanMessage

    result = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        if role in ("human", "user"):
            result.append(HumanMessage(content=content))
        elif role in ("ai", "assistant"):
            result.append(AIMessage(content=content))
    return result


def _make_edit_tool(cwd: Path | None) -> Any:
    from langchain_core.tools import StructuredTool

    workspace = (cwd or Path(".")).resolve()

    def edit_file(path: str, old_str: str, new_str: str) -> str:
        """Replace old_str with new_str in file at path (first occurrence).
        Restricted to the workspace directory."""
        try:
            candidate = (workspace / path).resolve()
            candidate.relative_to(workspace)
        except ValueError:
            return f"ERROR: edit rejected — {path!r} resolves outside workspace."
        except Exception as exc:
            return f"ERROR: invalid path {path!r}: {exc}"
        if not candidate.exists():
            return f"ERROR: {path} not found"
        text = candidate.read_text(encoding="utf-8")
        if old_str not in text:
            return f"ERROR: string not found in {path}"
        candidate.write_text(text.replace(old_str, new_str, 1), encoding="utf-8")
        return f"Edited {candidate}"

    return StructuredTool.from_function(edit_file, name="Edit")


def _make_grep_tool() -> Any:
    from langchain_core.tools import StructuredTool

    def grep_files(pattern: str, path: str = ".") -> str:
        """Search for pattern in files under path; return matching lines."""
        import subprocess

        try:
            result = subprocess.run(
                ["grep", "-r", "-n", pattern, path],
                capture_output=True, text=True, timeout=30,
            )
            return result.stdout or "(no matches)"
        except Exception as exc:
            return f"ERROR: {exc}"

    return StructuredTool.from_function(grep_files, name="Grep")


def _make_glob_tool(cwd: Path | None) -> Any:
    from langchain_core.tools import StructuredTool

    base = cwd or Path(".")

    def glob_files(pattern: str) -> str:
        """Find files matching a glob pattern relative to the workspace."""
        matches = sorted(str(p) for p in base.glob(pattern))
        return "\n".join(matches) if matches else "(no matches)"

    return StructuredTool.from_function(glob_files, name="Glob")


# --------------------------------------------------------------------------
# Bash + BashOutput + KillShell: one SDK-compatible surface for the non-Claude
# backends (Claude gets these from the real SDK). A command that exceeds its
# timeout is BACKGROUNDED (not killed) and returned with a bash_id, matching
# the SDK's auto-background; BashOutput polls it, KillShell stops it. The one
# deliberate deviation from the SDK is that the background is made VISIBLE
# (interrupted + bash_id in the result) so the agent never wakes up thinking a
# still-running job finished.
# --------------------------------------------------------------------------

_BASH_TIMEOUT_DEFAULT_MS = 300000    # 5 min (SDK default is 120000; ours is
_BASH_TIMEOUT_MAX_MS = 600000        # longer for hours-scale compute). Cap = SDK max.
_BASH_INLINE_CAP = 30000             # chars kept inline before spilling to a file


class _BashSession:
    """Per-adapter registry of shells backing Bash/BashOutput/KillShell.

    Background shells key on a ``bash_id``. Shared by the three tool factories
    (built together in ``_native_tool_map``) so a worker can launch with
    ``run_in_background`` (or hit the timeout) and then poll ``BashOutput`` /
    ``KillShell`` within the same ReAct turn.
    """

    def __init__(self, cwd: Path | None) -> None:
        self.cwd = str(cwd) if cwd else None
        self._bg: dict[str, dict] = {}   # bash_id -> {proc, spool, handle, cursor}
        self._n = 0
        self._lock = threading.Lock()

    def add_background(self, proc, spool: str, handle) -> str:
        with self._lock:
            self._n += 1
            bid = f"bash_{self._n}"
            self._bg[bid] = {"proc": proc, "spool": spool,
                             "handle": handle, "cursor": 0}
        return bid

    def get(self, bid: str) -> dict | None:
        with self._lock:
            return self._bg.get(bid)

    def drop(self, bid: str) -> None:
        with self._lock:
            self._bg.pop(bid, None)

    def register_governor(self, pid: int) -> None:
        """Best-effort: add a backgrounded pid to the run's governor_pids.jsonl
        so the memory watcher tracks/kills it per-delegation. If the run
        context is not resolvable from the env, the process still stays in the
        run's process group (no setsid), so the whole-run group-kill remains
        the backstop; we simply skip fine-grained tracking."""
        try:
            import json as _json
            from datetime import datetime, timezone
            store = os.environ.get("F3DASM_CANONICAL_STORE")
            did = os.environ.get("F3DASM_DELEGATION_ID")
            if not store or not did:
                return
            reg = Path(store).parent / "debug" / "governor_pids.jsonl"
            if not reg.parent.exists():
                return
            from ..resource_backend import get_resource_backend
            be = get_resource_backend()
            rec = {
                "delegation_id": did, "pid": pid,
                "start_time": be.proc_start_time(pid),
                "ts": datetime.now(tz=timezone.utc).isoformat(
                    timespec="seconds"),
            }
            with reg.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001
            pass


def _read_spill(spool: str, start: int) -> tuple[str, int]:
    """Read spool[start:] as text; return (text, new_end_offset)."""
    try:
        with open(spool, "rb") as f:
            f.seek(start)
            data = f.read()
        return data.decode("utf-8", errors="replace"), start + len(data)
    except OSError:
        return "", start


def _render(out: str, spool: str, *, exit_code=None, interrupted=False,
            bash_id=None, waited_s=None, nudge_msg=None) -> str:
    """SDK-shaped rendering as text (this backend delivers tool results as
    strings). Plain output on clean foreground success — for minimal
    disruption and backwards-compat — with a labeled trailer only when there
    is something the agent must know (nonzero exit, timeout/background)."""
    body = out if out else "(no output)"
    # Spill large output to the file and keep an inline tail.
    if len(body) > _BASH_INLINE_CAP:
        body = (f"[output truncated to last {_BASH_INLINE_CAP} chars; full "
                f"output at {spool}]\n" + body[-_BASH_INLINE_CAP:])
    parts = [body]
    if interrupted and bash_id is not None:
        parts.append(
            f"\n[interrupted: still running in the background after "
            f"~{waited_s}s — NOT killed]\nbash_id: {bash_id}  "
            f"(BashOutput('{bash_id}') for more output; "
            f"KillShell('{bash_id}') to stop it)")
    elif bash_id is not None:               # explicit run_in_background
        parts.append(
            f"\n[running in the background]\nbash_id: {bash_id}  "
            f"(BashOutput('{bash_id}') to poll; KillShell('{bash_id}') to stop)")
    elif exit_code not in (None, 0):
        parts.append(f"\n[exit {exit_code}]")
    if nudge_msg:
        parts.append(f"\n\n{nudge_msg}")
    return "".join(parts)


def _make_bash_tool(cwd: Path | None, nudge: Any = None,
                    session: _BashSession | None = None) -> Any:
    import subprocess
    import tempfile

    from langchain_core.tools import StructuredTool

    sess = session if session is not None else _BashSession(cwd)

    def bash(command: str, timeout: int = _BASH_TIMEOUT_DEFAULT_MS,
             run_in_background: bool = False, description: str | None = None,
             dangerouslyDisableSandbox: bool = False) -> str:
        """Run a shell command. Foreground by default; a command that exceeds
        `timeout` (ms) is BACKGROUNDED — not killed — and returned with a
        `bash_id` you poll via BashOutput / stop via KillShell. Set
        run_in_background=true to background immediately (for a long solve).
        stdout and stderr are merged. `description` is advisory;
        `dangerouslyDisableSandbox` is accepted for SDK compatibility and has
        no effect here (Bash is not sandbox-enforced on this backend)."""
        try:
            timeout_ms = int(timeout)
        except (TypeError, ValueError):
            timeout_ms = _BASH_TIMEOUT_DEFAULT_MS
        timeout_s = max(1.0, min(timeout_ms, _BASH_TIMEOUT_MAX_MS) / 1000.0)
        if isinstance(run_in_background, str):
            run_in_background = run_in_background.strip().lower() in (
                "true", "1", "yes")

        spool = tempfile.NamedTemporaryFile(
            prefix="f3dasm_bash_", suffix=".log", delete=False)
        spool_path = spool.name
        spool.close()
        handle = open(spool_path, "wb")
        # No start_new_session: the child MUST stay in the run's process group
        # so the watchdog group-kill and governor tree-walk can reach it.
        proc = subprocess.Popen(
            command, shell=True, cwd=sess.cwd,
            stdout=handle, stderr=subprocess.STDOUT)

        def _nudge_msg():
            return nudge("Bash", {"command": command}) if nudge else None

        if run_in_background:
            sess.register_governor(proc.pid)
            bid = sess.add_background(proc, spool_path, handle)
            out, _ = _read_spill(spool_path, 0)
            return _render(out, spool_path, bash_id=bid,
                           nudge_msg=_nudge_msg())
        try:
            proc.wait(timeout=timeout_s)
            handle.close()
            out, _ = _read_spill(spool_path, 0)
            os.unlink(spool_path) if len(out) <= _BASH_INLINE_CAP else None
            return _render(out, spool_path, exit_code=proc.returncode,
                           nudge_msg=_nudge_msg())
        except subprocess.TimeoutExpired:
            # Auto-background on timeout — matches the SDK, but made VISIBLE.
            sess.register_governor(proc.pid)
            bid = sess.add_background(proc, spool_path, handle)
            out, cursor = _read_spill(spool_path, 0)
            sess._bg[bid]["cursor"] = cursor
            return _render(out, spool_path, interrupted=True, bash_id=bid,
                           waited_s=int(timeout_s), nudge_msg=_nudge_msg())

    return StructuredTool.from_function(bash, name="Bash")


def _make_bashoutput_tool(session: _BashSession) -> Any:
    from langchain_core.tools import StructuredTool

    def bash_output(bash_id: str) -> str:
        """Return output produced by a background shell (started via Bash with
        run_in_background, or auto-backgrounded on timeout) SINCE THE LAST
        call. Reports whether it is still running or has exited (with code)."""
        entry = session.get(bash_id)
        if entry is None:
            return (f"ERROR: no background shell {bash_id!r}. Use the bash_id "
                    "returned by Bash.")
        new_out, cursor = _read_spill(entry["spool"], entry["cursor"])
        entry["cursor"] = cursor
        rc = entry["proc"].poll()
        if rc is None:
            status = "[still running]"
        else:
            try:
                entry["handle"].close()
            except OSError:
                pass
            session.drop(bash_id)
            status = f"[exited {rc}]"
        return f"{status}\n{new_out if new_out else '(no new output)'}"

    return StructuredTool.from_function(bash_output, name="BashOutput")


def _make_killshell_tool(session: _BashSession) -> Any:
    from langchain_core.tools import StructuredTool

    def kill_shell(bash_id: str) -> str:
        """Terminate a background shell and its process tree (SIGTERM then
        SIGKILL). The only per-delegation teardown for a long job."""
        entry = session.get(bash_id)
        if entry is None:
            return f"ERROR: no background shell {bash_id!r}."
        pid = entry["proc"].pid
        try:
            from ..resource_backend import get_resource_backend
            get_resource_backend().kill([pid])   # tree kill (recursive)
        except Exception:  # noqa: BLE001
            try:
                entry["proc"].kill()
            except Exception:  # noqa: BLE001
                pass
        try:
            entry["handle"].close()
        except OSError:
            pass
        session.drop(bash_id)
        return f"[killed {bash_id} (pid {pid}) and its process tree]"

    return StructuredTool.from_function(kill_shell, name="KillShell")


def _make_read_tool(cwd: Path | None) -> Any:
    from langchain_core.tools import StructuredTool

    def read_file(path: str) -> str:
        """Read a file and return its contents."""
        p = (
            Path(path) if Path(path).is_absolute()
            else (cwd or Path(".")) / path
        )
        if not p.exists():
            return f"ERROR: {p} not found"
        return p.read_text(encoding="utf-8")

    return StructuredTool.from_function(read_file, name="Read")


def _make_write_tool(cwd: Path | None, nudge: Any = None) -> Any:
    from langchain_core.tools import StructuredTool

    def write_file(path: str, content: str) -> str:
        """Write content to a file, creating it if it doesn't exist."""
        p = (
            Path(path) if Path(path).is_absolute()
            else (cwd or Path(".")) / path
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        out = f"Written: {p}"
        if nudge is not None:
            msg = nudge("Write", {"file_path": path, "content": content})
            if msg:
                out = f"{out}\n\n{msg}"
        return out

    return StructuredTool.from_function(write_file, name="Write")


def _native_tool_map(cwd: Path | None, nudge: Any = None) -> dict[str, Any]:
    # One BashSession shared by Bash/BashOutput/KillShell so background shells
    # launched by Bash are visible to the companion tools.
    session = _BashSession(cwd)
    return {
        "Bash":  _make_bash_tool(cwd, nudge, session=session),
        "BashOutput": _make_bashoutput_tool(session),
        "KillShell":  _make_killshell_tool(session),
        "Read":  _make_read_tool(cwd),
        "Write": _make_write_tool(cwd, nudge),
        "Edit":  _make_edit_tool(cwd),
        "Glob":  _make_glob_tool(cwd),
        "Grep":  _make_grep_tool(),
    }


def _build_arxiv_closures() -> dict:
    """Return arxiv tool callables keyed by their MCP-compatible tool name.

    Returns an empty dict if the ``arxiv`` package is not installed.
    Used by the OpenAI-compatible adapters (via _make_literature_tools) and
    LiteratureReviewAgent.build_closure_tools() so both backends share
    identical arxiv tool implementations.
    """
    try:
        import arxiv as _arxiv
    except ImportError:
        return {}

    def search_papers(query: str, max_results: int = 10) -> str:
        """Search arxiv for papers matching query."""
        # MCP string-in tools pass numbers as strings ("5"); the arxiv library
        # does int arithmetic on max_results internally → 'str - int' crash.
        max_results = int(max_results)
        client = _arxiv.Client()
        results = list(client.results(_arxiv.Search(query=query, max_results=max_results)))
        lines = []
        for r in results:
            lines.append(f"[{r.entry_id}] {r.title} ({r.published.year})\n  {r.summary[:300]}")
        return "\n\n".join(lines) or "(no results)"

    def list_papers(category: str, max_results: int = 10) -> str:
        """List recent arxiv papers in a category (e.g. 'cs.LG')."""
        max_results = int(max_results)
        client = _arxiv.Client()
        results = list(client.results(_arxiv.Search(query=f"cat:{category}", max_results=max_results)))
        return "\n".join(f"[{r.entry_id}] {r.title}" for r in results) or "(no results)"

    def _fetch_pdf(paper_id: str, dest: str) -> None:
        """Fetch an arXiv PDF straight from its canonical URL into ``dest``.

        Direct-by-URL on purpose: the arxiv library's Result.download_pdf was
        removed in recent versions ('Result' object has no attribute
        'download_pdf'), and resolving the id through client.results() costs a
        rate-limited round-trip we don't need — the id alone determines the PDF
        URL. So there is no library call to fail and no reason for the agent to
        re-try a different tool for the same result.
        """
        import urllib.request
        pid = paper_id.strip().rstrip("/").split("/")[-1]  # tolerate abs URLs
        url = f"https://arxiv.org/pdf/{pid}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "f3dasm-agent/1.0 (mailto:f3dasm@brown.edu)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        with open(dest, "wb") as f:
            f.write(data)

    def download_paper(paper_id: str, output_dir: str = ".") -> str:
        """Download a paper PDF from arxiv by its ID (direct URL)."""
        pid = paper_id.strip().rstrip("/").split("/")[-1]
        dest = str(Path(output_dir) / f"{pid}.pdf")
        _fetch_pdf(paper_id, dest)
        return f"Downloaded: {dest}"

    def read_paper(paper_id: str) -> str:
        """Download (direct URL) and extract text from an arxiv paper."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "paper.pdf")
            _fetch_pdf(paper_id, path)
            try:
                import fitz
                doc = fitz.open(path)
                return "\n".join(page.get_text() for page in doc)
            except ImportError:
                import subprocess
                result = subprocess.run(
                    ["pdftotext", path, "-"], capture_output=True, text=True
                )
                return result.stdout or "(could not extract text — install pymupdf or pdftotext)"

    return {
        "arxiv_search_papers": search_papers,
        "arxiv_list_papers": list_papers,
        "arxiv_download_paper": download_paper,
        "arxiv_read_paper": read_paper,
    }


def _make_literature_tools() -> list:
    """Build LangChain StructuredTools equivalent to the literature MCP stack."""
    from langchain_core.tools import StructuredTool

    tools = []

    # --- arxiv (via shared _build_arxiv_closures) ---
    for name, fn in _build_arxiv_closures().items():
        tools.append(StructuredTool.from_function(fn, name=name))

    # --- semantic scholar ---
    try:
        from semanticscholar import SemanticScholar as _SS

        _ss = _SS(api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))

        def get_paper_details(paper_id: str) -> str:
            """Get details for a paper by its S2, DOI, or arxiv ID."""
            import json
            paper = _ss.get_paper(paper_id, fields=[
                "title", "year", "venue", "citationCount",
                "influentialCitationCount", "tldr", "authors",
            ])
            return json.dumps({
                "title": paper.title,
                "year": paper.year,
                "venue": paper.venue,
                "citations": paper.citationCount,
                "influential_citations": paper.influentialCitationCount,
                "tldr": paper.tldr.get("text") if paper.tldr else None,
                "authors": [a["name"] for a in (paper.authors or [])],
            }, indent=2)

        def get_citations_and_references(paper_id: str) -> str:
            """Get citing papers and references for a paper."""
            import json
            paper = _ss.get_paper(paper_id, fields=["citations", "references"])
            refs = [{"title": r.get("title"), "paperId": r.get("paperId")} for r in (paper.references or [])]
            cits = [{"title": c.get("title"), "paperId": c.get("paperId")} for c in (paper.citations or [])]
            return json.dumps({"references": refs[:20], "citations": cits[:20]}, indent=2)

        tools += [
            StructuredTool.from_function(
                get_paper_details,
                name="mcp__semanticscholar__get_semantic_scholar_paper_details",
            ),
            StructuredTool.from_function(
                get_citations_and_references,
                name="mcp__semanticscholar__get_semantic_scholar_citations_and_references",
            ),
        ]
    except ImportError:
        pass  # semanticscholar package not installed

    # --- zotero (read-only) ---
    try:
        import os as _os

        from pyzotero import zotero as _pyzotero

        _lib_id = _os.environ.get("ZOTERO_LIBRARY_ID")
        _api_key = _os.environ.get("ZOTERO_API_KEY")

        if _lib_id and _api_key:
            _zot = _pyzotero.Zotero(_lib_id, "user", _api_key)

            def zotero_search_items(query: str, limit: int = 10) -> str:
                """Search Zotero library for items matching query."""
                import json
                items = _zot.items(q=query, limit=limit)
                return json.dumps([
                    {"key": i["key"], "title": i["data"].get("title"), "year": i["data"].get("date", "")[:4]}
                    for i in items
                ], indent=2)

            def zotero_get_item_metadata(item_key: str) -> str:
                """Get metadata for a Zotero item by key."""
                import json
                return json.dumps(_zot.item(item_key)["data"], indent=2)

            def zotero_get_item_fulltext(item_key: str) -> str:
                """Get full text content of a Zotero item."""
                try:
                    return _zot.fulltext_item(item_key).get("content", "(no fulltext)")
                except Exception as e:
                    return f"(fulltext unavailable: {e})"

            def zotero_semantic_search(query: str, limit: int = 10) -> str:
                """Semantic search over Zotero library."""
                return zotero_search_items(query, limit)  # fallback to keyword

            tools += [
                StructuredTool.from_function(zotero_search_items, name="mcp__zotero-mcp__zotero_search_items"),
                StructuredTool.from_function(zotero_get_item_metadata, name="mcp__zotero-mcp__zotero_get_item_metadata"),
                StructuredTool.from_function(zotero_get_item_fulltext, name="mcp__zotero-mcp__zotero_get_item_fulltext"),
                StructuredTool.from_function(zotero_semantic_search, name="mcp__zotero-mcp__zotero_semantic_search"),
            ]
    except ImportError:
        pass  # pyzotero not installed

    return tools


class OpenAICompatibleAdapter:
    """Base adapter for any OpenAI chat-completions-compatible backend.

    Concrete backends subclass this and set only the endpoint/auth class
    attributes below; the invoke/tool/usage machinery is inherited. Exposes
    the same public interface as ClaudeAdapter: a mutable closure_tools dict
    and invoke().

    Class attributes (override per backend)
    ---------------------------------------
    DEFAULT_BASE_URL : str
        Endpoint used when neither an explicit base_url nor BASE_URL_ENV is set.
    BASE_URL_ENV : str | None
        Environment variable consulted for the base_url (env > default).
    API_KEY : str | None
        Default API key when none is supplied / no API_KEY_ENV is set.
    API_KEY_ENV : str | None
        Environment variable consulted for the API key (env > API_KEY).

    Resolution precedence: explicit constructor arg > environment > default.
    """

    DEFAULT_BASE_URL: str = "http://localhost:11434/v1"
    BASE_URL_ENV: str | None = None
    API_KEY: str | None = "local"
    API_KEY_ENV: str | None = None

    @classmethod
    def select_native_tools(cls, agent_tools) -> list[str]:
        """Pick which of an agent's declared tools are NATIVE (CLI) tools.

        OpenAI-compatible backends run every tool through LangChain, so a tool
        is native unless it is one of the Python closure tools the node injects
        separately. (ClaudeAdapter overrides this with its own CLI-tool set.)
        """
        return [t for t in agent_tools if t not in _CLOSURE_TOOL_NAMES]

    def __init__(
        self,
        model: str,
        system_prompt: str,
        study_dir: Any = None,
        native_tools: list[str] | None = None,
        closure_tools: dict[str, Any] | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        extra_mcp_servers: dict | None = None,
        extra_allowed_tools: list[str] | None = None,
        persistent: bool = False,
        max_history_pairs: int = 5,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.study_dir = Path(study_dir) if study_dir else None
        # Use 'native_tools' (not '_native_tool_names') so ImplementerNode's
        # sandboxed-Write setup can find it by the same attribute name as
        # ClaudeAdapter.
        self.native_tools: list[str] = list(native_tools or [])
        self.closure_tools: dict[str, Any] = dict(closure_tools or {})
        # Endpoint + auth: explicit arg > environment > class default.
        if base_url is None and self.BASE_URL_ENV:
            base_url = os.environ.get(self.BASE_URL_ENV)
        self._base_url = base_url or self.DEFAULT_BASE_URL
        if api_key is None and self.API_KEY_ENV:
            api_key = os.environ.get(self.API_KEY_ENV)
        self._api_key = api_key if api_key is not None else self.API_KEY
        self.extra_mcp_servers: dict = dict(extra_mcp_servers or {})
        self.extra_allowed_tools: list[str] = list(extra_allowed_tools or [])
        # persistent and max_history_pairs kept for backward compatibility with
        # Agent subclasses and tests that read these attributes; not used
        # in the core invocation path (history is demand-driven via DelegationLog).
        self.persistent: bool = persistent
        self.max_history_pairs: int = max_history_pairs
        # Lock serializes concurrent delegations to the same shared adapter.
        self._lock: threading.Lock = threading.Lock()
        # Built lazily so that closure_tools are fully populated before first
        # invoke().  Reset to None whenever native_tools or closure_tools change
        # so the next invoke() picks up the updated tool set.
        self._agent: Any = None
        # route_watcher is set by StrategizerNode; unused by these adapters
        # (create_react_agent runs the full tool loop to completion) but must
        # be present so StrategizerNode.__init__ doesn't raise AttributeError.
        self.route_watcher: Any = None
        # Non-blocking raw-oracle nudge, capped per delegation (= per invoke).
        from .base import OracleNudgeBudget
        self._oracle_nudge = OracleNudgeBudget()
        # Populated after each invoke() with token counts for run-level accounting.
        self.last_usage: dict = {}

    def copy(self) -> OpenAICompatibleAdapter:
        """Always return self.

        Concurrent delegations share this adapter instance and are serialized
        via _lock in invoke(). Episodic memory is demand-driven via RecallHistory
        (backed by DelegationLog) rather than per-adapter history injection.
        """
        return self

    def _build_tools(self) -> list[Any]:
        from langchain_core.tools import StructuredTool

        native_map = _native_tool_map(self.study_dir, self._oracle_nudge.check)
        tools: list[Any] = [
            native_map[name]
            for name in self.native_tools
            if name in native_map
        ]
        for name, fn in self.closure_tools.items():
            tools.append(StructuredTool.from_function(fn, name=name))
        # Inject MCP-equivalent tools for declared extra_allowed_tools.
        if self.extra_allowed_tools:
            lit_tools = _make_literature_tools()
            allowed = set(self.extra_allowed_tools)
            tools += [t for t in lit_tools if t.name in allowed]
        return tools

    def _build_agent(self) -> Any:
        from langchain_core.messages import SystemMessage
        from langchain_openai import ChatOpenAI
        from langgraph.prebuilt import create_react_agent

        from ..tool_catalog import system_prompt_with_catalog
        llm = ChatOpenAI(
            model=self.model, base_url=self._base_url, api_key=self._api_key
        )
        return create_react_agent(
            llm,
            self._build_tools(),
            prompt=SystemMessage(content=system_prompt_with_catalog(
                self.system_prompt, self.closure_tools)),
        )

    def invoke(
        self, messages: list[dict], *,
        idle_timeout: float | None = None, retry_max: int | None = None,
    ) -> str:
        """Run one full agent turn; return final assistant text.

        Acquires _lock to serialize concurrent callers (e.g. parallel
        delegations to the same shared worker adapter). Transient API/network
        failures are retried with exponential backoff (see retry_on_transient).

        ``idle_timeout`` / ``retry_max`` give short advisory side-calls a tight
        budget. ``idle_timeout`` is accepted for signature parity with the
        Claude backend (HTTP requests carry their own socket timeout, so it is
        not separately applied here); ``retry_max`` caps retries for this call.
        """
        from .base import retry_on_transient
        with self._lock:
            return retry_on_transient(
                lambda: self._invoke_once(messages), max_attempts=retry_max)

    def _invoke_once(self, messages: list[dict]) -> str:
        """Core invoke logic — build agent if needed, run, return text."""
        # One invoke == one delegation's worker run; reset the per-delegation
        # nudge cap. The cached agent's tool closures read this live.
        self._oracle_nudge.reset()
        if self._agent is None:
            self._agent = self._build_agent()

        lc_msgs = _to_lc_messages(messages)
        result = self._agent.invoke(
            {"messages": lc_msgs},
            config={"configurable": {"thread_id": str(uuid.uuid4())}},
        )
        last = result["messages"][-1]

        # DEBUG: capture full reasoning + tool-calls (parity with Claude).
        from .base import append_transcript, debug_enabled
        if debug_enabled():
            for _m in result.get("messages", []):
                append_transcript({
                    "type": _m.__class__.__name__,
                    "text": str(getattr(_m, "content", "")),
                    "tools": getattr(_m, "tool_calls", None) or [],
                })

        # Extract token usage from LangChain response metadata.
        meta = getattr(last, "usage_metadata", None) or {}
        self.last_usage = {
            "input_tokens": meta.get("input_tokens", 0),
            "output_tokens": meta.get("output_tokens", 0),
            "cache_read_input_tokens": meta.get("input_token_details", {}).get("cache_read", 0),
            "cache_creation_input_tokens": meta.get("input_token_details", {}).get("cache_creation", 0),
            "total_cost_usd": None,  # not available from open-weight/self-hosted
        }

        return str(last.content)
