"""Claude-SDK-backed adapter for the f3dasm LangGraph agentic runtime."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect as _inspect
import threading
import time
from pathlib import Path
from typing import Any

__all__ = ["ClaudeAdapter"]

# The in-process MCP server name every f3dasm closure is registered under. The
# Claude SDK exposes each closure to the model ONLY by its qualified name
# ``mcp__<server>__<tool>`` (that is also what allowed_tools carries), so the
# prompt's <tools> catalog must advertise the SAME qualified names — a bare-name
# catalog tells the model to call a tool that does not exist ("No such tool
# available"). One constant + one helper feed BOTH the registration and the
# catalog so they can never drift apart.
_CLOSURE_MCP_SERVER = "f3dasm_agent_tools"


def _qualify_closure_names(closure_tools: dict) -> dict:
    """Re-key a bare closure dict by the MCP-qualified names the SDK exposes.

    Pure; returns ``{}`` unchanged on an empty dict. Used for both the
    allowed_tools list and the <tools> catalog so the model is shown exactly
    the names it can call.
    """
    return {
        f"mcp__{_CLOSURE_MCP_SERVER}__{name}": fn
        for name, fn in closure_tools.items()
    }


_SDK_AVAILABLE: bool | None = None  # None = not yet checked


def _require_sdk() -> None:
    global _SDK_AVAILABLE
    if _SDK_AVAILABLE is None:
        try:
            import claude_agent_sdk  # noqa: F401
            _SDK_AVAILABLE = True
        except ImportError:
            _SDK_AVAILABLE = False
    if not _SDK_AVAILABLE:
        raise ImportError(
            "claude-agent-sdk is required for the Claude backend. "
            "Install it with: uv add claude-agent-sdk"
        )


_TYPE_MAP: dict = {
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    dict: {"type": "object"},
    list: {"type": "array"},
    str: {"type": "string"},
}


def _infer_schema_from_callable(fn: Any) -> dict:
    """Build a JSON schema dict from a Python callable's type annotations."""
    sig = _inspect.signature(fn)
    props: dict = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        # Skip leading-underscore params: by convention these are closure
        # constants bound via the default-arg idiom (e.g. `_ws=delegation_ws`,
        # `_did=delegation_id`), NOT model inputs. Exposing them let a model
        # pass e.g. `_ws` as a string → `str / path` TypeError in a write tool.
        if pname.startswith("_"):
            continue
        ann = param.annotation
        if ann is _inspect.Parameter.empty:
            json_type = {"type": "string"}
        else:
            json_type = _TYPE_MAP.get(ann, {"type": "string"})
        props[pname] = json_type
        if param.default is _inspect.Parameter.empty:
            required.append(pname)
    schema: dict = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


def _run_async_safe(coro: Any) -> Any:
    """Run *coro* safely whether or not an event loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Nested-loop case (e.g. the critic invoked from inside the
        # strategizer's running ainvoke): we hop to a fresh thread, so
        # propagate the thread-local context (transcript sink + delegation id)
        # into it — otherwise the critic's stream/env would silently lose them.
        from .base import (
            get_delegation_id,
            get_transcript_sink,
            set_delegation_id,
            set_transcript_sink,
        )
        _sink, _did = get_transcript_sink(), get_delegation_id()

        def _runner() -> Any:
            set_transcript_sink(_sink)
            set_delegation_id(_did)
            return asyncio.run(coro)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_runner).result()
    else:
        return asyncio.run(coro)


def _format_messages_as_prompt(messages: list[dict]) -> str:
    """Convert LangChain-style message dicts to a plain conversation string."""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        if role in ("human", "user"):
            prefix = "Human"
        elif role in ("ai", "assistant"):
            prefix = "Assistant"
        else:
            continue  # skip system messages — passed via system_prompt
        parts.append(f"{prefix}: {content}")
    return "\n\n".join(parts)


_STREAM_DONE = object()


async def _anext_or_done(ait: Any) -> Any:
    """Return the next item, or the _STREAM_DONE sentinel when exhausted.

    Catches StopAsyncIteration INSIDE the coroutine so it never escapes into an
    asyncio Task (which would surface as a RuntimeError under wait_for).
    """
    try:
        return await ait.__anext__()
    except StopAsyncIteration:
        return _STREAM_DONE


async def _stream_with_idle_timeout(
    gen: Any,
    idle_timeout: float,
    *,
    tool_timeout: float = 0.0,
    classify: Any = None,
):
    """Yield messages from an SDK async stream, raising TimeoutError if the
    stream goes idle while AWAITING MODEL GENERATION for longer than
    ``idle_timeout`` seconds.

    This catches a *stalled* API response — an ESTABLISHED-but-silent
    connection that streams nothing and never errors — and turns it into a
    transient TimeoutError that ``retry_on_transient`` (wrapping ``invoke``)
    retries with backoff. It is an IDLE timeout (reset on every message), NOT a
    total cap, so a legitimately long call that keeps streaming tokens is never
    cut; only a true stall is.

    The window is scoped to *model generation*. While a TOOL is executing — a
    worker running a multi-minute Bash compute job emits no stream messages —
    the tight window stands down, because that silence is the tool working, not
    the API stalling. Penalising it false-fires and (via the retry) re-runs the
    whole, non-idempotent worker. ``classify(msg)`` reports the phase:
    ``True`` a tool is now executing (suspend the tight window), ``False``
    generation/tool-result (re-arm it), ``None`` leave the phase unchanged.
    While a tool is pending the next message is awaited with ``tool_timeout``
    (``<= 0`` means no cap — a runaway tool is the delegation watchdog's job,
    not this stream timeout's). With ``classify=None`` the tight window applies
    always (backward-compatible).
    """
    ait = gen.__aiter__()
    tool_pending = False
    while True:
        if tool_pending:
            _wait = tool_timeout if tool_timeout and tool_timeout > 0 else None
        else:
            _wait = idle_timeout
        try:
            msg = await asyncio.wait_for(_anext_or_done(ait), timeout=_wait)
        except asyncio.TimeoutError as exc:
            # Only generation silence reaches here; while a tool is pending
            # _wait is None (uncapped) so this never trips on tool execution.
            raise TimeoutError(
                f"Anthropic stream stalled: no model output for "
                f"{idle_timeout:.0f}s (transient; will retry)"
            ) from exc
        if msg is _STREAM_DONE:
            return
        if classify is not None:
            phase = classify(msg)
            if phase is True:
                tool_pending = True
            elif phase is False:
                tool_pending = False
        yield msg


def _build_session_env() -> dict:
    """Per-session env vars injected into the worker subprocess (thread-local).

    - ``F3DASM_DELEGATION_ID`` (race-safe) so get_evaluator() resolves without
      the worker cd-ing into its D### dir (audit Finding 2).
    - ``F3DASM_RUN_CONFIG`` — explicit path to run_config.json so get_evaluator()
      resolves it regardless of cwd (the SDK spawns the worker in study_dir, from
      which the old walk-up never reached runs/<id>/debug/run_config.json).
    - ``F3DASM_CANONICAL_STORE`` — the canonical store path, derived from
      run_config["store_dir"]. get_evaluator() reads store_dir from the config,
      but a worker's OWN campaign scripts read os.environ["F3DASM_CANONICAL_STORE"]
      directly; without this the var is empty in the worker shell, so a campaign
      defaults to the wrong namespace and can overwrite another delegation's
      scratch data (audit run 20260624T021359, D005→D006 sim-dir clobber).
    """
    import os
    import sys

    from .base import (
        get_delegation_id,
        get_namespace,
        get_run_config_path,
    )
    env: dict = {}
    # The agent's shell must run the SAME interpreter as the agent loop, so
    # `python`/`uv run python` in Bash can import whatever the framework can
    # (f3dasm, a3dasm, the study's deps). Without this, bash `python` resolves
    # via the inherited PATH — which need not include the run's venv when the
    # loop was launched by invoking the venv's python binary directly (no
    # activation), leaving the agent unable to import the package and forced
    # into off-ledger workarounds. Prepend the run interpreter's bin dir.
    _bin = os.path.dirname(sys.executable)
    if _bin:
        env["PATH"] = _bin + os.pathsep + os.environ.get("PATH", "")
    did = get_delegation_id()
    if did:
        env["F3DASM_DELEGATION_ID"] = did
    ns = get_namespace()
    if ns:
        # Scope this worker to a design namespace so get_evaluator() resolves
        # that namespace's oracle + ledger (Axis 3a). Absent → single-study path.
        env["F3DASM_NAMESPACE"] = ns
    rc = get_run_config_path()
    if rc:
        env["F3DASM_RUN_CONFIG"] = rc
        try:
            import json as _json
            store = _json.loads(Path(rc).read_text()).get("store_dir")
            if store:
                env["F3DASM_CANONICAL_STORE"] = str(store)
        except Exception:  # noqa: BLE001 — best-effort, never fatal
            pass
    return env


class ClaudeAdapter:
    """Wraps claude-agent-sdk; runs one agent turn and returns assistant text.

    The SDK handles its own tool-execution loop (Bash, Read, Write, Edit).
    This adapter converts a list of LangChain-style message dicts to the SDK
    format, runs the query, and assembles the final text response.

    Parameters
    ----------
    model : str
        Claude model identifier.
    system_prompt : str
        System prompt for the agent.
    study_dir : Path or None
        Working directory passed to the SDK as ``cwd``.
    native_tools : list[str]
        Tool names to enable (e.g. ``["Bash", "Read", "Write"]``).
    closure_tools : dict[str, callable] or None
        Extra Python callables exposed to the model as MCP tools.
    """

    # CLI tools the Claude SDK executes natively. A node's other declared tools
    # are injected as Python closures (MCP), not passed here.
    NATIVE_TOOLS = frozenset({
        "Bash", "Edit", "Read", "Write", "Glob", "Grep",
        # Bash's own SDK companions: poll a backgrounded shell / kill it. These
        # are SDK built-ins we previously omitted, so an agent that got a
        # backgroundTaskId (from auto-background on timeout) had no tool to act
        # on it. Granting them by declaration closes that awareness gap.
        "BashOutput", "KillShell",
        "Task", "WebFetch", "WebSearch", "computer",
    })

    @classmethod
    def select_native_tools(cls, agent_tools) -> list[str]:
        """Pick which of an agent's declared tools are native SDK CLI tools.

        Mirror of OpenAICompatibleAdapter.select_native_tools so the runtime
        can choose native tools generically for any backend (forward-compatible
        dispatch)."""
        return [t for t in agent_tools if t in cls.NATIVE_TOOLS]

    def __init__(
        self,
        model: str,
        system_prompt: str,
        study_dir: Path | None = None,
        native_tools: list[str] | None = None,
        closure_tools: dict[str, Any] | None = None,
        extra_mcp_servers: dict | None = None,
        extra_allowed_tools: list[str] | None = None,
        persistent: bool = False,
        max_history_pairs: int = 5,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.study_dir = Path(study_dir) if study_dir else None
        self.native_tools = list(native_tools or [])
        self.closure_tools = dict(closure_tools or {})
        self.extra_mcp_servers: dict = dict(extra_mcp_servers or {})
        self.extra_allowed_tools: list[str] = list(extra_allowed_tools or [])
        # persistent and max_history_pairs kept for backward compatibility with
        # Agent subclasses and tests that read these attributes; not used
        # in the core invocation path (history is demand-driven via DelegationLog).
        self.persistent: bool = persistent
        self.max_history_pairs: int = max_history_pairs
        # Lock serializes concurrent delegations to the same shared adapter.
        self._lock: threading.Lock = threading.Lock()
        # Set by StrategizerNode; when truthy, the generator is closed after
        # the next AssistantMessage so the session ends on a routing decision.
        self.route_watcher: Any = None
        # Populated after each ainvoke() with token counts from ResultMessage.
        self.last_usage: dict = {}

    def _compute_allowed_tools(self, qualified_mcp_tools) -> list[str]:
        """All allowed tool names, ALWAYS as a list (never None).

        The SDK does ``list(options.allowed_tools)`` when building its command,
        which raises ``TypeError`` on ``None`` — so a tool-less agent (e.g. the
        one-shot problem-statement reviewer) must still get ``[]`` here, not
        ``None``. An empty list correctly means "no tools allowed".
        """
        return (
            list(qualified_mcp_tools)
            + list(self.native_tools)
            + list(self.extra_allowed_tools)
        )

    def copy(self) -> ClaudeAdapter:
        """Always return self.

        Concurrent delegations share this adapter instance and are serialized
        via _lock in invoke(). Episodic memory is demand-driven via RecallHistory
        (backed by DelegationLog) rather than per-adapter history injection.
        """
        return self

    async def ainvoke(
        self, messages: list[dict], *, idle_timeout: float | None = None,
    ) -> str:
        """Run one agent turn asynchronously; return assembled text.

        ``idle_timeout`` overrides the run-wide ``llm_stream_idle_timeout`` for
        THIS call only — used by short advisory side-calls (e.g. the verdict
        validator) that must not inherit a real agent turn's generous window.
        """
        _require_sdk()
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            SdkMcpTool,
            StreamEvent,
            TextBlock,
            ToolUseBlock,
            UserMessage,
            create_sdk_mcp_server,
            query,
        )

        # Build MCP server from closure_tools if any
        mcp_servers: dict = {}
        qualified_mcp_tools: list[str] = []
        if self.closure_tools:
            server_name = _CLOSURE_MCP_SERVER
            sdk_tools: list[Any] = []
            for tool_name, fn in self.closure_tools.items():
                schema = _infer_schema_from_callable(fn)

                async def _handler(args: dict, bound_fn: Any = fn) -> dict:
                    try:
                        result = bound_fn(**args)
                    except Exception as exc:
                        return {
                            "content": [
                                {"type": "text", "text": f"ERROR: {exc}"}
                            ],
                            "is_error": True,
                        }
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    str(result) if result is not None else ""
                                ),
                            }
                        ]
                    }

                sdk_tools.append(
                    SdkMcpTool(
                        name=tool_name,
                        description=(
                            (fn.__doc__ or tool_name).split("\n")[0].strip()
                        ),
                        input_schema=schema,
                        handler=_handler,
                    )
                )

            mcp_cfg = create_sdk_mcp_server(
                name=server_name, tools=sdk_tools or None
            )
            mcp_servers = {server_name: mcp_cfg}
            qualified_mcp_tools = list(_qualify_closure_names(self.closure_tools))

        # Merge external stdio MCP servers declared by the Agent subclass.
        if self.extra_mcp_servers:
            mcp_servers.update(self.extra_mcp_servers)

        _base_disallowed = ["WebSearch", "WebFetch", "Task", "ExitPlanMode", "computer"]
        # Under permission_mode="bypassPermissions" the allowed_tools allowlist
        # is NOT enforced — disallowed_tools is the only thing that binds. So a
        # native tool the agent never declared (e.g. Bash/Write for a read-only
        # reviewer) would otherwise be silently usable. Disallow every native
        # tool this agent did not declare, making its declared toolset binding.
        _ungranted_native = [
            t for t in self.NATIVE_TOOLS if t not in self.native_tools
        ]
        _effective_disallowed = [
            t for t in dict.fromkeys([*_base_disallowed, *_ungranted_native])
            if t not in self.extra_allowed_tools
        ]

        # Non-blocking raw-oracle nudge: a PostToolUse hook that injects a
        # reminder (capped per delegation = per ainvoke) when a Bash/Write
        # call reaches the oracle directly instead of via get_evaluator().
        # Best-effort — if the SDK hook API is unavailable, run without it.
        _hooks = None
        try:
            from claude_agent_sdk import HookMatcher

            from .base import OracleNudgeBudget, oracle_registered
            # Silent until an oracle is registered: pre-registration work (the
            # datagenerator wrapping/validating its raw source) has no
            # get_evaluator() to use, so nudging it is a false positive.
            _nudge = OracleNudgeBudget(enabled=oracle_registered())
            # Expose on the adapter so the runtime can drain + log its
            # firings as direct evidence (see _record_intervention).
            self._oracle_nudge = _nudge

            async def _oracle_hook(input_data, tool_use_id, context):
                msg = _nudge.check(
                    input_data.get("tool_name", ""),
                    input_data.get("tool_input") or {},
                )
                if not msg:
                    return {}
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": msg,
                    }
                }

            _hooks = {"PostToolUse": [HookMatcher(hooks=[_oracle_hook])]}
        except Exception:  # noqa: BLE001 — nudge is best-effort, never fatal
            _hooks = None

        # Per-session env: the SDK MERGES this over the inherited environment
        # (PATH etc. preserved), so bare extra keys are safe. See
        # _build_session_env for what is injected and why.
        _sess_env: dict = _build_session_env()

        from ..settings import get_float
        _max_buf_mb = get_float("llm_max_buffer_mb", 30.0)
        _max_buf = int(_max_buf_mb * 1024 * 1024)

        # The SDK spawns the CLI with cwd=self.study_dir; if that directory
        # doesn't exist the subprocess dies with a cryptic CLIConnectionError
        # ("Working directory does not exist") mid-delegation. Create it
        # defensively so a missing worker workspace can never abort a run.
        if self.study_dir:
            try:
                self.study_dir.mkdir(parents=True, exist_ok=True)
            except Exception:  # noqa: BLE001 — best-effort; spawn surfaces real errors
                pass

        from ..tool_catalog import system_prompt_with_catalog
        # Catalog shows the QUALIFIED names (same helper as allowed_tools above),
        # so the AUTHORITATIVE <tools> block matches what the model can call.
        options = ClaudeAgentOptions(
            system_prompt=system_prompt_with_catalog(
                self.system_prompt, _qualify_closure_names(self.closure_tools)),
            model=self.model,
            cwd=str(self.study_dir) if self.study_dir else None,
            tools=self.native_tools or [],
            mcp_servers=mcp_servers if mcp_servers else {},
            allowed_tools=self._compute_allowed_tools(qualified_mcp_tools),
            disallowed_tools=_effective_disallowed,
            permission_mode="bypassPermissions",
            strict_mcp_config=bool(mcp_servers) or bool(self.extra_mcp_servers),
            # Hermetic session: load NO filesystem settings, so worker/critic
            # subprocesses don't inherit the developer's global ~/.claude hooks
            # (e.g. cbm-code-discovery-gate, which blocked legitimate Read calls
            # for workers AND the critic). Our own hooks are passed
            # programmatically via options.hooks below (audit/#1: fresh hooks).
            setting_sources=[],
            env=_sess_env,
            # Stream-message buffer ceiling. Default 1MB is far too small for a
            # literature reviewer whose tools return full PDFs — a single >1MB
            # MCP tool result overflowed it and FATALLY (non-retried) killed the
            # whole delegation (D003). 30MB clears realistic PDFs; non-PDF
            # results never approach it. A result still exceeding this is caught
            # gracefully below (turn cut short + marker), not a fatal crash.
            # Tune via F3DASM_LLM_MAX_BUFFER_MB.
            max_buffer_size=_max_buf,
            # Partial streaming → a fine-grained heartbeat: the stream emits a
            # StreamEvent sub-second while genuinely generating, so total
            # silence becomes a reliable stall signal and the
            # idle timeout can be bounded without false-positiving a
            # slow-but-working generation.
            include_partial_messages=True,
            **({"hooks": _hooks} if _hooks else {}),
        )

        prompt_str = _format_messages_as_prompt(messages)

        last_assistant = None
        last_result: Any = None
        _buffer_overflowed = False
        gen = query(prompt=prompt_str, options=options)
        # Idle-stream timeout — turns a silent stream into a retryable
        # TimeoutError. Resets on EVERY stream message. Scoped to model
        # generation: while a tool runs the window stands down (see _phase).
        # MEASURED (the stream_evt instrumentation settled this): the bundled
        # CLI forwards NO ping events — only message lifecycle — so during a
        # mid-generation pause we get ZERO liveness signal. Two supercompressible
        # (Sonnet) runs captured LEGITIMATE (recovered) active-generation gaps of
        # 53.6s AND 252-256s (one mid-tool_use-block composition, one
        # tool_result→next message_start delay). So earlier 60s/180s windows
        # would guillotine legitimate slow generations. The observed ~256s legit
        # gap left only ~44s of headroom under the prior 300s default, and with
        # NO ping liveness signal a longer legit generation is indistinguishable
        # from a stall — so the default is 600s: ample margin over the measured
        # legit gaps while still catching a truly dead (silent-forever) stream in
        # ~10 min. A genuine runaway is the delegation watchdog's concern, not
        # this window's. Knobs (config.yaml runtime block; env overrides):
        # llm_stream_idle_timeout (0 disables); llm_tool_idle_timeout caps tool
        # execution (0 = uncapped).
        from ..settings import get_float as _get_float
        _idle = (idle_timeout if idle_timeout is not None
                 else _get_float("llm_stream_idle_timeout", 600.0))
        _tool_idle = _get_float("llm_tool_idle_timeout", 0.0)

        def _phase(msg: Any):
            # True: a tool is now executing → suspend the tight idle window.
            # False: generation active / tool result returned → tight window.
            # None: leave phase unchanged.
            if isinstance(msg, AssistantMessage):
                return any(
                    isinstance(b, ToolUseBlock) for b in msg.content
                )
            if isinstance(msg, (StreamEvent, UserMessage)):
                return False
            return None

        from .base import append_transcript, debug_enabled

        def _record(msg: Any):
            # Full reasoning + tool-calls + tool-results; StreamEvent partials
            # are skipped (the assembled AssistantMessage carries the text).
            if isinstance(msg, AssistantMessage):
                texts, tools, thinking = [], [], []
                for b in msg.content:
                    if isinstance(b, TextBlock):
                        texts.append(b.text)
                    elif isinstance(b, ToolUseBlock):
                        tools.append({"name": b.name, "input": b.input})
                    else:
                        t = (getattr(b, "thinking", None)
                             or getattr(b, "text", None))
                        if t:
                            thinking.append(t)
                return {"type": "assistant", "text": "".join(texts),
                        "tools": tools, "thinking": thinking}
            if isinstance(msg, UserMessage):
                results = []
                for b in (getattr(msg, "content", None) or []):
                    results.append({
                        "tool_use_id": getattr(b, "tool_use_id", None),
                        "content": getattr(b, "content", b),
                    })
                return {"type": "tool_result", "results": results}
            if isinstance(msg, ResultMessage):
                return {"type": "result",
                        "usage": getattr(msg, "usage", None),
                        "cost_usd": getattr(msg, "total_cost_usd", None)}
            return None

        _capture = debug_enabled()
        # Partial-stream checkpointing: an agent that never completes a message
        # (infinite thinking / an unresolved turn) emits only StreamEvents and
        # would otherwise disclose nothing. Buffer the deltas and flush a
        # "partial" record every N events (and on any stream teardown) so the
        # transcript reveals what a stuck turn is doing in near-real-time.
        _PARTIAL_FLUSH_EVERY = 25
        _pbuf: list[str] = []
        _pcount = [0]

        def _extract_delta(ev: Any) -> str:
            if not isinstance(ev, dict):
                return ""
            d = ev.get("delta") or {}
            return (d.get("text") or d.get("thinking")
                    or d.get("partial_json") or "")

        def _flush_partial() -> None:
            if _pbuf:
                append_transcript({"type": "partial", "text": "".join(_pbuf),
                                   "events": _pcount[0]})
                _pbuf.clear()

        try:
            _stream = (
                _stream_with_idle_timeout(
                    gen, _idle,
                    tool_timeout=_tool_idle, classify=_phase,
                )
                if _idle > 0 else gen
            )
            # Measurement clock: the first event's gap below = time-to-first
            # stream event (≈ prefill latency).
            _last_evt = [time.monotonic()]
            async for msg in _stream:
                if _capture:
                    if isinstance(msg, StreamEvent):
                        _pcount[0] += 1
                        _ev = getattr(msg, "event", {}) or {}
                        _et = _ev.get("type", "?") if isinstance(
                            _ev, dict) else "?"
                        _now = time.monotonic()
                        _gap = _now - _last_evt[0]
                        _last_evt[0] = _now
                        # Record every NON-delta event (ping, message_start/
                        # stop, content_block_start/stop, message_delta) and any
                        # delta after a >2s pause — so we can see whether the
                        # stream stays alive (pings) during silent/prefill
                        # phases and the true inter-event gap distribution.
                        # This is what settles whether a 60s silence is a dead
                        # stream or a legitimately-slow first token.
                        if _et != "content_block_delta" or _gap > 2.0:
                            append_transcript({
                                "type": "stream_evt", "evt": _et,
                                "gap_s": round(_gap, 2)})
                        _d = _extract_delta(_ev)
                        if _d:
                            _pbuf.append(_d)
                        if _pcount[0] % _PARTIAL_FLUSH_EVERY == 0:
                            _flush_partial()
                    else:
                        # A complete message: flush any buffered partial first,
                        # then the structured record.
                        _flush_partial()
                        _rec = _record(msg)
                        if _rec is not None:
                            append_transcript(_rec)
                if isinstance(msg, AssistantMessage):
                    last_assistant = msg
                    if self.route_watcher and self.route_watcher():
                        break
                elif isinstance(msg, ResultMessage):
                    last_result = msg
                    break
        except Exception as exc:  # noqa: BLE001
            _m = str(exc).lower()
            if "maximum buffer size" in _m or "exceeded maximum" in _m:
                # Graceful contour: a single tool result (e.g. a huge PDF)
                # overflowed the stream buffer. The old behavior was a fatal,
                # NON-retried crash that killed the whole delegation (D003).
                # Instead, end the turn with what we have + a marker so the
                # agent can retry the fetch smaller — the delegation survives.
                _buffer_overflowed = True
            else:
                raise
        finally:
            if _capture:
                _flush_partial()  # disclose a stuck/torn-down turn's tail
            aclose = getattr(gen, "aclose", None)
            if aclose:
                try:
                    await aclose()
                except Exception:
                    pass

        # Capture token usage from ResultMessage for run-level accounting.
        if last_result is not None:
            self.last_usage = {
                **(last_result.usage or {}),
                "total_cost_usd": last_result.total_cost_usd,
            }
        else:
            self.last_usage = {}

        text = ""
        if last_assistant is not None:
            for block in last_assistant.content:
                if isinstance(block, TextBlock):
                    text += block.text
        if _buffer_overflowed:
            _note = (
                f"[STREAM NOTE: a tool returned more than {_max_buf_mb:.0f} MB "
                "in a single result and overflowed the message buffer; that "
                "result was dropped and this turn was cut short (the delegation "
                "did NOT crash). Re-run the tool with a smaller/narrower request "
                "— fewer items, or a summary/extract instead of full text.]"
            )
            text = (text + "\n\n" + _note) if text else _note
        return text

    def invoke(
        self, messages: list[dict], *,
        idle_timeout: float | None = None, retry_max: int | None = None,
    ) -> str:
        """Synchronous wrapper around :meth:`ainvoke`.

        Acquires _lock to serialize concurrent callers (e.g. parallel
        delegations to the same shared worker adapter). Transient API/network
        failures are retried with exponential backoff (see retry_on_transient).

        ``idle_timeout`` / ``retry_max`` override the run-wide stream-idle and
        retry budgets for THIS call only. A short advisory side-call (verdict
        validator) passes a tight idle + ``retry_max=1`` so a hung CLI stream
        aborts in ~that window instead of inheriting a real turn's
        5×600s budget (which once froze a whole run for ~89 min).
        """
        from .base import retry_on_transient
        with self._lock:
            return retry_on_transient(
                lambda: _run_async_safe(
                    self.ainvoke(messages, idle_timeout=idle_timeout)),
                max_attempts=retry_max,
            )
