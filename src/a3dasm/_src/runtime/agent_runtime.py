"""f3dasm agentic runtime — thin LangGraph wrapper."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from ..agents import ImplementerAgent, StrategizerAgent, _default_graph
from ..backends.base import Agent, Graph
from ..infra.container_runner import ContainerRunner
from ..infra.delegation_log import DelegationLog
from ..prompts.agent_prompts import (
    RUN_PATHS_PREAMBLE_TEMPLATE,
    WORKSPACE_PREAMBLE_TEMPLATE,
)
from . import settings
from .graph_builder import build_graph
from .graph_state import AgenticState, Delegation, Report, StudyConfig, Task
from .run_setup import (
    AgenticRunError,
    _archive_prior_pipeline_notebook,
    _ingest_precomputed_pool,
    _init_canonical_store,
    _load_study_config,
    _parse_budget_str,
    resolve_mem_cap_bytes,
)

__all__ = [
    "AgenticRun",
    "AgenticRunError",
    "DEFAULT_MODEL",
    "DEFAULT_OLLAMA_MODEL",
    "Delegation",
    "ImplementerAgent",
    "Report",
    "StrategizerAgent",
    "StudyConfig",
    "Task",
]

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OLLAMA_MODEL = "qwen2.5:1.5b"

# Text the Claude Code CLI backend returns as an ORDINARY assistant message
# (not a raised exception) when an org-level billing cap stops it mid-run —
# confirmed for real: a run's strategizer turn ended with this exact text as
# its final reply, made no further tool calls, and the run closed UNGATED
# indistinguishable from the strategizer simply deciding it was done. This
# is a distinct, externally-caused stop condition (not a science/critic
# failure and not a bug in the agent's reasoning) that deserves its own
# stop_reason and explicit resume guidance rather than being silently
# folded into an ordinary UNGATED close (BACKLOG #34).
_EXTERNAL_STOP_SIGNATURES = {
    "org_spend_limit": "org's monthly spend limit",
}


class AgenticRun:
    """Run an agentic loop over a study directory.

    The entry point of a3dasm. It reads ``PROBLEM_STATEMENT.md`` from the study
    directory, builds the agent graph, runs the strategizer's open loop to a
    gated deliverable, and returns the final report. Configuration not passed
    here is read from ``<study_dir>/config.yaml``; explicit arguments win.

    Parameters
    ----------
    study_dir : Path
        Root of the study tree. Must contain ``PROBLEM_STATEMENT.md``.
    graph : Graph, optional
        Custom agent graph. Defaults to the built-in strategizer-hub graph.
    model : str, optional
        LLM model identifier. Defaults to ``config.yaml`` ``model`` or
        ``DEFAULT_MODEL`` (``DEFAULT_OLLAMA_MODEL`` when the backend is Ollama).
    budget : float, optional
        Wall-clock budget in seconds, or an ``"HH:MM:SS"`` string in
        ``config.yaml``. ``None`` means unlimited. Soft: it nudges, it does not
        hard-kill the science.
    budget_usd : float, optional
        Hard USD cost ceiling. Honoured only when the backend reports per-call
        cost (the Claude backend); ``None`` means no ceiling.
    eval_budget : int, optional
        Soft cap on oracle evaluations across all delegations. Nudges the
        strategizer when approached; never stops a run on its own.
    interactive : bool, default True
        Whether the pre-run problem-statement review and in-graph FollowUp may
        prompt on stdin. Forced off automatically when there is no TTY, so a
        headless run never blocks on input.
    max_ask : int, default 1
        Maximum number of clarifying questions the interactive review may ask.
    container : bool, default False
        Run the loop inside a container via ``ContainerRunner`` instead of
        in-process.
    container_image : str, default "f3dasm-agentic:latest"
        Image used when ``container`` is True.
    resume_from : Path, optional
        A prior run directory to resume from (replays the LangGraph checkpoint).
        The run must have a ``debug/thread_id``.
    review_statement : bool, default True
        Run the advisory pre-run problem-statement review. Never blocks an
        autonomous run.

    Examples
    --------
    >>> from a3dasm import AgenticRun
    >>> report = AgenticRun(study_dir="studies/my_study").execute()
    """

    def __init__(
        self,
        study_dir: Path,
        *,
        graph: Graph | None = None,
        model: str | None = None,
        budget: float | None = None,
        budget_usd: float | None = None,
        eval_budget: int | None = None,
        interactive: bool = True,
        max_ask: int = 1,
        container: bool = False,
        container_image: str = "f3dasm-agentic:latest",
        resume_from: Path | None = None,
        review_statement: bool = True,
    ) -> None:
        self.study_dir = Path(study_dir).resolve()
        cfg = _load_study_config(self.study_dir)
        # config.yaml is the source of truth for run knobs (debug, timeouts,
        # retry, backstop, recursion_limit, …). Install the `runtime:` block so
        # scattered read sites resolve via settings (env still overrides).
        settings.configure(cfg.get("runtime") or {})

        _backend_cfg = cfg.get("backend", "claude")
        self._backend = _backend_cfg
        self._model = model or cfg.get("model") or (
            DEFAULT_OLLAMA_MODEL if _backend_cfg == "ollama" else DEFAULT_MODEL
        )
        self._eval_budget = (
            eval_budget if eval_budget is not None else cfg.get("eval_budget")
        )
        # Hard per-campaign memory cap (bytes) — the single HARD resource boundary
        # (host safety, not a science budget). config.yaml `mem_cap` wins; else
        # env F3DASM_MEM_CAP; else the SLURM allocation (real HPC budget); else
        # the default. See resolve_mem_cap_bytes.
        self._mem_cap_bytes = resolve_mem_cap_bytes(cfg.get("mem_cap"))
        self._required_deliverables = cfg.get("required_deliverables") or []

        # budget from config is HH:MM:SS string or seconds float
        if budget is not None:
            self._budget = budget
        elif "budget" in cfg:
            self._budget = _parse_budget_str(cfg["budget"])
        else:
            self._budget = None

        # Hard USD cost ceiling (None = no ceiling). Honoured only when the
        # backend reports per-call cost (claude); ollama has no cost data.
        self._budget_usd = (
            budget_usd if budget_usd is not None else cfg.get("budget_usd")
        )

        self._graph_spec = graph or _default_graph()
        # `interactive` requires a real terminal: a headless/background run (no
        # TTY) has a stdin that blocks on read but never EOFs, so any input()
        # would hang the whole run forever. The in-graph FollowUp path already
        # guards on isatty(); the pre-run problem-statement review keys off
        # self._interactive alone, so fold the TTY check in HERE so EVERY
        # input() path is non-interactive when there is no terminal.
        import sys as _sys
        self._interactive = bool(interactive) and (
            getattr(_sys.stdin, "isatty", lambda: False)()
        )
        self._max_ask = max_ask
        self._container = container
        self._container_image = container_image
        self._resume_from = (
            Path(resume_from) if resume_from is not None else None
        )
        # Pre-run problem-statement review (Item B). User-controllable; cfg can
        # also disable it. Default on.
        self._review_statement = (
            review_statement
            if review_statement is not None
            else cfg.get("review_statement", True)
        )
        self._run_dir = None  # set in execute()

    @staticmethod
    def _write_run_status(debug_dir: Path, **payload) -> None:
        """Persist ``debug/run_status.json`` — the terminal status the §1 analysis
        protocol reads FIRST. Written on every close (normal gate outcome, crash,
        watchdog kill) so a run's outcome is always on disk, not only in the
        notebook metadata + the longitudinal ledger. Best-effort: a status write
        must never fail a run."""
        try:
            (debug_dir / "run_status.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _maybe_start_slurm_llm(self, full_cfg, debug_dir, log):
        """Optionally own a vLLM server on a SLURM GPU node for this run.

        Guarded by the ``llm_slurm.enabled`` config block. Submits a
        ``vllm serve`` job (reusing f3dasm's ``SlurmCluster`` + the plain
        ``sbatch`` submit idiom — a persistent server is not an eval array),
        waits for the granted node and a ready server, then publishes
        ``VLLM_BASE_URL`` so the vllm/openai-compatible adapter reaches it over
        the cluster network. Returns the SLURM job id (for teardown) or None
        when disabled.

        No silent fallback: if the feature is enabled and the server cannot be
        brought up, this raises — a run the user asked to serve locally must not
        quietly fall back to a hosted API. The jobid is persisted to disk so the
        study watchdog can reap a leaked allocation even if this process dies.
        """
        cfg = (full_cfg or {}).get("llm_slurm") or {}
        if not cfg.get("enabled"):
            return None

        from f3dasm import SlurmCluster

        from ..infra import slurm_llm

        model = cfg.get("model") or self._model
        spec = slurm_llm.resolve_serve_spec(model, cfg)
        warn = slurm_llm.serve_throughput_warning(spec)
        if warn:
            log.warning("llm_slurm: %s", warn)

        cluster_cfg = cfg.get("cluster") or {}
        cluster = SlurmCluster(
            partition=cluster_cfg.get("partition", "batch"),
            account=cluster_cfg.get("account", "default"),
            env_setup=list(cluster_cfg.get("env_setup", []) or []),
            env_vars=dict(cluster_cfg.get("env_vars", {}) or {}),
            runner=cluster_cfg.get("runner", "python"),
        )
        port = spec.profile.port
        script = slurm_llm.render_serve_script(
            spec, cluster, port, str(debug_dir))
        script_path = debug_dir / "vllm_serve.sh"
        script_path.write_text(script, encoding="utf-8")

        jobid = slurm_llm.submit_serve_job(str(script_path))
        (debug_dir / "serve_job.jobid").write_text(jobid, encoding="utf-8")
        log.info("llm_slurm: submitted serve job %s (model=%s)", jobid, model)

        queue_timeout = float(cfg.get("queue_timeout", 3600))
        serve_timeout = float(cfg.get("serve_timeout", 900))
        node = slurm_llm.wait_until_running(jobid, queue_timeout)
        base_url = f"http://{node}:{port}/v1"
        log.info("llm_slurm: job %s RUNNING on %s; waiting for vLLM at %s",
                 jobid, node, base_url)
        slurm_llm.wait_until_ready(base_url, serve_timeout)
        os.environ["VLLM_BASE_URL"] = base_url
        log.info("llm_slurm: server ready; published VLLM_BASE_URL=%s", base_url)
        if self._backend not in ("vllm", "openai", "openai_compatible"):
            log.warning(
                "llm_slurm.enabled but backend=%r (not vllm) — the served "
                "endpoint will be IGNORED. Set `backend: vllm` in config.",
                self._backend)
        return jobid

    def render_architecture(self, out_path: Path | str | None = None) -> Path:
        """Render this run's agent graph — nodes, roles, tools, descriptions,
        edges — as a self-contained SVG and write it to disk.

        Works before or after ``execute()`` (the graph is fixed at
        construction). Default location: ``run_dir/debug/architecture.svg``
        once a run has started (``self._run_dir`` set), else
        ``study_dir/architecture.svg``. Returns the path written.
        """
        from .run_diagram import render_architecture_svg

        svg = render_architecture_svg(
            self._graph_spec,
            model=self._model,
            backend=self._backend,
            study_dir=self.study_dir,
        )
        if out_path is None:
            base = (
                self._run_dir / "debug" if self._run_dir is not None
                else self.study_dir
            )
            out_path = base / "architecture.svg"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(svg, encoding="utf-8")
        return out_path

    def serve_viewer(
        self, host: str = "127.0.0.1", port: int = 8765,
    ) -> None:
        """Launch the read-only live web viewer for this study's runs
        (blocking — run in a separate terminal/process from ``execute()``,
        the same way ``render_architecture()`` is a separate opt-in step,
        never called automatically). Requires the ``viewer``
        optional-dependency group (``pip install a3dasm[viewer]``); lazily
        imported here so the core package never depends on Starlette.

        Passes this run's own live ``Graph`` object (``self._graph_spec``)
        straight through — no reconstruction needed, unlike the standalone
        ``python -m a3dasm.viewer <study-dir>`` CLI, which runs as a
        separate process with no in-memory Graph and falls back to
        recovering one from the study's own ``run.py``/``build_graph()`` (or
        the stock default graph) instead.
        """
        from ..viewer.app import run_viewer

        run_viewer(
            self.study_dir, host=host, port=port, graph=self._graph_spec)

    def execute(self) -> str:
        """Run the agentic loop; return the final report text.

        Reads ``PROBLEM_STATEMENT.md`` from the study directory and passes it
        as the initial user message to the entry node.
        """
        if getattr(self, "_container", False):
            runner = ContainerRunner(
                self.study_dir,
                model=self._model,
                budget=getattr(self, "_budget", None),
                backend=getattr(self, "_backend", "claude"),
                image=getattr(self, "_container_image", "f3dasm-agentic:latest"),
            )
            exit_code = runner.run()
            if exit_code != 0:
                raise AgenticRunError(f"Container exited with code {exit_code}")
            return runner._latest_solution()

        problem_path = self.study_dir / "PROBLEM_STATEMENT.md"
        if not problem_path.exists():
            raise AgenticRunError(
                f"PROBLEM_STATEMENT.md not found in {self.study_dir}"
            )
        problem = problem_path.read_text(encoding="utf-8")

        # Create run directory (or reuse an existing one when resuming).
        # getattr default: some tests build AgenticRun via __new__.
        _resume = getattr(self, "_resume_from", None)
        if _resume is not None:
            run_dir = _resume.resolve()
            if not (run_dir / "debug" / "thread_id").exists():
                raise AgenticRunError(
                    f"resume_from={run_dir} is not a resumable run dir "
                    "(no debug/thread_id)"
                )
            ts = run_dir.name
        else:
            ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
            run_dir = self.study_dir / "runs" / ts
            _archive_prior_pipeline_notebook(self.study_dir)
        debug_dir = run_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot the EXACT PROBLEM_STATEMENT.md this run read, verbatim —
        # the live study_dir/PROBLEM_STATEMENT.md drifts between runs (a study
        # is meant to be run once per statement; when it isn't, a human
        # auditing a later run needs to see the statement THAT run actually
        # answered, not whatever the file has since been edited to say). The
        # notebook's Run metadata cell carries only the hash so a reader can
        # confirm which snapshot matches; the snapshot is the recoverable copy.
        # write-if-absent: a resumed run must keep its ORIGINAL snapshot, not
        # overwrite it with whatever PROBLEM_STATEMENT.md says at resume time.
        # The hash is derived from the SNAPSHOT's content, never re-read from
        # the live file, so a resume's stamp always matches what this run
        # actually answered even if PROBLEM_STATEMENT.md has since drifted.
        _ps_snapshot_path = debug_dir / "PROBLEM_STATEMENT_snapshot.md"
        if not _ps_snapshot_path.exists():
            _ps_snapshot_path.write_text(problem, encoding="utf-8")
        import hashlib
        _problem_statement_sha256 = hashlib.sha256(
            _ps_snapshot_path.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        # Captured BEFORE the constraint-snapshot preamble (budgets, elapsed
        # time — always different between runs) gets prepended to `problem`
        # below, so a resume's "did PROBLEM_STATEMENT.md change" check
        # compares the same kind of content on both sides instead of always
        # reporting "changed" (BACKLOG #35).
        _live_problem_sha256 = hashlib.sha256(
            problem.encode("utf-8")
        ).hexdigest()
        notes_dir = debug_dir / "strategizer_notes"
        notes_dir.mkdir(parents=True, exist_ok=True)
        self._run_dir = run_dir

        # Canonical store: experiment_data/ + run_config.json
        _full_cfg = _load_study_config(self.study_dir)
        _eval_cfg = _full_cfg.get("evaluator")
        canonical_cfg = _init_canonical_store(
            run_dir, self.study_dir, evaluator_config=_eval_cfg,
            eval_budget=getattr(self, "_eval_budget", None),
            mem_cap_bytes=getattr(self, "_mem_cap_bytes", None),
        )

        # Ingest a precomputed pool as D000 ground-truth rows. Two sources,
        # one ingestion path — D000 rows are never counted as evaluations
        # (_resolve_delegation_evals runs only for real delegations D001+):
        #   evaluator.lookup.pool  → pool IS the oracle (queried via
        #                            LookupDataGenerator) AND training data.
        #   training_data          → pool is ONLY training data; there is NO
        #                            live oracle (e.g. surrogate-only studies
        #                            where new evaluations cannot be run).
        _lookup_cfg = (_eval_cfg or {}).get("lookup")
        _training_data = _full_cfg.get("training_data")
        _pool_cfg = _lookup_cfg or (
            {"pool": _training_data} if _training_data else None
        )
        if _pool_cfg:
            _store_dir = Path(canonical_cfg["store_dir"])
            try:
                _n_ingested = _ingest_precomputed_pool(
                    _store_dir, self.study_dir, _pool_cfg
                )
                log_ingested = _n_ingested  # captured for log below
            except Exception as _exc:  # noqa: BLE001
                log_ingested = None
                _exc_msg = str(_exc)
        else:
            log_ingested = None
            _exc_msg = None

        # Set up run.log
        log = logging.getLogger(f"a3dasm.{ts}")
        log.setLevel(logging.INFO)
        handler = logging.FileHandler(debug_dir / "run.log")
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        log.addHandler(handler)
        log.info(f"Run starting: model={self._model}, study={self.study_dir}")
        if log_ingested is not None:
            log.info(
                f"D000: ingested {log_ingested} precomputed pool rows"
                f" from {_pool_cfg.get('pool', '?')}"
            )
        elif _exc_msg is not None:
            log.warning(f"D000 pool ingest failed: {_exc_msg}")

        # Wall-clock anchor, persisted in its OWN file for exactly the reason
        # thread_id is (run_config.json is rewritten mid-run, so it cannot
        # carry a start time). A resume MUST charge the wall time the run has
        # already spent: re-anchoring to now makes every budget check, every
        # constraint snapshot and the critic's run-adequacy judgement restart
        # from zero, so a run that has been going for a day reports hours.
        _start_path = debug_dir / "run_started_at"
        start_time = None
        if _resume is not None:
            try:
                start_time = float(_start_path.read_text().strip())
            except (OSError, ValueError):
                start_time = None          # unreadable: fall back to now
        if start_time is None:
            start_time = time.time()
            try:
                _start_path.write_text(repr(start_time))
            except OSError:
                pass                        # anchor is best-effort, never fatal

        # Create graph-wide delegation log for episodic memory.
        delegation_log_path = debug_dir / "delegation_log.jsonl"
        delegation_log = DelegationLog(delegation_log_path)

        # workspace_dir for worker delegations
        workspace_dir = debug_dir / "delegations"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Stable thread_id: persisted so a crashed run can be resumed against
        # the same LangGraph checkpoint. Resume reads it back; a fresh run
        # mints and stores it (own file: run_config.json is rewritten).
        _tid_path = debug_dir / "thread_id"
        if _resume is not None:
            thread_id = _tid_path.read_text().strip()
        else:
            thread_id = str(uuid.uuid4())
            _tid_path.write_text(thread_id)

        # Pre-run problem-statement review (advisory; interactive-refine when
        # enabled). Fresh runs only — a resume replays the checkpoint and must
        # not re-prompt. Skipped when a graph is injected programmatically
        # (a test affordance — _run_dir is set above so _make_adapter can build
        # the ephemeral reviewer session on real runs).
        if (
            _resume is None
            and getattr(self, "_review_statement", True)
            and getattr(self, "_graph", None) is None
        ):
            problem = self._review_problem_statement(problem, debug_dir)

        # Human -> strategizer is a delegation like any other (strategizer ->
        # worker, strategizer -> critic) — same constraint snapshot, single
        # source of truth (constraint_snapshot.py), so the very first message
        # the strategizer reads already states the budgets as facts instead
        # of leaving them latent in AgenticState (present to the node's
        # Python code, never rendered into words the model actually sees).
        from ..runtime.constraint_snapshot import (
            compute_constraint_snapshot,
        )
        _initial_snapshot = compute_constraint_snapshot(
            eval_budget=getattr(self, "_eval_budget", None),
            budget_seconds=getattr(self, "_budget", None),
            run_start=start_time,
            experiment_data_dir=run_dir / "experiment_data",
        )
        problem = _initial_snapshot.as_text() + "\n\n" + problem

        initial_state = AgenticState(
            messages=[HumanMessage(content=problem)],
            study_dir=str(self.study_dir),
            done=False,
            last_report=None,
            total_delegations=0,
            budget_seconds=getattr(self, "_budget", None),
            budget_usd=getattr(self, "_budget_usd", None),
            run_dir=str(run_dir),
            eval_budget=getattr(self, "_eval_budget", None),
            evals_used=0,
            start_time=start_time,
            return_to=None,
            required_deliverables=(
                getattr(self, "_required_deliverables", None) or None
            ),
            experiment_data_dir=canonical_cfg["store_dir"],
        )

        config: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            # 2000 ≈ hundreds of delegations; the old 500 (and the legacy 25 on
            # some branches) could crash a long multi-delegation run mid-flight
            # (GraphRecursionError). Knob: recursion_limit (config.yaml runtime
            # block; F3DASM_RECURSION_LIMIT overrides).
            "recursion_limit": settings.get_int("recursion_limit", 2000),
        }

        # Durable checkpoint to disk so the run survives a crash; resume passes
        # None as input (LangGraph convention: replay from last checkpoint).
        from langgraph.checkpoint.sqlite import SqliteSaver
        ckpt_path = debug_dir / "checkpoints.sqlite"
        log.info("Invoking graph")
        # Optionally own a vLLM server on a SLURM GPU node for this run; the
        # jobid is torn down in the finally on EVERY exit path (normal close,
        # crash, KeyboardInterrupt) so a killed run never leaks a GPU
        # allocation. None when llm_slurm is disabled — the common path.
        _serve_jobid = None
        try:
            _serve_jobid = self._maybe_start_slurm_llm(
                _full_cfg, debug_dir, log)
            with SqliteSaver.from_conn_string(str(ckpt_path)) as saver:
                graph = getattr(self, "_graph", None) or build_graph(
                    self._graph_spec, self._make_adapter,
                    study_dir=self.study_dir,
                    interactive=self._interactive, max_ask=self._max_ask,
                    notes_dir=notes_dir,
                    workspace_dir=workspace_dir,
                    delegation_log=delegation_log,
                    checkpointer=saver,
                )
                graph_input = None if _resume is not None else initial_state
                # A run that reached a terminal Command(goto=END) — i.e. EVERY
                # normal close (GATED/UNGATED/FAILED all go through the same
                # terminal branch in strategizer.py) — leaves the checkpoint
                # with an empty `.next`. LangGraph's invoke(None, config) on
                # such a checkpoint is a genuine no-op: no node re-runs, no
                # new model call happens, it just hands back the stale
                # last_report verbatim (confirmed empirically: a minimal
                # StateGraph reproduction showed the node's own call counter
                # never incremented on a second invoke(None) against an
                # already-END'd thread). BACKLOG #34's resume_from guidance
                # was silently useless for exactly the runs it targeted
                # (externally-stopped, therefore terminal) until this fix —
                # found because a "resumed" run replayed 19-hour-old cached
                # text and was mistaken for a live re-test of the same stop
                # condition (BACKLOG #35).
                #
                # Only a genuinely mid-flight interruption (crash, kill —
                # `.next` non-empty, real pending tasks) should still use the
                # plain invoke(None) replay-from-checkpoint path. A terminal
                # checkpoint needs FRESH input to force real re-execution from
                # the entry node (confirmed empirically too: invoke() with new
                # non-None input on an already-terminal thread DOES re-run the
                # node). The one case explicitly NOT worth resuming: the run
                # already closed cleanly (GATED, an accepted Done()) and
                # PROBLEM_STATEMENT.md hasn't changed since — there is
                # nothing new to do, so this refuses loudly rather than
                # silently no-op or silently redo finished work.
                if _resume is not None and hasattr(graph, "get_state"):
                    _resume_state = graph.get_state(config)
                    if not _resume_state.next:  # terminal: reached END
                        _prior_status: dict = {}
                        try:
                            _prior_status = json.loads(
                                (debug_dir / "run_status.json")
                                .read_text(encoding="utf-8")
                            )
                        except (OSError, json.JSONDecodeError):
                            pass
                        _resume_ps_changed = (
                            _live_problem_sha256 != _problem_statement_sha256
                        )
                        if (
                            _prior_status.get("status") == "GATED"
                            and not _resume_ps_changed
                        ):
                            raise AgenticRunError(
                                f"resume_from={run_dir} closed cleanly "
                                "(GATED, an accepted Done()) and "
                                "PROBLEM_STATEMENT.md is unchanged since — "
                                "there is nothing new for this run to do. "
                                "Resume is for a run that was interrupted or "
                                "stopped short of a real close; edit "
                                "PROBLEM_STATEMENT.md first if you want it "
                                "reconsidered, or start a fresh run instead."
                            )
                        _reason_bits = []
                        if _prior_status.get("stop_reason"):
                            _reason_bits.append(
                                "it was stopped by an external cause "
                                f"({_prior_status['stop_reason']}), not by "
                                "its own choice"
                            )
                        elif _prior_status.get("status", "GATED") != "GATED":
                            _reason_bits.append(
                                f"it closed {_prior_status['status']} "
                                "without an accepted Done()"
                            )
                        if _resume_ps_changed:
                            _reason_bits.append(
                                "PROBLEM_STATEMENT.md has been edited since "
                                "this run's original snapshot — the current "
                                "text follows below"
                            )
                        _resume_note = (
                            "[RESUME] This run previously closed, but "
                            + "; and ".join(
                                _reason_bits or ["you asked to resume it"]
                            )
                            + ". Continue using the accumulated conversation"
                            " history above — do not restart from scratch."
                        )
                        if _resume_ps_changed:
                            _resume_note += (
                                f"\n\nCurrent PROBLEM_STATEMENT.md:\n\n"
                                f"{problem}"
                            )
                        graph_input = {
                            "messages": [HumanMessage(content=_resume_note)],
                            "done": False,
                        }
                # On resume, the checkpointed state still carries the OLD
                # budgets and start_time. Re-seed them from this AgenticRun so a
                # run that halted on a budget can actually make progress after
                # the user raises it (cumulative token_totals persist in the
                # checkpoint, so the spend-so-far is still counted against the
                # new ceiling).
                if _resume is not None and hasattr(graph, "update_state"):
                    try:
                        graph.update_state(config, {
                            "budget_seconds": getattr(self, "_budget", None),
                            "budget_usd": getattr(self, "_budget_usd", None),
                            "eval_budget": getattr(self, "_eval_budget", None),
                            "start_time": start_time,
                        })
                    except Exception:  # noqa: BLE001
                        log.warning("resume state refresh failed",
                                    exc_info=True)
                try:
                    result = graph.invoke(graph_input, config=config)
                except BaseException as _exc:  # noqa: BLE001
                    # Any unhandled crash (GraphRecursionError,
                    # KeyboardInterrupt, OOM, …): record a resumable status so
                    # resume_from is always an option after a break, then
                    # re-raise (we do not swallow).
                    self._write_run_status(
                        debug_dir, status="crashed",
                        reason=f"{type(_exc).__name__}: {_exc}"[:500],
                        resumable=True, thread_id=thread_id,
                        # A crashed run's duration is exactly what the next
                        # resume needs to charge, so record it here too.
                        wall_s=round(time.time() - start_time, 1),
                    )
                    raise
        finally:
            if _serve_jobid:
                from ..infra.slurm_llm import cancel_job
                try:
                    cancel_job(_serve_jobid)
                    log.info("llm_slurm: scancel'd serve job %s", _serve_jobid)
                except Exception:  # noqa: BLE001
                    log.warning("llm_slurm: teardown failed", exc_info=True)
        # Merge per-call telemetry into an analysis-ready summary.json (additive,
        # off the decision path — a failure here must not fail the run).
        try:
            from ..infra.telemetry import Telemetry
            Telemetry.merge(debug_dir)
        except Exception:  # noqa: BLE001
            log.warning("telemetry merge failed", exc_info=True)
        report = result.get("last_report") or ""
        # Gate outcome, persisted so the ledger reports the TRUE terminal state.
        # The strategizer prepends a "⚠ UNGATED RUN" / FAILED banner to
        # last_report when the run closed without an accepted (critic-PASS)
        # Done(); a notebook-deliverable study has no solution.md for the ledger
        # to read this from, so without persisting it here every stamped notebook
        # reads as GATED — masking 3-strike and failed closes. (audit 20260622)
        if ("⛔" in report) or ("FAILED RUN" in report):
            _gate_outcome = "FAILED"
        elif ("⚠ UNGATED RUN" in report) or ("NOT validated" in report):
            _gate_outcome = "UNGATED"
        else:
            _gate_outcome = "GATED"
        _stop_reason = next(
            (name for name, sig in _EXTERNAL_STOP_SIGNATURES.items()
             if sig in report),
            None,
        )
        if _stop_reason is not None:
            log.warning(
                "Run stopped by an external cause (%s), not a normal close "
                "— resume it once the cause clears:\n"
                "    from a3dasm import AgenticRun\n"
                "    AgenticRun(study_dir=%r, graph=build_graph(),\n"
                "               interactive=False,\n"
                "               resume_from=%r).execute()",
                _stop_reason, str(self.study_dir), str(run_dir),
            )
        # Authoritative eval count = provenance-stamped rows in the canonical
        # ledger, NOT the run-state counter. evals_used is summed from a
        # registry that clears Done entries on loop-back, so it under-reports
        # (0) on any run that re-prompts (e.g. every UNGATED run). The ledger
        # never loses rows — and it also captures cancelled-but-completed
        # delegations whose evals are real.
        evals = result.get("evals_used", 0)
        try:
            # Sum across the canonical store AND every design namespace (Axis 3a):
            # namespace evals live in sibling stores the canonical-only count
            # missed (run 20260627T013812 reported 100 while 200 real evals ran).
            from ..evaluation.ledger_summary import total_ledgered_evals
            _total = total_ledgered_evals(debug_dir.parent / "experiment_data")
            if _total:
                evals = _total
        except Exception:  # noqa: BLE001
            log.warning("ledger eval-count failed; using state counter",
                        exc_info=True)
        tokens = result.get("token_totals") or {}
        error_counts = result.get("error_counts") or {}

        now_ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        elapsed = time.time() - start_time
        h, m, s = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)

        tokens_in = tokens.get("input_tokens", 0) or 0
        tokens_out = tokens.get("output_tokens", 0) or 0
        cache_read = tokens.get("cache_read_input_tokens", 0) or 0
        cache_create = tokens.get("cache_creation_input_tokens", 0) or 0
        cost = tokens.get("total_cost_usd")
        cost_str = f"${cost:.4f}" if cost is not None else "n/a"

        # Run metadata + token table — provenance appended to the deliverable.
        meta_md = (
            f"## Run metadata\n\n"
            f"- timestamp: {now_ts}\n"
            f"- model: {self._model}\n"
            f"- gate: {_gate_outcome}\n"
            f"- total_delegations: {len(delegation_log.query_all())}\n"
            f"- evals_used: {evals}\n"
            f"- run_dir: {run_dir}\n"
            f"- time_used: {h:02d}:{m:02d}:{s:02d}\n"
            f"- problem_statement_sha256: {_problem_statement_sha256}\n"
            f"  (verbatim snapshot: {debug_dir}/PROBLEM_STATEMENT_snapshot.md — "
            f"study_dir/PROBLEM_STATEMENT.md may since have been edited)\n\n"
            f"## Token usage\n\n"
            f"| Metric | Value |\n"
            f"|--------|-------|\n"
            f"| input_tokens | {tokens_in:,} |\n"
            f"| output_tokens | {tokens_out:,} |\n"
            f"| cache_read_tokens | {cache_read:,} |\n"
            f"| cache_creation_tokens | {cache_create:,} |\n"
            f"| total_tokens | {tokens_in + tokens_out:,} |\n"
            f"| estimated_cost | {cost_str} |\n"
            + (
                "\n## Tool-call errors per node\n\n"
                + "| node | error_count |\n"
                + "|------|-------------|\n"
                + "".join(
                    f"| {node} | {count} |\n"
                    for node, count in sorted(error_counts.items())
                )
                if error_counts else ""
            )
        )

        # The agent-authored pipeline.ipynb IS the deliverable (its leading
        # markdown cells hold the writeup). There is no solution.md — stamp run
        # provenance as a trailing metadata cell + notebook metadata.
        nb_path = self.study_dir / "pipeline.ipynb"
        if nb_path.exists():
            try:
                import nbformat

                from ..evaluation.notebook_exec import (
                    repair_code_cells,
                    stamp_run_provenance,
                )
                nb = nbformat.read(str(nb_path), as_version=4)
                repair_code_cells(nb)
                # Replace (not append) the provenance cell — the notebook is
                # study-scoped and persists across runs; appending accumulated
                # a prior run's stale metadata cell.
                stamp_run_provenance(nb, meta_md)
                nb.metadata.setdefault("agentic", {}).update(
                    {"model": self._model, "run": str(run_dir),
                     "timestamp": now_ts, "gate_outcome": _gate_outcome,
                     "problem_statement_sha256": _problem_statement_sha256})
                nbformat.write(nb, str(nb_path))
            except Exception:  # noqa: BLE001
                log.warning("notebook provenance stamp failed", exc_info=True)

        # Persist the terminal gate outcome to run_status.json on the NORMAL
        # close too (the crash path above writes its own). Without this a
        # cleanly-closed run leaves no run_status.json and the §1 protocol's
        # first KPI (gate outcome) is unreadable — the outcome would live only in
        # the notebook metadata + the ledger. (audit: 3 GATED runs, none had it.)
        self._write_run_status(
            debug_dir, status=_gate_outcome, model=self._model,
            evals_used=evals, timestamp=now_ts, run=str(run_dir),
            thread_id=thread_id, stop_reason=_stop_reason,
            # The §1 KPI table asks for wall clock and this file is what it
            # reads first; without it every consumer re-derives the duration
            # from file mtimes and gets a different answer.
            wall_s=round(elapsed, 1),
        )

        # Append a KPI row to the longitudinal ledger automatically (best
        # effort). The extraction logic lives in studies/run_ledger.py (the one
        # source of truth, writing studies/run_ledger.csv); we invoke it as a
        # subprocess when present so a run is always recorded without a manual
        # step. Absent (e.g. a non-studies install) → silently skipped.
        try:
            import subprocess
            import sys as _sys
            ledger_script = self.study_dir.parent / "run_ledger.py"
            if ledger_script.exists():
                proc = subprocess.run(
                    [_sys.executable, str(ledger_script), str(run_dir)],
                    capture_output=True, text=True, timeout=60,
                )
                if proc.returncode == 0:
                    log.info("KPI ledger: %s", proc.stdout.strip())
                else:
                    log.warning(
                        "KPI ledger append failed (rc=%s): %s",
                        proc.returncode, proc.stderr.strip())
        except Exception:
            log.warning("KPI ledger append errored", exc_info=True)

        log.info(
            f"Run complete. Evals: {evals}. "
            f"Tokens in/out: {tokens_in}/{tokens_out}. "
            f"Cost: {cost_str}. "
            + ("pipeline.ipynb stamped." if nb_path.exists()
               else "pipeline.ipynb was NEVER WRITTEN — the agent never "
               "called WriteDeliverable().")
        )
        log.removeHandler(handler)
        handler.close()

        return report

    def _review_problem_statement(
        self, problem: str, debug_dir: Path, *, adapter=None
    ) -> str:
        """Advisory pre-run well-posedness review (Item B).

        Always writes ``debug/problem_statement_review.md``.  When the run is
        interactive and gaps are found, offers a per-gap refine via the same
        ``input()`` channel the in-graph FollowUp uses, appending accepted
        clarifications to the statement (and to a saved addendum).  Returns the
        (possibly augmented) problem text.  NEVER blocks an autonomous run: any
        reviewer failure falls back to the original statement unchanged.
        """
        from ..epistemics.reviewer import (
            ProblemStatementReviewerAgent,
            format_review_markdown,
            parse_review,
            review_gaps,
        )

        try:
            if adapter is None:
                adapter = self._make_adapter(
                    "problem_statement_reviewer",
                    ProblemStatementReviewerAgent(),
                )
            raw = adapter.invoke([{"role": "user", "content": problem}])
            review = parse_review(raw)
        except Exception:  # noqa: BLE001 — advisory, never blocks
            return problem

        try:
            (debug_dir / "problem_statement_review.md").write_text(
                format_review_markdown(review, problem), encoding="utf-8"
            )
        except OSError:
            pass

        gaps = review_gaps(review)
        if not (gaps and self._interactive):
            return problem

        clarifications: list[tuple[str, str]] = []
        print(
            "\nThe problem statement may be under-specified. For each gap, "
            "type a clarification (or leave blank to skip):"
        )
        for g in gaps:
            try:
                ans = input(f"  [{g['element']}] {g['note']}\n  > ").strip()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            if ans:
                clarifications.append((g["element"], ans))

        if not clarifications:
            return problem

        addendum = "\n\n## Clarifications (added pre-run via HITL review)\n" + (
            "\n".join(f"- **{el}**: {ans}" for el, ans in clarifications)
        )
        try:
            (debug_dir / "problem_statement_addendum.md").write_text(
                addendum.strip() + "\n", encoding="utf-8"
            )
        except OSError:
            pass
        return problem + addendum

    def _resource_stanza(self, run_dir, *, for_worker: bool) -> str:
        """The static resource-envelope stanza — "what you HAVE" — injected into a
        worker/strategizer preamble at delegation start, so the agent stops
        running-and-hoping. O(1): cpu_count + one statvfs (no directory walk).
        Empty string on any failure (never fatal).

        ROLE-AWARE parallelism (deliberate): the cores/RAM/disk facts are shared,
        but only the WORKER is primed to parallelize — and only its EVALUATIONS
        within a campaign (compute speedup, same experiment/budget, epistemically
        neutral). The strategizer is NOT resource-nudged to fan out experiments:
        running multiple arms concurrently is an experimental-design decision with
        epistemic weight (budget splits, comparison validity) that lives in its own
        guidance — resource-priming it nudges breadth over disciplined comparison
        (observed run 20260628T224159: a 3-arm, unequal-budget, INCONCLUSIVE run)."""
        try:
            from ..infra.watchdog_cleanup import resource_envelope
            env = resource_envelope(run_dir or self.study_dir, self._mem_cap_bytes)
            cores = env["cores"]
            ram = (f"{env['ram_cap_bytes'] / 1024 ** 3:.1f} GB"
                   if env["ram_cap_bytes"] else "unset")
            disk = (f"{env['disk_free_bytes'] / 1024 ** 3:.0f} GB"
                    if env["disk_free_bytes"] is not None else "unknown")
            facts = (
                "resources: "
                f"~{cores} CPU cores · RAM cap {ram} per delegation (HARD — exceed "
                "it and your process is KILLED; stream/cache large data, don't load "
                f"it all at once) · disk free {disk}.\n"
            )
            if for_worker:
                facts += (
                    "Use the cores: parallelize the EVALUATIONS within your "
                    "campaign (e.g. gen.call(mode='parallel'), or concurrent "
                    "candidate evaluations) to finish faster — same experiment, "
                    "just quicker. Size concurrency to the RAM cap.\n"
                )
            # Non-campaign roles (strategizer, critic, datagenerator, literature)
            # get the facts only — NO parallelism imperative. Fanning out
            # experiments is the strategizer's design call (KB 0004: one
            # delegation = one experiment), not something to resource-nudge.
            return facts
        except Exception:  # noqa: BLE001 — telemetry must never break a run
            return ""

    def _kb_menu(self, role) -> str:
        """Audience-filtered handbook MENU injected at the head of an agent's
        prompt — so it always SEES the latent knowledge it can pull (mirroring
        how it always sees its tool list), instead of only discovering a chapter
        if it already thought to call ConsultHandbook. Cached; empty on failure."""
        try:
            if getattr(self, "_kb", None) is None:
                from ..knowledge import KnowledgeBase
                self._kb = KnowledgeBase.load()
            return self._kb.menu(audience=role)
        except Exception:  # noqa: BLE001 — a missing menu must never break a run
            return ""

    def _make_adapter(self, name: str, agent: Agent):
        run_dir = self._run_dir
        _role = getattr(agent, "role", None)

        # The run-aware cwd=study_dir + full <run_paths> preamble is for the
        # graph's ENTRY/orchestrator node ONLY — it alone needs full-repo
        # visibility and doesn't itself write delegation-scoped worker files.
        # This used to key off "has ANY outgoing edge", which also matched
        # datagenerator/implementer (each has its own edge to
        # literature_reviewer, for sub-delegating a lookup — see _graphs.py)
        # even though both are sandboxed WORKERS everywhere else in their
        # contract (Delegate's own docstring: "writes exclusively to {id}/
        # relative to their workspace in debug/delegations/"). That mismatch
        # split delegation output across TWO physical trees for these two
        # roles — study_dir/debug/delegations/D### (this cwd) vs the
        # run-scoped run_dir/debug/delegations/D### their own preamble
        # promised — different inodes, same D### ids, no single source of
        # truth (run 20260718T132852, D010's retrospective: "cost an extra
        # stat/inode-comparison round-trip to notice").
        is_entry = run_dir and name == getattr(self._graph_spec, "entry", None)
        if is_entry:
            notes_dir = Path(run_dir) / "debug" / "strategizer_notes"
            debug_dir = Path(run_dir) / "debug"
            preamble = RUN_PATHS_PREAMBLE_TEMPLATE.format(
                study_dir=self.study_dir,
                run_dir=run_dir,
                debug_dir=debug_dir,
                notes_dir=notes_dir,
                experiment_data_dir=Path(run_dir) / "experiment_data",
                resources=self._resource_stanza(run_dir, for_worker=False),
                knowledge=self._kb_menu(_role),
            )
            system_prompt = preamble + agent.system_prompt
            cwd = self.study_dir
        else:
            if run_dir is not None:
                workspace_dir = Path(run_dir) / "debug" / "delegations"
                workspace_dir.mkdir(parents=True, exist_ok=True)
            else:
                workspace_dir = self.study_dir
            # Only the implementer runs evaluation campaigns → only it gets the
            # eval-parallelism nudge; the critic/datagenerator/literature get the
            # resource facts alone.
            _is_campaign = getattr(agent, "role", None) == "implementer"
            preamble = WORKSPACE_PREAMBLE_TEMPLATE.format(
                workspace_dir=workspace_dir,
                study_dir=self.study_dir,
                resources=self._resource_stanza(run_dir, for_worker=_is_campaign),
                knowledge=self._kb_menu(_role),
            )
            system_prompt = preamble + agent.system_prompt
            # Critics read from the study tree, not from a delegation subfolder.
            if getattr(agent, "role", None) == "critic":
                cwd = self.study_dir
            else:
                cwd = workspace_dir

        # The deliverable is pipeline.ipynb. Inject its contract — ROLE-AWARE:
        # only the strategizer authors it (it alone has the notebook tools); the
        # implementer/critic get the same structure framed for their job (fit
        # your code to it / judge against it), never an "author it" imperative.
        # Gated on pipeline_deliverable (default True): its injected text is an
        # unconditional imperative ("this SUPERSEDES every ... instruction
        # above") that previously overrode even a PROBLEM_STATEMENT.md saying
        # there is no pipeline deliverable (BACKLOG #27) — a study that
        # explicitly turns this off has no notebook contract to inject at all.
        from ..evaluation.notebook_exec import notebook_deliverable_spec
        _role = getattr(agent, "role", None)
        if _role in ("strategizer", "implementer", "critic") and settings.get_bool(
            "pipeline_deliverable", True
        ):
            system_prompt = system_prompt + notebook_deliverable_spec(_role)

        model = agent.model or self._model
        backend = agent.backend or self._backend

        _persistent = not agent.reset_on_checkpoint
        _max_history_pairs = getattr(agent, "max_history_pairs", 5)

        # Study-scoped, NOT per-run: a study is typically run many times
        # (the same domain, evolving problem statement), and re-downloading
        # + re-embedding the same papers every run is pure waste with no
        # corresponding staleness risk — unlike cross-run FINDINGS/hypothesis
        # memory (rejected earlier as too risky, since a study's actual
        # scientific question genuinely can drift run to run), a paper's
        # relevance to a domain does not. Lives under runs/ (not the study
        # root) so it stays out of the user-facing study folder alongside
        # PROBLEM_STATEMENT.md/config.yaml/pipeline.ipynb — see
        # LiteratureCorpus for the cross-process FileLock this now requires
        # (two runs of the same study can genuinely overlap and both write).
        lit_reviewer_notes_dir = self.study_dir / "runs" / "lit_reviewer_notes"

        # Registry-driven, forward-compatible dispatch: resolve the adapter
        # class by backend name and let it choose its own native tools. Adding
        # a backend to backends/registry.py makes it dispatchable here with no
        # change to this method. Backend-specific endpoint/auth (base_url,
        # api_key) is resolved inside each adapter from env/defaults, so the
        # construction kwargs are common to every backend.
        from ..backends.registry import get_adapter_class

        adapter_cls = get_adapter_class(backend)
        native = adapter_cls.select_native_tools(agent.tools)

        _mcp = dict(getattr(agent, "mcp_servers", {}))
        _allowed = list(getattr(agent, "extra_allowed_tools", frozenset()))

        adapter = adapter_cls(
            model=model,
            system_prompt=system_prompt,
            study_dir=cwd,
            native_tools=native,
            extra_mcp_servers=_mcp,
            extra_allowed_tools=_allowed,
            persistent=_persistent,
            max_history_pairs=_max_history_pairs,
        )
        # Universal read-only handbook lookup: EVERY node's adapter gets it
        # here, equally, at construction (copy() returns self, so the
        # per-invocation worker/critic paths inherit it). Single injection
        # point — do not duplicate it per path. The tool's description is owned
        # by _consult_handbook's docstring (the backend infers the schema from
        # the callable).
        from ..nodes.parsing import _consult_handbook
        adapter.closure_tools["ConsultHandbook"] = _consult_handbook

        extra_closures = agent.build_closure_tools(
            self.study_dir,
            lit_reviewer_notes_dir=lit_reviewer_notes_dir,
        )
        if extra_closures:
            adapter.closure_tools.update(extra_closures)
        return adapter
