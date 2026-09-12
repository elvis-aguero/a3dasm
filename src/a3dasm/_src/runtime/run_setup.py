"""Pre-run setup for an agentic run: everything that happens before the graph.

:class:`.agent_runtime.AgenticRun` calls these in order while preparing a
run directory — resolve the hard per-delegation memory cap, read the
study's ``config.yaml``, archive the previous ``pipeline.ipynb``, create
the canonical experiment store and register the oracle entry point,
ingest any precomputed evaluation pool, parse the wall-clock budget. The
delegation tool re-enters :func:`register_evaluator_entrypoint` mid-run when
an implementer registers a generator.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

import yaml  # available via hydra-core

from ..evaluation._f3dasm_compat import PROTECTED_STORE_SENTINEL

__all__ = [
    "DEFAULT_MEM_CAP_BYTES",
    "AgenticRunError",
    "register_evaluator_entrypoint",
    "resolve_mem_cap_bytes",
]

# Default per-campaign-process hard memory cap (bytes) — the one HARD resource
# boundary. 4 GiB: comfortably above a healthy GP-BO campaign, below the runaway
# GP-on-5302-points blowup that pegged the host. See resolve_mem_cap_bytes for
# the resolution order (config -> env -> SLURM allocation -> this default).
DEFAULT_MEM_CAP_BYTES = 4 * 1024 ** 3


def resolve_mem_cap_bytes(explicit, env=None) -> int:
    """Resolve the hard per-delegation RAM cap (bytes).

    Precedence: an explicit config.yaml ``mem_cap`` > env ``F3DASM_MEM_CAP`` >
    the SLURM job's memory allocation > ``DEFAULT_MEM_CAP_BYTES``. The SLURM
    step matters on real HPC: SLURM already gives the job a memory allocation,
    so without this the watchdog kept a small hardcoded ceiling and silently
    throttled worker concurrency far below what the node actually granted. This
    only sets the cap's VALUE — it is still the one hard host-safety cap.
    """
    import os as _os
    env = _os.environ if env is None else env

    def _as_int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    if (v := _as_int(explicit)) and v > 0:
        return v
    if (v := _as_int(env.get("F3DASM_MEM_CAP"))) and v > 0:
        return v
    # SLURM reports memory in MB. Prefer the per-node allocation; otherwise
    # derive it from per-CPU * CPUs (SLURM_CPUS_ON_NODE may be "16" or "16(x2)").
    mb = _as_int(env.get("SLURM_MEM_PER_NODE"))
    if not mb:
        per_cpu = _as_int(env.get("SLURM_MEM_PER_CPU"))
        cpus_raw = env.get("SLURM_CPUS_ON_NODE") or env.get(
            "SLURM_JOB_CPUS_PER_NODE") or ""
        cpus = _as_int(str(cpus_raw).split("(")[0]) if cpus_raw else None
        if per_cpu and cpus:
            mb = per_cpu * cpus
    if mb and mb > 0:
        return mb * 1024 * 1024
    return DEFAULT_MEM_CAP_BYTES


class AgenticRunError(Exception):
    """Raised when an agentic run fails unrecoverably."""


def _load_study_config(study_dir: Path) -> dict:
    """Read study_dir/config.yaml if present; return empty dict otherwise."""
    cfg_path = study_dir / "config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _archive_prior_pipeline_notebook(study_dir: Path) -> None:
    """Archive (never delete) a prior run's leftover study_dir/pipeline.ipynb
    before a FRESH run starts.

    pipeline.ipynb is study-scoped, not run-scoped: _load_or_new_notebook
    (routing.py) loads it wholesale if one exists, and _emit_notebook
    preserves any cells not touched this run as trailing "extras" — so a
    prior run's stale cells/imports otherwise leak into a new run's
    deliverable. Only called on the non-resume path (a resumed run must keep
    working on the SAME notebook, which is the whole point of resume).

    Renamed to pipeline_<run_id>.ipynb, keyed by the run id the archived
    notebook actually belongs to (read from its own provenance stamp,
    stamp_run_provenance's nb.metadata["agentic"]["run"]) so the archive is
    traceable to the run that produced it, not the run that's starting.
    """
    nb_path = study_dir / "pipeline.ipynb"
    if not nb_path.exists():
        return
    prior_run_id = "unknown"
    try:
        import nbformat
        nb = nbformat.read(str(nb_path), as_version=4)
        prior_run = (nb.metadata.get("agentic") or {}).get("run")
        if prior_run:
            prior_run_id = Path(prior_run).name
    except Exception:  # noqa: BLE001 — corrupt/unreadable notebook, archive anyway
        pass
    archived = study_dir / f"pipeline_{prior_run_id}.ipynb"
    if archived.exists():  # same run id archived twice — don't clobber
        archived = study_dir / (
            f"pipeline_{prior_run_id}_{uuid.uuid4().hex[:8]}.ipynb"
        )
    nb_path.rename(archived)


def _init_canonical_store(
    run_dir: Path,
    study_dir: Path,
    evaluator_config: dict | None = None,
    eval_budget: int | None = None,
    mem_cap_bytes: int | None = None,
) -> dict:
    """Create canonical store dirs and write run_config.json sidecar.

    Creates:
    - ``<run_dir>/experiment_data/`` — shared ExperimentData project_dir
    - ``<run_dir>/debug/run_config.json`` — config read by get_evaluator()

    Parameters
    ----------
    run_dir : Path
        The timestamped run directory.
    study_dir : Path
        The study root (contains config.yaml, PROBLEM_STATEMENT.md, …).
    evaluator_config : dict or None, optional
        Parsed ``evaluator:`` block from config.yaml.  Keys:

        - ``entrypoint`` (str) — ``"path/to/file.py:attr"``
        - ``output_names`` (list[str]) — required for bare-fn entrypoints
        - ``lookup`` (dict) — ``{"pool": ..., "input_columns": ...,
          "output_columns": ...}``
        - ``fidelity_column`` (str or None)

    Returns the config dict that was written.
    """
    import json as _json

    store_dir = run_dir / "experiment_data"
    store_dir.mkdir(parents=True, exist_ok=True)
    # Mark this as the PROTECTED canonical store: ExperimentData.store() will
    # refuse any write that would shrink it, so a stray agent .store() can't
    # clobber the metered ledger (only get_evaluator() should write here).
    (store_dir / PROTECTED_STORE_SENTINEL).touch()

    eval_cfg = evaluator_config or {}

    # This function is called UNCONDITIONALLY at the top of every execute()
    # call — fresh run AND resume alike, with no `if _resume is None` guard —
    # so a resumed run reaches here again for the SAME run_dir. Without
    # reading back whatever is already on disk first, every dynamically
    # registered oracle (register_evaluator_entrypoint(), called mid-run by a
    # datagenerator delegation — the canonical entrypoint when config.yaml
    # declares none of its own, AND any namespaced "oracles" entry) would be
    # silently wiped the moment a run resumes, since the fresh config dict
    # below has no way to know about anything registered after the FIRST
    # call (BACKLOG #37). Preserve those specific fields from the existing
    # file when present; eval_budget/mem_cap_bytes still refresh from THIS
    # call's arguments unconditionally — the existing, deliberate "user
    # raises the budget, resume can progress" behavior.
    existing: dict = {}
    run_config_path = run_dir / "debug" / "run_config.json"
    if run_config_path.exists():
        try:
            existing = _json.loads(run_config_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError):
            existing = {}

    config: dict = {
        "store_dir": str(store_dir),
        # Co-locate the lock with the data (store_dir/experiment_data/) so
        # D000 ingestion (_ingest_precomputed_pool) and D001+ evaluations
        # (InstrumentedDataGenerator default) all lock the SAME file.
        "lock_path": str(store_dir / "experiment_data" / ".lock"),
        "evaluator_name": study_dir.name,
        "study_dir": str(study_dir),
        "fidelity_column": eval_cfg.get("fidelity_column"),
        # Extensible, oracle-stamped provenance columns ({col: value}) — an
        # open schema so any study can carry the metadata its science needs
        # (fidelity, regime, seed, mesh, …). Stamped on every ledger row by
        # InstrumentedDataGenerator, never authored by the agent.
        "provenance": eval_cfg.get("provenance"),
        "evaluator_entrypoint": existing.get(
            "evaluator_entrypoint", eval_cfg.get("entrypoint")),
        "evaluator_output_names": existing.get(
            "evaluator_output_names", eval_cfg.get("output_names")),
        "evaluator_lookup": existing.get(
            "evaluator_lookup", eval_cfg.get("lookup")),
        # Resource-governor knobs read by the in-process governor at the eval
        # boundary (get_evaluator/InstrumentedDataGenerator). eval_budget is a
        # SOFT cap (nudge only); mem_cap_bytes is the one HARD cap (host safety).
        "eval_budget": eval_budget,
        "mem_cap_bytes": mem_cap_bytes,
    }
    if "oracles" in existing:
        config["oracles"] = existing["oracles"]
    run_config_path.write_text(
        _json.dumps(config, indent=2), encoding="utf-8"
    )
    # No env-var export: run_config.json (written from config.yaml) is the single
    # channel — the campaign reads it via get_evaluator, the watcher reads it from
    # the run dir. Config is explicit in config.yaml, never through the environment.
    return config


def register_evaluator_entrypoint(
    run_config_path: Path,
    generator_file: Path | str,
    attr: str,
    output_names: list | None = None,
    namespace: str | None = None,
) -> str:
    """Register an agent-authored DataGenerator as an oracle.

    Atomically updates ``run_config.json`` so the next ``get_evaluator()``
    call (which re-reads the config on every invocation) resolves the
    authored generator with no manual config edit.  This is the runtime
    side of the datagenerator → oracle handoff: the agent writes the
    generator file (+ a registration manifest); the runtime points the
    entrypoint at it.

    Parameters
    ----------
    run_config_path : Path
        Path to the run's ``run_config.json``.
    generator_file : Path or str
        Path to the authored generator ``.py``.  An absolute path is made
        relative to the study root (``study_dir`` in the config); a relative
        path is assumed already study-relative and kept as-is.
        ``load_inner_evaluator`` resolves it as ``study_dir / file_part``.
    attr : str
        Name of the callable or ``DataGenerator`` subclass inside that file.
    output_names : list or None, optional
        Output column names — required when ``attr`` is a bare callable.
    namespace : str or None, optional
        The design namespace this oracle serves (Axis 3a). ``None`` (the
        default) registers the canonical single-study oracle as before. A
        non-``None`` namespace writes a ``run_config["oracles"][namespace]``
        block with its OWN isolated, protected store — leaving the canonical
        default oracle untouched, so opening a new design never disturbs the
        baseline study.

    Returns
    -------
    str
        The ``"file:attr"`` entrypoint that was written.
    """
    import json as _json
    import os as _os

    run_config_path = Path(run_config_path)
    config = _json.loads(run_config_path.read_text(encoding="utf-8"))

    gen_path = Path(generator_file)
    if gen_path.is_absolute():
        study_dir = Path(config["study_dir"]).resolve()
        file_part = str(gen_path.resolve().relative_to(study_dir))
    else:
        file_part = str(gen_path)

    entrypoint = f"{file_part}:{attr}"

    if namespace:
        # Per-experiment oracle: its own isolated, protected store; the default
        # oracle/store is left untouched. The experiment store is a sibling subdir
        # of the default store, so its name must not collide with the default's
        # own data dir ("experiment_data") — that name is how every accounting
        # helper distinguishes the default store from an experiment store.
        if namespace == "experiment_data":
            raise ValueError(
                "'experiment_data' is reserved (it is the default store's own "
                "data dir) — choose a different experiment/namespace name."
            )
        base_store = Path(config["store_dir"])
        ns_store = base_store / namespace
        ns_store.mkdir(parents=True, exist_ok=True)
        (ns_store / PROTECTED_STORE_SENTINEL).touch()
        oracles = config.setdefault("oracles", {})
        oracles[namespace] = {
            "store_dir": str(ns_store),
            "lock_path": str(ns_store / "experiment_data" / ".lock"),
            "evaluator_entrypoint": entrypoint,
            "evaluator_output_names": output_names,
            "evaluator_lookup": None,  # entrypoint takes precedence
        }
    else:
        prior_ep = config.get("evaluator_entrypoint")
        if prior_ep and prior_ep != entrypoint:
            # Overwriting the canonical oracle with a DIFFERENT file. This is
            # legitimate for a re-registration of THE baseline, but it is also
            # how a NEW-family oracle whose namespace failed to propagate
            # (Delegate arg + manifest both null) silently destroys the
            # baseline. Do not block it, but snapshot the prior config and warn
            # loudly so an accidental clobber is visible and recoverable
            # (run 20260706T204732: D006 replaced the baseline; D007 had to
            # self-heal via a hand-made .bak).
            bak = run_config_path.with_name(
                run_config_path.name + ".bak_preclobber")
            try:
                bak.write_text(
                    _json.dumps(config, indent=2), encoding="utf-8")
            except OSError:
                bak = None
            logging.getLogger("a3dasm").warning(
                "register_evaluator_entrypoint: OVERWRITING the canonical "
                "evaluator_entrypoint %r -> %r (namespace=None). If this "
                "oracle was meant for a NEW design family, register it under "
                "an explicit namespace instead; the prior config was "
                "snapshotted to %s.",
                prior_ep, entrypoint,
                bak if bak is not None else "(snapshot failed)")
        config["evaluator_entrypoint"] = entrypoint
        config["evaluator_output_names"] = output_names
        config["evaluator_lookup"] = None  # entrypoint takes precedence

    tmp = run_config_path.with_suffix(".json.tmp")
    tmp.write_text(_json.dumps(config, indent=2), encoding="utf-8")
    _os.replace(tmp, run_config_path)

    # Guardrail: keep study_dir/config.yaml's declared output_names in sync with
    # what was actually registered, so the human-facing config never goes stale
    # (a stale config.yaml is what let an agent build a pipeline around the wrong
    # objective column). Surgical line edit — preserves comments/formatting.
    # Only for the CANONICAL oracle: config.yaml describes the baseline study, so
    # a namespace's (possibly different) objective must not overwrite it.
    if namespace is None and output_names is not None and config.get("study_dir"):
        try:
            _sync_config_output_names(
                Path(config["study_dir"]) / "config.yaml", output_names)
        except Exception:  # noqa: BLE001 — best effort; never fail registration
            pass
    return entrypoint


def _sync_config_output_names(config_yaml: Path, output_names: list) -> bool:
    """Rewrite the ``output_names:`` value under ``evaluator:`` in config.yaml to
    match the registered names. Surgical (regex on the one line) so comments and
    the rest of the file are untouched. Returns True if a change was written.

    No-op if the file or the key is absent (we do not guess the YAML structure —
    run_config.json remains the authoritative runtime source either way).
    """
    import re as _re
    if not config_yaml.exists():
        return False
    text = config_yaml.read_text(encoding="utf-8")
    flow = "[" + ", ".join(str(n) for n in output_names) + "]"
    new_text, n = _re.subn(
        r"(?m)^(\s*output_names:\s*).*$", r"\g<1>" + flow, text)
    if n == 0 or new_text == text:
        return False
    config_yaml.write_text(new_text, encoding="utf-8")
    return True


def _ingest_precomputed_pool(
    store_dir: Path,
    study_dir: Path,
    lookup_cfg: dict,
) -> int:
    """Ingest a precomputed pool into the canonical store as D000 rows.

    Loads the pool ExperimentData from
    ``study_dir / lookup_cfg["pool"]`` and writes its rows into
    *store_dir* with provenance stamped as
    ``_delegation_id='D000'``, ``source='precomputed_pool'``.

    D000 rows are ground-truth data — they are *never* counted as
    evaluations.  ``_resolve_delegation_evals`` is called only for
    real delegation IDs (D001+) so D000 will never be counted.

    Parameters
    ----------
    store_dir : Path
        The run-level canonical store directory
        (``run_dir/experiment_data``).
    study_dir : Path
        Study root; pool path is resolved relative to this.
    lookup_cfg : dict
        The ``evaluator.lookup`` block from config.yaml.  Must have
        a ``"pool"`` key.

    Returns
    -------
    int
        Number of rows ingested.
    """
    from datetime import datetime, timezone

    from f3dasm import ExperimentData, ExperimentSample

    # Not yet public; flip to `from f3dasm import ...` after bessagroup/f3dasm#351.
    from f3dasm._src.errors import EmptyFileError, ReachMaximumTriesError
    from f3dasm._src.experimentsample import JobStatus
    from f3dasm.design import Domain
    from filelock import FileLock

    pool_project = study_dir / lookup_cfg["pool"]
    pool = ExperimentData.from_file(project_dir=pool_project)

    ts = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    n_rows = len(pool)

    df_in, df_out = pool.to_pandas()

    # Stamp provenance onto every output row.
    batch_samples: dict = {}
    for i in range(n_rows):
        row_in = (
            {} if df_in is None
            else {k: df_in.iloc[i][k] for k in df_in.columns}
        )
        row_out = (
            {} if df_out is None
            else {k: df_out.iloc[i][k] for k in df_out.columns}
        )
        row_out["_delegation_id"] = "D000"
        row_out["_source"] = "precomputed_pool"
        row_out["_ts"] = ts
        batch_samples[i] = ExperimentSample(
            _input_data=row_in,
            _output_data=row_out,
            job_status=JobStatus.FINISHED,
        )

    # Build batch domain: copy pool domain + provenance outputs.
    batch_domain = Domain()
    all_out_keys: set[str] = set()
    for s in batch_samples.values():
        all_out_keys.update(s._output_data.keys())
    for key in sorted(all_out_keys):
        batch_domain.add_output(key, exist_ok=True)

    batch = ExperimentData.from_data(
        data=batch_samples, domain=batch_domain
    )

    lock_path = store_dir / "experiment_data" / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path)):
        try:
            canon = ExperimentData.from_file(project_dir=store_dir)
        except (
            FileNotFoundError,
            EmptyFileError,
            ReachMaximumTriesError,
        ):
            canon = ExperimentData(domain=batch_domain)
        for col in ("_delegation_id", "_source", "_ts"):
            canon._domain.add_output(col, exist_ok=True)
        merged = canon + batch
        merged.store(project_dir=store_dir)

    return n_rows


def _parse_budget_str(value) -> float | None:
    """Parse budget: float seconds passthrough, or 'HH:MM:SS' string."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    parts = str(value).split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    return float(value)
