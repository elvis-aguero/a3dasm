"""get_evaluator: the one door from a campaign process to the run's oracle.

Resolves ``run_config.json`` (env pointer or cwd walk-up), the delegation id
(cwd name or env), the effective oracle block for a design namespace, and
the inner ``DataGenerator`` named by the registration — then hands back a
configured :class:`.instrumented.InstrumentedDataGenerator`. Also applies
the per-process memory governor at that entry point.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from f3dasm import DataGenerator, ExperimentData, ExperimentSample
from f3dasm._src.experimentsample import JobStatus

from .instrumented import InstrumentedDataGenerator

_DELEGATION_ID_RE = re.compile(r"^D\d+$")


# ==========================================================================


def load_inner_evaluator(
    run_config: dict, study_dir: Path
) -> DataGenerator | None:
    """Resolve and instantiate the inner evaluator from run_config.

    Resolution order
    ----------------
    1. ``evaluator_lookup`` present → build a
       :class:`.LookupDataGenerator` over the named pool.
    2. ``evaluator_entrypoint`` present → load via file-location
       import.  Entrypoint format: ``"path/to/file.py:AttrName"``
       (path is relative to *study_dir*; no package structure needed).

       - If the resolved attr is a :class:`.DataGenerator` subclass →
         instantiate no-args.
       - If it is a callable (bare function) →
         wrap with ``@datagenerator(output_names=...)`` applied
         functionally.  The callable **must** accept ``**kwargs``
         whose keys are the input column names of the sample.
         ``evaluator_output_names`` in run_config is required in this
         case.
    3. Neither present → return ``None``.

    Parameters
    ----------
    run_config : dict
        The dict loaded from ``run_config.json``.
    study_dir : Path
        Root of the study tree (pool paths and entrypoint paths are
        resolved relative to this directory).

    Returns
    -------
    DataGenerator or None
    """
    import importlib.util
    import sys

    lookup_cfg = run_config.get("evaluator_lookup")
    if lookup_cfg:
        from .lookup import LookupDataGenerator

        pool_rel = lookup_cfg["pool"]
        pool_project = study_dir / pool_rel
        pool = ExperimentData.from_file(project_dir=pool_project)
        return LookupDataGenerator(
            pool=pool,
            input_columns=lookup_cfg["input_columns"],
            output_columns=lookup_cfg.get("output_columns"),
        )

    entrypoint = run_config.get("evaluator_entrypoint")
    if entrypoint:
        if ":" not in entrypoint:
            raise ValueError(
                f"evaluator_entrypoint must be 'path/to/file.py:attr', "
                f"got {entrypoint!r}"
            )
        file_part, attr = entrypoint.rsplit(":", 1)
        abs_file = (study_dir / file_part).resolve()
        if not abs_file.exists():
            raise FileNotFoundError(
                f"Evaluator file not found: {abs_file} "
                f"(entrypoint={entrypoint!r})"
            )
        module_name = (
            "_f3dasm_eval_"
            + abs_file.stem.replace("-", "_").replace(".", "_")
        )
        spec = importlib.util.spec_from_file_location(
            module_name, abs_file
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Cannot load module from {abs_file}"
            )
        mod = importlib.util.module_from_spec(spec)
        # Temporarily add study_dir to sys.path so the loaded module
        # can perform its own relative imports if needed.
        _injected = str(study_dir) not in sys.path
        if _injected:
            sys.path.insert(0, str(study_dir))
        try:
            spec.loader.exec_module(mod)
        finally:
            if _injected and str(study_dir) in sys.path:
                sys.path.remove(str(study_dir))

        obj = getattr(mod, attr)

        # DataGenerator subclass → instantiate
        try:
            if isinstance(obj, type) and issubclass(obj, DataGenerator):
                return obj()
        except TypeError:
            pass

        # DataGenerator instance (already decorated) → return as-is
        if isinstance(obj, DataGenerator):
            return obj

        # Callable (bare function) → wrap with @datagenerator.
        # The contract: the callable accepts **kwargs whose keys are
        # the ExperimentSample's input column names.  We create a thin
        # adapter that passes all _input_data as kwargs so that both
        # VAR_KEYWORD (**kwargs) and named-parameter callables work.
        if callable(obj):
            output_names = run_config.get("evaluator_output_names")
            if not output_names:
                raise ValueError(
                    f"evaluator_entrypoint {entrypoint!r} resolves to a "
                    "callable (not a DataGenerator subclass).  "
                    "Set 'evaluator_output_names' in config.yaml "
                    "(e.g. output_names: [f])."
                )
            _fn = obj
            _out_names = list(output_names)

            class _BareCallableGen(DataGenerator):
                def execute(
                    self,
                    experiment_sample: ExperimentSample,
                    **kwargs,
                ) -> ExperimentSample:
                    result = _fn(**experiment_sample._input_data)
                    if isinstance(result, dict):
                        # map by declared output name, not order
                        result = [result[n] for n in _out_names]
                    elif not isinstance(result, (list, tuple)):
                        result = [result]
                    for name, val in zip(_out_names, result, strict=False):
                        experiment_sample._output_data[name] = val
                    experiment_sample.job_status = (
                        JobStatus.FINISHED
                    )
                    return experiment_sample

            return _BareCallableGen()

        raise ValueError(
            f"Resolved attr {attr!r} from {entrypoint!r} is neither a "
            "DataGenerator subclass nor a callable."
        )

    return None


# ==========================================================================

_GOVERNOR_PID_APPLIED = False


def _apply_process_governor(run_config: dict, store_dir: Path,
                            delegation_id: str) -> None:
    """At the oracle entry INSIDE a campaign process: apply the one hard memory
    cap to this process and register its PID so the run's watcher can sample +
    kill this delegation's tree regardless of how the agent's Bash launched it.
    Idempotent per process; best-effort — never blocks evaluations."""
    global _GOVERNOR_PID_APPLIED
    if _GOVERNOR_PID_APPLIED:
        return
    _GOVERNOR_PID_APPLIED = True
    try:
        from ..infra.resource_backend import get_resource_backend
        be = get_resource_backend()
        cap = run_config.get("mem_cap_bytes")
        if cap:
            be.set_self_limit(int(cap))  # no-op on psutil/stdlib; the RSS watcher enforces
        import json as _json
        import os as _os
        from datetime import datetime, timezone
        run_dir = store_dir.parent  # store_dir == <run_dir>/experiment_data
        reg = run_dir / "debug" / "governor_pids.jsonl"
        if reg.parent.exists():
            _pid = _os.getpid()
            rec = {
                "delegation_id": delegation_id,
                "pid": _pid,
                # Process start time: the watcher checks this before killing, so a
                # RECYCLED pid (a different program that inherited the number) is
                # never killed — the ownership guard.
                "start_time": be.proc_start_time(_pid),
                "ts": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            }
            with reg.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(rec) + "\n")
    except Exception:  # noqa: BLE001
        pass


def _effective_oracle_config(run_config: dict, namespace: str | None) -> dict:
    """Resolve the effective oracle config for a design ``namespace``.

    A namespace is a sub-config under ``run_config["oracles"][namespace]`` of the
    same shape as the flat keys. With ``namespace is None`` this returns the
    run_config unchanged — the single-study path is byte-for-byte today's. With a
    namespace, the namespace block's oracle + store keys are taken WHOLESALE
    (defaulting to ``None`` if absent, so a base ``evaluator_lookup`` cannot leak
    past a namespace that declares an entrypoint), while everything else
    (``study_dir``, budgets, governor knobs) is inherited from the base config.

    Raises ``ValueError`` naming the available namespaces if ``namespace`` is not
    registered.
    """
    if namespace is None:
        return run_config

    oracles = run_config.get("oracles") or {}
    if namespace not in oracles:
        available = ", ".join(sorted(oracles)) or "(none registered)"
        raise ValueError(
            f"Unknown design namespace {namespace!r}. Registered namespaces: "
            f"{available}. A namespace's oracle is authored and registered by "
            f"the datagenerator agent before it can be evaluated."
        )
    block = oracles[namespace] or {}
    eff = dict(run_config)
    # Oracle + store keys come wholesale from the namespace block (None if absent).
    for key in (
        "store_dir", "lock_path", "evaluator_entrypoint",
        "evaluator_output_names", "evaluator_lookup", "source",
        "fidelity_column", "provenance",
    ):
        if key in block:
            eff[key] = block[key]
        elif key in ("evaluator_entrypoint", "evaluator_output_names",
                     "evaluator_lookup"):
            eff[key] = None  # don't let a base oracle leak into the namespace
    return eff


def get_evaluator(namespace: str | None = None) -> InstrumentedDataGenerator:
    """The ONE door to a registered ground-truth oracle.

    Locates ``run_config.json`` by walking up from ``Path.cwd()``, reads
    all configuration from it, and derives the delegation ID from the
    current working directory name (expected pattern ``D###``).

    The oracle is the source registered for this run, and its evaluations are
    written to the canonical store with provenance. There is no way to substitute
    an arbitrary inner generator or redirect the store — that is what makes
    ground-truth metering airtight. (Surrogates, stubs, and analysis are the
    agent's own DataGenerators, run freely off-ledger — never through here.)

    Parameters
    ----------
    namespace : str or None, optional
        The design namespace whose oracle + ledger to resolve. ``None`` (the
        default) uses the flat single-study config and the canonical store — the
        behavior every existing study relies on. A non-``None`` namespace resolves
        ``run_config["oracles"][namespace]`` (its own oracle + its own isolated
        store). When omitted, the namespace falls back to the ``F3DASM_NAMESPACE``
        environment variable, so a delegation scoped to a namespace keeps the
        agent's call site a plain ``get_evaluator()``.

    Returns
    -------
    InstrumentedDataGenerator

    Raises
    ------
    ValueError
        If the cwd is not a ``D###`` directory and ``F3DASM_DELEGATION_ID`` is
        not set, if no evaluator source is registered, or if ``namespace`` is not
        a registered namespace.
    FileNotFoundError
        If ``run_config.json`` cannot be found by walking up from cwd.
    """
    delegation_id = _resolve_delegation_id()
    run_config = _load_run_config()

    namespace_from_env = False
    if namespace is None:
        env_ns = os.environ.get("F3DASM_NAMESPACE", "")
        if env_ns:
            namespace, namespace_from_env = env_ns, True
    try:
        cfg = _effective_oracle_config(run_config, namespace)
    except ValueError as exc:
        if namespace_from_env:
            # De-footgun: get_evaluator() SILENTLY inherits F3DASM_NAMESPACE, so a
            # namespace-scoped delegation asking for the DEFAULT/baseline oracle
            # gets a confusing "unknown namespace". Name the env source + the fix.
            raise ValueError(
                f"{exc} NOTE: namespace {namespace!r} was inherited from the "
                "F3DASM_NAMESPACE environment variable, not passed explicitly. "
                "For the default/baseline oracle, clear it first "
                "(`env -u F3DASM_NAMESPACE ...`) or call from a process where it "
                "is unset."
            ) from exc
        raise

    store_dir = Path(cfg["store_dir"])
    lock_path_str = cfg.get("lock_path")
    lock_path = (
        Path(lock_path_str)
        if lock_path_str
        else store_dir / "experiment_data" / ".lock"
    )
    source = cfg.get(
        "source",
        cfg.get("evaluator_name", ""),
    )
    fidelity_column = cfg.get("fidelity_column")
    # Extensible provenance declared for this run (open schema; stamped on
    # every row by the wrapper, never by the agent). Tolerate a non-dict.
    _prov = cfg.get("provenance")
    extra_provenance = _prov if isinstance(_prov, dict) else {}

    study_dir_str = cfg.get("study_dir")
    if study_dir_str is None:
        raise ValueError(
            "run_config.json is missing 'study_dir' key; "
            "re-run your study to regenerate it."
        )
    inner = load_inner_evaluator(cfg, Path(study_dir_str))
    if inner is None:
        raise ValueError(
            "No ground-truth oracle is registered for this run. A source "
            "must be authored and registered by the datagenerator agent "
            "(it writes a registration.json the runtime reads) before "
            "evaluations can be ledgered. If this study genuinely has no "
            "registerable oracle, report evaluation counts manually via "
            "ReportEvals (honour-system, off-ledger)."
        )

    # Hard memory cap + PID registration for THIS campaign process (the one
    # hard resource boundary). Done here because every campaign reaches the
    # oracle through get_evaluator, regardless of how it was launched.
    _apply_process_governor(cfg, store_dir, delegation_id)

    # F3DASM_DEDUP_SCOPE=all: set by notebook_exec.sandbox_env() for
    # reproduction-gate/deliverable execution, where the process is stamped
    # with a synthetic delegation id that never matches the real
    # delegation(s) that actually generated the ledger's rows — the default
    # per-delegation dedup scope would silently fail to recognize ANY
    # existing row as already-seen there. Real campaign delegations never
    # set this, so they keep the default "delegation" scope.
    dedup_scope = os.environ.get("F3DASM_DEDUP_SCOPE", "delegation")

    return InstrumentedDataGenerator(
        inner=inner,
        store_dir=store_dir,
        delegation_id=delegation_id,
        source=source,
        fidelity_column=fidelity_column,
        lock_path=lock_path,
        extra_provenance=extra_provenance,
        eval_budget=cfg.get("eval_budget"),
        dedup_scope=dedup_scope,
    )


# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------


def _resolve_delegation_id() -> str:
    """Return delegation ID from cwd name or env var, or raise."""
    cwd_name = Path.cwd().name
    if _DELEGATION_ID_RE.match(cwd_name):
        return cwd_name

    env_id = os.environ.get("F3DASM_DELEGATION_ID", "")
    if env_id and _DELEGATION_ID_RE.match(env_id):
        return env_id

    raise ValueError(
        "get_evaluator() needs a delegation id: either run inside a workspace "
        "whose directory is named D### (cwd), OR set F3DASM_DELEGATION_ID=D###. "
        f"Got cwd='{cwd_name}', "
        f"F3DASM_DELEGATION_ID='{env_id or '(unset)'}'. "
        "For an off-delegation or validation eval, just "
        "`export F3DASM_DELEGATION_ID=D000` — the id is only used to stamp "
        "provenance, so no directory needs to exist (no mkdir/cd required)."
    )


def _load_run_config() -> dict:
    """Locate run_config.json: explicit env var first, then walk up from cwd.

    ``F3DASM_RUN_CONFIG`` (injected per-session by the backend) points straight
    at the file, so resolution does not depend on the worker's cwd — the SDK
    spawns it in study_dir, from which a walk-UP can never reach the config that
    lives DOWN at runs/<id>/debug/. The cwd walk-up stays as a fallback for
    direct/standalone invocations (e.g. the reproduction gate sets cwd itself).
    """
    env_path = os.environ.get("F3DASM_RUN_CONFIG", "")
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return json.loads(candidate.read_text())
        raise FileNotFoundError(
            f"F3DASM_RUN_CONFIG points to '{env_path}', which does not exist."
        )
    current = Path.cwd()
    for _ in range(10):  # guard against infinite walk
        candidate = current / "run_config.json"
        if candidate.exists():
            return json.loads(candidate.read_text())
        parent = current.parent
        if parent == current:
            break
        current = parent
    raise FileNotFoundError(
        "run_config.json not found: F3DASM_RUN_CONFIG unset and no "
        f"run_config.json found by walking up from '{Path.cwd()}'."
    )


__all__ = [
    "get_evaluator",
    "load_inner_evaluator",
]
