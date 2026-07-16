"""Instrumented DataGenerator: canonical concurrency-safe eval ledger.

Provides :class:`InstrumentedDataGenerator`, a wrapper that:

1. Delegates execution to an inner ``DataGenerator``.
2. Stamps provenance metadata (delegation_id, source, UTC timestamp)
   onto each returned ``ExperimentSample``.
3. Buffers samples and flushes them to a shared ``ExperimentData`` store
   under a ``FileLock`` so concurrent delegations cannot corrupt the
   ledger.

Also provides :func:`get_evaluator`, a factory that reads
``run_config.json`` from the delegation workspace and returns a
configured :class:`InstrumentedDataGenerator`.

**Phase 1 scope** — wiring into agent_runtime/nodes is a separate task.
The ``fidelity_column`` parameter is accepted but unused in Phase 1;
it exists for forward-compatibility when fidelity-aware stamping is
added.
"""
from __future__ import annotations

#                                                                      Modules
# ==========================================================================
import json
import os
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from f3dasm import DataGenerator, ExperimentData, ExperimentSample

# Not yet re-exported by the public f3dasm API. They exist in stock f3dasm,
# only under _src. Flip to `from f3dasm import ...` once bessagroup/f3dasm#351
# lands and is pinned.
from f3dasm._src.errors import EmptyFileError, ReachMaximumTriesError
from f3dasm._src.experimentsample import JobStatus
from f3dasm.design import Domain
from filelock import FileLock

#                                                         Authorship & Credits
# ==========================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# ==========================================================================

_DELEGATION_ID_RE = re.compile(r"^D\d+$")


# ==========================================================================


class InstrumentedDataGenerator(DataGenerator):
    """Wrap an inner DataGenerator, stamp provenance, and flush to disk.

    Parameters
    ----------
    inner : DataGenerator
        The wrapped evaluator.  Callers are responsible for decorating a
        plain function with ``@datagenerator`` before passing it here.
    store_dir : Path or str
        Canonical project_dir — the directory that *contains*
        ``experiment_data/``.  Every delegation shares the same
        ``store_dir`` so their rows end up in one ledger.
    delegation_id : str
        Identifier for this delegation, e.g. ``"D003"``.
    source : str, optional
        Human-readable label stamped in the ``source`` provenance column
        (e.g. the evaluator name).  Default ``""``.
    fidelity_column : str or None, optional
        Name of the study's fidelity input column if any.  Unused in
        Phase 1 — accepted only for forward-compatibility.
    lock_path : Path or str or None, optional
        Path for the ``FileLock``.  Defaults to
        ``<store_dir>/experiment_data/.lock``.
    flush_every : int, optional
        Number of samples to buffer before a locked flush.  Default 1
        (flush on every execute).
    """

    def __init__(
        self,
        inner: DataGenerator,
        store_dir: Path | str,
        delegation_id: str,
        *,
        source: str = "",
        fidelity_column: Optional[str] = None,
        lock_path: Optional[Path | str] = None,
        flush_every: int = 1,
        extra_provenance: Optional[dict] = None,
        eval_budget: Optional[int] = None,
    ) -> None:
        self.inner = inner
        self.store_dir = Path(store_dir)
        self.delegation_id = delegation_id
        self.source = source
        # SOFT eval-budget governor (resource-governance L1). Fires at the flush
        # boundary — i.e. MID-delegation, where the strategizer's turn-gated check
        # is blind. eval_budget is soft (§4): it NUDGES the offender, never stops
        # the campaign. `_nudge_bands_hit` caps the nudge at one per threshold band
        # (flush_every defaults to 1 → per-row → uncapped would spam).
        self.eval_budget = eval_budget
        self._nudge_bands_hit: set[int] = set()
        self.fidelity_column = fidelity_column  # unused Phase 1
        self.flush_every = flush_every
        # Extensible, oracle-stamped provenance: arbitrary {column: value}
        # declared per run (config 'provenance' block, or set by the runtime).
        # Stamped into EVERY evaluated row at the metered call — so the schema
        # is open (any future problem can add columns: fidelity, regime, seed,
        # mesh, …) and the VALUES come from the oracle wrapper, never from the
        # agent (which keeps the audit trail trustworthy).
        self.extra_provenance: dict = dict(extra_provenance or {})

        if lock_path is None:
            lock_path = (
                self.store_dir / "experiment_data" / ".lock"
            )
        self.lock_path = Path(lock_path)

        self._buffer: list[ExperimentSample] = []
        # Designs (coord keys) queued for SUPERSEDE via supersede(): their stale
        # canon rows are dropped at flush so a corrected re-eval replaces them.
        self._supersede_keys: set = set()

    # ------------------------------------------------------------------

    def execute(
        self, experiment_sample: ExperimentSample, **kwargs
    ) -> ExperimentSample:
        """Run inner generator, stamp provenance, buffer, maybe flush.

        Parameters
        ----------
        experiment_sample : ExperimentSample
            Sample to evaluate.
        **kwargs
            Forwarded to ``inner.execute``.

        Returns
        -------
        ExperimentSample
            The evaluated sample (with provenance stamped into
            ``_output_data``).
        """
        _t0 = time.perf_counter()
        out = self.inner.execute(experiment_sample, **kwargs)
        _wall_ms = (time.perf_counter() - _t0) * 1000.0

        # Stamp provenance into the output dict.
        ts = datetime.now(tz=timezone.utc).isoformat(
            timespec="seconds"
        )
        out._output_data["_delegation_id"] = self.delegation_id
        out._output_data["_source"] = self.source
        out._output_data["_ts"] = ts
        # Generic per-eval wall-time (ms). A plain underscore-prefixed column:
        # to_numpy() drops it and it's excluded from value stats. Any grouping
        # (per-phase, per-fidelity) is df.groupby(col)["_wall_ms"] downstream —
        # no timing-specific code special-cases a dimension here.
        out._output_data["_wall_ms"] = round(_wall_ms, 3)
        # Extensible declared provenance (oracle-stamped, not agent-authored).
        for _col, _val in self.extra_provenance.items():
            out._output_data[_col] = _val

        # The inner generator returned normally, so this evaluation COMPLETED:
        # stamp FINISHED on the copy we buffer for the canonical store. f3dasm's
        # _run_sample marks finished on the agent's *working* ExperimentData, not
        # on this buffered deepcopy — without this, completed rows persist as
        # IN_PROGRESS in the canonical jobs.csv, defeating the FINISHED-regression
        # store guard, is_all_finished(), and resumption logic. (Errors raise out
        # of inner.execute before this line and are marked elsewhere.)
        out.mark("finished")

        self._buffer.append(deepcopy(out))

        if len(self._buffer) >= self.flush_every:
            self._flush()

        return out

    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush any remaining buffered samples to the store.

        Call at the end of a delegation to ensure no samples are lost.
        """
        if self._buffer:
            self._flush()

    # ------------------------------------------------------------------

    def _flush(self) -> None:
        """Flush the current buffer to disk under a FileLock.

        The entire read → merge → write sequence is executed inside the
        lock so concurrent threads/processes cannot interleave.
        """
        if not self._buffer:
            return

        # Ensure the lock parent directory exists.
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        n_skipped = 0
        _n_total = 0
        with FileLock(str(self.lock_path)):
            # Absent store (FileNotFoundError) OR a torn/empty CSV from an
            # interrupted prior write (EmptyFileError / retry-exhausted):
            # treat as fresh and let this locked write heal it. We do NOT
            # catch broader errors — a populated store that fails to parse
            # must propagate, never be silently overwritten with the batch.
            try:
                canon = ExperimentData.from_file(
                    project_dir=self.store_dir
                )
            except (
                FileNotFoundError,
                EmptyFileError,
                ReachMaximumTriesError,
            ):
                canon = None

            # Supersede (opt-in correction): drop the stale rows for any design
            # queued via supersede() BEFORE dedup, so the corrected re-eval
            # REPLACES them instead of being skipped as a duplicate. Net-count-
            # preserving (old row out, new row in) so the PROTECTED-store guard
            # still holds (it blocks only SHRINK and FINISHED-regression).
            if canon is not None and self._supersede_keys:
                canon = self._drop_canon_rows(canon, self._supersede_keys)

            # Dedup-on-write: drop buffered samples whose design (the
            # non-provenance input coords, rounded 10dp — the SAME key the
            # DUPLICATE_EVALUATION detector uses) is already in the canonical
            # store, or repeated within this batch. Keep-first: an existing row
            # is never mutated (correcting a stale row is a separate, opt-in
            # affordance, deliberately out of scope). This stops a re-launched
            # or retried campaign from re-appending designs already evaluated —
            # the retry-duplication that burned ~30h/2-designs in an earlier run.
            # NOTE: this hard-prevents a duplicate row (the DUPLICATE_EVALUATION
            # signal was previously only a soft nudge); safe because the oracle
            # is deterministic, so a repeat is pure waste, not a confirm-probe.
            n_skipped = self._drop_duplicate_buffer(canon)

            if self._buffer:
                batch_domain = self._build_batch_domain()
                batch = self._build_batch_experimentdata(batch_domain)
                if canon is None:
                    canon = ExperimentData(domain=batch_domain)

                # Ensure provenance columns are declared on the canon domain
                # (the fixed four + any extensible declared columns).
                for col in ("_delegation_id", "_source", "_ts", "_wall_ms",
                            *self.extra_provenance):
                    canon._domain.add_output(col, exist_ok=True)

                # Re-type the batch's input parameters to the canonical domain's:
                # the canonical store is authoritative on types. _build_batch_domain
                # declares inputs as untyped base Parameter(); merging one into a
                # typed canonical param (e.g. add_int → DiscreteParameter) raises
                # "Cannot add non-continuous parameter to continuous!". Adopting the
                # canonical typed param makes the per-key merge typed+typed.
                for key, cparam in canon._domain.input_space.items():
                    if key in batch._domain.input_space:
                        batch._domain.input_space[key] = cparam

                merged = canon + batch
                merged.store(project_dir=self.store_dir)
                _n_total = len(merged)
            elif canon is not None:
                _n_total = len(canon)

        # SOFT eval-budget nudge (lock released): fire MID-delegation at the eval
        # boundary, to THE OFFENDER (this campaign's own stdout → the implementer's
        # delegation report). Never stops the campaign; capped at one per band.
        if _n_total:
            self._maybe_nudge_budget(_n_total)
        if n_skipped:
            self._record_dedup(n_skipped)

        self._buffer.clear()
        self._supersede_keys.clear()

    @staticmethod
    def _coord_key(input_data: dict) -> tuple:
        """Order-independent design key: (col, value) pairs over the
        non-provenance inputs, values rounded to 10dp — the same convention
        ``duplicate_eval_stats`` and the reproduction gate use, so dedup-on-write
        matches duplicate DETECTION exactly."""
        def _r(v):
            try:
                return round(float(v), 10)
            except (TypeError, ValueError):
                return v
        return tuple(sorted(
            (str(k), _r(v)) for k, v in input_data.items()
            if k not in _PROVENANCE_COLS))

    def _drop_duplicate_buffer(self, canon) -> int:
        """Filter ``self._buffer`` in place to designs not already present in
        ``canon`` and not repeated earlier in the buffer (keep-first). Returns
        the count dropped. Best-effort: any failure leaves the buffer intact so
        the eval path never loses data to a dedup bug."""
        try:
            seen: set = set()
            if canon is not None:
                df_in, df_out = canon.to_pandas()
                if df_in is not None and not df_in.empty:
                    # Per-delegation dedup — matches duplicate_eval_stats's own
                    # per-delegation definition of "duplicate". Only THIS
                    # delegation's prior rows count: a DIFFERENT delegation
                    # legitimately re-measuring the same design is not waste
                    # (and collapsing it would corrupt concurrent campaigns).
                    if "_delegation_id" in df_out.columns:
                        mine = (df_out["_delegation_id"].astype(str)
                                == str(self.delegation_id)).to_numpy()
                        df_in = df_in[mine]
                    cols = [c for c in df_in.columns
                            if c not in _PROVENANCE_COLS]
                    for rec in df_in[cols].to_dict("records"):
                        seen.add(self._coord_key(rec))
            survivors = []
            for s in self._buffer:
                k = self._coord_key(s._input_data)
                if k in seen:
                    continue
                seen.add(k)
                survivors.append(s)
            n = len(self._buffer) - len(survivors)
            self._buffer = survivors
            return n
        except Exception:  # noqa: BLE001
            return 0

    def _record_dedup(self, n_skipped: int) -> None:
        """Best-effort DEDUP_SKIPPED audit line (never breaks the eval path)."""
        try:
            import json as _json
            from datetime import datetime, timezone
            diag = self.store_dir.parent / "debug" / "diagnostics.jsonl"
            if diag.parent.exists():
                rec = {
                    "ts": datetime.now(tz=timezone.utc).isoformat(
                        timespec="seconds"),
                    "node": self.delegation_id,
                    "error_type": "DEDUP_SKIPPED",
                    "message": (
                        f"{n_skipped} buffered eval(s) skipped: design already "
                        "in the ledger (dedup-on-write)"),
                }
                with diag.open("a", encoding="utf-8") as f:
                    f.write(_json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------

    def supersede(self, experiment_sample: ExperimentSample, **kwargs):
        """Re-evaluate a design and REPLACE its existing FINISHED ledger row(s).

        The opt-in correction for a stale/wrong row (e.g. a pre-oracle-fix read):
        dedup-on-write otherwise keep-firsts, so a plain re-eval would be dropped.
        This runs the oracle fresh and, at flush, drops the design's prior canon
        row(s) and writes the new one — net-count-preserving, so the PROTECTED-
        store guard still holds (it blocks only SHRINK and FINISHED-regression,
        neither of which a same-design replace does). Use sparingly; the ledger
        is otherwise append-only by design."""
        self._supersede_keys.add(
            self._coord_key(experiment_sample._input_data))
        out = self.execute(experiment_sample, **kwargs)
        self.flush()   # force: the drop + the new row must land in one write
        # Provenance-mutating op → leave an audit line (the old value is replaced
        # in-ledger, so the FACT of the correction must be traceable).
        self._record_supersede(dict(experiment_sample._input_data))
        return out

    def _record_supersede(self, input_data: dict) -> None:
        """Best-effort SUPERSEDE audit line (never breaks the eval path)."""
        try:
            import json as _json
            from datetime import datetime, timezone
            diag = self.store_dir.parent / "debug" / "diagnostics.jsonl"
            if diag.parent.exists():
                coords = {k: v for k, v in input_data.items()
                          if k not in _PROVENANCE_COLS}
                rec = {
                    "ts": datetime.now(tz=timezone.utc).isoformat(
                        timespec="seconds"),
                    "node": self.delegation_id,
                    "error_type": "SUPERSEDE",
                    "message": (
                        f"re-evaluated and REPLACED the ledger row for design "
                        f"{coords} (prior value overwritten in-ledger)"),
                }
                with diag.open("a", encoding="utf-8") as f:
                    f.write(_json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def _drop_canon_rows(self, canon, keys: set):
        """canon minus every row whose design (coord key) is in ``keys``, rebuilt
        via the same from_data(samples, domain) idiom the flush path uses. Kept
        rows are re-stamped FINISHED (they were) so the FINISHED-regression guard
        passes. Best-effort: on any error returns canon unchanged (the new row
        then merely appends — visible and safe, never lost)."""
        try:
            df_in, df_out = canon.to_pandas()
            if df_in is None or df_in.empty:
                return canon
            in_cols = list(df_in.columns)
            out_cols = list(df_out.columns)
            samples: dict[int, ExperimentSample] = {}
            j = 0
            for i in range(len(df_out)):
                if self._coord_key(df_in.iloc[i].to_dict()) in keys:
                    continue
                s = ExperimentSample(
                    _input_data={c: df_in.iloc[i][c] for c in in_cols},
                    _output_data={c: df_out.iloc[i][c] for c in out_cols})
                s.mark("finished")
                samples[j] = s
                j += 1
            return ExperimentData.from_data(
                data=samples, domain=canon._domain)
        except Exception:  # noqa: BLE001
            return canon

    # Soft eval-budget nudge bands (fraction of eval_budget). One nudge per band
    # max → at most 3 nudges per delegation (the fixed cap).
    _NUDGE_BANDS = (0.8, 1.0, 1.5)

    def _maybe_nudge_budget(self, n_total: int) -> None:
        """SOFT, capped, offender-directed eval-budget nudge. Best-effort: a
        governor must never break the eval path, so it swallows everything."""
        try:
            budget = self.eval_budget
            if not budget or budget <= 0:
                return
            # Bands crossed by this flush that we haven't nudged yet.
            crossed = [
                int(b * 100) for b in self._NUDGE_BANDS
                if n_total >= b * budget and int(b * 100) not in self._nudge_bands_hit
            ]
            if not crossed:
                return
            # Mark ALL crossed bands hit (so a big batch that jumps two bands
            # still nudges only once) and nudge for the highest.
            self._nudge_bands_hit.update(crossed)
            pct = round(100 * n_total / budget)
            msg = (
                f"[EVAL BUDGET — {self.delegation_id}] {n_total}/{budget} ledgered "
                f"evals ({pct}% of the SOFT budget). The budget is soft (not "
                "enforced), but this is the SHARED canonical ledger: every campaign "
                "re-run APPENDS to it, so re-running a full campaign to debug burns "
                "the budget fast. Debug on RunScratch / a stub, not the real oracle; "
                "re-plan rather than spend more real evaluations."
            )
            # Channel 1 — the campaign's OWN stdout → captured into the offender's
            # (implementer's) delegation report. This is the cross-process path to
            # the offender (the governor runs in the campaign subprocess).
            print(msg, flush=True)
            # Channel 2 — a BUDGET_WARN diagnostic line for the audit trail ONLY
            # (NOT the nudge). store_dir is <run_dir>/experiment_data.
            try:
                import json as _json
                from datetime import datetime, timezone
                diag = self.store_dir.parent / "debug" / "diagnostics.jsonl"
                if diag.parent.exists():
                    rec = {
                        "ts": datetime.now(tz=timezone.utc).isoformat(
                            timespec="seconds"),
                        "node": self.delegation_id,
                        "error_type": "BUDGET_WARN",
                        "message": f"{n_total}/{budget} evals ({pct}%)",
                    }
                    with diag.open("a", encoding="utf-8") as f:
                        f.write(_json.dumps(rec) + "\n")
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass

    def _build_batch_domain(self) -> Domain:
        """Build a Domain that covers inner inputs + outputs + provenance cols.

        Input columns are declared as base Parameter() (no bounds). This is
        intentional: the batch domain is merged with the canonical store's
        domain on every flush, which already carries the correct typed
        parameters (ContinuousParameter with bounds, etc.). Declaring the
        input keys here ensures that on the very first flush — when no
        canonical store exists yet — the written domain.json at least has
        the input column names, preventing a later from_file() load from
        silently omitting them and causing samplers to no-op without error.
        """
        d = Domain()
        # Declare input columns (base Parameter — bounds come from the
        # canonical domain on merge, not from this batch).
        all_input_keys: set[str] = set()
        for sample in self._buffer:
            all_input_keys.update(sample._input_data.keys())
        for key in sorted(all_input_keys):
            # Public API for a bounds-less base input column (constructs the
            # Parameter and registers it, replacing the private d._add).
            d.add_parameter(key)
        # Collect all output keys from the buffer.
        all_keys: set[str] = set()
        for sample in self._buffer:
            all_keys.update(sample._output_data.keys())
        for key in sorted(all_keys):
            d.add_output(key, exist_ok=True)
        return d

    def _build_batch_experimentdata(
        self, domain: Domain
    ) -> ExperimentData:
        """Build an ExperimentData from the current buffer."""
        data: dict[int, ExperimentSample] = {
            i: sample for i, sample in enumerate(self._buffer)
        }
        return ExperimentData.from_data(data=data, domain=domain)


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
        from .resource_backend import get_resource_backend
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

    return InstrumentedDataGenerator(
        inner=inner,
        store_dir=store_dir,
        delegation_id=delegation_id,
        source=source,
        fidelity_column=fidelity_column,
        lock_path=lock_path,
        extra_provenance=extra_provenance,
        eval_budget=cfg.get("eval_budget"),
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


# ==========================================================================
# Module-level mtime cache: {str(store_dir): (mtime, RunStateSummary)}
_RSS_CACHE: dict[str, tuple[float, RunStateSummary]] = {}
# Guards _RSS_CACHE: the summary is read from main (closure) threads and
# refreshed from background delegation threads. Benign on CPython, but
# the lock makes it correct on free-threaded builds too.
_RSS_CACHE_LOCK = __import__("threading").Lock()

_PROVENANCE_COLS = frozenset({"_delegation_id", "_source", "_ts", "_wall_ms"})


class RunStateSummary:
    """Read-only summary of the canonical evaluation ledger.

    Computed from the store at read time; never written by agents.
    Cached at module level by (store_dir, output.csv mtime) so repeated
    calls within a turn are O(1).
    """

    def __init__(
        self,
        *,
        n_rows: int,
        n_per_delegation: dict,
        n_per_source: dict,
        n_per_fidelity: dict | None,
        output_stats: dict,
        mean_eval_wall_ms: float | None = None,
        wall_per_delegation: dict | None = None,
    ) -> None:
        self.n_rows = n_rows
        self.n_per_delegation = n_per_delegation
        self.n_per_source = n_per_source
        self.n_per_fidelity = n_per_fidelity
        self.output_stats = output_stats
        self.mean_eval_wall_ms = mean_eval_wall_ms
        # {delegation_id: {"n", "median_ms", "max_ms", "total_ms"}} — per-eval
        # wall cost grouped by the delegation that wrote the rows. Lets a
        # finished delegation report its OWN measured sim cost, so budget
        # planning runs on observed reality, not an a priori per-sim estimate.
        self.wall_per_delegation = wall_per_delegation or {}

    # ------------------------------------------------------------------

    def delegation_footer(
        self,
        delegation_id: str,
        *,
        wall_remaining_s: float | None = None,
        wall_budget_s: float | None = None,
        peak_rss_bytes: int | None = None,
        ram_cap_bytes: int | None = None,
    ) -> str | None:
        """Compact KPI footer for ONE delegation's ledgered rows, or None.

        Auto-appended to the delegation report the strategizer receives, so the
        measured per-eval sim cost travels with every result — no on-demand
        lookup. When the wall budget is known, also reports time remaining so the
        median above is actionable (≈ remaining / median = sims still affordable).
        Plain measurements only; the interpretation is the strategizer's. Returns
        None when this delegation wrote no timed rows (lookup-direct/off-ledger).
        """
        kpi = self.wall_per_delegation.get(str(delegation_id))
        if not kpi or not kpi.get("n"):
            return None

        def _dur(ms: float) -> str:
            s = ms / 1000.0
            if s < 90:
                return f"{s:.1f}s"
            if s < 5400:
                return f"{s / 60:.1f}min"
            return f"{s / 3600:.2f}h"

        lines = [
            "\n\n---",
            f"LEDGER KPIs ({delegation_id}, measured from the "
            f"{kpi['n']} rows this delegation wrote):",
            f"  per-eval wall-time: median {_dur(kpi['median_ms'])} · "
            f"max {_dur(kpi['max_ms'])}",
            f"  total eval wall-time (this delegation): {_dur(kpi['total_ms'])}",
            f"  ledger total so far: {self.n_rows} evaluations",
        ]
        if wall_remaining_s is not None and wall_budget_s:
            lines.append(
                f"  wall budget remaining: "
                f"{_dur(max(0.0, wall_remaining_s) * 1000)} of "
                f"{_dur(wall_budget_s * 1000)}"
            )
        if peak_rss_bytes:
            _gb = peak_rss_bytes / 1024 ** 3
            cap = (f" of {ram_cap_bytes / 1024 ** 3:.1f} GB hard cap"
                   if ram_cap_bytes else "")
            lines.append(f"  peak RAM (this delegation): {_gb:.2f} GB{cap}")
        return "\n".join(lines)

    # ------------------------------------------------------------------

    @classmethod
    def from_store(
        cls,
        store_dir: Path | str,
        *,
        fidelity_column: Optional[str] = None,
    ) -> RunStateSummary | None:
        """Build a RunStateSummary from the canonical store.

        Returns ``None`` if the store does not exist or is empty.
        Caches the result by (store_dir, mtime) so the DataFrame is not
        re-parsed on every call within a turn.

        Parameters
        ----------
        store_dir : Path or str
            The run-level directory that *contains* ``experiment_data/``.
        fidelity_column : str or None
            Name of the fidelity input column.  Grouping is only done when
            this column is present in the ExperimentData INPUT columns.
        """
        store_dir = Path(store_dir)
        csv_path = store_dir / "experiment_data" / "output.csv"
        if not csv_path.exists():
            return None

        key = str(store_dir)
        mtime = csv_path.stat().st_mtime
        with _RSS_CACHE_LOCK:
            cached = _RSS_CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]

        # Re-parse
        try:
            data = ExperimentData.from_file(project_dir=store_dir)
            df_in, df_out = data.to_pandas()
        except Exception:  # noqa: BLE001 — empty/corrupt store
            return None

        if df_out.empty:
            return None

        n_rows = len(df_out)
        n_per_delegation: dict = {}
        if "_delegation_id" in df_out.columns:
            n_per_delegation = (
                df_out["_delegation_id"]
                .value_counts()
                .to_dict()
            )
        n_per_source: dict = {}
        if "_source" in df_out.columns:
            n_per_source = (
                df_out["_source"]
                .value_counts()
                .to_dict()
            )

        # Fidelity: only group when column is in INPUT columns
        n_per_fidelity: dict | None = None
        if (
            fidelity_column is not None
            and df_in is not None
            and fidelity_column in df_in.columns
        ):
            n_per_fidelity = (
                df_in[fidelity_column]
                .value_counts()
                .to_dict()
            )

        # Output stats: numeric, non-provenance output columns
        output_stats: dict = {}
        for col in df_out.columns:
            if col in _PROVENANCE_COLS:
                continue
            series = df_out[col]
            try:
                import pandas as _pd
                numeric = _pd.to_numeric(series, errors="coerce").dropna()
            except Exception:  # noqa: BLE001
                continue
            if numeric.empty:
                continue
            output_stats[col] = {
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
            }

        # Mean per-eval wall-time (ms), overall. Generic: just average the
        # _wall_ms column over real eval rows, dropping NaN and the D000
        # precomputed pool (not real evals). Any per-group breakdown is a
        # groupby on this same column downstream — none is computed here.
        mean_eval_wall_ms: float | None = None
        wall_per_delegation: dict = {}
        if "_wall_ms" in df_out.columns:
            import pandas as _pd
            wall = _pd.to_numeric(df_out["_wall_ms"], errors="coerce")
            if "_source" in df_out.columns:
                wall = wall[df_out["_source"] != "precomputed_pool"]
            wall = wall.dropna()
            if not wall.empty:
                mean_eval_wall_ms = float(wall.mean())
            # Per-delegation wall breakdown — same column, grouped by the
            # delegation that wrote each row (drops the precomputed pool above).
            if "_delegation_id" in df_out.columns:
                _gid = df_out["_delegation_id"].reindex(wall.index)
                for _did, _grp in wall.groupby(_gid):
                    if _grp.empty:
                        continue
                    wall_per_delegation[str(_did)] = {
                        "n": int(_grp.size),
                        "median_ms": float(_grp.median()),
                        "max_ms": float(_grp.max()),
                        "total_ms": float(_grp.sum()),
                    }

        summary = cls(
            n_rows=n_rows,
            n_per_delegation=n_per_delegation,
            n_per_source=n_per_source,
            n_per_fidelity=n_per_fidelity,
            output_stats=output_stats,
            mean_eval_wall_ms=mean_eval_wall_ms,
            wall_per_delegation=wall_per_delegation,
        )
        with _RSS_CACHE_LOCK:
            _RSS_CACHE[key] = (mtime, summary)
        return summary

    # ------------------------------------------------------------------

    def format(self) -> str:
        """Compact human/LLM-readable block, typically ≤ 25 lines."""
        lines: list[str] = [
            f"Canonical store: {self.n_rows} total ledgered evaluations "
            "(AUTHORITATIVE evaluation count — cite THIS; never hand-compute "
            "or use a worker's self-reported count)",
        ]
        if self.n_per_delegation:
            parts = ", ".join(
                f"{k}={v}" for k, v in sorted(self.n_per_delegation.items())
            )
            lines.append(f"  rows per delegation: {parts}")
        if self.n_per_source:
            parts = ", ".join(
                f"{k}={v}" for k, v in sorted(self.n_per_source.items())
            )
            lines.append(f"  rows per source: {parts}")
        if self.n_per_fidelity is not None:
            parts = ", ".join(
                f"{k}={v}"
                for k, v in sorted(
                    self.n_per_fidelity.items(), key=lambda kv: kv[0]
                )
            )
            lines.append(f"  rows per fidelity: {parts}")
        if self.mean_eval_wall_ms is not None:
            lines.append(
                f"  mean eval wall-time: {self.mean_eval_wall_ms / 1000:.3g}s "
                f"({self.mean_eval_wall_ms:.0f}ms) — use for time-budget "
                "planning (≈ remaining_seconds / this = evals that still fit)"
            )
        if self.output_stats:
            lines.append("  output ranges:")
            for col, stats in sorted(self.output_stats.items()):
                lines.append(
                    f"    {col}: min={stats['min']:.4g}"
                    f"  max={stats['max']:.4g}"
                    f"  mean={stats['mean']:.4g}"
                )
        return "\n".join(lines)


# ==========================================================================


def experiment_stores(store_root: Path | str) -> list[Path]:
    """Every ExperimentData store in a run: the canonical/default store plus one
    per design experiment.

    A run holds one clean ``ExperimentData`` per experiment. The default store is
    ``<store_root>`` itself (its data under ``<store_root>/experiment_data``);
    each additional experiment is a sibling subdir ``<store_root>/<name>`` with
    its own ``experiment_data/``. ``experiment_data`` is the default store's own
    data dir, never an experiment name (registration forbids that name), so it is
    skipped. The default store is always included.
    """
    store_root = Path(store_root)
    stores = [store_root]
    try:
        for sub in sorted(store_root.iterdir()):
            if (sub.is_dir() and sub.name != "experiment_data"
                    and (sub / "experiment_data" / "output.csv").exists()):
                stores.append(sub)
    except (FileNotFoundError, OSError):
        pass
    return stores


def total_ledgered_evals(store_root: Path | str) -> int:
    """Total REAL oracle evaluations in the run: provenance-stamped rows
    ATTRIBUTED to a delegation, summed across every experiment store.

    This is deliberately the SAME set that delegation_evals (and the
    UNLEDGERED_EVALS guard) reads — the per-delegation stamped rows — so the
    reported number (run_status evals_used, the budget spent, the deliverable
    count) and the guarded number can no longer diverge. That divergence was the
    structural root of the recurring eval-accounting backdoors (post-mortem
    989e7daa): this used to sum n_rows = len(df_out), the RAW physical row count,
    which ALSO counted (a) D000 precomputed-pool rows — ground-truth data the
    code elsewhere says are "never counted as evaluations" — and (b) any
    UNSTAMPED rows appended to the store outside get_evaluator(). Both are now
    excluded: D000 by id, unstamped rows because value_counts() drops them from
    n_per_delegation. (Cross-store summing is preserved — a single store would
    miss the design-namespace stores; observed run 20260626T231202.) Never
    raises. NOTE: this does not SEAL the unstamped-write path (public
    ExperimentData.store() can still append rows); it stops such rows from
    inflating the COUNT, and detecting/refusing the write is a separate step.
    """
    total = 0
    for store in experiment_stores(store_root):
        s = RunStateSummary.from_store(store)
        if s is not None:
            total += sum(
                int(c) for did, c in s.n_per_delegation.items()
                if did and str(did) != "D000")
    return total


def unstamped_row_count(store_root: Path | str) -> int:
    """Physical rows in the stores that carry NO provenance stamp, summed across
    every experiment store. These are rows appended outside get_evaluator() (the
    public ``ExperimentData.store()`` write-door): value_counts() drops them from
    n_per_delegation, so they are excluded from the COUNT — but their existence
    is itself a signal (an eval that ran without attribution). This surfaces the
    gap so it is visible instead of silent. Never raises.

    A row is attributable iff its _delegation_id is truthy (D000, D001, …);
    unattributable rows are those with a MISSING stamp (NaN — dropped by
    value_counts, so absent from n_per_delegation) OR an EMPTY stamp ("" — a key
    in n_per_delegation but excluded from evals by the same truthiness test
    total_ledgered_evals uses). Both are counted here.
    """
    total = 0
    for store in experiment_stores(store_root):
        s = RunStateSummary.from_store(store)
        if s is not None:
            attributed = sum(int(c) for did, c in s.n_per_delegation.items()
                             if did)
            total += max(0, int(s.n_rows) - attributed)
    return total


def delegation_evals(store_root: Path | str, delegation_id: str) -> int:
    """Rows stamped with this delegation_id across EVERY experiment store.

    Provenance-based: a delegation's evaluations are found by its stamp wherever
    they landed, so the count is correct no matter HOW the experiment was selected
    — ``Delegate(namespace=...)`` OR ``get_evaluator(namespace=...)`` at the call
    site. This is what makes the unledgered-evals guard immune to the selection
    path (run 20260627T045747: a delegation that wrote to the 'ring' store via the
    call site was falsely flagged off-ledger because the guard only knew the
    Delegate-arg experiment). Never raises.
    """
    total = 0
    for store in experiment_stores(store_root):
        s = RunStateSummary.from_store(store)
        if s is not None:
            total += int(s.n_per_delegation.get(delegation_id, 0))
    return total


def load_experiments(
    store_root: Path | str | None = None,
) -> dict[str, ExperimentData]:
    """Load EVERY experiment store of a run as a dict ``{name: ExperimentData}``.

    A namespaced run holds one clean ``ExperimentData`` per experiment at nested
    paths — the default store at ``<root>/experiment_data/`` and each design
    experiment at ``<root>/<name>/experiment_data/`` — so a single
    ``ExperimentData.from_file`` (the single-study idiom) loads only the default
    store and silently misses the rest. This is the multi-namespace load idiom
    for pipeline.ipynb: one call returns them all, keyed by experiment name
    (the default/baseline store is ``"default"``).

    ``store_root`` defaults to ``$F3DASM_CANONICAL_STORE`` (set in the notebook's
    execution env), so the notebook body is just
    ``experiments = load_experiments()``. Empty/absent stores are skipped; an
    empty run yields ``{}``. Never raises on a missing store.
    """
    if store_root is None:
        store_root = os.environ.get("F3DASM_CANONICAL_STORE", "")
    store_root = Path(store_root)
    out: dict[str, ExperimentData] = {}
    for store in experiment_stores(store_root):
        name = "default" if store == store_root else store.name
        try:
            data = ExperimentData.from_file(project_dir=store)
        except Exception:  # noqa: BLE001 — empty/absent store
            continue
        if len(data) > 0:
            out[name] = data
    return out


def ledger_breakdown(store_root: Path | str) -> list[dict]:
    """Per-experiment, per-delegation eval counts across the whole run.

    Returns one entry per experiment store (the default store is named
    ``"default"``; each design experiment by its registered name) with::

        {"experiment": <name>, "total": <rows>, "per_delegation": {<id>: <n>}}

    This is the report-time provenance the agent could not otherwise see: it
    exposes exactly what the provenance accounting counts, so a writeup DERIVES
    its eval counts from the ledger instead of hardcoding stale plan numbers
    (run 20260628T001710 hardcoded 70 polar evals; the real ledger held 90 →
    UNGATED). Sorted: default first, then experiments alphabetically. Never
    raises; an empty/absent store yields an empty list.
    """
    store_root = Path(store_root)
    out: list[dict] = []
    for store in experiment_stores(store_root):
        s = RunStateSummary.from_store(store)
        if s is None:
            continue
        name = "default" if store == store_root else store.name
        out.append({
            "experiment": name,
            "total": int(s.n_rows),
            "per_delegation": {k: int(v) for k, v in s.n_per_delegation.items()},
        })
    return out


def duplicate_eval_stats(store_root: Path | str) -> dict[str, dict]:
    """Per-delegation duplicate-design-point counts across every store.

    An evaluation that re-derives a design point already FINISHED, unchanged,
    in the ledger wastes eval budget without adding evidence — this exposes
    that waste so a caller (ScienceMonitor) can nudge about it. Real incident
    (run example_study/20260713T221841): a delegation's three separate scripts
    each independently re-sampled and re-evaluated an identical seed=42 LHS
    design, so 122 of its 160 stamped rows (76%) were exact repeats of 38
    unique points — backlog #24.

    Coordinates are the non-provenance INPUT columns, rounded to 10 decimals
    (same convention `_reproduction_gate`'s `_ledger_snapshot` uses) so a
    faithful re-store of identical values doesn't false-count as new.
    Namespace-aware: sums across every experiment_stores() store, the same
    primitive `ledger_breakdown`/`delegation_evals` use.

    Returns ``{delegation_id: {"total_rows", "unique_points",
    "duplicate_rows", "worst": (coords_dict, count) | None}}``. Never raises;
    an empty/absent store yields ``{}``.
    """
    store_root = Path(store_root)
    per_deleg: dict[str, dict] = {}
    for store in experiment_stores(store_root):
        try:
            data = ExperimentData.from_file(project_dir=store)
            df_in, df_out = data.to_pandas()
        except Exception:  # noqa: BLE001
            continue
        if df_out.empty or "_delegation_id" not in df_out.columns:
            continue
        input_cols = [
            c for c in (df_in.columns if df_in is not None else [])
            if c not in _PROVENANCE_COLS
        ]
        rounded = df_in[input_cols].round(10) if input_cols else None
        for pos, did in enumerate(df_out["_delegation_id"]):
            if not did:
                continue
            did = str(did)
            coords = tuple(rounded.iloc[pos]) if rounded is not None else ()
            bucket = per_deleg.setdefault(
                did, {"counts": {}, "cols": input_cols})
            bucket["counts"][coords] = bucket["counts"].get(coords, 0) + 1

    out: dict[str, dict] = {}
    for did, bucket in per_deleg.items():
        counts = bucket["counts"]
        total_rows = sum(counts.values())
        unique_points = len(counts)
        worst = None
        if counts:
            worst_coords, worst_count = max(
                counts.items(), key=lambda kv: kv[1])
            if worst_count > 1:
                worst = (
                    dict(zip(bucket["cols"], worst_coords, strict=True)),
                    worst_count,
                )
        out[did] = {
            "total_rows": total_rows,
            "unique_points": unique_points,
            "duplicate_rows": total_rows - unique_points,
            "worst": worst,
        }
    return out


__all__ = [
    "InstrumentedDataGenerator",
    "RunStateSummary",
    "duplicate_eval_stats",
    "get_evaluator",
    "ledger_breakdown",
    "load_experiments",
    "load_inner_evaluator",
    "total_ledgered_evals",
    "delegation_evals",
    "unstamped_row_count",
    "experiment_stores",
]
