"""Instrumented DataGenerator: canonical concurrency-safe eval ledger.

Provides :class:`InstrumentedDataGenerator`, a wrapper that:

1. Delegates execution to an inner ``DataGenerator``.
2. Stamps provenance metadata (delegation_id, source, UTC timestamp)
   onto each returned ``ExperimentSample``.
3. Buffers samples and flushes them to a shared ``ExperimentData`` store
   under a ``FileLock`` so concurrent delegations cannot corrupt the
   ledger.

The factory that builds a configured instance from ``run_config.json`` is
:func:`.oracle_resolution.get_evaluator`; the read-only ledger summaries
live in :mod:`.ledger_summary`.

The ``fidelity_column`` parameter is accepted but unused; it exists for
forward-compatibility when fidelity-aware stamping is added.
"""
from __future__ import annotations

#                                                                      Modules
# ==========================================================================
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
from f3dasm.design import Domain
from filelock import FileLock

#                                                         Authorship & Credits
# ==========================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# ==========================================================================

# Columns the wrapper stamps on every ledgered row. Everything else in an
# output frame is a real output; everything else in an input frame is a
# design coordinate.
_PROVENANCE_COLS = frozenset({"_delegation_id", "_source", "_ts", "_wall_ms"})


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
    dedup_scope : str, optional
        ``"delegation"`` (default): dedup-on-write only recognizes a design
        as already-seen if THIS delegation itself wrote it — a genuinely
        concurrent campaign delegation may legitimately re-measure a design
        another delegation also measured; collapsing across delegations
        would corrupt that. ``"all"``: dedup against the WHOLE ledger
        regardless of which delegation wrote each row — the correct
        semantics for a validation replay of already-generated data (e.g.
        the reproduction gate re-executing a notebook's ``data_generation``
        cell, which must add ZERO new rows to satisfy the "LAZY"
        reproduction invariant) rather than a live campaign delegation
        genuinely exploring in parallel.
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
        dedup_scope: str = "delegation",
    ) -> None:
        self.inner = inner
        self.store_dir = Path(store_dir)
        if dedup_scope not in ("delegation", "all"):
            raise ValueError(
                f"dedup_scope must be 'delegation' or 'all', got {dedup_scope!r}")
        self.dedup_scope = dedup_scope
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

    def call(self, data, mode: str = "sequential", pass_id: bool = False,
              **kwargs):
        """Same as f3dasm's ``DataGenerator.call`` — EXCEPT ``mode="parallel"``
        is refused outright, never delegated to ``super().call()``.

        f3dasm's own ``mode="parallel"`` falls through to a LOCAL
        ``multiprocessing.Pool`` — it spawns every solve as a subprocess on
        THIS process's own host, which in this architecture is the run's
        shared, resource-constrained orchestration node (not a SLURM
        allocation). N concurrent Abaqus solves there is CPU oversubscription
        and OOM that kills the whole run, not just this evaluation — a
        documentation warning telling agents "never use this" is not a
        control; a host-safety hard cap that never lets the call start is
        (mem_cap_bytes is the other one, §4 of the working contract — a run
        must not be able to OOM the shared node it runs on). Real
        parallelism belongs on a cluster scheduler (e.g. one evaluation per
        SLURM array task, each with its own node's resources) — whatever
        submission helper the study provides for that — never a local pool.
        """
        if mode == "parallel":
            raise ValueError(
                "gen.call(mode='parallel', ...) is disallowed — it falls "
                "through to f3dasm's local multiprocessing.Pool, spawning "
                "every solve as a subprocess on THIS run's own shared "
                "orchestration node (CPU oversubscription + OOM that kills "
                "the whole run, not just this evaluation). Use "
                "mode='sequential' here; get real parallelism through the "
                "study's cluster-array submission path (one evaluation per "
                "SLURM array task, each with its own node's resources) "
                "instead."
            )
        return super().call(data, mode=mode, pass_id=pass_id, **kwargs)

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
                    # dedup_scope="all" (reproduction-gate/deliverable replay
                    # of already-generated data, stamped with a synthetic id
                    # that never matches the real generating delegation(s))
                    # skips this narrowing entirely — every row in the
                    # ledger counts as already-seen, regardless of who wrote
                    # it, since the whole point of that replay is to add
                    # ZERO new rows.
                    if (self.dedup_scope == "delegation"
                            and "_delegation_id" in df_out.columns):
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


__all__ = [
    "InstrumentedDataGenerator",
]
