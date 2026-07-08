"""Re-home f3dasm-core behaviours a3dasm needs but stock f3dasm does not ship.

a3dasm depends on a released f3dasm and carries no copy of f3dasm core, so it
cannot patch core. Two behaviours it relies on were once added to a fork of
f3dasm core; this module reinstalls them on ``ExperimentData`` at import time,
using only the public surface (``to_pandas``, ``len``, the on-disk layout):

1. ``to_numpy()`` drops underscore-prefixed metadata columns (the provenance
   stamps ``_delegation_id`` / ``_source`` / ``_ts`` / ``_wall_ms``), so the
   returned array stays float instead of decaying to ``object`` dtype.
2. ``store()`` refuses a write that would SHRINK a PROTECTED store (a project
   dir marked with ``PROTECTED_STORE_SENTINEL``), so a stray partial
   ``.store()`` cannot clobber the metered canonical ledger.

Both are idempotent. Applied on top of a f3dasm that already has them, the
underscore drop is a no-op and the guard raises identically. The patch is
applied once, guarded by a sentinel attribute on ``ExperimentData``.
"""
from __future__ import annotations

from pathlib import Path

_APPLIED_FLAG = "_a3dasm_compat_applied"

# a3dasm owns these; they mirror f3dasm's on-disk layout constants.
PROTECTED_STORE_SENTINEL = ".f3dasm_protected"
_STORE_SUBFOLDER = "experiment_data"
_OUTPUT_CSV = "output.csv"
_JOBS_CSV = "jobs.csv"


def apply_f3dasm_compat() -> None:
    """Idempotently install the two ExperimentData behaviours a3dasm needs."""
    from f3dasm import ExperimentData

    if getattr(ExperimentData, _APPLIED_FLAG, False):
        return

    def to_numpy(self):
        """Return ``(input_array, output_array)`` with metadata columns dropped.

        Underscore-prefixed columns are provenance bookkeeping, not measured
        values; keeping them (some are strings) would force an ``object`` dtype.
        """
        df_input, df_output = self.to_pandas(keep_references=False)
        df_input = df_input.loc[
            :, ~df_input.columns.astype(str).str.startswith("_")
        ]
        df_output = df_output.loc[
            :, ~df_output.columns.astype(str).str.startswith("_")
        ]
        return df_input.to_numpy(), df_output.to_numpy()

    _orig_store = ExperimentData.store

    def store(self, project_dir=None, copy_references=False):
        """``ExperimentData.store`` guarded against corrupting a PROTECTED store.

        Two monotonicity guards on a store marked with the sentinel: refuse a
        write that would SHRINK the row count, and refuse one that would reset a
        row the oracle already marked FINISHED. Both fail open on a read error so
        a genuine write is never blocked by a parse hiccup.
        """
        pdir = (
            Path(project_dir)
            if project_dir is not None
            else getattr(self, "_project_dir", None)
        )
        if pdir is not None and (pdir / PROTECTED_STORE_SENTINEL).exists():
            subdir = pdir / _STORE_SUBFOLDER

            # (1) Shrink guard. Count LOGICAL csv records, not physical lines:
            # an output value can contain embedded newlines (e.g. an array repr
            # in one quoted field), so a raw line count over-counts and would
            # falsely reject a valid superset write.
            existing_out = subdir / _OUTPUT_CSV
            if existing_out.exists():
                try:
                    import csv as _csv

                    with open(existing_out, newline="") as f:
                        existing_rows = max(
                            sum(1 for _ in _csv.reader(f)) - 1, 0)  # - header
                except OSError:
                    existing_rows = 0
                if len(self) < existing_rows:
                    raise RuntimeError(
                        "Refusing to overwrite the PROTECTED canonical store at "
                        f"{pdir}: it holds {existing_rows} rows but this store() "
                        f"would write only {len(self)}, destroying "
                        f"{existing_rows - len(self)} metered evaluations. The "
                        "canonical store is written ONLY via get_evaluator(); "
                        "store your own ExperimentData to a different project_dir."
                    )

            # (2) FINISHED is monotonic. Refuse resetting a FINISHED row to a
            # non-finished status (a worker calling .store() after gen.call()).
            existing_jobs = subdir / _JOBS_CSV
            if existing_jobs.exists():
                try:
                    import pandas as _pd

                    disk = _pd.read_csv(
                        existing_jobs, index_col=0).iloc[:, 0].astype(str)
                    mine = self.jobs.astype(str)
                    regressed = [
                        i for i, s in disk.items()
                        if s.upper() == "FINISHED"
                        and str(mine.get(i, "")).upper() != "FINISHED"
                    ]
                    if regressed:
                        raise RuntimeError(
                            "Refusing to overwrite the PROTECTED canonical store "
                            f"at {pdir}: this store() would reset {len(regressed)}"
                            " FINISHED evaluation(s) to a non-finished status "
                            "(typically a worker calling data.store() after "
                            "gen.call()). The canonical store is written ONLY via "
                            "get_evaluator(); store your own ExperimentData to a "
                            "different project_dir."
                        )
                except RuntimeError:
                    raise
                except Exception:  # noqa: BLE001 — fail open on a read error
                    pass
        return _orig_store(
            self, project_dir=project_dir, copy_references=copy_references
        )

    ExperimentData.to_numpy = to_numpy
    ExperimentData.store = store
    setattr(ExperimentData, _APPLIED_FLAG, True)
