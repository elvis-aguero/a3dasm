"""Read-only views over the canonical evaluation ledger.

:class:`RunStateSummary` digests one ``ExperimentData`` store (cached by
mtime); the module functions sum across every experiment store of a run —
the provenance-based counts the budget, the guards, and the deliverable all
read from. Nothing here writes.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from f3dasm import ExperimentData

from .instrumented import _PROVENANCE_COLS

# ==========================================================================
# Module-level mtime cache: {str(store_dir): (mtime, RunStateSummary)}
_RSS_CACHE: dict[str, tuple[float, RunStateSummary]] = {}
# Guards _RSS_CACHE: the summary is read from main (closure) threads and
# refreshed from background delegation threads. Benign on CPython, but
# the lock makes it correct on free-threaded builds too.
_RSS_CACHE_LOCK = __import__("threading").Lock()


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
    "RunStateSummary",
    "delegation_evals",
    "duplicate_eval_stats",
    "experiment_stores",
    "ledger_breakdown",
    "load_experiments",
    "total_ledgered_evals",
    "unstamped_row_count",
]
