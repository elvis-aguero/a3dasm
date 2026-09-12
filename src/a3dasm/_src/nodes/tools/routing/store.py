"""Read-only ledger/store query tools: RecallStore, OracleStatus, QueryStore,
HypothesisList, HypothesisGet, ReadProblemStatement. Declaration-gated and
shared verbatim between an orchestrating node and a leaf (see
nodes/leaf.py) via build_declared_shared_closures()."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Returned verbatim by RecallStore and QueryStore alike; one string so the
# two cannot drift into saying different things about the same condition.
_EMPTY_STORE = (
    "Canonical store is empty — no instrumented evaluations recorded yet."
)

_FLOAT_EQ_RE = re.compile(
    r"\b([A-Za-z_]\w*)\s*==\s*[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b"
)


def _float_equality_where_hint(where: str, joined) -> str:
    """Return a warning suffix if ``where`` exact-``==``s a float-dtype column,
    else "".

    QueryStore's ``where=`` is a pandas ``.query()`` expression: an exact ``==``
    against a float column silently excludes a row whose stored value differs
    by float representation error alone (e.g. a value round-tripped through
    CSV/JSON), and the caller has no way to tell that apart from a genuine
    absence — observed independently by the strategizer and 3 of 8 critic
    rounds in run 20260825T012642, each paying a diagnosis cycle before
    switching to an ``abs(x-v)<1e-6`` tolerance predicate. Heuristic, not a
    parser: only flags a bare ``col==literal`` pattern where ``col`` resolves
    to a float-dtype column — deliberately does NOT touch int/bool/str exact
    equality (e.g. ``feasible==1``), which is not float-precision-sensitive
    and should stay exact.
    """
    import pandas as _pd

    for col in _FLOAT_EQ_RE.findall(where or ""):
        if col in joined.columns and _pd.api.types.is_float_dtype(joined[col]):
            return (
                " HINT: where= exact-compared a float column (`==`) — float "
                "storage/round-trip error can make a real row silently not "
                "match. Prefer a tolerance predicate, e.g. "
                f"abs({col}-VALUE)<1e-6."
            )
    return ""


# f3dasm marks infeasible / failed designs with a large-magnitude sentinel
# output (e.g. ±1e9; the resonance study treats resonance <= -1e8 as
# infeasible). "Best" must never return such a placeholder.
_INFEASIBLE_SENTINEL_MAG = 1e8


def _select_best_index(values, n_best, minimize=True, sentinel_mag=_INFEASIBLE_SENTINEL_MAG):
    """Index of the n best *feasible* values.

    Coerces to numeric, drops NaN and large-magnitude infeasibility
    placeholders, then picks the smallest (minimize) or largest (maximize).
    Returns a possibly-empty index when no feasible values remain.
    """
    import pandas as pd
    s = pd.to_numeric(values, errors="coerce").dropna()
    s = s[s.abs() < sentinel_mag]
    if s.empty:
        return s.index
    chosen = s.nsmallest(n_best) if minimize else s.nlargest(n_best)
    return chosen.index


def _decode_str_list(raw) -> list | None:
    """Decode a JSON-array string / comma-separated string / bare string /
    list into a list[str], or None if raw is None. Same permissive decoding
    QueryStore already uses for delegation_ids — an LLM caller passes any of
    these shapes interchangeably."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(x) for x in raw]
    s = str(raw).strip()
    if s.startswith("["):
        import json as _json
        try:
            decoded = _json.loads(s)
            return [str(x) for x in decoded] if isinstance(decoded, list) else [s]
        except _json.JSONDecodeError:
            return [s]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [s]


def _select_columns(df, columns, *, always_keep=("_namespace",)):
    """Narrow df to `columns` (decoded via _decode_str_list) plus
    always_keep, preserving df's own column order. Returns (narrowed_df,
    error_or_None) — error is an ERROR string (listing available columns,
    the same convention QueryStore's where= already uses) when a requested
    column doesn't exist; never raises.

    columns=None returns df unchanged — narrowing rows already existed
    (where=/delegation_ids=/n_best=); this is the same capability for
    columns, since a where=-narrowed row set can still be a wall of text
    when each row itself has dozens of columns (observed: 449,879 chars from
    two independent QueryStore calls in run 20260816T013744, overflowing the
    critic's token limit on calls that had already correctly narrowed rows).
    """
    if columns is None:
        return df, None
    requested = _decode_str_list(columns) or []
    missing = [c for c in requested if c not in df.columns]
    if missing:
        return None, (
            f"ERROR: column(s) {missing} not found. "
            f"Available columns: {list(df.columns)}"
        )
    keep = [c for c in df.columns if c in requested or c in always_keep]
    return df[keep], None

class StoreTools:
    """The read-only ledger/store tools, bound to one node.

    Capability tools granted to ANY node by DECLARATION. Single source of
    truth: a tool is exposed iff the agent lists it in its ``tools``. These are
    topology-agnostic — the entry node, a mid-tier delegating node and a leaf
    (e.g. the critic) all resolve the run's store/ledger through
    ``node._resolve_run_dir()`` / ``node._read_ledger()``, so they behave
    identically everywhere, and they are plain framework closures (so the
    Claude and Ollama backends expose an identical surface). All read-only:
    they mutate nothing.
    """

    def __init__(self, node: Any) -> None:
        self.node = node

    # ── Resolving the run's stores ───────────────────────────────────────────

    def _all_store_dirs(self) -> list[Path]:
        """Every experiment store in this run: the default one plus namespaces.

        Resolves run_dir via the node (entry: from its notes dir; any worker:
        from the shared delegation-log path) so these tools are never dead,
        then every store in the run: the canonical/default store PLUS every
        design-namespace sibling (run_dir/experiment_data/<namespace>/). A bare
        run_dir/experiment_data read misses namespace evals entirely — the same
        gap LedgerBreakdown/ScienceMonitor/GetStatus/CancelDelegation already
        avoid by going through experiment_stores() (backlog #21).
        """
        rd = self.node._resolve_run_dir()
        if rd is None:
            return []
        from ....evaluation.ledger_summary import experiment_stores
        return experiment_stores(rd / "experiment_data")

    def _load_frames(self):
        """Every store's rows, concatenated and namespace-tagged.

        Loads and concatenates EVERY store (default + every design namespace)
        so filtering sees all of them — a bare single-store read misses rows
        that landed in a namespace store (backlog #21). Mismatched columns
        across namespaces just NaN-fill; this is an in-memory read-side merge
        for reporting, not a rewrite of the canonical store.

        Returns ``(df_in, df_out)``, or ``(None, None)`` when nothing is
        readable anywhere.
        """
        import pandas as _pd
        from f3dasm import ExperimentData

        # Not yet public; flip after bessagroup/f3dasm#351.
        from f3dasm._src.errors import EmptyFileError, ReachMaximumTriesError

        df_ins, df_outs = [], []
        for i, store in enumerate(self._all_store_dirs()):
            try:
                data = ExperimentData.from_file(project_dir=store)
                d_in, d_out = data.to_pandas()
            except (FileNotFoundError, EmptyFileError,
                    ReachMaximumTriesError):
                continue
            if d_out.empty:
                continue
            # Tag every row with its source store BEFORE concatenating —
            # once merged, a namespace row is otherwise indistinguishable
            # from a baseline row (same convention as RecallStore's
            # "default" if i == 0 else store.name).
            d_out = d_out.copy()
            d_out["_namespace"] = "default" if i == 0 else store.name
            df_ins.append(d_in)
            df_outs.append(d_out)
        if not df_outs:
            return None, None
        df_out = _pd.concat(df_outs, ignore_index=True)
        df_in = (
            _pd.concat(df_ins, ignore_index=True)
            if all(d is not None for d in df_ins) else None
        )
        return df_in, df_out

    # ── The tools ────────────────────────────────────────────────────────────

    def RecallStore(self) -> str:
        """Summary of the run's canonical evaluation ledger: rows per
        delegation/source, output ranges. Call before deciding the next
        delegation."""
        from ....evaluation.ledger_summary import RunStateSummary
        stores = self._all_store_dirs()
        blocks: list[tuple[str, str]] = []
        for i, store in enumerate(stores):
            summary = RunStateSummary.from_store(store)
            if summary is None:
                continue
            label = "default" if i == 0 else store.name
            blocks.append((label, summary.format()))
        if not blocks:
            return _EMPTY_STORE
        if len(blocks) == 1:
            return blocks[0][1]
        return "\n\n".join(f"[{label}]\n{body}" for label, body in blocks)

    def OracleStatus(self) -> str:
        """The CURRENT canonical oracle registration — reads
        run_config.json fresh on every call, never from memory of an
        earlier notification. Call this before asserting anything about
        "the registered entrypoint" (e.g. in a delegation brief):
        register_evaluator_entrypoint() can repoint it BETWEEN
        delegations — a datagenerator delegation that authors/extends
        the generator repoints the canonical entrypoint the moment its
        report is processed — and trusting a stale assumption here has
        already cost one wasted real evaluation job (a delegation
        silently exercising the wrong generator, caught only by an
        agent noticing bit-identical outputs across differing inputs;
        run 20260717T014507)."""
        rd = self.node._resolve_run_dir()
        if rd is None:
            return "ERROR: run directory not resolved yet."
        cfg_path = rd / "debug" / "run_config.json"
        if not cfg_path.exists():
            return "No run_config.json yet — no oracle registered."
        import json as _json
        try:
            cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, _json.JSONDecodeError) as exc:
            return f"ERROR: could not read run_config.json: {exc}"
        ep = cfg.get("evaluator_entrypoint")
        lookup = cfg.get("evaluator_lookup")
        lines = []
        if ep:
            lines.append(f"canonical entrypoint: {ep}")
        elif lookup:
            lines.append(f"canonical lookup pool: {lookup}")
        else:
            lines.append("canonical oracle: NOT YET REGISTERED")
        out_names = cfg.get("evaluator_output_names")
        if out_names:
            lines.append(f"output_names: {out_names}")
        for ns, ns_cfg in (cfg.get("oracles") or {}).items():
            lines.append(
                f"namespace {ns!r}: "
                f"{ns_cfg.get('evaluator_entrypoint')}"
            )
        return "\n".join(lines)

    def QueryStore(
        self,
        delegation_ids: str | list | None = None,
        source: str | None = None,
        namespace: str | None = None,
        n_best: int | None = None,
        output_name: str | None = None,
        minimize: bool = True,
        where: str | None = None,
        limit: int | None = None,
        columns: str | list | None = None,
    ) -> str:
        """Filtered view of the evaluation ledger (e.g. rows from D001+D003
        only). Use to ground claims or to select training subsets; cite row
        values from here as evidence.

        columns narrows which COLUMNS are shown (the same idea as where=/
        limit= narrowing which ROWS are shown) — e.g.
        columns=["f", "feasible", "margin"]. Without it every column is
        shown for every matching row, which can overflow the response
        token limit on a wide store even after where=/limit= has already
        narrowed the rows down to a handful — narrow both when a store
        has many columns. `_namespace` is always kept regardless of
        columns=, per the invariant above. A requested column that
        doesn't exist returns an ERROR string listing the available
        columns; it never raises.

        Every row is tagged with a `_namespace` column ("default" for the
        baseline study, or the design-namespace name for a row that landed
        in an experiment opened via Delegate(namespace=...)) — this is
        ALWAYS shown, not just when filtering, since a namespace row can
        otherwise look indistinguishable from a baseline row (e.g. a
        near-identical value with a different mechanism). Pass namespace=
        to restrict to one store; note `source=` is NOT this — it filters
        the `_source` provenance column (the study name, identical across
        every namespace), so it can never disambiguate namespaces.

        n_best returns the best rows by output_name: smallest when
        minimize=True (default), largest when minimize=False (set this for
        MAXIMIZATION objectives). Infeasible/placeholder rows (large-
        magnitude sentinel outputs) are never returned as 'best'.

        where is a pandas query() expression over the JOINED inputs+outputs
        frame, for a compound feasibility predicate in one call — e.g.
        where="feasible==1 and margin>=0.10 and stress_ratio<=1.0".
        Arithmetic on input columns works too, so a derived quantity needs
        no stored column: where="x2/(2*x1) >= 10". Combine with n_best to
        rank the feasible subset. A bad expression returns an ERROR string
        listing the available columns; it never raises.

        limit caps the default (non-n_best) listing (default 20); raise it to
        pull a larger feasible set once where= has narrowed the rows."""
        df_in, df_out = self._load_frames()
        if df_out is None:
            return _EMPTY_STORE

        # Same permissive decoding for every list-shaped argument — an LLM
        # caller passes JSON / comma-joined / bare / list interchangeably.
        d_ids = _decode_str_list(delegation_ids)

        mask, joined, err = _filter_mask(
            df_in, df_out, d_ids, source, namespace, where)
        if err is not None:
            return err
        filtered = df_out[mask]
        filtered_in = df_in[mask] if df_in is not None else None
        if filtered.empty:
            return _no_match_message(
                df_out, d_ids, source, namespace, where, joined)

        n_best, limit, minimize, err = _coerce_query_args(
            n_best, limit, minimize)
        if err is not None:
            return err

        if n_best is not None and output_name is not None:
            return _best_rows_table(
                filtered, filtered_in, output_name, n_best, minimize, columns)
        return _default_listing(filtered, filtered_in, limit, columns)

    def HypothesisList(self, hypothesis_ids: list | None = None) -> str:
        """List all hypotheses with id, status, belief, statement.

        Takes no real arguments — it always lists ALL hypotheses. The
        optional `hypothesis_ids` is accepted-and-ignored so a stray kwarg
        (agents confuse this with Delegate/AskForFeedback) returns the list
        instead of crashing the turn with a TypeError.
        """
        led = self.node._read_ledger()
        if led is None:
            return "ERROR: hypothesis ledger not available in this run."
        items = led.list_all()
        if not items:
            return "No hypotheses proposed yet."
        return "\n".join(
            f"- {h['id']} [{h['current_status']}]"
            f" (belief {h['belief']}): {h['statement']}"
            for h in items
        )

    def HypothesisGet(self, hypothesis_id: str) -> str:
        """Get full hypothesis entry including status_log."""
        led = self.node._read_ledger()
        if led is None:
            return "ERROR: hypothesis ledger not available in this run."
        import json as _json
        entry = led.get(hypothesis_id)
        if entry is None:
            return f"ERROR: hypothesis {hypothesis_id!r} not found."
        return _json.dumps(entry, indent=2)

    def ReadProblemStatement(self) -> str:
        """The run's PROBLEM_STATEMENT.md verbatim: what this run is
        actually trying to establish, and its stated success/termination
        criteria (e.g. "do not stop until a good design is found or the
        budget is exhausted"). Available to every agent, not just the
        strategizer or literature reviewer — call this whenever a
        delegation's own brief doesn't make clear what the run as a whole
        is for, or before judging whether a result actually satisfies
        what was asked."""
        if not self.node._study_dir:
            return "ERROR: study_dir not set for this run."
        ps_path = Path(self.node._study_dir) / "PROBLEM_STATEMENT.md"
        if not ps_path.exists():
            return f"ERROR: {ps_path} not found."
        return ps_path.read_text(encoding="utf-8")


# ── QueryStore's phases ──────────────────────────────────────────────────────
# Pure functions of the frames and the arguments: no node state, so each one
# is testable on a DataFrame alone.

def _filter_mask(df_in, df_out, d_ids, source, namespace, where):
    """Build the read-only row mask. Returns ``(mask, joined, error_or_None)``.

    ``joined`` is the inputs+outputs frame the ``where`` predicate ran against
    (None when no ``where`` was given) — the caller needs it to explain an
    empty result.
    """
    import pandas as _pd
    mask = _pd.Series([True] * len(df_out), index=df_out.index)
    if d_ids is not None and "_delegation_id" in df_out.columns:
        mask &= df_out["_delegation_id"].isin(d_ids)
    if source is not None and "_source" in df_out.columns:
        mask &= df_out["_source"] == source
    if namespace is not None:
        mask &= df_out["_namespace"] == namespace
    if not where:
        return mask, None, None
    # where: compound predicate over the joined inputs+outputs frame.
    # Read-only (query on an in-memory copy); folds into the mask so both
    # the n_best path and the list path respect it. Never raises.
    if df_in is not None:
        joined = _pd.concat(
            [df_in.reset_index(drop=True),
             df_out.reset_index(drop=True)], axis=1)
        joined = joined.loc[:, ~joined.columns.duplicated()]
    else:
        joined = df_out.reset_index(drop=True)
    try:
        keep = joined.query(where).index
    except Exception as exc:  # noqa: BLE001
        return None, joined, (
            f"ERROR: could not evaluate where={where!r}: {exc}. "
            f"Available columns: {list(joined.columns)}"
        )
    mask &= _pd.Series(df_out.index.isin(keep), index=df_out.index)
    return mask, joined, None


def _no_match_message(df_out, d_ids, source, namespace, where, joined) -> str:
    """Explain an empty result — and whether it is a TRUE zero.

    Unambiguous absence: state how many rows were SCANNED. A true zero
    (columns exist, predicate excluded everything) is distinct from a bad
    column, which returns an ERROR from _filter_mask — never a 0. EXCEPT: an
    exact ``==`` against a float column can produce this same "zero rows"
    outcome on a row that's really there but off by float representation
    error — that is NOT a true zero, so don't assert one; hedge instead (see
    _float_equality_where_hint).
    """
    applied = []
    if d_ids is not None:
        applied.append(f"delegation_ids={d_ids}")
    if source is not None:
        applied.append(f"source={source!r}")
    if namespace is not None:
        applied.append(f"namespace={namespace!r}")
    if where:
        applied.append(f"where={where!r}")
    crit = "; ".join(applied) or "(no filter)"
    hint = _float_equality_where_hint(where, joined) if where else ""
    verdict = (
        "NOT necessarily a true zero — see hint below." if hint
        else "TRUE zero — the columns exist and were scanned "
        "(a non-existent column returns an ERROR, not 0)."
    )
    return (
        f"0 of {len(df_out)} scanned ledger row(s) match "
        f"[{crit}]. {verdict}{hint}"
    )


def _coerce_query_args(n_best, limit, minimize):
    """Coerce the MCP string-in arguments. Returns ``(n_best, limit, minimize, err)``.

    MCP string-in tools may pass n_best/limit as strings ("5") — coerce to int
    (pandas nsmallest does ``if n <= 0``, which TypeErrors on a str), and
    minimize as "false". Same string-arg-decoding discipline as delegation_ids.
    """
    if n_best is not None:
        try:
            n_best = int(n_best)
        except (TypeError, ValueError):
            return None, None, None, (
                f"ERROR: n_best must be an integer, got {n_best!r}."
            )
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return None, None, None, (
                f"ERROR: limit must be an integer, got {limit!r}."
            )
    if isinstance(minimize, str):
        minimize = minimize.strip().lower() not in (
            "false", "0", "no", "max", "maximize"
        )
    return n_best, limit, minimize, None


def _best_rows_table(
    filtered, filtered_in, output_name, n_best, minimize, columns
) -> str:
    """The n best feasible rows by ``output_name``, with their input columns."""
    import pandas as _pd

    from ....evaluation.instrumented import _PROVENANCE_COLS

    if output_name not in filtered.columns:
        return (
            f"ERROR: output column {output_name!r} not found. "
            f"Available: {list(filtered.columns)}"
        )
    best_idx = _select_best_index(
        filtered[output_name], n_best, minimize=minimize
    )
    if len(best_idx) == 0:
        return (
            f"No feasible rows to rank by {output_name!r} "
            "(all matching rows are infeasibility placeholders)."
        )
    best_rows = filtered.loc[best_idx]
    # Include input columns + output_name + _delegation_id
    show_cols = []
    if filtered_in is not None:
        show_cols.extend(
            c for c in filtered_in.columns if c not in _PROVENANCE_COLS
        )
    show_cols.append(output_name)
    if "_delegation_id" in best_rows.columns:
        show_cols.append("_delegation_id")
    if "_namespace" in best_rows.columns:
        show_cols.append("_namespace")

    if filtered_in is not None:
        best_in = filtered_in.loc[best_idx, [
            c for c in filtered_in.columns if c in show_cols
        ]]
        combined = _pd.concat(
            [best_in, best_rows[
                [c for c in show_cols if c not in best_in.columns]
            ]], axis=1
        )
    else:
        combined = best_rows[
            [c for c in show_cols if c in best_rows.columns]
        ]
    combined, _col_err = _select_columns(combined, columns)
    if _col_err:
        return _col_err
    return combined.to_string(index=False)


def _default_listing(filtered, filtered_in, limit, columns) -> str:
    """Count + the first ``limit`` rows (default 20), inputs included.

    INPUT columns are included so a design's coordinates (ratio_a, ratio_b, …)
    are directly verifiable — not just its outputs. Narrow with
    where=/delegation_ids= or rank with n_best= to see the rest.
    """
    import pandas as _pd
    cap = 20 if limit is None else limit
    n_shown = min(cap, len(filtered))
    if filtered_in is not None:
        show = _pd.concat(
            [filtered_in.reset_index(drop=True),
             filtered.reset_index(drop=True)], axis=1)
        show = show.loc[:, ~show.columns.duplicated()]
    else:
        show = filtered
    subset = show.iloc[:n_shown]
    subset, _col_err = _select_columns(subset, columns)
    if _col_err:
        return _col_err
    more = len(filtered) - n_shown
    tail = (
        f"\n… {more} more not shown — pass limit=<n>, a tighter "
        f"where=, or n_best= to rank." if more > 0 else ""
    )
    return (
        f"{len(filtered)} rows match. Showing {n_shown}:\n"
        + subset.to_string(index=False) + tail
    )


# NOTE: WaitForProcess was superseded by the SDK-compatible Bash surface
# (run_in_background -> BashOutput -> KillShell), which lives with the Bash
# tool per backend (SDK-native on Claude; openai_compatible on the rest).


def build_declared_shared_closures(node, agent_tools) -> dict:
    """The read-only capability tools this agent DECLARED, by registered name.

    Single source of truth: a tool is exposed iff the agent lists it in its
    ``tools``. Granted identically to any node, whatever its topology.
    """
    t = StoreTools(node)
    available = {
        "RecallStore": t.RecallStore,
        "OracleStatus": t.OracleStatus,
        "QueryStore": t.QueryStore,
        "HypothesisList": t.HypothesisList,
        "HypothesisGet": t.HypothesisGet,
        "ReadProblemStatement": t.ReadProblemStatement,
    }
    return {name: fn for name, fn in available.items() if name in agent_tools}
