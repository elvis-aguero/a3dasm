"""The strategizer's tool closures (Delegate/Parallel/GetStatus/Done/FollowUp/
WriteNote/ReadNote/WriteDeliverable/RecallStore/QueryStore/AskForFeedback +
hypothesis tools). Built per-node; the node is passed in so closures reach its
state. Extracted verbatim from StrategizerNode._build_routing_closures."""
from __future__ import annotations

import re
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from ...tool_catalog import tool_examples
from .._constants import backstop_enabled, run_backstop_multiple
from ..parsing import (
    _classify_response,
    _parse_verdict,
    _reconcile_delegation_evals,
    _stamped_eval_count,
)


def _strip_leading_md_header(text: str) -> str:
    """Drop a single leading markdown header line from author-supplied cell text.

    The notebook cell tools PREPEND the canonical heading themselves (e.g.
    ``## Hypotheses`` for the hypotheses cell, ``### doe`` for a pillar's
    WHY-explainer). When the author also opens their content with a header, the
    cell renders a duplicated heading (observed in run 20260623T212346:
    ``## Hypotheses\\n\\n## Hypotheses``, ``### analysis\\n\\n### analysis``). The
    tool owns the heading, so we strip a leading ``#``-header here to guarantee
    exactly one. Only a LEADING header is removed — sub-headings inside the body
    are preserved.
    """
    return re.sub(r"^\s*#{1,6}[^\n]*(?:\n+|$)", "", (text or "").lstrip(), count=1)

# Roles whose delegations actually reach the ground-truth oracle and so are
# subject to the eval-ledger guards (raw-oracle nudge, unledgered bounce,
# off-ledger reconciliation). Role-based (not node-name-based) so it stays
# forward-compatible across node renames. The literature_reviewer (no oracle),
# datagenerator (legitimately builds/validates the oracle), and critic/
# strategizer are NOT evaluators and must be exempt — else they get bounced /
# nudged for work that never touches get_evaluator(). DebuggerAgent inherits
# role "implementer"; "debugger" is listed too for when it carries its own.
_LEDGER_GUARD_ROLES = frozenset({"implementer", "debugger"})


# Forward-compatible delegation-target resolution. Agents repeatedly name a
# target by CAPABILITY rather than the exact graph node name — e.g.
# "pipeline"/"pipeline_executor" for the implementer (the "pipeline executor"),
# "data_generation" for the datagenerator — and bounce off "unknown target".
# Resolve in order: exact node name -> normalized name (case/separator-
# insensitive) -> normalized role. Resolution is by live node name or role
# only — there is NO hardcoded capability-synonym table. The strategizer prompt
# names targets by their hint/role, so it does not invent capability words like
# 'pipeline'/'oracle'; an unresolvable target returns None and the caller errors
# with the valid-target list (the agent then self-corrects). Resolving by role
# (not node name) keeps it forward-compatible across node renames.


def _norm_target(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


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


def resolve_target(
    requested: str, outgoing: list[str], roles: dict[str, str]
) -> str | None:
    """Map a requested delegation target to a valid outgoing node name, or None.

    ``roles`` maps node name -> its configured role. Resolution is exact-name →
    normalized-name → normalized-role. Only a unique, confident match resolves;
    anything else returns None (caller errors)."""
    if requested in outgoing:
        return requested
    rn = _norm_target(requested)
    if not rn:
        return None
    for t in outgoing:  # normalized node name (case/separator-insensitive)
        if _norm_target(t) == rn:
            return t
    for t in outgoing:  # normalized role
        if _norm_target(roles.get(t, "")) == rn:
            return t
    return None


# Post-Done exit interview for the strategizer. Asked as a SEPARATE turn only
# after the critic accepted the conclusion — so the strategizer never carries
# the interview in its working context (no pollution). It answers with one more
# Done() whose summary is just a ### Retrospective block.
_EXIT_INTERVIEW = (
    "Your conclusion has been accepted by the critic and recorded — the run "
    "is effectively closed. One last thing before we finalise: a quick "
    "question about the SYSTEM you worked within (its rules, tools, and the "
    "monitor/critic feedback), NOT the science. Call Done() ONE more time "
    "with a summary containing only a ### Retrospective block:\n"
    "- CONSISTENCY: ok | flagged — did any rule, tool, monitor message, or "
    "critic finding contradict another, or contradict what you were told "
    "elsewhere (e.g. a rule that rejected evidence you believe was correct)? "
    "Write 'flagged' and QUOTE both sides; otherwise 'ok'. (Most important.)\n"
    "- DECISION: the one strategic choice you were least sure the system "
    "wanted, and why you made it.\n"
    "- FRICTION: anything counterintuitive about the rules/tools, INCLUDING what "
    "you recovered from (an errored tool call, a wrong-guessed tool name), not "
    "just blockers; 'none' only if truly zero.\n"
    "- BLOCKED: was there anything you NEEDED to do your job but COULDN'T — a "
    "missing tool, permission, or way to test/inspect your own work (e.g. no "
    "way to run or debug a deliverable you had to author)? Name it specifically, "
    "or 'none'. (We want CAPABILITY GAPS, not just counterintuitive rules — be "
    "honest; an unreported gap can't be fixed.)\n"
    "This will NOT reopen the run."
)

# Retrospective for runs that did NOT pass (UNGATED / FAILED). The most painful
# runs carry the most friction signal, yet they used to close with no interview
# at all — so capability gaps (e.g. "I couldn't run my own deliverable") were
# never surfaced. Capture them here, with the same BLOCKED probe.
_FAILED_RETROSPECTIVE = (
    "The run is closing WITHOUT a passing conclusion (it is recorded as "
    "UNGATED/FAILED — this is final and will NOT reopen). Before it finalises, "
    "a quick retrospective about the SYSTEM, not the science. Call Done() ONE "
    "more time with a summary containing only a ### Retrospective block:\n"
    "- BLOCKED: the single biggest thing you NEEDED but COULDN'T do — a missing "
    "tool, permission, or way to test/inspect your own work (e.g. no way to run "
    "or debug the deliverable you had to author). Name it specifically. (This "
    "is the most important field — be candid; an unreported gap can't be "
    "fixed.)\n"
    "- BLOCKER: in one line, the proximate reason the run did not pass.\n"
    "- FRICTION: any rule/tool that worked against you, INCLUDING what you "
    "recovered from (an errored call, a wrong-guessed tool name), not just "
    "blockers; 'none' only if truly zero.\n"
    "This will NOT reopen the run."
)


def build_declared_shared_closures(node, agent_tools) -> dict:
    """Capability tools granted to ANY node type by DECLARATION.

    Single source of truth: a tool is exposed iff the agent lists it in its
    `tools`. These are node-type-agnostic — the entry strategizer, a delegating
    worker (implementer/datagenerator), and a leaf worker (critic) resolve the
    run's store/ledger through node._resolve_run_dir()/node._read_ledger(), so
    they behave identically everywhere, and they are plain framework closures
    (so Claude/Ollama backends expose an identical surface). These are all
    read-only and mutate nothing.
    """
    out: dict = {}

    def _all_store_dirs() -> list[Path]:
        # Resolve run_dir via the node (entry: from its notes dir; any worker:
        # from the shared delegation-log path) so these tools are never dead,
        # then every store in the run: the canonical/default store PLUS every
        # design-namespace sibling (run_dir/experiment_data/<namespace>/). A
        # bare run_dir/experiment_data read misses namespace evals entirely —
        # the same gap LedgerBreakdown/ScienceMonitor/GetStatus/CancelDelegation
        # already avoid by going through experiment_stores() (backlog #21).
        rd = node._resolve_run_dir()
        if rd is None:
            return []
        from ...instrumented import experiment_stores
        return experiment_stores(rd / "experiment_data")

    if "RecallStore" in agent_tools:
        def RecallStore() -> str:
            """Summary of the run's canonical evaluation ledger: rows per
            delegation/source, output ranges. Call before deciding the next
            delegation."""
            from ...instrumented import RunStateSummary
            stores = _all_store_dirs()
            blocks: list[tuple[str, str]] = []
            for i, store in enumerate(stores):
                summary = RunStateSummary.from_store(store)
                if summary is None:
                    continue
                label = "default" if i == 0 else store.name
                blocks.append((label, summary.format()))
            if not blocks:
                return (
                    "Canonical store is empty — no instrumented "
                    "evaluations recorded yet."
                )
            if len(blocks) == 1:
                return blocks[0][1]
            return "\n\n".join(f"[{label}]\n{body}" for label, body in blocks)
        out["RecallStore"] = RecallStore

    if "QueryStore" in agent_tools:
        def QueryStore(
            delegation_ids: str | list | None = None,
            source: str | None = None,
            n_best: int | None = None,
            output_name: str | None = None,
            minimize: bool = True,
            where: str | None = None,
            limit: int | None = None,
        ) -> str:
            """Filtered view of the evaluation ledger (e.g. rows from D001+D003
            only). Use to ground claims or to select training subsets; cite row
            values from here as evidence.

            n_best returns the best rows by output_name: smallest when
            minimize=True (default), largest when minimize=False (set this for
            MAXIMIZATION objectives). Infeasible/placeholder rows (large-
            magnitude sentinel outputs) are never returned as 'best'.

            where is a pandas query() expression over the JOINED inputs+outputs
            frame, for a compound feasibility predicate in one call — e.g.
            where="coilable==1 and max_compressive_strain>=0.90 and
            max_local_strain<=0.02". Arithmetic on input columns works too, so a
            derived quantity needs no stored column:
            where="ratio_pitch/(2*ratio_b) >= 10". Combine with n_best to rank
            the feasible subset. A bad expression returns an ERROR string listing
            the available columns; it never raises.

            limit caps the default (non-n_best) listing (default 20); raise it to
            pull a larger feasible set once where= has narrowed the rows."""
            import json as _json

            from f3dasm import ExperimentData

            # Not yet public; flip after bessagroup/f3dasm#351.
            from f3dasm._src.errors import (
                EmptyFileError,
                ReachMaximumTriesError,
            )

            from ...instrumented import _PROVENANCE_COLS

            stores = _all_store_dirs()
            if not stores:
                return (
                    "Canonical store is empty — no instrumented "
                    "evaluations recorded yet."
                )
            # Load and concatenate EVERY store (default + every design
            # namespace) before filtering — a bare single-store read misses
            # rows that landed in a namespace store (backlog #21). Mismatched
            # columns across namespaces just NaN-fill; this is an in-memory
            # read-side merge for reporting, not a rewrite of the canonical
            # store.
            import pandas as _pd
            df_ins, df_outs = [], []
            for store in stores:
                try:
                    data = ExperimentData.from_file(project_dir=store)
                    d_in, d_out = data.to_pandas()
                except (FileNotFoundError, EmptyFileError,
                        ReachMaximumTriesError):
                    continue
                if d_out.empty:
                    continue
                df_ins.append(d_in)
                df_outs.append(d_out)
            if not df_outs:
                return (
                    "Canonical store is empty — no instrumented "
                    "evaluations recorded yet."
                )
            df_out = _pd.concat(df_outs, ignore_index=True)
            df_in = (
                _pd.concat(df_ins, ignore_index=True)
                if all(d is not None for d in df_ins) else None
            )

            # Decode delegation_ids: JSON / comma / bare / list
            d_ids: list[str] | None = None
            if delegation_ids is not None:
                if isinstance(delegation_ids, list):
                    d_ids = [str(x) for x in delegation_ids]
                elif isinstance(delegation_ids, str):
                    raw = delegation_ids.strip()
                    if raw.startswith("["):
                        try:
                            decoded = _json.loads(raw)
                            d_ids = (
                                [str(h) for h in decoded]
                                if isinstance(decoded, list)
                                else [raw]
                            )
                        except _json.JSONDecodeError:
                            d_ids = [raw]
                    elif "," in raw:
                        d_ids = [
                            p.strip() for p in raw.split(",")
                            if p.strip()
                        ]
                    else:
                        d_ids = [raw]

            # Apply filters (read-only: build a boolean mask)
            mask = _pd.Series([True] * len(df_out), index=df_out.index)
            if d_ids is not None and "_delegation_id" in df_out.columns:
                mask &= df_out["_delegation_id"].isin(d_ids)
            if source is not None and "_source" in df_out.columns:
                mask &= df_out["_source"] == source

            # where: compound predicate over the joined inputs+outputs frame.
            # Read-only (query on an in-memory copy); folds into the mask so both
            # the n_best path and the list path respect it. Never raises.
            if where:
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
                    return (
                        f"ERROR: could not evaluate where={where!r}: {exc}. "
                        f"Available columns: {list(joined.columns)}"
                    )
                mask &= _pd.Series(
                    df_out.index.isin(keep), index=df_out.index)

            filtered = df_out[mask]
            filtered_in = df_in[mask] if df_in is not None else None

            if filtered.empty:
                return "No rows match the given filters."

            # n_best: return the n rows with smallest output_name value.
            # MCP string-in tools may pass n_best as a string ("5") — coerce
            # to int (pandas nsmallest does `if n <= 0`, which TypeErrors on
            # a str). Same string-arg-decoding discipline as delegation_ids.
            if n_best is not None:
                try:
                    n_best = int(n_best)
                except (TypeError, ValueError):
                    return (
                        f"ERROR: n_best must be an integer, got "
                        f"{n_best!r}."
                    )
            if limit is not None:
                try:
                    limit = int(limit)
                except (TypeError, ValueError):
                    return f"ERROR: limit must be an integer, got {limit!r}."
            if isinstance(minimize, str):  # MCP string-in may pass "false"
                minimize = minimize.strip().lower() not in (
                    "false", "0", "no", "max", "maximize"
                )
            if n_best is not None and output_name is not None:
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
                    input_cols = [
                        c for c in filtered_in.columns
                        if c not in _PROVENANCE_COLS
                    ]
                    show_cols.extend(input_cols)
                show_cols.append(output_name)
                if "_delegation_id" in best_rows.columns:
                    show_cols.append("_delegation_id")
                show_cols = [c for c in show_cols if c in best_rows.columns]

                if filtered_in is not None:
                    best_in = filtered_in.loc[best_idx, [
                        c for c in filtered_in.columns
                        if c in show_cols
                    ]]
                    combined = _pd.concat(
                        [best_in, best_rows[
                            [c for c in show_cols
                             if c not in best_in.columns]
                        ]], axis=1
                    )
                else:
                    combined = best_rows[
                        [c for c in show_cols if c in best_rows.columns]
                    ]
                return combined.to_string(index=False)

            # Default: return count + first `limit` rows (default 20). Narrow
            # with where=/delegation_ids= or rank with n_best= to see the rest.
            cap = 20 if limit is None else limit
            n_shown = min(cap, len(filtered))
            subset = filtered.iloc[:n_shown]
            more = len(filtered) - n_shown
            tail = (
                f"\n… {more} more not shown — pass limit=<n>, a tighter "
                f"where=, or n_best= to rank." if more > 0 else ""
            )
            return (
                f"{len(filtered)} rows match. Showing {n_shown}:\n"
                + subset.to_string(index=False) + tail
            )
        out["QueryStore"] = QueryStore

    if "HypothesisList" in agent_tools:
        def HypothesisList(hypothesis_ids: list | None = None) -> str:
            """List all hypotheses with id, status, belief, statement.

            Takes no real arguments — it always lists ALL hypotheses. The
            optional `hypothesis_ids` is accepted-and-ignored so a stray kwarg
            (agents confuse this with Delegate/AskForFeedback) returns the list
            instead of crashing the turn with a TypeError.
            """
            led = node._read_ledger()
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
        out["HypothesisList"] = HypothesisList

    if "HypothesisGet" in agent_tools:
        def HypothesisGet(hypothesis_id: str) -> str:
            """Get full hypothesis entry including status_log."""
            led = node._read_ledger()
            if led is None:
                return "ERROR: hypothesis ledger not available in this run."
            import json as _json
            entry = led.get(hypothesis_id)
            if entry is None:
                return f"ERROR: hypothesis {hypothesis_id!r} not found."
            return _json.dumps(entry, indent=2)
        out["HypothesisGet"] = HypothesisGet

    # NOTE: WaitForProcess was superseded by the SDK-compatible Bash surface
    # (run_in_background -> BashOutput -> KillShell), which lives with the Bash
    # tool per backend (SDK-native on Claude; openai_compatible on the rest).

    return out


def build_routing_tools(node) -> dict:
    route = node._route
    outgoing = node._outgoing
    study_dir = node._study_dir
    interactive = node._interactive

    # Build target hints from each connected agent's description.
    _target_hints = "\n  ".join(
        f"{t}: {node._spec.nodes[t].description}"
        for t in outgoing
        if t in node._spec.nodes
    )
    _delegate_doc = (
        "Fire a task to a connected agent.\n\n"
        "CHOOSE THE MODE DELIBERATELY — neither is the default-good answer:\n"
        "  wait=False (async): returns a D### ID immediately and the worker\n"
        "    runs in the background. Multiple workers can then be alive at\n"
        "    once — which is the ONLY way Confer (live worker-to-worker\n"
        "    messaging) can do anything, and the only way the run's wall-clock\n"
        "    is the longest single chain rather than the sum of every\n"
        "    delegation. Costs you GetStatus(id) polling to collect results.\n"
        "  wait=True (sync): blocks until the worker finishes and returns its\n"
        "    report directly, with zero polling. Simpler when this task must\n"
        "    fully finish before you can even decide the next one.\n"
        "  Ask yourself: could this run alongside other work, or might a peer\n"
        "  worker need to Confer with it mid-flight? If yes, async. If it is a\n"
        "  hard prerequisite for your very next decision, sync. Decide per\n"
        "  delegation; do not pick one mode reflexively for the whole run.\n\n"
        "CONTEXT PACKAGING: workers start each delegation with no memory of\n"
        "prior delegations. Include in the task message everything the worker\n"
        "needs: relevant paths, key findings from prior delegations, and the\n"
        "precise question to answer.\n\n"
        "hypothesis_ids must be non-empty when the ledger is active.\n"
        "The worker writes exclusively to {id}/ (relative to their workspace\n"
        "in debug/delegations/).\n\n"
        "Set is_falsification_attempt=True when this delegation attacks"
        " a hypothesis's stated falsification criterion.\n\n"
        "phase (optional): the f3dasm process stage this delegation advances —"
        " one of literature, doe, data_generation, ml, optimization, setup."
        " Tags the work's intent in the larger data-driven process; used by"
        " milestone gates, timing, and the critic.\n\n"
        "namespace (optional): open a NEW design parametrization as its own"
        " oracle + ledger. Leave it UNSET (the default) for the baseline study —"
        " that is most problems. Set namespace='some_name' only when the"
        " scientific question is a fundamentally different design REPRESENTATION"
        " (new variables / new geometry — e.g. 'elliptical_rings'): delegate a"
        " datagenerator with that namespace to build + register its oracle, then"
        " delegate implementers with the SAME namespace to evaluate in it. Each"
        " namespace keeps its own isolated ledger and the baseline is untouched;"
        " results compare across namespaces only insofar as they share the"
        " objective evaluator. A tool for creativity, not a requirement — open as"
        " many (or as few) as the science needs.\n\n"
        f"Available targets:\n  {_target_hints}"
    )

    def _falsification_checkpoint(delegation_id: str) -> str:
        """Read-time ritual text for a freshly-read Done report.

        Forces the strategizer to classify whether this delegation was a
        falsification ATTEMPT of a registered hypothesis and, if so, link it
        and record the verdict against the hypothesis's pre-registered
        (immutable) prediction. Returns "" when there is nothing to reconcile.
        Fires once per delegation (sets reconciled=True) to avoid nagging.
        """
        if node._ledger is None:
            return ""
        try:
            hyps = node._ledger.list_all()
        except Exception:  # noqa: BLE001
            return ""
        if not hyps:
            return ""

        def _pred(hid: str) -> str:
            e = node._ledger.get(hid) or {}
            return (
                e.get("prediction")
                or e.get("falsification_criterion")
                or "(no prediction on record)"
            )

        with node._registry_lock:
            entry = node._registry.get(delegation_id)
            if (
                entry is None
                or entry.get("status") != "Done"
                or entry.get("reconciled")
            ):
                return ""
            is_fals = bool(entry.get("is_falsification_attempt"))
            linked = list(entry.get("hypothesis_ids") or [])
            entry["reconciled"] = True  # fire once

        if is_fals and linked:
            tested = "; ".join(
                f"{h} (prediction: \"{_pred(h)}\")" for h in linked)
            return (
                f"⚖ FALSIFICATION CHECKPOINT — {delegation_id} was declared "
                f"a falsification attempt of {tested}. Record the VERDICT now "
                "via HypothesisUpdate(<id>, "
                "status=SUPPORTED|FALSIFIED|INCONCLUSIVE, "
                f"evidence={{'delegation': '{delegation_id}', "
                "'numbers': {…}}), judging THIS report against that "
                "pre-registered prediction. Do not move on with the "
                "hypothesis left OPEN."
            )
        open_h = [h for h in hyps if h.get("current_status") == "OPEN"]
        if not open_h:
            return ""  # nothing open to test → don't nag exploration
        listing = "; ".join(
            f"{h['id']} (prediction: \"{_pred(h['id'])}\")" for h in open_h)
        return (
            f"⚖ FALSIFICATION CHECKPOINT — was {delegation_id} an attempt to "
            "test a registered hypothesis's pre-registered prediction? Open: "
            f"{listing}. If YES: LinkFalsificationAttempt('{delegation_id}', "
            "'<id>') then record the verdict with HypothesisUpdate against "
            f"that prediction. Link ONLY if {delegation_id} genuinely tested "
            "that prediction — do not retrofit an exploratory result onto a "
            "hypothesis. If it was exploration/setup, there is no hypothesis "
            "to attach — continue (no action needed)."
        )

    @tool_examples(
        "Delegate('implementer', 'Run a 50-pt Latin sweep of t/L in "
        "[0.02,0.20]; evaluate via get_evaluator(); report top-5 by "
        "buckling_load_norm + the results CSV path + feasible count.', "
        "'top-5 t/L, their values, CSV path, n feasible', "
        "hypothesis_ids=['H1','H2'])",
        "Delegate('implementer', 'Falsification probe: dense grid n=20 of "
        "t/L in [0.10,0.14]; does any point beat buckling_load_norm 1.47?', "
        "'best value in range + pass/fail', hypothesis_ids=['H1'], "
        "is_falsification_attempt=True, phase='optimization')",
    )
    def Delegate(
        target: str,
        intent: str,
        expected_report: str,
        hypothesis_ids: list | None = None,
        wait: bool = False,
        is_falsification_attempt: bool = False,
        phase: str | None = None,
        namespace: str | None = None,
    ) -> str:
        _resolved = resolve_target(
            target, outgoing,
            {t: getattr(node._spec.nodes.get(t), "role", "") for t in outgoing},
        )
        if _resolved is None:
            return (
                f"ERROR: unknown target {target!r}."
                f" Valid targets: {outgoing}"
            )
        if _resolved != target:
            # Forward-compatible alias resolution: the agent named the target by
            # capability (e.g. 'pipeline_executor' -> 'implementer'). Proceed and
            # record it for observability instead of bouncing the agent.
            node._record_intervention(
                "TARGET_ALIAS", _resolved,
                f"delegation target {target!r} resolved to {_resolved!r}.",
            )
            target = _resolved
        worker_template = node._worker_adapters.get(target)
        if worker_template is None:
            return (
                f"ERROR: no worker adapter for target {target!r}."
                f" Available: {list(node._worker_adapters)}"
            )

        if isinstance(wait, str):  # MCP string-in tools may pass "false"
            wait = wait.strip().lower() not in ("false", "0", "no", "")

        # Enforce hypothesis linkage when ledger is active.
        # LLMs pass strings in several shapes — decode all of
        # them: '["H1","H2"]' (JSON), 'H1, H2' (joined), 'H1'.
        if isinstance(hypothesis_ids, str):
            raw = hypothesis_ids.strip()
            if raw.startswith("["):
                import json as _json
                try:
                    decoded = _json.loads(raw)
                    h_ids: list[str] = [
                        str(h) for h in decoded
                    ] if isinstance(decoded, list) else [raw]
                except _json.JSONDecodeError:
                    h_ids = [raw]
            elif "," in raw:
                h_ids = [
                    p.strip() for p in raw.split(",") if p.strip()
                ]
            else:
                h_ids = [raw]
        else:
            h_ids = [str(h) for h in (hypothesis_ids or [])]
        if node._ledger is not None:
            if not h_ids:
                # Defer this requirement while the process backlog is still open
                # (setup phase): you propose hypotheses AFTER engaging with the
                # problem and setting up the oracle, so early setup delegations
                # (oracle wrapping, literature review) have nothing to link to
                # yet. Once the backlog is cleared, every delegation must cite a
                # hypothesis. (Tied to the existing milestone backlog, not a
                # phase taxonomy the agent controls.)
                _ms = getattr(node, "_milestones", None)
                _backlog_open = _ms is not None and bool(_ms.pending())
                if not _backlog_open:
                    return (
                        "ERROR: hypothesis_ids must not be empty. "
                        "Every delegation must be linked to at "
                        "least one hypothesis. Call "
                        "HypothesisList() to see open hypotheses."
                    )
                # backlog still open → permit this setup-phase delegation
                # without a hypothesis link.
            known = {
                h["id"] for h in node._ledger.list_all()
            }
            unknown = [h for h in h_ids if h not in known]
            if unknown:
                return (
                    f"ERROR: unknown hypothesis IDs {unknown}."
                    f" Valid IDs: "
                    f"{sorted(known) or '(none proposed yet)'}."
                )

        # Resolve the optional process-phase tag (DoE/DataGeneration/ML/…).
        # Unknown/None → None (soft; never refuses), stored as the canonical
        # value string for the log + critic flags + downstream grouping.
        from ...phases import resolve_phase
        _phase_obj = resolve_phase(phase)
        _phase = _phase_obj.value if _phase_obj is not None else None

        # Milestone gate: the process backlog applies ONLY to delegations to the
        # f3dasm implementer (the agent that runs experiments) — never the
        # literature_reviewer/datagenerator that satisfy a milestone. Keyed on
        # the resolved TARGET ROLE (reliable), not the agent's self-declared
        # phase. This is a NUDGE, not a hard block: the milestones (assess-
        # literature, oracle-ready, …) are good prompts, not a safety invariant,
        # and a new design legitimately needs its own setup. Two-shot confirm,
        # RECURRING PER NAMESPACE — nudge once per namespace, proceed on a
        # re-delegate. MilestoneComplete/MilestoneSkip remain the clean path.
        _ms = getattr(node, "_milestones", None)
        _target_role = getattr(
            node._spec.nodes.get(target), "role", "") if node._spec else ""
        if _ms is not None and _target_role == "implementer":
            from ...milestones import implementer_block
            _pend = implementer_block(_ms, node)
            if _pend:
                _ns_key = namespace or "__default__"
                if not hasattr(node, "_milestone_ack"):
                    node._milestone_ack = set()
                if _ns_key not in node._milestone_ack:
                    node._milestone_ack.add(_ns_key)
                    _ids = ", ".join(f"{m['id']} ({m['description'][:50]}…)"
                                     for m in _pend)
                    node._record_intervention(
                        "MILESTONE_BLOCK", target,
                        f"{len(_pend)} backlog item(s) precede the implementer")
                    _scope = (f"design '{namespace}'" if namespace
                              else "this study")
                    return (
                        f"[CONFIRM] process backlog still open for {_scope}: "
                        f"{_ids}. The usual path is to resolve each first — "
                        "MilestoneComplete(id, brief), or MilestoneSkip(id, "
                        "reason) if it doesn't apply. If you mean to run the "
                        "implementer anyway, re-delegate (same target) to "
                        "confirm. (Not a tool error; a process nudge.)"
                    )
                # confirmed (and re-nudges for each new namespace) → fall through

        # Delegation ID — allocated AFTER the milestone gate so a blocked attempt
        # does not BURN an ID (next_id() advances a monotonic counter on every
        # call; allocating before the gate left a permanent gap in the sequence,
        # e.g. the milestone-blocked first implementer attempt always ate D002).
        # Globally unique when a shared DelegationLog is present (multiple
        # orchestrating nodes share one log); per-node counter otherwise.
        if node._delegation_log is not None:
            delegation_id = node._delegation_log.next_id()
            # Keep per-node seq in sync so checkpoint/WorkerNode paths that read
            # _delegation_seq stay consistent.
            with node._registry_lock:
                try:
                    node._delegation_seq = int(delegation_id[1:])
                except (ValueError, IndexError):
                    pass
        else:
            with node._registry_lock:
                node._delegation_seq += 1
                delegation_id = f"D{node._delegation_seq:03d}"

        start_time_mono = time.monotonic()
        started_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

        _followup_event = threading.Event()
        with node._registry_lock:
            node._registry[delegation_id] = {
                "status": "Working",
                "result": None,
                "evals": 0,
                "start_time": start_time_mono,
                "hypothesis_ids": h_ids,
                "is_falsification_attempt": bool(
                    is_falsification_attempt
                ),
                "phase": _phase,
                "started_at": started_at,
                "target": target,
                "namespace": (namespace or None),
                "followup_question": None,
                "followup_answer": None,
                "followup_event": _followup_event,
                "getstatus_count": 0,
                "followup_count": 0,
                # Read-time falsification ritual: flips True once the
                # checkpoint has been shown for this delegation's report
                # (fire-once anti-nag). The Done()-gate dangling check is
                # content-based and independent of this flag.
                "reconciled": False,
            }

        # Provenance: log a RUNNING entry NOW, at dispatch — before the worker
        # runs and flushes ledger rows. If this delegation is cancelled or
        # killed mid-flight (wall/eval budget), its ledgered evals stay traceable
        # to a logged delegation instead of becoming orphan rows. The terminal
        # DONE/FAILED record (same id) supersedes this via last-wins collapse.
        if node._delegation_log is not None:
            node._delegation_log.record_started(
                id=delegation_id,
                from_node=node._name,
                to_node=target,
                task=intent,
                hypothesis_ids=h_ids,
                started_at=started_at,
                is_falsification_attempt=bool(is_falsification_attempt),
                phase=_phase,
            )

        # Build task message
        edge = node._spec.edge(node._name, target)
        preamble = edge.preamble if edge else ""
        task_msg = (
            f"<workspace_subfolder>{delegation_id}/</workspace_subfolder>\n\n"
            + intent
        )
        if expected_report:
            task_msg += (
                f"\n\n**Required deliverables / acceptance"
                f" criteria:**\n{expected_report}"
            )
        if preamble:
            task_msg = preamble + "\n\n" + task_msg

        # Prepend time-budget banner so every worker starts
        # time-aware.  GetStatus handles mid-run updates.
        _b = node._budget_seconds
        _rs = node._run_start
        if _b is not None and _rs is not None:
            _el = time.time() - _rs
            _pct = (_el / _b) * 100
            _banner = (
                f"[Time budget: {_el:.0f}s / {_b:.0f}s used"
                f" ({_pct:.0f}%). Work efficiently and"
                f" return a report promptly.]\n\n"
            )
            task_msg = _banner + task_msg

        # Inject PROBLEM_STATEMENT for agents that request it
        _target_agent = node._spec.nodes.get(target) if node._spec else None
        if getattr(_target_agent, "inject_problem_statement", False) and node._study_dir:
            _ps_path = Path(node._study_dir) / "PROBLEM_STATEMENT.md"
            if _ps_path.exists():
                _ps_text = _ps_path.read_text(encoding="utf-8")
                task_msg = (
                    f"<problem_statement>\n{_ps_text}\n</problem_statement>\n\n"
                    + task_msg
                )

        # Each delegation gets its OWN adapter copy (D1/D2 concurrency fix).
        worker = worker_template.copy() if hasattr(worker_template, "copy") else worker_template

        # Inject per-delegation Write sandbox: writes go to {delegation_id}/ only.
        _study = node._study_dir
        if _study is not None:
            _workspace = (
                node._workspace_dir.resolve()
                if node._workspace_dir is not None
                else (Path(node._study_dir or ".") / "debug" / "delegations").resolve()
            )
            _delegation_ws = (_workspace / delegation_id).resolve()

            def _sandboxed_write(path: str, body: str, _ws=_delegation_ws, _did=delegation_id) -> str:
                """Write restricted to {delegation_id}/."""
                # Absorb a redundant leading "{delegation_id}/": the sandbox is
                # ALREADY rooted at {delegation_id}/, but the prompt calls it
                # "your D### subfolder", so agents naturally prefix paths with it
                # — which would nest D###/D###/. Strip one leading D### component
                # (only when a real filename remains after it). Do NOT lstrip
                # "/": an absolute path must stay absolute so the relative_to
                # boundary check below still rejects it.
                _norm = (path or "").strip()
                _first, _sep, _rest = _norm.partition("/")
                if _first == _did and _rest:
                    _norm = _rest
                try:
                    candidate = (_ws / _norm).resolve()
                except Exception as exc:  # noqa: BLE001
                    return f"ERROR: invalid path {path!r}: {exc}"
                try:
                    candidate.relative_to(_ws)
                except ValueError:
                    return (
                        f"ERROR: write rejected — {candidate} is outside "
                        f"{_ws}. Write only to {_did}/."
                    )
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_text(body, encoding="utf-8")
                return f"Written: {candidate}"

            worker.closure_tools["Write"] = node._wrap_closure(_sandboxed_write, target)

        def _run() -> None:
            evals_box: dict = {"count": 0}

            def ReportEvals(count: int) -> str:
                """Report the total ground-truth evaluations you performed this
                task. Call once per task, ALWAYS — even if 0. When you use
                get_evaluator() the canonical ledger is authoritative for the
                count, but this call also ARMS the unledgered-evals safety check,
                so never skip it."""
                evals_box["count"] = int(count)
                # Drain any queued budget warnings for this delegation.
                with node._pending_worker_msgs_lock:
                    msgs = node._pending_worker_msgs.pop(delegation_id, [])
                prefix = ("\n".join(msgs) + "\n\n") if msgs else ""
                return prefix + f"Recorded {count} evaluations."

            def FollowUp(question: str) -> str:
                """Ask your delegating party one clarifying question before proceeding.

                Routes to whoever sent you this task: the agent that delegated
                to you.  One FollowUp per delegation.  The answer is injected
                directly into your context.  If no answer arrives, proceed with
                best judgment.
                """
                with node._registry_lock:
                    entry = node._registry.get(delegation_id, {})
                    if entry.get("followup_count", 0) >= 1:
                        return (
                            "FollowUp limit reached (1 per delegation). "
                            "Proceed with best judgment."
                        )
                    node._registry[delegation_id]["followup_question"] = question
                    node._registry[delegation_id]["followup_count"] = 1
                    node._registry[delegation_id]["status"] = "FollowUp"
                    evt = node._registry[delegation_id]["followup_event"]
                with node._notifications_lock:
                    node._notifications.append(
                        f"[{delegation_id} FollowUp: {question!r} "
                        f"→ call Reply('{delegation_id}', answer)]"
                    )
                evt.wait(timeout=300)  # 5-minute patience; proceed if no reply
                with node._registry_lock:
                    answer = node._registry[delegation_id].get("followup_answer")
                    node._registry[delegation_id]["status"] = "Working"
                # Drain any queued budget warnings alongside the answer.
                with node._pending_worker_msgs_lock:
                    msgs = node._pending_worker_msgs.pop(delegation_id, [])
                budget_prefix = ("\n".join(msgs) + "\n\n") if msgs else ""
                base = answer or "No answer received. Proceed with best judgment."
                return budget_prefix + base

            def ReportProgress(note: str) -> str:
                """Leave a short progress note (<=200 chars) your delegator sees
                when it polls you. NON-BLOCKING — you keep working immediately;
                no answer comes back. Use it so the delegator can tell you are
                making progress rather than stuck (which prevents needless
                cancellation): e.g. 'LHS done, 250 evals; fitting GP next' or
                'BO round 3/10, best f=-0.81 so far'.
                """
                _n = (note or "").strip()[:200]
                with node._registry_lock:
                    e = node._registry.get(delegation_id)
                    if e is not None:
                        e["progress_note"] = (_n, time.monotonic())
                return "Progress noted (your delegator will see it on poll)."

            worker.closure_tools["ReportEvals"] = ReportEvals  # not wrapped: never errors
            worker.closure_tools["ReportProgress"] = ReportProgress  # not wrapped
            worker.closure_tools["FollowUp"] = node._wrap_closure(FollowUp, target)
            # Confer: async messaging to the orchestrator (or any woken peer).
            # The worker drains its OWN inbox collect-on-send (see _build_confer).
            worker.closure_tools["Confer"] = node._wrap_closure(
                _build_confer(target), target)
            # ConsultHandbook is injected universally at adapter construction
            # (agent_runtime._make_adapter) — every node gets it equally there.
            try:
                from ...agent_prompts import build_report_retry_prompt
                from ...backends.base import (
                    debug_enabled as _dbg,
                )
                from ...backends.base import (
                    set_delegation_id as _set_did,
                )
                from ...backends.base import (
                    set_namespace as _set_ns,
                )
                from ...backends.base import (
                    set_oracle_registered as _set_oracle_reg,
                )
                from ...backends.base import (
                    set_run_config_path as _set_rc,
                )
                from ...backends.base import (
                    set_transcript_sink as _set_sink,
                )

                # Bind the delegation id for this worker thread so the
                # backend can inject F3DASM_DELEGATION_ID into the session
                # env → get_evaluator() resolves without a mandatory cd
                # into D### (audit Finding 2).
                _set_did(delegation_id)

                # Scope this worker to its design namespace (Axis 3a/3b) so the
                # backend injects F3DASM_NAMESPACE → get_evaluator() resolves the
                # namespace's oracle + ledger. None → single-study canonical path.
                _set_ns(namespace or None)

                # Bind the run_config.json path too, so the backend injects
                # F3DASM_RUN_CONFIG → get_evaluator() resolves by explicit path
                # rather than walking up from the worker's cwd (study_dir),
                # which can never reach runs/<id>/debug/run_config.json.
                _notes_for_rc = node._current_notes_dir
                if _notes_for_rc is not None:
                    _set_rc(str(_notes_for_rc.parent / "run_config.json"))

                # The eval-ledger guards apply ONLY to evaluator roles
                # (implementer/debugger) AND only once a canonical oracle is
                # registered. Non-evaluator roles (literature_reviewer,
                # datagenerator, critic) never reach get_evaluator(), so
                # nudging/bouncing them is a false positive. This one flag
                # gates all three guards below.
                _guard_agent = (
                    node._spec.nodes.get(target) if node._spec else None
                )
                _target_role = getattr(_guard_agent, "role", None)
                _enforce_ledger = (
                    node._canonical_source_registered()
                    and _target_role in _LEDGER_GUARD_ROLES
                )
                _set_oracle_reg(_enforce_ledger)

                # DEBUG: stream this worker's full reasoning + tool-calls
                # to debug/transcripts/{delegation_id}.jsonl (thread-local;
                # this _run is the worker's own thread).
                if _dbg() and node._current_notes_dir is not None:
                    _set_sink(str(
                        node._current_notes_dir.parent / "transcripts"
                        / f"{delegation_id}.jsonl"))

                messages = [{"role": "user", "content": task_msg}]
                text = worker.invoke(messages)

                # Validate against THIS agent's declared report_sections
                # (audit Finding 4 — report_sections is now the single
                # source of truth, not a hardcoded list), so e.g. a missing
                # ### Retrospective earns one corrective retry.
                _req_sections = list(
                    getattr(_target_agent, "report_sections", None) or []
                ) or None
                diagnosis = _classify_response(text, _req_sections)
                if diagnosis is not None:
                    retry_messages = messages + [
                        {"role": "ai", "content": text},
                        {
                            "role": "user",
                            "content": (
                                f"{build_report_retry_prompt(_req_sections)}"
                                f"\n\nDiagnosis: {diagnosis}"
                            ),
                        },
                    ]
                    text = worker.invoke(retry_messages)

                # Unledgered evals are a CORRECTIVE flag, not a re-run. The
                # reconciliation below counts this delegation's stamped rows
                # across every experiment store (provenance-based); if a source
                # is registered and it claimed evals but stamped none anywhere,
                # it is flagged OFF_LEDGER_EVALS (a corrective tip the strategizer
                # reads) and counted as 0. We do NOT bounce it to re-run: re-
                # running a whole campaign to re-ledger is wasted wall-time (it
                # helped blow the watchdog in run 20260627T045747), and the
                # critic's headline-provenance check at the gate is the real
                # floor. Cooperative agents rarely bypass get_evaluator() on
                # purpose; when they do, the tip says so — once.

                # Direct evidence: log any raw-oracle nudge firings from
                # this delegation (drained from the adapter's budget).
                _onb = getattr(worker, "_oracle_nudge", None)
                _evs = list(getattr(_onb, "events", []) or [])
                for _ev in _evs:
                    node._record_intervention(
                        "RAW_ORACLE_NUDGE", target,
                        f"{delegation_id}: a {_ev.get('tool')} call "
                        "reached the oracle directly; nudged toward "
                        "get_evaluator().",
                        snippet=_ev.get("snip", ""),
                    )
                if _onb is not None:
                    _onb.events = []

                # Accumulate token usage from this worker invocation
                # (_usage is also consumed downstream by the delegation log).
                _usage = getattr(worker, "last_usage", {}) or {}
                node._record_worker_usage(worker, target, delegation_id)

                # Scan worker report for MCP tool errors (infrastructure, not agent fault).
                # MCP errors appear as lines containing "error" near tool names in the report.
                import re as _re
                _MCP_ERROR_PATTERNS = [
                    r"(mcp__\w+__\w+)[^\n]*?(429|rate.?limit|timeout|timed.?out|unavailable|connection.?error)",
                    r"(HTTP\s+(?:429|500|502|503))[^\n]*",
                    r"(rate.?limit(?:ed|ing)?)[^\n]*",
                ]
                for _pat in _MCP_ERROR_PATTERNS:
                    for _m in _re.finditer(_pat, text, _re.IGNORECASE):
                        node._record_tool_error(
                            target,
                            _m.group(1) if _m.lastindex and _m.lastindex >= 1 else "mcp_tool",
                            "MCP_REPORTED",
                            _m.group(0)[:200],
                        )
                        # Override fault to "system" for MCP infrastructure errors
                        # (already classified correctly by _record_tool_error)
                        break  # one log entry per pattern match per delegation

                # Ledger row count preferred over self-reported
                # Count rows per delegation_id in the store the delegation
                # ACTUALLY wrote to (its design namespace, or canonical when
                # None); fall back to the honour-system evals_box only when it
                # wrote no rows there. Reading the canonical store for a
                # namespaced delegation undercounts it to 0 and falsely flags it
                # off-ledger (run 20260627T011059 D006/annular: 100 real evals
                # logged as 0). namespace is the Delegate() arg, in scope here.
                _notes = node._current_notes_dir
                _run_exp = (
                    _notes.parent.parent / "experiment_data"
                    if _notes is not None else None
                )
                # Honesty reconciliation (runs on BOTH the normal-return and the
                # cancel/detach path): the ledger — summed across EVERY experiment
                # store — is the source of truth. If a source is registered and the
                # worker CLAIMED evals but NONE are provenance-stamped in ANY store,
                # trust the ledger (count 0) and flag it; otherwise the count is
                # found wherever the delegation wrote (provenance-based, so a
                # call-site experiment selection is reconciled correctly).
                _claimed_evals = evals_box["count"]
                _evals, _off_ledger, _stamped_evals = (
                    _reconcile_delegation_evals(
                        _run_exp,
                        delegation_id,
                        _claimed_evals,
                        _enforce_ledger,
                    )
                )
                # Auto-append measured per-eval KPIs for THIS delegation's rows
                # (median/max wall-time, ledger total) so the strategizer plans
                # its budget on observed sim cost. Read the summary of the store
                # that actually holds this delegation's rows (its experiment),
                # found by provenance — not assumed to be the default store.
                # Best-effort: a KPI footer must never fail a delegation.
                try:
                    from ...instrumented import (
                        RunStateSummary,
                        experiment_stores,
                    )
                    _summary = None
                    for _st in (experiment_stores(_run_exp)
                                if _run_exp is not None else []):
                        _s = RunStateSummary.from_store(_st)
                        if _s is not None and _s.n_per_delegation.get(
                                delegation_id, 0) > 0:
                            _summary = _s
                            break
                    if _summary is not None:
                            # Wall budget remaining (telemetry, not a hard stop):
                            # makes the median above actionable. None when no
                            # wall budget is set or the run hasn't started timing.
                            _bs = getattr(node, "_budget_seconds", None)
                            _rs = getattr(node, "_run_start", None)
                            _rem = (
                                _bs - (time.time() - _rs)
                                if _bs and _rs else None
                            )
                            # Peak RAM this delegation reached (watcher high-water,
                            # free) against the hard cap, so the strategizer learns
                            # the memory footprint like it learns the time cost.
                            from ...watchdog_cleanup import delegation_peak_rss
                            _peak = delegation_peak_rss(delegation_id)
                            _cap = None
                            try:
                                import json as _json
                                _cfgp = (_notes.parent.parent / "debug"
                                         / "run_config.json")
                                if _cfgp.exists():
                                    _cap = _json.loads(
                                        _cfgp.read_text()).get("mem_cap_bytes")
                            except Exception:  # noqa: BLE001
                                _cap = None
                            _footer = _summary.delegation_footer(
                                delegation_id,
                                wall_remaining_s=_rem,
                                wall_budget_s=_bs,
                                peak_rss_bytes=_peak,
                                ram_cap_bytes=_cap,
                            )
                            if _footer:
                                text = text + _footer
                except Exception:  # noqa: BLE001
                    pass
                with node._registry_lock:
                    _detached = (
                        node._registry[delegation_id].get("status")
                        == "Cancelled"
                    )
                    if _detached:
                        # CancelDelegation detached this while it ran: keep it
                        # Cancelled and DISCARD the deliverable. Usage was
                        # already recorded above (the worker did spend tokens).
                        node._registry[delegation_id]["evals"] = _evals
                    else:
                        node._registry[delegation_id].update({
                            "status": "Done",
                            "result": text,
                            "evals": _evals,
                            "usage": _usage,
                        })
                        # This target made progress → clear its consecutive
                        # error streak (the repeated-errors halt is for a
                        # target stuck failing, not one that recovers).
                        node._consecutive_errors[target] = 0
                with node._notifications_lock:
                    node._notifications.append(
                        f"[Delegation {delegation_id} "
                        + ("completed after cancellation — result discarded]"
                           if _detached else "Done]")
                    )
                # Loud, single record of the off-ledger condition — on the
                # cancel/detach path too (which the bounce above never reaches),
                # so a cancelled delegation that evaluated off-ledger can no
                # longer vanish silently.
                if _off_ledger:
                    node._record_intervention(
                        "OFF_LEDGER_EVALS", target,
                        f"{delegation_id} claimed {_claimed_evals} evals but none "
                        "are provenance-stamped in any experiment store — counted "
                        "as 0"
                        + (" (delegation was cancelled/detached)"
                           if _detached else "")
                        + ". These didn't go through get_evaluator(), so they "
                        "cannot anchor a reproducible headline — that wasn't the "
                        "right way to evaluate. If this delegation's numbers feed "
                        "your conclusion, re-run it through get_evaluator(); and "
                        "route evaluations through get_evaluator() from the start "
                        "next time. (Not re-run for you: re-running a whole "
                        "campaign to re-ledger wastes wall-time.)",
                        claimed=_claimed_evals,
                        stamped=_stamped_evals,
                        detached=_detached,
                    )
                # Write delegation record to graph-wide delegation log
                if node._delegation_log is not None:
                    node._delegation_log.record(
                        id=delegation_id,
                        from_node=node._name,
                        to_node=target,
                        task=intent,
                        deliverable=text,
                        hypothesis_ids=h_ids,
                        started_at=started_at,
                        completed_at=datetime.now(
                            tz=timezone.utc
                        ).isoformat(timespec="seconds"),
                        status="DONE",
                        tokens_in=(
                            _usage.get("input_tokens", 0) or 0
                        ),
                        tokens_out=(
                            _usage.get("output_tokens", 0) or 0
                        ),
                        cost_usd=_usage.get("total_cost_usd"),
                        is_falsification_attempt=bool(
                            is_falsification_attempt
                        ),
                        evals=_evals,
                        phase=_phase,
                    )
                    if node._science_monitor is not None:
                        try:
                            node._science_monitor\
                                .on_delegation_complete(
                                delegation_id
                            )
                        except Exception:  # noqa: BLE001
                            pass

                # Registration handoff: when a datagenerator delegation
                # authors an oracle, it drops a registration.json manifest
                # in its workspace. Point the canonical entrypoint at it so
                # the next get_evaluator() (re-reads config each call)
                # resolves it — no manual config edit. Best-effort: never
                # fail a delegation over registration.
                try:
                    _tgt = node._spec.nodes.get(target)
                    if (
                        _notes is not None
                        and getattr(_tgt, "role", None) == "datagenerator"
                    ):
                        _run_dir = _notes.parent.parent
                        _ws = (
                            _run_dir / "debug" / "delegations"
                            / delegation_id / "generators"
                        )
                        _manifest = _ws / "registration.json"
                        if _manifest.exists():
                            import json as _json

                            from ...agent_runtime import (
                                register_evaluator_entrypoint,
                            )
                            _m = _json.loads(_manifest.read_text())
                            _gf = _m["generator_file"]
                            _gf_path = Path(_gf)
                            if not _gf_path.is_absolute():
                                # Resolve against the manifest dir, then
                                # the run dir; take the first that exists.
                                for _base in (_ws, _run_dir):
                                    _cand = (_base / _gf).resolve()
                                    if _cand.exists():
                                        _gf_path = _cand
                                        break
                            # A datagenerator delegation scoped to a design
                            # namespace registers that namespace's oracle (its
                            # own isolated store); the manifest may also name one.
                            # The delegation's namespace is authoritative.
                            _ns = namespace or _m.get("namespace") or None
                            _ep = register_evaluator_entrypoint(
                                _run_dir / "debug" / "run_config.json",
                                _gf_path,
                                _m["attr"],
                                output_names=_m.get("output_names"),
                                namespace=_ns,
                            )
                            with node._notifications_lock:
                                _ns_tag = f" ns={_ns}" if _ns else ""
                                node._notifications.append(
                                    f"[Evaluator registered: {_ep}{_ns_tag}]"
                                )
                except Exception:  # noqa: BLE001
                    pass

                # Worker retrospective (every node has a 'job done'
                # moment — see _record_retrospective).
                node._record_retrospective(
                    node._role_of(target), delegation_id, text
                )
            except Exception:  # noqa: BLE001
                tb = traceback.format_exc()
                _usage = getattr(worker, "last_usage", {}) or {}
                node._record_worker_usage(worker, target, delegation_id)
                with node._registry_lock:
                    node._registry[delegation_id].update({
                        "status": "Errored",
                        "result": tb,
                        "evals": evals_box["count"],
                        "usage": _usage,
                    })
                    node._consecutive_errors[target] = (
                        node._consecutive_errors.get(target, 0) + 1
                    )
                with node._notifications_lock:
                    node._notifications.append(
                        f"[Delegation {delegation_id} Errored]"
                    )
                if node._delegation_log is not None:
                    node._delegation_log.record(
                        id=delegation_id,
                        from_node=node._name,
                        to_node=target,
                        task=intent,
                        # Keep the TAIL: the root exception is on
                        # the last line of a traceback.
                        deliverable="ERROR: " + tb[-2000:],
                        hypothesis_ids=h_ids,
                        started_at=started_at,
                        completed_at=datetime.now(
                            tz=timezone.utc
                        ).isoformat(timespec="seconds"),
                        status="FAILED",
                        tokens_in=(
                            _usage.get("input_tokens", 0) or 0
                        ),
                        tokens_out=(
                            _usage.get("output_tokens", 0) or 0
                        ),
                        cost_usd=_usage.get("total_cost_usd"),
                        is_falsification_attempt=bool(
                            is_falsification_attempt
                        ),
                        phase=_phase,
                    )

        t = threading.Thread(target=_run, daemon=True, name=delegation_id)
        with node._registry_lock:
            node._threads[delegation_id] = t  # A3: inside lock
        t.start()

        # Reset two-shot Done() gate so the next Done() warns again.
        node._done_warned = False

        if wait:
            # Synchronous mode: block until the delegation finishes.
            t.join()
            with node._registry_lock:
                entry = dict(node._registry.get(delegation_id, {}))
            status = entry.get("status", "Errored")
            if status == "Done":
                cp = _falsification_checkpoint(delegation_id)
                body = f"Done\n\n{entry['result']}"
                return body + (("\n\n" + cp) if cp else "")
            return f"Errored:\n{entry.get('result', '(no details)')}"

        return (
            f"Delegation started. ID: {delegation_id!r}. "
            f"Use GetStatus('{delegation_id}') to poll for completion."
        )

    Delegate.__doc__ = _delegate_doc

    def GetStatus(delegation_id: str) -> str:
        """Poll a background delegation; also delivers push notifications.

        Returns one of:
          'Working (running for Xs, polled N times)' — still running
          'Done\\n\\n<full report>'                  — completed
          'Errored:\\n<traceback>'                   — failed

        Polling is for async (wait=False) delegations. If this task had no
        reason to overlap other work, Delegate(wait=True) would have returned
        the result directly with no polling — worth a thought next time.
        """
        prefix = node._drain_notifications()

        # Drain any budget warnings queued for this delegation.
        with node._pending_worker_msgs_lock:
            worker_msgs = node._pending_worker_msgs.pop(delegation_id, [])
        if worker_msgs:
            prefix += "\n".join(worker_msgs) + "\n\n"

        with node._registry_lock:
            entry = node._registry.get(delegation_id)
            if entry is None:
                # Cache miss: consult the authoritative log. The in-memory
                # registry can be rebuilt empty after a node reconstruction
                # while the log retains every delegation — this is the
                # "Known IDs: []" symptom (audit BF-0).
                _lstatus, _ldeliv = node._log_status(delegation_id)
                if _lstatus == "DONE":
                    return prefix + f"Done\n\n{_ldeliv}"
                if _lstatus == "FAILED":
                    return prefix + f"Errored:\n{_ldeliv}"
                if _lstatus == "RUNNING":
                    return prefix + (
                        "Working (still running; live progress is unavailable "
                        "after a session rebuild — re-poll shortly and the "
                        "result will appear here when it completes)"
                    )
                known = list(node._registry)
                return (
                    prefix +
                    f"ERROR: unknown delegation ID {delegation_id!r}. "
                    f"Known IDs: {known}"
                )
            status = entry["status"]
            if status in ("Working", "FollowUp"):
                # Increment poll count and record timing.
                entry["getstatus_count"] = entry.get("getstatus_count", 0) + 1
                poll_count = entry["getstatus_count"]
                last_poll = entry.get("last_getstatus_time")
                now_mono = time.monotonic()
                entry["last_getstatus_time"] = now_mono
                start_mono = entry["start_time"]
                elapsed = int(now_mono - start_mono)
                prev_stamped = entry.get("last_stamped", 0)
                last_progress = entry.get("last_progress_time", start_mono)
                progress_note = entry.get("progress_note")

        # Status token FIRST (contract: callers dispatch on the
        # leading word); queued notifications follow the report.
        _tail = (
            ("\n\n" + prefix.rstrip()) if prefix.strip() else ""
        )
        if status == "Done":
            cp = _falsification_checkpoint(delegation_id)
            body = f"Done\n\n{entry['result']}"
            if cp:
                body += "\n\n" + cp
            return body + _tail
        if status not in ("Working", "FollowUp"):
            return f"Errored:\n{entry['result']}" + _tail

        # --- Still working: build informative response ---
        # Ledger progress (feature c): surface real progress so the delegator
        # can tell "progressing" from "stuck" instead of inferring it from
        # wall-time (the blindness that drove over-cancelling). Also folds in
        # backlog #6: zero stamped after a long wall-time IS the stuck signal.
        # Count the delegation's rows across EVERY experiment store (provenance-
        # based) — else a campaign that wrote to its experiment's store polls as
        # 0 progress and the stuck-signal above would over-cancel a healthy worker.
        _run_exp = (
            node._current_notes_dir.parent.parent / "experiment_data"
            if node._current_notes_dir is not None else None
        )
        cur_stamped = (
            _stamped_eval_count(_run_exp, delegation_id) if _run_exp else 0
        )
        delta = cur_stamped - prev_stamped
        if cur_stamped > prev_stamped:
            last_progress = now_mono
        with node._registry_lock:
            _e = node._registry.get(delegation_id)
            if _e is not None:
                _e["last_stamped"] = cur_stamped
                _e["last_progress_time"] = last_progress
        stale = int(now_mono - last_progress)
        if cur_stamped > 0 and delta > 0:
            progress_desc = (
                f"{cur_stamped} evals stamped (+{delta} since last poll) "
                "— progressing")
        elif cur_stamped > 0:
            progress_desc = (
                f"{cur_stamped} evals stamped, none new for {stale}s")
        else:
            progress_desc = f"0 evals stamped after {elapsed}s"
        # Per-delegation memory telemetry (resource-governance L3): surface this
        # delegation's process-tree RSS so the strategizer can SEE a fat campaign
        # and Confer the implementer. Best-effort; appended only if known.
        if _run_exp is not None:
            try:
                from ...watchdog_cleanup import (
                    delegation_peak_rss,
                    delegation_rss,
                )
                _rss = delegation_rss(_run_exp.parent, delegation_id)
                if _rss > 0:
                    progress_desc += f"; ~{_rss / 1024 ** 2:.0f} MB RSS"
                    _peak = delegation_peak_rss(delegation_id)
                    if _peak > _rss:
                        progress_desc += f" (peak ~{_peak / 1024 ** 2:.0f} MB)"
            except Exception:  # noqa: BLE001
                pass
        note_desc = ""
        if progress_note:
            _ntext, _nts = progress_note
            note_desc = (
                f" · worker note: {_ntext!r} ({int(now_mono - _nts)}s ago)")

        hints: list[str] = []

        # Rate warning: polled too recently.
        if last_poll is not None and (now_mono - last_poll) < 30:
            hints.append(
                f"NOTE: you polled {delegation_id} only "
                f"{now_mono - last_poll:.0f}s ago. "
                "The worker runs in a background thread — polling faster "
                "does not make it finish sooner. Do other work in the "
                "meantime."
            )

        # Poll-count escalation. Polling does NOT make the worker finish
        # sooner, so from the first escalation we spell out the three real
        # ways forward (same options as the premature-Done nudge) — so the
        # agent never grinds out 30 status checks when it could just wait.
        if poll_count >= 5:
            firmness = (
                "STOP polling in a tight loop. " if poll_count >= 15
                else ""
            )
            if cur_stamped > 0:
                # Demonstrably progressing — anchor the nudge on the numbers so
                # the agent doesn't cancel a healthy campaign out of impatience.
                hints.append(
                    f"{firmness}Polled {poll_count}× ({elapsed}s) — but "
                    f"{delegation_id} IS progressing ({progress_desc}). Polling "
                    "won't speed it up. Best move: (a) do other work now; or "
                    "(b) just wait and poll occasionally. Do NOT cancel a "
                    "progressing delegation to save time — its ledgered evals "
                    "persist regardless, so cancelling only discards its report. "
                    "(If this task had nothing to overlap, wait=True would have "
                    "blocked without any of this polling.)"
                )
            else:
                # Zero stamped (backlog #6 stuck signal): cancelling is now a
                # defensible call, but only here.
                hints.append(
                    f"{firmness}Polled {poll_count}× ({elapsed}s) and "
                    f"{progress_desc}. Options: (a) do other work; (b) just "
                    "wait — a worker may still be setting up before its first "
                    "eval. A delegation that has stamped NOTHING for a long "
                    "time may be genuinely stuck; the run watchdog will reclaim "
                    "it. (A task with nothing to overlap could have been "
                    "wait=True — blocking with zero polling.)"
                )

        # Budget broadcast: check if a new 10%-overbudget threshold is reached.
        budget = node._budget_seconds
        run_start = node._run_start
        if budget is not None and run_start is not None:
            elapsed_wall = time.time() - run_start
            pct = (elapsed_wall / budget) * 100
            # Thresholds: 80, 90, 100, 110, 120, …
            threshold = int(pct // 10) * 10
            if threshold >= 80:
                with node._pending_worker_msgs_lock:
                    already_sent = node._budget_notified_pcts
                    if threshold not in already_sent:
                        already_sent.add(threshold)
                        _backstop_mult = run_backstop_multiple()
                        _over = (
                            backstop_enabled()
                            and pct >= _backstop_mult * 100
                        )
                        msg = (
                            f"BACKSTOP IMMINENT: {pct:.0f}% of time "
                            "budget — past the "
                            f"{int(_backstop_mult)}x cost "
                            "backstop. Stop polling and call Done() "
                            "NOW with a partial report."
                        ) if _over else (
                            f"BUDGET: {pct:.0f}% of time budget consumed. "
                            "Wrap up your current work and return a partial "
                            "report as soon as possible."
                        )
                        # Queue for all currently Working delegations.
                        with node._registry_lock:
                            active = [
                                did for did, e in node._registry.items()
                                if e["status"] in ("Working", "FollowUp")
                                and did != delegation_id
                            ]
                        for did in active:
                            node._pending_worker_msgs.setdefault(did, []).append(msg)
                        # Include in this response too.
                        hints.append(msg)

        # Status token FIRST (documented contract: callers may
        # dispatch on the leading word); hints and queued
        # notifications follow.
        hint_str = ("\n\n" + "\n".join(hints)) if hints else ""
        tail = ("\n\n" + prefix.rstrip()) if prefix.strip() else ""
        return (
            f"Working (running for {elapsed}s, polled {poll_count}× · "
            + progress_desc + note_desc + ")" + hint_str + tail
        )

    def CancelDelegation(delegation_id: str) -> str:
        """Detach a delegation whose RESULT you no longer want.

        Cancel ONLY when the output is genuinely unwanted — a wrong approach, a
        superseded plan, a true dead-end. Do NOT cancel a delegation because it
        is slow: a Working delegation is almost always still producing real,
        ledgered evaluations (the worker runs in a background thread — slow is
        not stuck). Cancelling discards its REPORT, so its findings never reach
        your conclusion; its already-written ledger rows remain (and still
        count). If you just want to make progress meanwhile, do other work in
        parallel and let it finish. A delegation that has already produced
        ledgered evals is two-shot: call twice to confirm."""
        prefix = node._drain_notifications()
        with node._registry_lock:
            entry = node._registry.get(delegation_id)
            if entry is None:
                return (
                    prefix + f"No delegation {delegation_id!r}. "
                    f"Known: {list(node._registry)}"
                )
            st = entry.get("status")
            if st not in ("Working", "FollowUp"):
                return (
                    prefix + f"Delegation {delegation_id} is {st!r}, not "
                    "running — nothing to cancel."
                )
            # Harden against impatience: a delegation already writing ledgered
            # evals is progressing, not stuck. Require a deliberate second call
            # so a slow-but-healthy campaign can't be discarded on a whim. Count
            # across EVERY experiment store (provenance-based) — else a campaign
            # that wrote to its experiment's store reads 0 and loses this guard.
            _run_exp = (
                node._current_notes_dir.parent.parent / "experiment_data"
                if node._current_notes_dir is not None else None
            )
            _stamped = (
                _stamped_eval_count(_run_exp, delegation_id) if _run_exp else 0
            )
            if _stamped > 0 and not entry.get("cancel_pending"):
                entry["cancel_pending"] = True
                return (
                    prefix + f"HOLD: {delegation_id} has already written "
                    f"{_stamped} provenance-stamped evaluation(s) to the "
                    "canonical ledger — it is progressing, not stuck. "
                    "Cancelling discards its REPORT (its findings won't reach "
                    "your conclusion); the evals remain. If it is merely slow, "
                    "do other work in parallel and let it finish. If its result "
                    "is genuinely unwanted, call CancelDelegation('"
                    + delegation_id + "') again to confirm."
                )
            entry["status"] = "Cancelled"
        with node._notifications_lock:
            node._notifications.append(
                f"[Delegation {delegation_id} cancelled — detached; its "
                "report is discarded (its ledgered evals remain)]"
            )
        return (
            prefix + f"Delegation {delegation_id} cancelled (detached): "
            "excluded from the run, its result will be ignored. You may "
            "proceed (e.g. call Done() if nothing else is running) or start "
            "other work."
        )

    def Wait(delegation_id: str) -> str:
        """Block until delegation_id finishes (Done or Errored), then return
        its result. Use instead of polling with GetStatus() — holds the current
        turn open with no extra turns consumed.

        Returns the same text as GetStatus() once the delegation completes."""
        prefix = node._drain_notifications()
        with node._registry_lock:
            entry = node._registry.get(delegation_id)
            if entry is None:
                return prefix + (
                    f"ERROR: unknown delegation {delegation_id!r}. "
                    f"Known: {list(node._registry)}"
                )
            if entry["status"] in ("Done", "Errored"):
                cp = entry.get("checkpoint", "")
                body = f"{entry['status']}\n\n{entry.get('result', '')}"
                return prefix + body + (("\n\n" + cp) if cp else "")
            t = node._threads.get(delegation_id)

        if t is not None:
            while t.is_alive():
                t.join(timeout=10.0)
                with node._notifications_lock:
                    _notifs = list(node._notifications)
                    node._notifications.clear()
                for _n in _notifs:
                    prefix += _n + "\n\n"
                # Wake a strategizer that's asleep in Wait() for a live
                # campaign — without this, a science-monitor nudge (e.g.
                # DUPLICATE_EVALUATION) only surfaces on the NEXT tool call,
                # by which point the whole delegation (and its eval budget)
                # has already finished. Same drain() every other call site
                # uses; each poll tick is a fresh check, not a repeat.
                if node._science_monitor is not None:
                    drift = node._science_monitor.drain()
                    if drift:
                        prefix += drift + "\n"

        with node._registry_lock:
            entry = node._registry.get(delegation_id, {})
        cp = entry.get("checkpoint", "")
        body = f"{entry.get('status', 'Unknown')}\n\n{entry.get('result', '')}"
        return prefix + body + (("\n\n" + cp) if cp else "")

    def Reply(delegation_id: str, answer: str) -> str:
        """Answer a worker's FollowUp question and unblock it.

        Call this after GetStatus returns 'FollowUp: <question>'.
        The answer is injected into the worker's context and it resumes.
        """
        prefix = node._drain_notifications()
        with node._registry_lock:
            entry = node._registry.get(delegation_id)
            if entry is None:
                return prefix + f"ERROR: unknown delegation {delegation_id!r}."
            if entry.get("status") != "FollowUp":
                return (
                    prefix +
                    f"ERROR: delegation {delegation_id!r} is not awaiting a "
                    f"FollowUp (status: {entry.get('status')!r})."
                )
            entry["followup_answer"] = answer
            evt = entry["followup_event"]
        evt.set()
        return prefix + f"Reply sent to {delegation_id}. Worker resuming."

    def Done(summary: str) -> str:
        """Signal end of run with a summary of findings (two-shot).

        Call only when: a best design is in hand with numerical support from
        Reports; at least one falsification attempt has been carried out; every
        PRIMARY success criterion is MET (an INCONCLUSIVE/unmet one is not — run
        the affordable experiment that would settle it if budget remains); and
        pipeline.ipynb has been authored via WriteDeliverable("pipeline.ipynb", …).
        summary should state the best design + supporting numbers + the
        falsification outcome + remaining uncertainty + (if closing with budget
        left) why the remaining budget cannot settle any unmet criterion.

        First call: issues a WARNING and lists any open delegations or
        unmet conditions; does NOT close.
        Second call: closes the run.

        Refused if any delegation is still Working — call GetStatus()
        on all pending delegations first. (A delegation you launched
        wait=True would already be collected here, with no pending poll.)
        """
        prefix = node._drain_notifications()
        # Liveness reconciled against the authoritative persistent log: a
        # delegation the log shows terminal is never "pending", so a stale
        # in-memory cache can no longer make Done() refuse forever (audit
        # BF-0/BF-2: the run4 deadlock, where a finished delegation read
        # "Working" until the watchdog killed the run UNGATED).
        pending = node._pending_delegations()
        if pending:
            # Soft nudge (NOT an "ERROR:" return, so it isn't counted as a
            # tool error): closing now is premature, but offer the three real
            # ways forward instead of a dead-end bounce.
            return (
                prefix +
                f"{len(pending)} delegation(s) still running: {pending}. "
                "Closing now is premature — you have two options:\n"
                "  (a) keep working: inspect results so far, write notes, or "
                "start another delegation while these finish;\n"
                "  (b) wait, then GetStatus(<id>) on each and interpret its "
                "results before you conclude.\n"
                "Re-call Done() once none are still running."
            )
        # Exit-interview capture (final stage): the conclusion is already
        # accepted + recorded; this Done() carries ONLY the retrospective.
        # Capture it, then actually close. The strategizer hears about the
        # interview ONLY after the critic accepted — never during its
        # working turns, so its orchestration context stays clean.
        if node._awaiting_retro:
            node._awaiting_retro = False
            node._record_retrospective("strategizer", "DONE", summary)
            node._done_warned = False
            route["kind"] = "done"
            route["summary"] = node._final_summary
            return prefix + "Run complete."
        # Milestone close-gate (Spec #1, HARD): every milestone must be DONE or
        # SKIPPED before the run can close. Auto-satisfy first (met gates tick
        # themselves). Forces engagement; MilestoneSkip(id, reason) is the
        # escape so it never deadlocks. Checked on every Done() call (the agent
        # can't bypass via the two-shot).
        _ms = getattr(node, "_milestones", None)
        if _ms is not None:
            _ms.auto_satisfy(node)
            _pend = _ms.pending()
            if _pend:
                node._milestone_block_count = getattr(
                    node, "_milestone_block_count", 0) + 1
                if node._milestone_block_count <= 3:
                    items = "; ".join(
                        f"{m['id']} ({m['description']})" for m in _pend)
                    return (
                        prefix + "Cannot close yet — these milestones are "
                        f"still PENDING: {items}. For each: "
                        "MilestoneComplete(id, brief) with a one-line why, or "
                        "MilestoneSkip(id, reason) if this study doesn't need "
                        "it. Then re-call Done(). (Process gate, not a tool "
                        "error.)"
                    )
                # Bounded escape (mirrors the 3-strikes UNGATED gate): after 3
                # blocked closes, auto-skip the rest so the run can never
                # deadlock. Recorded — the critic sees the forced skips and can
                # flag them.
                for m in _pend:
                    _ms.skip(m["id"],
                             "auto-skipped: unresolved after 3 close attempts")
                node._record_intervention(
                    "MILESTONE_AUTO_SKIP", "",
                    f"{len(_pend)} milestone(s) auto-skipped after 3 close "
                    "attempts")
        # Two-shot gate: first call warns, second call closes.
        if not node._done_warned:
            node._done_warned = True
            open_hypotheses: list[str] = []
            if node._ledger is not None:
                open_hypotheses = [
                    h["id"]
                    for h in node._ledger.list_all()
                    if h.get("current_status") == "OPEN"
                ]
            warn_parts = [
                "WARNING: first Done() call — confirm you are ready to close.",
                "Call Done() again to confirm and end the run.",
            ]
            if open_hypotheses:
                warn_parts.append(
                    f"Open hypotheses still in OPEN state: "
                    f"{open_hypotheses}. "
                    "Consider updating their status before closing."
                )
            # Dangling falsification attempts: a delegation flagged (or
            # post-hoc linked) as a falsification attempt whose hypothesis is
            # still OPEN means a test ran but its verdict was never recorded.
            # Content-based (independent of the read-time fire-once flag).
            if open_hypotheses:
                _open = set(open_hypotheses)
                dangling: list[str] = []
                with node._registry_lock:
                    for d_id, e in node._registry.items():
                        if (
                            e.get("status") == "Done"
                            and e.get("is_falsification_attempt")
                        ):
                            tied = [
                                h for h in (e.get("hypothesis_ids") or [])
                                if h in _open
                            ]
                            if tied:
                                dangling.append(f"{d_id}→{tied}")
                if dangling:
                    warn_parts.append(
                        "Falsification attempts whose hypotheses are still "
                        f"OPEN (record their verdict first): {dangling}. "
                        "Use HypothesisUpdate against each one's pre-registered "
                        "prediction."
                    )
            # (Pending milestones are now a HARD close-gate handled above — by
            # the time we reach this two-shot warn, all milestones are resolved.)
            # WARNING goes first; then any pending notifications.
            return "  ".join(warn_parts) + (("\n\n" + prefix.rstrip()) if prefix.strip() else "")
        # Pre-critic reproducibility gate (HARD): the deliverable must EXECUTE
        # and reproduce lazily BEFORE any critic consult is spent on it. A
        # broken / non-lazy pipeline.ipynb bounces straight back to the strategizer
        # to fix — the critic never wastes a turn reviewing a deliverable that
        # cannot even run. Bounded (N=3) so a persistently-broken pipeline still
        # lets the run end (the post-accept gate then marks it UNGATED).
        _REPRO_MAX = 6  # bounce budget before the run is declared FAILED
        _repro = node._reproduction_gate()
        if _repro is not None:
            node._repro_attempts = getattr(node, "_repro_attempts", 0) + 1
            n = node._repro_attempts
            if n <= _REPRO_MAX:
                node._record_intervention(
                    "REPRO_GATE_BOUNCE", "",
                    f"pre-critic reproduction gate failed (attempt {n}/{_REPRO_MAX})")
                # Escalate: after the first failure, push the agent to DEBUG with
                # CheckDeliverable() rather than re-Write blindly — it is the only
                # way to run pipeline.ipynb and see the real error.
                escalate = (
                    "" if n == 1 else
                    f"\n\nThis is failure {n}/{_REPRO_MAX}. Do NOT re-Write "
                    "pipeline.ipynb blindly. Use CheckDeliverable() to RUN it and "
                    "read the full error, fix the EXACT problem, CheckDeliverable() "
                    "again until it PASSES, then call Done(). After "
                    f"{_REPRO_MAX} failures the run is closed FAILED.")
                return (
                    prefix + "Cannot close yet — pipeline.ipynb failed the "
                    "reproduction gate (the runtime ran it before involving the "
                    "critic):\n\n" + _repro
                    + "\n\nFix it via WriteDeliverable('pipeline.ipynb', …) — and "
                    "verify with CheckDeliverable() before re-calling Done()."
                    + escalate)
            # Genuine non-convergence: the agent could not produce a reproducing
            # deliverable in _REPRO_MAX sighted attempts. Close FAILED (loud,
            # distinct from GATED/UNGATED) via the retrospective round — do NOT
            # spend the critic on a deliverable that does not even reproduce.
            node._record_intervention(
                "REPRO_GATE_FAILED", "",
                f"pipeline.ipynb never reproduced after {n} attempts — run FAILED")
            banner = (
                "## ⛔ FAILED RUN — DELIVERABLE NEVER REPRODUCED\n\n"
                f"pipeline.ipynb failed the reproduction gate on all {n} attempts; "
                "the run could not produce a runnable, lazy deliverable that "
                "re-derives the headline from the canonical ledger. This is a "
                "hard failure, not a gated or ungated conclusion.\n\n"
                "### Last gate error\n" + _repro + "\n\n---\n\n"
            )
            node._awaiting_retro = True
            node._final_summary = banner + summary
            node._done_warned = False
            return prefix + _FAILED_RETROSPECTIVE

        # Second call — run critic gate if critic is in the graph.
        if node._find_critic_name() is not None:
            # Synchronous critic gate
            _notes_dir = node._current_notes_dir
            _study_dir = node._study_dir
            _debug_dir = (
                _notes_dir.parent
                if _notes_dir is not None else None
            )
            notes_path = str(_notes_dir or "")
            _store_dir = (
                str(_debug_dir.parent / "experiment_data")
                if _debug_dir is not None else "(unknown)"
            )
            task_msg = (
                "<mode>GATE</mode>\n\n"
                "Final gate check before run closes. PASS to accept the "
                "conclusion; REVISE/REJECT only for a CRITICAL or MAJOR "
                "objection.\n\n"
                "<paths>\n"
                f"study_dir             = {_study_dir}\n"
                f"debug_dir             = {_debug_dir}\n"
                f"canonical_store       = {_store_dir}\n"
                "  ^ audit the ledger yourself: call RecallStore()/"
                "QueryStore(...) (namespace-aware — they see every design "
                "namespace, not just the default store). Do NOT hand-roll "
                "ExperimentData.from_file(project_dir=...) directly on this "
                "path alone — it misses any namespace store. Do NOT rely "
                "solely on the strategizer's self-reported numbers.\n"
                "delegation_log        = "
                f"{_debug_dir}/delegation_log.jsonl\n"
                f"diagnostics           = "
                f"{_debug_dir}/diagnostics.jsonl\n"
                f"strategizer_notes     = {notes_path}\n"
                "delegations_workspace = "
                f"{_debug_dir}/delegations/\n"
                f"deliverable           = {_study_dir}/pipeline.ipynb "
                "(the runtime EXECUTES the notebook lazily after this gate to "
                "verify the headline re-derives from the ledger with zero new "
                "evals; the notebook's own markdown cells ARE the writeup — "
                "there is no solution.md, do NOT flag it as missing)\n"
                "</paths>\n\n"
                # FULL conclusion — never truncate what the adversarial gate
                # must validate (a head-excerpt would let an over-claim in
                # the body pass unseen). A Done() summary is small; context
                # is not a concern, and final_summary.md is also available.
                f"Proposed conclusion:\n{summary}"
            )
            import json as _json
            ledger_dump = "(no hypotheses)"
            if node._ledger is not None:
                ledger_dump = _json.dumps(
                    {
                        h["id"]: node._ledger.get(h["id"])
                        for h in node._ledger.list_all()
                    },
                    indent=2,
                )
            attempts: list[str] = []
            if node._delegation_log is not None:
                attempts = [
                    f"{r['id']}: "
                    f"phase={r.get('phase')} "
                    f"hypotheses={r.get('hypothesis_ids')} "
                    f"is_falsification_attempt="
                    f"{r.get('is_falsification_attempt', False)}"
                    + (
                        " (linked post-hoc — scrutinise adequacy)"
                        if r.get("attempt_linked_post_hoc") else ""
                    )
                    for r in node._delegation_log.query_all()
                ]
            # Budget-aware framing: tell the critic what was actually
            # spent so it judges the BEST HONEST conclusion reachable
            # within budget, rather than demanding falsification work the
            # budget no longer allows (which strands the close).
            # Eval count = canonical ledger rows (the single source of truth),
            # NOT the sum of logged delegations' self-reported evals: a
            # delegation killed/cancelled mid-flight flushes rows whose evals
            # never reach the log, so the log-sum undercounts and the budget
            # silently overruns. Read the store directly; fall back to the
            # log-sum only for lookup-direct studies that wrote no store.
            _spent = 0
            _notes_sp = getattr(node, "_current_notes_dir", None)
            if _notes_sp is not None:
                try:
                    # Sum across the canonical store AND every design namespace,
                    # so the critic's budget framing reflects ALL real evals (a
                    # canonical-only count under-reports a multi-namespace run and
                    # would have the critic demand work the budget can't afford).
                    from ...instrumented import total_ledgered_evals
                    _spent = int(total_ledgered_evals(
                        _notes_sp.parent.parent / "experiment_data"))
                except Exception:  # noqa: BLE001
                    _spent = 0
            if _spent == 0 and node._delegation_log is not None:
                _spent = sum(
                    (r.get("evals") or 0)
                    for r in node._delegation_log.query_all()
                )
            _bud = getattr(node, "_eval_budget", None)
            _exhausted = _bud is not None and _spent >= _bud
            # Milestones: show the critic each process milestone's resolution +
            # note/reason, so it can flag a hollow SKIP (a study that skipped a
            # gate it actually needed) — skips are unilateral, audited here.
            _ms_lines = []
            _ms_obj = getattr(node, "_milestones", None)
            if _ms_obj is not None:
                for m in _ms_obj.list_all():
                    _ms_lines.append(
                        f"{m['id']} [{m['status']}]: {m['description']}"
                        + (f" — note: {m['note']}" if m.get('note') else ""))
            _ms_block = "\n".join(_ms_lines) or "(none)"
            task_msg += (
                "\n\n<hypothesis_ledger>\n" + ledger_dump
                + "\n</hypothesis_ledger>\n\n"
                "<milestones>\n" + _ms_block
                + "\n</milestones>\n\n"
                "<delegation_flags>\n"
                + "\n".join(attempts)
                + "\n</delegation_flags>\n\n"
                "<budget>\n"
                f"Evaluation budget: {_bud if _bud else 'unspecified'}; "
                f"ledgered evaluations spent: {_spent}"
                + (" (EXHAUSTED)." if _exhausted else ".") + "\n"
                "If the budget is exhausted, judge the BEST HONEST "
                "conclusion reachable within the evals actually spent: an "
                "honest INCONCLUSIVE/negative result whose falsification "
                "attempts were adequate FOR THE REMAINING BUDGET can PASS. "
                "Do NOT REVISE solely to demand evaluations the budget no "
                "longer allows — note them as future work instead.\n"
                "</budget>\n\n"
                "For each hypothesis, judge whether its stated "
                "falsification_criterion was actually tested by "
                "a delegation flagged is_falsification_attempt "
                "— adequacy of the test (given the budget), not mere "
                "presence of the flag."
            )
            critique_text = node._invoke_critic(task_msg)
            verdict = _parse_verdict(critique_text)

            if verdict == "PASS":
                # Conclusion accepted + recorded. Now — and only now —
                # ask the exit interview as a separate turn.
                node._done_warned = False
                node._revise_count = 0
                node._awaiting_retro = True
                node._final_summary = summary
                return prefix + _EXIT_INTERVIEW

            # Non-PASS: reset the two-shot and count the revision internally.
            # After a bounded number of unsatisfiable verdicts, close GRACEFULLY
            # UNGATED rather than looping to recursion-limit. Policy (user, this
            # session): disclose the SUBSTANCE — the critic's standing objections
            # and that leaving them unresolved closes the run UNGATED — but NOT
            # the numeric attempt count (advertising "call Done() N times to
            # close" teaches the agent to exhaust the critic instead of earning a
            # PASS). So the agent is never blindsided by an unexplained ending,
            # and the limit stays un-gameable.
            node._done_warned = False
            node._revise_count = getattr(node, "_revise_count", 0) + 1
            if node._revise_count >= 3:
                banner = (
                    "## ⚠ UNGATED RUN\n\n"
                    "This run is NOT validated: the conclusion did not earn a "
                    "critic PASS — its objections (below) remained unresolved. "
                    "Closing honestly with them on record rather than looping.\n\n"
                    "### Outstanding critic findings\n"
                    + critique_text.strip() + "\n\n---\n\n"
                )
                node._revise_count = 0
                # Route through the retrospective round (D): even an UNGATED
                # close gets interviewed for capability gaps — these are the
                # runs that carry the most friction signal.
                node._awaiting_retro = True
                node._final_summary = banner + summary
                node._done_warned = False
                return prefix + _FAILED_RETROSPECTIVE
            return (
                prefix +
                f"Critic verdict: {verdict}. Address the findings below and "
                "revise your conclusion, then call Done() again — a run closes "
                "only on a critic PASS. If these objections stay unresolved, the "
                "run will close UNGATED with them on record, so resolve the "
                "substance rather than re-submitting unchanged.\n\n" + critique_text
            )

        # No critic — no review stage, so no exit interview; close
        # directly (the interview is "your work was reviewed, now a
        # question", which only applies when a critic gate ran).
        node._done_warned = False
        route["kind"] = "done"
        route["summary"] = summary
        return prefix + "Run complete."

    def FollowUp(question: str) -> str:
        """Ask your delegating party one clarifying question before proceeding.

        Routes to whoever sent you this task: the human operator if you are
        the entry node, or the agent that delegated to you if you are a worker.
        One FollowUp per delegation.  The answer is injected directly into
        your context.  If no answer is available, proceed with best judgment.
        Use it only for genuine briefing ambiguities (or a result so
        surprising it may signal a bug) — never for rhetorical/confirmatory
        questions or to replace your own reasoning.
        """
        if node._ask_count >= node._max_ask:
            return (
                f"FollowUp limit reached ({node._max_ask} per run). "
                "Proceed autonomously with the information you have."
            )
        node._ask_count += 1
        # Prompt the human ONLY when interactive AND stdin is a real terminal.
        # A headless/background run (no TTY) must never block on input() — that
        # raises EOFError. In that case, and whenever the operator gives no
        # answer, notify the agent that no operator is present and let it
        # proceed autonomously.
        import sys as _sys
        _can_prompt = (
            interactive
            and getattr(_sys.stdin, "isatty", lambda: False)()
        )
        if _can_prompt:
            print(f"\n[Node {node._name}] {question}\nAnswer: ", end="", flush=True)
            try:
                answer = input()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer.strip():
                return answer
        return (
            "No operator is present to answer. Proceed autonomously "
            "using only information available in the task message "
            "and files in the study directory."
        )

    def WriteNote(path: str, body: str) -> str:
        """Write a Markdown (.md) note to strategizer_notes/ — free-form
        reasoning: why you chose each Delegate, interim findings, open issues.
        Do NOT write code in notes (embed it in Delegate().intent as plain
        text), and do NOT record priors/posteriors here — those live ONLY in
        the hypothesis ledger (HypothesisPropose/Update)."""
        prefix = node._drain_notifications()
        notes_dir = node._current_notes_dir
        if notes_dir is None:
            return "ERROR: notes_dir not set (run_dir missing from state)."
        bare = Path(path).name
        if bare.endswith(".ipynb"):
            if study_dir is None:
                return prefix + "ERROR: study_dir not set — cannot write .ipynb."
            target = Path(study_dir).resolve() / bare
        else:
            if not bare.endswith(".md"):
                bare = bare + ".md"
            target = Path(notes_dir) / bare
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        return prefix + f"Written: {target}"

    def ReadNote(path: str) -> str:
        """Read a file — or LIST a directory — from the study directory. Use it
        to load PROBLEM_STATEMENT.md, review prior notes, and (importantly) to
        reuse the implementers' work: point it at a delegation workspace
        (workspace_dir/D###/) to LIST its files, then read the script you want
        to consolidate into pipeline.ipynb. Read what you need, not everything."""
        prefix = node._drain_notifications()
        if study_dir is None:
            return "ERROR: study_dir not set."
        # Contain to the study directory. An absolute or ..-escaping path — e.g.
        # ReadNote("/") — otherwise resolves OUTSIDE the study tree, and a
        # recursive listing of "/" walks the entire filesystem and HANGS the run
        # (the tool call never returns, so the strategizer turn never ends and
        # the time backstop, checked only between turns, never fires).
        study_root = Path(study_dir).resolve()
        target = (study_root / path).resolve()
        try:
            target.relative_to(study_root)
        except ValueError:
            return prefix + (
                f"ERROR: {path!r} resolves outside the study directory; "
                "use a path within it (e.g. 'PROBLEM_STATEMENT.md' or 'D001/')."
            )
        if not target.exists():
            return prefix + f"NOT FOUND: {target}"
        if target.is_dir():
            # List files recursively so the agent can discover what an
            # implementer wrote — but BOUND the walk: stop after _MAX files so a
            # large tree can never hang the run.
            _MAX = 300
            found: list[str] = []
            try:
                for p in target.rglob("*"):
                    if p.is_file():
                        found.append(str(p.relative_to(target)))
                        if len(found) > _MAX:
                            break
            except Exception:  # noqa: BLE001
                found = [p.name for p in target.iterdir()]
            truncated = len(found) > _MAX
            entries = sorted(found[:_MAX])
            listing = "\n".join(f"  {e}" for e in entries) or "  (empty)"
            if truncated:
                listing += "\n  … (more files — narrow the path)"
            return prefix + (
                f"{target} is a directory. Files (read one with "
                f"ReadNote('{path.rstrip('/')}/<file>')):\n{listing}"
            )
        return prefix + target.read_text(encoding="utf-8")

    # RecallHistory: demand-driven episodic memory via delegation log.
    if node._delegation_log is not None:
        _dlog = node._delegation_log

        def RecallHistory(n: int = 5) -> str:
            """Return the last n delegations received by this node as (task, deliverable) pairs.
            Call at the start of a delegation to recall prior work. Returns oldest-first."""
            try:
                n = int(n)  # the model may pass "5"; query_received does [-n:]
            except (TypeError, ValueError):
                n = 5
            records = _dlog.query_received(node._name, n)
            if not records:
                return "No prior delegations found."
            parts = []
            for i, r in enumerate(records, 1):
                parts.append(
                    f"== Prior delegation {i} ==\n"
                    f"Task: {r['task']}\n\n"
                    f"Deliverable:\n{r['deliverable']}"
                )
            return "\n\n---\n\n".join(parts)

    def _build_confer(sender_name: str):
        """Factory: returns a Confer closure for any node in the graph.

        Works identically for the orchestrating node, a worker, or a peer —
        sender_name is the only difference. Async: never blocks. A message is
        queued in the TARGET's inbox and delivered when the target next drains
        (orchestrator: each turn via _drain_notifications; worker: collect-on-
        send the next time IT calls Confer). Faithful port of the stashed
        Confer design.
        """
        def Confer(target: str, message: str) -> str:
            """Send an async message to another node in the run.

            Returns immediately — neither side blocks. The message is delivered
            to the target when it next drains its inbox (the orchestrator on its
            next turn; a worker the next time it itself calls Confer). target
            must be a node delegated to at least once this run (ever-woken
            guard), or the orchestrating node itself. Reply by convention with
            Confer(sender_name, "re #N: <answer>").
            """
            with node._registry_lock:
                ever_woken = (target == node._name) or any(
                    e.get("target") == target
                    for e in node._registry.values()
                )
            if not ever_woken:
                return (
                    f"ERROR: {target!r} has never been delegated to "
                    "— cannot Confer with a node that was never woken."
                )
            seq = node._next_confer_seq()
            envelope = (
                f"[Confer #{seq} from {sender_name} → {target}]: {message}\n"
                f"→ reply with Confer(\"{sender_name}\", \"re #{seq}: "
                "<answer>\")"
            )
            with node._confer_inbox_lock:
                node._confer_inbox.setdefault(target, []).append(envelope)
                # Collect-on-send: drain any messages addressed to this sender
                # so replies arrive alongside the send confirmation.
                inbox = node._confer_inbox.pop(sender_name, [])
            inbox_text = ("\n\n".join(inbox) + "\n\n") if inbox else ""
            return (
                inbox_text
                + f"Message queued for {target!r} (confer #{seq}); "
                "delivered when they next drain their inbox."
            )
        return Confer

    def _orchestrator_confer(target: str, message: str) -> str:
        """Confer for the orchestrating node — prepends the notification drain
        (which also delivers this node's own inbox)."""
        prefix = node._drain_notifications()
        return prefix + _build_confer(node._name)(target, message)

    # Topology-injected tools: granted to every orchestrating node because the
    # ability to delegate/recall derives from having outgoing edges, not from a
    # static class declaration. Capability tools (RecallStore/QueryStore/
    # Hypothesis*/Milestone*/...) are declaration-gated below, NOT here.
    closures: dict = {
        "Delegate": Delegate,
        "Wait": Wait,
        "Reply": Reply,
        "FollowUp": FollowUp,
    }

    if node._delegation_log is not None:
        closures["RecallHistory"] = RecallHistory

    # Agent-declared closure tools: inject only what the subclass opted in to.
    # Done / WriteNote / ReadNote are declared in StrategizerAgent.tools;
    # an ImplementerAgent or DebuggerAgent that gains outgoing edges does not
    # declare them and therefore does not receive them.
    _agent_tools: frozenset = frozenset()
    if node._spec is not None:
        _ag = node._spec.nodes.get(node._name)
        if _ag is not None:
            _agent_tools = _ag.tools

    def WriteDeliverable(filename: str, content: str) -> str:
        """Author the deliverable pipeline.ipynb (the ONLY deliverable).

        pipeline.ipynb is the single merged artifact — the writeup AND the
        runnable, lazily-reproducible recipe in one. Its code cells form a
        COMPLETE f3dasm pipeline: every phase (sampling, surrogate/BO, local
        search, validation) actually CALLS get_evaluator(); NO stubs, NO
        placeholder functions, NO 'would go here' comments — AND it resumes
        lazily on the shipped ledger (zero new evals). A notebook that only
        loads the ledger and prints the headline is NOT acceptable, and a stub
        dressed up as real code is still a stub — the gate and the critic reject
        it. Do not try to disguise one. See <deliverable_format> for the cell
        structure (the four f3dasm pillars + the Popperian spine).

        DO NOT REINVENT IT. The implementers you delegated already WROTE and
        VALIDATED this code under workspace_dir/D###/ (see <run_paths>): the LHS
        sampler, the BO loop, the local search that actually found the optimum.
        Before authoring, ReadNote their working scripts and CONSOLIDATE them
        into the notebook's cells — lift proven code, don't re-derive from
        scratch (re-deriving is where you hit bugs and run out of room).

        filename is normally pipeline.ipynb (content must be valid nbformat-v4
        JSON); files declared in config.yaml required_deliverables (e.g.
        replicate.py) may also be written here, verbatim. Verify with
        CheckDeliverable() before Done().
        """
        prefix = node._drain_notifications()
        if node._study_dir is None:
            return "ERROR: study_dir not available."

        p = Path(filename)
        if "/" in filename or "\\" in filename:
            return "ERROR: filename must be a bare name (no path separators)."
        # The primary deliverable is a Jupyter notebook: the notebook IS both the
        # runnable pipeline and the writeup. The ONLY other files writable here are
        # the AUX deliverables the study declared in config.yaml
        # (required_deliverables) — the Done() gate REQUIRES those, so the writing
        # tool must accept them or the run deadlocks (gate demands a file the tool
        # refuses — audit run 20260624T021359). Any other suffix is rejected loudly
        # so the agent doesn't ship a script the gate would never execute.
        _required_aux = {
            Path(x).name for x in (getattr(node, "_required_deliverables", None) or [])
        }
        if p.suffix != ".ipynb" and p.name not in _required_aux:
            return (
                f"ERROR: the deliverable must be pipeline.ipynb (a notebook), "
                f"not {filename!r}. There is no pipeline.py / solution.md — the "
                "notebook's markdown cells ARE the writeup. (Files declared in "
                "config.yaml required_deliverables may also be written here.) See "
                "<deliverable_format>."
            )
        # A .ipynb must be valid notebook JSON — reject a malformed notebook here
        # rather than letting the gate fail opaquely later. Aux files (e.g. a .py)
        # are written verbatim.
        if p.suffix == ".ipynb":
            try:
                import nbformat
                nbformat.reads(content, as_version=4)
            except Exception as exc:  # noqa: BLE001
                return (
                    f"ERROR: {filename!r} is not valid notebook JSON ({exc}). "
                    "Prefer the structured tools (AddPipelineMarkdownCell / "
                    "AddPipelineCell); if you author raw, write valid nbformat v4."
                )

        # Write directly to study_dir/ — the user-visible output location.
        target = Path(node._study_dir) / p.name
        target.write_text(content, encoding="utf-8")
        return prefix + f"Written: {target}"

    def CheckDeliverable() -> str:
        """Dry-run pipeline.ipynb through the SAME controlled reproduction gate
        the runtime applies at Done(), and return the full result WITHOUT closing
        the run. This is how you DEBUG pipeline.ipynb before closing: it executes
        the notebook lazily against the canonical ledger and checks it (a)
        runs cleanly, (b) adds zero new evals, (c) doesn't modify the ledger.
        It also surfaces the printed 'REPRODUCED: <value>' headline (the critic
        checks its provenance; the runtime no longer machine-matches it, so a
        constrained optimum is a valid headline). On failure you get
        the full error (stderr) to fix the exact problem; on success the Done()
        gate will pass. It runs ONLY pipeline.ipynb through the gate — not
        arbitrary code. Call it repeatedly until it passes, THEN call Done()."""
        from ...notebook_exec import required_deliverable_name
        _dname = required_deliverable_name()
        prefix = node._drain_notifications()
        if not (Path(node._study_dir) / _dname).exists():
            # A no-op (nothing to check) does NOT consume the budget.
            return (prefix + f"No {_dname} yet — write it first via "
                    f"WriteDeliverable('{_dname}', …), then CheckDeliverable().")
        # Fail fast if the canonical store holds no computed evaluations — the
        # notebook cannot reproduce a result that was never computed. Key on
        # POPULATED output.csv rows, NOT jobs.csv 'FINISHED' status: reproduction
        # loads via ExperimentData.from_file() (output.csv), which carries values
        # regardless of job status, and a stray worker `data.store()` can reset
        # FINISHED→IN_PROGRESS without dropping any data. Checking jobs.csv here
        # made this guard misfire on a store that is fully present but whose
        # statuses were clobbered (the "no FINISHED rows" false positive).
        # Checked across EVERY store (default + every design namespace) via
        # experiment_stores() — a namespace-only run's default output.csv can
        # exist but be empty, which false-blocked CheckDeliverable even though
        # the ledger was populated (backlog #21's sibling gap).
        _notes = getattr(node, "_current_notes_dir", None)
        if _notes is not None:
            from ...instrumented import RunStateSummary, experiment_stores
            _run_dir = Path(_notes).parent.parent
            _store_root = _run_dir / "experiment_data"
            _has_rows = any(
                RunStateSummary.from_store(s) is not None
                for s in experiment_stores(_store_root)
            )
            if not _has_rows:
                return (prefix +
                    "CheckDeliverable: canonical store has no evaluations yet. "
                    "Run at least one evaluation campaign before calling CheckDeliverable.")
        _BUDGET = 10
        prior = getattr(node, "_check_deliverable_calls", 0)
        if prior >= _BUDGET:
            return (prefix + f"CheckDeliverable budget exhausted ({_BUDGET}/"
                    f"{_BUDGET} used). Stop iterating — write a correct lazy "
                    "pipeline in ONE decisive edit (re-read the LAST error; the "
                    "fix is usually 'load the ledger and skip finished rows', "
                    "not a fresh rewrite), or call Done() to close now (the run "
                    "is recorded FAILED if it still doesn't reproduce).")
        node._check_deliverable_calls = prior + 1
        used = node._check_deliverable_calls
        left = _BUDGET - used
        # Show the budget on EVERY call so the agent paces itself and never hits
        # an unseen wall.
        footer = (
            f"\n\n[CheckDeliverable: {used}/{_BUDGET} used — {left} check"
            f"{'s' if left != 1 else ''} left before you must close with "
            "Done().]")
        problem = node._reproduction_gate()
        if problem is None:
            ok = getattr(node, "_repro_ok_detail", "reproduces cleanly")
            return (prefix + "PASS — pipeline.ipynb " + ok
                    + ". Call Done() now to close." + footer)
        return (prefix + "NOT YET — pipeline.ipynb failed the reproduction gate. "
                "Fix the exact problem below and CheckDeliverable() again:\n\n"
                + problem + footer)

    # ── Structured notebook authoring ────────────────────────────────────────
    # The four f3dasm pillars + the Popperian spine are PLUMBING here: the cell
    # name (= pillar) and the WHY-explainer are REQUIRED arguments, so the agent
    # cannot author a structureless notebook or forget the rationale. Each call
    # builds/updates study_dir/pipeline.ipynb in canonical order via nbformat —
    # no hand-written JSON, no live kernel.
    _PILLARS = ("doe", "data_generation", "ml", "optimization", "analysis")
    _NB_ORDER = ["problem", "hypotheses"]
    for _p in _PILLARS:
        _NB_ORDER += [f"{_p}__why", _p]

    def _load_or_new_notebook():
        import nbformat
        nb_path = Path(node._study_dir) / "pipeline.ipynb"
        if nb_path.exists():
            try:
                return nbformat.read(str(nb_path), as_version=4), nb_path
            except Exception:  # noqa: BLE001 — corrupt → start clean
                pass
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3",
                                     "display_name": "Python 3",
                                     "language": "python"}
        return nb, nb_path

    def _emit_notebook(by_name: dict, nb, nb_path):
        """Re-emit cells: canonical pillars first, then any CUSTOM named cells
        (a confirmed non-pillar section, in insertion order), then any unnamed
        extras (e.g. a runtime provenance stamp) preserved at the end."""
        import nbformat
        ordered = [by_name[k] for k in _NB_ORDER if k in by_name]
        custom = [by_name[k] for k in by_name if k not in _NB_ORDER]
        extras = [c for c in nb.cells
                  if (c.get("metadata", {}) or {}).get("name") not in by_name]
        nb.cells = ordered + custom + extras
        nbformat.write(nb, str(nb_path))

    def _by_name(nb) -> dict:
        out = {}
        for c in nb.cells:
            nm = (c.get("metadata", {}) or {}).get("name")
            if nm:
                out[nm] = c
        return out

    def _rev(source: str) -> str:
        """Stateless content revision tag — a short hash of a cell's source.
        Same source → same rev (survives reloads/checkpoints); computed
        on-demand, never stored, so the gate's name/order metadata is untouched.
        Used as an optimistic-concurrency token: an edit/delete must present the
        rev it last saw, so it cannot blindly overwrite a cell that changed."""
        import hashlib
        return hashlib.sha256((source or "").encode("utf-8")).hexdigest()[:8]

    # The two leading narrative (markdown-only) cells + their heading.
    _NARRATIVE = {"problem": "# Problem & objective", "hypotheses": "## Hypotheses"}

    def AddPipelineMarkdownCell(name: str, content: str) -> str:
        """CREATE one standalone narrative markdown cell in pipeline.ipynb. `name`
        is 'problem' (the question, min/max, success criterion) or 'hypotheses'
        (the registered hypotheses + their falsifiable predictions — the Popperian
        setup). The canonical heading is added for you. CREATE-ONLY: if it already
        exists this errors — change it with EditPipelineCell(name, content=…) or
        remove it with DeletePipelineCell. Creates pipeline.ipynb if absent. (For
        the executable f3dasm pillars use AddPipelineCell instead.)"""
        import nbformat
        prefix = node._drain_notifications()
        if node._study_dir is None:
            return "ERROR: study_dir not available."
        name = (name or "").strip()
        if name not in _NARRATIVE:
            return (f"ERROR: name must be one of {sorted(_NARRATIVE)}, got "
                    f"{name!r}. (Pillar code cells go through AddPipelineCell.)")
        if not (content or "").strip():
            return f"ERROR: `content` is empty for {name!r}."
        nb, nb_path = _load_or_new_notebook()
        by = _by_name(nb)
        if name in by:
            return (prefix + f"ERROR: {name!r} already exists "
                    f"(rev {_rev(by[name].get('source', ''))}). "
                    "AddPipelineMarkdownCell is create-only — change it with "
                    "EditPipelineCell or remove it with DeletePipelineCell first.")
        cell = nbformat.v4.new_markdown_cell(
            _NARRATIVE[name] + "\n\n" + _strip_leading_md_header(content))
        cell.metadata["name"] = name
        by[name] = cell
        _emit_notebook(by, nb, nb_path)
        return prefix + f"Added {name} cell (rev {_rev(cell['source'])}) to pipeline.ipynb."

    def AddPipelineCell(phase: str, why: str, code: str) -> str:
        """CREATE one f3dasm-pillar cell in pipeline.ipynb, preceded by its
        WHY-explainer. `phase` is usually one of: doe, data_generation, ml,
        optimization, analysis — these are the standard pillars. A non-standard
        phase is allowed (a custom design/analysis section); you are asked to
        confirm it once, then it is appended after the standard pillars. `why`
        is the rationale markdown (cite the
        literature; if the pillar was not run, say 'NOT executed (budget)' and
        why). `code` is the cell's Python. Cells are kept in canonical pillar
        order regardless of call order. CREATE-ONLY: if the phase already exists
        this errors — change it with EditPipelineCell or remove it with
        DeletePipelineCell (so you never blindly overwrite a cell that changed
        since you last saw it). The analysis cell must derive the headline from
        the ledger and print exactly 'REPRODUCED: <value>'. Creates
        pipeline.ipynb if absent."""
        import nbformat
        prefix = node._drain_notifications()
        if node._study_dir is None:
            return ("ERROR: no study directory is set for this run — an "
                    "infrastructure condition, not something you did. The "
                    "notebook can't be written; report it if unexpected.")
        phase = (phase or "").strip()
        if not phase:
            return ("ERROR: `phase` is required — the section this cell belongs "
                    "to. Use a standard pillar (doe, data_generation, ml, "
                    "optimization, analysis) or a custom name for a bespoke "
                    "design/analysis section.")
        # The five pillars are the USUAL shape, not a fence. A custom design or
        # analysis can warrant its own section. Adding a cell is fully reversible
        # (edit/delete it), so a non-pillar phase just PROCEEDS with a tip — never
        # a refusal or a confirm. The deliverable's structure must not constrain
        # what science can be expressed.
        _custom_phase = phase not in _PILLARS
        if not (why or "").strip():
            return ("ERROR: `why` is required — every pillar cell needs its "
                    "rationale (the WHY-explainer the writeup always lacked).")
        if not (code or "").strip():
            return f"ERROR: `code` is empty for phase {phase!r}."
        nb, nb_path = _load_or_new_notebook()
        by = _by_name(nb)
        if phase in by:
            return (prefix + f"ERROR: phase {phase!r} already exists "
                    f"(rev {_rev(by[phase].get('source', ''))}). AddPipelineCell "
                    "is create-only — change it with EditPipelineCell or remove "
                    "it with DeletePipelineCell first.")
        wc = nbformat.v4.new_markdown_cell(
            f"### {phase}\n\n" + _strip_leading_md_header(why))
        wc.metadata["name"] = f"{phase}__why"
        cc = nbformat.v4.new_code_cell(code)
        cc.metadata["name"] = phase
        cc.metadata["tags"] = [phase]
        by[f"{phase}__why"], by[phase] = wc, cc
        _emit_notebook(by, nb, nb_path)
        if _custom_phase:
            return (prefix + f"Added custom '{phase}' cell (rev {_rev(code)}) to "
                    f"pipeline.ipynb, after the standard pillars. '{phase}' isn't "
                    f"one of the usual pillars {_PILLARS} — fine for a custom "
                    "design/analysis section; edit or remove it any time with "
                    "EditPipelineCell / DeletePipelineCell.")
        present = [p for p in _PILLARS if p in by]
        missing = [p for p in _PILLARS if p not in by]
        return (prefix + f"Added {phase} cell (rev {_rev(code)}) to "
                f"pipeline.ipynb. Pillars present: {present}."
                + (f" Still missing: {missing}." if missing else
                   " All pillars present — verify with CheckDeliverable()."))

    def EditPipelineCell(name: str, why: str = None, code: str = None,
                         content: str = None, old: str = None, new: str = None,
                         expected_rev: str = None) -> str:
        """Patch an EXISTING cell in pipeline.ipynb. `name` is any named cell: a
        pillar (doe, data_generation, ml, optimization, analysis), its
        '<pillar>__why' explainer, or a narrative cell (problem, hypotheses). Modes:
        • SURGICAL: pass `old`/`new` — literal find/replace on the cell's source;
          `old` must occur EXACTLY once. Self-guarding (if the cell changed, `old`
          won't match), so no `expected_rev` needed. ShowNotebook('<name>') first
          to copy an accurate `old`.
        • FULL-FIELD (REQUIRES `expected_rev` — the rev you last saw, from
          ShowNotebook or a prior Add/Edit; a stale rev is rejected): for a PILLAR
          pass `code=` and/or `why=`; for a MARKDOWN cell (problem / hypotheses /
          <pillar>__why) pass `content=`.
        Create cells with AddPipelineCell / AddPipelineMarkdownCell — this only edits."""
        import nbformat
        prefix = node._drain_notifications()
        if node._study_dir is None:
            return "ERROR: study_dir not available."
        name = (name or "").strip()
        if name not in _NB_ORDER:
            return f"ERROR: unknown cell {name!r}. Valid names: {_NB_ORDER}."
        surgical = old is not None or new is not None
        full = [k for k, v in (("code", code), ("why", why), ("content", content))
                if v is not None]
        if surgical and full:
            return ("ERROR: pass EITHER surgical `old`/`new` OR a full-field value "
                    f"({'/'.join(full)}), not both.")
        if not surgical and not full:
            return ("ERROR: pass `old`/`new` (surgical), or a full-field value: "
                    "`code`/`why` for a pillar, `content` for a markdown cell.")
        nb, nb_path = _load_or_new_notebook()
        by = _by_name(nb)
        if name not in by:
            return (prefix + f"ERROR: {name!r} is not in pipeline.ipynb yet — create "
                    "it with AddPipelineCell / AddPipelineMarkdownCell first. "
                    f"Present: {[k for k in _NB_ORDER if k in by]}.")
        cur_rev = _rev(by[name].get("source", ""))
        is_pillar = name in _PILLARS
        if surgical:
            if old is None or new is None:
                return "ERROR: surgical edit needs BOTH `old` and `new`."
            src = by[name].get("source", "")
            cnt = src.count(old)
            if cnt == 0:
                return (prefix + f"ERROR: `old` not found in {name!r} (it may have "
                        f"changed; current rev {cur_rev}). ShowNotebook('{name}') and retry.")
            if cnt > 1:
                return (prefix + f"ERROR: `old` occurs {cnt}× in {name!r} — include "
                        "surrounding context so it matches exactly once.")
            by[name]["source"] = src.replace(old, new, 1)
        else:
            if expected_rev is None:
                return (prefix + f"ERROR: full-field edit requires `expected_rev` "
                        f"({name!r} is at rev {cur_rev}). ShowNotebook('{name}') to "
                        "confirm the content, then pass that rev.")
            if expected_rev != cur_rev:
                return (prefix + f"ERROR: {name!r} changed since rev {expected_rev} "
                        f"(now {cur_rev}). ShowNotebook('{name}') to see the current "
                        "content, then retry.")
            if is_pillar:
                if content is not None:
                    return (f"ERROR: {name!r} is a pillar — use `code=` and/or "
                            "`why=`, not `content=`.")
                if code is not None:
                    if not code.strip():
                        return f"ERROR: `code` is empty for {name!r}."
                    by[name]["source"] = code
                if why is not None:
                    if not why.strip():
                        return "ERROR: `why` is empty."
                    wname = f"{name}__why"
                    body = f"### {name}\n\n" + _strip_leading_md_header(why)
                    if wname in by:
                        by[wname]["source"] = body
                    else:
                        wc = nbformat.v4.new_markdown_cell(body)
                        wc.metadata["name"] = wname
                        by[wname] = wc
            else:
                # markdown cell: problem / hypotheses / <pillar>__why
                if code is not None or why is not None:
                    return (f"ERROR: {name!r} is a markdown cell — use `content=`, "
                            "not `code=`/`why=`.")
                if not content.strip():
                    return f"ERROR: `content` is empty for {name!r}."
                heading = (_NARRATIVE[name] if name in _NARRATIVE
                           else f"### {name[:-len('__why')]}")
                by[name]["source"] = heading + "\n\n" + _strip_leading_md_header(content)
        _emit_notebook(by, nb, nb_path)
        new_rev = _rev(by[name].get("source", ""))
        return prefix + f"Edited {name} in pipeline.ipynb (rev {cur_rev} → {new_rev})."

    def DeletePipelineCell(name: str, expected_rev: str = None) -> str:
        """Remove a cell from pipeline.ipynb. `name` is any named cell: a pillar
        (doe, data_generation, ml, optimization, analysis) — which also removes its
        '<pillar>__why' explainer — or a narrative cell (problem, hypotheses) or a
        '<pillar>__why'. Use it to drop a part you decided not to keep instead of
        leaving dead/placeholder content. REQUIRES `expected_rev` (the rev you last
        saw, from ShowNotebook or a prior Add/Edit) so you cannot delete a cell that
        changed since you last saw it."""
        import nbformat
        prefix = node._drain_notifications()
        if node._study_dir is None:
            return "ERROR: study_dir not available."
        name = (name or "").strip()
        if name not in _NB_ORDER:
            return f"ERROR: unknown cell {name!r}. Valid names: {_NB_ORDER}."
        nb, nb_path = _load_or_new_notebook()
        by = _by_name(nb)
        if name not in by:
            return prefix + f"Nothing to delete: {name!r} not in pipeline.ipynb."
        cur_rev = _rev(by[name].get("source", ""))
        if expected_rev is None:
            return (prefix + f"ERROR: delete requires `expected_rev` ({name!r} is at "
                    f"rev {cur_rev}). ShowNotebook('{name}') to confirm, then pass that rev.")
        if expected_rev != cur_rev:
            return (prefix + f"ERROR: {name!r} changed since rev {expected_rev} "
                    f"(now {cur_rev}). ShowNotebook('{name}') to see the current "
                    "content, then retry the delete.")
        # A pillar drops with its WHY-explainer; any other named cell drops alone.
        targets = {name, f"{name}__why"} if name in _PILLARS else {name}
        nb.cells = [c for c in nb.cells
                    if (c.get("metadata", {}) or {}).get("name") not in targets]
        nbformat.write(nb, str(nb_path))
        present = [p for p in _PILLARS if p in _by_name(nb)]
        return (prefix + f"Deleted {name} from pipeline.ipynb. "
                f"Pillars present: {present}.")

    def ShowNotebook(name: str = None) -> str:
        """Read pipeline.ipynb back. NO argument → a BRIEF table of contents
        LISTING EVERY CELL BY NAME in canonical order, each with its type, rev,
        and first source line, plus which pillars are present/missing — call this
        to see what exists before editing. With `name` (problem, hypotheses, a
        pillar, or a '<pillar>__why' explainer) → the FULL source of that cell and
        its rev (the rev you then pass as `expected_rev` to EditPipelineCell /
        DeletePipelineCell). Read-only; never creates the file; free."""
        prefix = node._drain_notifications()
        if node._study_dir is None:
            return "ERROR: study_dir not available."
        nb_path = Path(node._study_dir) / "pipeline.ipynb"
        if not nb_path.exists():
            return prefix + ("pipeline.ipynb does not exist yet — author it with "
                             "AddPipelineMarkdownCell / AddPipelineCell.")
        nb, _ = _load_or_new_notebook()
        by = _by_name(nb)
        if name is not None:
            name = name.strip()
            if name not in by:
                return (prefix + f"ERROR: no cell named {name!r}. Present: "
                        f"{[k for k in _NB_ORDER if k in by]}.")
            c = by[name]
            src = c.get("source", "")
            ctype = c.get("cell_type", "?")
            return (prefix + f"--- {name} ({ctype}, rev {_rev(src)}) ---\n"
                    + (src or "(empty)"))
        # Brief: every named cell, canonical order then extras.
        names = [k for k in _NB_ORDER if k in by]
        names += [k for k in by if k not in _NB_ORDER]
        lines = []
        for nm in names:
            c = by[nm]
            src = c.get("source", "")
            first = (src.splitlines()[0] if src.strip() else "(empty)")[:80]
            lines.append(f"  {nm} ({c.get('cell_type', '?')}, rev {_rev(src)}): "
                         f"{first}")
        present = [p for p in _PILLARS if p in by]
        missing = [p for p in _PILLARS if p not in by]
        return (prefix + "pipeline.ipynb cells (canonical order):\n"
                + "\n".join(lines)
                + f"\n\nPillars present: {present}."
                + (f" Missing: {missing}." if missing else " All present."))

    def LedgerBreakdown() -> str:
        """Show, per experiment and per delegation, how many ledgered evaluations
        each contributed — read live from the canonical store. Use this at REPORT
        time to DERIVE eval counts for the writeup/hypothesis evidence instead of
        copying numbers from a plan or a delegation's notes (those drift from what
        actually landed in the ledger). Read-only; does NOT spend eval budget.

        Output is one line per experiment (the baseline store is 'default'; each
        design parametrization by its registered name) with its total and a
        per-delegation split, e.g.::

            polar: 90 total  (D006: 50, D004: 40)

        The numbers here are the ones pipeline.ipynb will reproduce — quote THESE,
        never a remembered figure."""
        prefix = node._drain_notifications()
        notes = getattr(node, "_current_notes_dir", None)
        if notes is None:
            return prefix + "ERROR: no run context available."
        store_root = notes.parent.parent / "experiment_data"
        from ...instrumented import ledger_breakdown
        rows = ledger_breakdown(store_root)
        if not rows:
            return (prefix + "No ledgered evaluations yet — the canonical store "
                    "is empty. Run a campaign delegation first.")
        lines = []
        for r in rows:
            split = ", ".join(
                f"{d}: {n}" for d, n in sorted(r["per_delegation"].items()))
            lines.append(
                f"{r['experiment']}: {r['total']} total"
                + (f"  ({split})" if split else ""))
        grand = sum(r["total"] for r in rows)
        # Ground the spent/remaining number against the budget so it is READ,
        # not hand-computed (agents flip spent<->remaining: run 20260628T130525
        # asserted "200 remain" when 200 were spent of 300 → UNGATED).
        budget = None
        try:
            import json as _json
            _cfg = notes.parent.parent / "debug" / "run_config.json"
            if _cfg.exists():
                budget = _json.loads(_cfg.read_text()).get("eval_budget")
        except Exception:  # noqa: BLE001
            budget = None
        if budget:
            lines.append(
                f"— run total: {grand} of {int(budget)} eval budget spent "
                f"— {max(int(budget) - grand, 0)} remaining")
        else:
            lines.append(f"— run total: {grand} ledgered evaluations")
        return prefix + "\n".join(lines)

    def RunScratch(code: str) -> str:
        """Run a short Python snippet against a COPY of the canonical ledger and
        return its stdout/stderr — your scratchpad for INSPECTING state before
        committing it to pipeline.ipynb. f3dasm and a3dasm are importable and
        F3DASM_CANONICAL_STORE points at a temp copy of the ledger, so you can
        e.g. ``from a3dasm import load_experiments; experiments =
        load_experiments()`` (returns every store — default AND every design
        namespace — as ``{name: ExperimentData}``; a bare
        ``ExperimentData.from_file(os.environ['F3DASM_CANONICAL_STORE'])`` only
        sees the default store and silently misses namespace evals), print best
        values, check a path resolves, or verify a DataFrame populates. Runs
        against a COPY — it cannot touch the real ledger or pipeline.ipynb —
        and does NOT count toward the eval budget. Use it to debug instead of
        guessing (e.g. 'does hypotheses.json load? does h_dict populate?')
        rather than discovering a silent bug only at CheckDeliverable."""
        import json as _json
        import shutil as _shutil
        import subprocess as _sub
        import tempfile as _tempfile
        prefix = node._drain_notifications()
        if not (code or "").strip():
            return prefix + "ERROR: `code` is empty."
        notes = getattr(node, "_current_notes_dir", None)
        if notes is None:
            return prefix + "ERROR: no run context available for scratch execution."
        run_dir = notes.parent.parent
        store_dir = run_dir / "experiment_data"
        run_config = run_dir / "debug" / "run_config.json"
        sandbox = Path(_tempfile.mkdtemp(prefix="f3dasm_scratch_"))
        try:
            sb_store = sandbox / "experiment_data"
            if store_dir.exists():
                _shutil.copytree(store_dir, sb_store)
            else:
                sb_store.mkdir(parents=True, exist_ok=True)
            _cfg = (_json.loads(run_config.read_text())
                    if run_config.exists() else {})
            _cfg["store_dir"] = str(sb_store)
            sb_cfg = sandbox / "run_config.json"
            sb_cfg.write_text(_json.dumps(_cfg))
            from ...notebook_exec import sandbox_env
            env = sandbox_env(sb_store, sb_cfg, study_root=node._study_dir)
            snippet = sandbox / "_scratch.py"
            snippet.write_text(code)
            try:
                from ...notebook_exec import run_deliverable
                proc = run_deliverable(
                    snippet, cwd=sandbox, env=env, timeout=120)
            except _sub.TimeoutExpired:
                return (prefix + "Scratch snippet exceeded 120s and was killed. "
                        "Keep it lightweight — load the ledger and print; do not "
                        "re-run a campaign.")
            out = (proc.stdout or "")[-4000:]
            err = (proc.stderr or "")[-2000:]
            return (prefix + f"[scratch exit {proc.returncode}]\n--- stdout ---\n"
                    + (out or "(empty)")
                    + (f"\n--- stderr ---\n{err}" if err.strip() else ""))
        finally:
            _shutil.rmtree(sandbox, ignore_errors=True)

    def RunPipelineCell(name: str = None) -> str:
        """Execute pipeline.ipynb against a COPY of the canonical ledger and return
        a PER-CELL trace — the cell-level debugger CheckDeliverable's binary
        pass/fail lacks. With `name` (a pillar cell: doe / data_generation / ml /
        optimization / analysis) it runs top-to-bottom UP TO AND INCLUDING that
        cell and reports it — cells share kernel state, so this pinpoints WHICH
        cell breaks reproduction and shows its exact traceback + stdout. With no
        name it runs the WHOLE notebook and reports every cell plus the first
        failure. Runs against a COPY (cannot touch the real ledger or
        pipeline.ipynb) and does NOT count toward the eval budget. Use it to
        localize a CheckDeliverable failure to one cell before editing, instead of
        re-running the binary gate blindly."""
        import json as _json
        import shutil as _shutil
        import subprocess as _sub
        import tempfile as _tempfile
        prefix = node._drain_notifications()
        nb_path = Path(node._study_dir) / "pipeline.ipynb"
        if not nb_path.exists():
            return prefix + ("No pipeline.ipynb yet — author it first "
                             "(AddPipelineCell / AddPipelineMarkdownCell).")
        notes = getattr(node, "_current_notes_dir", None)
        if notes is None:
            return prefix + "ERROR: no run context available for cell execution."
        run_dir = notes.parent.parent
        store_dir = run_dir / "experiment_data"
        run_config = run_dir / "debug" / "run_config.json"
        sandbox = Path(_tempfile.mkdtemp(prefix="f3dasm_cell_"))
        try:
            sb_store = sandbox / "experiment_data"
            if store_dir.exists():
                _shutil.copytree(store_dir, sb_store)
            else:
                sb_store.mkdir(parents=True, exist_ok=True)
            _cfg = (_json.loads(run_config.read_text())
                    if run_config.exists() else {})
            _cfg["store_dir"] = str(sb_store)
            sb_cfg = sandbox / "run_config.json"
            sb_cfg.write_text(_json.dumps(_cfg))
            from ...notebook_exec import sandbox_env
            env = sandbox_env(sb_store, sb_cfg, study_root=node._study_dir)
            try:
                from ...notebook_exec import diagnose_notebook
                trace = diagnose_notebook(
                    nb_path, cwd=sandbox, env=env, timeout=180, upto_name=name)
            except _sub.TimeoutExpired:
                return (prefix + "Notebook diagnosis exceeded 180s and was killed "
                        "— a cell is running a real campaign; it should load the "
                        "ledger lazily, not recompute.")
            if trace.get("missing_name"):
                return (prefix + f"No cell named {name!r}. Valid pillar names: "
                        "doe, data_generation, ml, optimization, analysis "
                        "(use ShowNotebook() to see the cells).")
            lines = []
            scope = (f"up to & including '{name}'" if name
                     else "whole notebook")
            for c in trace["cells"]:
                if c["cell_type"] != "code":
                    continue
                tag = "ERROR" if c["errored"] else "ok"
                nm = c["name"] or f"cell{c['index']}"
                lines.append(f"  [{tag}] {nm}")
                out = (c["stdout"] or "").strip()
                if out:
                    lines.append("        stdout: " + out[-300:].replace("\n", "\n        "))
                if c["errored"]:
                    lines.append("        " + (c["error"] or "").strip()[-700:].replace("\n", "\n        "))
            fe = trace["first_error"]
            head = (
                f"{'TIMED OUT — ' if trace['timed_out'] else ''}"
                f"per-cell trace ({scope}, against a COPY of the ledger):\n"
                + ("\n".join(lines) or "  (no code cells)")
            )
            verdict = (
                f"\n\nFIRST FAILURE: cell '{fe['name'] or fe['index']}' — fix this "
                "cell, then RunPipelineCell() again or CheckDeliverable()."
                if fe else
                "\n\nAll code cells ran without error against the copy. If "
                "CheckDeliverable still fails, the issue is the gate's checks "
                "(zero-new-evals / REPRODUCED line / ledger unchanged), not a cell "
                "exception."
            )
            return prefix + head + verdict
        finally:
            _shutil.rmtree(sandbox, ignore_errors=True)

    if "Done" in _agent_tools:
        closures["Done"] = Done
    if "EditPipelineCell" in _agent_tools:
        closures["EditPipelineCell"] = EditPipelineCell
    if "DeletePipelineCell" in _agent_tools:
        closures["DeletePipelineCell"] = DeletePipelineCell
    if "ShowNotebook" in _agent_tools:
        closures["ShowNotebook"] = ShowNotebook
    if "RunScratch" in _agent_tools:
        closures["RunScratch"] = RunScratch
    if "LedgerBreakdown" in _agent_tools:
        closures["LedgerBreakdown"] = LedgerBreakdown
    if "RunPipelineCell" in _agent_tools:
        closures["RunPipelineCell"] = RunPipelineCell
    if "WriteNote" in _agent_tools:
        closures["WriteNote"] = WriteNote
    if "ReadNote" in _agent_tools:
        closures["ReadNote"] = ReadNote
    if "WriteDeliverable" in _agent_tools:
        closures["WriteDeliverable"] = WriteDeliverable
    if "CheckDeliverable" in _agent_tools:
        closures["CheckDeliverable"] = CheckDeliverable
    if "AddPipelineCell" in _agent_tools:
        closures["AddPipelineCell"] = AddPipelineCell
    if "AddPipelineMarkdownCell" in _agent_tools:
        closures["AddPipelineMarkdownCell"] = AddPipelineMarkdownCell
    if "Confer" in _agent_tools:
        closures["Confer"] = _orchestrator_confer
    # GetStatus / CancelDelegation are now OPT-IN (plug-and-play), not always-on.
    # Their defs above are intact; they are simply not granted unless an agent
    # lists them in its `tools`. PRODUCTION agents do not, so:
    #   - GetStatus is dropped: completions are PUSHED via _notifications each
    #     turn + Confer supersedes polling.
    #   - CancelDelegation is dropped (drop-but-don't-delete) pending the
    #     cooperative-stop decision; restore by adding the name to an agent's
    #     `tools` (one line), exactly like the debugger agent is plug-and-play.
    if "GetStatus" in _agent_tools:
        closures["GetStatus"] = GetStatus
    if "CancelDelegation" in _agent_tools:
        closures["CancelDelegation"] = CancelDelegation
    # ConsultHandbook is injected universally at adapter construction
    # (agent_runtime._make_adapter) — no per-node duplication here.

    # Capability closures are DECLARATION-GATED (single source of truth = the
    # Agent's `tools`), exactly like the notebook/Done/notes tools above.
    # Hypothesis MUTATE tools go only to agents that declare them (the
    # strategizer); a stateless worker must never mutate the shared ledger.
    _hyp = node._build_hypothesis_closures()
    for _t in ("HypothesisPropose", "HypothesisUpdate",
               "LinkFalsificationAttempt"):
        if _t in _agent_tools and _t in _hyp:
            closures[_t] = _hyp[_t]
    # Milestone tools (process policy) — declaration-gated too.
    if hasattr(node, "_build_milestone_closures"):
        for _t, _fn in node._build_milestone_closures().items():
            if _t in _agent_tools:
                closures[_t] = _fn
    # Read-only ledger/store tools — declaration-gated and shared verbatim with
    # leaf WorkerNodes (see WorkerNode.__init__), so the exposure surface is
    # identical across node types.
    closures.update(build_declared_shared_closures(node, _agent_tools))

    # AskForFeedback is only injected when a critic node is
    # connected AND this is the entry node (only the entry node
    # gates Done).
    critic_name: str | None = node._find_critic_name()
    spec = node._spec

    if critic_name is not None:
        _critic_name = critic_name
        _node = node
        _critic_desc = spec.nodes[critic_name].description if spec else ""

        def AskForFeedback(hypothesis_ids: list | None = None) -> str:
            """Placeholder — __doc__ overridden below."""
            # Resolve hypothesis IDs
            h_ids: list[str] = []
            if hypothesis_ids is not None:
                h_ids = list(hypothesis_ids)
            elif _node._ledger is not None:
                h_ids = [h["id"] for h in _node._ledger.list_all()]

            started_at = datetime.now(tz=timezone.utc).isoformat(
                timespec="seconds"
            )
            task_msg = _node._build_feedback_task_msg(h_ids)
            text = _node._invoke_critic(task_msg)

            # Log to delegation log
            if _node._delegation_log is not None:
                _node._delegation_log.record(
                    id=f"FB{datetime.now(tz=timezone.utc).strftime('%H%M%S')}",
                    from_node=_node._name,
                    to_node=_critic_name,
                    task="AskForFeedback (synchronous audit)",
                    deliverable=text,
                    hypothesis_ids=h_ids,
                    started_at=started_at,
                    completed_at=datetime.now(
                        tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                    status="FEEDBACK",
                    tokens_in=0,
                    tokens_out=0,
                    cost_usd=None,
                )

            return text

        AskForFeedback.__doc__ = (
            f"Synchronous find-only audit by: {_critic_desc} "
            "PASS is not a valid verdict — only REVISE or REJECT. "
            "hypothesis_ids: H-ids to focus on; None = all hypotheses auto-injected. "
            "Use Done() for the final gate check."
        )
        closures["AskForFeedback"] = AskForFeedback

    # Wrap every closure so ERROR returns and exceptions are counted.
    _node_name = node._name
    return {k: node._wrap_closure(v, _node_name) for k, v in closures.items()}
