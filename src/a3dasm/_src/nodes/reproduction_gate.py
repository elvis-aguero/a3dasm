"""The reproduction gate: is ``pipeline.ipynb`` a faithful, lazy re-derivation?

``_missing_deliverables`` says which required deliverables are absent;
``_reproduction_gate`` executes the notebook in a hermetic sandbox copy of
the canonical store and checks it exits cleanly, adds zero oracle rows,
rewrites none, and (when it declares both) states the headline it computes.
Called from ``CheckDeliverable`` and ``Done`` (``nodes/tools/routing``). A
mixin on the strategizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime.graph_state import AgenticState


def _headline_consistency(stdout: str) -> str | None:
    """Cross-check the notebook's STATED answer against its COMPUTED one.

    If the deliverable prints BOTH a freshly-computed ``REPRODUCED: <v>`` and a
    ``CLAIMED_HEADLINE: <v>`` (the value its write-up states), assert they
    agree within a relative tolerance. Returns an error string on mismatch,
    else None. LENIENT by design: if either marker is absent it returns None,
    so it adds no new failure mode (and no wait) when the convention isn't used
    — it only catches an internally self-contradicting deliverable (run
    20260705T181941: prose said 0.3644 while an idxmax cell printed a 0.3648
    noise row, and the gate waved it through for 4 rounds).
    """
    import re as _re

    def _grab(tag: str):
        m = _re.search(
            tag + r":\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
            stdout or "")
        return float(m.group(1)) if m else None

    rep, claim = _grab("REPRODUCED"), _grab("CLAIMED_HEADLINE")
    if rep is None or claim is None:
        return None
    if abs(rep - claim) > 1e-9 + 1e-3 * abs(claim):
        return (
            f"Headline inconsistency: the write-up states CLAIMED_HEADLINE="
            f"{claim} but the notebook's own computation prints REPRODUCED="
            f"{rep}. The reported answer must be what the notebook computes — "
            "fix the selection (e.g. an idxmax picking a noise/near-duplicate "
            "row) or the prose so the stated and computed headlines agree.")
    return None


class ReproductionGateMixin:
    def _missing_deliverables(self, state: AgenticState) -> list[str]:
        """Return required deliverable paths not present at study_dir yet.

        The single deliverable (pipeline.ipynb) is required, authored before
        Done() is accepted, UNLESS the study turns off pipeline_deliverable —
        it is the human-readable recipe AND the reproduction in one notebook:
        the runtime executes it lazily (see _reproduction_gate) to verify the
        headline re-derives from the ledger with zero new evals, which makes
        no sense for a study with no ledger at all. Requiring it unconditionally
        (BACKLOG #30) meant a pure-derivation study's strategizer could never
        satisfy Done() regardless of what its own tools/prompt said — this was
        the third of three places that assumption was baked in (the other two,
        the injected notebook_deliverable_spec() preamble and the notebook-
        authoring tools themselves, are already gated the same way). Additional
        paths can still be declared in state['required_deliverables']
        regardless of this flag.
        """
        from ..evaluation.notebook_exec import required_deliverable_name
        from ..runtime import settings
        study_dir = Path(state.get("study_dir", "."))
        # WriteDeliverable writes BARE names to study_dir/ (it rejects path
        # separators). Normalise any configured path to its basename so a stray
        # 'workspace/…' prefix in a study config can't spuriously flag a present
        # deliverable as missing.
        required = list(state.get("required_deliverables") or [])
        if settings.get_bool("pipeline_deliverable", True):
            required = [required_deliverable_name()] + required
        seen: set[str] = set()
        missing: list[str] = []
        for p in required:
            name = Path(p).name
            if name in seen:
                continue
            seen.add(name)
            if not (study_dir / name).exists():
                missing.append(name)
        return missing

    def _reproduction_gate(self, state: AgenticState | None = None) -> str | None:
        """Execute pipeline.ipynb under a CONTROLLED reproduction gate.

        The binding reproducibility check. The pipeline must:
          (a) finish cleanly within a time ceiling (no heavy from-scratch run);
          (b) add ZERO new oracle rows (lazy: skip FINISHED evals);
          (c) NOT modify/delete existing ledger rows (integrity — no faking the
              zero-delta by delete+re-add or value rewrite);
          (d) print ``REPRODUCED: <value>`` — an informational headline marker
              for the critic/human; the runtime does NOT gate on it. Headline
              grounding (the value traces to a real ledger row) is the critic's
              HEADLINE PROVENANCE check, not an independent runtime extremum
              match (which wrongly rejected constrained optima).
        Returns None on PASS (and stashes ``self._repro_ok_detail``), else a
        problem string. Skips silently when there is no run context. Callable
        without ``state`` — study dir comes from ``self._study_dir``.

        On PASS the deliverable is a faithful, lightweight, lazy reproduction —
        not a script doing "sneaky stuff" unrelated to validating the pipeline.
        """
        import json as _json
        import re
        import shutil
        import subprocess
        import tempfile

        study_dir = (
            Path(self._study_dir) if getattr(self, "_study_dir", None) is not None
            else Path((state or {}).get("study_dir", "."))
        )
        # The deliverable is pipeline.ipynb. (A .py is still executable by the
        # executor-agnostic gate, kept only as a fallback for gate-logic tests;
        # the notebook is preferred when both are present.) Absence is left to
        # _missing_deliverables.
        deliverable = next(
            (study_dir / n for n in ("pipeline.ipynb", "pipeline.py")
             if (study_dir / n).exists()),
            None,
        )
        if deliverable is None:
            return None  # absence is handled by _missing_deliverables
        notes = self._current_notes_dir
        if notes is None:
            return None  # no run dir context (e.g. non-debug) — skip the gate
        run_dir = notes.parent.parent              # …/runs/<id>
        store_dir = run_dir / "experiment_data"
        run_config = run_dir / "debug" / "run_config.json"

        def _ledger_snapshot(store_root: Path) -> tuple[int, str]:
            """(row_count, content_hash) across EVERY store under store_root:
            the canonical/default store PLUS every design-namespace sibling
            (store_root/<namespace>/), via the same experiment_stores()
            aggregation LedgerBreakdown/ScienceMonitor already use. A single-
            store read here would miss a non-lazy write into a namespace
            store during "reproduction" — the sandbox copy this is called
            against is already namespace-complete (namespace stores nest
            under store_root), only the read needs to look past the default.

            content_hash is order-independent (sorted rounded values, tagged
            by store so identical values in two different stores can't
            false-collide) so a faithful lazy re-store doesn't false-trip it.
            """
            import hashlib

            from ..evaluation.ledger_summary import experiment_stores

            total_rows = 0
            all_rows: list[tuple] = []
            for store in experiment_stores(store_root):
                try:
                    from f3dasm import ExperimentData
                    data = ExperimentData.from_file(project_dir=store)
                    _, out = data.to_pandas()
                except Exception:  # noqa: BLE001
                    continue
                total_rows += len(out)
                cols = [c for c in out.columns if not str(c).startswith("_")]
                if not cols:
                    continue
                vals = out[cols].round(10)
                all_rows.extend(
                    (store.name,) + tuple(r) for r in vals.to_numpy().tolist()
                )
            # key=repr, not a bare sort: a row is (store.name, *column values),
            # and a design space can legitimately mix numeric columns with
            # non-numeric ones (e.g. a string mechanism/family label). Two
            # same-namespace rows that diverge at a non-numeric column crash
            # Python's tuple comparison (float < str is undefined) the moment
            # they're compared — repr() is always string-comparable regardless
            # of what each column holds, and dropping non-numeric columns
            # instead would silently weaken the reproduction hash itself.
            h = hashlib.sha256(
                repr(sorted(all_rows, key=repr)).encode()
            ).hexdigest()
            return total_rows, h

        # ── HERMETIC SANDBOX ──────────────────────────────────────────────────
        # CRITICAL: run the deliverable against a COPY of the canonical store, never
        # the live one. A faithful lazy pipeline adds nothing; a NON-lazy one
        # (re-evaluating) writes its evals into the THROWAWAY copy — we detect
        # that as "not lazy" while the real ledger stays pristine. Without this,
        # checking a non-lazy pipeline pollutes + inflates the canonical store
        # (and CheckDeliverable could be looped to balloon it without bound).
        before_n, before_hash = _ledger_snapshot(store_dir)
        if before_n == 0:
            return (
                "Canonical store has no rows — the campaign has not been "
                "evaluated yet. Run the delegation pipeline first so the "
                "ledger is populated, then the notebook can be reproduced "
                "lazily against those rows.")
        sandbox = Path(tempfile.mkdtemp(prefix="f3dasm_repro_"))
        try:
            sb_store = sandbox / "experiment_data"
            if store_dir.exists():
                shutil.copytree(store_dir, sb_store)
            else:
                sb_store.mkdir(parents=True, exist_ok=True)
            # A sandbox run_config so get_evaluator() also writes to the COPY
            # (it resolves the store from run_config["store_dir"], not the env).
            sb_run_config = sandbox / "run_config.json"
            if run_config.exists():
                _cfg = _json.loads(run_config.read_text())
            else:
                _cfg = {}
            _cfg["store_dir"] = str(sb_store)
            _cfg["lock_path"] = str(sb_store / "experiment_data" / ".lock")
            sb_run_config.write_text(_json.dumps(_cfg))

            from ..evaluation.notebook_exec import sandbox_env
            env = sandbox_env(
                sb_store, sb_run_config, study_root=self._study_dir)
            _timeout = (
                max(0.1 * self._budget_seconds, 180.0)
                if self._budget_seconds else 300.0
            )
            try:
                # Executor-agnostic: a .ipynb runs via nbclient (in-env kernel),
                # a .py via subprocess — both return a CompletedProcess and raise
                # TimeoutExpired on timeout, so the asserts below are unchanged.
                from ..evaluation.notebook_exec import run_deliverable
                proc = run_deliverable(
                    deliverable, cwd=sandbox, env=env, timeout=_timeout)
            except subprocess.TimeoutExpired:
                return (
                    f"{deliverable.name} did not finish within {_timeout:.0f}s. A "
                    "reproduction must be lightweight — load the ledger and skip "
                    "finished evals and heavy refits (cache-or-load surrogates). "
                    "Make it lazy.")
            after_n, after_hash = _ledger_snapshot(sb_store)
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

        # (a) clean exit — surface a generous stderr tail for sighted debugging.
        if proc.returncode != 0:
            return (
                f"{deliverable.name} FAILED to run (exit {proc.returncode}). It must "
                "load the ledger and derive the headline cleanly. Stderr:\n"
                + (proc.stderr or "")[-3000:]
                + ("\n\nStdout tail:\n" + proc.stdout[-800:]
                   if proc.stdout else ""))
        # (b) zero new evals (lazy).
        if after_n != before_n:
            return (
                f"{deliverable.name} is NOT lazy: re-running it changed the ledger row "
                f"count ({before_n} → {after_n}). It must LOAD the ledger "
                "(ExperimentData.from_file) and reach the oracle only via "
                "get_evaluator() so FINISHED rows are skipped — zero new evals.")
        # (c) integrity — existing rows unchanged.
        if before_hash and after_hash and before_hash != after_hash:
            return (
                f"{deliverable.name} MODIFIED existing ledger rows. A reproduction must "
                "read the ledger READ-ONLY (it may re-store identical rows, but "
                "must not rewrite values or delete+re-add). Do not tamper with "
                "the canonical store.")
        # (d) The printed ``REPRODUCED:`` line is an informational headline
        # marker for the critic / human reader — the runtime no longer gates on
        # it. Headline GROUNDING (the value traces to a real ledger row) is
        # owned by the critic's HEADLINE PROVENANCE check; an independent
        # runtime extremum match wrongly rejected legitimate CONSTRAINED optima
        # (a constrained best is, by definition, not an objective extremum), so
        # it forced studies to headline their infeasible unconstrained extremum
        # — see audit run 20260624T021359.
        # (e) internal consistency — if the notebook declares CLAIMED_HEADLINE
        # (the value its write-up states), it must equal the freshly-computed
        # REPRODUCED. Lenient: skips when the marker is absent.
        _hc = _headline_consistency(proc.stdout or "")
        if _hc is not None:
            return _hc
        m = re.search(r"REPRODUCED:\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
                      proc.stdout or "")
        headline = f", REPRODUCED={m.group(1)}" if m else ""
        self._repro_ok_detail = (
            f"reproduced cleanly ({before_n} rows, unchanged, 0 new evals, "
            f"ran in <{_timeout:.0f}s{headline})")
        return None
