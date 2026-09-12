"""Notebook/deliverable-authoring tool closures: WriteDeliverable,
CheckDeliverable, AddPipelineMarkdownCell, AddPipelineCell, EditPipelineCell,
DeletePipelineCell, ShowNotebook, LedgerBreakdown, RunScratch,
RunPipelineCell. Built per-node via build_notebook_closures(node) — the node
is passed in so closures reach its state. Extracted verbatim from the former
build_routing_tools()."""
from __future__ import annotations

import re
from pathlib import Path

# Tool names whose entire purpose is authoring/checking pipeline.ipynb.
# Stripped from a strategizer's effective toolset when pipeline_deliverable
# is false (BACKLOG #30) — Done is deliberately NOT in this set, it is still
# needed to close a run regardless of whether a notebook exists.
_NOTEBOOK_TOOL_NAMES = frozenset({
    "WriteDeliverable", "CheckDeliverable",
    "AddPipelineCell", "AddPipelineMarkdownCell",
    "EditPipelineCell", "DeletePipelineCell",
    "ShowNotebook", "RunPipelineCell",
})



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



def build_notebook_closures(node) -> dict:
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

                from ....evaluation.notebook_exec import repair_code_cells
                nb = nbformat.reads(content, as_version=4)
                # nbformat.reads() accepts a code cell missing `outputs` with
                # no validation error (confirmed empirically) — repair it here
                # so a hand-authored notebook can't crash a later
                # nbformat.write elsewhere (AddPipelineMarkdownCell, the
                # final provenance stamp) with AttributeError: outputs.
                repair_code_cells(nb)
                content = nbformat.writes(nb)
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
        from ....evaluation.notebook_exec import required_deliverable_name
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
            from ....evaluation.ledger_summary import (
                RunStateSummary,
                experiment_stores,
            )
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
        # <deliverable_format> step 7 substitutes a literal '## Verdict &
        # result' heading for the usual WHY-explainer immediately ahead of
        # the analysis pillar's code cell — same "narrative heading then
        # code" shape as every other pillar, just with a fixed name/text
        # instead of a free-form WHY paragraph.
        if _p == "analysis":
            _NB_ORDER += ["verdict"]
        _NB_ORDER += [f"{_p}__why", _p]


    def _load_or_new_notebook():
        import nbformat

        from ....evaluation.notebook_exec import repair_code_cells
        nb_path = Path(node._study_dir) / "pipeline.ipynb"
        if nb_path.exists():
            try:
                nb = nbformat.read(str(nb_path), as_version=4)
                repair_code_cells(nb)
                return nb, nb_path
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


    # The standalone narrative (markdown-only) cells + their heading. "verdict"
    # is <deliverable_format>'s step 7 ('## Verdict & result', immediately
    # ahead of the analysis pillar) — without an entry here, AddPipelineMarkdownCell
    # rejected any name a strategizer guessed for it (e.g. "verdict_heading"),
    # a real, reproducible spec/tool mismatch (2 independent critic reviews,
    # run 20260820T003819) since the spec instructs this cell in exactly the
    # same literal-heading-text shape as "problem"/"hypotheses" but named
    # neither.
    _NARRATIVE = {
        "problem": "# Problem & objective",
        "hypotheses": "## Hypotheses",
        "verdict": "## Verdict & result",
    }

    def AddPipelineMarkdownCell(name: str, content: str) -> str:
        """CREATE one standalone narrative markdown cell in pipeline.ipynb.
        Three names are RESERVED and get a canonical heading added for you:
        'problem' (the question, min/max, success criterion), 'hypotheses'
        (the registered hypotheses + their falsifiable predictions — the
        Popperian setup), and 'verdict' ('## Verdict & result', immediately
        ahead of the analysis pillar). Any OTHER name is also allowed — a
        bespoke narrative section (a caveat, a background note, anything
        <deliverable_format> doesn't already name) — it is appended after the
        standard cells with your `content` used verbatim, no forced heading;
        the deliverable's structure must not block what you need to say. Only
        a pillar name or a `<pillar>__why` name is rejected (those belong to
        AddPipelineCell). CREATE-ONLY: if it already exists this errors —
        change it with EditPipelineCell(name, content=…) or remove it with
        DeletePipelineCell. Creates pipeline.ipynb if absent."""
        import nbformat
        prefix = node._drain_notifications()
        if node._study_dir is None:
            return "ERROR: study_dir not available."
        name = (name or "").strip()
        if not name:
            return "ERROR: `name` is required."
        if name in _PILLARS or (name.endswith("__why")
                                 and name[:-len("__why")] in _PILLARS):
            return (f"ERROR: {name!r} belongs to a pillar's code cell or its "
                    "WHY-explainer — use AddPipelineCell instead.")
        if not (content or "").strip():
            return f"ERROR: `content` is empty for {name!r}."
        nb, nb_path = _load_or_new_notebook()
        by = _by_name(nb)
        if name in by:
            return (prefix + f"ERROR: {name!r} already exists "
                    f"(rev {_rev(by[name].get('source', ''))}). "
                    "AddPipelineMarkdownCell is create-only — change it with "
                    "EditPipelineCell or remove it with DeletePipelineCell first.")
        _custom_name = name not in _NARRATIVE
        source = (
            content if _custom_name
            else _NARRATIVE[name] + "\n\n" + _strip_leading_md_header(content)
        )
        cell = nbformat.v4.new_markdown_cell(source)
        cell.metadata["name"] = name
        by[name] = cell
        _emit_notebook(by, nb, nb_path)
        if _custom_name:
            return (prefix + f"Added custom {name!r} markdown cell (rev "
                    f"{_rev(cell['source'])}) to pipeline.ipynb, after the "
                    "standard cells. Edit or remove it any time with "
                    "EditPipelineCell / DeletePipelineCell.")
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
        '<pillar>__why' explainer, or a narrative cell (problem, hypotheses,
        verdict, or any custom name created via AddPipelineMarkdownCell). Modes:
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
        # No static name-whitelist gate here: a custom cell (created via
        # AddPipelineMarkdownCell's free-form path, or AddPipelineCell's
        # custom-phase path) is a real cell in the notebook but isn't in the
        # canonical _NB_ORDER list — the check below (against the notebook's
        # ACTUAL contents) is what correctly distinguishes "doesn't exist yet"
        # from "exists, edit it" for both canonical and custom names alike.
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
                # markdown cell: problem / hypotheses / verdict / <pillar>__why /
                # a free-form custom name (AddPipelineMarkdownCell's custom path)
                if code is not None or why is not None:
                    return (f"ERROR: {name!r} is a markdown cell — use `content=`, "
                            "not `code=`/`why=`.")
                if not content.strip():
                    return f"ERROR: `content` is empty for {name!r}."
                if name in _NARRATIVE:
                    heading = _NARRATIVE[name]
                elif name.endswith("__why"):
                    heading = f"### {name[:-len('__why')]}"
                else:
                    heading = None  # free-form custom cell — no forced heading
                by[name]["source"] = (
                    heading + "\n\n" + _strip_leading_md_header(content)
                    if heading is not None else content
                )
        _emit_notebook(by, nb, nb_path)
        new_rev = _rev(by[name].get("source", ""))
        return prefix + f"Edited {name} in pipeline.ipynb (rev {cur_rev} → {new_rev})."


    def DeletePipelineCell(name: str, expected_rev: str = None) -> str:
        """Remove a cell from pipeline.ipynb. `name` is any named cell: a pillar
        (doe, data_generation, ml, optimization, analysis) — which also removes its
        '<pillar>__why' explainer — or a narrative cell (problem, hypotheses,
        verdict, or a custom name) or a '<pillar>__why'. Use it to drop a part you decided not to keep instead of
        leaving dead/placeholder content. REQUIRES `expected_rev` (the rev you last
        saw, from ShowNotebook or a prior Add/Edit) so you cannot delete a cell that
        changed since you last saw it."""
        import nbformat
        prefix = node._drain_notifications()
        if node._study_dir is None:
            return "ERROR: study_dir not available."
        name = (name or "").strip()
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
        to see what exists before editing. With `name` (problem, hypotheses, verdict, a custom cell, a
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
                _present = ([k for k in _NB_ORDER if k in by]
                            + [k for k in by if k not in _NB_ORDER])
                return (prefix + f"ERROR: no cell named {name!r}. Present: "
                        f"{_present}.")
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
        from ....evaluation.ledger_summary import ledger_breakdown
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
            from ....evaluation.notebook_exec import sandbox_env
            env = sandbox_env(sb_store, sb_cfg, study_root=node._study_dir)
            snippet = sandbox / "_scratch.py"
            snippet.write_text(code)
            try:
                from ....evaluation.notebook_exec import run_deliverable
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
        pass/fail lacks. With `name` (any named CODE cell — a standard pillar
        doe/data_generation/ml/optimization/analysis, OR a custom-phase cell
        you added via AddPipelineCell) it runs top-to-bottom UP TO AND
        INCLUDING that cell and reports it — cells share kernel state, so this
        pinpoints WHICH cell breaks reproduction and shows its exact traceback
        + stdout. With no name it runs the WHOLE notebook and reports every
        cell plus the first failure. Runs against a COPY (cannot touch the
        real ledger or pipeline.ipynb) and does NOT count toward the eval
        budget. Use it to localize a CheckDeliverable failure to one cell
        before editing, instead of re-running the binary gate blindly."""
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
            from ....evaluation.notebook_exec import sandbox_env
            env = sandbox_env(sb_store, sb_cfg, study_root=node._study_dir)
            try:
                from ....evaluation.notebook_exec import diagnose_notebook
                trace = diagnose_notebook(
                    nb_path, cwd=sandbox, env=env, timeout=180, upto_name=name)
            except _sub.TimeoutExpired:
                return (prefix + "Notebook diagnosis exceeded 180s and was killed "
                        "— a cell is running a real campaign; it should load the "
                        "ledger lazily, not recompute.")
            if trace.get("missing_name"):
                return (prefix + f"No CODE cell named {name!r} — this can be a "
                        "standard pillar (doe, data_generation, ml, "
                        "optimization, analysis) or a custom-phase cell you "
                        "added via AddPipelineCell, but it must be a code "
                        "cell (a markdown-only cell like 'problem' or "
                        "'verdict' has no execution state to run up to). "
                        "Use ShowNotebook() to see the cells.")
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


    return {
        "WriteDeliverable": WriteDeliverable,
        "CheckDeliverable": CheckDeliverable,
        "AddPipelineMarkdownCell": AddPipelineMarkdownCell,
        "AddPipelineCell": AddPipelineCell,
        "EditPipelineCell": EditPipelineCell,
        "DeletePipelineCell": DeletePipelineCell,
        "ShowNotebook": ShowNotebook,
        "LedgerBreakdown": LedgerBreakdown,
        "RunScratch": RunScratch,
        "RunPipelineCell": RunPipelineCell,
    }
