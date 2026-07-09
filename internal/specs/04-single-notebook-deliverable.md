# Spec 04 — Single Jupyter-notebook deliverable

**Backlog #4.** Priority: low (polish). **Status: SUPERSEDED — IMPLEMENTED.**

> ⚠️ **This spec is a historical design record, not a description of the live
> system.** It is written in the future tense and lists open questions that have
> since been settled. What actually shipped:
> - `pipeline.ipynb` is the **sole, always-required** deliverable — NOT the
>   opt-in extra this spec recommended. There is no `solution.md` and no
>   `pipeline.py`; both are gone.
> - The reproduction gate executes the notebook lazily via **nbclient**
>   (`notebook_exec.run_deliverable` / `_reproduction_gate`) with the same
>   zero-new-evals + `REPRODUCED:` asserts.
> - The **agent authors the cells** through structured CRUD tools
>   (`AddPipelineCell` / `AddPipelineMarkdownCell` / `EditPipelineCell` /
>   `DeletePipelineCell` / `ShowNotebook`); the prose lives in the notebook's
>   leading markdown cells. The runtime no longer writes `solution.md`.
> - Open questions below are resolved: the **agent** authors the prose cells;
>   authoring is via the structured tools (not raw JSON); the notebook is
>   required for all agentic runs (not opt-in).
>
> For the live contract see [`../authoring-a-study.md`](../authoring-a-study.md),
> `notebook_exec.py`, and `tests/agentic/test_study_contract.py`. The line
> numbers and "current mechanics" below describe the pre-migration code and are
> stale. The original write-up is kept for the record.

Merge the two deliverables — `solution.md` (prose) + `pipeline.py` (executable) —
into **one `.ipynb`**: markdown cells for the writeup, code cells for the lazy
create→run→analyze pipeline. One human-readable, runnable artifact that is also
the reproduction.

## Problem (evidence)
Today a run emits **two** artifacts with split authorship:
- `pipeline.py` — agent-authored in-graph via `WriteDeliverable`
  (`nodes/tools/routing.py:1841-1863`), required by `_missing_deliverables`
  (`nodes/strategizer.py:604`: `required = ["pipeline.py"] + …`), executed by the
  reproduction gate.
- `solution.md` — runtime-authored **post-graph** from the `Done()` summary
  (`agent_runtime.py:610-653`, `solution_path = self.study_dir / "solution.md"`),
  with no agent tool to write it.

The prose and the code drift apart, and the reader holds two files. A notebook
unifies them and keeps the reproduction property.

## Current mechanics (primary evidence)
- `WriteDeliverable` accepts **only** `{".py", ".md"}` (`routing.py:1851`),
  rejects path separators (`:1857`), writes the **bare name** to `study_dir/`
  (`:1861`), verbatim (`:1862`).
- Required list (`strategizer.py:604`): `pipeline.py` hardcoded + state-injected
  `required_deliverables`; existence-checked at `:613`.
- Reproduction gate (`strategizer.py:617-703`, called from `Done()` at
  `routing.py:1345`): runs `subprocess.run([sys.executable, str(pipeline_py)], …,
  timeout=_timeout)` (`:677-681`); PASS iff exit 0 (`:690`) **and** zero new
  ledger rows (`:696`); injects `F3DASM_CANONICAL_STORE` (`:665`).
- `solution.md` authored at `agent_runtime.py:623-651` from `report`
  (= strategizer's final `last_report`) + a metadata table.
- **Tooling gap (evidence):** `nbclient`, `nbformat`, `nbconvert`, `jupyter`,
  `papermill` are **NOT** dependencies (`pyproject.toml` has only
  `mkdocs-ipynb` for docs). All would be new deps.

## Design (DRY — change the *mechanism*, keep the *contract*)
The lazy + zero-new-eval contract is sound and must NOT be duplicated — only the
execution mechanism changes from "run a .py" to "execute a notebook".

1. **Dependency:** add `nbclient>=0.7` + `nbformat>=5.9` to a **new
   `agentic-notebook` extra** (not core; orthogonal to f3dasm). Gate the feature
   on import availability so a non-notebook install still runs `pipeline.py`.
2. **`WriteDeliverable` accepts `.ipynb`** (`routing.py:1851` add to
   `allowed_exts`). Content is the notebook JSON the agent authors (or a small
   `nbformat` helper builds from a cells list — TBD, see open question).
3. **Required deliverable becomes notebook-aware** (`strategizer.py:604`): require
   `pipeline.ipynb` when the notebook extra is active, else `pipeline.py`. DRY:
   one resolver `_required_deliverable_name()` returns the active form; both the
   missing-check and the gate read it.
4. **Reproduction gate executes the notebook** (`strategizer.py:677-681`): when
   the deliverable is `.ipynb`, run it via `nbclient.NotebookClient` (or
   `jupyter nbconvert --execute --inplace`) with the **same** env injection
   (`F3DASM_CANONICAL_STORE`, `F3DASM_RUN_CONFIG`, `F3DASM_DELEGATION_ID`), and
   apply the **identical** asserts (exit clean + `after == before` rows). Factor
   the "run deliverable in subprocess, count rows before/after" body into one
   helper that branches only on the executor — the assert logic is shared.
   **Verify** the env vars propagate into the notebook kernel (kernel envs are a
   known footgun).
5. **solution.md prose → leading markdown cells.** Decide authorship (open
   question): either the runtime injects the `Done()` summary as the notebook's
   first markdown cell post-graph (mirrors today), or the agent authors the prose
   cells and the runtime stops writing `solution.md`. Keep ONE source of the
   headline.

## TDD plan (tests first)
1. `test_required_deliverable_is_notebook_when_extra_active` — `_missing_deliverables`
   asks for `pipeline.ipynb`, not `.py`, when notebook mode is on; `.py` otherwise.
2. `test_write_deliverable_accepts_ipynb` — `.ipynb` allowed; bad JSON rejected.
3. `test_repro_gate_executes_notebook_lazily` — a notebook that loads the ledger
   and asserts the headline → gate PASS, zero new rows (mirror
   `test_reproduction_gate.py::test_gate_passes_for_clean_lazy_pipeline` with a
   tiny notebook fixture).
4. `test_repro_gate_fails_when_notebook_adds_evals` — a notebook that re-evaluates
   → FAIL "not lazy" (mirror the existing `.py` test).
5. `test_repro_gate_env_vars_reach_kernel` — assert `F3DASM_CANONICAL_STORE`/
   `F3DASM_RUN_CONFIG` are visible inside the executed notebook.
6. `test_shared_assert_helper_is_executor_agnostic` — the row-count/exit assert
   helper gives the same verdict for an equivalent `.py` and `.ipynb` (DRY guard).

## Risks / out of scope
- **Risk:** kernel env propagation + nbclient timeout semantics differ from
  `subprocess.run`; pin behavior with test 5.
- **Risk:** notebook JSON authoring by the agent is error-prone → prefer an
  `nbformat` helper that assembles cells from a simple structure.
- **Out of scope:** rich notebook outputs/plots in the gate (only exit + zero
  evals are binding); making the notebook the live campaign spine (that's the
  deferred "Phase B" of the pipeline redesign).

## Done when
A run emits a single `pipeline.ipynb`; the gate executes it lazily (exit clean +
zero new evals) with env vars reaching the kernel; the prose lives in its leading
markdown cells; tests 1-6 pass; non-notebook installs still work on `pipeline.py`.

## Open questions
- Who authors the prose cells (runtime injection vs agent)?
- `WriteDeliverable` agent-authored JSON vs an `nbformat` cell-list helper?
- Required for all agentic installs, or an opt-in extra (recommended: opt-in)?
