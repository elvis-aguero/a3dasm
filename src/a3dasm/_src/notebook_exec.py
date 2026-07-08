"""Execute the run deliverable — `pipeline.ipynb` — under the reproduction gate.

The system is committed to the notebook: `pipeline.ipynb` is THE deliverable.
The gate's contract (exit-clean + ZERO new oracle evals + the `REPRODUCED:`
headline grounded in the ledger) executes it lazily via nbclient. The nbclient/
nbformat/ipykernel deps ship with the `agentic` extra; `run_deliverable()`
returns a `subprocess.CompletedProcess`-shaped result and raises
`subprocess.TimeoutExpired` on timeout, so the gate branches on nothing.

Agents author the notebook cell-by-cell via the structured AddPipelineMarkdownCell
/ AddPipelineCell / EditPipelineCell / DeletePipelineCell / ShowNotebook closures
(pure nbformat, name-addressed) — there is no live kernel.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

__all__ = [
    "notebook_available",
    "required_deliverable_name",
    "run_deliverable",
    "build_notebook",
    "notebook_deliverable_spec",
    "sandbox_env",
    "stamp_run_provenance",
]

# Stable name for the run-provenance metadata cell (see stamp_run_provenance).
RUN_PROVENANCE_CELL = "_run_provenance"


def stamp_run_provenance(nb, meta_md: str):
    """Stamp the run-provenance metadata cell onto the deliverable notebook,
    REPLACING any prior one instead of appending.

    pipeline.ipynb is study-scoped and persists across runs, so appending an
    UNNAMED stamp cell each close accumulated stale metadata — a run's
    deliverable carried the PREVIOUS run's run_dir / evals_used (run
    20260630T164908 shipped 20260629's stamp). The cell is given a stable
    ``metadata["name"]`` so (a) the next close replaces it rather than piling
    up, and (b) it is visible/removable via ShowNotebook / DeletePipelineCell
    like every other named cell.
    """
    import nbformat
    nb.cells = [
        c for c in nb.cells
        if (getattr(c, "metadata", None) or {}).get("name")
        != RUN_PROVENANCE_CELL
    ]
    cell = nbformat.v4.new_markdown_cell(meta_md)
    cell.metadata["name"] = RUN_PROVENANCE_CELL
    nb.cells.append(cell)
    return nb


def sandbox_env(
    sb_store,
    sb_run_config,
    study_root=None,
    delegation_id: str = "D999",
    base=None,
) -> dict:
    """Build the environment for running a deliverable / cell / scratch snippet
    against a SANDBOX copy of the ledger.

    F3DASM_CANONICAL_STORE / F3DASM_RUN_CONFIG point at the sandbox copy so no
    execution can touch the real store. F3DASM_STUDY_ROOT is a READ-ONLY anchor
    to the real study repo, so pipeline cells can locate non-ledger resources
    (e.g. ``bo/cei_core.py`` for a surrogate self-check) deterministically
    instead of hand-rolling multi-candidate path searches relative to the
    store. Isolation is unaffected: only the store is a copy; the study root is
    read-only reference code.
    """
    env = dict(os.environ if base is None else base)
    env["F3DASM_CANONICAL_STORE"] = str(sb_store)
    env["F3DASM_RUN_CONFIG"] = str(sb_run_config)
    if study_root is not None:
        env["F3DASM_STUDY_ROOT"] = str(study_root)
    env.setdefault("F3DASM_DELEGATION_ID", str(delegation_id))
    return env


# The canonical notebook structure — ONE source, referenced by the agent prompt
# (notebook_deliverable_spec), KB 0009, and any fallback builder. The spine is the
# Popperian loop (hypotheses → falsification attempt → verdict); the body mirrors
# f3dasm's four pillars (Phase enum: doe / data_generation / ml / optimization),
# each a name-tagged code cell preceded by a WHY explainer markdown cell.
def notebook_deliverable_spec(role: str = "strategizer") -> str:
    """The deliverable contract — injected into the relevant agent prompts so
    the .ipynb is a reproducible scientific narrative, not a ported .py.

    Role-aware: ONLY the strategizer authors the notebook (it alone is granted
    AddPipelineMarkdownCell / AddPipelineCell), so only it gets the "author with
    these tools" imperative. The implementer (writes phase code to its workspace) and
    the critic (judges the notebook) get the same STRUCTURE + RULES so their
    work fits / is judged against it — but no instruction to call tools they do
    not have."""
    if role == "strategizer":
        intro = (
            "DELIVERABLE = pipeline.ipynb. This SUPERSEDES every 'pipeline.py' /\n"
            "WriteDeliverable('pipeline.py') / solution.md instruction above —\n"
            "for THIS run there is exactly one deliverable, pipeline.ipynb, and\n"
            "no pipeline.py and no solution.md. It is the single merged artifact\n"
            "— the writeup AND the runnable, lazily-reproducible recipe in one. A\n"
            "SCIENTIFIC NARRATIVE, not a dumped script. AUTHOR IT WITH THE\n"
            "STRUCTURED TOOLS — they make the structure unforgeable plumbing:\n"
            "  - AddPipelineMarkdownCell(name, content): CREATE a narrative cell;\n"
            "    name is 'problem' or 'hypotheses' (the canonical heading is added).\n"
            "    Call early. Create-only — change it later with EditPipelineCell.\n"
            "  - AddPipelineCell(phase, why, code): CREATE one pillar cell + its\n"
            "    REQUIRED WHY-explainer; phase in {doe, data_generation, ml,\n"
            "    optimization, analysis}. Cells stay in canonical order. The pillar\n"
            "    name and rationale are required, so you cannot ship a structureless\n"
            "    notebook or omit the WHY. Create-only. Do NOT hand-write notebook JSON.\n"
            "  - ShowNotebook(name?): no arg → list every cell by NAME with its rev;\n"
            "    with a name → that cell's full source + rev. READ before you edit.\n"
            "  - EditPipelineCell(name, ...): change ANY existing cell (pillar,\n"
            "    <pillar>__why, problem, or hypotheses). Surgical old/new find-replace\n"
            "    (self-guarding), or full-field — code=/why= for a pillar, content= for\n"
            "    a markdown cell — which REQUIRES expected_rev (the rev from\n"
            "    ShowNotebook), so you can't clobber a cell that changed since you saw it.\n"
            "  - DeletePipelineCell(name, expected_rev): drop a cell you decided not to\n"
            "    keep (a pillar also drops its __why). Don't leave dead/placeholder content.\n"
            "  - RunScratch(code): run a snippet against a COPY of the ledger and\n"
            "    see its output — INSPECT before you commit. Verify the ledger\n"
            "    loads, a path resolves, a value is what you think, the analysis\n"
            "    cell would populate — rather than discovering a silent bug only\n"
            "    when the critic reads it. Free (no eval-budget cost, no mutation).\n"
            "    (WriteDeliverable('pipeline.ipynb', …) is a raw fallback only.)\n\n"
        )
    elif role == "implementer":
        intro = (
            "DELIVERABLE CONTEXT: the run's single deliverable is pipeline.ipynb,\n"
            "which the STRATEGIZER assembles from your work (you do NOT author it\n"
            "— you have no notebook tools). Write your phase code so it drops\n"
            "cleanly into one of the pillar cells below; return runnable,\n"
            "non-stub code that reaches the oracle ONLY via get_evaluator().\n\n"
        )
    else:  # critic / reviewer
        intro = (
            "DELIVERABLE CONTEXT: the run's single deliverable is pipeline.ipynb\n"
            "(no pipeline.py, no solution.md — the notebook's markdown IS the\n"
            "writeup). You JUDGE it against the structure below; you do not author\n"
            "it.\n\n"
        )
    return (
        "\n<deliverable_format>\n"
        + intro +
        "STRUCTURE (the Popperian spine + f3dasm's four pillars). Each code cell\n"
        "carries name metadata = its pillar so the structure is machine-checkable:\n"
        "  1. md  '# Problem & objective'  — question, min/max, success criterion.\n"
        "  2. md  '## Hypotheses'          — registered hypotheses + falsifiable\n"
        "         predictions (the Popperian setup; mirror the hypothesis ledger).\n"
        "  3. md WHY-explainer + code name='doe'             — Domain + sampler\n"
        "         (LOAD-OR-CREATE: load the ledger if present, else build the DoE).\n"
        "  4. md WHY-explainer + code name='data_generation' — LAZY eval pattern:\n"
        "         canonical_store = os.environ['F3DASM_CANONICAL_STORE']\n"
        "         try:\n"
        "             from a3dasm import get_evaluator\n"
        "             gen = get_evaluator()   # raises ValueError if oracle not registered\n"
        "             gen.call(data, mode='sequential')  # skips FINISHED rows → 0 new\n"
        "         except ValueError:\n"
        "             data = ExperimentData.from_file(project_dir=canonical_store)\n"
        "             assert len(data) > 0, 'Canonical store empty and no oracle registered'\n"
        "         MULTI-EXPERIMENT LOADING: a run may hold more than one experiment\n"
        "         (a baseline + design parametrizations), each its OWN ExperimentData\n"
        "         store at a nested path — a single from_file() loads ONLY the default\n"
        "         and silently misses the rest, and there is NO namespace column to\n"
        "         split a single store on. Load them ALL with the one canonical call:\n"
        "             from a3dasm import load_experiments\n"
        "             experiments = load_experiments()  # {'default': ExperimentData, 'polar': ...}\n"
        "         then iterate experiments.items() per experiment. Works for a single-\n"
        "         experiment run too (returns {'default': ...}).\n"
        "         (Never use f3dasm._src.* paths — internal, unversioned, not public.)\n"
        "         @datagenerator SCALAR RULE: if you use @datagenerator(output_names=['y'])\n"
        "         directly, return the scalar value itself — NOT a dict. 'return val' not\n"
        "         'return {\"y\": val}'. The output_names mapping already handles the name.\n"
        "  5. md WHY-explainer + code name='ml'              — fit the surrogate.\n"
        "  6. md WHY-explainer + code name='optimization'    — acquisition / BO loop.\n"
        "  7. md '## Verdict & result' + code name='analysis' — for each hypothesis\n"
        "         state SUPPORTED/FALSIFIED + WHY from the evidence; derive the\n"
        "         headline value FROM the ledger — never hardcode the result —\n"
        "         and print exactly 'REPRODUCED: <value>'. Also print exactly\n"
        "         'CLAIMED_HEADLINE: <value>' = the SAME number your write-up\n"
        "         states as the answer, so the gate can verify your prose matches\n"
        "         your computation (they must agree; a mismatch — e.g. an idxmax\n"
        "         selecting a noise row — is REJECTED). 'Derive from the ledger'\n"
        "         means COMPUTE it in code from the loaded store (this is still\n"
        "         LAZY — loading + computing adds zero new oracle rows); it does\n"
        "         NOT mean pasting a number you remember. For per-experiment /\n"
        "         per-delegation eval COUNTS, call LedgerBreakdown() and quote\n"
        "         those — counts in a plan or a delegation's notes drift from what\n"
        "         actually landed in the ledger.\n\n"
        "RULES:\n"
        "- NOTEBOOK-LEDGER SYNC: the hypotheses cell (## Hypotheses) and the analysis\n"
        "  cell MUST reflect the CURRENT status of every hypothesis in hypotheses.json.\n"
        "  When you call HypothesisUpdate (e.g. SUPPORTED → INCONCLUSIVE), you MUST\n"
        "  immediately update BOTH to match, BEFORE CheckDeliverable, with\n"
        "  EditPipelineCell: the hypotheses cell — EditPipelineCell('hypotheses',\n"
        "  content=…) — and the analysis cell — EditPipelineCell('analysis', code=…).\n"
        "  ShowNotebook('<name>') first for each cell's current rev (full-field edits\n"
        "  require expected_rev; AddPipelineCell/AddPipelineMarkdownCell are create-only).\n"
        "  A notebook that shows a stale status will be REJECTED. Ledger and notebook\n"
        "  must agree on every hypothesis status at gate time.\n"
        "- STORE PATH = PORTABILITY. The reproduction must not depend on the machine\n"
        "  that authored it, so read the canonical store path ONLY from the injected\n"
        "  env var, never from a fallback or default that bakes in a local path:\n"
        "  `canonical_store = os.environ['F3DASM_CANONICAL_STORE']` (bracket access —\n"
        "  raises if missing). Read it at the TOP of every cell that needs it; the gate\n"
        "  re-executes cells independently, so do not rely on a variable from a prior cell.\n"
        "- NON-LEDGER REPO RESOURCES: to locate study code that is NOT in the ledger\n"
        "  (e.g. a surrogate helper like `bo/cei_core.py`), anchor paths to\n"
        "  `os.environ['F3DASM_STUDY_ROOT']` (the study repo root) — do NOT derive them\n"
        "  from the store path, which points at a temp sandbox copy unrelated to the repo.\n"
        "- OBJECTIVE COLUMN NAME: name your study's objective column EXPLICITLY —\n"
        "  it is a fixed property of the study (e.g. `output_col = 'lambda_cr_nd'`).\n"
        "  Do NOT auto-detect it by output order: `data.domain.output_names` is\n"
        "  SORTED, so with MULTIPLE outputs a non-objective column (e.g. a binary\n"
        "  constraint flag like 'coilable') can sort BEFORE the objective and get\n"
        "  silently picked — the optimization then targets the wrong column and\n"
        "  reports a bogus best. If you must derive it programmatically, read the\n"
        "  REGISTERED objective from run_config['evaluator_output_names'][0], not\n"
        "  the first non-provenance column.\n"
        "- The four pillar cells (doe/data_generation/ml/optimization) are ALWAYS\n"
        "  present. A pillar you did NOT run stays present but its explainer says\n"
        "  plainly 'NOT executed (budget)'. Never silently drop a pillar.\n"
        "- Every WHY-explainer justifies the methodological choice (cite the\n"
        "  literature you gathered) — this is the rationale, not just description.\n"
        "- LAZY + reproducible: the runtime executes the notebook against the\n"
        "  shipped ledger and requires ZERO new oracle evals. Reach the oracle\n"
        "  ONLY via get_evaluator(). Print the 'REPRODUCED: <value>' headline\n"
        "  derived from the ledger — it is an informational marker the critic\n"
        "  checks for provenance (it must trace to a real ledger row); the\n"
        "  runtime no longer machine-matches it, so a constrained optimum is a\n"
        "  valid headline even though it is not an objective extremum.\n"
        "  LAZY is a cell-level invariant: a cell satisfies LAZY only if executing\n"
        "  it against the shipped ledger produces zero new oracle rows. A guard that\n"
        "  is reached only after computation has already begun does not satisfy this\n"
        "  invariant — the cell is not lazy, it merely crashes late.\n"
        "- CACHE-OR-LOAD HEAVY BLOCKS. Row-laziness covers ONLY oracle evals. A\n"
        "  fitted surrogate or costly analysis YOU build must persist and\n"
        "  load-if-present (cache-or-load), never refit on a re-run — else a re-run\n"
        "  recomputes for minutes/hours though it adds zero oracle evals.\n"
        "- NEVER call data.store() after evaluator.call(). The InstrumentedDataGenerator\n"
        "  behind get_evaluator() already writes FINISHED rows to the canonical store.\n"
        "  Calling data.store() afterwards overwrites those FINISHED rows with\n"
        "  IN_PROGRESS — silently corrupting the ledger. Reload if you need the\n"
        "  updated outputs: ExperimentData.from_file(project_dir=canonical_store).\n"
        "- GUARD OPTIONAL IMPORTS = PORTABILITY. Heavy packages (torch, botorch,\n"
        "  gpytorch, jax) are NOT guaranteed installed; an unguarded `import torch`\n"
        "  breaks the notebook on any machine without it (again: reproduction must not\n"
        "  assume the authoring environment). Guard them: `import importlib.util; if\n"
        "  importlib.util.find_spec(\"torch\") is None: # use sklearn/scipy fallback`.\n"
        "- WRITEUP STANDARD — the notebook is the companion to a paper, not a code\n"
        "  dump. Write for a SKEPTICAL reader who accepts the result only if every\n"
        "  claim is earned; someone who reads ONLY this notebook must come away\n"
        "  agreeing WHAT the solution is AND WHY. Concretely:\n"
        "    * CALIBRATE CONFIDENCE TO EVIDENCE. State a conclusion no more strongly\n"
        "      than its test and its hypothesis posterior support. A hypothesis\n"
        "      closed at a modest posterior is 'supported, with residual uncertainty',\n"
        "      not a settled fact — the prose must carry that uncertainty. Reserve\n"
        "      flat declarative claims ('X IS the global minimum') for what the\n"
        "      evidence decisively shows; otherwise write 'best found', 'strong but\n"
        "      not conclusive evidence that…'.\n"
        "    * NO FAITH GAPS. Every quantitative claim (a 'best', a threshold, a\n"
        "      'global'/'unique') traces to a ledger value or a cited delegation\n"
        "      result. If a claim rests on COVERAGE/sampling ('no better region\n"
        "      exists'), say so and state the residual risk — a sampling sweep is\n"
        "      evidence, not a proof; a deceptive landscape can hide a narrow basin\n"
        "      the sweep missed. Do not present a sampling argument as a proof.\n"
        "    * ONE PRINCIPLED THRESHOLD per claim. Any cutoff used to judge a\n"
        "      hypothesis is stated once, justified, and used identically in every\n"
        "      cell — never a different number for the same claim across cells.\n"
        "    * NO FILLER. Every sentence is rationale, evidence, or a stated\n"
        "      limitation; cut restated boilerplate.\n"
        "</deliverable_format>\n"
    )


def notebook_available() -> bool:
    """True iff nbclient + nbformat are importable (ship with the agentic extra)."""
    try:
        import nbclient  # noqa: F401
        import nbformat  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def required_deliverable_name() -> str:
    """The single run deliverable. The system is committed to the notebook —
    pipeline.ipynb is THE deliverable (there is no pipeline.py / solution.md)."""
    return "pipeline.ipynb"


@contextlib.contextmanager
def _patched_environ(env: dict | None):
    """Temporarily replace os.environ with `env` so a freshly-spawned Jupyter
    kernel (which inherits os.environ at launch) sees the gate's injected vars
    (F3DASM_CANONICAL_STORE / F3DASM_RUN_CONFIG / F3DASM_STUDY_ROOT /
    F3DASM_DELEGATION_ID)."""
    if env is None:
        yield
        return
    saved = dict(os.environ)
    try:
        os.environ.clear()
        os.environ.update(env)
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _execute_notebook(path: Path, cwd: Path, env: dict, timeout: float):
    """Execute a notebook in THIS interpreter's env (so f3dasm/get_evaluator
    import) and return a CompletedProcess-shaped result. Raises
    subprocess.TimeoutExpired on timeout (mirrors the .py path)."""
    import nbformat
    from jupyter_client.manager import KernelManager
    from nbclient import NotebookClient
    from nbclient.exceptions import CellTimeoutError, DeadKernelError

    nb = nbformat.read(str(path), as_version=4)
    # Pin the kernel to THIS interpreter so the notebook runs in the env that
    # has f3dasm installed (the global "python3" kernelspec may point elsewhere).
    km = KernelManager(kernel_name="python3")
    try:
        km.kernel_spec.argv[0] = sys.executable
    except Exception:  # noqa: BLE001 — fall back to the spec's python
        pass
    client = NotebookClient(
        nb, km=km, timeout=int(timeout),
        allow_errors=True,                      # capture errors, don't raise
        resources={"metadata": {"path": str(cwd)}},
    )
    with _patched_environ(env):
        try:
            client.execute()
        except (CellTimeoutError, DeadKernelError) as exc:
            raise subprocess.TimeoutExpired(cmd=str(path), timeout=timeout) from exc

    out_parts, err_parts, errored = [], [], False
    for cell in nb.cells:
        for o in cell.get("outputs", []):
            ot = o.get("output_type")
            if ot == "stream":
                (out_parts if o.get("name") == "stdout" else err_parts).append(
                    o.get("text", ""))
            elif ot == "error":
                errored = True
                err_parts.append(
                    f"{o.get('ename', '')}: {o.get('evalue', '')}\n"
                    + "\n".join(o.get("traceback", [])))
    return subprocess.CompletedProcess(
        args=[str(path)],
        returncode=1 if errored else 0,
        stdout="".join(out_parts),
        stderr="".join(err_parts),
    )


def run_deliverable(path: Path, *, cwd: Path, env: dict, timeout: float):
    """Run the deliverable; return a subprocess.CompletedProcess. `.ipynb` →
    nbclient (in-env kernel); anything else → `python <file>` subprocess.
    Raises subprocess.TimeoutExpired on timeout in BOTH paths."""
    path = Path(path)
    if path.suffix == ".ipynb" and notebook_available():
        return _execute_notebook(path, Path(cwd), env, timeout)
    return subprocess.run(
        [sys.executable, str(path)],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout,
    )


def diagnose_notebook(path: Path, *, cwd: Path, env: dict, timeout: float,
                      upto_name: str | None = None) -> dict:
    """Execute a notebook and return a PER-CELL execution trace — the granular
    diagnostic the binary ``run_deliverable`` gate aggregates away (#13).

    Mirrors ``_execute_notebook`` exactly (same interpreter-pinned kernel + env
    patch + ``allow_errors=True``) so it reproduces the gate's environment, but
    keeps each cell's outputs instead of merging them. With ``upto_name`` it runs
    only the cells up to AND INCLUDING the code cell carrying that
    ``metadata.name`` — cells share kernel state top-to-bottom, so this localizes
    WHERE reproduction breaks without spuriously failing on missing prior state.

    Returns ``{"cells": [{"index","name","cell_type","errored","stdout",
    "error"}], "first_error": <that cell|None>, "executed": int,
    "truncated": bool, "timed_out": bool, "missing_name": bool}``.
    """
    import re as _re

    import nbformat
    from jupyter_client.manager import KernelManager
    from nbclient import NotebookClient
    from nbclient.exceptions import CellTimeoutError, DeadKernelError

    _ansi = _re.compile(r"\x1b\[[0-9;]*m")  # strip terminal colour codes from tracebacks

    nb = nbformat.read(str(path), as_version=4)
    truncated = False
    if upto_name is not None:
        idx = next(
            (i for i, c in enumerate(nb.cells)
             if c.get("cell_type") == "code"
             and (c.get("metadata", {}) or {}).get("name") == upto_name),
            None,
        )
        if idx is None:
            return {"cells": [], "first_error": None, "executed": 0,
                    "truncated": False, "timed_out": False, "missing_name": True}
        nb.cells = nb.cells[:idx + 1]
        truncated = True

    km = KernelManager(kernel_name="python3")
    try:
        km.kernel_spec.argv[0] = sys.executable
    except Exception:  # noqa: BLE001 — fall back to the spec's python
        pass
    client = NotebookClient(
        nb, km=km, timeout=int(timeout), allow_errors=True,
        resources={"metadata": {"path": str(cwd)}},
    )
    timed_out = False
    with _patched_environ(env):
        try:
            client.execute()
        except (CellTimeoutError, DeadKernelError):
            timed_out = True

    cells, first_error = [], None
    for i, cell in enumerate(nb.cells):
        out_parts, err_parts, errored = [], [], False
        for o in cell.get("outputs", []):
            ot = o.get("output_type")
            if ot == "stream":
                (out_parts if o.get("name") == "stdout" else err_parts).append(
                    o.get("text", ""))
            elif ot == "error":
                errored = True
                err_parts.append(_ansi.sub(
                    "",
                    f"{o.get('ename', '')}: {o.get('evalue', '')}\n"
                    + "\n".join(o.get("traceback", []))))
        rec = {
            "index": i,
            "name": (cell.get("metadata", {}) or {}).get("name"),
            "cell_type": cell.get("cell_type"),
            "errored": errored,
            "stdout": "".join(out_parts),
            "error": "".join(err_parts),
        }
        cells.append(rec)
        if errored and first_error is None:
            first_error = rec
    return {"cells": cells, "first_error": first_error, "executed": len(cells),
            "truncated": truncated, "timed_out": timed_out, "missing_name": False}


def build_notebook(cells: list[dict]):
    """Assemble a notebook from a simple cell list (avoids hand-written JSON).

    Each cell: {"type": "markdown"|"code", "source": str, "name": str|None}.
    The optional ``name`` is written to cell.metadata.name AND cell.metadata.tags
    (mirrors the f3dasm Phase: doe/data_generation/ml/optimization/analysis), so
    phase-presence is machine-checkable. Returns an nbformat NotebookNode.
    """
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                 "language": "python"}
    built = []
    for c in cells:
        src = c.get("source", "")
        if c.get("type") == "markdown":
            cell = nbformat.v4.new_markdown_cell(src)
        else:
            cell = nbformat.v4.new_code_cell(src)
        name = c.get("name")
        if name:
            cell.metadata["name"] = name
            cell.metadata["tags"] = [name]
        built.append(cell)
    nb.cells = built
    return nb
