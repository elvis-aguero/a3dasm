# Porting a3dasm to its own repository

This directory holds the standalone-repo scaffolding for a3dasm, adapted from
f3dasm's own files so the new repo matches f3dasm's bar (uv + src-layout, Ruff,
pytest with markers and an 85% coverage gate, MkDocs Material + mkdocstrings,
Read the Docs, PR-matrix CI). It is inert inside this fork. When the private
`bessagroup/a3dasm` repo exists, assemble the new repo as below.

## 1. Extract the source and history (git filter-repo)

Run on a fresh CLONE (filter-repo rewrites history and is irreversible). Keep the
original clone as a backup until the result is validated. This keeps the agentic
package, its public shim, tests, docs, and the agentic studies, and renames them
into the a3dasm layout while preserving blame:

```bash
git clone <this-fork> a3dasm && cd a3dasm
git filter-repo \
  --path src/f3dasm/_src/agentic/ \
  --path src/f3dasm/agentic/ \
  --path tests/agentic/ \
  --path docs/agentic/ \
  --path CLAUDE.md \
  --path README-agentic.md \
  --path-rename src/f3dasm/_src/agentic/:src/a3dasm/_src/ \
  --path-rename src/f3dasm/agentic/:src/a3dasm/ \
  --path-rename tests/agentic/:tests/ \
  --path-rename docs/agentic/:docs/
```

Studies are not renamed above because `studies/` mixes agentic and non-agentic
work. Add explicit `--path studies/agentic_black_box_3d/` (and the other
`agentic_*` dirs plus `studies/run_ledger.py`, `studies/audit_run.py`,
`studies/_oracle_src/`) if you want them to travel. Do a `--dry-run` first and
review the file list.

## 2. Drop in the scaffolding

Copy everything from `packaging/a3dasm/` to the new repo root:

```bash
cp -r packaging/a3dasm/. .
```

Then add the meta files not staged here: `LICENSE` (BSD-3-Clause, bessagroup),
`.gitignore`, and optionally `CITATION.cff`, `CODE_OF_CONDUCT.md`,
`CONTRIBUTING.md`, `.github/ISSUE_TEMPLATE/` (copy f3dasm's and adjust names).

## 3. Rename the package references (careful sweep, not a blind sed)

- `f3dasm._src.agentic` -> `a3dasm._src`
- `f3dasm.agentic`      -> `a3dasm`
- Fix relative-import depths shifted by the `_src/agentic/` -> `_src/` move.

Do NOT rename f3dasm references inside agent prompt strings that instruct
generated code (for example `from f3dasm import ExperimentData`,
`from f3dasm.design import Domain`, `create_sampler(...)`). Those target the real
f3dasm dependency and must stay `f3dasm`. Review each hit.

## 4. Remaining work to hit the full bar

- [ ] Re-home the two f3dasm-core behaviours a3dasm still relies on, so a3dasm
      works against STOCK upstream f3dasm (this fork's vendored copy has them, a
      released f3dasm does not):
      - `to_numpy` dropping underscore-prefixed provenance columns. Handle it in
        a3dasm where it reads arrays (select non-underscore columns), or wrap.
      - the protected-store guard (refuse a shrinking write to the canonical
        ledger). Wrap `ExperimentData.store` at a3dasm import time, or guard at
        a3dasm's write boundary. `PROTECTED_STORE_SENTINEL` is already
        a3dasm-owned in agent_runtime.
- [ ] After `bessagroup/f3dasm#351` merges and is released: bump the `f3dasm`
      pin in `pyproject.toml`, and flip the interim
      `from f3dasm._src.errors import EmptyFileError, ReachMaximumTriesError`
      back to `from f3dasm import ...` (3 sites, all marked with a TODO).
      `JobStatus` stays on `f3dasm._src` unless it is also promoted.
- [ ] numpydoc pass over public classes and functions in `src/a3dasm/_src/`
      (module narratives are already strong; class/function docstrings need the
      typed Parameters/Returns/Raises sections so mkdocstrings renders cleanly).
- [ ] Author `docs/notebooks/quickstart.ipynb` (a small end-to-end AgenticRun,
      distinct from the internal `studies/*`) and flesh `docs/concepts.md`.
- [ ] Confirm the ported `FEATURES.md`, `BACKLOG.md`, and `authoring-a-study.md`
      render in the docs nav (already referenced in `mkdocs.yml`).
- [ ] Update `CLAUDE.md` paths for the standalone layout (it currently assumes
      the monorepo `studies/` and `README-agentic.md` locations).

## 5. Validate

```bash
uv sync --extra all --extra tests
uv run pytest -m "not integration and not ollama"   # green, coverage >= 85
uv run --with pre-commit pre-commit run --all-files  # clean
uv run --extra docs mkdocs build                     # strict build passes
python -m a3dasm --help                              # entry point works
```
