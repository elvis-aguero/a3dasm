# Spec 09 — `QueryStore` compound-predicate filter (`where`) + explicit `limit`

## Problem

`QueryStore` (`src/a3dasm/_src/nodes/tools/routing.py`) is the read tool agents use to
ground design decisions in the evaluation ledger. As shipped it could filter only by
`delegation_ids`/`source`, and its default listing hard-capped at 20 rows. Its `n_best`
mode already drops the *objective* infeasibility sentinel (`_select_best_index`), but it
could not apply a **compound feasibility predicate across separate columns**.

For the supercompressible study, feasibility is `coilable==1 AND max_compressive_strain>=0.90
AND max_local_strain<=0.02 AND slenderness>=10` — four columns (one derived from inputs).
The single most common discovery question — *"of everything evaluated, which designs are
feasible, and which is best?"* — was therefore inexpressible, so agents fell back to
`ExperimentData.from_file()` + pandas by hand every run (attested: run `20260715T002538`
D005 implementer and critic-1; recurs in ≥3 prior runs). This is the one source-verified,
a3dasm-owned friction that survived a recency audit (many others were already fixed).

## Design (backward-compatible; two optional params)

1. **`where: str | None`** — a pandas `query()` expression evaluated over the **joined
   inputs+outputs** frame (`QueryStore` already loads both `df_in` and `df_out`). Applied as
   an extra mask *after* the `delegation_ids`/`source` filters and *before* `n_best`/listing,
   so both the ranking path and the list path honour it. Because the frame is joined and
   `query()` supports arithmetic, a derived quantity needs no stored column
   (`where="ratio_pitch/(2*ratio_b) >= 10"`).
2. **`limit: int | None`** — caps the default (non-`n_best`) listing; defaults to 20 (current
   behaviour). Coerced from string like `n_best`.

Killer query, one call:
```
QueryStore(where="coilable==1 and max_compressive_strain>=0.90 and max_local_strain<=0.02",
           n_best=5, output_name="<objective>", minimize=False)
```

## Safety / edges

- `where` runs on an in-memory joined copy (read-only; no store mutation).
- A bad expression is caught and returned as
  `"ERROR: could not evaluate where=<...>; Available columns: [...]"` — self-documenting,
  never raises (matches the existing `output_name`-not-found contract).
- Duplicate columns across the in/out join are de-duplicated (keep-first) before `query()`.
- `df_in is None` (mismatched namespaces) → `where` applies over outputs only.

## Scope / non-goals

Read-side query tool only. Does **not** touch the f3dasm-core `from_data` row-reordering /
domain-clamp behaviour (upstream), the oracle, or any physics. One implementation file
(`routing.py`) + one test file (`tests/test_querystore_where.py`), plus the FEATURES.md entry.

## Tests (`tests/test_querystore_where.py`, headless, deterministic)

- compound `where` masks correctly (AND over coilable + mcs);
- `where` + `n_best` ranks the feasible subset only (infeasible top-objective excluded);
- `where` arithmetic on input columns;
- bad `where` returns an `ERROR:` string, does not raise;
- `limit` caps the listing and reports the remainder; no-args unchanged.
