---
id: experimentdata-gotchas
title: ExperimentData gotchas — input shaping and to_numpy()
tags: [experimentdata, to_numpy, domain, gotcha, idiom]
audience: [implementer]
---
The exact, CI-verified call forms (sampling, `to_numpy`, candidate
construction, best-N) live in the core f3dasm idioms injected into your
prompt — use those verbatim. This entry only records the *why* behind the
gotchas they encode, so you trust them:

- **Domain variable names are your contract.** Build the `Domain` with the
  parameter names the study uses (e.g. `x1..x8`, or `k`/`m`) and keep input
  dicts consistent with them. Mixing `x0..x7` with a domain declared `x1..x8`
  silently misaligns columns.

- **Column order follows the order you ADD variables, not your task wording or
  dict key order.** `domain.input_names`, the `to_numpy()` X columns, and the
  stored CSV all follow Domain *insertion* order (it is NOT alphabetised). So if
  the task says "(θ, r_offset)" but you call `add_float("r_offset", …)` first,
  column 0 is `r_offset`. Key your `ExperimentSample(_input_data={...})` by NAME
  (a dict, so order-independent) and trust `input_names` for positional arrays —
  do not assume a sort.

- **Load with an ABSOLUTE `project_dir`.** `ExperimentData.from_file(project_dir=
  os.environ["F3DASM_CANONICAL_STORE"])` (or any absolute path) always works. A
  *relative* path only re-resolves against Hydra's original cwd, and outside a
  Hydra run (e.g. under `uv run`) it is taken as-is against the current cwd — so
  pass the absolute store path and skip the ambiguity.

- **`to_numpy()` returns a tuple `(X, y)` — it takes no argument.** Not
  `to_numpy("input")`. The output array can carry metadata columns; if in
  doubt about the objective column, go through pandas (`_, df_out =
  data.to_pandas(); y = df_out["<output_name>"].to_numpy()`).

- **There is no `ExperimentData.from_numpy` and no `data.sample(...)`.** To
  wrap proposed points you build `ExperimentSample(_input_data={...})` (the
  kwarg is `_input_data`, not `input`) and pass them to
  `ExperimentData.from_data`; to sample you use `create_sampler(...).call(...)`.
  Both are shown in the injected core idioms.

- **The Domain input methods are `add_float` / `add_int` / `add_category` /
  `add_constant` (also `add_array`).** Do NOT invent names from other
  frameworks: there is no `add_continuous_input`, `add_continuous`,
  `add_discrete`, `add_categorical`, `add_variable`, or
  `add_parameter(type="continuous")`. A continuous bound is
  `domain.add_float("x0", low, high)`; an integer is
  `domain.add_int("k", low, high)`. A wrong method name raises `AttributeError`
  the moment the reproduction gate executes your DoE cell — so the run bounces
  on it; use the real names from the start.
