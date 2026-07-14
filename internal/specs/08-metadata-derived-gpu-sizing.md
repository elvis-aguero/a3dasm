# Spec 08 — Size the vLLM serve job from checkpoint metadata, not the model-name string

**GitHub issue #4.** Priority: medium (unblocks credible local-LLM runs on the
cluster). Status: **spec** (independent design; does not adopt the issue's
proposed solution wholesale — see §7).

## 1. Problem statement

`llm_slurm` decides how much GPU/CPU/RAM to request for a `vllm serve` job by
**string-matching the model-name against a hand-maintained table**. Two distinct
defects follow, one acute and one structural.

### 1a. Acute: the prefix matcher silently misfires on real HF ids
`_match_profile` (`slurm_llm.py:93-103`) lowercases the model string and keeps the
longest key for which `model.startswith(key)` holds:

```python
m = (model or "").lower()
for key in MODEL_SERVE_PROFILES:      # keys: "gemma-4", "llama-3", "qwen"
    if m.startswith(key) and len(key) > len(best_key):
        best_key = key
```

Real Hugging Face checkpoint ids are namespaced `org/repo`
(`"google/gemma-4-e4b-it"`, `"meta-llama/Llama-3.1-8B-Instruct"`). A key like
`"gemma-4"` can **never** be a prefix of `"google/gemma-4-..."`, so the match
fails and `resolve_serve_spec` (`slurm_llm.py:115-122`) falls through to
`_DEFAULT_PROFILE` (`gres="gpu:1"`, `mem="32G"`) with **only a `log.warning`** —
no error, no config-time stop. A 70B checkpoint therefore gets sized as a 1-GPU
32 GB job and OOMs on the GPU node minutes later, after the queue wait, instead of
failing loudly at submit. The existing tests only ever pass bare names
(`"gemma-4-27b-it"`, `test_slurm_llm.py:17-44`), so the namespaced-id path is
untested and the bug is invisible to the suite.

### 1b. Structural: sizing is guessed from the name at all
Even with the prefix bug fixed (e.g. match on the repo basename), the mechanism is
a table a human must extend for every new family (Gemma 4, Llama 4, Qwen 3, …).
The table *duplicates information the checkpoint already publishes* — parameter
count, dtype, architecture — so it drifts out of sync with reality by
construction. The prefix bug is the first symptom of that duplication, not the
disease.

Note the split in how the name is used today:
- **Sizing** (`gres`/`mem`/`cpus`/`tensor_parallel`) comes **only** from the
  profile table (`MODEL_SERVE_PROFILES`, `slurm_llm.py:71-81`). This is the part
  that mis-sizes.
- **Throughput advisory** already tries to read `params_b` from the name via
  `parse_params_billions` (`slurm_llm.py:199-203`), used only by
  `serve_throughput_warning` (`slurm_llm.py:221-251`) — it never feeds sizing.

So `ServeProfile` already carries the right fields (`params_b`, `gpu_model`,
`dtype_bytes`, `tensor_parallel`, `slurm_llm.py:60-64`); they are simply never
populated from fact and never used to size the allocation.

## 2. Requirements / non-goals

**Requirements**
- R1. A literal HF id (`"google/gemma-4-e4b-it"`) is sized from the checkpoint's
  own published metadata (parameter count + dtype), not the name string. This is
  the regression fix for §1a.
- R2. A short name (`"gemma-4"`) resolves to a canonical literal id before sizing,
  so `model: "gemma-4"` alone launches a correctly-sized server.
- R3. Explicit `llm_slurm` config overrides continue to win outright over anything
  derived (the hand-tuning escape hatch, `slurm_llm.py:123-125`).
- R4. Metadata unavailable (offline node, gated/private repo, network error, or a
  local filesystem path with no published metadata) degrades to a conservative
  default **plus a loud warning** — never raises, never silently mis-sizes below
  what the artifact needs.
- R5. No new heavyweight or auth-requiring runtime dependency. The metadata path
  must work with what is already installed.
- R6. Everything testable headless: no live HF network call, no live SLURM, no
  live LLM in the default suite.

**Non-goals**
- NG1. A comprehensive, always-current model registry. The alias layer stays tiny
  and curated (or config-supplied); metadata carries correctness for everything
  else. (Same posture the issue states.)
- NG2. Exact VRAM accounting (activation memory, exact KV-cache geometry per
  batch/context). We compute a *conservative leading-order* VRAM requirement,
  consistent with the module's existing "order-of-magnitude bound, not a
  guarantee" stance (`slurm_llm.py:181-185`).
- NG3. Auto-detecting which physical GPU SLURM will grant. GPU-count derivation
  needs a declared target GPU (see §3, Layer 2 caveat); we do not probe the
  cluster.

## 3. Recommended design

Keep the layered, most-explicit-wins shape `resolve_serve_spec` already has, and
change only *where the base comes from*: a fact fetched from the checkpoint,
instead of a guess keyed on its name.

Resolution order for `resolve_serve_spec(model, overrides)`:

```
model string
  │
  ├─ Layer 1  alias expansion      short name → canonical HF id (if known)
  │
  ├─ Layer 2  metadata sizing      fetch params+dtype → derive gres/mem/tp
  │             (base ServeProfile)
  │
  ├─ Layer 1.5 family serve-hints  merge non-sizing hints (vllm_args, e.g.
  │             (name-keyed table)  --max-model-len) that metadata cannot supply
  │
  └─ Layer 0  config overrides     llm_slurm.{gres,mem,cpus,vllm_args,...} win
```

### Layer 1 — alias expansion (small, and mostly config-driven)
A `resolve_model_id(model, cfg)` helper expands a short name to a canonical id:
1. If `model` is a key in a **config-supplied** `llm_slurm.aliases` map, use that.
2. Else if it is a key in a **tiny built-in** `MODEL_ALIASES` table, use that.
3. Else return `model` unchanged (today's behavior — literal id or local path).

Built-in aliases must point only at ids **verified to exist** at spec-implementation
time; an alias to a non-existent id reintroduces exactly the drift this issue
kills (see §7 for why I keep this layer smaller than the issue proposes). The map
is `alias -> id`; the `name:tag` ergonomics the issue admires (`gemma3:4b`) is
just alias keys that happen to contain a colon — no separate parser needed.

### Layer 2 — derive the base `ServeProfile` from checkpoint metadata
Given a literal id, fetch metadata **weights-free** and populate
`params_b`/`dtype_bytes`, then derive the allocation.

**Metadata source (diverges from the issue — see §7):** a single HTTP GET of the
Hub model-info JSON, `https://huggingface.co/api/models/{id}`, via the already-present
`requests` dependency (`pyproject.toml:38`). That payload carries a `safetensors`
block with the total parameter count and a per-dtype parameter breakdown — enough
for both `params_b` and `dtype_bytes` — without pulling weights and without a new
dependency. **Prefer a local read first:** if `HF_HOME`/the HF cache already holds
the checkpoint's `config.json` (common on a cluster that pre-stages models), read
`config.json` (`torch_dtype`, and param count if present) locally — this makes the
happy path work on an offline compute node. Order: local cache → Hub HTTP API →
Layer 3 fallback. Wrap the source behind one module-level function
(`_fetch_model_metadata(id) -> ModelMeta | None`) so tests stub exactly one seam,
mirroring the `_run`/`_probe` isolation the module already uses (`slurm_llm.py:19`).

> Field-name caveat for the implementer: confirm the exact JSON keys
> (`safetensors.total`, `safetensors.parameters.<dtype>`) against the live API
> before hard-coding them; treat a missing/renamed field as a fetch failure
> (→ Layer 3), never a crash.

**Params → allocation.** Metadata gives parameters and dtype; turning that into
`gres`/`mem` still needs to know each GPU's VRAM capacity. That is the honest core
of the design: we do not eliminate a hardware table, we **move the table from
fast-drifting model families to slowly-drifting GPU hardware.** Extend the
existing `_GPU_BW_TBPS` table (`slurm_llm.py:188-191`) with a parallel
`_GPU_VRAM_GB` capacity map (same keys), and derive:

- `weight_gb ≈ params_b * dtype_bytes` (fp16 70B ≈ 140 GB).
- `required_vram_gb ≈ weight_gb * KV_OVERHEAD` with a conservative
  `KV_OVERHEAD ≈ 1.3` (KV cache + fragmentation + CUDA-graph headroom). Leading
  order, per NG2.
- `tensor_parallel = ceil(required_vram_gb / per_gpu_vram_gb)` when a target GPU
  is known; `gres = "gpu:{tensor_parallel}"`.
- `mem` (host RAM for staging, distinct from VRAM) `≈ max(32, ceil(1.5 *
  weight_gb))` G, floored at today's default so small models are unaffected.

**Layer 2 caveat (issue glosses this):** `gres` derivation needs a per-GPU VRAM,
i.e. a known target GPU. If neither `llm_slurm.gpu_model` nor a
partition→GPU mapping is set, we **cannot** safely pick a GPU count. In that case
we still populate `params_b`/`dtype_bytes` (so the throughput advisory and `mem`
derivation work) but leave `gres`/`tensor_parallel` at the conservative default
and emit the same loud warning telling the user to set `gpu_model`. Metadata fixes
the *silent* mis-size (§1a) even here, because the warning now names a real
parameter count instead of guessing.

### Layer 1.5 — keep family profiles, but only for non-sizing serve hints
Do **not** retire `MODEL_SERVE_PROFILES` (diverges from issue §"Why this ordering":
"can be retired"). Metadata publishes params/dtype/architecture; it does **not**
tell you the right `--max-model-len` or other `vllm serve` tuning for a family —
that is operational knowledge, not an artifact fact. Reframe the table as a
**serve-hints** table (`vllm_args`, `port`, default `time`) merged *after* Layer 2
sizing and *before* config overrides. Match it on the **repo basename** (strip
`org/`, lowercase) so it stops silently missing namespaced ids — this alone is a
cheap, independent fix for §1a even before Layer 2 lands, and it makes the
name-keyed knowledge that *is* legitimate (context length) keep working.

### Layer 0 — config overrides (unchanged)
The existing merge (`slurm_llm.py:123-125`: only `ServeProfile` fields, non-`None`,
unknown keys ignored) stays exactly as is and wins last.

### Layer 3 — conservative fallback (unchanged contract)
Any failure in Layers 1-2 (fetch returns `None`, no metadata, gated repo) lands on
`_DEFAULT_PROFILE` with the existing loud warning (`slurm_llm.py:116-122`), now
worded to name *why* it fell back (offline / gated / unknown-GPU). Never raises.

## 4. Alternatives considered

| Option | Trade-off | Verdict |
|---|---|---|
| `transformers.AutoConfig.from_pretrained(id)` (issue's alt) | Heavy dep; `from_pretrained` can trigger downloads/imports; overkill to read a JSON. | Rejected (R5). |
| `huggingface_hub.HfApi().model_info(id)` (issue's primary) | Correct + clean, but a **new runtime dependency** for one JSON GET already doable with `requests`. | Rejected as default; acceptable as a thin optional wrapper if `requests`-direct proves brittle. |
| Direct HTTP GET of `/api/models/{id}` via `requests` | No new dep (R5); metadata-only; public repos need no auth. Must hard-code/confirm JSON keys and handle gated 401. | **Chosen.** |
| Fully retire `MODEL_SERVE_PROFILES` | Loses per-family `--max-model-len`/serve tuning metadata can't supply. | Rejected; reframed as serve-hints (Layer 1.5). |
| Derive GPU count with no GPU-VRAM table (pure metadata) | Impossible — params alone don't yield a GPU count without per-GPU capacity. | Rejected; hence `_GPU_VRAM_GB`. |
| Make the whole thing a hard config-time error on OOM-risk | Violates CLAUDE.md §4 (science budgets/nudges stay soft; no new hard stops without approval). | Rejected; stays warn-only. |

## 5. Assessment of the issue author's proposal

**Where I agree:**
- The diagnosis is exactly right and well-evidenced: prefix matching cannot match
  a namespaced `org/repo` id, and the deeper rot is duplicating artifact-published
  facts in a hand table. The `accelerate estimate-memory` precedent (size from the
  checkpoint's own `config.json`/safetensors) is the correct mental model.
- The most-explicit-wins layering, and the "alias table is explicitly *not* a
  registry" non-goal, are the right shape. I keep both.
- Keeping Layer 3's never-raise/never-silently-guess contract is correct.

**Where I diverge, and why:**
1. **Metadata source.** The issue offers `HfApi().model_info` *or*
   `AutoConfig.from_pretrained`. `transformers` is a heavyweight dependency and
   `from_pretrained` can download; `huggingface_hub` is a new dep for one JSON GET.
   The repo already ships `requests` (`pyproject.toml:38`) and none of these HF
   libs. I use a direct `requests` GET of `/api/models/{id}`, plus a **local HF-cache
   read first** so the happy path survives an offline compute node — a case the
   issue treats only as a failure/fallback.
2. **"Metadata sizes the allocation" is incomplete.** Params + dtype give a VRAM
   *requirement*; producing `gres` (a GPU *count*) still needs per-GPU VRAM
   capacity. The issue says it will "populate `mem`/`gres`/`tensor_parallel`" from
   the fetch without acknowledging this. I make it explicit: add `_GPU_VRAM_GB`,
   and when no target GPU is declared, derive `mem`/`params_b` but *not* `gres`,
   and say so loudly. The real win is reframed honestly: **move the maintained
   table from fast-drifting model families to slow-drifting GPU hardware**, not
   "eliminate the table."
3. **Do not retire `MODEL_SERVE_PROFILES`.** The issue says it can be retired once
   Layer 2 lands. Metadata cannot supply `--max-model-len`/serve tuning — that is
   operational, not an artifact fact. I keep it, reframed as a *serve-hints* layer
   matched on repo basename (which also independently fixes the prefix bug).
4. **The alias table's example ids look invented** (`google/gemma-4-31B-it`,
   `gemma-4:31b`). Codifying guessed ids reintroduces the exact drift the issue
   attacks — a built-in alias to a non-existent checkpoint is worse than no alias.
   I keep the built-in table minimal (verified ids only) and push the bulk to a
   **config-supplied `llm_slurm.aliases`**, so the ergonomics live where the user
   controls them and no framework release is needed to add a shortcut.
5. **Fix the prefix matcher regardless.** The issue frames the basename fix as a
   side effect of Layer 2. I make basename-normalized matching a small,
   independently landable, independently tested change (Layer 1.5) so the acute
   silent-misfire is closed even if the metadata path is deferred or unavailable.

## 6. Risks / failure modes / backward-compat

- **Backward compat.** Existing configs that pass a bare family name and rely on
  `MODEL_SERVE_PROFILES` keep working via Layer 1.5 (basename match is a superset
  of today's prefix match for bare names). Config-override precedence is byte-for-byte
  unchanged. `test_slurm_llm.py:17-44` must still pass (or be updated only where it
  asserts the *old* silent-default behavior we are deliberately changing).
- **Launch latency + network at submit.** Layer 2 adds one metadata GET on the
  submission host (where the agent already has internet — it talks to the LLM API).
  Guard with a short timeout; on timeout → Layer 3. Never block the run on it.
- **Gated/private repos** return 401/403 without a token → treat as fetch failure
  → Layer 3 warning naming "gated repo, set HF token or llm_slurm overrides."
- **Local filesystem model paths** (not HF ids) have no Hub metadata; local
  `config.json` read covers many, otherwise → Layer 3. Must not crash on a path.
- **VRAM heuristic is leading-order** (NG2). It can under-size a model with an
  unusually large KV cache / long context. Mitigated by the conservative
  `KV_OVERHEAD` and the existing `serve_throughput_warning`; documented as an
  estimate, and config overrides remain the escape hatch.
- **Stale/absent `safetensors` block** (repos that publish only `.bin`, or omit
  the field) → fetch returns `None` → Layer 3. Do not guess from the name to
  "rescue" it (that would resurrect §1b).

## 7. Test plan (headless, deterministic)

New/extended tests in `tests/test_slurm_llm.py` (mirrors its existing stub-one-seam
style; all network/SLURM stubbed).

Metadata + sizing:
1. `test_metadata_sizing_from_stubbed_hub` — monkeypatch `_fetch_model_metadata`
   to return `(params_b=70, dtype_bytes=2.0)`; with `gpu_model="a100"` (80 GB),
   `resolve_serve_spec("meta-llama/Llama-3.1-70B-Instruct", {"gpu_model":"a100"})`
   yields `tensor_parallel=2`, `gres="gpu:2"`, and `mem` ≥ derived floor.
2. `test_metadata_sizing_small_model_stays_one_gpu` — 7B fp16 on a100 →
   `gres="gpu:1"`, `tensor_parallel=1`.
3. `test_namespaced_id_is_sized_not_silently_defaulted` — **regression for §1a**:
   `"google/gemma-4-e4b-it"` with stubbed metadata is sized from metadata, and
   does *not* silently land on `_DEFAULT_PROFILE`.
4. `test_metadata_fetch_failure_falls_back_and_warns` — stub returns `None`;
   `resolve_serve_spec` → `_DEFAULT_PROFILE` + a WARNING (caplog), no raise (R4).
5. `test_no_target_gpu_derives_mem_but_not_gres` — metadata present, `gpu_model`
   unset → `params_b` populated, `gres` left at default, warning names `gpu_model`.
6. `test_local_cache_read_preferred_over_network` — stub a local `config.json`
   path; assert the Hub HTTP seam is *not* called (offline-node happy path).

Alias layer:
7. `test_config_alias_expands_before_sizing` — `llm_slurm.aliases={"m":"org/Repo-7B"}`;
   `resolve_model_id("m", cfg) == "org/Repo-7B"`.
8. `test_builtin_alias_expands` and `test_unknown_name_passes_through_unchanged`
   (literal id / local path preserved — nothing breaks for existing configs).
9. `test_config_alias_beats_builtin` (precedence).

Prefix-bug fix (Layer 1.5, independent of metadata):
10. `test_basename_match_finds_profile_for_namespaced_id` — `"google/gemma-4-9b"`
    resolves the gemma serve-hints (`--max-model-len` present), which
    `startswith("gemma-4")` never did.
11. `test_config_overrides_still_win_over_derived` — extend the existing
    `test_config_overrides_beat_profile_defaults` to also override a
    *metadata-derived* field and assert config wins (R3).

Wiring: `test_slurm_llm_wiring.py` — assert `_maybe_start_slurm_llm`
(`agent_runtime.py:643-644`) still calls `resolve_serve_spec` with the raw config
`model` and that a stubbed-metadata run renders a script with the derived `--gres`.

**KPI / done when.**
- *Mechanism:* tests 1-11 pass; the namespaced-id regression (test 3) fails on
  `main` and passes after.
- *Behavioral (needs a real GPU-cluster run, not claimed by mechanism):* a study
  with `model: "gemma-4"` (or a real 70B id) and `backend: vllm` launches a server
  whose `--gres`/`--mem` match the checkpoint's true size — no OOM on the serve
  node, no manual `llm_slurm` sizing overrides needed. Track against the prior
  behavior (silent 1-GPU/32 GB default) on the same config.

## 8. FEATURES.md

Update the existing entry **"Framework-owned local LLM on a SLURM GPU node
(vLLM)"** (`internal/FEATURES.md:266-294`) in the same commit that lands the code:
the sizing sentence changes from "model-keyed serve defaults" to
"metadata-derived sizing (params/dtype from the checkpoint) + config-supplied
aliases, with a name-keyed serve-hints fallback." No new declared *tool* (this is
infrastructure, not an agent tool), so `test_features_documented.py` is not
triggered; honor the catalog contract by hand per CLAUDE.md §5. Add the
`llm_slurm.aliases` config key to the **Config** bullet.
