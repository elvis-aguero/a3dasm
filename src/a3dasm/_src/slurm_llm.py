"""Framework-owned local LLM backend on SLURM (vLLM on a GPU node).

The agent loop stays on its current node; this module owns the *separate* GPU
allocation that serves the model with vLLM, and drives its lifecycle:

    resolve_serve_spec(model, overrides)   # model-keyed defaults, config overrides
        -> submit_serve_job(...)           # sbatch a `vllm serve` job (reuses
                                           #   f3dasm SlurmCluster/SlurmResources)
        -> wait_until_running(jobid)       # poll squeue for the granted node
        -> wait_until_ready(base_url)      # poll /v1/models past the cold load
        -> (publish VLLM_BASE_URL)         # the vllm adapter resolves it from env
        -> cancel_job(jobid)               # scancel on EVERY exit path

Design notes:
- A persistent server is NOT an f3dasm eval array, so we do not route it through
  Pipeline/SlurmExecutor; we reuse the config objects (SlurmCluster/
  SlurmResources) + the plain `sbatch` submit idiom. GPU is requested via
  SlurmResources.extra_sbatch={"gres": ...}.
- All external calls (sbatch/squeue/scancel, the HTTP probe) go through small
  module-level functions so tests can stub them without a real cluster.
- Phase 2 (walltime chaining + a stable local proxy) is designed in the plan but
  not built here; this owns a single allocation for the run's lifetime.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import re
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from . import settings

log = logging.getLogger("a3dasm.slurm_llm")

# --------------------------------------------------------------------------
# Model-keyed serve profiles
# --------------------------------------------------------------------------


@dataclass
class ServeProfile:
    """Default SLURM + vLLM serving parameters for a model.

    Sizing fields (``gres``/``mem``/``tensor_parallel``/``params_b``/
    ``dtype_bytes``) are DERIVED from the checkpoint's own published metadata
    (see :func:`resolve_serve_spec`); the family serve-hints table below carries
    only *operational* knowledge metadata cannot publish (``vllm_args`` such as
    context length, ``time``, ``port``). A study's config overrides any field.
    GPU is requested via ``gres`` (SLURM ``--gres``). ``partition`` None means
    "use the cluster's configured default partition".
    """
    gres: str = "gpu:1"
    mem: str = "32G"
    cpus_per_task: int = 8
    time: str = "08:00:00"
    partition: str | None = None
    port: int = 8000
    # Extra args appended to `vllm serve <model>` (e.g. context length).
    vllm_args: list[str] = field(default_factory=list)
    # Build-time throughput-estimate + sizing inputs:
    gpu_model: str | None = None      # e.g. "l40s" — a key of _GPU_VRAM_GB
    dtype_bytes: float = 2.0          # fp16=2, fp8/int8=1, int4=0.5
    tensor_parallel: int = 1          # GPUs the weights are sharded across
    params_b: float | None = None     # model size in billions; derived from metadata


# Family serve-HINTS, keyed by a lowercase repo-basename prefix (longest match
# wins). These carry ONLY non-sizing operational defaults metadata can't publish
# (context length via vllm_args, walltime). Sizing (gres/mem/tp) is derived from
# the checkpoint metadata and wins over anything here. Optional and small — a
# model with no hint just uses the bare defaults + metadata sizing.
MODEL_SERVE_PROFILES: dict[str, ServeProfile] = {
    # gemma-4 keeps its FULL 256K context (a hard requirement) and an FP8 KV
    # cache so that context fits one 48 GB L40S. Context-aware sizing (below)
    # reads --max-model-len + --kv-cache-dtype from here to budget KV correctly.
    # NB: FP8 KV also needs an FP8-capable GPU (Ada/Hopper/Blackwell) — the same
    # class the FP8 weight default targets; on Ampere set --kv-cache-dtype auto.
    "gemma-4": ServeProfile(time="08:00:00",
        vllm_args=["--max-model-len", "262144", "--kv-cache-dtype", "fp8"]),
    "llama-3": ServeProfile(time="08:00:00", vllm_args=["--max-model-len", "8192"]),
    "qwen": ServeProfile(time="08:00:00", vllm_args=["--max-model-len", "16384"]),
    "mistral": ServeProfile(time="08:00:00", vllm_args=["--max-model-len", "16384"]),
}

# MINIMAL, CURRENT short-name -> canonical HF id table. The whole point of issue
# #4 is to NOT maintain a broad dictionary of model facts: this stays tiny and
# current, and live metadata derivation carries correctness for everything else.
# Users add their own via ``llm_slurm.aliases`` (config wins). A full HF id is
# sized correctly with NO alias at all — this is pure convenience.
#
# gemma-4 (Google, released 2026-04) is the intended cheap-but-SOTA family.
# Owner-verified, case-sensitive HF ids — use exactly:
#   E4B  = the cheap default (~4B-effective), fits a single 24-48 GB card;
#   31B  = "max power for one 48 GB L40S": 30.7B served FP8 (~31 GB) leaves KV
#          headroom on a single card (see the quantization knob below).
_GEMMA_4_E4B = "google/gemma-4-E4B-it"    # cheap default
_GEMMA_4_31B = "google/gemma-4-31B-it"    # max-power-for-48GB (FP8, single card)
MODEL_ALIASES: dict[str, str] = {
    "gemma-4": _GEMMA_4_E4B,        # primary/default: smallest instruct variant
    "gemma-4-e4b": _GEMMA_4_E4B,    # explicit cheap variant
    "gemma-4-31b": _GEMMA_4_31B,    # explicit power variant
}

_DEFAULT_PROFILE = ServeProfile()

# Host RAM floor (G). Fixed VRAM headroom (GB) added to weight+KV for the CUDA
# context, activations, and fragmentation.
_DEFAULT_MEM_GB = 32
_FIXED_OVERHEAD_GB = 2.0
# FALLBACK ONLY: a flat multiplier over raw weight bytes, used when the model
# config lacks the fields needed for a real context-aware KV estimate (see
# _estimate_vram_gb). Context-blind and known to under-budget long-context KV —
# never the primary path when the config is available.
_KV_OVERHEAD = 1.3

# Per-GPU usable VRAM (GB), keyed by the EXACT Slurm gres names on the Oscar
# (Brown CCV) cluster — these are the strings ``--gres=gpu:<name>:<n>`` must use.
# GPU hardware drifts far slower than model releases, so this table is the
# maintainable place to hold sizing knowledge (vs. a per-model-family table that
# drifts every release). No a100 exists on this cluster — do not add one.
_GPU_VRAM_GB: dict[str, float] = {
    "nvidia_geforce_rtx_3090": 24,
    "nvidia_rtx_a5000": 24,
    "a5500": 24,
    "quadro_rtx_6000": 24,
    "a2": 16,
    "l40s": 48,
    "nvidia_a40": 48,
    "nvidia_rtx_a6000": 48,
    "h100": 80,
    "nvidia_rtx_pro_6000_blackwell": 96,
    "nvidia_b200": 180,
}

# Default target GPU when the study sets none: a card actually schedulable on the
# general ``gpu`` partition (48 GB, plentiful). Never a card absent from
# _GPU_VRAM_GB. Zero-config runs size against this.
_DEFAULT_GPU_MODEL = "l40s"

# GPUs with native FP8 tensor cores (Ada/Hopper/Blackwell). Only these can serve
# FP8 weights; the Ampere/Turing cards below have none. The DEFAULT quant is
# GPU-aware (FP8 only here); an explicit llm_quantization=fp8 still wins.
_FP8_CAPABLE: frozenset[str] = frozenset({
    "l40s", "h100", "nvidia_rtx_pro_6000_blackwell", "nvidia_b200",
})

# Serve-time quantization -> bytes/param actually resident in VRAM during decode.
# The SERVE dtype can differ from the checkpoint's native dtype (a BF16 weight
# can be served FP8), so VRAM sizing must use this, not the metadata dtype.
_QUANT_BYTES: dict[str, float] = {
    "bf16": 2.0, "bfloat16": 2.0, "fp16": 2.0, "float16": 2.0, "f16": 2.0,
    "fp8": 1.0, "float8": 1.0, "f8": 1.0, "int8": 1.0, "i8": 1.0, "w8a8": 1.0,
    "q4": 0.5, "int4": 0.5, "awq": 0.5, "gptq": 0.5, "w4a16": 0.5, "nf4": 0.5,
    "bitsandbytes": 0.5,
}
# vLLM ``--quantization`` value for a serve quant, when one is needed (BF16/FP16
# are the native path — no flag). Only methods vLLM names directly.
_QUANT_VLLM: dict[str, str] = {
    "fp8": "fp8", "float8": "fp8", "f8": "fp8",
    "awq": "awq", "gptq": "gptq", "bitsandbytes": "bitsandbytes",
}
# Default serve quant: FP8. Chosen so the zero-config power path (gemma-4-31B,
# 30.7B) fits a single 48 GB L40S (~31 GB weights + KV headroom). A study picks
# another via the ``llm_quantization`` runtime knob; ``auto`` keeps native dtype.
_DEFAULT_QUANT = "fp8"


def _explicit_quant() -> str:
    """The serve quant the study set EXPLICITLY (``llm_quantization`` in the
    runtime block; env override ``F3DASM_LLM_QUANTIZATION``), normalized — or ""
    if unset. An explicit value is the user's call and always wins, even FP8 on
    a non-FP8 GPU."""
    return settings.get_str("llm_quantization", "").strip().lower()


def _resolve_quant(gpu_model: str | None, meta, context_len,
                   kv_dtype_bytes: float) -> str:
    """The serve quant to use. Explicit config wins; otherwise a GPU-AWARE
    default: FP8 on FP8-capable cards, else the safe path — BF16 if it fits one
    card, else Q4 (4-bit) — so a zero-config run never asks an Ampere/Turing GPU
    to serve FP8 it cannot."""
    explicit = _explicit_quant()
    if explicit:
        return explicit
    if str(gpu_model or "").lower() in _FP8_CAPABLE:
        return _DEFAULT_QUANT                      # "fp8"
    vram = _GPU_VRAM_GB.get(str(gpu_model or "").lower())
    if vram and meta is not None and meta.params_b:
        bf16_vram, _ = _estimate_vram_gb(meta, 2.0, context_len, kv_dtype_bytes)
        return "bf16" if bf16_vram <= vram else "q4"
    return "bf16"


def _serve_dtype_bytes(quant: str, native_bytes: float | None) -> float:
    """Bytes/param resident in VRAM for the serve quant. ``auto``/``native``/
    ``none`` keep the checkpoint's own dtype; unknown quant falls back to it."""
    if quant in ("", "auto", "native", "none"):
        return native_bytes or 2.0
    return _QUANT_BYTES.get(quant, native_bytes or 2.0)


def _quant_vllm_args(quant: str) -> list[str]:
    """The ``vllm serve`` flag(s) that actually apply the serve quant, or []
    (BF16/FP16 need none)."""
    method = _QUANT_VLLM.get(quant)
    return ["--quantization", method] if method else []


# FP8 KV-cache dtype flag values (1 byte/elem); anything else is fp16/bf16.
_FP8_KV = {"fp8", "fp8_e5m2", "fp8_e4m3"}


def _arg_value(args: list[str], flag: str) -> str | None:
    """The value following ``flag`` in a vLLM arg list, or None."""
    try:
        return args[args.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _int_arg(args: list[str], flag: str) -> int | None:
    v = _arg_value(args, flag)
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _kv_cache_dtype_bytes(args: list[str]) -> float:
    """Bytes/element the KV cache occupies, from ``--kv-cache-dtype``. Default
    (``auto``/absent) is the model dtype = 2 bytes (fp16/bf16); FP8 = 1."""
    return 1.0 if (_arg_value(args, "--kv-cache-dtype") or "").lower() in _FP8_KV else 2.0

# torch_dtype / safetensors dtype string -> bytes per parameter.
_DTYPE_BYTES: dict[str, float] = {
    "float32": 4, "f32": 4, "fp32": 4,
    "float16": 2, "f16": 2, "fp16": 2, "bfloat16": 2, "bf16": 2,
    "float8": 1, "f8": 1, "fp8": 1, "int8": 1, "i8": 1, "uint8": 1, "u8": 1,
    "int4": 0.5, "i4": 0.5, "uint4": 0.5, "u4": 0.5, "q4": 0.5,
}


@dataclass
class ModelMeta:
    """Facts we size from. ``params_b``/``dtype_bytes`` are enough for a
    weight-only estimate; the rest (from the model ``config.json``) enable a
    context-aware KV-cache estimate. Any of the KV fields may be None — the
    sizer falls back to a flat heuristic when they are missing."""
    params_b: float | None = None
    dtype_bytes: float | None = None
    # KV-cache geometry (config.json):
    num_layers: int | None = None
    num_kv_heads: int | None = None          # sliding/default-layer KV heads
    head_dim: int | None = None              # sliding/default-layer head dim
    sliding_window: int | None = None        # None -> full attention everywhere
    num_sliding_layers: int | None = None
    num_global_layers: int | None = None
    global_kv_heads: int | None = None       # gemma-style separate global dims
    global_head_dim: int | None = None
    max_position_embeddings: int | None = None


def _kv_fields_from_config(cfg: dict) -> dict:
    """Extract KV-cache geometry from a HF ``config.json`` mapping. Tolerant:
    any absent field stays None (the sizer then falls back). Handles both the
    explicit ``layer_types`` list and the ``sliding_window_pattern`` int form
    for the sliding/global attention split (gemma-2/3/4)."""
    out: dict = {}
    layers = cfg.get("num_hidden_layers")
    out["num_layers"] = layers
    out["num_kv_heads"] = (cfg.get("num_key_value_heads")
                           or cfg.get("num_attention_heads"))
    head_dim = cfg.get("head_dim")
    if head_dim is None and cfg.get("hidden_size") and cfg.get("num_attention_heads"):
        head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    out["head_dim"] = head_dim
    out["sliding_window"] = cfg.get("sliding_window")
    out["max_position_embeddings"] = cfg.get("max_position_embeddings")
    out["global_kv_heads"] = cfg.get("num_global_key_value_heads")
    out["global_head_dim"] = cfg.get("global_head_dim")
    n_global = n_sliding = None
    lt = cfg.get("layer_types")
    if isinstance(lt, list) and lt:
        n_global = sum(1 for t in lt
                       if "full" in str(t).lower() or "global" in str(t).lower())
        n_sliding = len(lt) - n_global
    elif cfg.get("sliding_window_pattern") and layers:
        # gemma convention: every P-th layer is global attention
        p = int(cfg["sliding_window_pattern"])
        if p > 0:
            n_global = layers // p
            n_sliding = layers - n_global
    out["num_global_layers"] = n_global
    out["num_sliding_layers"] = n_sliding
    return out


@dataclass
class ServeSpec:
    """A resolved plan to serve one model: the model id + the merged profile."""
    model: str
    profile: ServeProfile


def _model_basename(model: str) -> str:
    """Repo basename, lowercased: 'google/Gemma-2-9b-it' -> 'gemma-2-9b-it'.

    HF ids are namespaced ``org/repo``; matching on the bare basename is what a
    prefix key like 'gemma-2' can actually match (the old ``startswith`` on the
    full id could never match a namespaced id — issue #4)."""
    return (model or "").rsplit("/", 1)[-1].lower()


def _match_profile(model: str) -> tuple[ServeProfile, bool]:
    """Return (serve-hints profile, matched). Longest prefix match on the repo
    basename; (default, False) if nothing matches."""
    m = _model_basename(model)
    best_key = ""
    for key in MODEL_SERVE_PROFILES:
        if m.startswith(key) and len(key) > len(best_key):
            best_key = key
    if best_key:
        return MODEL_SERVE_PROFILES[best_key], True
    return _DEFAULT_PROFILE, False


def resolve_model_id(model: str, overrides: dict | None = None) -> str:
    """Layer 1: expand a short alias to a canonical HF id. Config-supplied
    ``aliases`` win over the built-in table; an unknown name passes through
    unchanged (already a literal id or a local path)."""
    overrides = overrides or {}
    cfg_aliases = overrides.get("aliases") or {}
    if model in cfg_aliases:
        return str(cfg_aliases[model])
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    return model


def _dtype_bytes_from(name: str | None) -> float | None:
    """Map a torch_dtype/safetensors dtype string to bytes/param, or None."""
    if not name:
        return None
    return _DTYPE_BYTES.get(str(name).strip().lower())


def _hf_cache_roots() -> list:
    """Candidate HF hub cache roots, most-specific first. No I/O here."""
    from pathlib import Path
    roots = []
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        roots.append(Path(hub))
    home = os.environ.get("HF_HOME")
    if home:
        roots.append(Path(home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def _read_local_metadata(model_id: str) -> ModelMeta | None:
    """Read params/dtype from a pre-staged HF cache snapshot, weights-free.

    Prefer this over the network so the happy path works on an offline compute
    node. Reads ``config.json`` (torch_dtype) and the safetensors index
    (``metadata.total_size`` bytes) from the newest snapshot. Returns None if no
    usable local metadata (caller then tries the network). Never raises."""
    import json
    from pathlib import Path
    try:
        if "/" not in model_id:            # local path or bare name, not org/repo
            p = Path(model_id)
            snap_dirs = [p] if p.is_dir() else []
        else:
            folder = "models--" + model_id.replace("/", "--")
            snap_dirs = []
            for root in _hf_cache_roots():
                snaps = root / folder / "snapshots"
                if snaps.is_dir():
                    snap_dirs.extend(sorted(snaps.iterdir(),
                                            key=lambda d: d.stat().st_mtime,
                                            reverse=True))
        for snap in snap_dirs:
            cfg_path = snap / "config.json"
            if not cfg_path.is_file():
                continue
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            dtype_bytes = _dtype_bytes_from(cfg.get("torch_dtype")) or 2.0
            idx = snap / "model.safetensors.index.json"
            params_b = None
            if idx.is_file():
                meta = json.loads(idx.read_text(encoding="utf-8")).get("metadata", {})
                total_bytes = meta.get("total_size")
                if total_bytes:
                    params_b = float(total_bytes) / dtype_bytes / 1e9
            if params_b:
                return ModelMeta(params_b=params_b, dtype_bytes=dtype_bytes,
                                 **_kv_fields_from_config(cfg))
        return None
    except Exception:  # noqa: BLE001 — local read is best-effort; fall through
        return None


def _fetch_hub_config(model_id: str, timeout: float) -> dict:
    """GET the model's ``config.json`` from the Hub (weights-free). Isolated so
    tests stub it; returns {} on any failure. Never raises."""
    try:
        import requests
        resp = requests.get(
            f"https://huggingface.co/{model_id}/resolve/main/config.json",
            timeout=timeout)
        return resp.json() if resp.status_code == 200 else {}
    except Exception:  # noqa: BLE001 — best-effort; KV estimate falls back
        return {}


def _fetch_hub_metadata(model_id: str, timeout: float) -> ModelMeta | None:
    """Read params/dtype + KV geometry from the Hub, weights-free.

    A metadata GET of ``/api/models/{id}`` (no weight download, no auth for
    public repos) via the already-present ``requests`` dep gives the ``safetensors``
    block (total parameter count + per-dtype breakdown); a second GET of the
    model ``config.json`` gives the KV-cache geometry for context-aware sizing.
    Returns None if params can't be read (offline, gated repo 401, timeout,
    missing field); a missing config.json just leaves the KV fields None (the
    sizer then falls back). Never raises."""
    if "/" not in model_id:                # not an org/repo id — nothing to fetch
        return None
    try:
        import requests
        resp = requests.get(
            f"https://huggingface.co/api/models/{model_id}", timeout=timeout)
        if resp.status_code != 200:
            return None
        st = (resp.json() or {}).get("safetensors") or {}
        total = st.get("total")
        params = st.get("parameters") or {}
        if not total and params:
            total = sum(v for v in params.values() if isinstance(v, (int, float)))
        if not total:
            return None
        dtype_bytes = 2.0
        if params:                         # dominant dtype by parameter count
            dominant = max(params, key=lambda k: params.get(k) or 0)
            dtype_bytes = _dtype_bytes_from(dominant) or 2.0
        kv = _kv_fields_from_config(_fetch_hub_config(model_id, timeout))
        return ModelMeta(params_b=float(total) / 1e9, dtype_bytes=dtype_bytes, **kv)
    except Exception:  # noqa: BLE001 — network is best-effort; fall through
        return None


def fetch_model_metadata(model_id: str) -> ModelMeta | None:
    """Layer 2 metadata: local HF cache first (offline-safe), then the Hub HTTP
    API, then None (caller falls back to conservative defaults + a loud warning).

    Fetching is on by default and needs zero config. ``llm_metadata_fetch: false``
    (runtime block) disables the network hop (local-cache-only); the timeout is
    ``llm_metadata_timeout_s`` (default 8s)."""
    local = _read_local_metadata(model_id)
    if local is not None:
        return local
    if not settings.get_bool("llm_metadata_fetch", True):
        return None
    timeout = settings.get_float("llm_metadata_timeout_s", 8.0)
    return _fetch_hub_metadata(model_id, timeout)


def _kv_cache_gb(meta: ModelMeta, context_len: int | None,
                 kv_dtype_bytes: float) -> float | None:
    """Context-aware KV-cache size (GB) from the model config, or None when the
    config lacks the geometry (caller then falls back). Per token per layer the
    cache holds K and V: ``2 * kv_heads * head_dim * kv_dtype_bytes``. Sliding
    layers cap their length at the sliding window; global (full-attention) layers
    scale with the full context and may use separate gemma-style head dims."""
    if not (meta.num_layers and meta.num_kv_heads and meta.head_dim
            and context_len):
        return None

    def per_layer_bytes(kv_heads, head_dim, length):
        return 2 * kv_heads * head_dim * kv_dtype_bytes * length

    have_split = (meta.num_sliding_layers is not None
                  and meta.num_global_layers is not None and meta.sliding_window)
    if have_split:
        g_heads = meta.global_kv_heads or meta.num_kv_heads
        g_dim = meta.global_head_dim or meta.head_dim
        sliding_len = min(context_len, meta.sliding_window)
        total = (meta.num_global_layers * per_layer_bytes(g_heads, g_dim, context_len)
                 + meta.num_sliding_layers
                 * per_layer_bytes(meta.num_kv_heads, meta.head_dim, sliding_len))
    else:                              # plain full attention: every layer full ctx
        total = meta.num_layers * per_layer_bytes(
            meta.num_kv_heads, meta.head_dim, context_len)
    return total / 1e9


def _estimate_vram_gb(meta: ModelMeta, weight_dtype_bytes: float,
                      context_len: int | None,
                      kv_dtype_bytes: float) -> tuple[float, bool]:
    """(required VRAM GB, used_fallback). Prefers weight + context-aware KV + a
    small fixed overhead; falls back to the flat weight * _KV_OVERHEAD multiplier
    only when the config lacks KV geometry."""
    weight_gb = (meta.params_b or 0.0) * weight_dtype_bytes
    kv_gb = _kv_cache_gb(meta, context_len, kv_dtype_bytes)
    if kv_gb is None:
        return weight_gb * _KV_OVERHEAD, True
    return weight_gb + kv_gb + _FIXED_OVERHEAD_GB, False


def _derive_sizing(meta: ModelMeta, gpu_model: str | None, weight_dtype_bytes: float,
                   context_len: int | None, kv_dtype_bytes: float) -> tuple[dict, bool]:
    """Turn (params, serve-dtype, context, kv-dtype) into ServeProfile sizing
    overrides; returns (overrides, used_kv_fallback). VRAM is sized on the SERVE
    weight dtype plus a context-aware KV estimate (see _estimate_vram_gb). GPU-
    count derivation needs a known per-GPU VRAM; if ``gpu_model`` is unknown we
    still set mem + params_b/dtype_bytes but leave gres/tp at the default. When
    the GPU is known the derived gres names the card (``gpu:<gres_name>:<n>``) so
    Slurm grants the exact type the size was computed for — not just any GPU."""
    import math
    out: dict = {}
    if not meta.params_b:
        return out, False
    out["params_b"] = meta.params_b
    out["dtype_bytes"] = weight_dtype_bytes
    weight_gb = meta.params_b * weight_dtype_bytes
    out["mem"] = f"{max(_DEFAULT_MEM_GB, math.ceil(1.5 * weight_gb))}G"
    required_gb, used_fallback = _estimate_vram_gb(
        meta, weight_dtype_bytes, context_len, kv_dtype_bytes)
    key = (gpu_model or "").lower()
    vram = _GPU_VRAM_GB.get(key)
    if vram:
        tp = max(1, math.ceil(required_gb / vram))
        out["tensor_parallel"] = tp
        out["gres"] = f"gpu:{key}:{tp}"
        out["gpu_model"] = key      # record so the throughput check can run too
    return out, used_fallback


_PROFILE_FIELDS = {f.name for f in dataclasses.fields(ServeProfile)}


def resolve_serve_spec(model: str, overrides: dict | None = None) -> ServeSpec:
    """Resolve a model string to a sized serve plan, most-explicit-wins.

    Layered (each a clean fallback for the one above): config overrides win
    outright > metadata-derived sizing (a fact the checkpoint publishes) >
    family serve-hints (basename-matched, non-sizing) > conservative default.
    Never raises: if metadata is unavailable (offline, gated repo, unknown GPU)
    it degrades to the conservative default with a loud warning, never a silent
    mis-size. ``overrides`` is the study's ``llm_slurm`` config block; only keys
    that are ServeProfile fields are applied (unknown keys ignored, not errors).
    """
    overrides = overrides or {}
    model_id = resolve_model_id(model, overrides)             # Layer 1

    hints, _matched = _match_profile(model_id)                # Layer 1.5 (base)
    merged = {f: getattr(hints, f) for f in _PROFILE_FIELDS}

    # Zero-config default: a card actually schedulable on this cluster, so a run
    # that sets nothing still gets a correctly-sized, grantable gres.
    gpu_model = overrides.get("gpu_model") or hints.gpu_model or _DEFAULT_GPU_MODEL
    # Sizing inputs read from the EFFECTIVE vLLM args (config replaces hints):
    # context length and KV-cache dtype drive the context-aware KV budget.
    eff_vllm_args = (overrides["vllm_args"] if overrides.get("vllm_args") is not None
                     else list(hints.vllm_args))
    context_len = _int_arg(eff_vllm_args, "--max-model-len")
    kv_dtype_bytes = _kv_cache_dtype_bytes(eff_vllm_args)

    meta = fetch_model_metadata(model_id)                     # Layer 2
    if meta is not None and meta.params_b:
        if context_len is None:                # fall back to the model's own max
            context_len = meta.max_position_embeddings
        # GPU-aware serve quant (explicit config wins; FP8 only on FP8 cards).
        quant = _resolve_quant(gpu_model, meta, context_len, kv_dtype_bytes)
        weight_dtype_bytes = _serve_dtype_bytes(quant, meta.dtype_bytes)
        derived, used_fallback = _derive_sizing(
            meta, gpu_model, weight_dtype_bytes, context_len, kv_dtype_bytes)
        merged.update(derived)                                # fact beats hint
        if not _GPU_VRAM_GB.get(str(gpu_model).lower()):
            log.warning(
                "llm_slurm: sized %r from metadata (~%.1fB, serve %g B/param) but "
                "gpu_model=%r is not a known cluster GPU — gres left at %s. Set "
                "llm_slurm.gpu_model (one of %s) so the GPU count can be derived.",
                model_id, meta.params_b, weight_dtype_bytes,
                gpu_model, merged["gres"], sorted(_GPU_VRAM_GB))
        elif used_fallback:
            log.warning(
                "llm_slurm: %r config lacks KV-cache geometry — sized with the "
                "context-blind fallback multiplier (%gx weights), which under-"
                "budgets long-context KV. Verify gres=%s or set llm_slurm.gres.",
                model_id, _KV_OVERHEAD, merged.get("gres"))
    else:                                                     # Layer 3 fallback
        quant = _resolve_quant(gpu_model, None, context_len, kv_dtype_bytes)
        log.warning(
            "llm_slurm: no checkpoint metadata for %r (offline, gated repo, or "
            "no published safetensors) — using conservative defaults (%s, gres="
            "%s, %s). Set llm_slurm overrides (gres/mem/time/vllm_args) or stage "
            "the model in the HF cache for metadata-derived sizing.",
            model_id, _DEFAULT_PROFILE.mem, _DEFAULT_PROFILE.gres,
            _DEFAULT_PROFILE.time)

    for k, v in overrides.items():                            # Layer 0 wins last
        if k in _PROFILE_FIELDS and v is not None:
            merged[k] = v

    # Apply the serve quant to the vLLM launch (after overrides, unless the user
    # already set --quantization). Keeps sizing and the actual serve dtype in
    # agreement: FP8 sizing means an FP8 serve.
    q_args = _quant_vllm_args(quant)
    vllm_args = list(merged.get("vllm_args") or [])
    if q_args and "--quantization" not in vllm_args:
        merged["vllm_args"] = vllm_args + q_args

    return ServeSpec(model=model_id, profile=ServeProfile(**merged))


# --------------------------------------------------------------------------
# SLURM lifecycle (each external call isolated for stubbing)
# --------------------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Run a short SLURM CLI command. Isolated so tests stub one function."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def render_serve_script(spec: ServeSpec, cluster, base_url_port: int,
                        log_dir: str) -> str:
    """Render the sbatch script that serves the model with vLLM. Reuses the
    f3dasm SlurmCluster fields (partition/account/env_setup/env_vars/runner)."""
    partition = spec.profile.partition or getattr(cluster, "partition", "batch")
    account = getattr(cluster, "account", "default")
    env_setup = list(getattr(cluster, "env_setup", []) or [])
    env_vars = dict(getattr(cluster, "env_vars", {}) or {})
    runner = getattr(cluster, "runner", "python") or "python"

    sb = [
        "#!/bin/bash",
        "#SBATCH -J vllm-serve",
        f"#SBATCH -p {partition}",
        f"#SBATCH -A {account}",
        "#SBATCH -N 1",
        f"#SBATCH -c {spec.profile.cpus_per_task}",
        f"#SBATCH --mem={spec.profile.mem}",
        f"#SBATCH --gres={spec.profile.gres}",
        f"#SBATCH --time={spec.profile.time}",
        f"#SBATCH -o {log_dir}/vllm-%j.out",
        f"#SBATCH -e {log_dir}/vllm-%j.err",
    ]
    body = [""]
    for k, v in env_vars.items():
        body.append(f"export {k}={shlex.quote(str(v))}")
    body.extend(env_setup)
    tp = spec.profile.tensor_parallel
    args = list(spec.profile.vllm_args)
    if tp and tp > 1:
        args += ["--tensor-parallel-size", str(tp)]
    vllm_extra = " ".join(shlex.quote(a) for a in args)
    body.append(
        f"{runner} -m vllm.entrypoints.openai.api_server "
        f"--model {shlex.quote(spec.model)} --host 0.0.0.0 "
        f"--port {base_url_port} {vllm_extra}".rstrip())
    return "\n".join(sb + body) + "\n"


# --------------------------------------------------------------------------
# Build-time, leading-order throughput bound (no run needed)
# --------------------------------------------------------------------------
# Decode is memory-bandwidth bound: tok/s ~= (aggregate HBM bandwidth * util) /
# (params * dtype_bytes). Every input here is config-known (GPU model, model
# size, dtype, tensor-parallel). This is an ORDER-OF-MAGNITUDE bound to flag an
# unwise config — NOT a guarantee; measured tok/s from telemetry (tokens/wall)
# refines the real numbers afterward.

# Peak HBM bandwidth (TB/s) per GPU. Extend as needed.
_GPU_BW_TBPS: dict[str, float] = {
    "v100": 0.9, "a100": 2.0, "a100-80g": 2.0, "h100": 3.35, "h200": 4.8,
    "a6000": 0.77, "l40": 0.86, "l40s": 0.86, "a40": 0.70, "rtx4090": 1.01,
}

# Fraction of peak bandwidth realistically achieved (leading-order haircut), and
# the extra loss from sharding weights across GPUs over the interconnect.
_BW_UTIL = 0.6
_TP_EFF = 0.85   # per-extra-GPU efficiency for tensor parallelism


def parse_params_billions(model: str) -> float | None:
    """Best-effort model size in billions from the name (e.g. '...-27b' -> 27).
    Returns None if not parseable (caller should then require params_b in cfg)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model or "")
    return float(m.group(1)) if m else None


def estimate_decode_tok_s(params_b: float, gpu_model: str,
                          dtype_bytes: float = 2.0,
                          tensor_parallel: int = 1) -> float | None:
    """Leading-order single-stream decode throughput (tokens/s), or None if the
    GPU is unknown. Deliberately conservative; batching raises real aggregate
    throughput above this per-stream floor."""
    bw = _GPU_BW_TBPS.get((gpu_model or "").lower())
    if bw is None or params_b <= 0:
        return None
    tp = max(1, int(tensor_parallel))
    agg_bw = bw * (1 + (tp - 1) * _TP_EFF) * 1e12   # bytes/s
    bytes_per_token = params_b * 1e9 * dtype_bytes
    return (agg_bw * _BW_UTIL) / bytes_per_token


def serve_throughput_warning(spec: ServeSpec,
                             choke_tok_s: float = 5.0) -> str | None:
    """Config-time nudge: return a loud warning string if the configuration is
    likely to be painfully slow (or can't be estimated), else None. Never
    blocks — the user is told their choice is unwise and may proceed."""
    p = spec.profile
    params = p.params_b if p.params_b else parse_params_billions(spec.model)
    if params is None:
        return (f"Can't estimate throughput for {spec.model!r}: model size "
                "unknown. Set llm_slurm.params_b so the config-time speed check "
                "can run.")
    if not p.gpu_model:
        return (f"Can't estimate throughput: GPU model unknown. Set "
                f"llm_slurm.gpu_model (one of {sorted(_GPU_BW_TBPS)}) to enable "
                "the config-time speed check.")
    est = estimate_decode_tok_s(params, p.gpu_model, p.dtype_bytes,
                                p.tensor_parallel)
    if est is None:
        return (f"Unknown GPU {p.gpu_model!r} (known: {sorted(_GPU_BW_TBPS)}) — "
                "can't run the config-time speed check.")
    if est < choke_tok_s:
        return (f"LIKELY TOO SLOW: {spec.model!r} (~{params:g}B, {p.dtype_bytes:g}"
                f" B/param) on {p.gpu_model} x{p.tensor_parallel} estimates "
                f"~{est:.1f} decode tok/s (leading-order bound) — below the "
                f"{choke_tok_s:g} tok/s floor. Consider a bigger/more GPUs, a "
                "smaller model, or quantization (lower dtype_bytes). This is an "
                "estimate; proceeding anyway.")
    log.info("Config-time throughput estimate for %s on %s x%d: ~%.0f decode "
             "tok/s (leading-order).", spec.model, p.gpu_model,
             p.tensor_parallel, est)
    return None


def submit_serve_job(script_path: str) -> str:
    """sbatch the serve script; return the SLURM job id. Mirrors SlurmExecutor's
    `subprocess.run(["sbatch", ...])` submit."""
    proc = _run(["sbatch", "--parsable", script_path])
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed: {proc.stderr.strip() or proc.stdout}")
    # --parsable prints "<jobid>" (optionally "<jobid>;<cluster>")
    return proc.stdout.strip().split(";")[0].strip()


def _job_state_and_node(jobid: str) -> tuple[str, str]:
    """(state, nodelist) for a job via squeue; ("", "") if not listed."""
    proc = _run(["squeue", "-h", "-j", jobid, "-o", "%T %N"])
    out = (proc.stdout or "").strip()
    if not out:
        return "", ""
    parts = out.split()
    state = parts[0] if parts else ""
    node = parts[1] if len(parts) > 1 else ""
    return state, node


def wait_until_running(jobid: str, timeout: float, poll: float = 10.0,
                       sleep=time.sleep) -> str:
    """Poll squeue until the job is RUNNING; return the allocated node hostname.
    Raises TimeoutError if the allocation is not granted within `timeout` (GPU
    queues can be long — this is a bounded wait with a clear failure)."""
    start = time.monotonic()
    while True:
        state, node = _job_state_and_node(jobid)
        if state == "RUNNING" and node:
            return node
        if state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                     "OUT_OF_MEMORY", "BOOT_FAIL"):
            raise RuntimeError(f"serve job {jobid} entered state {state}")
        if time.monotonic() - start > timeout:
            raise TimeoutError(
                f"serve job {jobid} not RUNNING after {timeout:.0f}s "
                f"(last state {state!r}). GPU queue may be busy — raise "
                "llm_slurm.queue_timeout or check `squeue`.")
        sleep(poll)


def _probe(url: str, timeout: float = 5.0) -> bool:
    """True if the OpenAI-compatible server answers /v1/models."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= getattr(r, "status", 200) < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_until_ready(base_url: str, timeout: float, poll: float = 10.0,
                     sleep=time.sleep) -> None:
    """Poll <base_url>/models until the server answers (cold vLLM load is
    minutes — this is the gate the per-request transient retry cannot cover).
    Raises TimeoutError past `timeout`."""
    models_url = base_url.rstrip("/") + "/models"
    start = time.monotonic()
    while True:
        if _probe(models_url):
            return
        if time.monotonic() - start > timeout:
            raise TimeoutError(
                f"vLLM server at {base_url} did not answer within "
                f"{timeout:.0f}s (model still loading, or unreachable — check "
                "the serve job's log and agent->GPU-node network reachability).")
        sleep(poll)


def cancel_job(jobid: str) -> None:
    """scancel the serve job. Best-effort and idempotent — a leaked GPU
    allocation is expensive, so this must never raise on the teardown path."""
    if not jobid:
        return
    try:
        _run(["scancel", jobid])
    except Exception:  # noqa: BLE001
        log.warning("scancel %s failed — check `squeue` for a leaked allocation",
                    jobid, exc_info=True)


def reap_run_serve_job(run_dir) -> None:
    """scancel the serve job a run persisted, if any. For the study watchdog's
    hard-kill path (``os._exit`` bypasses ``execute()``'s finally, so the serve
    allocation would otherwise leak). Reads ``<run_dir>/debug/serve_job.jobid``;
    a no-op when the file is absent. Never raises."""
    try:
        from pathlib import Path
        jobid_file = Path(run_dir) / "debug" / "serve_job.jobid"
        if jobid_file.exists():
            cancel_job(jobid_file.read_text(encoding="utf-8").strip())
    except Exception:  # noqa: BLE001
        log.warning("reap_run_serve_job(%s) failed", run_dir, exc_info=True)
