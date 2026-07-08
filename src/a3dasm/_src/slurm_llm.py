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
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("a3dasm.slurm_llm")

# --------------------------------------------------------------------------
# Model-keyed serve profiles
# --------------------------------------------------------------------------


@dataclass
class ServeProfile:
    """Default SLURM + vLLM serving parameters for a model family.

    These are STARTING defaults, tuned per model as they are validated on a real
    cluster; a study's config overrides any field (see resolve_serve_spec). GPU
    is requested via ``gres`` (SLURM ``--gres``). ``partition`` None means "use
    the cluster's configured default partition".
    """
    gres: str = "gpu:1"
    mem: str = "32G"
    cpus_per_task: int = 8
    time: str = "08:00:00"
    partition: str | None = None
    port: int = 8000
    # Extra args appended to `vllm serve <model>` (e.g. context length).
    vllm_args: list[str] = field(default_factory=list)
    # Build-time throughput-estimate inputs (all config-known, no run needed):
    gpu_model: str | None = None      # e.g. "a100" — keys _GPU_BW_TBPS
    dtype_bytes: float = 2.0          # fp16=2, fp8/int8=1, int4=0.5
    tensor_parallel: int = 1          # GPUs the weights are sharded across
    params_b: float | None = None     # model size in billions; None -> parse from name


# Keyed by a lowercase model-name PREFIX (longest match wins). Illustrative
# starting points — extend/tune as models are validated on the cluster; a study
# always overrides via config. Unknown models fall back to _DEFAULT_PROFILE with
# a loud log line naming what to set.
MODEL_SERVE_PROFILES: dict[str, ServeProfile] = {
    "gemma-4": ServeProfile(
        gres="gpu:1", mem="48G", cpus_per_task=8, time="08:00:00",
        vllm_args=["--max-model-len", "8192"]),
    "llama-3": ServeProfile(
        gres="gpu:1", mem="48G", cpus_per_task=8, time="08:00:00",
        vllm_args=["--max-model-len", "8192"]),
    "qwen": ServeProfile(
        gres="gpu:2", mem="80G", cpus_per_task=16, time="08:00:00",
        tensor_parallel=2, vllm_args=["--max-model-len", "16384"]),
}

_DEFAULT_PROFILE = ServeProfile()


@dataclass
class ServeSpec:
    """A resolved plan to serve one model: the model id + the merged profile."""
    model: str
    profile: ServeProfile


def _match_profile(model: str) -> tuple[ServeProfile, bool]:
    """Return (profile, matched). Longest prefix match on the lowercased model
    name; (default, False) if nothing matches."""
    m = (model or "").lower()
    best_key = ""
    for key in MODEL_SERVE_PROFILES:
        if m.startswith(key) and len(key) > len(best_key):
            best_key = key
    if best_key:
        return MODEL_SERVE_PROFILES[best_key], True
    return _DEFAULT_PROFILE, False


_PROFILE_FIELDS = {f.name for f in dataclasses.fields(ServeProfile)}


def resolve_serve_spec(model: str, overrides: dict | None = None) -> ServeSpec:
    """Merge model default -> config overrides -> hard default into a ServeSpec.

    ``overrides`` is the study's ``llm_slurm`` config block; only keys that are
    ServeProfile fields are applied (unknown keys are ignored, not an error).
    """
    base, matched = _match_profile(model)
    if not matched:
        log.warning(
            "No serve profile for model %r — using conservative defaults "
            "(%s, %s, gres=%s, time=%s). Set llm_slurm overrides (gres/mem/"
            "time/vllm_args) in config for this model.",
            model, _DEFAULT_PROFILE.mem, f"{_DEFAULT_PROFILE.cpus_per_task} cpus",
            _DEFAULT_PROFILE.gres, _DEFAULT_PROFILE.time)
    applied = {k: v for k, v in (overrides or {}).items()
               if k in _PROFILE_FIELDS and v is not None}
    profile = dataclasses.replace(base, **applied) if applied else base
    return ServeSpec(model=model, profile=profile)


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
        f"#SBATCH -J vllm-serve",
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

import re as _re

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
    m = _re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model or "")
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
