"""Framework-owned vLLM-on-SLURM lifecycle (Phase 1) — fully headless.

All external calls (sbatch/squeue/scancel via _run, the HTTP readiness probe)
are stubbed, so this needs no cluster. Covers: model-profile resolution +
config-override precedence, the sbatch script, submit/wait-running/wait-ready
gates, and teardown-never-raises.
"""
from __future__ import annotations

import pytest

from a3dasm._src import settings
from a3dasm._src import slurm_llm as S

# --- profiles / resolution ------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_settings():
    """Clear the global run-knob mapping around every test so the serve-quant
    default (FP8) is deterministic and a test that sets it can't leak."""
    settings.configure({})
    yield
    settings.configure({})


@pytest.fixture
def no_metadata(monkeypatch):
    """Force the metadata layer to return None (offline) so a test exercises the
    name/hint/fallback layers without any network or local-cache dependency."""
    monkeypatch.setattr(S, "fetch_model_metadata", lambda mid: None)


def _meta(params_b, dtype_bytes=2.0):
    def _fake(model_id):
        return S.ModelMeta(params_b=params_b, dtype_bytes=dtype_bytes)
    return _fake


def _set_quant(q):
    settings.configure({"llm_quantization": q})


def test_family_serve_hints_apply_by_basename(no_metadata):
    # basename 'gemma-4-e4b-it' matches the 'gemma-4' serve-hint (context length)
    spec = S.resolve_serve_spec("gemma-4-e4b-it")
    assert "--max-model-len" in spec.profile.vllm_args
    assert spec.profile.gres == "gpu:1"     # no metadata -> conservative default


def test_config_overrides_beat_profile_defaults(no_metadata):
    spec = S.resolve_serve_spec(
        "gemma-4-31b", overrides={"mem": "80G", "time": "12:00:00",
                                  "gres": "gpu:2", "bogus_key": "ignored"})
    assert spec.profile.mem == "80G"
    assert spec.profile.time == "12:00:00"
    assert spec.profile.gres == "gpu:2"
    assert not hasattr(spec.profile, "bogus_key")  # unknown key ignored, no error


def _is_conservative_default(profile):
    # the fallback keeps the default SIZING (vllm_args may carry the serve-quant
    # flag, which applies on every path — so compare sizing, not the whole dc).
    d = S._DEFAULT_PROFILE
    return (profile.gres == d.gres and profile.mem == d.mem
            and profile.tensor_parallel == d.tensor_parallel
            and profile.params_b is None)


def test_unknown_model_falls_back_to_default_and_warns(caplog, no_metadata):
    import logging
    with caplog.at_level(logging.WARNING, logger="a3dasm.slurm_llm"):
        spec = S.resolve_serve_spec("some-unheard-of-model")
    assert _is_conservative_default(spec.profile)
    assert any("no checkpoint metadata" in r.getMessage() for r in caplog.records)


def test_longest_prefix_match_wins(no_metadata):
    spec = S.resolve_serve_spec("llama-3.1-8b")
    assert spec.profile.gres == "gpu:1"


# --- GPU inventory: real Oscar cards, keyed by exact Slurm gres names -----

def test_gpu_table_is_the_oscar_inventory_with_no_a100():
    # No a100 exists on this cluster — sizing must never target one.
    assert "a100" not in S._GPU_VRAM_GB
    assert not any("a100" in k for k in S._GPU_VRAM_GB)
    # exact gres names + capacities the cluster actually schedules
    assert S._GPU_VRAM_GB["nvidia_geforce_rtx_3090"] == 24
    assert S._GPU_VRAM_GB["nvidia_rtx_a5000"] == 24
    assert S._GPU_VRAM_GB["l40s"] == 48
    assert S._GPU_VRAM_GB["h100"] == 80
    assert S._GPU_VRAM_GB["nvidia_b200"] == 180


def test_default_gpu_is_schedulable_on_the_gpu_partition():
    # the zero-config default must be a card that is actually in the inventory
    assert S._DEFAULT_GPU_MODEL == "l40s"
    assert S._DEFAULT_GPU_MODEL in S._GPU_VRAM_GB


# --- metadata-derived sizing (the core of issue #4) -----------------------

def test_namespaced_id_is_sized_from_metadata_not_silently_defaulted(monkeypatch):
    # regression for the org/repo prefix bug: a real namespaced id with a large
    # model must be sized up, NOT silently left at the 1-GPU/32G default.
    _set_quant("bf16")
    monkeypatch.setattr(S, "fetch_model_metadata", _meta(70.0, 2.0))
    spec = S.resolve_serve_spec(
        "meta-llama/Llama-3.1-70B-Instruct", overrides={"gpu_model": "l40s"})
    # 70B bf16 = ~140GB weights * 1.3 / 48GB per L40S -> 4 GPUs; gres names the card
    assert spec.profile.tensor_parallel == 4
    assert spec.profile.gres == "gpu:l40s:4"
    assert spec.profile.params_b == 70.0
    assert spec.profile != S._DEFAULT_PROFILE


def test_zero_config_defaults_to_schedulable_l40s(monkeypatch):
    # no gpu_model AND no quant set -> sizes against the defaults (l40s + FP8),
    # gres is grantable. 70B fp8 = 70GB * 1.3 / 48 -> 2 GPUs.
    monkeypatch.setattr(S, "fetch_model_metadata", _meta(70.0, 2.0))
    spec = S.resolve_serve_spec("org/Big-70B")
    assert spec.profile.gres == "gpu:l40s:2"
    assert spec.profile.tensor_parallel == 2
    assert spec.profile.gpu_model == "l40s"   # recorded, so throughput check runs
    assert spec.profile.dtype_bytes == 1.0    # FP8 default


@pytest.mark.parametrize("gpu,tp", [
    ("l40s", 1),                    # 24GB need / 48GB -> 1
    ("nvidia_rtx_a5000", 1),        # 24GB need / 24GB -> 1
    ("nvidia_geforce_rtx_3090", 1),
])
def test_small_model_stays_one_gpu_on_real_cards(monkeypatch, gpu, tp):
    monkeypatch.setattr(S, "fetch_model_metadata", _meta(7.0, 2.0))
    spec = S.resolve_serve_spec("org/Small-7B", overrides={"gpu_model": gpu})
    assert spec.profile.tensor_parallel == tp
    assert spec.profile.gres == f"gpu:{gpu}:{tp}"


@pytest.mark.parametrize("gpu,tp", [
    ("l40s", 4),                    # 182GB need / 48GB -> 4
    ("nvidia_rtx_a5000", 8),        # 182GB need / 24GB -> 8
    ("nvidia_geforce_rtx_3090", 8),
])
def test_large_model_shards_across_real_cards(monkeypatch, gpu, tp):
    _set_quant("bf16")   # bf16 so the 70B genuinely needs multiple cards
    monkeypatch.setattr(S, "fetch_model_metadata", _meta(70.0, 2.0))
    spec = S.resolve_serve_spec("org/Big-70B", overrides={"gpu_model": gpu})
    assert spec.profile.tensor_parallel == tp
    assert spec.profile.gres == f"gpu:{gpu}:{tp}"


# --- GPU-aware default quant (FP8 only on FP8-capable cards) ---------------

def test_fp8_capable_set_matches_the_cluster_ada_hopper_blackwell():
    assert S._FP8_CAPABLE == frozenset(
        {"l40s", "h100", "nvidia_rtx_pro_6000_blackwell", "nvidia_b200"})
    # Ampere/Turing cards must NOT be in it
    for ampere in ("nvidia_geforce_rtx_3090", "nvidia_rtx_a5000",
                   "quadro_rtx_6000", "nvidia_a40", "nvidia_rtx_a6000"):
        assert ampere not in S._FP8_CAPABLE


def test_default_quant_is_fp8_on_l40s(monkeypatch):
    # no llm_quantization set + FP8-capable card -> FP8 weights.
    monkeypatch.setattr(S, "fetch_model_metadata", _meta(7.0, 2.0))
    spec = S.resolve_serve_spec("org/M-7B", overrides={"gpu_model": "l40s"})
    assert spec.profile.dtype_bytes == 1.0
    assert spec.profile.vllm_args[-2:] == ["--quantization", "fp8"]


def test_default_quant_is_not_fp8_on_ampere_a5000(monkeypatch):
    # no llm_quantization set + non-FP8 card -> safe quant, never FP8.
    monkeypatch.setattr(S, "fetch_model_metadata", _meta(7.0, 2.0))
    spec = S.resolve_serve_spec("org/M-7B", overrides={"gpu_model": "nvidia_rtx_a5000"})
    assert spec.profile.dtype_bytes != 1.0          # NOT fp8
    assert "fp8" not in spec.profile.vllm_args


def test_explicit_fp8_wins_even_on_ampere(monkeypatch):
    # user's explicit choice is respected, GPU-capability notwithstanding.
    _set_quant("fp8")
    monkeypatch.setattr(S, "fetch_model_metadata", _meta(7.0, 2.0))
    spec = S.resolve_serve_spec("org/M-7B", overrides={"gpu_model": "nvidia_rtx_a5000"})
    assert spec.profile.dtype_bytes == 1.0


# --- context-aware KV sizing: gemma-4-31B on one 48GB L40S -----------------

def _gemma_31b_meta(model_id=None):
    # gemma-4-31B config facts: 60 layers = 50 sliding@window1024 + 10 global;
    # global attention kv_heads=4, head_dim=512; 256K positions.
    return S.ModelMeta(
        params_b=30.7, dtype_bytes=2.0, num_layers=60, num_kv_heads=8,
        head_dim=256, sliding_window=1024, num_sliding_layers=50,
        num_global_layers=10, global_kv_heads=4, global_head_dim=512,
        max_position_embeddings=262144)


def test_kv_fields_from_config_parses_gemma_split():
    cfg = {"num_hidden_layers": 60, "num_key_value_heads": 8, "head_dim": 256,
           "sliding_window": 1024, "sliding_window_pattern": 6,
           "num_global_key_value_heads": 4, "global_head_dim": 512,
           "max_position_embeddings": 262144}
    out = S._kv_fields_from_config(cfg)
    assert out["num_layers"] == 60
    assert out["num_global_layers"] == 10 and out["num_sliding_layers"] == 50
    assert out["global_kv_heads"] == 4 and out["global_head_dim"] == 512


def test_gemma_4_31b_fp8_weights_fp8_kv_256k_fits_one_l40s(monkeypatch):
    # computed from config (NOT hardcoded): weight ~31GB + fp8 KV ~11GB + 2GB
    # overhead ~= 44GB < 48GB -> a single L40S. Default quant (FP8) on L40S; the
    # gemma-4 serve-hint supplies --max-model-len 262144 + --kv-cache-dtype fp8.
    monkeypatch.setattr(S, "fetch_model_metadata", _gemma_31b_meta)
    spec = S.resolve_serve_spec("gemma-4-31b", overrides={"gpu_model": "l40s"})
    assert spec.profile.tensor_parallel == 1
    assert spec.profile.gres == "gpu:l40s:1"
    assert spec.profile.dtype_bytes == 1.0                    # FP8 weights
    assert "--max-model-len" in spec.profile.vllm_args        # 256K NOT capped
    assert "262144" in spec.profile.vllm_args
    # the computed requirement really is under one card
    vram, used_fallback = S._estimate_vram_gb(_gemma_31b_meta(), 1.0, 262144, 1.0)
    assert not used_fallback and vram < S._GPU_VRAM_GB["l40s"]


def test_gemma_4_31b_fp8_weights_fp16_kv_256k_needs_two_l40s(monkeypatch):
    # same model + context, but fp16 KV (~22GB) -> ~53GB -> 2 L40S. Drop the
    # fp8 KV hint via a config vllm_args override (keeps the 256K context).
    monkeypatch.setattr(S, "fetch_model_metadata", _gemma_31b_meta)
    spec = S.resolve_serve_spec(
        "gemma-4-31b",
        overrides={"gpu_model": "l40s", "vllm_args": ["--max-model-len", "262144"]})
    assert spec.profile.tensor_parallel == 2
    assert spec.profile.gres == "gpu:l40s:2"
    assert spec.profile.dtype_bytes == 1.0                    # weights still FP8


def test_unknown_gpu_model_derives_mem_but_not_gres(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(S, "fetch_model_metadata", _meta(70.0, 2.0))
    with caplog.at_level(logging.WARNING, logger="a3dasm.slurm_llm"):
        # an explicitly unknown card (e.g. the absent a100) can't yield a count
        spec = S.resolve_serve_spec("org/Big-70B", overrides={"gpu_model": "a100"})
    assert spec.profile.params_b == 70.0
    assert spec.profile.mem != S._DEFAULT_PROFILE.mem       # mem still derived
    assert spec.profile.gres == S._DEFAULT_PROFILE.gres     # gres NOT derived
    assert any("not a known cluster GPU" in r.getMessage() for r in caplog.records)


def test_metadata_fetch_failure_falls_back_and_warns(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(S, "fetch_model_metadata", lambda mid: None)
    with caplog.at_level(logging.WARNING, logger="a3dasm.slurm_llm"):
        spec = S.resolve_serve_spec("gated/Private-Model")
    assert _is_conservative_default(spec.profile)   # never raises, never mis-sizes
    assert any("no checkpoint metadata" in r.getMessage() for r in caplog.records)


def test_config_override_beats_metadata_derived_field(monkeypatch):
    monkeypatch.setattr(S, "fetch_model_metadata", _meta(70.0, 2.0))
    spec = S.resolve_serve_spec(
        "org/Big-70B", overrides={"gpu_model": "l40s", "gres": "gpu:1"})
    assert spec.profile.gres == "gpu:1"   # explicit config wins over derived


# --- metadata source: local-cache-first, then Hub HTTP --------------------

def test_local_cache_preferred_over_network(monkeypatch):
    monkeypatch.setattr(S, "_read_local_metadata",
                        lambda mid: S.ModelMeta(params_b=13.0, dtype_bytes=2.0))
    def _boom(mid, timeout):
        raise AssertionError("network must not be hit when the cache has it")
    monkeypatch.setattr(S, "_fetch_hub_metadata", _boom)
    meta = S.fetch_model_metadata("org/Repo-13B")
    assert meta.params_b == 13.0


def test_hub_fetch_used_when_cache_misses(monkeypatch):
    monkeypatch.setattr(S, "_read_local_metadata", lambda mid: None)
    seen = {}
    def _hub(mid, timeout):
        seen["id"] = mid
        return S.ModelMeta(params_b=8.0, dtype_bytes=2.0)
    monkeypatch.setattr(S, "_fetch_hub_metadata", _hub)
    meta = S.fetch_model_metadata("org/Repo-8B")
    assert meta.params_b == 8.0 and seen["id"] == "org/Repo-8B"


def test_hub_fetch_disabled_by_setting(monkeypatch):
    from a3dasm._src import settings
    monkeypatch.setattr(S, "_read_local_metadata", lambda mid: None)
    monkeypatch.setattr(S, "_fetch_hub_metadata",
                        lambda mid, timeout: (_ for _ in ()).throw(
                            AssertionError("network hit despite fetch disabled")))
    settings.configure({"llm_metadata_fetch": False})
    try:
        assert S.fetch_model_metadata("org/Repo") is None
    finally:
        settings.configure({})


def test_hub_metadata_parses_safetensors_total(monkeypatch):
    class _Resp:
        status_code = 200
        def json(self):
            return {"safetensors": {"total": 8_000_000_000,
                                    "parameters": {"BF16": 8_000_000_000}}}
    monkeypatch.setattr("requests.get", lambda url, timeout: _Resp())
    meta = S._fetch_hub_metadata("org/Repo-8B", timeout=1.0)
    assert round(meta.params_b, 3) == 8.0
    assert meta.dtype_bytes == 2.0


def test_hub_metadata_bare_name_or_error_returns_none(monkeypatch):
    assert S._fetch_hub_metadata("gemma-2", timeout=1.0) is None   # not org/repo
    class _Resp:
        status_code = 401
        def json(self):
            return {}
    monkeypatch.setattr("requests.get", lambda url, timeout: _Resp())
    assert S._fetch_hub_metadata("gated/Repo", timeout=1.0) is None  # gated 401


def test_local_metadata_reads_cache_snapshot(tmp_path, monkeypatch):
    import json
    root = tmp_path / "hub"
    snap = root / "models--org--Repo-7B" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
    (snap / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 14_000_000_000}}))
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    meta = S._read_local_metadata("org/Repo-7B")
    assert meta is not None
    assert round(meta.params_b, 2) == 7.0    # 14GB / 2 B/param / 1e9
    assert meta.dtype_bytes == 2.0


# --- alias expansion (Layer 1, purely optional convenience) ---------------

def test_config_alias_expands_and_wins_over_builtin():
    mid = S.resolve_model_id("gemma-4", {"aliases": {"gemma-4": "me/Custom-4B"}})
    assert mid == "me/Custom-4B"


def test_gemma_4_is_the_primary_builtin_alias():
    # gemma-4 (the cheap-but-SOTA default) is the primary curated alias; the
    # table stays minimal and relies on live metadata for everything else.
    assert "gemma-4" in S.MODEL_ALIASES
    assert S.resolve_model_id("gemma-4") == S.MODEL_ALIASES["gemma-4"]
    assert len(S.MODEL_ALIASES) <= 4   # deliberately tiny, not a model registry


def test_gemma_4_alias_ids_are_the_verified_case_sensitive_strings():
    # cheap default variant, and the max-power-for-48GB variant, exact case.
    assert S.resolve_model_id("gemma-4-e4b") == "google/gemma-4-E4B-it"
    assert S.resolve_model_id("gemma-4-31b") == "google/gemma-4-31B-it"
    assert S.resolve_model_id("gemma-4") == "google/gemma-4-E4B-it"


def test_unknown_name_passes_through_unchanged():
    assert S.resolve_model_id("org/Some-Literal-Id") == "org/Some-Literal-Id"
    assert S.resolve_model_id("/scratch/models/local-ckpt") == \
        "/scratch/models/local-ckpt"


# --- sbatch script --------------------------------------------------------

class _Cluster:
    partition = "gpu"
    account = "acct"
    env_setup = ["module load cuda", "module load vllm"]
    env_vars = {"HF_HOME": "/scratch/hf"}
    runner = "uv run python"


def test_render_serve_script_has_gres_model_port_and_setup():
    spec = S.resolve_serve_spec("gemma-4-27b", overrides={"gres": "gpu:1"})
    script = S.render_serve_script(spec, _Cluster(), base_url_port=8000,
                                   log_dir="/tmp/logs")
    assert "--gres=gpu:1" in script
    assert "#SBATCH -p gpu" in script
    assert "module load cuda" in script
    assert "export HF_HOME=/scratch/hf" in script
    assert "--model gemma-4-27b" in script
    assert "--port 8000" in script
    assert "vllm.entrypoints.openai.api_server" in script


# --- submit / wait-running ------------------------------------------------

def test_submit_parses_jobid(monkeypatch):
    monkeypatch.setattr(S, "_run", lambda cmd, timeout=30.0: _CP(0, "12345;cluster\n", ""))
    assert S.submit_serve_job("/tmp/serve.sh") == "12345"


def test_submit_raises_on_sbatch_failure(monkeypatch):
    monkeypatch.setattr(S, "_run", lambda cmd, timeout=30.0: _CP(1, "", "boom"))
    with pytest.raises(RuntimeError):
        S.submit_serve_job("/tmp/serve.sh")


def test_wait_until_running_returns_node(monkeypatch):
    seq = iter([("PENDING", ""), ("PENDING", ""), ("RUNNING", "gpu042")])
    monkeypatch.setattr(S, "_job_state_and_node", lambda jid: next(seq))
    node = S.wait_until_running("1", timeout=100, poll=0, sleep=lambda _: None)
    assert node == "gpu042"


def test_wait_until_running_times_out(monkeypatch):
    monkeypatch.setattr(S, "_job_state_and_node", lambda jid: ("PENDING", ""))
    with pytest.raises(TimeoutError):
        S.wait_until_running("1", timeout=-1, poll=0, sleep=lambda _: None)


def test_wait_until_running_raises_on_failed_job(monkeypatch):
    monkeypatch.setattr(S, "_job_state_and_node", lambda jid: ("FAILED", ""))
    with pytest.raises(RuntimeError):
        S.wait_until_running("1", timeout=100, poll=0, sleep=lambda _: None)


# --- wait-ready -----------------------------------------------------------

def test_wait_until_ready_polls_past_cold_load(monkeypatch):
    seq = iter([False, False, True])   # 503, 503, then 200
    monkeypatch.setattr(S, "_probe", lambda url, timeout=5.0: next(seq))
    S.wait_until_ready("http://gpu042:8000/v1", timeout=100, poll=0,
                       sleep=lambda _: None)  # returns without raising


def test_wait_until_ready_times_out(monkeypatch):
    monkeypatch.setattr(S, "_probe", lambda url, timeout=5.0: False)
    with pytest.raises(TimeoutError):
        S.wait_until_ready("http://gpu042:8000/v1", timeout=-1, poll=0,
                           sleep=lambda _: None)


# --- teardown -------------------------------------------------------------

def test_cancel_job_never_raises(monkeypatch):
    def boom(cmd, timeout=30.0):
        raise OSError("scancel missing")
    monkeypatch.setattr(S, "_run", boom)
    S.cancel_job("999")   # must not raise
    S.cancel_job("")      # no-op on empty


class _CP:
    def __init__(self, rc, out, err):
        self.returncode, self.stdout, self.stderr = rc, out, err


# --- build-time throughput bound -----------------------------------------

def test_parse_params_billions():
    assert S.parse_params_billions("gemma-4-27b-it") == 27
    assert S.parse_params_billions("llama-3-70b") == 70
    assert S.parse_params_billions("meta/Llama-3.1-8B-Instruct") == 8
    assert S.parse_params_billions("mystery-model") is None


def test_estimate_decode_tok_s_ballpark():
    # 7B fp16 on one A100 (~2 TB/s): order ~85 tok/s single-stream
    est = S.estimate_decode_tok_s(7, "a100", dtype_bytes=2.0, tensor_parallel=1)
    assert 60 < est < 120
    # unknown GPU -> None
    assert S.estimate_decode_tok_s(7, "nonesuch") is None


def test_warning_fires_for_a_slow_config():
    spec = S.resolve_serve_spec("llama-3-70b", overrides={"gpu_model": "v100"})
    msg = S.serve_throughput_warning(spec, choke_tok_s=5.0)
    assert msg is not None and "TOO SLOW" in msg


def test_no_warning_for_a_reasonable_config():
    spec = S.resolve_serve_spec(
        "gemma-4-9b", overrides={"gpu_model": "a100", "params_b": 9})
    assert S.serve_throughput_warning(spec, choke_tok_s=5.0) is None


def test_warning_when_gpu_or_size_unknown():
    # size known (parsed) but no gpu_model -> can't check
    spec = S.resolve_serve_spec("gemma-4-9b")
    assert "GPU model unknown" in (S.serve_throughput_warning(spec) or "")
    # gpu known but size unparseable and not set -> can't check
    spec2 = S.resolve_serve_spec("mystery", overrides={"gpu_model": "a100"})
    assert "model size unknown" in (S.serve_throughput_warning(spec2) or "")


def test_reap_run_serve_job_cancels_persisted_jobid(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(S, "cancel_job", seen.append)
    debug = tmp_path / "debug"
    debug.mkdir()
    (debug / "serve_job.jobid").write_text("4242\n")
    S.reap_run_serve_job(tmp_path)
    assert seen == ["4242"]


def test_reap_run_serve_job_noop_without_file(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(S, "cancel_job", seen.append)
    S.reap_run_serve_job(tmp_path)   # no debug/ at all -> no-op, no raise
    assert seen == []
