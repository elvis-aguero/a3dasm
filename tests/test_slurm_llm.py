"""Framework-owned vLLM-on-SLURM lifecycle (Phase 1) — fully headless.

All external calls (sbatch/squeue/scancel via _run, the HTTP readiness probe)
are stubbed, so this needs no cluster. Covers: model-profile resolution +
config-override precedence, the sbatch script, submit/wait-running/wait-ready
gates, and teardown-never-raises.
"""
from __future__ import annotations

import pytest

from a3dasm._src import slurm_llm as S


# --- profiles / resolution ------------------------------------------------

def test_known_model_uses_its_profile():
    spec = S.resolve_serve_spec("gemma-4-27b-it")
    assert spec.profile.gres == "gpu:1"
    assert "--max-model-len" in spec.profile.vllm_args


def test_config_overrides_beat_profile_defaults():
    spec = S.resolve_serve_spec(
        "gemma-4-27b", overrides={"mem": "80G", "time": "12:00:00",
                                  "gres": "gpu:2", "bogus_key": "ignored"})
    assert spec.profile.mem == "80G"
    assert spec.profile.time == "12:00:00"
    assert spec.profile.gres == "gpu:2"
    assert not hasattr(spec.profile, "bogus_key")  # unknown key ignored, no error


def test_unknown_model_falls_back_to_default_and_warns(caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="a3dasm.slurm_llm"):
        spec = S.resolve_serve_spec("some-unheard-of-model")
    assert spec.profile == S._DEFAULT_PROFILE
    assert any("No serve profile" in r.getMessage() for r in caplog.records)


def test_longest_prefix_match_wins():
    # both "llama-3" registered; a longer more-specific key would win if present
    spec = S.resolve_serve_spec("llama-3.1-8b")
    assert spec.profile.gres == "gpu:1"


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
