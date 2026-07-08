"""Hard per-delegation RAM cap resolution: config > env > SLURM > default.

The watchdog cap must reflect the real HPC budget. Previously it read only
config.yaml `mem_cap` (else a 4 GB default) despite a comment claiming env +
SLURM support, so on SLURM it kept a small hardcoded ceiling that silently
throttled worker concurrency (run 20260706T204732: ~3x wall-clock, D005).
"""
from __future__ import annotations

from a3dasm._src.agent_runtime import (
    DEFAULT_MEM_CAP_BYTES,
    resolve_mem_cap_bytes,
)

GB = 1024 ** 3
MB = 1024 ** 2


def test_explicit_config_wins():
    assert resolve_mem_cap_bytes(12 * GB, env={"F3DASM_MEM_CAP": str(8 * GB),
                                                "SLURM_MEM_PER_NODE": "4096"}) == 12 * GB


def test_env_beats_slurm_and_default():
    assert resolve_mem_cap_bytes(None, env={"F3DASM_MEM_CAP": str(8 * GB),
                                             "SLURM_MEM_PER_NODE": "4096"}) == 8 * GB


def test_slurm_per_node_used_when_no_config_or_env():
    assert resolve_mem_cap_bytes(None, env={"SLURM_MEM_PER_NODE": "131072"}) == 131072 * MB


def test_slurm_per_cpu_times_cpus():
    env = {"SLURM_MEM_PER_CPU": "4096", "SLURM_CPUS_ON_NODE": "16(x2)"}
    assert resolve_mem_cap_bytes(None, env=env) == 4096 * 16 * MB


def test_falls_back_to_default():
    assert resolve_mem_cap_bytes(None, env={}) == DEFAULT_MEM_CAP_BYTES
    assert resolve_mem_cap_bytes("garbage", env={}) == DEFAULT_MEM_CAP_BYTES
