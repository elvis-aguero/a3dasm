"""Commit 1: eval_budget + mem_cap are plumbed into run_config.json + env so the
in-process governor (at the eval boundary) can read them — they previously lived
only in graph state, invisible to a campaign process."""
from __future__ import annotations

import json
import os

from a3dasm._src.agent_runtime import (
    DEFAULT_MEM_CAP_BYTES,
    _init_canonical_store,
)


def test_run_config_carries_eval_budget_and_mem_cap(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "debug").mkdir(parents=True)
    study_dir = tmp_path / "study"
    study_dir.mkdir()

    cfg = _init_canonical_store(
        run_dir, study_dir, evaluator_config=None,
        eval_budget=1000, mem_cap_bytes=DEFAULT_MEM_CAP_BYTES,
    )
    # The knobs travel via run_config.json (sourced from config.yaml) — the single
    # channel. Both the returned dict and the on-disk sidecar carry them.
    assert cfg["eval_budget"] == 1000
    assert cfg["mem_cap_bytes"] == DEFAULT_MEM_CAP_BYTES
    on_disk = json.loads((run_dir / "debug" / "run_config.json").read_text())
    assert on_disk["eval_budget"] == 1000
    assert on_disk["mem_cap_bytes"] == DEFAULT_MEM_CAP_BYTES


def test_config_is_not_exported_to_env(tmp_path):
    # Config is explicit in config.yaml, NEVER through env vars (policy). The
    # init must not export F3DASM_* — the run_config.json sidecar is the channel.
    run_dir = tmp_path / "run"
    (run_dir / "debug").mkdir(parents=True)
    _init_canonical_store(
        run_dir, tmp_path / "s", eval_budget=1000,
        mem_cap_bytes=DEFAULT_MEM_CAP_BYTES,
    )
    assert "F3DASM_EVAL_BUDGET" not in os.environ
    assert "F3DASM_MEM_CAP" not in os.environ


def test_default_mem_cap_is_sane():
    # 4 GiB: above a healthy campaign, below the runaway blowup.
    assert DEFAULT_MEM_CAP_BYTES == 4 * 1024 ** 3
