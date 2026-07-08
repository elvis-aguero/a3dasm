"""sandbox_env: the environment for running the deliverable/cells against a
sandbox ledger copy must also expose F3DASM_STUDY_ROOT.

Regression for run 20260705T181941 (strategizer DONE FRICTION #2): the repro/
CheckDeliverable sandbox provided only a temp store path with no relationship
to the study repo root, so any pillar cell needing a non-ledger repo resource
(e.g. bo/cei_core.py for the ml surrogate self-check) had to hand-roll
multi-candidate path search. A read-only F3DASM_STUDY_ROOT anchor removes that.
"""
from __future__ import annotations

from a3dasm._src.notebook_exec import sandbox_env


def test_sandbox_env_sets_store_config_and_study_root(tmp_path):
    env = sandbox_env(
        tmp_path / "sb_store", tmp_path / "cfg.json",
        study_root=tmp_path / "study", base={})
    assert env["F3DASM_CANONICAL_STORE"] == str(tmp_path / "sb_store")
    assert env["F3DASM_RUN_CONFIG"] == str(tmp_path / "cfg.json")
    assert env["F3DASM_STUDY_ROOT"] == str(tmp_path / "study")
    assert env["F3DASM_DELEGATION_ID"] == "D999"


def test_sandbox_env_preserves_existing_delegation_id(tmp_path):
    env = sandbox_env(
        tmp_path / "s", tmp_path / "c",
        base={"F3DASM_DELEGATION_ID": "D042"})
    assert env["F3DASM_DELEGATION_ID"] == "D042"


def test_sandbox_env_omits_study_root_when_not_given(tmp_path):
    env = sandbox_env(tmp_path / "s", tmp_path / "c", base={})
    assert "F3DASM_STUDY_ROOT" not in env
