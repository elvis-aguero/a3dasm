"""Axis 3b — the design namespace flows delegation → worker session env.

A namespace bound on the delegating thread is injected as ``F3DASM_NAMESPACE``
into the worker subprocess, so the worker's plain ``get_evaluator()`` resolves
the right oracle (Axis 3a) without changing the agent's call site. Default is
None → no env var → today's single-study path.
"""
from __future__ import annotations

import json
from pathlib import Path


def test_namespace_threadlocal_roundtrip():
    from a3dasm._src.backends.base import (
        get_namespace,
        set_namespace,
    )

    assert get_namespace() is None  # unset by default
    set_namespace("elliptical_rings")
    assert get_namespace() == "elliptical_rings"
    set_namespace(None)
    assert get_namespace() is None


def test_session_env_injects_namespace(tmp_path, monkeypatch):
    from a3dasm._src.backends.base import (
        set_delegation_id,
        set_namespace,
        set_run_config_path,
    )
    from a3dasm._src.backends.claude import _build_session_env

    rc = tmp_path / "run_config.json"
    rc.write_text(json.dumps({"store_dir": str(tmp_path / "store")}))

    set_delegation_id("D003")
    set_run_config_path(str(rc))
    set_namespace("elliptical_rings")
    try:
        env = _build_session_env()
        assert env["F3DASM_NAMESPACE"] == "elliptical_rings"
        assert env["F3DASM_DELEGATION_ID"] == "D003"
    finally:
        set_namespace(None)
        set_delegation_id(None)
        set_run_config_path(None)


def test_session_env_omits_namespace_when_unset(tmp_path):
    from a3dasm._src.backends.base import (
        set_delegation_id,
        set_namespace,
    )
    from a3dasm._src.backends.claude import _build_session_env

    set_namespace(None)
    set_delegation_id("D003")
    try:
        env = _build_session_env()
        assert "F3DASM_NAMESPACE" not in env  # default path untouched
    finally:
        set_delegation_id(None)


def test_delegation_dataclass_has_namespace_default_none():
    from a3dasm._src.graph_state import Delegation

    d = Delegation(target="implementer", task="optimize")
    assert d.namespace is None
    d2 = Delegation(target="datagenerator", task="build oracle",
                    namespace="elliptical_rings")
    assert d2.namespace == "elliptical_rings"


def test_namespace_capability_is_documented_for_the_strategizer():
    """The strategizer must KNOW it can open a namespace (else Axis 3 is dead
    code from the agent's view) — but framed as optional, not a routine step."""
    from a3dasm._src.agents.strategizer import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    low = STRATEGIZER_SYSTEM_PROMPT.lower()
    assert "namespace" in low
    assert "optional" in low  # presented as a capability, not an obligation

