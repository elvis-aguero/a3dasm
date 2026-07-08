"""Tests for the multi-namespace oracle resolution layer (Axis 3a).

A "design namespace" lets a run carry more than one oracle + ledger — one per
design parametrization the agent invents. The layer is ADDITIVE: with no
``oracles`` block and no namespace, ``get_evaluator()`` resolves the flat keys
exactly as before (single-study path unchanged).

Resolution rules:
- ``get_evaluator()`` with no arg and no ``F3DASM_NAMESPACE`` env → flat keys (today).
- ``get_evaluator("ns")`` or ``F3DASM_NAMESPACE=ns`` → ``run_config["oracles"]["ns"]``
  overlaid onto the base config (study_dir / budgets inherited; oracle + store keys
  taken wholesale from the namespace block so a base lookup can't leak).
- unknown namespace → ValueError naming the available ones.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from f3dasm._src.design.domain import Domain
from f3dasm._src.experimentdata import ExperimentData
from f3dasm._src.experimentsample import ExperimentSample, JobStatus

from tests.test_evaluator_resolution import (
    _make_delegation_dir,
    _write_run_config,
)


def _sample(**inputs) -> ExperimentSample:
    return ExperimentSample(
        _input_data=dict(inputs), _output_data={}, job_status=JobStatus.OPEN
    )


def _write_evaluator(study_dir: Path, name: str, body: str) -> None:
    (study_dir / name).write_text(textwrap.dedent(body))


def _add_oracle_block(cfg_path: Path, namespace: str, block: dict) -> None:
    """Insert ``oracles[namespace] = block`` into an existing run_config.json."""
    cfg = json.loads(cfg_path.read_text())
    cfg.setdefault("oracles", {})[namespace] = block
    cfg_path.write_text(json.dumps(cfg))


# ---------------------------------------------------------------------------
# Back-compat: no namespace, no oracles block → today's exact behavior.
# ---------------------------------------------------------------------------


def test_default_namespace_resolves_flat_keys(tmp_path, monkeypatch):
    from a3dasm._src.instrumented import get_evaluator

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    _write_evaluator(
        study_dir, "tiny.py",
        "def f_kw(**kwargs):\n    return float(sum(kwargs.values()))\n",
    )
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir, delegation_dir = _make_delegation_dir(tmp_path)
    _write_run_config(
        debug_dir, store_dir, study_dir,
        evaluator_entrypoint="tiny.py:f_kw", evaluator_output_names=["f"],
    )

    monkeypatch.chdir(delegation_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)
    monkeypatch.delenv("F3DASM_NAMESPACE", raising=False)

    out = get_evaluator().execute(_sample(x0=0.3, x1=0.2))
    assert out._output_data["f"] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# A named namespace resolves its own oracle + writes its own store.
# ---------------------------------------------------------------------------


def test_named_namespace_resolves_own_oracle(tmp_path, monkeypatch):
    from a3dasm._src.instrumented import get_evaluator

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    # default oracle: sum; namespace 'ell' oracle: product.
    _write_evaluator(study_dir, "base.py",
                     "def s(**k):\n    return float(sum(k.values()))\n")
    _write_evaluator(study_dir, "ell.py",
                     "def p(**k):\n    v=1.0\n    "
                     "for x in k.values():\n        v*=x\n    return float(v)\n")
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    ell_store = tmp_path / "store_ell"
    ell_store.mkdir()
    debug_dir, delegation_dir = _make_delegation_dir(tmp_path)
    _write_run_config(
        debug_dir, store_dir, study_dir,
        evaluator_entrypoint="base.py:s", evaluator_output_names=["f"],
    )
    _add_oracle_block(
        debug_dir / "run_config.json", "ell",
        {
            "store_dir": str(ell_store),
            "lock_path": str(ell_store / "experiment_data" / ".lock"),
            "evaluator_entrypoint": "ell.py:p",
            "evaluator_output_names": ["f"],
        },
    )

    monkeypatch.chdir(delegation_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)
    monkeypatch.delenv("F3DASM_NAMESPACE", raising=False)

    # default → sum
    assert get_evaluator().execute(_sample(a=3.0, b=4.0)
                                   )._output_data["f"] == pytest.approx(7.0)
    # 'ell' → product
    assert get_evaluator("ell").execute(_sample(a=3.0, b=4.0)
                                        )._output_data["f"] == pytest.approx(12.0)

    # Stores are isolated: each ledger holds only its own row.
    assert len(ExperimentData.from_file(project_dir=store_dir)) == 1
    assert len(ExperimentData.from_file(project_dir=ell_store)) == 1


# ---------------------------------------------------------------------------
# The namespace can arrive via env (so the agent's call site stays get_evaluator()).
# ---------------------------------------------------------------------------


def test_namespace_from_env_var(tmp_path, monkeypatch):
    from a3dasm._src.instrumented import get_evaluator

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    _write_evaluator(study_dir, "base.py",
                     "def s(**k):\n    return float(sum(k.values()))\n")
    _write_evaluator(study_dir, "ell.py",
                     "def p(**k):\n    v=1.0\n    "
                     "for x in k.values():\n        v*=x\n    return float(v)\n")
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    ell_store = tmp_path / "store_ell"
    ell_store.mkdir()
    debug_dir, delegation_dir = _make_delegation_dir(tmp_path)
    _write_run_config(
        debug_dir, store_dir, study_dir,
        evaluator_entrypoint="base.py:s", evaluator_output_names=["f"],
    )
    _add_oracle_block(
        debug_dir / "run_config.json", "ell",
        {
            "store_dir": str(ell_store),
            "lock_path": str(ell_store / "experiment_data" / ".lock"),
            "evaluator_entrypoint": "ell.py:p",
            "evaluator_output_names": ["f"],
        },
    )

    monkeypatch.chdir(delegation_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)
    monkeypatch.setenv("F3DASM_NAMESPACE", "ell")

    # no explicit arg, but env routes to 'ell' → product
    assert get_evaluator().execute(_sample(a=3.0, b=4.0)
                                   )._output_data["f"] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Unknown namespace → clear error naming the available ones.
# ---------------------------------------------------------------------------


def test_unknown_namespace_raises(tmp_path, monkeypatch):
    from a3dasm._src.instrumented import get_evaluator

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    _write_evaluator(study_dir, "base.py",
                     "def s(**k):\n    return float(sum(k.values()))\n")
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    debug_dir, delegation_dir = _make_delegation_dir(tmp_path)
    _write_run_config(
        debug_dir, store_dir, study_dir,
        evaluator_entrypoint="base.py:s", evaluator_output_names=["f"],
    )
    _add_oracle_block(
        debug_dir / "run_config.json", "ell",
        {"store_dir": str(tmp_path / "se"),
         "evaluator_entrypoint": "base.py:s", "evaluator_output_names": ["f"]},
    )

    monkeypatch.chdir(delegation_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)
    monkeypatch.delenv("F3DASM_NAMESPACE", raising=False)

    with pytest.raises(ValueError, match="ell"):
        get_evaluator("does_not_exist")


# ---------------------------------------------------------------------------
# A base lookup must NOT leak into a namespace whose oracle is an entrypoint.
# ---------------------------------------------------------------------------


def test_namespace_oracle_keys_taken_wholesale(tmp_path, monkeypatch):
    """If the base config uses a lookup but the namespace declares an entrypoint,
    the namespace must resolve the entrypoint — the base lookup must not win
    (load_inner_evaluator checks lookup first)."""
    from a3dasm._src.instrumented import get_evaluator
    from a3dasm._src.lookup import LookupDataGenerator

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    # base: a lookup pool
    pool_project = study_dir / "pool"
    pool_project.mkdir()
    dom = Domain()
    dom.add_float("a", 0.0, 1.0)
    dom.add_output("z", exist_ok=True)
    ExperimentData.from_data(
        data={0: ExperimentSample(_input_data={"a": 0.0},
                                  _output_data={"z": 9.0},
                                  job_status=JobStatus.FINISHED)},
        domain=dom,
    ).store(project_dir=pool_project)
    _write_evaluator(study_dir, "ell.py",
                     "def p(**k):\n    return float(sum(k.values()))\n")

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    ell_store = tmp_path / "store_ell"
    ell_store.mkdir()
    debug_dir, delegation_dir = _make_delegation_dir(tmp_path)
    _write_run_config(
        debug_dir, store_dir, study_dir,
        evaluator_lookup={"pool": "pool", "input_columns": ["a"],
                          "output_columns": ["z"]},
    )
    _add_oracle_block(
        debug_dir / "run_config.json", "ell",
        {
            "store_dir": str(ell_store),
            "lock_path": str(ell_store / "experiment_data" / ".lock"),
            "evaluator_entrypoint": "ell.py:p",
            "evaluator_output_names": ["f"],
        },
    )

    monkeypatch.chdir(delegation_dir)
    monkeypatch.delenv("F3DASM_DELEGATION_ID", raising=False)
    monkeypatch.delenv("F3DASM_NAMESPACE", raising=False)

    gen = get_evaluator("ell")
    assert not isinstance(gen.inner, LookupDataGenerator)
    assert gen.execute(_sample(a=2.0, b=5.0)
                       )._output_data["f"] == pytest.approx(7.0)
