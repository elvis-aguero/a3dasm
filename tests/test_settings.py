"""Central run-knob accessor: config.yaml is source of truth, env overrides.

Precedence per knob: env F3DASM_<KEY> > configured config.yaml value > default.
"""
from __future__ import annotations

import pytest

from a3dasm._src import settings


@pytest.fixture(autouse=True)
def _clean_settings(monkeypatch):
    # Each test starts from an empty config and no relevant env vars.
    settings.configure({})
    for var in ("F3DASM_FOO", "F3DASM_FLAG", "F3DASM_N", "F3DASM_X"):
        monkeypatch.delenv(var, raising=False)
    yield
    settings.configure({})


def test_default_when_unset():
    assert settings.get_int("n", 7) == 7
    assert settings.get_float("x", 1.5) == 1.5
    assert settings.get_bool("flag", True) is True
    assert settings.get_str("foo", "d") == "d"


def test_config_value_used_over_default():
    settings.configure({"n": 12, "x": 2.5, "flag": False, "foo": "bar"})
    assert settings.get_int("n", 7) == 12
    assert settings.get_float("x", 1.5) == 2.5
    assert settings.get_bool("flag", True) is False
    assert settings.get_str("foo", "d") == "bar"


def test_env_overrides_config(monkeypatch):
    settings.configure({"n": 12, "flag": False})
    monkeypatch.setenv("F3DASM_N", "99")
    monkeypatch.setenv("F3DASM_FLAG", "true")
    assert settings.get_int("n", 7) == 99       # env beats config
    assert settings.get_bool("flag", True) is True


def test_env_overrides_default_when_no_config(monkeypatch):
    monkeypatch.setenv("F3DASM_X", "3.25")
    assert settings.get_float("x", 1.0) == 3.25


def test_bool_accepts_yaml_native_and_string_forms():
    settings.configure({"flag": True})
    assert settings.get_bool("flag", False) is True
    settings.configure({"flag": "on"})
    assert settings.get_bool("flag", False) is True
    settings.configure({"flag": "0"})
    assert settings.get_bool("flag", True) is False


def test_blank_env_falls_back_to_default_for_numbers(monkeypatch):
    monkeypatch.setenv("F3DASM_N", "   ")
    assert settings.get_int("n", 5) == 5  # blank string is not a number


def test_int_tolerates_float_like_string():
    settings.configure({"n": "12.0"})
    assert settings.get_int("n", 0) == 12


def test_configure_replaces_not_merges():
    settings.configure({"n": 1})
    settings.configure({"x": 2.0})
    assert settings.get_int("n", 0) == 0  # prior key gone
    assert settings.get_float("x", 0.0) == 2.0
