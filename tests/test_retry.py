"""Tests for transient-error retry in backend adapters."""
import threading
import time

import pytest

from a3dasm._src.backends.base import (
    is_transient_error,
    retry_on_transient,
)


@pytest.mark.parametrize("msg", [
    "Error: overloaded_error", "HTTP 429 Too Many Requests",
    "503 Service Unavailable", "Read timed out", "Connection error",
    "rate limit exceeded",
])
def test_transient_messages_are_transient(msg):
    assert is_transient_error(RuntimeError(msg)) is True


@pytest.mark.parametrize("msg", [
    "401 Unauthorized", "400 invalid request", "tool 'Bash' not found",
    "KeyError: 'foo'",
])
def test_nontransient_messages_are_not(msg):
    assert is_transient_error(RuntimeError(msg)) is False


def test_timeout_and_connection_types_are_transient():
    assert is_transient_error(TimeoutError("x")) is True
    assert is_transient_error(ConnectionError("x")) is True


def test_retry_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("overloaded_error")
        return "ok"

    assert retry_on_transient(flaky, max_attempts=5, base_delay=0.01) == "ok"
    assert calls["n"] == 3


def test_retry_reraises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)

    def always_429():
        raise RuntimeError("HTTP 429")

    with pytest.raises(RuntimeError, match="429"):
        retry_on_transient(always_429, max_attempts=3, base_delay=0.01)


def test_retry_does_not_retry_nontransient(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def bad_auth():
        calls["n"] += 1
        raise RuntimeError("401 Unauthorized")

    with pytest.raises(RuntimeError, match="401"):
        retry_on_transient(bad_auth, max_attempts=5, base_delay=0.01)
    assert calls["n"] == 1


def test_ollama_invoke_retries_invoke_once(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    from a3dasm._src.backends.ollama import OllamaAdapter
    a = OllamaAdapter.__new__(OllamaAdapter)  # bypass __init__/server
    a._lock = threading.Lock()
    calls = {"n": 0}

    def flaky(_msgs):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("503 Service Unavailable")
        return "done"

    monkeypatch.setattr(a, "_invoke_once", flaky)
    assert a.invoke([{"role": "user", "content": "hi"}]) == "done"
    assert calls["n"] == 2


def test_claude_invoke_retries(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_: None)
    from a3dasm._src.backends import claude as cl
    a = cl.ClaudeAdapter.__new__(cl.ClaudeAdapter)
    a._lock = threading.Lock()
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("overloaded_error")
        return "done"

    monkeypatch.setattr(cl, "_run_async_safe", lambda _coro: flaky())
    monkeypatch.setattr(a, "ainvoke", lambda _m, **_k: None)  # coro arg unused
    assert a.invoke([{"role": "user", "content": "hi"}]) == "done"
    assert calls["n"] == 2
