"""DEBUG flag + transcript capture (F3DASM_DEBUG, off by default).

The capture must be a strict no-op unless F3DASM_DEBUG is on AND a per-thread
sink is set, and it must never raise into the agent loop.
"""
from __future__ import annotations

import json

from a3dasm._src.backends.base import (
    append_transcript,
    debug_enabled,
    get_transcript_sink,
    set_transcript_sink,
)


def test_debug_disabled_by_default(monkeypatch):
    monkeypatch.delenv("F3DASM_DEBUG", raising=False)
    assert debug_enabled() is False


def test_debug_enabled_truthy_values(monkeypatch):
    for v in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("F3DASM_DEBUG", v)
        assert debug_enabled() is True
    for v in ("", "0", "false", "no"):
        monkeypatch.setenv("F3DASM_DEBUG", v)
        assert debug_enabled() is False


def test_append_noop_when_debug_off(tmp_path, monkeypatch):
    monkeypatch.delenv("F3DASM_DEBUG", raising=False)
    sink = tmp_path / "t.jsonl"
    set_transcript_sink(str(sink))
    append_transcript({"type": "assistant", "text": "hello"})
    assert not sink.exists()  # nothing written while debug is off


def test_append_noop_when_no_sink(monkeypatch):
    monkeypatch.setenv("F3DASM_DEBUG", "1")
    set_transcript_sink(None)
    # Must not raise even though debug is on but no sink is set.
    append_transcript({"type": "assistant", "text": "hello"})


def test_append_writes_jsonl_when_on(tmp_path, monkeypatch):
    monkeypatch.setenv("F3DASM_DEBUG", "1")
    sink = tmp_path / "sub" / "D001.jsonl"  # parent dir auto-created
    set_transcript_sink(str(sink))
    append_transcript({"type": "assistant", "text": "thinking...",
                       "tools": [{"name": "Bash", "input": {"cmd": "ls"}}]})
    append_transcript({"type": "tool_result", "results": [{"content": "ok"}]})
    set_transcript_sink(None)

    lines = sink.read_text().strip().splitlines()
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["type"] == "assistant"
    assert rec0["tools"][0]["name"] == "Bash"
    assert "ts" in rec0  # stamped
    assert json.loads(lines[1])["type"] == "tool_result"


def test_append_never_raises_on_bad_sink(monkeypatch):
    monkeypatch.setenv("F3DASM_DEBUG", "1")
    # A path under a file (cannot be a dir) — write fails, must be swallowed.
    set_transcript_sink("/dev/null/cannot/exist.jsonl")
    append_transcript({"type": "assistant", "text": "x"})  # no raise
    set_transcript_sink(None)


def test_sink_is_thread_local():
    import threading
    set_transcript_sink("main.jsonl")
    seen = {}

    def worker():
        seen["before"] = get_transcript_sink()  # not the main thread's sink
        set_transcript_sink("worker.jsonl")
        seen["after"] = get_transcript_sink()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["before"] is None  # isolated per thread
    assert seen["after"] == "worker.jsonl"
    assert get_transcript_sink() == "main.jsonl"  # main unchanged
    set_transcript_sink(None)
