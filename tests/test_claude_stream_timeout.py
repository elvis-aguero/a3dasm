"""Idle-stream timeout for the Claude backend.

A stalled API response (an ESTABLISHED-but-silent connection that streams
nothing and never errors) must become a transient TimeoutError so the existing
retry_on_transient retries it — rather than hanging the run forever. The timeout
is an IDLE timeout (reset per message) and CONSERVATIVE, so a legitimately long
single-message generation is never false-positived.
"""
from __future__ import annotations

import asyncio

import pytest

from a3dasm._src.backends.claude import (
    _anext_or_done,
    _STREAM_DONE,
    _stream_with_idle_timeout,
)


class _CleanGen:
    """Yields n messages then stops — a normal stream."""

    def __init__(self, n: int) -> None:
        self._n, self._i = n, 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= self._n:
            raise StopAsyncIteration
        self._i += 1
        return f"m{self._i}"


class _StallGen:
    """Yields one message, then stalls forever (the hang we observed)."""

    def __init__(self) -> None:
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._sent:
            self._sent = True
            return "first"
        await asyncio.sleep(3600)  # silent ESTABLISHED stream


def test_clean_stream_completes():
    async def run():
        return [m async for m in _stream_with_idle_timeout(_CleanGen(3), 5.0)]
    assert asyncio.run(run()) == ["m1", "m2", "m3"]


def test_stall_raises_transient_timeout():
    async def run():
        got = []
        with pytest.raises(TimeoutError):
            async for m in _stream_with_idle_timeout(_StallGen(), 0.2):
                got.append(m)
        return got
    got = asyncio.run(run())
    assert got == ["first"]  # delivered what arrived, then tripped on the stall


def test_timeouterror_is_transient_so_retry_catches_it():
    # The whole point: the raised error must be retryable by retry_on_transient.
    from a3dasm._src.backends.base import is_transient_error
    assert is_transient_error(TimeoutError("Anthropic stream stalled"))


def test_anext_or_done_sentinel_on_exhaustion():
    async def run():
        return await _anext_or_done(_CleanGen(0))
    assert asyncio.run(run()) is _STREAM_DONE


# --- Model-generation scoping ------------------------------------------------
# The tight idle window must measure silence ONLY while awaiting model tokens.
# While a TOOL is executing (a worker running a multi-minute Bash script), the
# stream is legitimately silent — that silence must NOT trip the timeout, or it
# false-fires and (via retry) re-runs the whole worker. classify(msg) reports:
#   True  -> a tool is now executing (suspend the tight window)
#   False -> generation/tool-result (re-arm the tight window)
#   None  -> leave the phase unchanged.

def _phase_by_marker(msg):
    if msg == "tool_use":
        return True
    if msg in ("tool_result", "stream"):
        return False
    return None


class _ToolThenSilentGen:
    """Emits a tool_use, then goes silent far past idle_timeout (tool running),
    then finishes. Mimics a worker stuck in a long Bash compute job."""

    def __init__(self, silence: float) -> None:
        self._step, self._silence = 0, silence

    def __aiter__(self):
        return self

    async def __anext__(self):
        self._step += 1
        if self._step == 1:
            return "tool_use"
        if self._step == 2:
            await asyncio.sleep(self._silence)  # tool executing — legit silence
            return "tool_result"
        raise StopAsyncIteration


class _ToolReturnsThenStallsGen:
    """tool_use -> tool_result -> then GENERATION goes silent forever.
    Post-tool generation silence must still trip (a real stall)."""

    def __init__(self) -> None:
        self._step = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self._step += 1
        if self._step == 1:
            return "tool_use"
        if self._step == 2:
            return "tool_result"
        await asyncio.sleep(3600)  # stalled API stream during generation


def test_tool_execution_silence_does_not_trip():
    # idle window 0.1s, but the tool is "running" for 0.3s — must NOT raise
    # because tool_timeout is uncapped (0) while a tool is pending.
    async def run():
        return [
            m async for m in _stream_with_idle_timeout(
                _ToolThenSilentGen(0.3), 0.1,
                tool_timeout=0.0, classify=_phase_by_marker,
            )
        ]
    assert asyncio.run(run()) == ["tool_use", "tool_result"]


def test_generation_stall_after_tool_returns_still_trips():
    async def run():
        got = []
        with pytest.raises(TimeoutError):
            async for m in _stream_with_idle_timeout(
                _ToolReturnsThenStallsGen(), 0.2,
                tool_timeout=0.0, classify=_phase_by_marker,
            ):
                got.append(m)
        return got
    # tool_use + tool_result delivered; the post-tool generation silence trips.
    assert asyncio.run(run()) == ["tool_use", "tool_result"]


def test_no_classify_keeps_tight_window_always():
    # Backward compatible: without classify, any silence trips (old behavior).
    async def run():
        with pytest.raises(TimeoutError):
            async for _ in _stream_with_idle_timeout(_StallGen(), 0.2):
                pass
    asyncio.run(run())
