"""Async-able literature tool pool: wait=False (default) fans external-provider
calls out concurrently across providers; same-provider calls serialize;
CollectSearches gathers results. Mirrors Delegate(wait=False).
"""
from __future__ import annotations

import inspect
import threading
import time

from a3dasm._src.agents.literature import _make_search_async_pool


def _overlap_probe():
    """Returns (fn, peak) where fn sleeps and peak['max'] tracks max concurrency."""
    peak = {"now": 0, "max": 0}
    lock = threading.Lock()

    def fn(query):
        with lock:
            peak["now"] += 1
            peak["max"] = max(peak["max"], peak["now"])
        time.sleep(0.25)
        with lock:
            peak["now"] -= 1
        return f"result:{query}"

    return fn, peak


def test_default_is_sync():
    # Default wait=True: returns the result inline (backward-compatible contract).
    asyncable, _ = _make_search_async_pool()
    f = asyncable("openalex", lambda query: f"R:{query}")
    out = f(query="x")  # no wait arg → default True → sync
    assert out == "R:x"


def test_wait_true_runs_sync_inline():
    asyncable, _ = _make_search_async_pool()
    f = asyncable("openalex", lambda query: f"R:{query}")
    assert f(query="x", wait=True) == "R:x"


def test_wait_false_returns_handle_then_collect():
    asyncable, collect = _make_search_async_pool()
    f = asyncable("openalex", lambda query: f"R:{query}")
    h = f(query="x", wait=False)
    assert "Started async" in h and "openalex#" in h
    out = collect()  # no handle → collect all
    assert "R:x" in out


def test_collect_specific_handle():
    asyncable, collect = _make_search_async_pool()
    f = asyncable("arxiv", lambda query: f"R:{query}")
    msg = f(query="y", wait=False)
    handle = msg.split("handle ")[1].split(")")[0]
    out = collect(handle)
    assert "R:y" in out


def test_wait_string_false_coerces_to_async():
    # MCP string-in: "false" must mean async, not truthy-string → sync.
    asyncable, collect = _make_search_async_pool()
    f = asyncable("openalex", lambda query: f"R:{query}")
    h = f(query="x", wait="false")
    assert "Started async" in h


def test_wait_string_true_is_sync():
    asyncable, _ = _make_search_async_pool()
    f = asyncable("openalex", lambda query: f"R:{query}")
    assert f(query="x", wait="true") == "R:x"


def test_same_provider_calls_serialize():
    asyncable, collect = _make_search_async_pool()
    fn, peak = _overlap_probe()
    f = asyncable("openalex", fn)
    f(query="a", wait=False)
    f(query="b", wait=False)  # both async, SAME provider
    collect()
    assert peak["max"] == 1, "same-provider calls must not overlap"


def test_different_providers_run_concurrently():
    asyncable, collect = _make_search_async_pool()
    fn, peak = _overlap_probe()
    f1 = asyncable("openalex", fn)
    f2 = asyncable("arxiv", fn)
    f1(query="a", wait=False)
    f2(query="b", wait=False)  # different providers
    collect()
    assert peak["max"] == 2, "different providers must run concurrently"


def test_signature_exposes_wait_and_original_params():
    asyncable, _ = _make_search_async_pool()
    f = asyncable("openalex", lambda query, top_k=5: query)
    params = inspect.signature(f).parameters
    assert "query" in params and "top_k" in params and "wait" in params
    assert params["wait"].annotation is bool


def test_async_error_captured_in_collect():
    asyncable, collect = _make_search_async_pool()
    def boom(query):
        raise ValueError("api down")
    f = asyncable("openalex", boom)
    f(query="x", wait=False)
    out = collect()
    assert "ERROR" in out and "api down" in out


def test_collect_with_nothing_pending():
    _, collect = _make_search_async_pool()
    assert "No async searches pending" in collect()


def test_literature_agent_registers_collect_and_async_tools(tmp_path):
    # End-to-end wiring: build_closure_tools wraps providers + adds CollectSearches.
    from a3dasm._src.agents.literature import LiteratureReviewAgent
    tools = LiteratureReviewAgent().build_closure_tools(study_dir=str(tmp_path))
    assert "CollectSearches" in tools
    # a wrapped provider tool exposes `wait`
    if "search_openalex" in tools:
        assert "wait" in inspect.signature(tools["search_openalex"]).parameters
