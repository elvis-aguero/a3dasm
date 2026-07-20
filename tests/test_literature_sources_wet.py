"""Wet integration test: all three literature sources, real network.

Simulates how a real literature-reviewer agent actually calls these tools —
fire all three searches wait=False (async, fanned out concurrently across
providers, same-provider calls serialize), then CollectSearches() once to
gather everything — rather than three independent blocking calls, which is
not the pattern the agent's own tool docstrings tell it to use.

Marked `integration` — NOT run in CI (no `SEMANTIC_SCHOLAR_API_KEY` there,
and CI shouldn't depend on live third-party APIs anyway). Run manually with:
    uv run pytest tests/test_literature_sources_wet.py -v -s --no-cov -m integration

Requires network + the `arxiv`, `semanticscholar`, and `requests` packages
(already project dependencies). A configured SEMANTIC_SCHOLAR_API_KEY (or
semantic_scholar_api_key in config.yaml's runtime: block) is what actually
exercises the authenticated path end to end — the whole reason this test
exists is to catch a REGRESSION exactly like commits 693a971/b5e655f (S2
either 403ing outright, or the pacing interval silently under-pacing the
authenticated tier) the moment it happens, instead of discovering it days
later on a real run.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

QUERY = "graph neural network surrogate model"


def _build_tools():
    from a3dasm._src.agents.literature import LiteratureReviewAgent
    td = tempfile.mkdtemp(prefix="a3dasm_lit_sources_wet_")
    study = Path(td)
    (study / "runs").mkdir()
    agent = LiteratureReviewAgent()
    return agent.build_closure_tools(study)


@pytest.mark.integration
def test_arxiv_search_returns_real_results():
    tools = _build_tools()
    out = tools["arxiv_search_papers"](query=QUERY, max_results=3)
    assert out != "(no results)", f"arxiv returned no results for {QUERY!r}"
    assert not out.startswith("ERROR"), f"arxiv search failed: {out!r}"
    # Each hit line starts with an arxiv entry id in brackets.
    assert "[http://arxiv.org/abs/" in out or "[https://arxiv.org/abs/" in out, (
        f"arxiv output missing expected entry_id format: {out[:300]!r}"
    )


@pytest.mark.integration
def test_openalex_search_returns_real_results():
    tools = _build_tools()
    out = tools["search_openalex"](query=QUERY, n_results=3)
    assert not out.startswith("ERROR"), f"OpenAlex search failed: {out!r}"
    papers = json.loads(out)
    assert papers, f"OpenAlex returned zero results for {QUERY!r}"
    assert all("title" in p for p in papers)


@pytest.mark.integration
def test_semantic_scholar_search_returns_real_results():
    """This is the one that actually needs a working key to prove anything
    beyond 'the unauthenticated tier hasn't been exhausted right now' — see
    the module docstring. Skips (not fails) if the semanticscholar package
    isn't installed, matching every other S2 test's convention."""
    tools = _build_tools()
    if "search_semantic_scholar" not in tools:
        pytest.skip("semanticscholar not installed")
    out = tools["search_semantic_scholar"](query=QUERY, num_results=3)
    assert not out.startswith("ERROR"), (
        f"Semantic Scholar search failed: {out!r} — if this is a 403/"
        "cooldown, check semantic_scholar_api_key is actually configured "
        "and not being silently under-paced (regression class: 693a971, "
        "b5e655f)"
    )
    papers = json.loads(out)
    assert papers, f"Semantic Scholar returned zero results for {QUERY!r}"
    assert all("title" in p for p in papers)


@pytest.mark.integration
def test_all_three_sources_via_async_pool_like_a_real_agent():
    """The actual agent-facing pattern: wait=False on all three (fanned out
    concurrently, per-provider serialized), then one CollectSearches()."""
    tools = _build_tools()
    if "search_semantic_scholar" not in tools:
        pytest.skip("semanticscholar not installed")

    handles = {}
    for name, kwargs in (
        ("arxiv_search_papers", {"query": QUERY, "max_results": 3}),
        ("search_openalex", {"query": QUERY, "n_results": 3}),
        ("search_semantic_scholar", {"query": QUERY, "num_results": 3}),
    ):
        started = tools[name](wait=False, **kwargs)
        assert "Started async on" in started, (
            f"{name} did not enter the async pool: {started!r}"
        )
        handles[name] = started

    collected = tools["CollectSearches"]()
    for name, provider in (
        ("arxiv_search_papers", "arxiv"),
        ("search_openalex", "openalex"),
        ("search_semantic_scholar", "semantic_scholar"),
    ):
        assert f"=== {provider}#" in collected, (
            f"{provider}'s section missing from CollectSearches output: "
            f"{collected[:500]!r}"
        )
    assert "ERROR" not in collected, (
        f"one or more sources failed in the collected output: {collected!r}"
    )
    assert "still running after 600s" not in collected
