"""Regression for lit-bug #3 (backlog #8): arxiv_search_papers / arxiv_list_papers
received max_results as a string ("5") from MCP string-in tools and passed it to
the arxiv library, whose internal pagination arithmetic did 'str - int' →
"unsupported operand type(s) for -: 'str' and 'int'". They must coerce to int.
"""
from __future__ import annotations

import sys
import types


def _fake_arxiv(captured):
    mod = types.ModuleType("arxiv")

    class Search:
        def __init__(self, query=None, max_results=None, id_list=None):
            captured["max_results"] = max_results
            captured["type"] = type(max_results)

    class Client:
        def results(self, search):
            return iter([])

    mod.Search = Search
    mod.Client = Client
    return mod


def test_search_papers_coerces_string_max_results(monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "arxiv", _fake_arxiv(captured))
    from a3dasm._src.backends.openai_compatible import _build_arxiv_closures
    tools = _build_arxiv_closures()
    # The exact repro: model passes max_results as the STRING "5".
    tools["arxiv_search_papers"]("query", max_results="5")
    assert captured["type"] is int
    assert captured["max_results"] == 5


def test_list_papers_coerces_string_max_results(monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "arxiv", _fake_arxiv(captured))
    from a3dasm._src.backends.openai_compatible import _build_arxiv_closures
    tools = _build_arxiv_closures()
    tools["arxiv_list_papers"]("cs.LG", max_results="10")
    assert captured["type"] is int and captured["max_results"] == 10


def test_int_max_results_still_works(monkeypatch):
    captured = {}
    monkeypatch.setitem(sys.modules, "arxiv", _fake_arxiv(captured))
    from a3dasm._src.backends.openai_compatible import _build_arxiv_closures
    tools = _build_arxiv_closures()
    tools["arxiv_search_papers"]("query", max_results=7)
    assert captured["max_results"] == 7
