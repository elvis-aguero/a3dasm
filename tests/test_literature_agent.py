"""Tests for LiteratureReviewAgent.build_closure_tools() — covers the corpus
and discovery tool closures without network calls."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_agent():
    from a3dasm._src.agents.literature import LiteratureReviewAgent
    return LiteratureReviewAgent()


# ---------------------------------------------------------------------------
# Basic agent attributes
# ---------------------------------------------------------------------------


def test_literature_agent_has_correct_tools():
    """LiteratureReviewAgent declares Read, Grep, Glob tools."""
    agent = _make_agent()
    assert "Read" in agent.tools
    assert "Grep" in agent.tools
    assert "Glob" in agent.tools


def test_literature_agent_injects_problem_statement():
    """LiteratureReviewAgent has inject_problem_statement=True."""
    agent = _make_agent()
    assert agent.inject_problem_statement is True


def test_literature_agent_has_system_prompt():
    """LiteratureReviewAgent has a non-empty system_prompt."""
    agent = _make_agent()
    assert isinstance(agent.system_prompt, str)
    assert len(agent.system_prompt) > 100


# ---------------------------------------------------------------------------
# build_closure_tools returns corpus tools
# ---------------------------------------------------------------------------


def test_build_closure_tools_returns_corpus_tools(tmp_path):
    """build_closure_tools returns CorpusAdd, CorpusSearch, etc."""
    agent = _make_agent()
    tools = agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=tmp_path / "lit",
    )

    assert "CorpusAdd" in tools
    assert "CorpusSearch" in tools
    assert "CorpusGetPaper" in tools
    assert "CorpusList" in tools


def test_corpus_add_closure_works(tmp_path):
    """CorpusAdd closure adds a .md file to the corpus."""
    agent = _make_agent()
    lit_dir = tmp_path / "lit"
    tools = agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=lit_dir,
    )

    md_file = tmp_path / "paper.md"
    md_file.write_text("<!-- page 1 -->\nContent about transformers.\n", encoding="utf-8")

    result = tools["CorpusAdd"](
        str(md_file),
        title="Transformer Paper",
        authors="A. Author",
        year="2017",
        arxiv_id="1706.03762",
    )

    assert result == "arxiv_1706_03762"


def test_corpus_list_closure_works(tmp_path):
    """CorpusList closure lists all papers in the corpus."""
    agent = _make_agent()
    lit_dir = tmp_path / "lit"
    tools = agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=lit_dir,
    )

    # Empty corpus
    result = tools["CorpusList"]()
    assert result == "Corpus is empty."


def test_corpus_search_closure_on_empty_corpus(tmp_path):
    """CorpusSearch returns an informative response on empty corpus."""
    agent = _make_agent()
    lit_dir = tmp_path / "lit"
    tools = agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=lit_dir,
    )

    result = tools["CorpusSearch"]("neural networks")
    # Empty corpus has no full-text papers → ERROR guidance or no results
    assert "No results found." in result or "ERROR" in result


def test_corpus_get_paper_not_found(tmp_path):
    """CorpusGetPaper returns ERROR for unknown paper."""
    agent = _make_agent()
    lit_dir = tmp_path / "lit"
    tools = agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=lit_dir,
    )

    result = tools["CorpusGetPaper"]("no_such_paper")
    assert "ERROR" in result


# ---------------------------------------------------------------------------
# build_closure_tools with fallback lit_reviewer_notes_dir
# ---------------------------------------------------------------------------


def test_build_closure_tools_fallback_dir(tmp_path):
    """build_closure_tools uses study_dir/delegations/literature when lit_reviewer_notes_dir=None."""
    agent = _make_agent()
    tools = agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=None,
    )

    # Should still return the corpus tools
    assert "CorpusAdd" in tools
    assert "CorpusList" in tools


# ---------------------------------------------------------------------------
# search_openalex tool (mocked network)
# ---------------------------------------------------------------------------


def test_search_openalex_returns_results_on_success(tmp_path):
    """search_openalex returns JSON results when requests.get succeeds."""
    agent = _make_agent()
    lit_dir = tmp_path / "lit"
    tools = agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=lit_dir,
    )

    if "search_openalex" not in tools:
        pytest.skip("search_openalex not in tools")

    oa_result = {
        "results": [
            {
                "id": "W123",
                "title": "Test Paper",
                "publication_year": 2023,
                "doi": "10.1234/test",
                "authorships": [{"author": {"display_name": "A. Author"}}],
                "primary_location": {"pdf_url": "https://example.com/paper.pdf"},
                "best_oa_location": None,
                "open_access": {"oa_url": None},
                "abstract_inverted_index": {"test": [0], "abstract": [1]},
            }
        ]
    }
    import json as _json
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.headers = {"Content-Type": "application/json"}
    mock_resp.text = _json.dumps(oa_result)
    mock_resp.json.return_value = oa_result

    with patch("requests.get", return_value=mock_resp):
        with patch("a3dasm._src.literature_corpus._sleep"):
            result = tools["search_openalex"]("neural networks", n_results=5)

    import json
    data = json.loads(result)
    assert len(data) >= 1
    titles = [d["title"] for d in data]
    assert "Test Paper" in titles


def test_search_openalex_returns_error_on_failure(tmp_path):
    """search_openalex returns ERROR when requests.get raises."""
    agent = _make_agent()
    lit_dir = tmp_path / "lit"
    tools = agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=lit_dir,
    )

    if "search_openalex" not in tools:
        pytest.skip("search_openalex not in tools")

    import requests as _requests
    with patch("requests.get", side_effect=_requests.RequestException("Connection failed")):
        with patch("a3dasm._src.literature_corpus._sleep"):
            result = tools["search_openalex"]("neural networks")

    assert "ERROR" in result


# ---------------------------------------------------------------------------
# get_semantic_scholar_recommendations tool (mocked)
# ---------------------------------------------------------------------------


def test_get_ss_recommendations_returns_json(tmp_path):
    """get_semantic_scholar_recommendations returns JSON on success."""
    agent = _make_agent()
    lit_dir = tmp_path / "lit"
    tools = agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=lit_dir,
    )

    if "get_semantic_scholar_recommendations" not in tools:
        pytest.skip("get_semantic_scholar_recommendations not available")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.headers = {"Content-Type": "application/json"}
    mock_resp.json.return_value = {
        "recommendedPapers": [
            {
                "paperId": "P1",
                "title": "Similar Paper",
                "year": 2022,
                "authors": [{"name": "B. Author"}],
                "abstract": "Abstract text.",
                "externalIds": {},
                "openAccessPdf": None,
            }
        ]
    }

    with patch("requests.post", return_value=mock_resp):
        with patch("a3dasm._src.literature_corpus._sleep"):
            result = tools["get_semantic_scholar_recommendations"](
                "1706.03762", n_results=5
            )

    import json
    data = json.loads(result)
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["title"] == "Similar Paper"


def test_get_ss_recommendations_returns_error_on_failure(tmp_path):
    """get_semantic_scholar_recommendations returns ERROR on network failure."""
    agent = _make_agent()
    lit_dir = tmp_path / "lit"
    tools = agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=lit_dir,
    )

    if "get_semantic_scholar_recommendations" not in tools:
        pytest.skip("get_semantic_scholar_recommendations not available")

    import requests as _requests
    with patch("requests.post", side_effect=_requests.RequestException("Network error")):
        with patch("a3dasm._src.literature_corpus._sleep"):
            result = tools["get_semantic_scholar_recommendations"]("1706.03762")

    assert "ERROR" in result


# ---------------------------------------------------------------------------
# CorpusRank tool (if present)
# ---------------------------------------------------------------------------


def test_corpus_rank_tool_present(tmp_path):
    """CorpusRank tool is injected when available."""
    agent = _make_agent()
    lit_dir = tmp_path / "lit"
    tools = agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=lit_dir,
    )

    if "CorpusRank" not in tools:
        pytest.skip("CorpusRank not injected (rank_bm25 or other dep missing)")

    # CorpusRank expects a single string containing passage blocks
    passages_str = "--- Paper (2024), p.1 ---\nSome passage.\n"
    result = tools["CorpusRank"](passages_str, "relevant query")
    assert isinstance(result, str)


def test_cap_result_truncates_oversized_payloads():
    """A search/read payload bigger than the cap is truncated with a marker, so it
    never overflows the tool-result token limit and gets dropped whole (observed
    every run: 'exceeds maximum allowed tokens')."""
    from a3dasm._src.agents.literature import _cap_result, _MAX_RESULT_CHARS
    small = "ok" * 10
    assert _cap_result(small) == small               # under cap: untouched
    big = "x" * (_MAX_RESULT_CHARS + 5000)
    out = _cap_result(big)
    assert len(out) < len(big) and "truncated" in out
    assert out.startswith("x" * 100)                 # keeps the head
    assert _cap_result(12345) == "12345"             # coerces non-str
