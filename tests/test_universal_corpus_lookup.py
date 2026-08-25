"""Tests for Agent.build_closure_tools()'s default: every agent gets read-only
literature-corpus lookup (CorpusSearch/CorpusList/CorpusGetPaper) without
needing its own override — the same way QueryStore lets every node read the
canonical evaluation ledger without delegating to the data generator.
Acquisition (CorpusAdd, external search) stays literature_reviewer-only,
since finding/vetting a NEW paper needs judgment a raw tool call can't
supply — see LiteratureReviewAgent.build_closure_tools, which overrides the
default entirely rather than extending it."""
from __future__ import annotations

from a3dasm._src.agents.critic import AdversarialCritiqueAgent
from a3dasm._src.agents.datagenerator import DataGeneratorAgent
from a3dasm._src.agents.debugger import DebuggerAgent
from a3dasm._src.agents.implementer import F3dasmImplementerAgent
from a3dasm._src.agents.literature import LiteratureReviewAgent
from a3dasm._src.agents.strategizer import StrategizerAgent

_NON_LITERATURE_AGENTS = (
    StrategizerAgent, F3dasmImplementerAgent, DataGeneratorAgent,
    AdversarialCritiqueAgent, DebuggerAgent,
)


def test_every_non_literature_agent_gets_read_only_corpus_lookup(tmp_path):
    """None of these agents override build_closure_tools — they inherit the
    base default, which must give each of them CorpusSearch/CorpusList/
    CorpusGetPaper (read-only), but never CorpusAdd (acquisition stays
    literature_reviewer-only)."""
    for Ag in _NON_LITERATURE_AGENTS:
        agent = Ag()
        tools = agent.build_closure_tools(study_dir=tmp_path)
        assert "CorpusSearch" in tools, f"{Ag.__name__} missing CorpusSearch"
        assert "CorpusList" in tools, f"{Ag.__name__} missing CorpusList"
        assert "CorpusGetPaper" in tools, f"{Ag.__name__} missing CorpusGetPaper"
        assert "CorpusAdd" not in tools, (
            f"{Ag.__name__} must not get CorpusAdd — acquisition needs "
            "judgment and stays gated behind a real literature_reviewer "
            "delegation."
        )


def test_non_literature_agent_corpus_lookup_actually_works(tmp_path):
    """The injected closures are not stubs — CorpusList on a fresh corpus
    behaves exactly like the literature_reviewer's own CorpusList."""
    agent = StrategizerAgent()
    tools = agent.build_closure_tools(study_dir=tmp_path)
    assert tools["CorpusList"]() == "Corpus is empty."


def test_non_literature_agent_sees_papers_added_by_literature_reviewer(tmp_path):
    """Regression: the corpus is SHARED, study-scoped storage (see
    agent_runtime.py's _make_adapter and literature_corpus.py) — a paper the
    literature_reviewer adds during its own delegation must show up for any
    other agent's CorpusSearch/CorpusList, since both resolve the identical
    default path (study_dir/runs/lit_reviewer_notes) when
    lit_reviewer_notes_dir is not explicitly overridden."""
    md_file = tmp_path / "paper.md"
    md_file.write_text(
        "<!-- page 1 -->\n"
        + "Content about tensegrity metamaterials. " * 200,
        encoding="utf-8",
    )

    lit_agent = LiteratureReviewAgent()
    lit_tools = lit_agent.build_closure_tools(study_dir=tmp_path)
    paper_id = lit_tools["CorpusAdd"](
        str(md_file), title="Tensegrity Paper", arxiv_id="9999.99999",
    )
    assert paper_id == "arxiv_9999_99999"

    other_agent = DataGeneratorAgent()
    other_tools = other_agent.build_closure_tools(study_dir=tmp_path)
    listing = other_tools["CorpusList"]()
    assert "Tensegrity Paper" in listing

    result = other_tools["CorpusSearch"]("tensegrity metamaterials")
    assert "tensegrity" in result.lower() or "Tensegrity" in result


def test_literature_reviewer_still_gets_corpus_add(tmp_path):
    """LiteratureReviewAgent.build_closure_tools fully overrides the base
    default (does not call super()) — confirm it still returns CorpusAdd,
    i.e. this refactor didn't silently drop the literature_reviewer's own
    acquisition capability while adding the universal read-only default."""
    agent = LiteratureReviewAgent()
    tools = agent.build_closure_tools(study_dir=tmp_path)
    assert "CorpusAdd" in tools
    assert "CorpusSearch" in tools
    assert "CorpusList" in tools
    assert "CorpusGetPaper" in tools


def test_lit_reviewer_notes_dir_override_respected_by_default(tmp_path):
    """The base default must honor an explicit lit_reviewer_notes_dir (same
    contract as study_dir/runs/lit_reviewer_notes fallback), so callers that
    pass the resolved path (as _make_adapter does) get the SAME corpus a
    literature_reviewer delegation writes to, not a stray default."""
    custom_dir = tmp_path / "custom_corpus_location"
    agent = StrategizerAgent()
    tools = agent.build_closure_tools(
        study_dir=tmp_path, lit_reviewer_notes_dir=custom_dir,
    )
    tools["CorpusList"]()
    assert custom_dir.is_dir()
    assert not (tmp_path / "runs" / "lit_reviewer_notes").exists()
