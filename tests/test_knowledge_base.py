"""Draft agentic knowledge base: corpus integrity + search API contract."""
from __future__ import annotations

from a3dasm._src.knowledge import KBEntry, KnowledgeBase


class TestCorpusIntegrity:
    def test_loads_seeded_entries(self):
        kb = KnowledgeBase.load()
        assert len(kb.entries) >= 8, "expected the seeded corpus to load"

    def test_every_entry_has_required_frontmatter(self):
        kb = KnowledgeBase.load()
        for e in kb.entries:
            assert e.id and e.id != e.body, f"missing id: {e}"
            assert e.title, f"missing title: {e.id}"
            assert e.tags, f"entry {e.id} has no tags (needed for search)"
            assert e.body.strip(), f"entry {e.id} has empty body"

    def test_entry_ids_are_unique(self):
        kb = KnowledgeBase.load()
        ids = [e.id for e in kb.entries]
        assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"

    def test_get_by_id(self):
        kb = KnowledgeBase.load()
        e = kb.get("evaluate-through-get-evaluator")
        assert e is not None
        assert "get_evaluator()" in e.body


class TestSearch:
    def test_finds_evaluation_entry(self):
        kb = KnowledgeBase.load()
        hits = kb.search("how do I make my evaluations count in the ledger")
        assert hits
        assert hits[0].id == "evaluate-through-get-evaluator"

    def test_finds_pipeline_entry(self):
        kb = KnowledgeBase.load()
        hits = kb.search("pipeline reproducibility headline from store")
        assert any(h.id == "pipeline-reproduces-from-store" for h in hits)

    def test_respects_k(self):
        kb = KnowledgeBase.load()
        assert len(kb.search("evaluation store provenance", k=2)) <= 2

    def test_empty_query_returns_nothing(self):
        kb = KnowledgeBase.load()
        assert kb.search("   ") == []

    def test_irrelevant_query_returns_nothing(self):
        kb = KnowledgeBase.load()
        assert kb.search("zzzznonsensetokenqqq") == []


class TestParser:
    def test_frontmatter_lists_parse(self, tmp_path):
        d = tmp_path / "entries"
        d.mkdir()
        (d / "x.md").write_text(
            "---\nid: t\ntitle: T\ntags: [a, b, c]\naudience: [impl]\n"
            "---\nbody here\n"
        )
        kb = KnowledgeBase.load(d)
        e = kb.get("t")
        assert e is not None
        assert e.tags == ["a", "b", "c"]
        assert e.audience == ["impl"]
        assert "body here" in e.body

    def test_render_is_human_readable(self):
        kb = KnowledgeBase.load()
        out = kb.entries[0].render()
        assert out.startswith("## ")
        assert "[tags:" in out


class TestConsultHandbookTool:
    def test_returns_relevant_entry(self):
        from a3dasm._src.nodes import _consult_handbook
        out = _consult_handbook("how do I make my evaluations count in the ledger")
        assert "get_evaluator" in out

    def test_no_match_is_graceful(self):
        from a3dasm._src.nodes import _consult_handbook
        out = _consult_handbook("zzzznonsenseqqq")
        assert "No chapter" in out
        assert "list the available chapters" in out

    def test_never_raises(self):
        from a3dasm._src.nodes import _consult_handbook
        # odd inputs must not raise into the agent loop
        for q in ("", "   ", 123):
            assert isinstance(_consult_handbook(q), str)

    def test_no_arg_returns_table_of_contents(self):
        from a3dasm._src.nodes import _consult_handbook
        toc = _consult_handbook()
        assert "available chapters" in toc
        # every chapter id is listed, including the charter
        assert "falsification-charter" in toc
        assert "evaluate-through-get-evaluator" in toc

    def test_chapter_id_returns_full_chapter(self):
        from a3dasm._src.nodes import _consult_handbook
        out = _consult_handbook("falsification-charter")
        assert "if and only if" in out  # the full charter body, not a summary
        assert "Duhem" in out


class TestCharterChapter:
    def test_charter_is_a_loaded_chapter_from_the_constant(self):
        from a3dasm._src.knowledge.charter import (
            FALSIFICATION_CHARTER,
        )
        kb = KnowledgeBase.load()
        e = kb.get("falsification-charter")
        assert e is not None
        # single source: the chapter body IS the injected constant
        assert e.body == FALSIFICATION_CHARTER

    def test_toc_lists_every_chapter_with_a_summary(self):
        kb = KnowledgeBase.load()
        toc = kb.toc()
        for e in kb.entries:
            assert e.id in toc


# ── KB menu (injected into the system prompt, like the tool list) ────────────

def test_kb_menu_is_audience_filtered_and_points_at_consulthandbook():
    """The injected menu lists only entries for the agent's role and tells it how
    to pull a full chapter — so it SEES its latent knowledge without first having
    to guess that ConsultHandbook exists."""
    from a3dasm._src.knowledge import KnowledgeBase
    kb = KnowledgeBase.load()

    strat = kb.menu(audience="strategizer")
    impl = kb.menu(audience="implementer")
    assert "ConsultHandbook" in strat
    # a strategizer-only entry shows for the strategizer, not the implementer
    assert "one-delegation-one-experiment" in strat
    assert "one-delegation-one-experiment" not in impl
    # an implementer-only entry shows for the implementer, not the strategizer
    assert "experimentdata-gotchas" in impl
    assert "experimentdata-gotchas" not in strat
    # the charter (all-audience) shows for both
    assert "falsification-charter" in strat and "falsification-charter" in impl


def test_kb_menu_empty_for_unknown_audience_is_safe():
    from a3dasm._src.knowledge import KnowledgeBase
    # 'literature_reviewer' has no targeted entries → menu lists only the
    # universal ones (or is empty), never raises.
    out = KnowledgeBase.load().menu(audience="nobody_role")
    assert isinstance(out, str)


def test_every_kb_entry_title_is_at_most_100_chars():
    """The menu descriptor (the title) is the one-line summary injected into every
    prompt — cap it at 100 chars so the menu stays a terse, scannable index, not a
    paragraph. A new entry with a long title fails here on purpose."""
    from a3dasm._src.knowledge import KnowledgeBase
    for e in KnowledgeBase.load().entries:
        assert len(e.title) <= 100, (
            f"KB entry {e.id!r} title is {len(e.title)} chars (>100): {e.title!r}"
        )
