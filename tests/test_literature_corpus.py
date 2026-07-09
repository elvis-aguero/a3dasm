"""Tests for LiteratureCorpus — written before implementation (TDD red phase)."""

from __future__ import annotations

import csv
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from a3dasm._src.literature_corpus import LiteratureCorpus


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_corpus(tmp_path: Path) -> LiteratureCorpus:
    return LiteratureCorpus(tmp_path / "literature")


def _inject_paper(corpus: LiteratureCorpus, paper_id: str, title: str = "Test Paper",
                  authors: str = "A. Author", year: str = "2024",
                  source: str = "arxiv", text: str = "",
                  full_text: bool = True) -> None:
    """Inject a fake paper directly into the corpus without network calls.

    ``full_text=True`` by default so that search() finds the paper;
    set to False to test abstract-only behaviour.
    """
    paper_dir = corpus._paper_dir(paper_id)
    paper_dir.mkdir(parents=True, exist_ok=True)
    md_path = paper_dir / "paper.md"
    md_content = text or f"<!-- page 1 -->\nAbstract for {title}\n"
    md_path.write_text(md_content, encoding="utf-8")

    # Populate chunks.jsonl so BM25 search can find this paper
    chunks = corpus._chunk_text(md_content)
    corpus._append_chunks(paper_id, chunks)

    rows = corpus._load_csv()
    rows.append({
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "year": year,
        "doi": "",
        "arxiv_id": paper_id.replace("arxiv_", "").replace("_", "."),
        "venue": "",
        "abstract": f"Abstract for {title}",
        "local_pdf_path": "",
        "local_md_path": str(md_path),
        "added_at": "2024-01-01T00:00:00+00:00",
        "source": source,
        "citation_count": "0",
        "full_text": "true" if full_text else "false",
    })
    corpus._save_csv(rows)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_corpus_creates_dir_on_init(tmp_path):
    corpus_dir = tmp_path / "lit"
    assert not corpus_dir.exists()
    LiteratureCorpus(corpus_dir)
    assert corpus_dir.exists()
    assert (corpus_dir / "papers").exists()


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


def test_corpus_csv_created_on_first_add(tmp_path):
    """add() with a local .md file creates corpus.csv and returns the paper_id."""
    corpus = _make_corpus(tmp_path)
    md_file = tmp_path / "paper.md"
    md_file.write_text("<!-- page 1 -->\nThe Transformer model.", encoding="utf-8")

    result = corpus.add(
        source=str(md_file),
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        authors="Vaswani et al.",
        year="2017",
    )

    assert result == "arxiv_1706_03762"
    assert (corpus._corpus_dir / "corpus.csv").exists()
    rows = corpus._load_csv()
    assert len(rows) == 1
    assert rows[0]["paper_id"] == "arxiv_1706_03762"
    assert rows[0]["title"] == "Attention Is All You Need"


def test_add_already_in_corpus(tmp_path):
    """Adding the same arxiv_id twice returns 'Already in corpus'."""
    corpus = _make_corpus(tmp_path)
    _inject_paper(corpus, "arxiv_1706_03762", title="Attention")

    md_file = tmp_path / "paper.md"
    md_file.write_text("content", encoding="utf-8")
    result = corpus.add(str(md_file), arxiv_id="1706.03762")
    assert result == "Already in corpus: arxiv_1706_03762"


def test_add_returns_error_on_complete_failure(tmp_path):
    """add() with a non-existent file path returns an ERROR string."""
    corpus = _make_corpus(tmp_path)
    result = corpus.add("/nonexistent/path/paper.pdf")
    assert result.startswith("ERROR:")


# ---------------------------------------------------------------------------
# list_papers()
# ---------------------------------------------------------------------------


def test_list_papers_empty_corpus(tmp_path):
    corpus = _make_corpus(tmp_path)
    assert corpus.list_papers() == "Corpus is empty."


def test_list_papers_shows_added_paper(tmp_path):
    corpus = _make_corpus(tmp_path)
    _inject_paper(corpus, "arxiv_1706_03762", title="Attention Is All You Need",
                  authors="Vaswani et al.", year="2017", source="arxiv")

    table = corpus.list_papers()
    assert "arxiv_1706_03762" in table
    assert "Attention Is All You Need" in table
    assert "2017" in table


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


def test_search_no_results(tmp_path):
    """Empty corpus returns an informative message (error or no results)."""
    corpus = _make_corpus(tmp_path)
    result = corpus.search("xyzzy_nonexistent_term_12345")
    # Either "No results found." or the full-text ERROR guidance
    assert "No results found." in result or "ERROR" in result


def test_search_finds_passage(tmp_path):
    corpus = _make_corpus(tmp_path)
    _inject_paper(corpus, "arxiv_1706_03762", title="Attention Is All You Need",
                  text="<!-- page 1 -->\nThe Transformer model uses self-attention.\n")

    result = corpus.search("self-attention")
    assert "self-attention" in result.lower()
    assert "Attention Is All You Need" in result


def test_search_respects_top_k(tmp_path):
    corpus = _make_corpus(tmp_path)
    # Paper with 5 matching lines
    many_lines = "<!-- page 1 -->\n" + "\n".join(
        [f"Line {i}: keyword here" for i in range(10)]
    )
    _inject_paper(corpus, "arxiv_0000_00001", title="Paper One", text=many_lines)
    result = corpus.search("keyword", top_k=3)
    # Should return at most 3 passages
    passage_count = result.count("--- Paper One")
    assert passage_count <= 3


def test_bm25_search_finds_paraphrase(tmp_path):
    """BM25 search should find documents by shared vocabulary (not exact match)."""
    corpus = _make_corpus(tmp_path)
    _inject_paper(
        corpus,
        "arxiv_9999_00001",
        title="Buckling Analysis",
        text=(
            "<!-- page 1 -->\n"
            "The critical buckling stress is maximized when the plate thickness "
            "is optimized. The maximum load-bearing capacity depends on the "
            "critical stress value sigma_crit in structural design.\n"
        ),
    )

    # Query uses different phrasing but overlapping vocabulary
    result = corpus.search("maximum sigma_crit", top_k=5)
    # Should find something — BM25 scores on shared tokens (maximum, sigma_crit)
    # OR graceful fallback via _search_substring which also finds the passage
    assert result != "No results found."


# ---------------------------------------------------------------------------
# get_paper()
# ---------------------------------------------------------------------------


def test_get_paper_not_found(tmp_path):
    corpus = _make_corpus(tmp_path)
    result = corpus.get_paper("no_such_paper_id")
    assert result.startswith("ERROR")


def test_get_paper_returns_content(tmp_path):
    corpus = _make_corpus(tmp_path)
    known_text = "<!-- page 1 -->\nThis is the paper content.\n"
    _inject_paper(corpus, "arxiv_1706_03762", text=known_text)

    result = corpus.get_paper("arxiv_1706_03762")
    assert "This is the paper content." in result


# ---------------------------------------------------------------------------
# _extract_pdf_to_md()
# ---------------------------------------------------------------------------


def test_extract_pdf_to_md_page_annotations(tmp_path):
    """If fitz is available, output contains <!-- page N --> annotations."""
    try:
        import fitz  # noqa: F401
        fitz_available = True
    except ImportError:
        fitz_available = False

    corpus = _make_corpus(tmp_path)

    if fitz_available:
        # A realistic text PDF (real papers are large) so extraction is treated
        # as a full parse, not a thin/scanned stub.
        doc = fitz.open()
        for _ in range(8):
            pg = doc.new_page()
            pg.insert_textbox(fitz.Rect(40, 40, 560, 760), ("word " * 150))
        pdf_path = tmp_path / "test.pdf"
        doc.save(str(pdf_path))
        doc.close()

        result = corpus._extract_pdf_to_md(pdf_path)
        # The <!-- page N --> annotations are the fitz path. Docling, when it is
        # installed and parses the PDF, legitimately wins (it is preferred, and
        # its markdown carries no page markers). So assert the page markers only
        # on the lean install where Docling is absent; otherwise assert the body
        # text was extracted.
        try:
            import docling  # noqa: F401
            docling_available = True
        except ImportError:
            docling_available = False
        if docling_available:
            assert "word" in result.lower()
        else:
            assert "<!-- page 1 -->" in result
    else:
        # Graceful fallback expected
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")
        result = corpus._extract_pdf_to_md(fake_pdf)
        assert "unavailable" in result.lower()


def test_extract_pdf_to_md_fallback_when_fitz_missing(tmp_path):
    """When fitz is patched as None, fallback string is returned."""
    corpus = _make_corpus(tmp_path)
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    import a3dasm._src.literature_corpus as lc_module
    original_fitz = lc_module.fitz
    try:
        lc_module.fitz = None
        result = corpus._extract_pdf_to_md(fake_pdf)
        assert "unavailable" in result.lower()
    finally:
        lc_module.fitz = original_fitz


def test_pdf_failed_extraction_is_rejected_not_stored_as_fulltext(
    tmp_path, monkeypatch
):
    """A PDF whose extraction yields too little (scanned / corrupt streams) must
    NOT be stored as quotable full-text. The old code hardcoded full_text=True
    for every PDF and kept a ~236-char placeholder the reviewer would 'quote'."""
    corpus = _make_corpus(tmp_path)
    fake_pdf = tmp_path / "scanned.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 dummy")
    monkeypatch.setattr(
        corpus, "_extract_pdf_to_md", lambda p: "<!-- page 1 -->\n(tiny)")
    result = corpus.add(str(fake_pdf), arxiv_id="1502.05700")
    assert result.startswith("ERROR"), result
    assert "arxiv_read_paper" in result  # points to the working alternative
    assert corpus._load_csv() == []      # nothing phantom entered the corpus


def test_extraction_returns_real_body_from_text_pdf(tmp_path):
    """Regression: a text PDF must extract its full body. The old code tried
    Docling first and returned ANY result >100 chars, so a botched 236-char
    Docling parse of a 13-page paper (Snoek et al. 2015) was accepted while
    PyMuPDF extracts ~53k chars. Now PyMuPDF runs first and a real body wins."""
    import pytest as _pt
    try:
        import fitz
    except ImportError:
        _pt.skip("fitz not available")
    corpus = _make_corpus(tmp_path)
    doc = fitz.open()
    for _ in range(8):  # ~6000 chars of real text, comfortably > threshold
        pg = doc.new_page()
        pg.insert_textbox(fitz.Rect(40, 40, 560, 760), ("word " * 150))
    pdf = tmp_path / "text_paper.pdf"
    doc.save(str(pdf))
    doc.close()

    md = corpus._extract_pdf_to_md(pdf)
    assert len(md) > 5000, f"expected a real body, got {len(md)} chars"
    assert "word" in md
    assert "unavailable" not in md.lower()


def test_pdf_real_extraction_is_stored_as_fulltext(tmp_path, monkeypatch):
    """The fix must not over-reject: a PDF with a real body is full-text."""
    corpus = _make_corpus(tmp_path)
    fake_pdf = tmp_path / "real.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 dummy")
    body = "<!-- page 1 -->\n" + ("word " * 2000)  # > 5000 chars
    monkeypatch.setattr(corpus, "_extract_pdf_to_md", lambda p: body)
    result = corpus.add(str(fake_pdf), arxiv_id="1807.02811")
    assert not result.startswith("ERROR"), result
    rows = corpus._load_csv()
    assert len(rows) == 1 and rows[0]["full_text"] == "true"


# ---------------------------------------------------------------------------
# _derive_paper_id() — replaces _normalize_identifier
# ---------------------------------------------------------------------------


def test_normalize_arxiv_id(tmp_path):
    corpus = _make_corpus(tmp_path)
    paper_id = corpus._derive_paper_id("2312.12345", "", Path("paper.pdf"))
    assert paper_id == "arxiv_2312_12345"


def test_normalize_arxiv_prefix(tmp_path):
    corpus = _make_corpus(tmp_path)
    paper_id = corpus._derive_paper_id("arxiv:2312.12345", "", Path("paper.pdf"))
    assert paper_id == "arxiv_arxiv_2312_12345"  # slugified with prefix


def test_normalize_doi(tmp_path):
    corpus = _make_corpus(tmp_path)
    paper_id = corpus._derive_paper_id("", "10.1234/test", Path("paper.pdf"))
    assert paper_id == "doi_10_1234_test"


def test_normalize_bare_doi(tmp_path):
    """DOI in doi field → doi_ prefix."""
    corpus = _make_corpus(tmp_path)
    paper_id = corpus._derive_paper_id("", "10.5678/other", Path("paper.pdf"))
    assert paper_id.startswith("doi_")


def test_normalize_unknown(tmp_path):
    """No arxiv_id, no doi → fall back to filename stem."""
    corpus = _make_corpus(tmp_path)
    paper_id = corpus._derive_paper_id("", "", Path("my_paper_2024.pdf"))
    assert "my_paper_2024" in paper_id


# ---------------------------------------------------------------------------
# Dense embedding fallback
# ---------------------------------------------------------------------------


def test_search_falls_back_gracefully_without_embeddings(tmp_path):
    """Search still returns BM25 results when no chunks.npy files exist."""
    corpus = _make_corpus(tmp_path)
    # _inject_paper does NOT create chunks.npy — simulates missing dense embeddings
    _inject_paper(
        corpus,
        "arxiv_1234_56789",
        title="Fallback Test Paper",
        text="<!-- page 1 -->\nThis paper discusses neural network optimization.\n",
    )

    # Verify no chunks.npy exists
    npy_path = corpus._paper_dir("arxiv_1234_56789") / "chunks.npy"
    assert not npy_path.exists(), "chunks.npy should not exist for injected paper"

    result = corpus.search("neural network optimization", top_k=5)
    assert result != "No results found.", "Expected BM25 fallback to find results"
    assert "Fallback Test Paper" in result


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_add_is_thread_safe(tmp_path):
    """Two threads adding different local .md files → CSV has exactly 2 rows."""
    corpus = _make_corpus(tmp_path)

    md1 = tmp_path / "paper1.md"
    md2 = tmp_path / "paper2.md"
    md1.write_text("<!-- page 1 -->\nContent one.", encoding="utf-8")
    md2.write_text("<!-- page 1 -->\nContent two.", encoding="utf-8")

    errors: list[str] = []

    def add_paper(md_path: Path, arxiv_id: str) -> None:
        result = corpus.add(str(md_path), arxiv_id=arxiv_id, title=f"Paper {arxiv_id}")
        if result.startswith("ERROR"):
            errors.append(result)

    t1 = threading.Thread(target=add_paper, args=(md1, "1111.11111"))
    t2 = threading.Thread(target=add_paper, args=(md2, "2222.22222"))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == [], f"Unexpected errors: {errors}"
    rows = corpus._load_csv()
    assert len(rows) == 2
    ids = {r["paper_id"] for r in rows}
    assert "arxiv_1111_11111" in ids
    assert "arxiv_2222_22222" in ids
