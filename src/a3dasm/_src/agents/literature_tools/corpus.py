"""Local corpus tools: index, search, read, rank, download."""

from __future__ import annotations

from ...literature.http_client import SourceCooldownError, _robust_get


def build_corpus_closures(corpus, cache_dir) -> dict:
    """Tools over the study-scoped LiteratureCorpus at *corpus*."""
    # Defined as named functions (not lambdas) so each carries a docstring:
    # the generated <tools> catalog renders these, making it the single
    # source of tool docs — no hand-written list in the prompt to drift.
    def CorpusAdd(file_path: str, title: str = "", authors: str = "",
                  year: str = "", doi: str = "", arxiv_id: str = "",
                  venue: str = "", abstract: str = "",
                  citation_count: int = 0):
        """Index a LOCAL file (a saved PDF or full-text markdown) into the
        corpus so its passages become searchable.

        file_path is a PATH ON DISK to a .pdf/.md/.txt you already downloaded
        (e.g. "papers/2506.14097.pdf") — NOT a provider name and NOT a paper
        id. Download the file first with arxiv_download_paper, or write the
        text from arxiv_read_paper with the Write tool. The remaining
        arguments are metadata about that file, copied from the search result
        that identified it; arxiv_id/doi are recorded as metadata and used to
        derive a stable paper_id, they are not fetched. citation_count boosts
        BM25 retrieval weight (log10(c+1) scaling) — pass the citationCount
        from Semantic Scholar or OpenAlex."""
        return corpus.add(
            file_path, title=title, authors=authors, year=year, doi=doi,
            arxiv_id=arxiv_id, venue=venue, abstract=abstract,
            citation_count=int(citation_count or 0))

    def CorpusSearch(query: str, top_k: int = 10):
        """Passage search across the FULL-TEXT papers in the corpus only.
        Returns an ERROR string if no full-text papers have been added yet —
        add papers first via the search → download → CorpusAdd chain."""
        return corpus.search(query, int(top_k))

    def CorpusGetPaper(paper_id: str):
        """Return the full extracted (page-annotated) text of one corpus paper."""
        return corpus.get_paper(paper_id)

    def CorpusList():
        """List corpus metadata — each paper tagged [full-text] or
        [abstract-only] so you know which you may quote from. The corpus
        persists across every run of this study, so this may already
        list papers a prior run added — check here before re-searching
        the databases for something already present."""
        return corpus.list_papers()

    def CorpusRank(passages: str, question: str) -> str:
        """Re-rank corpus passages by BM25 relevance to question.

        Pass the raw output of CorpusSearch as ``passages``.
        Returns passages reordered from most to least relevant.
        """
        if not passages or passages == "No results found.":
            return passages

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return passages  # no-op if not installed

        import re as _re
        blocks = _re.split(
            r"(?=--- .+ \(\d*\), p\.\d+ ---)",
            passages.strip(),
        )
        blocks = [b.strip() for b in blocks if b.strip()]
        if len(blocks) <= 1:
            return passages

        tokenized = [b.lower().split() for b in blocks]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(question.lower().split())
        ranked = sorted(
            zip(blocks, scores, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )
        return "\n\n".join(b for b, _ in ranked)

    def DownloadPdf(url: str, filename: str) -> str:
        """Fetch a PDF URL and save to disk.

        Parameters
        ----------
        url:
            Direct URL to the PDF (content-type must contain
            'pdf' or body must start with ``%PDF``).
        filename:
            Destination filename.  If not absolute, saved under
            the corpus papers directory.

        Returns
        -------
        str
            Absolute path to the saved file, or ``"ERROR: …"``.
        """
        from pathlib import Path as _Path
        try:
            resp = _robust_get(
                url,
                cache_dir=cache_dir,
                headers={
                    "User-Agent": (
                        "f3dasm-agent/1.0"
                        " (mailto:f3dasm@brown.edu)"
                    ),
                },
            )
        except SourceCooldownError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:
            return f"ERROR: DownloadPdf fetch failed: {exc}"

        # Validate content
        ct = ""
        if hasattr(resp, "headers"):
            ct = resp.headers.get("Content-Type", "")
        elif hasattr(resp, "_content_type"):
            ct = resp._content_type
        body = resp.content
        if "pdf" not in ct.lower() and not body.startswith(
            b"%PDF"
        ):
            return (
                "ERROR: not a PDF — content-type is"
                f" {ct!r} and body does not start with %PDF."
                " Check the URL."
            )

        dest = _Path(filename)
        if not dest.is_absolute():
            dest = corpus._papers_dir / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
        return str(dest)

    return {
        "CorpusAdd": CorpusAdd, "CorpusSearch": CorpusSearch,
        "CorpusGetPaper": CorpusGetPaper, "CorpusList": CorpusList,
        "CorpusRank": CorpusRank, "DownloadPdf": DownloadPdf,
    }
