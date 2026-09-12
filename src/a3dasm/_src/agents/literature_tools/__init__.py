"""Runtime tool closures for the literature reviewer, one module per
provider. ``build_literature_tools`` composes them in the order the
rendered <tools> catalog has always listed them."""

from __future__ import annotations

from pathlib import Path

from .async_pool import _make_search_async_pool
from .corpus import build_corpus_closures
from .openalex import build_openalex_closures
from .semantic_scholar import (
    build_recommendations_closure,
    build_semantic_scholar_closures,
)

# Which async provider lock each slow tool serializes on.
_PROVIDER_OF = {
    "search_semantic_scholar": "semantic_scholar",
    "get_semantic_scholar_paper_details": "semantic_scholar",
    "get_semantic_scholar_recommendations": "semantic_scholar",
    "search_openalex": "openalex",
    "get_openalex_citations": "openalex",
    "get_openalex_references": "openalex",
    "arxiv_search_papers": "arxiv",
    "arxiv_read_paper": "arxiv",
    "arxiv_download_paper": "arxiv",
    "DownloadPdf": "http",
}


def build_literature_tools(study_dir, lit_reviewer_notes_dir=None) -> dict:
    """Corpus + discovery tools as runtime closures; {} if the optional
    literature stack is not installed."""
    try:
        from ...literature.literature_corpus import LiteratureCorpus
    except ImportError:
        return {}

    corpus_dir = (
        Path(lit_reviewer_notes_dir)
        if lit_reviewer_notes_dir is not None
        else Path(study_dir) / "runs" / "lit_reviewer_notes"
    )
    corpus = LiteratureCorpus(corpus_dir)
    cache_dir = corpus._http_cache_dir

    corpus_tools = build_corpus_closures(corpus, cache_dir)
    tools = {
        k: corpus_tools[k]
        for k in ("CorpusAdd", "CorpusSearch", "CorpusGetPaper", "CorpusList")
    }
    tools.update(build_semantic_scholar_closures())
    tools.update(build_openalex_closures(cache_dir))
    tools.update(build_recommendations_closure())
    tools["CorpusRank"] = corpus_tools["CorpusRank"]
    tools["DownloadPdf"] = corpus_tools["DownloadPdf"]

    # arxiv tools — Python-native, same for Claude and Ollama.
    from ...backends.ollama import _build_arxiv_closures
    tools.update(_build_arxiv_closures())

    # Make the SLOW external-provider tools async-able (wait=False default)
    # so the reviewer fans out across providers concurrently instead of
    # blocking ~minutes per call. Same-provider calls serialize. The fast
    # local corpus tools (CorpusAdd/Search/List/Rank, Read) stay synchronous.
    _asyncable, _collect = _make_search_async_pool()
    for _nm, _pv in _PROVIDER_OF.items():
        if _nm in tools:
            tools[_nm] = _asyncable(_pv, tools[_nm])
    tools["CollectSearches"] = _collect
    return tools
