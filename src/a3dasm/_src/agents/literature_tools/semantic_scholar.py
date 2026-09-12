"""Semantic Scholar tools — client-library calls (throttled) and the
recommendations endpoint (via http_client's _robust_post)."""

from __future__ import annotations

import json as _json
import logging
import os
import re

from ...literature.http_client import SourceCooldownError, _robust_post
from .throttle import _throttled_ss

log = logging.getLogger(__name__)

# arXiv ids: "2506.14097", "2506.14097v2", or the old "math.AG/0309136" form.
_ARXIV_ID = re.compile(r"^(\d{4}\.\d{4,5}(v\d+)?|[a-z-]+(\.[A-Z]{2})?/\d{7}(v\d+)?)$")


def _s2_paper_id(paper_id: str) -> str:
    """Namespace a bare external id for Semantic Scholar's API.

    S2 resolves a non-S2 identifier only when it carries its namespace
    (``ARXIV:2506.14097``, ``DOI:10.1016/j.ijnonlinmec.2013.01.010``); a bare
    arXiv id 404s with "Paper with id … not found". Our own arXiv search hands
    the agent bare ids, so the id that identifies a paper in one tool has to be
    accepted by the next one. Anything already namespaced, or a 40-hex S2
    paperId, passes through untouched.
    """
    pid = (paper_id or "").strip()
    if ":" in pid:  # already namespaced (ARXIV:, DOI:, CorpusId:, …)
        return pid
    if pid.lower().startswith("arxiv"):  # "arxiv2506.14097", "arXiv 2506.14097"
        return "ARXIV:" + pid[5:].lstrip(": ")
    if _ARXIV_ID.match(pid):
        return f"ARXIV:{pid}"
    if pid.startswith("10."):
        return f"DOI:{pid}"
    return pid


def build_semantic_scholar_closures() -> dict:
    """The four semanticscholar-library tools; {} (with a warning) when
    the library is not installed."""
    tools: dict = {}
    # Semantic Scholar tools via the semanticscholar library.
    try:
        from semanticscholar import SemanticScholar as _SS

        from ...runtime.settings import get_str
        # config.yaml's runtime: block (or F3DASM_SEMANTIC_SCHOLAR_API_KEY)
        # is the explicit-config channel; the bare env var is honoured too
        # since it is Semantic Scholar's own documented convention, not
        # ours to rename out from under anyone already using it.
        _ss_api_key = (
            get_str("semantic_scholar_api_key", "")
            or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        )
        if not _ss_api_key:
            log.warning(
                "semantic_scholar_api_key not configured — proceeding "
                "with unauthenticated Semantic Scholar access (very low "
                "rate limit). Set it in config.yaml's runtime: block "
                "(or SEMANTIC_SCHOLAR_API_KEY / "
                "F3DASM_SEMANTIC_SCHOLAR_API_KEY) for reliable access."
            )
        # retry=False: the library's OWN internal 429 retry (tenacity,
        # up to 10 attempts, 5-60s exponential backoff EACH — legitimately
        # several minutes for one call) would otherwise silently absorb
        # every 429 before it ever reaches _throttled_ss, so our own
        # pacing/backoff/circuit-breaker (http_client's
        # _rate_limit_wait/_record_429, meant to be the SOLE retry
        # authority for this traffic — see _throttled_ss) never sees a
        # 429 until an entire hidden multi-minute retry storm has
        # already run underneath it. Disabling it here makes every
        # individual request's real outcome surface immediately, which
        # is also what makes _call_in_fresh_thread's 30s timeout
        # actually bound the call to ~30s instead of ~12 minutes.
        _sch = _SS(api_key=_ss_api_key or None, retry=False)

        def _ss_forbidden_error() -> str:
            """Message for a 403 (PermissionError): NOT retried, since
            the shared unauthenticated quota resets on a much longer
            window than a request backoff — retrying immediately would
            just burn the delegation's time on a door that is shut."""
            hint = (
                "" if _ss_api_key else
                ", or set semantic_scholar_api_key in config.yaml's "
                "runtime: block for reliable access"
            )
            return (
                "ERROR: Semantic Scholar access forbidden (403) — the "
                "shared unauthenticated quota is exhausted; retrying "
                f"will not help. Use OpenAlex/arXiv instead{hint}."
            )

        def search_semantic_scholar(
            query: str, num_results: int = 10
        ) -> str:
            """Search for papers on Semantic Scholar."""
            def _search():
                # search_paper() itself is NON-blocking — it returns a
                # LAZY PaginatedResults shell with no network call made
                # yet (unlike get_paper/get_author, which fetch eagerly
                # inside the call). The real HTTP request + retry only
                # fires on iteration — materializing it HERE, inside the
                # _throttled_ss-wrapped call, is what actually puts the
                # network I/O under the timeout/rate-limiter/circuit-
                # breaker. Iterating outside (the original shape) left
                # the real work completely unprotected: _throttled_ss
                # would return instantly having "successfully" produced
                # an empty shell, and the genuine multi-minute hang
                # happened afterwards, in code with no timeout at all.
                return list(_sch.search_paper(
                    query,
                    limit=int(num_results),
                    fields=[
                        "title", "authors", "year", "abstract",
                        "externalIds", "venue", "citationCount",
                    ],
                ))
            try:
                results = _throttled_ss(_search)
            except TimeoutError:
                return (
                    "ERROR: Semantic Scholar request timed out"
                    " after 30s. Try again or use OpenAlex."
                )
            except PermissionError:
                return _ss_forbidden_error()
            except SourceCooldownError as exc:
                return f"ERROR: {exc}"
            except Exception as exc:
                return f"ERROR: Semantic Scholar search failed: {exc}"
            papers = []
            for p in results:
                papers.append({
                    "paperId": p.paperId,
                    "title": p.title,
                    "year": p.year,
                    "venue": p.venue,
                    "citationCount": p.citationCount,
                    "authors": [
                        a["name"] for a in (p.authors or [])
                    ],
                    "abstract": (p.abstract or "")[:300],
                    "externalIds": p.externalIds or {},
                })
            return _json.dumps(papers, indent=2)

        def get_semantic_scholar_paper_details(
            paper_id: str,
        ) -> str:
            """Get details for a paper. paper_id may be a bare arXiv id
            ("2506.14097"), a bare DOI ("10.1016/j.cma.2020.113029"), an
            already-namespaced id ("ARXIV:2506.14097") or an S2 paperId —
            bare ids are namespaced for you."""
            try:
                paper = _throttled_ss(
                    _sch.get_paper,
                    _s2_paper_id(paper_id),
                    fields=[
                        "title", "authors", "year", "abstract",
                        "venue", "citationCount",
                        "influentialCitationCount",
                        "tldr", "externalIds",
                    ],
                )
            except TimeoutError:
                return (
                    "ERROR: Semantic Scholar request timed out"
                    " after 30s. Try again or use OpenAlex."
                )
            except PermissionError:
                return _ss_forbidden_error()
            except SourceCooldownError as exc:
                return f"ERROR: {exc}"
            except Exception as exc:
                return (
                    f"ERROR: Semantic Scholar paper details failed: {exc}"
                )
            return _json.dumps({
                "paperId": paper.paperId,
                "title": paper.title,
                "year": paper.year,
                "venue": paper.venue,
                "citationCount": paper.citationCount,
                "influentialCitationCount": (
                    paper.influentialCitationCount
                ),
                "tldr": (paper.tldr or {}).get("text"),
                "authors": [
                    a["name"] for a in (paper.authors or [])
                ],
                "abstract": paper.abstract,
                "externalIds": paper.externalIds or {},
            }, indent=2)

        def get_semantic_scholar_author_details(
            author_id: str,
        ) -> str:
            """Get details for an author by their S2 author ID."""
            try:
                author = _throttled_ss(
                    _sch.get_author,
                    author_id,
                    fields=[
                        "name", "affiliations", "paperCount",
                        "citationCount", "hIndex",
                    ],
                )
            except TimeoutError:
                return (
                    "ERROR: Semantic Scholar request timed out"
                    " after 30s. Try again or use OpenAlex."
                )
            except PermissionError:
                return _ss_forbidden_error()
            except SourceCooldownError as exc:
                return f"ERROR: {exc}"
            except Exception as exc:
                return (
                    f"ERROR: Semantic Scholar author details failed: {exc}"
                )
            return _json.dumps({
                "authorId": author.authorId,
                "name": author.name,
                "affiliations": author.affiliations,
                "paperCount": author.paperCount,
                "citationCount": author.citationCount,
                "hIndex": author.hIndex,
            }, indent=2)

        def get_semantic_scholar_citations_and_references(
            paper_id: str,
        ) -> str:
            """Get citing papers and references (≤20 each). paper_id takes
            the same forms as get_semantic_scholar_paper_details: a bare arXiv
            id or DOI is namespaced for you."""
            try:
                paper = _throttled_ss(
                    _sch.get_paper,
                    _s2_paper_id(paper_id),
                    fields=["citations", "references"],
                )
            except TimeoutError:
                return (
                    "ERROR: Semantic Scholar request timed out"
                    " after 30s. Try again or use OpenAlex."
                )
            except PermissionError:
                return _ss_forbidden_error()
            except SourceCooldownError as exc:
                return f"ERROR: {exc}"
            except Exception as exc:
                return (
                    "ERROR: Semantic Scholar citations/references"
                    f" failed: {exc}"
                )
            refs = [
                {
                    "paperId": r.get("paperId"),
                    "title": r.get("title"),
                }
                for r in (paper.references or [])[:20]
            ]
            cits = [
                {
                    "paperId": c.get("paperId"),
                    "title": c.get("title"),
                }
                for c in (paper.citations or [])[:20]
            ]
            return _json.dumps(
                {"references": refs, "citations": cits}, indent=2
            )

        tools.update({
            "search_semantic_scholar": search_semantic_scholar,
            "get_semantic_scholar_paper_details": (
                get_semantic_scholar_paper_details
            ),
            "get_semantic_scholar_author_details": (
                get_semantic_scholar_author_details
            ),
            "get_semantic_scholar_citations_and_references": (
                get_semantic_scholar_citations_and_references
            ),
        })
    except ImportError:
        log.warning(
            "semanticscholar not installed — S2 tools not"
            " registered for literature_reviewer"
        )
    return tools


def build_recommendations_closure() -> dict:
    """The recommendations-API tool (no client library involved)."""
    def get_semantic_scholar_recommendations(
        paper_id: str, n_results: int = 10
    ) -> str:
        """Find semantically similar papers (no citation link).

        paper_id: an S2 paperId, or a DOI or arXiv id in either bare
        ("2506.14097") or namespaced ("ARXIV:2506.14097") form.
        """
        import json as _j
        try:
            resp = _robust_post(
                "https://api.semanticscholar.org"
                "/recommendations/v1/papers/",
                json={"positivePaperIds": [_s2_paper_id(paper_id)]},
                params={
                    "fields": (
                        "paperId,title,authors,year,abstract"
                        ",externalIds,openAccessPdf"
                    ),
                    "limit": min(int(n_results), 50),
                },
            )
            papers = resp.json().get("recommendedPapers", [])
        except SourceCooldownError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:
            return f"ERROR: S2 recommendations failed: {exc}"

        out = []
        for p in papers:
            oa = p.get("openAccessPdf") or {}
            out.append({
                "paperId": p.get("paperId", ""),
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "authors": [
                    a["name"]
                    for a in (p.get("authors") or [])[:3]
                ],
                "abstract": (p.get("abstract") or "")[:300],
                "externalIds": p.get("externalIds") or {},
                "pdf_url": oa.get("url", ""),
            })
        return _j.dumps(out, indent=2)
    return {"get_semantic_scholar_recommendations": get_semantic_scholar_recommendations}
