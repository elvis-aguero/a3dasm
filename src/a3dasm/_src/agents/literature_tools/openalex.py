"""OpenAlex tools: search and citation-graph traversal."""

from __future__ import annotations

import logging
import os

from ...literature.http_client import SourceCooldownError, _robust_get

log = logging.getLogger(__name__)


def build_openalex_closures(cache_dir) -> dict:
    """search_openalex + citation/reference traversal, sharing one
    polite-pool header set."""
    # OpenAlex headers — polite pool always; Bearer key if available.
    _oa_key = os.environ.get("OPENALEX_API_KEY")
    if not _oa_key:
        log.warning(
            "OPENALEX_API_KEY not set — proceeding with polite-pool "
            "access (lower rate limit). Set OPENALEX_API_KEY for "
            "better throughput."
        )
    _oa_headers = {"User-Agent": "f3dasm-agent/1.0 (mailto:f3dasm@brown.edu)"}
    if _oa_key:
        _oa_headers["Authorization"] = f"Bearer {_oa_key}"

    def search_openalex(
        query: str, n_results: int = 10
    ) -> str:
        """Search OpenAlex (300M+ works; open-access PDF URLs).

        Returns papers with metadata and open-access PDF URLs
        where available (best_oa_location.pdf_url / oa_url).
        Prefer for papers NOT on arXiv.
        """
        import json as _j
        try:
            resp = _robust_get(
                "https://api.openalex.org/works",
                params={
                    "search": query,
                    "per-page": min(int(n_results), 25),
                    "select": (
                        "id,title,authorships,publication_year"
                        ",doi,primary_location,open_access"
                        ",best_oa_location"
                        ",abstract_inverted_index"
                    ),
                },
                headers=_oa_headers,
                cache_dir=cache_dir,
            )
            works = resp.json().get("results", [])
        except SourceCooldownError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:
            return f"ERROR: OpenAlex search failed: {exc}"

        out = []
        for w in works:
            authors = ", ".join(
                a["author"]["display_name"]
                for a in (w.get("authorships") or [])[:3]
            )
            doi = (w.get("doi") or "").replace(
                "https://doi.org/", ""
            )
            # Best OA location first, then primary_location
            boa = w.get("best_oa_location") or {}
            loc = w.get("primary_location") or {}
            oa = w.get("open_access") or {}
            pdf_url = (
                boa.get("pdf_url")
                or oa.get("oa_url")
                or loc.get("pdf_url")
                or loc.get("landing_page_url")
                or ""
            )
            # reconstruct abstract from inverted index
            inv = w.get("abstract_inverted_index") or {}
            abstract = ""
            if inv:
                pairs = [
                    (pos, word)
                    for word, positions in inv.items()
                    for pos in positions
                ]
                pairs.sort()
                abstract = " ".join(
                    word for _, word in pairs
                )[:400]
            out.append({
                "id": w.get("id", ""),
                "title": w.get("title", ""),
                "year": w.get("publication_year", ""),
                "authors": authors,
                "doi": doi,
                "pdf_url": pdf_url,
                "abstract": abstract,
            })
        return _j.dumps(out, indent=2)

    def get_openalex_citations(
        work_id: str, n_results: int = 20
    ) -> str:
        """Fetch papers that cite *work_id* via OpenAlex citation-graph traversal; works during Semantic Scholar cooldowns.

        work_id may be a bare OpenAlex ID (``W123…``) or a full URL
        (``https://openalex.org/W123…``).  Returns a JSON list of
        ``{id, title, year, cited_by_count, doi, pdf_url}`` sorted
        by citation count descending.
        """
        import json as _j
        # Normalise: strip full URL prefix if present
        _wid = work_id.strip()
        if _wid.startswith("https://openalex.org/"):
            _wid = _wid[len("https://openalex.org/"):]
        try:
            resp = _robust_get(
                "https://api.openalex.org/works",
                params={
                    "filter": f"cites:{_wid}",
                    "per-page": min(int(n_results), 50),
                    "sort": "cited_by_count:desc",
                    "select": (
                        "id,title,publication_year,doi"
                        ",cited_by_count,best_oa_location"
                        ",open_access,primary_location"
                    ),
                },
                headers=_oa_headers,
                cache_dir=cache_dir,
            )
            works = resp.json().get("results", [])
        except SourceCooldownError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:
            return f"ERROR: OpenAlex citations failed: {exc}"

        out = []
        for w in works:
            doi = (w.get("doi") or "").replace(
                "https://doi.org/", ""
            )
            boa = w.get("best_oa_location") or {}
            loc = w.get("primary_location") or {}
            oa = w.get("open_access") or {}
            pdf_url = (
                boa.get("pdf_url")
                or oa.get("oa_url")
                or loc.get("pdf_url")
                or loc.get("landing_page_url")
                or ""
            )
            out.append({
                "id": w.get("id", ""),
                "title": w.get("title", ""),
                "year": w.get("publication_year", ""),
                "cited_by_count": w.get("cited_by_count", 0),
                "doi": doi,
                "pdf_url": pdf_url,
            })
        return _j.dumps(out, indent=2)

    def get_openalex_references(work_id: str) -> str:
        """Fetch the reference list of *work_id* hydrated from OpenAlex; citation-graph traversal fallback when S2 is rate-limited.

        Returns a JSON list of ``{id, title, year, cited_by_count,
        doi, pdf_url}`` for the first 40 referenced works, or a
        plain message when no references are listed.
        """
        import json as _j
        _wid = work_id.strip()
        if _wid.startswith("https://openalex.org/"):
            _wid = _wid[len("https://openalex.org/"):]
        try:
            work_resp = _robust_get(
                f"https://api.openalex.org/works/{_wid}",
                params={
                    "select": "id,referenced_works",
                },
                headers=_oa_headers,
                cache_dir=cache_dir,
            )
            ref_ids = work_resp.json().get(
                "referenced_works", []
            )[:40]
        except SourceCooldownError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:
            return f"ERROR: OpenAlex references failed: {exc}"

        if not ref_ids:
            return f"No references listed for {_wid}."

        # Strip URL prefixes to get bare W-ids for the filter
        bare_ids = [
            r.replace("https://openalex.org/", "")
            for r in ref_ids
        ]
        filter_str = "openalex_id:" + "|".join(bare_ids)
        try:
            hydrate_resp = _robust_get(
                "https://api.openalex.org/works",
                params={
                    "filter": filter_str,
                    "per-page": 50,
                    "select": (
                        "id,title,publication_year,doi"
                        ",cited_by_count,best_oa_location"
                        ",open_access,primary_location"
                    ),
                },
                headers=_oa_headers,
                cache_dir=cache_dir,
            )
            works = hydrate_resp.json().get("results", [])
        except SourceCooldownError as exc:
            return f"ERROR: {exc}"
        except Exception as exc:
            return f"ERROR: OpenAlex reference hydration failed: {exc}"

        out = []
        for w in works:
            doi = (w.get("doi") or "").replace(
                "https://doi.org/", ""
            )
            boa = w.get("best_oa_location") or {}
            loc = w.get("primary_location") or {}
            oa = w.get("open_access") or {}
            pdf_url = (
                boa.get("pdf_url")
                or oa.get("oa_url")
                or loc.get("pdf_url")
                or loc.get("landing_page_url")
                or ""
            )
            out.append({
                "id": w.get("id", ""),
                "title": w.get("title", ""),
                "year": w.get("publication_year", ""),
                "cited_by_count": w.get("cited_by_count", 0),
                "doi": doi,
                "pdf_url": pdf_url,
            })
        return _j.dumps(out, indent=2)

    return {
        "search_openalex": search_openalex,
        "get_openalex_citations": get_openalex_citations,
        "get_openalex_references": get_openalex_references,
    }
