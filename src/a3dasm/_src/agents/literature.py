"""LiteratureReviewAgent — specialist for scientific literature."""

from __future__ import annotations

import logging
import os
import time

from ..backends.base import Agent


def _call_in_fresh_thread(fn, *args, timeout=30.0, **kwargs):
    """Run *fn* in a thread with no running event loop.

    The semanticscholar sync client manages its own asyncio loop and
    breaks when called from within an already-running loop (the
    Claude SDK closure context). A fresh thread has no loop.

    Deliberately does NOT use `with ThreadPoolExecutor(...) as pool:` —
    that form's __exit__ calls shutdown(wait=True), which BLOCKS until the
    worker thread actually finishes, even after future.result(timeout=...)
    has already raised TimeoutError to us. If fn is genuinely stuck (e.g.
    the semanticscholar library's own internal 429 retry — 10 attempts,
    5-60s exponential backoff each — can legitimately run several minutes
    on ONE call), that silently re-introduces an unbounded wait on top of
    the timeout this function exists to enforce, hanging the whole agent
    turn. Observed directly: search_semantic_scholar hung 10+ minutes
    despite this function's 30s timeout, traced to exactly this. shutdown
    (wait=False) lets the caller return the instant the timeout fires; the
    one leaked worker thread finishing on its own later is an acceptable
    trade for never hanging the caller.
    """
    import concurrent.futures as _cf
    pool = _cf.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args, **kwargs)
    try:
        result = future.result(timeout=timeout)
    except _cf.TimeoutError as exc:
        pool.shutdown(wait=False)
        # On Python 3.10 concurrent.futures.TimeoutError is a DISTINCT class
        # from builtins.TimeoutError (they were merged in 3.11). Normalise to
        # builtins.TimeoutError so every caller's `except TimeoutError`
        # catches a pool timeout on all supported Python versions.
        raise TimeoutError(str(exc)) from exc
    pool.shutdown(wait=False)
    return result


# ---------------------------------------------------------------------------
# Semantic Scholar rate throttle
# ---------------------------------------------------------------------------
# The semanticscholar CLIENT LIBRARY issues its own HTTP requests, bypassing
# literature_corpus's _robust_get/_robust_post — but it hits the exact same
# remote quota as api.semanticscholar.org fetches made through those helpers
# (get_semantic_scholar_recommendations, citation-graph lookups). Both paths
# share ONE per-domain rate limiter + circuit breaker (literature_corpus's
# _rate_limit_wait/_record_429/_reset_429) so a 429 seen by either path
# counts against the same cooldown, instead of the client-library path
# tracking nothing and hard-failing on the first 403/429 it hits.
_SS_DOMAIN = "api.semanticscholar.org"
_SS_MAX_RETRIES = 3
# The search/paper/author/citations endpoints this module calls are ~100
# req/5min authenticated (~1 per 3s) — stricter than the domain's shared
# 1.0s default (calibrated for the recommendations endpoint's looser quota
# via _robust_post). Passing this override to _rate_limit_wait keeps the
# shared circuit breaker while pacing THIS traffic at its own real limit —
# using the domain default here silently paced 3x too fast, which is real
# heavy throttling even WITH a correctly-configured, correctly-resolved
# SEMANTIC_SCHOLAR_API_KEY (a key raises the ceiling; it does not exempt you
# from the pacing needed to stay under it over a sustained session).
_SS_MIN_INTERVAL = 3.0

# A single raw search/read payload can be enormous (full paper text, hundreds of
# hits) and overflow the tool-result token cap — the harness then drops the whole
# result and the reviewer loses the search (observed every run, e.g. "exceeds
# maximum allowed tokens"). Cap each result so the reviewer always gets a usable
# (if truncated) payload; deep reads go through targeted reads, not bulk search.
_MAX_RESULT_CHARS: int = 6000


def _cap_result(result) -> str:
    """Truncate an oversized search/read payload with a clear marker."""
    s = str(result)
    if len(s) <= _MAX_RESULT_CHARS:
        return s
    return (s[:_MAX_RESULT_CHARS]
            + f"\n\n[...truncated {len(s) - _MAX_RESULT_CHARS} chars — this "
            "result was too large to return whole. Narrow the query, or read a "
            "specific paper by id instead of bulk-searching.]")


def _throttled_ss(fn, *args, **kwargs):
    """Call fn via _call_in_fresh_thread, sharing literature_corpus's
    per-domain rate limiter + circuit breaker for api.semanticscholar.org.

    429 (the semanticscholar library raises ConnectionRefusedError for HTTP
    429 — "too many requests", transient) retries with exponential backoff,
    same discipline as _robust_get/_robust_post; three consecutive trips the
    shared breaker (SourceCooldownError, same message _robust_get raises).
    403 (raised as PermissionError — the shared UNAUTHENTICATED quota is
    exhausted) is NOT retried here, mirroring _robust_get's "4xx (non-429):
    raise immediately, no retry" rule: waiting a few seconds does not help a
    quota that resets on a much longer window.

    fn is constructed with retry=False (see the _SS(...) construction site),
    which routes through the library's OWN tenacity wrapper with
    stop_after_attempt(1) rather than bypassing tenacity entirely — so a
    failure still surfaces as tenacity.RetryError wrapping the real
    exception, not the real exception directly. Unwrapped here (inside the
    background thread, before it crosses back to the caller) via
    RetryError.reraise(), so the except clauses below see the actual
    ConnectionRefusedError/PermissionError exactly as if retry= didn't
    exist, with no duplicated handling logic.
    """
    # Deferred import: literature_corpus is an optional-dependency module
    # (see the try/except ImportError around its import a few frames up the
    # call stack) — by the time _throttled_ss is ever actually called, that
    # import has already succeeded (the whole tool set returns {} otherwise).
    from tenacity import RetryError

    from ..literature_corpus import (
        SourceCooldownError,
        _cooldown_message,
        _rate_limit_wait,
        _record_429,
        _reset_429,
    )

    def _fn_unwrapping_retry_error(*a, **kw):
        try:
            return fn(*a, **kw)
        except RetryError as exc:
            exc.reraise()

    last_exc: Exception | None = None
    for attempt in range(_SS_MAX_RETRIES):
        _rate_limit_wait(_SS_DOMAIN, _SS_MIN_INTERVAL)  # may raise SourceCooldownError
        try:
            result = _call_in_fresh_thread(
                _fn_unwrapping_retry_error, *args, **kwargs)
        except ConnectionRefusedError as exc:
            last_exc = exc
            if _record_429(_SS_DOMAIN):
                raise SourceCooldownError(
                    _cooldown_message(_SS_DOMAIN)) from exc
            if attempt < _SS_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            continue
        _reset_429(_SS_DOMAIN)
        return result
    raise last_exc


log = logging.getLogger(__name__)

# Module-level constant kept for backward compatibility with
# agent_prompts.py re-export.
LITERATURE_REVIEW_SYSTEM_PROMPT = """\
<role>
You are the Literature Reviewer. You answer specific research questions
by building a corpus of primary literature and quoting exact passages.

NEVER cite memory — corpus quotes only. Format: > "..." — Author et al., Year, p. X
If the corpus does not contain evidence, write: "Not found in corpus."
</role>

<primary_source_rule>
Quote ONLY from full-text papers. Abstract-only corpus entries are leads, not
sources — a corpus search will not return their text.

Acquisition chain: SEARCH the databases → DOWNLOAD or read a paper's full text
→ ADD it to the corpus → SEARCH the corpus for quotable passages. Until a paper
is in the corpus from full text (>5000 chars), do not quote from it. (The exact
tool for each step is in the <tools> catalog below.)
</primary_source_rule>

<tools_note>
Your exact, callable tools are listed in the <tools> catalog appended to this
prompt — that is the single authoritative source, generated from the tools the
runtime actually registered. Call tools by the EXACT names shown there; do not
guess names. The catalog covers your three capabilities: literature SEARCH
(arXiv, Semantic Scholar, OpenAlex — and citation-graph traversal), PAPER
ACQUISITION (download a PDF / read a paper directly), and the CORPUS (add a
local full-text file, then search/rank/list its passages).

The corpus lives under delegations/literature/ in the run's debug dir
(corpus.csv = metadata index; papers/{id}/paper.md = page-annotated text).
</tools_note>

<workflow>
1. Expand the question into 3-5 domain keywords and SEARCH all three literature
   databases (arXiv, Semantic Scholar, OpenAlex — OpenAlex is strongest for
   non-arXiv journals like JMPS / CMAME / Acta Materialia). These are SLOW
   external calls, so fan them out CONCURRENTLY: fire each provider's search
   with wait=False (returns a handle immediately) so different providers run in
   parallel, then gather all results in one collect step before reading them
   (the collect tool and exact names are in the <tools> catalog). Only
   same-provider calls serialize. Note any pdf_url.
2. For each relevant paper, ACQUIRE its full text — read it directly, or
   download the PDF — then ADD it to the corpus. Until a paper is in the corpus
   from full text (>5000 chars), you may not quote it.
3. SEARCH the corpus for passages (try multiple phrasings; re-rank when merging
   results from several searches).
4. Quote verbatim with a citation (Author et al., Year, p. X); never paraphrase.
5. If no passage answers a question, say "Not found in corpus." and list the
   queries you tried.
</workflow>

<operating_principles>
1. VERBATIM QUOTES ONLY: Copy exactly; never paraphrase.
   Cite every quote: Author et al., Year, p. X

2. CITATION REQUIRED
   Every factual claim in Key findings and Conclusions needs a citation.
   Never cite a paper you have not added to the corpus and read.

3. CORPUS FIRST: Try multiple phrasings before concluding "not found".

4. NO MEMORY SYNTHESIS: Don't fill gaps with background knowledge.
   "Not found in corpus." is valid.

5. RATE LIMIT HANDLING: If a tool returns ERROR containing
   "rate-limited", switch to a different source (e.g. arXiv instead of
   Semantic Scholar) or wait before retrying.
</operating_principles>

<output_format>
## Report

### Papers reviewed
- {paper_id}: Author et al. (Year). "Title". Venue/Source.

### Key findings
**Q: {question from delegation}**
> "{exact verbatim quote}" — Author et al., Year, p. X
Relevance: {one sentence}.

OR: Not found in corpus. Searched for: {list of queries tried}.

### Conclusions

### Numbers
questions_addressed: N
papers_consulted: M
new_papers_added: K
quotes_used: Q

### Retrospective
This audits the SYSTEM you worked within — its instructions, contracts, and
tools — NOT your findings. Be concrete; quote specifics. Exactly:
- CONSISTENCY: ok | flagged — did any instruction, contract, or message
  contradict another, or contradict what you were told elsewhere? Write
  "flagged" and QUOTE both conflicting sides; otherwise "ok". (Highest
  priority.)
- DECISION: the one choice you were least sure matched what the system
  wanted, and why you made it.
- FRICTION: anything counterintuitive or unclear about the tools/contracts,
  or "none". (Lowest priority.)
- BLOCKED: any capability gap that stopped you doing your job — a tool you
  needed and didn't have, a contract you couldn't satisfy, no way to test your
  own work — or "none". Name it specifically; an unreported gap can't be fixed.
</output_format>
"""


def _make_search_async_pool():
    """Factory → (asyncable, CollectSearches) sharing one fresh registry.

    External-provider calls (OpenAlex / Semantic Scholar / arXiv / PDF fetch)
    dominate literature wall-time — a single OpenAlex search can take minutes.
    `asyncable(provider, fn)` wraps a synchronous tool so the model can run it
    sync or async (the wait param, like Delegate's):
      • wait=True (DEFAULT): blocks and returns the result inline (the existing
        contract — a search returns results, not a handle).
      • wait=False: runs in a background thread, returns a handle immediately so
        the reviewer can fire independent searches on OTHER providers
        concurrently, then gather them with CollectSearches.
    Calls to the SAME provider serialize (one lock per provider — rate-limit
    safety); different providers run in parallel. `CollectSearches(handle?)`
    blocks until the named handle (or ALL pending) finish and returns results.
    A fresh pool is created per delegation (no cross-run state).
    """
    import inspect
    import itertools
    import threading

    prov_locks: dict = {}
    prov_guard = threading.Lock()
    reg: dict = {}
    reg_guard = threading.Lock()
    seq = itertools.count(1)

    def provider_lock(p):
        with prov_guard:
            return prov_locks.setdefault(p, threading.Lock())

    def asyncable(provider, fn):
        base_sig = inspect.signature(fn)
        wait_p = inspect.Parameter("wait", inspect.Parameter.KEYWORD_ONLY,
                                   default=True, annotation=bool)

        def wrapper(*args, **kwargs):
            _w = kwargs.pop("wait", True)
            # MCP string-in tools may pass "false"; coerce like Delegate.wait.
            wait = (_w if isinstance(_w, bool)
                    else str(_w).strip().lower() not in ("false", "0", "no", ""))
            if wait:
                with provider_lock(provider):
                    return _cap_result(fn(*args, **kwargs))
            h = f"{provider}#{next(seq)}"
            label = str((args[0] if args else None) or kwargs.get("query")
                        or kwargs.get("url") or kwargs.get("paper_id") or "")[:60]
            ev = threading.Event()
            rec = {"event": ev, "result": None, "label": label}
            with reg_guard:
                reg[h] = rec

            def run():
                try:
                    with provider_lock(provider):
                        rec["result"] = fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    rec["result"] = f"ERROR: {exc}"
                ev.set()

            threading.Thread(target=run, daemon=True, name=h).start()
            return (f"Started async on '{provider}' (handle {h}). Fire more "
                    "searches on OTHER providers now — they run concurrently — "
                    "then CollectSearches() to get all results before using them.")

        wrapper.__name__ = getattr(fn, "__name__", "tool")
        _doc = (fn.__doc__ or "").rstrip()
        wrapper.__doc__ = _doc + (
            "\n\nASYNC: pass wait=False to run this in the background and get a "
            "handle immediately, so you can fire independent searches on OTHER "
            "providers concurrently, then CollectSearches() for results; "
            "same-provider calls serialize. Default wait=True blocks and returns "
            "the result inline.")
        try:
            wrapper.__signature__ = base_sig.replace(
                parameters=list(base_sig.parameters.values()) + [wait_p])
        except (ValueError, TypeError):
            pass
        return wrapper

    def CollectSearches(handle: str = None) -> str:
        """Collect async (wait=False) search results. With a handle: block until
        that search finishes and return its result. With NO handle: block until
        ALL pending async searches finish and return them all. Call this before
        using async search results — same-provider searches were serialized,
        different providers ran concurrently."""
        with reg_guard:
            handles = [handle] if handle else list(reg)
        if not handles:
            return "No async searches pending."
        parts = []
        for h in handles:
            rec = reg.get(h)
            if rec is None:
                parts.append(f"{h}: unknown or already-collected handle")
                continue
            done = rec["event"].wait(timeout=600)
            parts.append(f"=== {h} ({rec['label']}) ===\n" + (
                _cap_result(rec["result"]) if done else "(still running after 600s)"))
            with reg_guard:
                reg.pop(h, None)
        return "\n\n".join(parts)

    return asyncable, CollectSearches


class LiteratureReviewAgent(Agent):
    """Literature reviewer: answers epistemic questions from a corpus.

    Owns debug/lit_reviewer_notes/corpus.csv and papers/.
    Never answers from memory — all claims must cite exact passages.
    inject_problem_statement=True ensures research domain is visible.
    """

    inject_problem_statement = True
    # Declare the role explicitly — the base default is "implementer", and
    # inheriting it makes implementer-only logic (the milestone-backlog nudge,
    # the eval-parallelism resource nudge, role telemetry) mis-fire on the
    # literature reviewer.
    role = "literature_reviewer"
    tools = frozenset({"Read", "Grep", "Glob"})
    reset_on_checkpoint = True
    description = (
        "Searches and synthesises primary scientific literature to"
        " answer epistemic questions: what methods exist, what has"
        " been tried, what the field recommends. Use before committing"
        " to a strategy you are uncertain about, or when you need to"
        " know the state of the art. Never for questions answerable"
        " from workspace data."
    )
    report_sections = (
        "### Papers reviewed",
        "### Key findings",
        "### Conclusions",
        "### Numbers",
        "### Retrospective",
    )
    mcp_servers = {}
    extra_allowed_tools = frozenset()

    system_prompt = LITERATURE_REVIEW_SYSTEM_PROMPT

    def build_closure_tools(
        self,
        study_dir,
        delegation_id=None,
        lit_reviewer_notes_dir=None,
    ):
        """Inject corpus + discovery tools as runtime closures."""
        import json as _json
        from pathlib import Path as _Path

        try:
            from ..literature_corpus import (
                LiteratureCorpus,
                SourceCooldownError,
                _robust_get,
                _robust_post,
            )
        except ImportError:
            return {}

        corpus_dir = (
            _Path(lit_reviewer_notes_dir)
            if lit_reviewer_notes_dir is not None
            else _Path(study_dir) / "delegations" / "literature"
        )
        corpus = LiteratureCorpus(corpus_dir)
        cache_dir = corpus._http_cache_dir

        # Defined as named functions (not lambdas) so each carries a docstring:
        # the generated <tools> catalog renders these, making it the single
        # source of tool docs — no hand-written list in the prompt to drift.
        def CorpusAdd(source, title="", authors="", year="", doi="",
                      arxiv_id="", venue="", abstract="", citation_count=0):
            """Index a LOCAL file (a saved PDF or full-text markdown) into the
            corpus so its passages become searchable. citation_count boosts BM25
            retrieval weight (log10(c+1) scaling) — pass the citationCount from
            Semantic Scholar or OpenAlex."""
            return corpus.add(
                source, title=title, authors=authors, year=year, doi=doi,
                arxiv_id=arxiv_id, venue=venue, abstract=abstract,
                citation_count=int(citation_count or 0))

        def CorpusSearch(query, top_k=10):
            """Passage search across the FULL-TEXT papers in the corpus only.
            Returns an ERROR string if no full-text papers have been added yet —
            add papers first via the search → download → CorpusAdd chain."""
            return corpus.search(query, int(top_k))

        def CorpusGetPaper(paper_id):
            """Return the full extracted (page-annotated) text of one corpus paper."""
            return corpus.get_paper(paper_id)

        def CorpusList():
            """List corpus metadata — each paper tagged [full-text] or
            [abstract-only] so you know which you may quote from."""
            return corpus.list_papers()

        tools = {
            "CorpusAdd": CorpusAdd, "CorpusSearch": CorpusSearch,
            "CorpusGetPaper": CorpusGetPaper, "CorpusList": CorpusList,
        }

        # Semantic Scholar tools via the semanticscholar library.
        try:
            from semanticscholar import SemanticScholar as _SS

            from ..settings import get_str
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
            # pacing/backoff/circuit-breaker (literature_corpus's
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
                """Get details for a paper by S2/DOI/arxiv ID."""
                try:
                    paper = _throttled_ss(
                        _sch.get_paper,
                        paper_id,
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
                """Get citing papers and references (≤20 each)."""
                try:
                    paper = _throttled_ss(
                        _sch.get_paper,
                        paper_id,
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

        def get_semantic_scholar_recommendations(
            paper_id: str, n_results: int = 10
        ) -> str:
            """Find semantically similar papers (no citation link).

            paper_id: S2 paperId, DOI, or 'arXiv:XXXX.XXXXX'.
            """
            import json as _j
            try:
                resp = _robust_post(
                    "https://api.semanticscholar.org"
                    "/recommendations/v1/papers/",
                    json={"positivePaperIds": [paper_id]},
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

        tools["search_openalex"] = search_openalex
        tools["get_openalex_citations"] = get_openalex_citations
        tools["get_openalex_references"] = get_openalex_references
        tools["get_semantic_scholar_recommendations"] = (
            get_semantic_scholar_recommendations
        )
        tools["CorpusRank"] = CorpusRank
        tools["DownloadPdf"] = DownloadPdf

        # arxiv tools — Python-native, same for Claude and Ollama.
        from ..backends.ollama import _build_arxiv_closures
        tools.update(_build_arxiv_closures())

        # Make the SLOW external-provider tools async-able (wait=False default)
        # so the reviewer fans out across providers concurrently instead of
        # blocking ~minutes per call. Same-provider calls serialize. The fast
        # local corpus tools (CorpusAdd/Search/List/Rank, Read) stay synchronous.
        _asyncable, _collect = _make_search_async_pool()
        _provider_of = {
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
        for _nm, _pv in _provider_of.items():
            if _nm in tools:
                tools[_nm] = _asyncable(_pv, tools[_nm])
        tools["CollectSearches"] = _collect

        return tools
