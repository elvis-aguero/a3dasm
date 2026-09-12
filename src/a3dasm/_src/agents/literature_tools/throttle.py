"""Semantic Scholar client throttle and result capping.

The semanticscholar client library issues its own HTTP requests, so it
is paced here against the same per-domain rate limiter + circuit
breaker that literature/http_client uses for its direct fetches.
"""

from __future__ import annotations

import time


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
# http_client's _robust_get/_robust_post — but it hits the exact same
# remote quota as api.semanticscholar.org fetches made through those helpers
# (get_semantic_scholar_recommendations, citation-graph lookups). Both paths
# share ONE per-domain rate limiter + circuit breaker (http_client's
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
    """Call fn via _call_in_fresh_thread, sharing http_client's
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
    # Deferred import: literature/ is an optional-dependency package
    # (see the try/except ImportError around its import a few frames up the
    # call stack) — by the time _throttled_ss is ever actually called, that
    # import has already succeeded (the whole tool set returns {} otherwise).
    from tenacity import RetryError

    from ...literature.http_client import (
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
