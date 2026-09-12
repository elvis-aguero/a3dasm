"""Robust HTTP for the literature providers: pacing, breaker, cache.

One process-wide per-domain rate limiter and circuit breaker shared by
every caller that talks to a literature API — the direct fetches in
``_robust_get``/``_robust_post`` here and the client-library paths in
``agents/literature_tools`` (see ``throttle.py``). Sharing ONE state table
is what keeps combined traffic to the same host under its real quota.

``_sleep`` is module-level so tests can monkeypatch it and run the retry
and pacing logic without real delays.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time as _time_module
from pathlib import Path
from typing import Optional

__all__ = [
    "SourceCooldownError",
    "_robust_get",
    "_robust_post",
]

# ---------------------------------------------------------------------------
# Injectable sleep — monkeypatch in tests to avoid real delays
# ---------------------------------------------------------------------------
_sleep = _time_module.sleep

# ---------------------------------------------------------------------------
# Per-domain rate limiting (min-interval + circuit breaker)
# ---------------------------------------------------------------------------

_DOMAIN_MIN_INTERVAL: dict[str, float] = {
    "api.semanticscholar.org": 1.0,
    "api.openalex.org": 0.2,
}
_ARXIV_MIN_INTERVAL = 3.0
_DEFAULT_MIN_INTERVAL = 0.5

_COOLDOWN_SECONDS = 60
_CIRCUIT_BREAKER_THRESHOLD = 3  # consecutive 429s before cooldown

# Protected by _rate_lock
_domain_last_request: dict[str, float] = {}
_domain_consecutive_429: dict[str, int] = {}
_domain_cooldown_until: dict[str, float] = {}
_rate_lock = threading.Lock()


class SourceCooldownError(Exception):
    """Raised when a domain is in circuit-breaker cooldown."""


def _domain_key(url: str) -> str:
    """Extract domain from URL for rate-limiting purposes."""
    # Simple extraction without urllib to keep it lightweight
    # e.g. "https://api.openalex.org/works" → "api.openalex.org"
    try:
        no_scheme = url.split("://", 1)[1]
        domain = no_scheme.split("/")[0].split("?")[0]
        return domain
    except (IndexError, AttributeError):
        return url


def _min_interval_for(domain: str) -> float:
    if "arxiv.org" in domain:
        return _ARXIV_MIN_INTERVAL
    return _DOMAIN_MIN_INTERVAL.get(domain, _DEFAULT_MIN_INTERVAL)


def _rate_limit_wait(domain: str, min_interval: float | None = None) -> None:
    """Sleep if needed to honour min-interval for *domain*.

    Must be called OUTSIDE the rate lock to avoid holding the lock
    during sleep.  We use a two-step approach: read under lock,
    sleep outside, then record under lock.

    min_interval overrides the domain's registered default (_DOMAIN_MIN_
    INTERVAL) for THIS call only — the shared _domain_last_request timestamp
    and circuit breaker state are untouched, so combined traffic to the same
    domain from a stricter caller and a looser one still paces off one
    another correctly. Needed because different endpoints under the same
    domain can have different real quotas (e.g. api.semanticscholar.org's
    search/paper-details endpoints are ~100 req/5min authenticated — needing
    ~3s between calls — while its recommendations endpoint tolerates the
    domain's looser 1.0s default); passing the domain default here for
    every caller would silently under-pace the stricter endpoint.
    """
    with _rate_lock:
        now = _time_module.monotonic()
        # Circuit breaker check
        until = _domain_cooldown_until.get(domain, 0.0)
        if until > now:
            remaining = int(until - now)
            raise SourceCooldownError(
                f"{domain} is rate-limited (cooldown {_COOLDOWN_SECONDS}s"
                f" remaining: {remaining}s). Use a different literature"
                " source or retry later."
            )
        # Rate-limit wait calculation
        last = _domain_last_request.get(domain, 0.0)
        if min_interval is None:
            min_interval = _min_interval_for(domain)
        jitter = random.uniform(0, 1.0) if "arxiv.org" in domain else 0.0
        required = min_interval + jitter
        wait = required - (now - last)

    if wait > 0:
        _sleep(wait)

    with _rate_lock:
        _domain_last_request[domain] = _time_module.monotonic()


def _record_429(domain: str) -> bool:
    """Record a 429 for domain; return True if circuit breaker fires."""
    with _rate_lock:
        count = _domain_consecutive_429.get(domain, 0) + 1
        _domain_consecutive_429[domain] = count
        if count >= _CIRCUIT_BREAKER_THRESHOLD:
            _domain_cooldown_until[domain] = (
                _time_module.monotonic() + _COOLDOWN_SECONDS
            )
            return True
        return False


def _reset_429(domain: str) -> None:
    """Reset consecutive 429 counter on a successful request."""
    with _rate_lock:
        _domain_consecutive_429.pop(domain, None)


def _cooldown_message(domain: str) -> str:
    """Human-readable remaining-cooldown text for domain's circuit breaker.

    Shared by every caller that trips the breaker (direct HTTP fetches here
    and the semanticscholar client-library path in agents/literature.py) so
    both report the SAME remaining time for the SAME underlying quota.
    """
    with _rate_lock:
        remaining = int(
            _domain_cooldown_until.get(domain, 0.0)
            - _time_module.monotonic()
        )
    return (
        f"{domain} is rate-limited (cooldown {_COOLDOWN_SECONDS}s remaining: "
        f"{max(remaining, 0)}s). Use a different literature source or retry "
        "later."
    )


# ---------------------------------------------------------------------------
# On-disk HTTP GET cache
# ---------------------------------------------------------------------------

class _CachedResponse:
    """Lightweight response-like object from cache."""

    def __init__(self, status_code: int, text: str,
                 content_type: str) -> None:
        self.status_code = status_code
        self._text = text
        self._content_type = content_type

    @property
    def text(self) -> str:
        return self._text

    @property
    def content(self) -> bytes:
        return self._text.encode("utf-8", errors="replace")

    def json(self):
        return json.loads(self._text)


def _cache_key(url: str, params) -> str:
    parts = url
    if params:
        if isinstance(params, dict):
            parts += json.dumps(
                sorted(params.items()), separators=(",", ":")
            )
        else:
            parts += str(params)
    return hashlib.sha256(parts.encode()).hexdigest()


def _cache_get(
    cache_dir: Optional[Path], url: str, params, ttl: float
) -> Optional[_CachedResponse]:
    if cache_dir is None:
        return None
    key = _cache_key(url, params)
    cache_file = cache_dir / (key + ".json")
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        age = _time_module.time() - data.get("ts", 0)
        if age > ttl:
            return None  # expired
        return _CachedResponse(
            data["status"],
            data["body"],
            data.get("content_type", ""),
        )
    except Exception:
        return None


def _cache_put(
    cache_dir: Optional[Path],
    url: str,
    params,
    status: int,
    body: str,
    content_type: str,
) -> None:
    if cache_dir is None or status != 200:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(url, params)
        cache_file = cache_dir / (key + ".json")
        cache_file.write_text(
            json.dumps({
                "status": status,
                "body": body,
                "content_type": content_type,
                "ts": _time_module.time(),
            }),
            encoding="utf-8",
        )
    except Exception:
        pass  # cache write failure is non-fatal


# ---------------------------------------------------------------------------
# Robust HTTP helpers (retry + rate limiting + 429 discipline + cache)
# ---------------------------------------------------------------------------

def _robust_get(
    url: str,
    *,
    params=None,
    headers=None,
    retries: int = 3,
    timeout: float = 15.0,
    cache_dir: Optional[Path] = None,
    ttl: float = 86400,
):
    """GET *url* with per-domain rate limiting, 429 discipline, and cache.

    Cache check happens BEFORE rate limiting — a hit never sleeps.
    Returns a :class:`requests.Response` (or cached equivalent) on
    success; raises :class:`SourceCooldownError` if circuit breaker is
    active; raises on final failure.
    """
    import requests  # type: ignore[import]

    # Cache check first (no rate-limit sleep on hit)
    cached = _cache_get(cache_dir, url, params, ttl)
    if cached is not None:
        return cached

    domain = _domain_key(url)
    last_exc: Exception | None = None
    for attempt in range(retries):
        # May raise SourceCooldownError — let it propagate immediately
        _rate_limit_wait(domain)
        try:
            resp = requests.get(
                url, params=params, headers=headers, timeout=timeout
            )
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                _sleep(2 ** attempt)
            continue

        if resp.status_code in (429, 503):
            fired = _record_429(domain)
            retry_after_str = resp.headers.get("Retry-After")
            if retry_after_str is not None:
                try:
                    wait = min(float(retry_after_str), 60.0)
                except ValueError:
                    wait = 2 ** attempt
            else:
                wait = 2 ** attempt
            if fired:
                # Circuit breaker just fired
                raise SourceCooldownError(_cooldown_message(domain))
            last_exc = Exception(
                f"HTTP {resp.status_code} from {domain}"
            )
            if attempt < retries - 1:
                _sleep(wait)
            continue

        if 400 <= resp.status_code < 500:
            # 4xx (non-429): raise immediately, no retry
            resp.raise_for_status()

        # 5xx or success
        try:
            resp.raise_for_status()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                _sleep(2 ** attempt)
            continue

        # Success
        _reset_429(domain)
        _cache_put(
            cache_dir,
            url,
            params,
            resp.status_code,
            resp.text,
            resp.headers.get("Content-Type", ""),
        )
        return resp

    raise last_exc  # type: ignore[misc]


def _robust_post(
    url: str,
    *,
    json=None,
    params=None,
    headers=None,
    retries: int = 3,
    timeout: float = 15.0,
):
    """POST *url* with per-domain rate limiting and 429 discipline.

    Returns a :class:`requests.Response` on success; raises
    :class:`SourceCooldownError` if circuit breaker is active; raises
    on final failure.
    """
    import requests as _requests  # type: ignore[import]

    domain = _domain_key(url)
    last_exc: Exception | None = None
    for attempt in range(retries):
        _rate_limit_wait(domain)
        try:
            resp = _requests.post(
                url, json=json, params=params,
                headers=headers, timeout=timeout,
            )
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                _sleep(2 ** attempt)
            continue

        if resp.status_code in (429, 503):
            fired = _record_429(domain)
            retry_after_str = resp.headers.get("Retry-After")
            if retry_after_str is not None:
                try:
                    wait = min(float(retry_after_str), 60.0)
                except ValueError:
                    wait = 2 ** attempt
            else:
                wait = 2 ** attempt
            if fired:
                raise SourceCooldownError(_cooldown_message(domain))
            last_exc = Exception(
                f"HTTP {resp.status_code} from {domain}"
            )
            if attempt < retries - 1:
                _sleep(wait)
            continue

        if 400 <= resp.status_code < 500:
            resp.raise_for_status()

        try:
            resp.raise_for_status()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                _sleep(2 ** attempt)
            continue

        _reset_429(domain)
        return resp

    raise last_exc  # type: ignore[misc]
