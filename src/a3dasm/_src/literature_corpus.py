"""LiteratureCorpus — owns a canonical paper corpus on disk.

Manages a directory with::

    corpus_dir/
        corpus.csv          — metadata index
        papers/
            <paper_id>/
                paper.pdf   — source file (optional copy)
                paper.md    — extracted text, page-annotated
        .http_cache/        — on-disk GET response cache (TTL 24h)

**Design principle:** this module does NO network I/O in the corpus
methods.  Discovery and download are the agent's responsibility via
MCP tools or ``DownloadPdf``.  ``CorpusAdd`` only accepts paths to
files already on disk.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
import shutil
import threading
import time as _time_module
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import fitz  # type: ignore[import]
    # Silence MuPDF's per-object C-library stderr ("zlib error: incorrect header
    # check" on PDFs with non-standard FlateDecode streams). These are non-fatal
    # — fitz still extracts the readable text — but they spam the run log. Genuine
    # extraction failure is detected by the extracted-text length instead (see
    # add()), not by these warnings.
    try:
        fitz.TOOLS.mupdf_display_errors(False)
    except Exception:  # noqa: BLE001 — older/newer pymupdf may differ
        pass
except ImportError:
    fitz = None  # type: ignore[assignment]

__all__ = [
    "LiteratureCorpus",
    "SourceCooldownError",
    "_robust_get",
    "_robust_post",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Injectable sleep — monkeypatch in tests to avoid real delays
# ---------------------------------------------------------------------------
_sleep = _time_module.sleep

# ---------------------------------------------------------------------------
# Out-of-process embedder (isolated Python 3.12 env via uv)
# ---------------------------------------------------------------------------

# fastembed runs bge-small via onnxruntime (light: ONNX, no torch). onnxruntime
# ships no CPython-3.13 wheel, so it cannot import in a 3.13 host. The worker
# runs in an ISOLATED uv env, so we just pin that env to Python 3.12 (where
# fastembed+onnxruntime+numpy all have wheels) — no host pins needed, and the
# host's numpy-2 is irrelevant across the process boundary.
_EMBED_WITH = "fastembed>=0.3"
_EMBED_PYTHON = "3.12"

# Tri-state cache for subprocess embedder availability:
#   None   → not yet probed
#   instance of _SubprocessEmbedder → available and ready
#   False  → probed and unavailable (log once, never re-probe)
_subprocess_embedder_state: _SubprocessEmbedder | None | bool = None
_subprocess_embedder_warned = False


class _SubprocessEmbedder:
    """.embed(texts) via _embed_worker.py in an ephemeral uv env.

    Used when fastembed cannot import in-process (NumPy-2 host env
    with only NumPy-1.x onnxruntime wheels, e.g. Intel macOS). The
    ephemeral env is resolved once by uv and cached; the model cache
    persists across calls. First call may take minutes (downloads).
    """

    def __init__(self, timeout: float = 600.0) -> None:
        self._timeout = timeout

    def embed(self, texts):
        import json as _json
        import subprocess
        from pathlib import Path as _P
        worker = _P(__file__).parent / "_embed_worker.py"
        cmd = [
            "uv", "run", "--no-project", "--quiet",
            "--python", _EMBED_PYTHON,
            "--with", _EMBED_WITH,
            "python", str(worker),
        ]
        proc = subprocess.run(
            cmd, input=_json.dumps({"texts": list(texts)}),
            capture_output=True, text=True, timeout=self._timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"embed worker failed: {proc.stderr[-500:]}"
            )
        return _json.loads(proc.stdout)["vectors"]


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


def _rate_limit_wait(domain: str) -> None:
    """Sleep if needed to honour min-interval for *domain*.

    Must be called OUTSIDE the rate lock to avoid holding the lock
    during sleep.  We use a two-step approach: read under lock,
    sleep outside, then record under lock.
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


_CSV_FIELDS = [
    "paper_id",
    "title",
    "authors",
    "year",
    "doi",
    "arxiv_id",
    "venue",
    "abstract",
    "local_pdf_path",
    "local_md_path",
    "added_at",
    "source",
    "citation_count",
    "full_text",
]

_ARXIV_BARE_RE = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_DOI_BARE_RE = re.compile(r"^10\.\d{4,}/\S+$")

_FULL_TEXT_MD_THRESHOLD = 5000  # chars; beyond this → real body, not abstract


def _slugify(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "_", s).strip("_")


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


_PUNCT_RE = re.compile(r"[^\w\-]", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase and strip trailing/leading punctuation from each token.

    Preserves hyphens so that ``self-attention`` stays as one token.
    """
    return [
        _PUNCT_RE.sub("", w).lower()
        for w in text.split()
        if _PUNCT_RE.sub("", w)
    ]


class LiteratureCorpus:
    """Local paper corpus: extraction, indexing, search.

    No network I/O — the agent downloads files via MCP tools and hands
    local paths to :meth:`add`.

    Parameters
    ----------
    corpus_dir:
        Root directory.  Created on construction if absent.
    """

    def __init__(self, corpus_dir: Path) -> None:
        self._corpus_dir = Path(corpus_dir)
        self._papers_dir = self._corpus_dir / "papers"
        self._csv_path = self._corpus_dir / "corpus.csv"
        self._chunks_path = self._corpus_dir / "chunks.jsonl"
        self._http_cache_dir = self._corpus_dir / ".http_cache"
        self._lock = threading.Lock()
        self._corpus_dir.mkdir(parents=True, exist_ok=True)
        self._papers_dir.mkdir(parents=True, exist_ok=True)
        self._embedding_model = None  # lazy-loaded on first CorpusAdd
        self._fastembed_warned = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        source: str,
        title: str = "",
        authors: str = "",
        year: str = "",
        doi: str = "",
        arxiv_id: str = "",
        venue: str = "",
        abstract: str = "",
        citation_count: int = 0,
    ) -> str:
        """Add a paper to the corpus from a local file.

        Parameters
        ----------
        source:
            Absolute or relative path to a PDF (``.pdf``) or extracted
            text (``.md`` / ``.txt``) file already on disk.  Download
            the file first with ``arxiv_download_paper`` or write
            the text returned by ``arxiv_read_paper`` via the
            Write tool.
        title, authors, year, doi, arxiv_id, venue, abstract:
            Optional metadata.  Pass values obtained from the MCP
            search result that identified this paper.

        Returns
        -------
        str
            ``paper_id`` on success, ``"Already in corpus: {id}"``,
            or ``"ERROR: …"`` on failure.
        """
        src = Path(source)
        if not src.exists():
            return f"ERROR: file not found: {source!r}. Download it first."

        # Derive a stable paper_id from arxiv_id, doi, or filename
        paper_id = self._derive_paper_id(arxiv_id, doi, src)

        with self._lock:
            rows = self._load_csv()
            if any(r["paper_id"] == paper_id for r in rows):
                return f"Already in corpus: {paper_id}"

        paper_dir = self._paper_dir(paper_id)
        paper_dir.mkdir(parents=True, exist_ok=True)

        # Copy the source file into the corpus directory
        suffix = src.suffix.lower()
        local_pdf_path = ""
        local_md_path = ""
        md_content = ""
        full_text = False

        if suffix == ".pdf":
            dest_pdf = paper_dir / "paper.pdf"
            if src.resolve() != dest_pdf.resolve():
                shutil.copy2(src, dest_pdf)
            local_pdf_path = str(dest_pdf)
            # Extract text
            md_content = self._extract_pdf_to_md(dest_pdf)
            dest_md = paper_dir / "paper.md"
            dest_md.write_text(md_content, encoding="utf-8")
            local_md_path = str(dest_md)
            # Full-text ONLY if extraction produced a real body. A scanned or
            # corrupt PDF can extract near-nothing; the old code hardcoded
            # full_text=True and stored a 236-char placeholder as quotable
            # full-text. Treat a failed extraction as a hard error with an
            # actionable alternative — do NOT enter a phantom paper.
            if len(md_content) <= _FULL_TEXT_MD_THRESHOLD:
                return (
                    f"ERROR: extracted only {len(md_content)} chars from this "
                    f"PDF (need >{_FULL_TEXT_MD_THRESHOLD}) — it is likely "
                    "scanned or has corrupt text streams, so it is NOT quotable "
                    "and was not added. Get the full text another way (e.g. "
                    "arxiv_read_paper for an arXiv id) and add that, or choose a "
                    "different source."
                )
            full_text = True
        elif suffix in {".md", ".txt"}:
            dest_md = paper_dir / "paper.md"
            if src.resolve() != dest_md.resolve():
                shutil.copy2(src, dest_md)
            md_content = dest_md.read_text(encoding="utf-8")
            local_md_path = str(dest_md)
            # Full-text if markdown body exceeds threshold
            full_text = len(md_content) > _FULL_TEXT_MD_THRESHOLD
        else:
            return (
                f"ERROR: unsupported file type {suffix!r}. "
                "Provide a .pdf, .md, or .txt file."
            )

        # Chunk and index the text
        chunks = self._chunk_text(md_content)
        self._append_chunks(paper_id, chunks)

        # Embed chunks and save per-paper .npy for dense search
        model = self._get_embedding_model()
        if model is not None and chunks:
            import numpy as _np
            texts = [c["text"] for c in chunks]
            embs = _np.array(
                list(model.embed(texts)), dtype=_np.float32
            )
            _np.save(str(paper_dir / "chunks.npy"), embs)

        # Infer source label
        source_label = (
            "arxiv" if arxiv_id else ("doi" if doi else "local")
        )

        row = {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "venue": venue,
            "abstract": abstract,
            "local_pdf_path": local_pdf_path,
            "local_md_path": local_md_path,
            "added_at": _now_iso(),
            "source": source_label,
            "citation_count": str(int(citation_count)),
            "full_text": "true" if full_text else "false",
        }

        with self._lock:
            rows = self._load_csv()
            if any(r["paper_id"] == paper_id for r in rows):
                return f"Already in corpus: {paper_id}"
            rows.append(row)
            self._save_csv(rows)

        return paper_id

    def _get_embedding_model(self):
        """Lazy-load bge-small-en-v1.5 via fastembed.

        Resolution order:
        1. In-process fastembed (fast path).
        2. Out-of-process via _embed_worker.py in an ephemeral uv env
           with numpy<2 (for Intel macOS where onnxruntime wheels are
           NumPy-1.x builds).  Probed once per process.
        3. None — BM25-only fallback (warning emitted once when BOTH
           routes fail).
        """
        global _subprocess_embedder_state, _subprocess_embedder_warned

        if self._embedding_model is not None:
            return self._embedding_model

        # --- fast path: in-process fastembed ---
        try:
            from fastembed import TextEmbedding
            self._embedding_model = TextEmbedding(
                "BAAI/bge-small-en-v1.5"
            )
            return self._embedding_model
        except ImportError:
            pass  # fall through to subprocess route
        except Exception:
            self._embedding_model = None
            return self._embedding_model

        # --- subprocess route (probe once per process) ---
        if _subprocess_embedder_state is None:
            # Not yet probed — check uv availability then probe
            import shutil
            if shutil.which("uv") is not None:
                candidate = _SubprocessEmbedder()
                try:
                    candidate.embed(["probe"])
                    _subprocess_embedder_state = candidate
                    log.info(
                        "dense retrieval via out-of-process embed"
                        " worker (ephemeral numpy<2 env)"
                    )
                except Exception:
                    _subprocess_embedder_state = False
            else:
                _subprocess_embedder_state = False

            if _subprocess_embedder_state is False:
                if not _subprocess_embedder_warned:
                    log.warning(
                        "fastembed unavailable in-process and embed"
                        " worker probe failed — CorpusSearch falls"
                        " back to BM25-only"
                    )
                    _subprocess_embedder_warned = True

        if _subprocess_embedder_state is not False and \
                _subprocess_embedder_state is not None:
            self._embedding_model = _subprocess_embedder_state
            return self._embedding_model

        # Both routes failed
        self._embedding_model = None
        return self._embedding_model

    def _load_all_embeddings(self) -> tuple[list[dict], object]:
        """Load all chunks and embeddings.

        Returns (chunks, embeddings_matrix) or (chunks, None).
        """
        import numpy as _np
        chunks = self._load_chunks()
        if not chunks:
            return chunks, None

        # Group chunks by paper_id to load matching .npy files
        paper_embs: dict = {}
        for paper_id in dict.fromkeys(c["paper_id"] for c in chunks):
            npy_path = self._paper_dir(paper_id) / "chunks.npy"
            if npy_path.exists():
                paper_embs[paper_id] = _np.load(str(npy_path))

        if not paper_embs:
            return chunks, None

        # Build aligned embedding matrix (rows match chunks order)
        paper_chunk_idx: dict = {}
        rows = []
        for chunk in chunks:
            pid = chunk["paper_id"]
            if pid not in paper_embs:
                # missing embeddings for at least one paper → fallback
                return chunks, None
            idx = paper_chunk_idx.get(pid, 0)
            if idx >= len(paper_embs[pid]):
                return chunks, None
            rows.append(paper_embs[pid][idx])
            paper_chunk_idx[pid] = idx + 1

        return chunks, _np.array(rows, dtype=_np.float32)

    def search(self, query: str, top_k: int = 10) -> str:
        """Search FULL-TEXT papers for passages relevant to *query*.

        Only chunks belonging to papers with ``full_text=true`` are
        searched.  If the corpus contains no full-text papers, returns
        an actionable error guiding the agent to acquire full text.

        Uses Reciprocal Rank Fusion (BM25 + dense embeddings) when
        both ``rank_bm25`` and ``fastembed`` are installed.  Falls back
        to BM25-only, then substring search.

        Returns up to *top_k* formatted passages with page citations,
        or ``"No results found."``.
        """
        import math as _math

        import numpy as _np

        csv_rows = self._load_csv()
        full_text_ids = {
            r["paper_id"]
            for r in csv_rows
            if r.get("full_text", "false").lower() == "true"
        }

        if not full_text_ids:
            n = len(csv_rows)
            return (
                f"ERROR: corpus contains no full-text papers"
                f" ({n} abstract-only entr{'y' if n == 1 else 'ies'})."
                " Quotes require full text — download the PDF or full"
                " text first (DownloadPdf / arxiv_download_paper),"
                " then CorpusAdd it."
            )

        all_chunks, emb_matrix_all = self._load_all_embeddings()
        if not all_chunks:
            return "No results found."

        # Filter to full-text chunks only
        full_idx = [
            i for i, c in enumerate(all_chunks)
            if c["paper_id"] in full_text_ids
        ]
        if not full_idx:
            return "No results found."

        chunks = [all_chunks[i] for i in full_idx]
        if emb_matrix_all is not None:
            emb_matrix = _np.array(
                [emb_matrix_all[i] for i in full_idx],
                dtype=_np.float32,
            )
        else:
            emb_matrix = None

        citation_counts = {
            r["paper_id"]: int(r.get("citation_count") or 0)
            for r in csv_rows
        }
        meta = {r["paper_id"]: r for r in csv_rows}

        K_RRF = 60  # standard RRF constant

        # --- BM25 ranking ---
        bm25_ranks: dict = {}
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import]
            tokenized = [_tokenize(c["text"]) for c in chunks]
            bm25 = BM25Okapi(tokenized)
            bm25_scores = bm25.get_scores(_tokenize(query))
            # Apply citation weight
            weighted = [
                bm25_scores[i]
                * (1 + _math.log10(
                    citation_counts.get(chunks[i]["paper_id"], 0) + 1
                ))
                for i in range(len(chunks))
            ]
            bm25_order = sorted(
                range(len(chunks)),
                key=lambda i: weighted[i],
                reverse=True,
            )
            for rank, idx in enumerate(bm25_order):
                bm25_ranks[idx] = rank
        except ImportError:
            return self._search_substring(
                query, top_k, full_text_ids=full_text_ids
            )

        # --- Dense ranking (if embeddings available) ---
        dense_ranks: dict = {}
        if emb_matrix is not None:
            model = self._get_embedding_model()
            if model is not None:
                q_emb = _np.array(
                    # model.embed() may return a LIST (not a generator); next()
                    # on a list raises "'list' object is not an iterator".
                    # iter() makes next() work for both forms (audit: this crashed
                    # CorpusSearch dense ranking in the bb3d wet run 20260621).
                    next(iter(model.embed([query]))), dtype=_np.float32
                )
                q_norm = q_emb / (_np.linalg.norm(q_emb) + 1e-9)
                norms = _np.linalg.norm(
                    emb_matrix, axis=1, keepdims=True
                )
                emb_n = emb_matrix / (norms + 1e-9)
                cos_scores = emb_n @ q_norm
                dense_order = _np.argsort(cos_scores)[::-1].tolist()
                for rank, idx in enumerate(dense_order):
                    dense_ranks[idx] = rank

        # --- RRF fusion ---
        if dense_ranks:
            rrf_scores = {
                i: (2.0 / 3.0) / (
                    K_RRF + dense_ranks.get(i, len(chunks))
                )
                + (1.0 / 3.0) / (
                    K_RRF + bm25_ranks.get(i, len(chunks))
                )
                for i in range(len(chunks))
            }
        else:
            rrf_scores = {
                i: 1.0 / (K_RRF + bm25_ranks.get(i, len(chunks)))
                for i in range(len(chunks))
            }

        top_idx = sorted(
            rrf_scores, key=lambda i: rrf_scores[i], reverse=True
        )[:top_k]

        passages = []
        for idx in top_idx:
            if rrf_scores[idx] <= 0:
                break
            chunk = chunks[idx]
            pid = chunk["paper_id"]
            m = meta.get(pid, {})
            title = m.get("title", pid)
            year = m.get("year", "")
            passages.append(
                f"--- {title} ({year}), p.{chunk['page']} ---\n"
                f"{chunk['text']}\n"
            )

        return "\n".join(passages) if passages else "No results found."

    def _search_substring(
        self,
        query: str,
        top_k: int,
        full_text_ids: Optional[set] = None,
    ) -> str:
        """Fallback substring search (rank_bm25 not installed)."""
        query_lower = query.lower()
        rows = self._load_csv()
        passages: list[str] = []

        for row in rows:
            if len(passages) >= top_k:
                break
            pid = row.get("paper_id", "")
            if full_text_ids is not None and pid not in full_text_ids:
                continue
            md_path_str = row.get("local_md_path", "")
            if not md_path_str:
                continue
            md_path = Path(md_path_str)
            if not md_path.exists():
                continue

            text = md_path.read_text(encoding="utf-8")
            lines = text.splitlines()
            current_page = 1

            for i, line in enumerate(lines):
                if len(passages) >= top_k:
                    break
                page_match = re.match(
                    r"<!--\s*page\s*(\d+)\s*-->", line
                )
                if page_match:
                    current_page = int(page_match.group(1))
                    continue
                if query_lower in line.lower():
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    context = "\n".join(lines[start:end])
                    title = row.get("title", row["paper_id"])
                    year = row.get("year", "")
                    passages.append(
                        f"--- {title} ({year}), p.{current_page}"
                        f" ---\n{context}\n"
                    )

        return "\n".join(passages) if passages else "No results found."

    def get_paper(self, paper_id: str) -> str:
        """Return full extracted Markdown of *paper_id*, or ERROR."""
        for row in self._load_csv():
            if row["paper_id"] == paper_id:
                md_path_str = row.get("local_md_path", "")
                if not md_path_str:
                    return (
                        f"ERROR: paper '{paper_id}' has no extracted"
                        " text. Add it with CorpusAdd using a local"
                        " PDF or .md file."
                    )
                md_path = Path(md_path_str)
                if not md_path.exists():
                    return (
                        f"ERROR: text file not found at"
                        f" {md_path_str!r}."
                    )
                return md_path.read_text(encoding="utf-8")
        return f"ERROR: paper '{paper_id}' not found in corpus."

    def list_papers(self) -> str:
        """Return a Markdown table of all papers, or 'Corpus is empty.'

        Each row is annotated with ``[full-text]`` or
        ``[abstract-only]`` to show primary-source status.
        """
        rows = self._load_csv()
        if not rows:
            return "Corpus is empty."
        header = (
            "| paper_id | title | authors | year"
            " | source | full_text |"
        )
        sep = "| --- | --- | --- | --- | --- | --- |"
        lines = [header, sep]
        for r in rows:
            ft = r.get("full_text", "false").lower() == "true"
            ft_label = "[full-text]" if ft else "[abstract-only]"
            lines.append(
                f"| {r.get('paper_id','')} | {r.get('title','')} | "
                f"{r.get('authors','')} | {r.get('year','')} | "
                f"{r.get('source','')} | {ft_label} |"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # PDF extraction
    # ------------------------------------------------------------------

    def _extract_pdf_to_md(self, pdf_path: Path) -> str:
        """Extract page-annotated Markdown from *pdf_path*.

        Docling first (layout-aware — the accuracy path) WHERE AVAILABLE; fall
        back to PyMuPDF (``fitz``; the same library) when Docling is absent
        (e.g. Intel macOS, where the pyproject marker excludes it by design) or
        returns a thin/failed parse. Return the longer of the two.

        The only real bug in the original was the acceptance floor: it returned
        ANY Docling result > 100 chars, so a *failed* ~236-char Docling parse
        would beat the PyMuPDF fallback. Require a real body (> the full-text
        threshold) before trusting Docling; otherwise fall through.
        """
        docling_md = ""
        # 1. Docling — layout-aware, the accuracy path. Absent on Intel macOS.
        try:
            from docling.document_converter import (  # type: ignore
                DocumentConverter,
            )
            result = DocumentConverter().convert(str(pdf_path))
            docling_md = result.document.export_to_markdown() or ""
            if len(docling_md.strip()) > _FULL_TEXT_MD_THRESHOLD:
                return docling_md  # Docling actually parsed it — trust it.
        except Exception:  # noqa: BLE001
            pass

        # 2. PyMuPDF (fitz) — the lean fallback; the only extractor on Intel
        #    macOS. Page-annotated so the reviewer can cite by page.
        pymupdf_md = ""
        if fitz is not None:
            try:
                doc = fitz.open(str(pdf_path))
                pymupdf_md = "\n\n".join(
                    f"<!-- page {n} -->\n{pg.get_text()}"
                    for n, pg in enumerate(doc, start=1))
                doc.close()
            except Exception:  # noqa: BLE001
                pass

        # Docling was absent or thin → take whichever recovered more text.
        best = max((docling_md, pymupdf_md), key=lambda c: len(c.strip()))
        if len(best.strip()) > 100:
            return best
        return "(PDF extraction unavailable — install docling or pymupdf)"

    # ------------------------------------------------------------------
    # Chunking infrastructure
    # ------------------------------------------------------------------

    def _chunk_text(
        self, md_text: str, chunk_size: int = 150, overlap: int = 50
    ) -> list[dict]:
        """Split page-annotated markdown into overlapping word chunks.

        Respects ``<!-- page N -->`` markers: each chunk records its
        starting page.

        Returns
        -------
        list[dict]
            Each entry has ``{"page": int, "text": str}``.
        """
        lines = md_text.splitlines()
        current_page = 1
        all_words: list[tuple[str, int]] = []  # (word, page)
        for line in lines:
            pm = re.match(r"<!--\s*page\s*(\d+)\s*-->", line)
            if pm:
                current_page = int(pm.group(1))
                continue
            for w in line.split():
                all_words.append((w, current_page))

        chunks: list[dict] = []
        i = 0
        while i < len(all_words):
            window = all_words[i: i + chunk_size]
            if not window:
                break
            page = window[0][1]
            text = " ".join(w for w, _ in window)
            chunks.append({"page": page, "text": text})
            i += max(1, chunk_size - overlap)
        return chunks

    def _append_chunks(
        self, paper_id: str, chunks: list[dict]
    ) -> None:
        """Append *chunks* for *paper_id* to chunks.jsonl."""
        with self._lock:
            with self._chunks_path.open("a", encoding="utf-8") as f:
                for i, chunk in enumerate(chunks):
                    f.write(
                        json.dumps(
                            {
                                "paper_id": paper_id,
                                "chunk_id": i,
                                "page": chunk["page"],
                                "text": chunk["text"],
                            }
                        )
                        + "\n"
                    )

    def _load_chunks(self) -> list[dict]:
        """Load all chunks from chunks.jsonl."""
        if not self._chunks_path.exists():
            return []
        with self._lock:
            with self._chunks_path.open(encoding="utf-8") as f:
                return [
                    json.loads(line)
                    for line in f
                    if line.strip()
                ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _derive_paper_id(
        self, arxiv_id: str, doi: str, src: Path
    ) -> str:
        """Derive a stable paper_id from available metadata."""
        if arxiv_id:
            return "arxiv_" + _slugify(arxiv_id.strip())
        if doi:
            return "doi_" + _slugify(doi.strip())
        # Fall back to the source filename without extension
        return _slugify(src.stem) or "paper_unknown"

    def _paper_dir(self, paper_id: str) -> Path:
        return self._papers_dir / paper_id

    def _load_csv(self) -> list[dict]:
        if not self._csv_path.exists():
            return []
        with self._csv_path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _save_csv(self, rows: list[dict]) -> None:
        with self._csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {k: row.get(k, "") for k in _CSV_FIELDS}
                )
