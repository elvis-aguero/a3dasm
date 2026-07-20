"""Guardrail tests for literature corpus — no real network.

Tests:
1. Rate limiter enforces min interval per domain.
2. 429 with Retry-After → requested sleep; success resets breaker.
3. 404 → raises immediately (one request, no retry).
4. Three consecutive 429s → SourceCooldownError on 4th call.
5. Cache: hit skips request; expired ts → re-fetches.
6. add() full_text flag (PDF → True; long md → True; short md → False).
7. search() abstract-only corpus → ERROR; full-text paper → results.
8. DownloadPdf validates content (PDF magic → ok; html → ERROR).
9. Preflight warning: fastembed missing → warning in caplog.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import a3dasm._src.literature_corpus as lc_mod
from a3dasm._src.literature_corpus import (
    LiteratureCorpus,
    SourceCooldownError,
    _cache_get,
    _cache_put,
    _domain_key,
    _robust_get,
    _robust_post,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corpus(tmp_path: Path) -> LiteratureCorpus:
    return LiteratureCorpus(tmp_path / "literature")


def _mock_resp(status: int = 200, body: str = "{}",
               headers: dict | None = None) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = body
    resp.content = body.encode()
    resp.headers = headers or {}
    try:
        resp.json.return_value = json.loads(body) if body else {}
    except json.JSONDecodeError:
        resp.json.side_effect = json.JSONDecodeError("", "", 0)
    if status >= 400:
        import requests
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"HTTP {status}", response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _reset_rate_state(domain: str) -> None:
    """Clear all rate-limiting state for a domain between tests."""
    with lc_mod._rate_lock:
        lc_mod._domain_last_request.pop(domain, None)
        lc_mod._domain_consecutive_429.pop(domain, None)
        lc_mod._domain_cooldown_until.pop(domain, None)


def _inject_paper(
    corpus: LiteratureCorpus,
    paper_id: str,
    title: str = "Test Paper",
    authors: str = "A. Author",
    year: str = "2024",
    source: str = "arxiv",
    text: str = "",
    full_text: bool = False,
) -> None:
    """Inject a fake paper directly (no file copy or network)."""
    paper_dir = corpus._paper_dir(paper_id)
    paper_dir.mkdir(parents=True, exist_ok=True)
    md_path = paper_dir / "paper.md"
    md_content = text or f"<!-- page 1 -->\nAbstract for {title}\n"
    md_path.write_text(md_content, encoding="utf-8")
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
# Test 1: Rate limiter enforces min interval per domain
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def setup_method(self):
        _reset_rate_state("export.arxiv.org")
        _reset_rate_state("api.openalex.org")
        _reset_rate_state("api.semanticscholar.org")

    def test_arxiv_min_interval_enforced(self, monkeypatch):
        """Two rapid calls to an arxiv URL record a sleep ≥ 3 s."""
        sleeps: list[float] = []
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: sleeps.append(s))

        domain = "export.arxiv.org"
        _reset_rate_state(domain)

        mock_resp = _mock_resp(200, '{"results": []}')
        with patch("requests.get", return_value=mock_resp):
            _robust_get("https://export.arxiv.org/search/", retries=1)
            # Second call — rate limiter should request a sleep
            _reset_rate_state.__doc__  # just to have something before second call
            _robust_get("https://export.arxiv.org/search/", retries=1)

        # At least one sleep ≥ 3 s was issued
        assert any(s >= 3.0 for s in sleeps), (
            f"Expected sleep ≥ 3 s for arxiv, got sleeps={sleeps}"
        )

    def test_different_domains_do_not_block_each_other(self, monkeypatch):
        """An openalex call does not sleep due to arxiv rate limiting."""
        sleeps_by_call: list[float] = []
        monkeypatch.setattr(
            lc_mod, "_sleep", lambda s: sleeps_by_call.append(s)
        )

        _reset_rate_state("export.arxiv.org")
        _reset_rate_state("api.openalex.org")

        mock_resp = _mock_resp(200, '{"results": []}')
        with patch("requests.get", return_value=mock_resp):
            # Saturate arxiv rate limit
            _robust_get("https://export.arxiv.org/search/", retries=1)

            before = len(sleeps_by_call)
            # openalex call — should not inherit arxiv's wait
            _robust_get("https://api.openalex.org/works", retries=1)
            openalex_sleeps = sleeps_by_call[before:]

        # If openalex sleeps at all, it should be a very small amount
        # (its own 0.2 s interval, not arxiv's 3 s)
        for s in openalex_sleeps:
            assert s < 3.0, (
                f"openalex call slept {s} s — looks like arxiv leak"
            )

    def test_semanticscholar_min_interval(self, monkeypatch):
        """Two rapid calls to S2 API record a sleep close to 1 s."""
        sleeps: list[float] = []
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: sleeps.append(s))

        domain = "api.semanticscholar.org"
        _reset_rate_state(domain)

        mock_resp = _mock_resp(200, '{}')
        with patch("requests.get", return_value=mock_resp):
            _robust_get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                retries=1,
            )
            _robust_get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                retries=1,
            )

        # Allow for tiny timing jitter (monotonic clock precision)
        assert any(s >= 0.9 for s in sleeps), (
            f"Expected sleep ≥ 0.9 s for S2 (1 s interval), got sleeps={sleeps}"
        )


# ---------------------------------------------------------------------------
# Test 2: 429 with Retry-After header
# ---------------------------------------------------------------------------

class Test429RetryAfter:
    def setup_method(self):
        _reset_rate_state("api.openalex.org")

    def test_retry_after_header_respected(self, monkeypatch):
        """Retry-After: 7 → at least one sleep of 7 s requested."""
        sleeps: list[float] = []
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: sleeps.append(s))
        _reset_rate_state("api.openalex.org")

        resp_429 = _mock_resp(
            429, "", headers={"Retry-After": "7"}
        )
        resp_429.raise_for_status.return_value = None
        resp_ok = _mock_resp(200, '{"results": []}')

        call_count = [0]

        def side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return resp_429
            return resp_ok

        with patch("requests.get", side_effect=side_effect):
            result = _robust_get(
                "https://api.openalex.org/works", retries=3
            )

        assert result is resp_ok
        assert any(s >= 7.0 for s in sleeps), (
            f"Expected sleep ≥ 7 s (Retry-After), got {sleeps}"
        )

    def test_success_after_429_resets_breaker(self, monkeypatch):
        """A successful request clears the consecutive-429 counter."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        resp_429 = _mock_resp(429, "", headers={})
        resp_429.raise_for_status.return_value = None
        resp_ok = _mock_resp(200, '{"results": []}')

        call_count = [0]

        def side_effect(*a, **kw):
            call_count[0] += 1
            if call_count[0] <= 2:
                return resp_429
            return resp_ok

        with patch("requests.get", side_effect=side_effect):
            _robust_get("https://api.openalex.org/works", retries=5)

        with lc_mod._rate_lock:
            count = lc_mod._domain_consecutive_429.get(
                "api.openalex.org", 0
            )
        assert count == 0, (
            f"Counter should be reset after success, got {count}"
        )


# ---------------------------------------------------------------------------
# Test 3: 404 raises immediately, exactly one request
# ---------------------------------------------------------------------------

class Test404:
    def test_404_raises_immediately_no_retry(self, monkeypatch):
        """HTTP 404 raises at once — no retry loop."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        import requests as _requests
        resp_404 = _mock_resp(404, "Not found")

        call_count = [0]

        def side_effect(*a, **kw):
            call_count[0] += 1
            return resp_404

        with patch("requests.get", side_effect=side_effect):
            with pytest.raises(_requests.HTTPError):
                _robust_get(
                    "https://api.openalex.org/works/W999",
                    retries=3,
                )

        assert call_count[0] == 1, (
            f"Expected exactly 1 request for 404, got {call_count[0]}"
        )


# ---------------------------------------------------------------------------
# Test 4: Circuit breaker — 3 consecutive 429s → SourceCooldownError
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def setup_method(self):
        _reset_rate_state("api.openalex.org")

    def test_three_429s_trigger_cooldown(self, monkeypatch):
        """Three consecutive 429s → 4th call raises SourceCooldownError."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        resp_429 = _mock_resp(429, "", headers={})
        resp_429.raise_for_status.return_value = None

        with patch("requests.get", return_value=resp_429):
            # Three calls should all 429 but not yet raise breaker
            for _ in range(3):
                with pytest.raises(Exception):
                    _robust_get(
                        "https://api.openalex.org/works",
                        retries=1,
                    )

        # The 4th call should raise SourceCooldownError immediately
        with pytest.raises(SourceCooldownError) as exc_info:
            _robust_get(
                "https://api.openalex.org/works", retries=1
            )

        assert "rate-limited" in str(exc_info.value).lower()

    def test_cooldown_error_message_contains_domain(self, monkeypatch):
        """SourceCooldownError message names the domain."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        resp_429 = _mock_resp(429, "", headers={})
        resp_429.raise_for_status.return_value = None

        with patch("requests.get", return_value=resp_429):
            for _ in range(3):
                with pytest.raises(Exception):
                    _robust_get(
                        "https://api.openalex.org/works",
                        retries=1,
                    )

        with pytest.raises(SourceCooldownError) as exc_info:
            _robust_get("https://api.openalex.org/works", retries=1)

        assert "api.openalex.org" in str(exc_info.value)

    def test_cooldown_no_request_issued(self, monkeypatch):
        """While in cooldown, no HTTP request is issued."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        # Manually set cooldown
        with lc_mod._rate_lock:
            lc_mod._domain_cooldown_until["api.openalex.org"] = (
                time.monotonic() + 9999
            )

        call_count = [0]

        def counting_get(*a, **kw):
            call_count[0] += 1
            return _mock_resp(200, "{}")

        with patch("requests.get", side_effect=counting_get):
            with pytest.raises(SourceCooldownError):
                _robust_get(
                    "https://api.openalex.org/works", retries=3
                )

        assert call_count[0] == 0, (
            "No request should be issued during cooldown"
        )


# ---------------------------------------------------------------------------
# Test 5: On-disk HTTP GET cache
# ---------------------------------------------------------------------------

class TestHttpCache:
    def test_cache_hit_skips_request(self, tmp_path, monkeypatch):
        """Second identical GET issues no request and returns same text."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        cache_dir = tmp_path / ".http_cache"
        resp_ok = _mock_resp(200, '{"results": [1, 2, 3]}')
        resp_ok.headers = {"Content-Type": "application/json"}

        call_count = [0]

        def counting_get(*a, **kw):
            call_count[0] += 1
            return resp_ok

        url = "https://api.openalex.org/works"
        params = {"search": "transformer"}

        with patch("requests.get", side_effect=counting_get):
            r1 = _robust_get(
                url, params=params, cache_dir=cache_dir, retries=1
            )

        # Second call — should hit cache
        with patch("requests.get", side_effect=counting_get):
            r2 = _robust_get(
                url, params=params, cache_dir=cache_dir, retries=1
            )

        assert call_count[0] == 1, (
            f"Expected 1 real request, got {call_count[0]}"
        )
        assert r1.text == r2.text

    def test_cache_file_written_after_200(self, tmp_path, monkeypatch):
        """A 200 response writes a cache file."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        cache_dir = tmp_path / ".http_cache"
        resp_ok = _mock_resp(200, '{"ok": true}')
        resp_ok.headers = {"Content-Type": "application/json"}

        with patch("requests.get", return_value=resp_ok):
            _robust_get(
                "https://api.openalex.org/works",
                cache_dir=cache_dir,
                retries=1,
            )

        cache_files = list(cache_dir.glob("*.json"))
        assert len(cache_files) == 1

    def test_expired_cache_refetches(self, tmp_path, monkeypatch):
        """An expired cache entry triggers a fresh request."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        cache_dir = tmp_path / ".http_cache"
        url = "https://api.openalex.org/works"
        params = None

        # Write an already-expired cache entry
        _cache_put(
            cache_dir, url, params, 200, '{"old": true}', "application/json"
        )
        # Backdate the ts by 2 days
        from a3dasm._src.literature_corpus import _cache_key
        key = _cache_key(url, params)
        cache_file = cache_dir / (key + ".json")
        data = json.loads(cache_file.read_text())
        data["ts"] = time.time() - 172800  # 2 days ago
        cache_file.write_text(json.dumps(data))

        call_count = [0]
        resp_fresh = _mock_resp(200, '{"new": true}')
        resp_fresh.headers = {"Content-Type": "application/json"}

        def counting_get(*a, **kw):
            call_count[0] += 1
            return resp_fresh

        with patch("requests.get", side_effect=counting_get):
            result = _robust_get(
                url, cache_dir=cache_dir, ttl=86400, retries=1
            )

        assert call_count[0] == 1, "Should re-fetch on expired cache"
        assert result.text == '{"new": true}'

    def test_cache_does_not_store_non_200(self, tmp_path, monkeypatch):
        """Non-200 responses are not cached."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        cache_dir = tmp_path / ".http_cache"

        _cache_put(cache_dir, "https://example.com", None, 404,
                   "not found", "text/html")

        cache_files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
        assert len(cache_files) == 0


# ---------------------------------------------------------------------------
# Test 6: add() full_text flag
# ---------------------------------------------------------------------------

class TestFullTextFlag:
    def test_long_markdown_is_full_text(self, tmp_path):
        """Markdown > 5000 chars → full_text=True in CSV."""
        corpus = _make_corpus(tmp_path)
        long_text = "<!-- page 1 -->\n" + ("word " * 1500)  # >5000 chars
        md_file = tmp_path / "long_paper.md"
        md_file.write_text(long_text, encoding="utf-8")

        paper_id = corpus.add(
            str(md_file),
            arxiv_id="1234.56789",
            title="Long Paper",
        )
        assert paper_id == "arxiv_1234_56789"
        rows = corpus._load_csv()
        assert len(rows) == 1
        assert rows[0].get("full_text", "false") == "true"

    def test_short_markdown_is_abstract_only(self, tmp_path):
        """Markdown ≤ 5000 chars → full_text=False in CSV."""
        corpus = _make_corpus(tmp_path)
        short_text = "<!-- page 1 -->\nThis is a short abstract.\n"
        md_file = tmp_path / "short_paper.md"
        md_file.write_text(short_text, encoding="utf-8")

        paper_id = corpus.add(
            str(md_file),
            arxiv_id="9999.00001",
            title="Short Abstract Paper",
        )
        assert paper_id == "arxiv_9999_00001"
        rows = corpus._load_csv()
        assert rows[0].get("full_text", "true") == "false"

    def test_pdf_full_text_gated_on_real_extraction(self, tmp_path):
        """A PDF is full-text ONLY if extraction yields a real body. A PDF that
        extracts almost nothing (scanned / corrupt streams) is REJECTED with
        guidance — not stored as quotable full-text. The previous code marked
        every PDF full-text "regardless of extraction", which kept a 236-char
        placeholder as if it were a quotable paper. (The real-body positive case
        is covered deterministically in test_literature_corpus.py.)"""
        import pytest as _pt
        try:
            import fitz
        except ImportError:
            _pt.skip("fitz not available")

        corpus = _make_corpus(tmp_path)
        doc = fitz.open()
        doc.new_page()
        doc[0].insert_text((50, 50), "Hello from PDF page one.")  # ~25 chars
        pdf_path = tmp_path / "scanned.pdf"
        doc.save(str(pdf_path))
        doc.close()

        result = corpus.add(str(pdf_path), arxiv_id="2024.00001", title="Scanned")
        assert result.startswith("ERROR"), f"expected rejection, got: {result}"
        assert "arxiv_read_paper" in result  # points to the working alternative
        assert corpus._load_csv() == []       # no phantom full-text entry stored

    def test_csv_contains_full_text_column(self, tmp_path):
        """corpus.csv includes the full_text column."""
        corpus = _make_corpus(tmp_path)
        md_file = tmp_path / "paper.md"
        md_file.write_text("content", encoding="utf-8")
        corpus.add(str(md_file), title="Any")

        rows = corpus._load_csv()
        assert len(rows) == 1
        assert "full_text" in rows[0]


# ---------------------------------------------------------------------------
# Test 7: search() — abstract-only vs full-text
# ---------------------------------------------------------------------------

class TestSearchFullTextOnly:
    def test_abstract_only_corpus_returns_error(self, tmp_path):
        """search() returns ERROR when no full-text papers exist."""
        corpus = _make_corpus(tmp_path)
        _inject_paper(
            corpus, "arxiv_0001_00001",
            title="Abstract Only Paper",
            text="<!-- page 1 -->\nShort abstract.",
            full_text=False,
        )

        result = corpus.search("abstract")
        assert result.startswith("ERROR:"), result
        assert "full-text" in result.lower()
        assert "DownloadPdf" in result or "download" in result.lower()

    def test_full_text_paper_returns_results(self, tmp_path):
        """search() returns passages when at least one full-text paper."""
        corpus = _make_corpus(tmp_path)
        long_text = (
            "<!-- page 1 -->\n"
            + "The transformer uses self-attention mechanisms. " * 300
        )
        _inject_paper(
            corpus, "arxiv_1706_03762",
            title="Attention Is All You Need",
            text=long_text,
            full_text=True,
        )

        result = corpus.search("self-attention")
        assert result != "No results found."
        assert "Attention Is All You Need" in result

    def test_search_ignores_abstract_only_in_mixed_corpus(self, tmp_path):
        """Abstract-only papers are excluded even in a mixed corpus."""
        corpus = _make_corpus(tmp_path)

        # Abstract-only paper with matching text
        _inject_paper(
            corpus, "arxiv_abstract_only",
            title="Abstract Only With Match",
            text="<!-- page 1 -->\nkeyword_unique_sentinel",
            full_text=False,
        )

        # Full-text paper without matching text
        long_text = (
            "<!-- page 1 -->\n"
            + "This paper is about neural architecture search. " * 300
        )
        _inject_paper(
            corpus, "arxiv_fulltext_paper",
            title="Full Text Paper No Match",
            text=long_text,
            full_text=True,
        )

        result = corpus.search("keyword_unique_sentinel")
        # Should not find the abstract-only paper's content
        assert "Abstract Only With Match" not in result

    def test_error_message_mentions_count(self, tmp_path):
        """ERROR message includes the number of abstract-only entries."""
        corpus = _make_corpus(tmp_path)
        for i in range(3):
            _inject_paper(
                corpus, f"arxiv_0000_0000{i}",
                title=f"Abstract {i}",
                full_text=False,
            )

        result = corpus.search("anything")
        assert "3" in result  # count mentioned


# ---------------------------------------------------------------------------
# Test 8: DownloadPdf validates content
# ---------------------------------------------------------------------------

class TestDownloadPdf:
    def test_pdf_magic_bytes_accepted(self, tmp_path, monkeypatch):
        """Body starting with %PDF is saved and path returned."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("arxiv.org")

        corpus = _make_corpus(tmp_path)
        cache_dir = corpus._http_cache_dir

        from a3dasm._src.agents.literature import (
            LiteratureReviewAgent,
        )
        agent = LiteratureReviewAgent()
        tools = agent.build_closure_tools(
            study_dir=tmp_path,
            lit_reviewer_notes_dir=tmp_path / "lit",
        )

        pdf_body = b"%PDF-1.4 this is fake pdf content"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = pdf_body.decode("latin-1")
        mock_resp.content = pdf_body
        mock_resp.headers = {"Content-Type": "application/pdf"}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            result = tools["DownloadPdf"](
                "https://arxiv.org/pdf/1706.03762",
                "test_paper.pdf",
            )

        assert not result.startswith("ERROR"), result
        assert result.endswith(".pdf")
        assert Path(result).exists()

    def test_html_body_rejected(self, tmp_path, monkeypatch):
        """HTML body → ERROR 'not a PDF'."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("example.com")

        from a3dasm._src.agents.literature import (
            LiteratureReviewAgent,
        )
        agent = LiteratureReviewAgent()
        tools = agent.build_closure_tools(
            study_dir=tmp_path,
            lit_reviewer_notes_dir=tmp_path / "lit",
        )

        html_body = b"<html><body>Login required</body></html>"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = html_body.decode()
        mock_resp.content = html_body
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            result = tools["DownloadPdf"](
                "https://example.com/paper.pdf",
                "bad_paper.pdf",
            )

        assert result.startswith("ERROR")
        assert "not a PDF" in result

    def test_content_type_pdf_accepted(self, tmp_path, monkeypatch):
        """Content-Type containing 'pdf' is accepted even without magic."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("example.com")

        from a3dasm._src.agents.literature import (
            LiteratureReviewAgent,
        )
        agent = LiteratureReviewAgent()
        tools = agent.build_closure_tools(
            study_dir=tmp_path,
            lit_reviewer_notes_dir=tmp_path / "lit",
        )

        pdf_body = b"some pdf bytes without magic"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = pdf_body.decode("latin-1")
        mock_resp.content = pdf_body
        mock_resp.headers = {"Content-Type": "application/pdf; charset=utf-8"}
        mock_resp.raise_for_status.return_value = None

        with patch("requests.get", return_value=mock_resp):
            result = tools["DownloadPdf"](
                "https://example.com/paper",
                "ct_paper.pdf",
            )

        assert not result.startswith("ERROR"), result

    def test_download_failure_returns_error(self, tmp_path, monkeypatch):
        """Network failure → ERROR string."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("example.com")

        from a3dasm._src.agents.literature import (
            LiteratureReviewAgent,
        )
        agent = LiteratureReviewAgent()
        tools = agent.build_closure_tools(
            study_dir=tmp_path,
            lit_reviewer_notes_dir=tmp_path / "lit",
        )

        import requests as _req
        with patch(
            "requests.get",
            side_effect=_req.RequestException("Connection refused"),
        ):
            result = tools["DownloadPdf"](
                "https://example.com/paper.pdf",
                "fail_paper.pdf",
            )

        assert result.startswith("ERROR")


# ---------------------------------------------------------------------------
# Test 9: Preflight warnings
# ---------------------------------------------------------------------------

class TestPreflightWarnings:
    def setup_method(self):
        # Reset the subprocess-embedder tri-state cache before each test
        # so the probe path is exercised fresh.
        lc_mod._subprocess_embedder_state = None
        lc_mod._subprocess_embedder_warned = False

    def teardown_method(self):
        lc_mod._subprocess_embedder_state = None
        lc_mod._subprocess_embedder_warned = False

    def test_fastembed_missing_logs_warning(self, tmp_path, caplog):
        """ImportError from fastembed AND no uv → warning logged; result is None."""
        import logging
        corpus = _make_corpus(tmp_path)

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return real_import(name, *args, **kwargs)

        with caplog.at_level(logging.WARNING, logger="a3dasm._src.literature_corpus"):
            with patch("builtins.__import__", side_effect=mock_import):
                with patch("shutil.which", return_value=None):
                    result = corpus._get_embedding_model()

        assert result is None
        warning_messages = [r.message for r in caplog.records]
        assert any(
            "fastembed" in str(m).lower() for m in warning_messages
        ), f"No fastembed warning found in: {warning_messages}"

    def test_fastembed_warning_emitted_only_once(self, tmp_path, caplog):
        """The fastembed+worker warning is logged only once (tri-state cache)."""
        import logging
        corpus = _make_corpus(tmp_path)

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("No module named 'fastembed'")
            return real_import(name, *args, **kwargs)

        with caplog.at_level(logging.WARNING, logger="a3dasm._src.literature_corpus"):
            with patch("builtins.__import__", side_effect=mock_import):
                with patch("shutil.which", return_value=None):
                    corpus._get_embedding_model()
                    corpus._embedding_model = None
                    corpus._get_embedding_model()
                    corpus._embedding_model = None
                    corpus._get_embedding_model()

        fastembed_warnings = [
            r for r in caplog.records
            if "fastembed" in str(r.message).lower()
        ]
        assert len(fastembed_warnings) == 1, (
            f"Expected 1 warning, got {len(fastembed_warnings)}"
        )

    def test_semanticscholar_missing_logs_warning(
        self, tmp_path, caplog
    ):
        """Missing semanticscholar logs a warning when building tools."""
        import logging
        import builtins

        from a3dasm._src.agents.literature import (
            LiteratureReviewAgent,
        )

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "semanticscholar" or name.startswith(
                "semanticscholar."
            ):
                raise ImportError(
                    "No module named 'semanticscholar'"
                )
            return real_import(name, *args, **kwargs)

        with caplog.at_level(
            logging.WARNING,
            logger="a3dasm._src.agents.literature",
        ):
            with patch("builtins.__import__", side_effect=mock_import):
                agent = LiteratureReviewAgent()
                agent.build_closure_tools(
                    study_dir=tmp_path,
                    lit_reviewer_notes_dir=tmp_path / "lit",
                )

        warning_messages = [r.message for r in caplog.records]
        assert any(
            "semanticscholar" in str(m).lower()
            for m in warning_messages
        ), f"No semanticscholar warning in: {warning_messages}"


# ---------------------------------------------------------------------------
# Test: list_papers marks full_text correctly
# ---------------------------------------------------------------------------

class TestListPapers:
    def test_list_papers_marks_full_text(self, tmp_path):
        """list_papers shows [full-text] and [abstract-only] labels."""
        corpus = _make_corpus(tmp_path)
        _inject_paper(
            corpus, "arxiv_ft_paper",
            title="Full Text Paper",
            full_text=True,
        )
        _inject_paper(
            corpus, "arxiv_abs_paper",
            title="Abstract Only Paper",
            full_text=False,
        )

        result = corpus.list_papers()
        assert "[full-text]" in result
        assert "[abstract-only]" in result

    def test_list_papers_empty_corpus(self, tmp_path):
        corpus = _make_corpus(tmp_path)
        assert corpus.list_papers() == "Corpus is empty."


# ---------------------------------------------------------------------------
# Test: search_openalex includes OA PDF URL (in agent tools)
# ---------------------------------------------------------------------------

class TestOpenAlexPdfUrl:
    def test_openalex_returns_oa_pdf_url(self, tmp_path, monkeypatch):
        """search_openalex includes best_oa_location pdf_url."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        from a3dasm._src.agents.literature import (
            LiteratureReviewAgent,
        )
        agent = LiteratureReviewAgent()
        tools = agent.build_closure_tools(
            study_dir=tmp_path,
            lit_reviewer_notes_dir=tmp_path / "lit",
        )

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "application/json"}
        oa_pdf = "https://open-access.example.com/paper.pdf"
        mock_resp.json.return_value = {
            "results": [{
                "id": "W123",
                "title": "OA Paper",
                "publication_year": 2023,
                "doi": "10.1234/test",
                "authorships": [
                    {"author": {"display_name": "A. Author"}}
                ],
                "best_oa_location": {"pdf_url": oa_pdf},
                "primary_location": {"pdf_url": None},
                "open_access": {"oa_url": None},
                "abstract_inverted_index": {},
            }]
        }
        mock_resp.text = json.dumps(mock_resp.json.return_value)

        with patch("requests.get", return_value=mock_resp):
            result = tools["search_openalex"]("neural networks")

        data = json.loads(result)
        assert len(data) >= 1
        assert data[0]["pdf_url"] == oa_pdf


# ---------------------------------------------------------------------------
# Tests: get_openalex_citations and get_openalex_references
# ---------------------------------------------------------------------------

def _make_tools(tmp_path):
    """Helper: build closure tools for an agent, returning the tools dict."""
    from a3dasm._src.agents.literature import LiteratureReviewAgent
    agent = LiteratureReviewAgent()
    return agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=tmp_path / "lit",
    )


def _work_payload(**overrides):
    """Minimal OpenAlex work object for use in mock responses."""
    base = {
        "id": "https://openalex.org/W999",
        "title": "Citing Paper",
        "publication_year": 2024,
        "cited_by_count": 42,
        "doi": "https://doi.org/10.1234/citing",
        "best_oa_location": {"pdf_url": "https://oa.example.com/paper.pdf"},
        "primary_location": {},
        "open_access": {},
    }
    base.update(overrides)
    return base


class TestGetOpenAlexCitations:
    """Tests for get_openalex_citations closure."""

    def test_issues_correct_filter_and_returns_titles(
        self, tmp_path, monkeypatch
    ):
        """One GET with filter=cites:W123; parsed titles returned."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        tools = _make_tools(tmp_path)
        if "get_openalex_citations" not in tools:
            pytest.skip("get_openalex_citations not in tools")

        resp_body = json.dumps({"results": [_work_payload()]})
        mock_resp = _mock_resp(200, resp_body)
        mock_resp.headers = {"Content-Type": "application/json"}

        captured_params = {}

        def fake_get(url, params=None, headers=None, timeout=None):
            captured_params.update(params or {})
            return mock_resp

        with patch("requests.get", side_effect=fake_get):
            result = tools["get_openalex_citations"]("W123", n_results=5)

        assert "cites:W123" == captured_params.get("filter"), (
            f"Expected filter=cites:W123, got {captured_params}"
        )
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["title"] == "Citing Paper"
        assert data[0]["pdf_url"] == "https://oa.example.com/paper.pdf"

    def test_accepts_full_url_work_id(self, tmp_path, monkeypatch):
        """Full URL work_id produces the same filter as a bare W-id."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        tools = _make_tools(tmp_path)
        if "get_openalex_citations" not in tools:
            pytest.skip("get_openalex_citations not in tools")

        resp_body = json.dumps({"results": [_work_payload()]})
        mock_resp = _mock_resp(200, resp_body)
        mock_resp.headers = {"Content-Type": "application/json"}

        captured_params = {}

        def fake_get(url, params=None, headers=None, timeout=None):
            captured_params.update(params or {})
            return mock_resp

        with patch("requests.get", side_effect=fake_get):
            result = tools["get_openalex_citations"](
                "https://openalex.org/W123", n_results=5
            )

        assert "cites:W123" == captured_params.get("filter"), (
            f"Expected cites:W123 (bare id), got {captured_params}"
        )
        data = json.loads(result)
        assert isinstance(data, list)

    def test_cooldown_error_returns_error_string(
        self, tmp_path, monkeypatch
    ):
        """SourceCooldownError from _robust_get → 'ERROR: ...' string."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)

        tools = _make_tools(tmp_path)
        if "get_openalex_citations" not in tools:
            pytest.skip("get_openalex_citations not in tools")

        # Force the circuit breaker into cooldown for openalex
        _reset_rate_state("api.openalex.org")
        with lc_mod._rate_lock:
            lc_mod._domain_cooldown_until["api.openalex.org"] = (
                time.monotonic() + 9999
            )

        call_count = [0]

        def counting_get(*a, **kw):
            call_count[0] += 1
            return _mock_resp(200, '{"results": []}')

        with patch("requests.get", side_effect=counting_get):
            result = tools["get_openalex_citations"]("W123")

        assert result.startswith("ERROR:"), result
        assert "rate-limited" in result.lower() or "cooldown" in result.lower()
        assert call_count[0] == 0, "No request should be issued during cooldown"


class TestGetOpenAlexReferences:
    """Tests for get_openalex_references closure."""

    def test_two_gets_work_then_hydrate(self, tmp_path, monkeypatch):
        """Makes two GETs: single-work fetch then batched hydration."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        tools = _make_tools(tmp_path)
        if "get_openalex_references" not in tools:
            pytest.skip("get_openalex_references not in tools")

        work_body = json.dumps({
            "id": "https://openalex.org/W42",
            "referenced_works": [
                "https://openalex.org/W10",
                "https://openalex.org/W20",
            ],
        })
        hydrate_body = json.dumps({
            "results": [
                _work_payload(
                    id="https://openalex.org/W10",
                    title="Reference One",
                    cited_by_count=5,
                ),
                _work_payload(
                    id="https://openalex.org/W20",
                    title="Reference Two",
                    cited_by_count=3,
                ),
            ]
        })

        call_urls = []

        def fake_get(url, params=None, headers=None, timeout=None):
            call_urls.append(url)
            if "W42" in url and not params.get("filter"):
                return _mock_resp(200, work_body)
            return _mock_resp(200, hydrate_body)

        with patch("requests.get", side_effect=fake_get):
            result = tools["get_openalex_references"]("W42")

        assert len(call_urls) == 2, (
            f"Expected 2 GETs, got {len(call_urls)}: {call_urls}"
        )
        data = json.loads(result)
        assert isinstance(data, list)
        titles = {d["title"] for d in data}
        assert "Reference One" in titles
        assert "Reference Two" in titles

    def test_hydrate_uses_pipe_joined_filter(self, tmp_path, monkeypatch):
        """Batched hydration uses openalex_id:W10|W20 filter."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        tools = _make_tools(tmp_path)
        if "get_openalex_references" not in tools:
            pytest.skip("get_openalex_references not in tools")

        work_body = json.dumps({
            "id": "https://openalex.org/W42",
            "referenced_works": [
                "https://openalex.org/W10",
                "https://openalex.org/W20",
            ],
        })
        hydrate_body = json.dumps({"results": []})

        captured_hydrate_params = {}

        def fake_get(url, params=None, headers=None, timeout=None):
            params = params or {}
            if params.get("filter", "").startswith("openalex_id:"):
                captured_hydrate_params.update(params)
            if "W42" in url and not params.get("filter"):
                return _mock_resp(200, work_body)
            return _mock_resp(200, hydrate_body)

        with patch("requests.get", side_effect=fake_get):
            tools["get_openalex_references"]("W42")

        filt = captured_hydrate_params.get("filter", "")
        assert filt.startswith("openalex_id:"), (
            f"Expected openalex_id: filter, got {filt!r}"
        )
        assert "W10" in filt and "W20" in filt

    def test_empty_referenced_works_returns_message(
        self, tmp_path, monkeypatch
    ):
        """Work with no referenced_works → plain 'No references listed' message."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
        _reset_rate_state("api.openalex.org")

        tools = _make_tools(tmp_path)
        if "get_openalex_references" not in tools:
            pytest.skip("get_openalex_references not in tools")

        work_body = json.dumps({
            "id": "https://openalex.org/W99",
            "referenced_works": [],
        })

        def fake_get(url, params=None, headers=None, timeout=None):
            return _mock_resp(200, work_body)

        with patch("requests.get", side_effect=fake_get):
            result = tools["get_openalex_references"]("W99")

        assert "No references listed" in result, (
            f"Expected no-references message, got: {result!r}"
        )

    def test_references_cooldown_error_returns_error_string(
        self, tmp_path, monkeypatch
    ):
        """SourceCooldownError on first GET → 'ERROR: ...' string."""
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)

        tools = _make_tools(tmp_path)
        if "get_openalex_references" not in tools:
            pytest.skip("get_openalex_references not in tools")

        # Force the circuit breaker into cooldown for openalex
        _reset_rate_state("api.openalex.org")
        with lc_mod._rate_lock:
            lc_mod._domain_cooldown_until["api.openalex.org"] = (
                time.monotonic() + 9999
            )

        call_count = [0]

        def counting_get(*a, **kw):
            call_count[0] += 1
            return _mock_resp(200, '{"results": []}')

        with patch("requests.get", side_effect=counting_get):
            result = tools["get_openalex_references"]("W42")

        assert result.startswith("ERROR:"), result
        assert "rate-limited" in result.lower() or "cooldown" in result.lower()
        assert call_count[0] == 0, "No request should be issued during cooldown"


# ---------------------------------------------------------------------------
# Tests: S2 closures run in a thread with no running event loop
# ---------------------------------------------------------------------------

def _make_s2_tools(tmp_path, monkeypatch=None):
    """Helper: build closure tools for an agent."""
    from a3dasm._src.agents.literature import LiteratureReviewAgent
    if monkeypatch is not None:
        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)
    agent = LiteratureReviewAgent()
    return agent.build_closure_tools(
        study_dir=tmp_path,
        lit_reviewer_notes_dir=tmp_path / "lit",
    )


class TestS2EventLoopSafety:
    """S2 closures must run in a fresh thread (no running event loop)."""

    def test_search_semantic_scholar_runs_in_thread_without_loop(
        self, tmp_path, monkeypatch
    ):
        """search_semantic_scholar called from inside a running event loop
        executes the S2 client in a thread where no event loop is running.

        The fake S2 client asserts asyncio.get_running_loop() raises
        RuntimeError in its own calling thread — proving _call_in_fresh_thread
        routed the call off the event-loop thread.
        """
        import asyncio
        import json as _json

        loop_errors_seen: list[bool] = []

        class FakeResult:
            paperId = "fake_id_1"
            title = "Fake Paper"
            year = 2024
            venue = "NeurIPS"
            citationCount = 42
            authors = [{"name": "A. Author"}]
            abstract = "Abstract text."
            externalIds = {}

        class FakeSS:
            def search_paper(self, query, limit=10, fields=None):
                # If we are NOT in an event loop thread, this raises.
                try:
                    asyncio.get_running_loop()
                    # A loop IS running in this thread — bad.
                    loop_errors_seen.append(False)
                except RuntimeError:
                    # No loop here — correct.
                    loop_errors_seen.append(True)
                return [FakeResult()]

        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)

        # Patch the SemanticScholar constructor inside the module scope.
        with patch(
            "semanticscholar.SemanticScholar",
            return_value=FakeSS(),
        ):
            tools = _make_s2_tools(tmp_path)

        if "search_semantic_scholar" not in tools:
            pytest.skip("semanticscholar not installed")

        # Invoke the closure from INSIDE a running event loop.
        async def driver():
            return tools["search_semantic_scholar"](query="transformers", num_results=1)

        result_json = asyncio.run(driver())

        # The fake saw no running loop in its own thread.
        assert loop_errors_seen, "FakeSS.search_paper was never called"
        assert all(loop_errors_seen), (
            "S2 client was called from within a running event loop "
            f"(loop_errors_seen={loop_errors_seen})"
        )

        # Result parses correctly.
        data = _json.loads(result_json)
        assert isinstance(data, list)
        assert data[0]["title"] == "Fake Paper"

    def test_call_in_fresh_thread_does_not_block_on_shutdown_after_timeout(self):
        """Regression: `with ThreadPoolExecutor(...) as pool:` blocks on
        shutdown(wait=True) in __exit__ even after future.result(timeout=...)
        has already raised TimeoutError — silently re-introducing an
        unbounded wait on top of the timeout. Observed directly:
        search_semantic_scholar hung 10+ minutes despite a 30s timeout,
        traced to exactly this (the semanticscholar library's own internal
        429 retry can legitimately run several minutes on one call, and the
        blocking shutdown waited for it regardless of the timeout firing).
        Assert the CALLER gets control back near the declared timeout, not
        after however long the hung function actually takes."""
        import time as _time
        import a3dasm._src.agents.literature as lit_mod

        def hangs_forever():
            _time.sleep(5.0)  # much longer than the 0.1s timeout below
            return "should never get here in time"

        t0 = _time.monotonic()
        with pytest.raises(TimeoutError):
            lit_mod._call_in_fresh_thread(hangs_forever, timeout=0.1)
        elapsed = _time.monotonic() - t0
        assert elapsed < 1.0, (
            f"_call_in_fresh_thread took {elapsed:.2f}s to return after a "
            "0.1s timeout — it blocked on the hung worker thread instead of "
            "returning immediately (the exact regression this test guards)"
        )

    def test_search_semantic_scholar_timeout_returns_error_string(
        self, tmp_path, monkeypatch
    ):
        """When the S2 call exceeds the timeout, return a timed-out ERROR string."""
        import time as _time

        class SlowFakeSS:
            def search_paper(self, query, limit=10, fields=None):
                _time.sleep(5.0)  # longer than injected timeout
                return []

        monkeypatch.setattr(lc_mod, "_sleep", lambda s: None)

        with patch(
            "semanticscholar.SemanticScholar",
            return_value=SlowFakeSS(),
        ):
            tools = _make_s2_tools(tmp_path)

        if "search_semantic_scholar" not in tools:
            pytest.skip("semanticscholar not installed")

        # Monkey-patch _call_in_fresh_thread to use a tiny timeout.
        import a3dasm._src.agents.literature as lit_mod
        orig = lit_mod._call_in_fresh_thread

        def _fast_timeout(fn, *args, timeout=0.05, **kwargs):
            return orig(fn, *args, timeout=0.05, **kwargs)

        monkeypatch.setattr(lit_mod, "_call_in_fresh_thread", _fast_timeout)

        result = tools["search_semantic_scholar"](query="something")

        assert result.startswith("ERROR"), (
            f"Expected ERROR string on timeout, got: {result!r}"
        )
        assert "timed out" in result.lower(), (
            f"ERROR should mention 'timed out': {result!r}"
        )
