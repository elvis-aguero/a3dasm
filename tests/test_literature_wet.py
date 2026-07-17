"""Wet integration test for LiteratureReviewAgent.

Runs a real delegation with actual LLM + arxiv MCP tools.
Marked `integration` — not run in CI, requires claude-agent-sdk + network.

Research question: Gaussian Process surrogate modeling for high-dimensional
design-of-experiments — directly relevant to f3dasm's core use case.

Run with:
    uv run pytest tests/agentic/test_literature_wet.py -v -s --no-cov
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Deterministic: SS throttle / circuit breaker / API key sourcing
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_ss_rate_state():
    """Isolate literature_corpus's shared per-domain rate state and
    settings._config across tests — both are module globals."""
    from a3dasm._src import literature_corpus as lc_mod
    from a3dasm._src import settings as settings_mod
    domain = "api.semanticscholar.org"
    lc_mod._domain_consecutive_429.pop(domain, None)
    lc_mod._domain_cooldown_until.pop(domain, None)
    lc_mod._domain_last_request.pop(domain, None)
    settings_mod.configure({})
    yield
    lc_mod._domain_consecutive_429.pop(domain, None)
    lc_mod._domain_cooldown_until.pop(domain, None)
    lc_mod._domain_last_request.pop(domain, None)
    settings_mod.configure({})


def test_ss_throttle_shares_domain_rate_limiter(monkeypatch):
    """_throttled_ss paces itself through literature_corpus's per-domain
    limiter (the SAME one _robust_get/_robust_post use for this host) rather
    than a private, unauthenticated-tier-blind fixed interval."""
    import a3dasm._src.agents.literature as lit
    from a3dasm._src import literature_corpus as lc_mod

    calls = []
    monkeypatch.setattr(
        lc_mod, "_rate_limit_wait", lambda domain: calls.append(domain))
    monkeypatch.setattr(lit, "_call_in_fresh_thread", lambda fn, *a, **kw: fn())

    dummy = lambda: "ok"  # noqa: E731
    assert lit._throttled_ss(dummy) == "ok"
    assert calls == [lit._SS_DOMAIN]


def test_ss_429_retries_with_backoff_then_succeeds(monkeypatch):
    """A 429 (ConnectionRefusedError) is transient — retry with backoff
    instead of hard-failing the tool call on the first attempt."""
    import a3dasm._src.agents.literature as lit
    from a3dasm._src import literature_corpus as lc_mod

    monkeypatch.setattr(lc_mod, "_rate_limit_wait", lambda domain: None)
    slept = []
    monkeypatch.setattr(lit.time, "sleep", lambda s: slept.append(s))

    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ConnectionRefusedError("HTTP status 429 Too Many Requests.")
        return "ok"

    monkeypatch.setattr(
        lit, "_call_in_fresh_thread", lambda fn, *a, **kw: fn())
    assert lit._throttled_ss(flaky) == "ok"
    assert attempts["n"] == 2
    assert len(slept) == 1, "one backoff sleep expected before the retry"


def test_ss_403_is_not_retried(monkeypatch):
    """A 403 (PermissionError) means the shared unauthenticated quota is
    exhausted — retrying immediately cannot help, so it must propagate
    without _throttled_ss silently eating time on doomed retries."""
    import a3dasm._src.agents.literature as lit
    from a3dasm._src import literature_corpus as lc_mod

    monkeypatch.setattr(lc_mod, "_rate_limit_wait", lambda domain: None)
    attempts = {"n": 0}

    def forbidden():
        attempts["n"] += 1
        raise PermissionError("HTTP status 403 Forbidden.")

    monkeypatch.setattr(
        lit, "_call_in_fresh_thread", lambda fn, *a, **kw: fn())
    with pytest.raises(PermissionError):
        lit._throttled_ss(forbidden)
    assert attempts["n"] == 1, "403 must not be retried"


def test_ss_three_consecutive_429s_trip_shared_breaker(monkeypatch):
    """Three consecutive 429s trip literature_corpus's circuit breaker —
    the SAME breaker _robust_get/_robust_post use for this host — raising
    SourceCooldownError instead of a bare ConnectionRefusedError."""
    import a3dasm._src.agents.literature as lit
    from a3dasm._src import literature_corpus as lc_mod

    monkeypatch.setattr(lc_mod, "_rate_limit_wait", lambda domain: None)
    monkeypatch.setattr(lit.time, "sleep", lambda s: None)

    def always_429():
        raise ConnectionRefusedError("HTTP status 429 Too Many Requests.")

    monkeypatch.setattr(
        lit, "_call_in_fresh_thread", lambda fn, *a, **kw: fn())
    with pytest.raises(lc_mod.SourceCooldownError):
        lit._throttled_ss(always_429)


def test_ss_missing_key_warns(monkeypatch, caplog):
    """A missing semantic_scholar_api_key emits a warning, not an error."""
    import logging
    import tempfile
    import a3dasm._src.agents.literature as lit

    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

    # build_closure_tools needs a real study dir to resolve corpus paths.
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        study = Path(td)
        (study / "runs").mkdir()
        agent = lit.LiteratureReviewAgent()
        with caplog.at_level(logging.WARNING, logger=lit.__name__):
            agent.build_closure_tools(study)

    assert any("semantic_scholar_api_key" in r.message for r in caplog.records), (
        "expected a warning about missing semantic_scholar_api_key"
    )


def test_ss_key_settable_via_config_yaml(monkeypatch):
    """semantic_scholar_api_key resolves through settings (config.yaml's
    runtime: block, or F3DASM_SEMANTIC_SCHOLAR_API_KEY), not only via the
    bare SEMANTIC_SCHOLAR_API_KEY env var — matching every other run knob."""
    import tempfile
    from pathlib import Path
    import a3dasm._src.agents.literature as lit
    from a3dasm._src import settings as settings_mod

    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("F3DASM_SEMANTIC_SCHOLAR_API_KEY", raising=False)
    settings_mod.configure({"semantic_scholar_api_key": "from-config-yaml"})

    captured = {}

    class _FakeSS:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key

    monkeypatch.setattr(
        "semanticscholar.SemanticScholar", _FakeSS, raising=False)

    with tempfile.TemporaryDirectory() as td:
        study = Path(td)
        (study / "runs").mkdir()
        agent = lit.LiteratureReviewAgent()
        tools = agent.build_closure_tools(study)

    if "search_semantic_scholar" not in tools:
        pytest.skip("semanticscholar not installed")
    assert captured["api_key"] == "from-config-yaml"


def test_openalex_missing_key_warns(monkeypatch, caplog):
    """A missing OPENALEX_API_KEY emits a warning, not an error."""
    import logging
    import tempfile
    import a3dasm._src.agents.literature as lit

    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)

    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        study = Path(td)
        (study / "runs").mkdir()
        agent = lit.LiteratureReviewAgent()
        with caplog.at_level(logging.WARNING, logger=lit.__name__):
            agent.build_closure_tools(study)

    assert any("OPENALEX_API_KEY" in r.message for r in caplog.records), (
        "expected a warning about missing OPENALEX_API_KEY"
    )


def test_arxiv_read_download_use_direct_url(monkeypatch, tmp_path):
    """arxiv read/download must fetch the PDF by direct URL — not via the
    removed Result.download_pdf() (the 'no attribute download_pdf' failure that
    made the agent re-call DownloadPdf). No network: urlopen is stubbed."""
    from a3dasm._src.backends.openai_compatible import (
        _build_arxiv_closures,
    )
    tools = _build_arxiv_closures()
    if not tools:
        import pytest as _pt
        _pt.skip("arxiv package not installed")

    import urllib.request

    captured = {}

    class _Resp:
        def __init__(self, d):
            self._d = d

        def read(self):
            return self._d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _Resp(b"%PDF-1.5 fake body")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    # versioned id and an abs-URL form both normalise to the bare id
    out = tools["arxiv_download_paper"]("2301.12345v2", str(tmp_path))
    assert "Downloaded" in out
    assert captured["url"] == "https://arxiv.org/pdf/2301.12345v2"
    assert (tmp_path / "2301.12345v2.pdf").read_bytes() == b"%PDF-1.5 fake body"

    tools["arxiv_download_paper"]("http://arxiv.org/abs/1706.03762", str(tmp_path))
    assert captured["url"] == "https://arxiv.org/pdf/1706.03762"


def test_arxiv_client_self_throttles():
    """arxiv.Client default delay is 3 s — no extra throttle needed."""
    import arxiv
    c = arxiv.Client()
    assert c.delay_seconds == 3.0, (
        f"arxiv.Client default delay changed to {c.delay_seconds!r}; "
        "review whether _build_arxiv_closures needs its own throttle"
    )

from a3dasm._src.agent_runtime import AgenticRun
from a3dasm._src.agents import LiteratureReviewAgent, StrategizerAgent
from a3dasm._src.backends.base import Edge, Graph


# ---------------------------------------------------------------------------
# Research question and study setup
# ---------------------------------------------------------------------------

PROBLEM_STATEMENT = """\
# Research Problem: Surrogate Modelling for Design-of-Experiments

## Background
We are running a design-of-experiments campaign for a structural mechanics
optimisation problem. The design space is three-dimensional (continuous) and
evaluations are expensive (FEM simulations). We want to guide the search
using a surrogate model.

## Research questions for literature review
1. What Gaussian Process surrogate modelling strategies have been proposed
   for high-dimensional or expensive design-of-experiments problems?
2. What acquisition functions are recommended when the objective landscape
   has multiple local optima?
3. Has any prior work combined GP surrogates with physics-based constraints
   or coilability-type feasibility filters?

## Scope
Focus on methods published since 2015. Prefer papers with open-source
implementations or reproducible benchmarks.
"""

DELEGATION_TASK = """\
Answer the following research questions from primary literature.
Cite exact passages with page and paragraph references.
Do NOT answer from memory — every claim must come from a paper in the corpus.

Build a corpus of AT MOST 4 full-text papers (the most relevant ones).
Stop adding papers once 4 are in the corpus.
Prefer arXiv/open-access PDFs.

Questions:
1. What Gaussian Process surrogate modelling strategies have been proposed
   for high-dimensional or expensive design-of-experiments problems?
2. What acquisition functions are recommended when the objective landscape
   has multiple local optima?
3. Has any prior work combined GP surrogates with physics-based constraints
   or feasibility filters?
"""


@pytest.mark.integration
def test_literature_review_wet(tmp_path, capfd):
    """End-to-end wet test: LiteratureReviewAgent with real LLM + arxiv MCP."""

    # --- Study setup -------------------------------------------------------
    study = tmp_path / "study"
    study.mkdir()
    (study / "PROBLEM_STATEMENT.md").write_text(PROBLEM_STATEMENT)

    # --- Scripted strategist -----------------------------------------------
    # Delegates once to literature_reviewer then calls Done().
    # Does NOT use two-shot Done() guard since no critic in this graph.
    from tests.test_nodes import StubAdapter
    import re, time as _time

    class LitReviewStrategist(StubAdapter):
        def invoke(self, messages):
            # Propose a hypothesis to satisfy the ledger (if active).
            # On re-invocation (graph loop), reuse the existing open
            # hypothesis instead of proposing duplicates until the
            # MAX_OPEN cap rejects and the ID fallback loops forever.
            h_id = None
            if "HypothesisPropose" in self.closure_tools:
                listing = self.closure_tools["HypothesisList"]()
                m_existing = re.search(r"\bH\d+\b", listing or "")
                if m_existing:
                    h_id = m_existing.group()
                else:
                    h_id = self.closure_tools["HypothesisPropose"](
                        statement="GP surrogates with acquisition"
                                  " functions are the dominant approach"
                                  " for expensive DOE.",
                        falsification_criterion=(
                            "a non-GP method outperforms GP on benchmark"
                        ),
                        prediction="GP wins on majority of DOE benchmarks",
                        prior=0.7,
                    )

            # Delegate to literature reviewer
            if not h_id or h_id.startswith("ERROR"):
                raise AssertionError(
                    f"could not obtain a hypothesis ID: {h_id!r}"
                )
            h_ids = [h_id]
            result = self.closure_tools["Delegate"](
                target="literature_reviewer",
                intent=DELEGATION_TASK,
                expected_report=(
                    "### Papers reviewed\n"
                    "### Key findings\n"
                    "### Conclusions\n"
                    "### Numbers"
                ),
                hypothesis_ids=h_ids,
            )

            # Poll until Done/Errored — up to 20 min (budget is 15)
            worker_report = ""
            m = re.search(r"D\d{3}", result)
            if m:
                d_id = m.group()
                for _ in range(2400):  # 2400 × 0.5s = 20 min ceiling
                    status = self.closure_tools["GetStatus"](d_id)
                    if not status.strip().startswith("Working"):
                        worker_report = re.sub(r"^Done\s*\n+", "", status, flags=re.DOTALL)
                        break
                    _time.sleep(0.5)

            summary = worker_report or "Literature review complete."
            # pipeline.ipynb is a hard Done() requirement — author it via the
            # structured tools (a minimal self-asserting analysis cell).
            if "AddPipelineCell" in self.closure_tools:
                self.closure_tools["AddPipelineMarkdownCell"](
                    "problem", "Literature review of the problem.")
                self.closure_tools["AddPipelineCell"](
                    "analysis", "re-run the literature review delegation",
                    "print('REPRODUCED: 0.0')")
            # Two-shot Done(): first call → warning; second → closes.
            # If delegation is still working, wait briefly and retry.
            r1 = self.closure_tools["Done"](summary=summary)
            if "ERROR" in r1 and "still running" in r1:
                _time.sleep(10)
                self.closure_tools["Done"](summary=summary)
                self.closure_tools["Done"](summary=summary)
            else:
                self.closure_tools["Done"](summary=summary)
            return "Done."

    # --- Graph -------------------------------------------------------------
    class _StratSpec(StrategizerAgent):
        description = "Orchestrates the literature review run."

    class _LitSpec(LiteratureReviewAgent):
        pass  # inherits everything

    graph = Graph(
        nodes={
            "strategizer": _StratSpec(),
            "literature_reviewer": _LitSpec(),
        },
        edges=(Edge("strategizer", "literature_reviewer"),),
        entry="strategizer",
    )

    # --- Run ---------------------------------------------------------------
    run = AgenticRun(
        study_dir=study,
        graph=graph,
        budget=20 * 60,  # 20-minute wall-clock cap
    )

    # Inject scripted strategist adapter
    strat_adapter = LitReviewStrategist()

    def _mock_make_adapter(name, agent):
        if name == "strategizer":
            return strat_adapter
        # LiteratureReviewAgent: build the real adapter with corpus tools
        from a3dasm._src.backends.claude import ClaudeAdapter
        native = [t for t in agent.tools
                  if t in {"Bash", "Edit", "Read", "Write", "Glob", "Grep"}]
        # The adapter spawns the claude CLI with cwd=study_dir; a
        # nonexistent cwd makes subprocess.Popen fail instantly.
        (study / "workspace").mkdir(parents=True, exist_ok=True)
        adapter = ClaudeAdapter(
            model="claude-haiku-4-5-20251001",
            system_prompt=agent.system_prompt,
            study_dir=study / "workspace",
            native_tools=native,
            extra_mcp_servers=dict(agent.mcp_servers),
            extra_allowed_tools=list(agent.extra_allowed_tools),
        )
        extra = agent.build_closure_tools(study)
        if extra:
            adapter.closure_tools.update(extra)
        return adapter

    run._make_adapter = _mock_make_adapter

    start = time.time()
    report = run.execute()
    elapsed = time.time() - start

    # --- Assertions: did the reviewer DO ITS JOB? -------------------------
    # The bar is the reviewer's contract, not surface plausibility: its tools
    # were callable, it gathered REAL full-text evidence (no phantom papers, no
    # MuPDF noise), and the review is grounded in that corpus (not memory).
    import csv as _csv

    run_dir = next((study / "runs").iterdir())
    corpus_dir = study / "delegations" / "literature"

    # The literature DELEGATION's report — NOT the run's final headline, which is
    # "⚠ UNGATED" boilerplate when the scripted Done() is refused by the critic.
    records = [json.loads(l) for l in
               (run_dir / "debug" / "delegation_log.jsonl").read_text().splitlines()
               if l.strip()]
    lit_done = [r for r in records
                if r.get("to_node") == "literature_reviewer" and r.get("deliverable")]
    assert lit_done, "no completed literature delegation produced a report"
    lit = lit_done[-1]
    review = lit["deliverable"]
    assert len(review) > 100, f"literature report too short: {review!r}"

    # (The bare-vs-qualified "No such tool available" contract is guarded
    # deterministically and offline by test_no_agent_bare_advertises_its_closures;
    # this wet test asserts the end-to-end OUTCOME instead.)

    # 1. CLEAN TOOLS (deterministic): PDF parsing emitted no MuPDF stderr spam.
    cap = capfd.readouterr()
    assert "MuPDF error" not in (cap.out + cap.err), (
        "PDF extraction emitted MuPDF errors — half-baked tooling")

    # 2. NO PHANTOM FULL-TEXT (deterministic): every paper marked full-text has a
    #    real extracted body (>5000 chars) — never a placeholder stored as
    #    quotable (the arxiv_1502_05700 236-char bug).
    corpus_csv = corpus_dir / "corpus.csv"
    assert corpus_csv.exists(), "no corpus.csv — the reviewer indexed nothing"
    rows = list(_csv.DictReader(corpus_csv.open()))
    fulltext = [r for r in rows if r.get("full_text") == "true"]
    for r in fulltext:
        md = Path(r["local_md_path"])
        n = len(md.read_text(encoding="utf-8")) if md.exists() else 0
        assert n > 5000, (
            f"{r['paper_id']} marked full-text but body is {n} chars — "
            "a failed extraction stored as quotable")

    # 3. REAL EVIDENCE (the job): at least two full-text papers were gathered.
    assert len(fulltext) >= 2, (
        f"only {len(fulltext)} full-text papers gathered — too thin to ground a "
        "review")

    # 4. GROUNDED (the job): the review references papers that are IN the corpus
    #    (by arxiv id / doi / first-author surname) — not synthesised from memory.
    rl = review.lower()
    anchors = []
    for r in fulltext:
        for k in (r.get("arxiv_id"), r.get("doi")):
            if k:
                anchors.append(k.lower())
        au = (r.get("authors") or "").replace(",", " ").split()
        if au:
            anchors.append(au[0].lower())  # first-author surname/given
    grounded = sum(1 for a in anchors if a and a in rl)
    assert grounded >= 1, (
        "the review references no corpus paper by id or author — possible "
        "memory synthesis")

    # 5. token table stamped into the deliverable notebook
    import nbformat
    nb = nbformat.read(str(study / "pipeline.ipynb"), as_version=4)
    md = "\n\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    assert "## Token usage" in md

    print(f"\n✓ Wet test passed in {elapsed:.0f}s: {len(fulltext)} full-text "
          f"papers, {grounded} grounded refs, clean tools, no MuPDF errors")
