"""Out-of-process dense embedder (bge-small via fastembed, isolated uv env).

Used by :class:`.literature_corpus.LiteratureCorpus` when fastembed cannot
import in the host interpreter. The worker script is ``_embed_worker.py``
next to this module; the env spec below is the only place its pins live.
"""

from __future__ import annotations

__all__ = [
    "_EMBED_PYTHON",
    "_EMBED_WITH",
    "_SubprocessEmbedder",
]

# fastembed runs bge-small via onnxruntime (light: ONNX, no torch). onnxruntime
# ships no CPython-3.13 wheel, so it cannot import in a 3.13 host. The worker
# runs in an ISOLATED uv env, so we just pin that env to Python 3.12 (where
# fastembed+onnxruntime+numpy all have wheels) — no host pins needed, and the
# host's numpy-2 is irrelevant across the process boundary.
#
# Upper-bounded (not just >=0.3): an unbounded floating spec here is exactly
# the footgun that caused docling's CI segfault (2026-08-26 — a transitive
# default changed under a version bump nobody asked for). Observed directly
# (run 20260830T004106, delegation D035): "ImportError: cannot import name
# 'TextEmbedding' from 'fastembed' (unknown location)" — confirmed
# `TextEmbedding` is still exported by fastembed 0.8.0 upstream (not an API
# removal), and the "(unknown location)" phrasing is the fingerprint of a
# namespace-package resolution (no real __init__.py) rather than a genuine
# API break — most likely `uv`'s ephemeral env transiently resolving a
# broken/partial install (possibly colliding with the fastembed-gpu variant,
# which provides the same top-level namespace) in that one throwaway
# environment; two other delegations in the SAME run had working
# CorpusSearch calls, so this looks environment-specific, not deterministic.
# This upper bound doesn't necessarily fix that particular transient failure
# (unverified — Oscar is read-only, so it can't be reproduced/confirmed
# there), but it's the same cheap, low-risk insurance against a FUTURE real
# break either way.
_EMBED_WITH = "fastembed>=0.3,<0.9"
_EMBED_PYTHON = "3.12"


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
