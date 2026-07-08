"""Tests for the out-of-process bge-small embed worker and its host integration.

Test inventory:
1. _SubprocessEmbedder.embed builds the right command; vectors parsed;
   np.array round-trip gives correct shape.
2. Nonzero returncode → RuntimeError containing stderr tail.
3. _get_embedding_model fallback chain:
   a. in-process ImportError + successful subprocess probe → returns
      _SubprocessEmbedder; subprocess.run called exactly once.
   b. in-process ImportError + failed probe → returns None; exactly one
      warning logged; second call does NOT re-probe (tri-state cache).
4. _embed_worker.main() protocol: fake fastembed module injected into
   sys.modules; stdin/stdout via StringIO; valid JSON out.
5. @pytest.mark.integration: real uv subprocess; 384-dim vectors.
   Skipped by default — may download ~100 MB on first run.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import a3dasm._src.literature_corpus as lc_mod
from a3dasm._src.literature_corpus import (
    LiteratureCorpus,
    _SubprocessEmbedder,
)

# ---------------------------------------------------------------------------
# Module-level path to worker script
# ---------------------------------------------------------------------------

_WORKER_PATH = (
    Path(__file__).parent.parent
    / "src" / "a3dasm" / "_src" / "_embed_worker.py"
)


def _reset_subprocess_cache():
    """Reset the module-level tri-state cache between tests."""
    lc_mod._subprocess_embedder_state = None
    lc_mod._subprocess_embedder_warned = False


# ---------------------------------------------------------------------------
# Test 1: _SubprocessEmbedder builds correct command and parses vectors
# ---------------------------------------------------------------------------

class TestSubprocessEmbedderCommand:
    def test_correct_command_built_and_vectors_parsed(self, monkeypatch):
        """embed() builds the right uv command and returns list-of-lists."""
        captured = {}

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = json.dumps({"vectors": [[0.1, 0.2], [0.3, 0.4]]})
        fake_result.stderr = ""

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["input"] = kwargs.get("input", "")
            return fake_result

        monkeypatch.setattr("subprocess.run", fake_run)

        embedder = _SubprocessEmbedder(timeout=30.0)
        result = embedder.embed(["hello", "world"])

        cmd = captured["cmd"]
        # uv run --no-project --quiet --python 3.12 --with <spec> python <worker>
        assert cmd[0] == "uv"
        assert "--no-project" in cmd
        assert "--quiet" in cmd
        assert "--with" in cmd
        with_idx = cmd.index("--with")
        assert "fastembed" in cmd[with_idx + 1]
        # Isolated env pinned to Python 3.12 — onnxruntime/fastembed have cp312
        # wheels; the 3.13 host has none. No numpy/onnxruntime pins needed since
        # the subprocess env is independent of the host's numpy-2.
        assert "--python" in cmd
        assert cmd[cmd.index("--python") + 1] == "3.12"
        assert cmd[-1].endswith("_embed_worker.py")

        # Input is valid JSON with "texts"
        payload = json.loads(captured["input"])
        assert payload["texts"] == ["hello", "world"]

        # Output is list of lists
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_numpy_array_roundtrip_shape(self, monkeypatch):
        """np.array(list(embedder.embed(...)), dtype=float32) has correct shape."""
        vectors_384 = [[float(i) / 384 for i in range(384)],
                       [float(i + 1) / 384 for i in range(384)]]

        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = json.dumps({"vectors": vectors_384})
        fake_result.stderr = ""

        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_result)

        embedder = _SubprocessEmbedder()
        arr = np.array(list(embedder.embed(["a", "b"])), dtype=np.float32)

        assert arr.shape == (2, 384)
        assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# Test 2: Nonzero returncode → RuntimeError with stderr tail
# ---------------------------------------------------------------------------

class TestSubprocessEmbedderError:
    def test_nonzero_returncode_raises_runtime_error(self, monkeypatch):
        """returncode != 0 → RuntimeError mentioning stderr tail."""
        long_stderr = "X" * 1000 + "the_real_error"

        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = ""
        fake_result.stderr = long_stderr

        monkeypatch.setattr("subprocess.run", lambda *a, **kw: fake_result)

        embedder = _SubprocessEmbedder()
        with pytest.raises(RuntimeError) as exc_info:
            embedder.embed(["test"])

        # Error message contains the tail of stderr (last 500 chars)
        assert "the_real_error" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 3: _get_embedding_model fallback chain
# ---------------------------------------------------------------------------

class TestGetEmbeddingModelFallbackChain:
    def setup_method(self):
        _reset_subprocess_cache()

    def teardown_method(self):
        _reset_subprocess_cache()

    def test_inprocess_importerror_successful_probe_returns_subprocess_embedder(
        self, tmp_path, monkeypatch, caplog
    ):
        """In-process ImportError + successful probe → _SubprocessEmbedder returned.
        subprocess.run called exactly once (for the probe).
        """
        import logging

        run_calls = []

        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            result = MagicMock()
            result.returncode = 0
            result.stdout = json.dumps({"vectors": [[0.1] * 384]})
            result.stderr = ""
            return result

        # Force in-process fastembed to fail
        monkeypatch.setattr("subprocess.run", fake_run)
        _reset_subprocess_cache()

        corpus = LiteratureCorpus(tmp_path / "corp")

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("intentional test ImportError")
            return real_import(name, *args, **kwargs)

        with caplog.at_level(logging.INFO,
                             logger="a3dasm._src.literature_corpus"):
            with patch("builtins.__import__", side_effect=mock_import):
                with patch("shutil.which", return_value="/usr/local/bin/uv"):
                    model = corpus._get_embedding_model()

        assert isinstance(model, _SubprocessEmbedder), (
            f"Expected _SubprocessEmbedder, got {type(model)}"
        )
        assert len(run_calls) == 1, (
            f"Expected exactly 1 subprocess.run call for probe, got {len(run_calls)}"
        )
        # Info log present
        info_msgs = [r.message for r in caplog.records
                     if r.levelno == logging.INFO]
        assert any("out-of-process" in m for m in info_msgs), (
            f"Expected info about out-of-process worker: {info_msgs}"
        )

    def test_inprocess_importerror_failed_probe_returns_none_one_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        """In-process ImportError + probe failure → None returned; exactly one warning."""
        import logging

        _reset_subprocess_cache()

        def fake_run(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "probe failed for test"
            return result

        monkeypatch.setattr("subprocess.run", fake_run)

        corpus = LiteratureCorpus(tmp_path / "corp2")

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("intentional test ImportError")
            return real_import(name, *args, **kwargs)

        with caplog.at_level(logging.WARNING,
                             logger="a3dasm._src.literature_corpus"):
            with patch("builtins.__import__", side_effect=mock_import):
                with patch("shutil.which", return_value="/usr/local/bin/uv"):
                    result = corpus._get_embedding_model()

        assert result is None, f"Expected None, got {result}"

        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING
                    and "fastembed" in r.message]
        assert len(warnings) == 1, (
            f"Expected exactly 1 warning, got {len(warnings)}: "
            f"{[r.message for r in warnings]}"
        )
        assert "BM25" in warnings[0].message

    def test_tristate_cache_second_call_does_not_reprobe(
        self, tmp_path, monkeypatch
    ):
        """After probe failure, second call does NOT re-probe subprocess."""
        _reset_subprocess_cache()

        run_calls = []

        def fake_run(cmd, **kwargs):
            run_calls.append(cmd)
            result = MagicMock()
            result.returncode = 1
            result.stdout = ""
            result.stderr = "fail"
            return result

        monkeypatch.setattr("subprocess.run", fake_run)

        corpus = LiteratureCorpus(tmp_path / "corp3")

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("intentional test ImportError")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with patch("shutil.which", return_value="/usr/local/bin/uv"):
                corpus._get_embedding_model()
                # Reset per-instance cache so it goes through global check
                corpus._embedding_model = None
                corpus._get_embedding_model()
                corpus._embedding_model = None
                corpus._get_embedding_model()

        assert len(run_calls) == 1, (
            f"Subprocess probed {len(run_calls)} times; expected exactly 1 "
            "(tri-state cache should prevent re-probing)"
        )

    def test_uv_not_found_goes_straight_to_none(
        self, tmp_path, monkeypatch, caplog
    ):
        """When uv is not on PATH, subprocess route is skipped immediately."""
        import logging
        _reset_subprocess_cache()

        run_calls = []
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: run_calls.append(a) or MagicMock(returncode=0),
        )

        corpus = LiteratureCorpus(tmp_path / "corp4")

        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("intentional test ImportError")
            return real_import(name, *args, **kwargs)

        with caplog.at_level(logging.WARNING,
                             logger="a3dasm._src.literature_corpus"):
            with patch("builtins.__import__", side_effect=mock_import):
                with patch("shutil.which", return_value=None):
                    result = corpus._get_embedding_model()

        assert result is None
        assert len(run_calls) == 0, "No subprocess.run when uv not found"


# ---------------------------------------------------------------------------
# Test 4: _embed_worker.main() protocol
# ---------------------------------------------------------------------------

class TestEmbedWorkerMain:
    def test_worker_main_outputs_valid_json(self):
        """_embed_worker.main reads JSON from stdin, writes vectors to stdout."""
        # Inject a fake fastembed module so no real import is needed
        fake_fastembed = types.ModuleType("fastembed")

        class FakeTextEmbedding:
            def __init__(self, model_name):
                pass

            def embed(self, texts):
                for _ in texts:
                    yield [0.5] * 4  # short fake vectors

        fake_fastembed.TextEmbedding = FakeTextEmbedding  # type: ignore[attr-defined]

        # Load the worker module fresh (avoiding cached state)
        spec = importlib.util.spec_from_file_location(
            "_embed_worker_test", str(_WORKER_PATH)
        )
        worker_mod = importlib.util.module_from_spec(spec)

        input_payload = json.dumps({"texts": ["hello", "world"]})
        fake_stdin = StringIO(input_payload)
        fake_stdout = StringIO()

        old_modules = sys.modules.copy()
        sys.modules["fastembed"] = fake_fastembed  # type: ignore[assignment]
        try:
            spec.loader.exec_module(worker_mod)
            old_stdin, old_stdout = sys.stdin, sys.stdout
            sys.stdin = fake_stdin
            sys.stdout = fake_stdout
            try:
                exit_code = worker_mod.main()
            finally:
                sys.stdin = old_stdin
                sys.stdout = old_stdout
        finally:
            # Restore sys.modules
            for key in list(sys.modules.keys()):
                if key not in old_modules:
                    del sys.modules[key]
            sys.modules.update(old_modules)

        assert exit_code == 0
        output = fake_stdout.getvalue()
        data = json.loads(output)
        assert "vectors" in data
        assert len(data["vectors"]) == 2
        assert data["vectors"][0] == [0.5] * 4

    def test_worker_script_exists(self):
        """The worker script is present at the expected path."""
        assert _WORKER_PATH.exists(), (
            f"Worker script not found at {_WORKER_PATH}"
        )


# ---------------------------------------------------------------------------
# Test 5: Integration — real uv subprocess (skipped by default)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_embed_worker_real_uv_384_dim():
    """Real uv subprocess: embed(['hello']) → 384-dim vector.

    Skipped by default. May download ~100 MB of model weights on first run.
    Run with: pytest -m integration tests/agentic/test_embed_worker.py
    """
    import shutil
    if shutil.which("uv") is None:
        pytest.skip("uv not found on PATH")

    embedder = _SubprocessEmbedder(timeout=600.0)
    result = embedder.embed(["hello world"])

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) == 1, f"Expected 1 vector, got {len(result)}"
    assert len(result[0]) == 384, (
        f"Expected 384-dim bge-small vector, got dim={len(result[0])}"
    )
    arr = np.array(result, dtype=np.float32)
    assert arr.shape == (1, 384)
    assert arr.dtype == np.float32
