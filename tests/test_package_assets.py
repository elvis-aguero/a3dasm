"""Every non-Python file the package loads at runtime must be packaged.

The failure this guards is invisible in the repo and fatal on install:
``MANIFEST.in`` said ``recursive-include src *.py *.pyi``, so
``viewer/templates/graph.html`` was simply absent from the wheel. Every
test passed, the repo checkout worked, and ``pip install a3dasm[viewer]``
produced a viewer whose only human-facing page returned 500 — the API
answered 200, so even a smoke check of the endpoints would have missed it.

Asserting the file exists on disk cannot catch this, because in the repo it
always does. So this checks the packaging RULES instead: every asset under
``src/a3dasm`` must be matched by a ``MANIFEST.in`` include line.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"

# Byproducts, not assets.
_IGNORED_SUFFIXES = {".pyc", ".pyo", ".so", ".pyd"}
_IGNORED_DIRS = {"__pycache__", ".egg-info"}


def _manifest_patterns() -> list[str]:
    """Glob patterns MANIFEST.in includes, relative to the repo root."""
    patterns: list[str] = []
    for raw in (_ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        verb = parts[0].lower()
        if verb == "include":
            patterns.extend(parts[1:])
        elif verb == "recursive-include" and len(parts) >= 3:
            base = parts[1].rstrip("/")
            patterns.extend(f"{base}/**/{pat}" for pat in parts[2:])
            # recursive-include also matches directly inside the base dir.
            patterns.extend(f"{base}/{pat}" for pat in parts[2:])
    return patterns


def _assets() -> list[Path]:
    out = []
    for path in _SRC.rglob("*"):
        if not path.is_file() or path.suffix == ".py":
            continue
        if path.suffix in _IGNORED_SUFFIXES:
            continue
        if any(part in _IGNORED_DIRS or part.endswith(".egg-info")
               for part in path.parts):
            continue
        out.append(path)
    return out


def test_every_runtime_asset_is_covered_by_the_manifest():
    patterns = _manifest_patterns()
    missing = []
    for path in _assets():
        rel = path.relative_to(_ROOT).as_posix()
        if not any(fnmatch.fnmatch(rel, pat) for pat in patterns):
            missing.append(rel)
    assert not missing, (
        "these files ship inside the package but no MANIFEST.in rule "
        f"includes them, so they will be absent from the wheel: {missing}"
    )


def test_the_viewer_template_is_one_of_those_assets():
    """Pins the specific file whose absence broke the installed viewer.

    Without this, a refactor that moved or renamed the template would leave
    the generic check above passing against an empty set.
    """
    template = _SRC / "a3dasm" / "_src" / "viewer" / "templates" / "graph.html"
    assert template.is_file(), "the viewer's only page template is missing"
    assert template in _assets()
