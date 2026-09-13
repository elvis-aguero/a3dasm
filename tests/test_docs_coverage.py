"""The docs must keep covering what the package actually exposes.

Three ways documentation rots silently, all of them caught here rather than
by a reader who could not find the thing:

* an export is added to ``a3dasm.__all__`` and never written about — the
  reason ``Workspace`` (the symbolic-derivation DSL) and the whole live
  viewer were once absent from ``docs/`` entirely;
* a page is written and never added to the nav, so nothing links to it;
* an internal link points at a page that moved or was renamed.

No network, no build step: pure text over ``docs/`` and ``mkdocs.yml``.
``mkdocs build --strict`` still owns the rendering-level checks (a ``:::``
directive that will not resolve, a broken link into the generated API pages).
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "docs"
_MKDOCS = _ROOT / "mkdocs.yml"


def _docs_text() -> str:
    return "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(_DOCS.rglob("*.md"))
    )


def _exports() -> list[str]:
    tree = ast.parse((_ROOT / "src" / "a3dasm" / "__init__.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError("a3dasm/__init__.py defines no __all__")


def _nav_pages() -> list[str]:
    """Every page path in mkdocs.yml's nav, without a YAML parser.

    The nav is a flat-enough list of ``'x.md'`` / ``'Label': 'x.md'`` entries
    that a targeted regex is more robust here than depending on PyYAML's
    handling of the surrounding theme block.
    """
    text = _MKDOCS.read_text(encoding="utf-8")
    nav = text.split("\nnav:", 1)[1]
    return re.findall(r"'([^']+\.(?:md|ipynb))'", nav)


def test_every_public_export_is_mentioned_in_the_docs():
    text = _docs_text()
    missing = [name for name in _exports() if name not in text]
    assert not missing, (
        f"exported from a3dasm but absent from docs/: {missing}. "
        "Add it to the API reference or write about it in a guide."
    )


def test_every_nav_page_exists():
    missing = [page for page in _nav_pages() if not (_DOCS / page).exists()]
    assert not missing, f"mkdocs.yml nav points at missing pages: {missing}"


def test_every_docs_page_is_reachable_from_the_nav():
    nav = {page for page in _nav_pages()}
    orphans = [
        str(p.relative_to(_DOCS))
        for p in sorted(_DOCS.rglob("*.md"))
        if str(p.relative_to(_DOCS)) not in nav
        and not str(p.relative_to(_DOCS)).startswith("_")
    ]
    assert not orphans, (
        f"pages not in mkdocs.yml nav, so nothing links to them: {orphans}"
    )


@pytest.mark.parametrize(
    "page", sorted(p.relative_to(_DOCS) for p in _DOCS.rglob("*.md"))
)
def test_internal_markdown_links_resolve(page):
    """Relative links between docs pages point at files that exist."""
    text = (_DOCS / page).read_text(encoding="utf-8")
    broken = []
    for target in re.findall(r"\]\(([^)#\s]+\.md)(?:#[^)\s]*)?\)", text):
        if target.startswith(("http://", "https://", "/")):
            continue
        if not ((_DOCS / page).parent / target).resolve().exists():
            broken.append(target)
    assert not broken, f"{page} links to missing pages: {broken}"
