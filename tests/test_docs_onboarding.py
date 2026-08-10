"""Onboarding-doc consistency — no live calls, no API cost.

The install command drifted into THREE different, two of them broken
(README.md and the quickstart notebook both pointed at a private ssh:// URL,
the wrong GitHub org, and a git tag that doesn't exist) while
docs/installation.md quietly had the correct one. These tests pin a single
source of truth so that can't happen silently again.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _pip_install_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if "pip install" in line]


def test_readme_and_installation_doc_agree_on_the_install_command():
    readme = (_ROOT / "README.md").read_text()
    install_doc = (_ROOT / "docs" / "installation.md").read_text()

    readme_lines = _pip_install_lines(readme)
    install_lines = _pip_install_lines(install_doc)

    assert readme_lines, "README.md has no pip install line"
    assert install_lines, "docs/installation.md has no pip install line"
    assert readme_lines[0] in install_lines, (
        f"README's install command {readme_lines[0]!r} does not appear in "
        "docs/installation.md — the two have drifted apart again"
    )


def test_install_command_is_not_a_broken_ssh_or_pinned_tag_url():
    for path in (_ROOT / "README.md", _ROOT / "docs" / "installation.md"):
        text = path.read_text()
        for line in _pip_install_lines(text):
            assert "git+ssh://" not in line, (
                f"{path}: install command uses a private ssh:// URL, which "
                f"requires SSH-key access most readers won't have: {line!r}"
            )
            assert not re.search(r"@v\d+\.\d+\.\d+", line), (
                f"{path}: install command pins a git tag ({line!r}) — verify "
                "it actually exists (`git tag -l`) before shipping this"
            )


def test_quickstart_notebook_is_valid_and_does_not_duplicate_install_command():
    nb_path = _ROOT / "docs" / "notebooks" / "quickstart.ipynb"
    nb = json.loads(nb_path.read_text())
    assert nb["cells"], "quickstart notebook has no cells"

    src = "".join(
        "".join(cell["source"]) for cell in nb["cells"]
    )
    # The notebook should point at Installation, not repeat (and risk
    # re-diverging from) its own copy of the install command.
    assert "pip install" not in src, (
        "quickstart notebook repeats the install command inline instead of "
        "linking to installation.md — this is exactly how it drifted before"
    )
    assert "installation.md" in src or "Installation" in src, (
        "quickstart notebook should point readers at the Installation page"
    )
