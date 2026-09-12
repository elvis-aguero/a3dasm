"""The `runtime:` config block is an explicit, documented contract.

Two failure modes this file exists to prevent, both of the same kind — a
setting that silently does nothing:

1. A knob is read in the code but missing from ``KNOWN_KEYS``, so putting it
   in a study's ``config.yaml`` triggers the "unrecognised key" warning even
   though it works.
2. A knob is documented or listed but nothing reads it, so a user sets it and
   nothing happens.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from a3dasm._src.runtime import settings

_SRC = Path(__file__).resolve().parent.parent / "src" / "a3dasm" / "_src"
_DOC = (Path(__file__).resolve().parent.parent
        / "docs" / "authoring-a-study.md")

# get_bool("debug", False) / get_str("semantic_scholar_api_key", "") / ...
# `_?` so aliased imports (`from ...settings import get_int as _get_int`)
# are matched too — missing those made three live knobs look orphaned.
_CALL = re.compile(
    r"_?get_(?:bool|int|float|str)\(\s*[\"']([a-z0-9_]+)[\"']\s*,\s*([^)]*)\)")


def _keys_read_in_source() -> set[str]:
    found: set[str] = set()
    for path in _SRC.rglob("*.py"):
        if path.name == "settings.py":
            continue  # its own signatures, not call sites
        found |= {m[0] for m in _CALL.findall(path.read_text(encoding="utf-8"))}
    return found


def test_every_knob_read_in_the_code_is_a_known_key():
    """Otherwise setting it in config.yaml warns that it is unrecognised
    while quietly working — the config lying about itself."""
    missing = _keys_read_in_source() - settings.KNOWN_KEYS
    assert not missing, (
        f"read via settings.get_* but absent from KNOWN_KEYS: {sorted(missing)}"
    )


def test_every_known_key_is_actually_read_somewhere():
    """A knob nothing reads is a setting a user can set with no effect."""
    orphans = settings.KNOWN_KEYS - _keys_read_in_source()
    assert not orphans, (
        f"in KNOWN_KEYS but never read: {sorted(orphans)}"
    )


def test_unrecognised_runtime_key_warns(caplog):
    with caplog.at_level("WARNING"):
        settings.configure({"debug": True, "dbeug": True})
    assert "dbeug" in caplog.text
    assert "unrecognised" in caplog.text.lower()
    settings.configure(None)


def test_recognised_keys_do_not_warn(caplog):
    with caplog.at_level("WARNING"):
        settings.configure({"debug": True, "recursion_limit": 100})
    assert "unrecognised" not in caplog.text.lower()
    settings.configure(None)


@pytest.mark.parametrize("key", sorted(settings.KNOWN_KEYS))
def test_runtime_block_is_documented_for_study_authors(key):
    """Explicit over implicit: a knob a user is expected to set belongs in the
    study-authoring docs, not only in the source that reads it."""
    text = _DOC.read_text(encoding="utf-8")
    assert "`runtime`" in text or "runtime:" in text, (
        "docs/authoring-a-study.md does not document the runtime: block at all"
    )
    assert key in text, f"runtime knob {key!r} is undocumented for study authors"


def _defaults_in_source() -> dict[str, str]:
    out: dict[str, str] = {}
    for path in _SRC.rglob("*.py"):
        if path.name == "settings.py":
            continue
        for key, default in _CALL.findall(path.read_text(encoding="utf-8")):
            out[key] = default.strip()
    return out


@pytest.mark.parametrize("key", sorted(settings.KNOWN_KEYS))
def test_documented_default_matches_the_source(key):
    """A documented default that is not the real one is worse than no docs:
    the reader makes a decision on it. Seven of this table's defaults were
    wrong when it was first written, which is why this test exists."""
    src_default = _defaults_in_source().get(key)
    if src_default is None:
        pytest.skip(f"{key} read without a literal default")

    row = next((ln for ln in _DOC.read_text(encoding="utf-8").splitlines()
                if ln.startswith(f"| `{key}` |")), None)
    assert row, f"{key} has no row in the runtime table"
    documented = row.rsplit("|", 2)[1].strip().strip("`")

    norm = {"False": "false", "True": "true", '""': "none"}
    expected = norm.get(src_default, src_default)
    assert documented == expected, (
        f"{key}: docs say {documented!r}, source default is {src_default!r}"
    )
