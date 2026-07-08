"""Commit 2: the soft, capped, offender-directed eval-budget governor at the flush
boundary. Fires MID-delegation (where the strategizer's turn-gated check is blind),
nudges to the campaign's own stdout (→ the implementer's report), never stops the
campaign, and is capped at one nudge per band."""
from __future__ import annotations

import json

from a3dasm._src.instrumented import InstrumentedDataGenerator


def _gen(tmp_path, budget):
    (tmp_path / "debug").mkdir(parents=True, exist_ok=True)
    return InstrumentedDataGenerator(
        inner=object(),  # governor never touches inner
        store_dir=tmp_path / "experiment_data",
        delegation_id="D001",
        eval_budget=budget,
    )


def test_no_nudge_below_first_band(tmp_path, capsys):
    g = _gen(tmp_path, 1000)
    g._maybe_nudge_budget(700)  # 70% < 80%
    assert capsys.readouterr().out == ""
    assert g._nudge_bands_hit == set()


def test_nudge_at_80_percent_goes_to_stdout(tmp_path, capsys):
    g = _gen(tmp_path, 1000)
    g._maybe_nudge_budget(800)
    out = capsys.readouterr().out
    assert "EVAL BUDGET" in out and "D001" in out and "80%" in out
    assert "RunScratch" in out  # steers debugging off the real oracle
    assert 80 in g._nudge_bands_hit
    # audit-only diagnostic written
    diag = tmp_path / "debug" / "diagnostics.jsonl"
    rec = json.loads(diag.read_text().splitlines()[-1])
    assert rec["error_type"] == "BUDGET_WARN"


def test_capped_one_per_band_max_three(tmp_path, capsys):
    g = _gen(tmp_path, 1000)
    for n in (820, 840, 860):       # all in the 80% band
        g._maybe_nudge_budget(n)
    assert capsys.readouterr().out.count("EVAL BUDGET") == 1  # one nudge, not three
    g._maybe_nudge_budget(1000)     # 100% band
    g._maybe_nudge_budget(1500)     # 150% band
    g._maybe_nudge_budget(3000)     # already past all bands
    extra = capsys.readouterr().out
    assert extra.count("EVAL BUDGET") == 2          # 100 + 150 only
    assert g._nudge_bands_hit == {80, 100, 150}      # fixed cap = 3 bands


def test_big_jump_nudges_once_and_marks_all_crossed(tmp_path, capsys):
    g = _gen(tmp_path, 1000)
    g._maybe_nudge_budget(1600)  # jumps 0 → 160% in one flush
    assert capsys.readouterr().out.count("EVAL BUDGET") == 1
    assert g._nudge_bands_hit == {80, 100, 150}
    g._maybe_nudge_budget(2000)
    assert capsys.readouterr().out == ""  # nothing left to nudge


def test_no_budget_no_nudge(tmp_path, capsys):
    for budget in (None, 0):
        g = _gen(tmp_path, budget)
        g._maybe_nudge_budget(10_000)
        assert capsys.readouterr().out == ""


def test_governor_never_raises(tmp_path):
    # store_dir without a debug dir → diagnostic write is skipped, no raise.
    g = InstrumentedDataGenerator(
        inner=object(), store_dir=tmp_path / "nope",
        delegation_id="D001", eval_budget=100,
    )
    g._maybe_nudge_budget(200)  # must not raise
