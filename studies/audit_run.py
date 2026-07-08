#!/usr/bin/env python3
"""Audit one agentic run — gather the evidence, don't render the verdict.

Consolidates the checks done by hand over and over when reviewing an
agentic_* run: liveness/hang, provenance (orphan ledger rows), delegations,
diagnostics/friction, solution.md health, pipeline.py faithfulness, agent
retrospectives, critic reviews (handbook use + optional pointer), and the
strategizer's per-turn tool-call sequence.

It FLAGS (with ⚠/✗/✓) and prints evidence; the human reads and judges. It never
decides PASS/FAIL — that is the point of having a person in the loop.

Usage
-----
    uv run python studies/audit_run.py <run_dir>        # a run/ or preserved dir
    uv run python studies/audit_run.py <study_dir>      # audits its latest run
    uv run python studies/audit_run.py                  # latest run of every study

A "run dir" is anything containing debug/ (and usually experiment_data/). Works
on a live run (flags a likely hang) and on a preserved copy.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

LEDGER = Path(__file__).parent / "run_ledger.csv"


# ── small helpers ─────────────────────────────────────────────────────────────
def _hr(title: str) -> None:
    print(f"\n{'═' * 78}\n{title}\n{'═' * 78}")


def _sub(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 73 - len(title)))


def _jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def _deliverable(run: Path, name: str) -> Path | None:
    """Find solution.md / pipeline.py. A live run keeps them in the STUDY dir
    (run.parent.parent); a preserved copy keeps them beside debug/."""
    for cand in (run / name, run.parent.parent / name):
        if cand.exists():
            return cand
    return None


def _output_csv(run: Path) -> Path | None:
    hits = glob.glob(str(run / "experiment_data" / "**" / "output.csv"),
                     recursive=True)
    return Path(hits[0]) if hits else None


def _collapse_last_wins(records: list[dict]) -> dict[str, dict]:
    """One record per delegation id, last write wins (matches DelegationLog)."""
    latest: dict[str, dict] = {}
    for r in records:
        rid = r.get("id")
        if rid is not None:
            latest[rid] = r
    return latest


# ── sections ──────────────────────────────────────────────────────────────────
def liveness(run: Path) -> None:
    _sub("LIVENESS / HANG")
    log = run / "debug" / "run.log"
    started = invoked = completed = False
    if log.exists():
        txt = log.read_text(errors="replace")
        started = "Run starting" in txt
        invoked = "Invoking graph" in txt
        completed = "Run complete" in txt
        print(txt.strip() or "(empty run.log)")
    else:
        print("(no run.log)")
    # newest written file vs now — a frozen run with idle mtime is a likely hang
    files = [Path(p) for p in glob.glob(str(run / "debug" / "**" / "*"),
                                        recursive=True) if Path(p).is_file()]
    if files:
        newest = max(files, key=lambda p: p.stat().st_mtime)
        age = time.time() - newest.stat().st_mtime
        print(f"\nnewest debug write: {newest.name} ({age/60:.1f} min ago)")
        if started and not invoked:
            print("  ⚠ reached 'Run starting' but NOT 'Invoking graph' — "
                  "stalled in startup (e.g. the problem-statement review).")
        if not completed and age > 20 * 60:
            print(f"  ⚠ no write for {age/60:.1f} min and not complete — "
                  "likely HUNG (check the process).")
        if completed:
            print("  ✓ run completed.")


def ledger_row(run: Path) -> None:
    _sub("LEDGER ROW (run_ledger.csv)")
    run_id = run.name
    if not LEDGER.exists():
        print("(no run_ledger.csv)")
        return
    rows = list(csv.DictReader(LEDGER.open()))
    match = [r for r in rows if r.get("run_id") == run_id]
    if not match:
        print(f"(no ledger row for run_id={run_id} — may be preserved/renamed "
              "or still running)")
        return
    r = match[-1]
    for k in ("outcome", "critic_consults", "delegations", "ledger_rows",
              "cost_usd", "time_used", "diagnostics"):
        print(f"  {k:16}: {r.get(k)}")


def provenance(run: Path) -> None:
    _sub("PROVENANCE (orphan ledger rows?)")
    out = _output_csv(run)
    if out is None:
        print("(no canonical output.csv)")
        return
    rows = list(csv.DictReader(out.open()))
    stamps = Counter(r.get("_delegation_id", "?") for r in rows)
    obj_cols = [c for c in (rows[0].keys() if rows else [])
                if c and not c.startswith("_")]
    print(f"  ledger rows: {len(rows)} | per-delegation: {dict(stamps)}")
    print(f"  objective column(s): {obj_cols}")
    for c in obj_cols:
        vals = [float(r[c]) for r in rows if r.get(c) not in (None, "")]
        if vals:
            print(f"    {c}: min={min(vals):.6g} max={max(vals):.6g} "
                  f"n={len(vals)}")
    logged = _collapse_last_wins(_jsonl(run / "debug" / "delegation_log.jsonl"))
    orphans = [d for d in stamps if d not in logged and d != "?"]
    if orphans:
        print(f"  ✗ ORPHANS (stamped, NOT logged): {orphans}")
    else:
        print("  ✓ every stamped delegation is logged (no orphans)")


def delegations(run: Path) -> None:
    _sub("DELEGATIONS (delegation_log, last-wins)")
    logged = _collapse_last_wins(_jsonl(run / "debug" / "delegation_log.jsonl"))
    if not logged:
        print("(no delegation_log)")
        return
    for did, r in sorted(logged.items()):
        print(f"  {did} [{r.get('status'):8}] to={r.get('to_node'):20} "
              f"evals={r.get('evals')} phase={r.get('phase')}")


def diagnostics(run: Path) -> None:
    _sub("DIAGNOSTICS / FRICTION CODES")
    recs = _jsonl(run / "debug" / "diagnostics.jsonl")
    if not recs:
        print("(none)")
        return
    c = Counter(r.get("error_type") or r.get("rule") or r.get("type") or "?"
                for r in recs)
    print("  " + (", ".join(f"{k}:{v}" for k, v in c.most_common()) or "(none)"))


def solution_md(run: Path) -> None:
    _sub("solution.md")
    sol = _deliverable(run, "solution.md")
    if sol is None:
        print("(no solution.md)")
        return
    txt = sol.read_text(errors="replace")
    # headline vs ledger min
    out = _output_csv(run)
    nums = [float(x) for x in re.findall(r"-?\d+\.\d{3,}", txt)]
    if out:
        rows = list(csv.DictReader(out.open()))
        objs = [c for c in (rows[0].keys() if rows else []) if c and not c.startswith("_")]
        if objs and rows:
            vals = [float(r[objs[0]]) for r in rows if r.get(objs[0]) not in (None, "")]
            if vals:
                mn = min(vals)
                hit = any(abs(n - mn) < 1e-3 for n in nums)
                print(f"  ledger min({objs[0]}) = {mn:.6g} | "
                      f"{'✓ appears in solution.md' if hit else '⚠ ledger min NOT found in prose (headline mismatch?)'}")
    # navigability heuristics (the 'where to look / what to do' bar)
    coords = bool(re.search(r"\[?\s*[-+]?\d+\.\d+\s*,\s*[-+]?\d+\.\d+", txt))
    print(f"  states best-point coordinates: {'✓' if coords else '⚠ no'}")
    for kw, label in [("basin", "landscape characterization"),
                      ("INCONCLUSIVE|FALSIFIED|SUPPORTED|OPEN", "hypothesis verdicts"),
                      ("future work|next step", "next steps")]:
        print(f"  mentions {label}: {'✓' if re.search(kw, txt, re.I) else '⚠ no'}")
    print(f"  length: {len(txt.splitlines())} lines")


def pipeline_py(run: Path) -> None:
    _sub("pipeline.py (valid + faithful to the method)")
    p = _deliverable(run, "pipeline.py")
    if p is None:
        print("(no pipeline.py)")
        return
    src = p.read_text(errors="replace")
    checks = {
        "calls get_evaluator() (real oracle step)": "get_evaluator(" in src,
        "composable Pipeline/Step": bool(re.search(r"\bPipeline\(|\bStep\(", src)),
        "load-or-create (from_file + sampler)": ("from_file" in src and
                                                 "sampler" in src.lower()),
        "derives headline (idxmin/min)": bool(re.search(r"idxmin|\.min\(", src)),
        "real surrogate/optimizer (BO/GP/minimize)":
            bool(re.search(r"GaussianProcess|ExpectedImprovement|scipy.optimize|minimize\(|BayesOpt|acquisition", src, re.I)),
        "⚠ fake API add_continuous_* (Domain has add_float)":
            "add_continuous" in src,
        "⚠ stub marker (# in production / TODO / pass-only phase)":
            bool(re.search(r"in production this would|# TODO|would run here", src, re.I)),
    }
    for label, ok in checks.items():
        mark = "✓" if (ok and not label.startswith("⚠")) else (
            "✗" if (not ok and not label.startswith("⚠")) else
            ("✗" if ok else "✓"))
        print(f"  {mark} {label}: {ok}")
    print(f"  length: {len(src.splitlines())} lines")


def retrospectives(run: Path) -> None:
    _sub("RETROSPECTIVES (reflection/interview)")
    recs = _jsonl(run / "debug" / "retrospectives.jsonl")
    if not recs:
        print("(none)")
        return
    for d in recs:
        who = d.get("agent") or d.get("role") or "?"
        txt = d.get("text") or ""
        print(f"\n  ### [{who}]")
        for key in ("CONSISTENCY", "FRICTION", "BLOCKED", "BLOCKER"):
            m = re.search(r"\*\*" + key + r".*?(?=\*\*[A-Z]|\Z)", txt, re.S)
            if m:
                line = " ".join(m.group(0).split())
                flag = " ⚠" if ("flagged" in line.lower() or
                                key.startswith("BLOCK")) else ""
                print(f"    {line[:300]}{flag}")


def critic_reviews(run: Path) -> None:
    _sub("CRITIC REVIEWS (verdict + handbook use + optional pointer)")
    rev_dir = run / "debug" / "critic_reviews"
    files = sorted(rev_dir.glob("*.md")) if rev_dir.exists() else []
    if not files:
        print("(no critic reviews)")
        return
    for f in files:
        txt = f.read_text(errors="replace")
        verdict = re.search(r"verdict:\s*(\w+)", txt)
        used_hb = bool(re.search(r"consult.*handbook|ConsultHandbook", txt, re.I))
        pointer = "Handbook pointer" in txt
        print(f"  {f.name}: verdict={verdict.group(1) if verdict else '?'} "
              f"| handbook-consulted={'✓' if used_hb else '—'} "
              f"| optional-pointer={'✓' if pointer else '—'}")


def transcript_toolcalls(run: Path) -> None:
    _sub("STRATEGIZER TOOL-CALLS (per turn)")
    tdir = run / "debug" / "transcripts" / "strategizer"
    turns = sorted(tdir.glob("turn_*.jsonl")) if tdir.exists() else []
    if not turns:
        print("(no strategizer transcripts)")
        return
    for tf in turns:
        seq = []
        for d in _jsonl(tf):
            if d.get("type") == "assistant" and d.get("tools"):
                for t in d["tools"]:
                    seq.append(t.get("name", "").replace("mcp__f3dasm_agent_tools__", ""))
        c = Counter(seq)
        errs = len(re.findall(r"ERROR[: ]", tf.read_text(errors="replace")))
        print(f"  {tf.name}: {len(seq)} calls | {dict(c)} | "
              f"ERROR-substrings≈{errs}")


def audit(run: Path) -> None:
    _hr(f"AUDIT: {run}")
    for fn in (liveness, ledger_row, provenance, delegations, diagnostics,
               solution_md, pipeline_py, retrospectives, critic_reviews,
               transcript_toolcalls):
        try:
            fn(run)
        except Exception as exc:  # noqa: BLE001 — a tool, never crash mid-audit
            print(f"  (section {fn.__name__} errored: {exc})")


def _resolve(target: str | None) -> list[Path]:
    base = Path(__file__).parent
    if target:
        p = Path(target)
        if (p / "debug").exists():
            return [p]
        runs = sorted((p / "runs").glob("*/"), key=lambda d: d.stat().st_mtime)
        return [runs[-1]] if runs else []
    # no arg: latest run of every agentic_* study
    out = []
    for study in sorted(base.glob("agentic_*")):
        runs = sorted((study / "runs").glob("*/"), key=lambda d: d.stat().st_mtime)
        if runs:
            out.append(runs[-1])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target", nargs="?", help="run dir, study dir, or omit for latest-of-each")
    runs = _resolve(ap.parse_args(argv).target)
    if not runs:
        print("no run dir found", file=sys.stderr)
        return 1
    for r in runs:
        audit(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
