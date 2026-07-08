"""Append one row of run telemetry to studies/run_ledger.csv.

Longitudinal record so we can measure whether changes make agentic runs better
or worse over time. Each row pins the commit SHA + run id and the headline
process/outcome metrics.

Usage:
    python studies/run_ledger.py <run_dir> [--commit <sha>]

<run_dir> is a study's runs/<timestamp>/ directory. --commit overrides the
detected HEAD (use it to backfill a row for a run that executed on an older
commit).
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "run_ledger.csv"
COLUMNS = [
    "commit", "study", "run_id", "outcome", "critic_consults", "delegations",
    "ledger_rows", "mean_wall_ms", "input_tokens", "output_tokens",
    "cost_usd", "time_used", "milestones_done", "milestones_skipped",
    "milestones_pending", "diagnostics",
]


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _solution_field(text: str, key: str) -> str:
    # matches "- key: value" or "| key | value |" style metadata lines
    m = re.search(rf"{re.escape(key)}\s*[:|]\s*([^\n|]+)", text)
    return m.group(1).strip() if m else ""


def extract(run_dir: Path) -> dict:
    debug = run_dir / "debug"
    study = run_dir.parent.parent.name  # studies/<study>/runs/<id>
    row = {c: "" for c in COLUMNS}
    row["study"] = study
    row["run_id"] = run_dir.name

    # outcome
    status_f = debug / "run_status.json"
    sol = run_dir.parent.parent / "solution.md"
    nb = run_dir.parent.parent / "pipeline.ipynb"
    if status_f.exists():
        try:
            row["outcome"] = json.loads(status_f.read_text()).get(
                "status", "halted")
        except Exception:
            row["outcome"] = "halted?"
    elif sol.exists() and (
        "FAILED RUN" in (_s := sol.read_text()) or "⛔" in _s
    ):
        # Distinct, loud terminal state: the deliverable never reproduced.
        row["outcome"] = "FAILED"
    elif sol.exists() and "UNGATED" in sol.read_text():
        row["outcome"] = "UNGATED"
    elif sol.exists():
        row["outcome"] = "GATED"
    elif nb.exists():
        # Notebook deliverable (no solution.md). The stamp records the TRUE gate
        # outcome (GATED/UNGATED/FAILED) — see agent_runtime stamping. A stamped
        # notebook without that field is a pre-fix run: fall back to GATED. No
        # stamp at all = the run never closed cleanly = FAILED.
        try:
            import nbformat as _nbf
            _ag = _nbf.read(str(nb), as_version=4).metadata.get("agentic", {})
            if not _ag.get("run"):
                row["outcome"] = "FAILED"
            else:
                row["outcome"] = _ag.get("gate_outcome", "GATED")
        except Exception:
            row["outcome"] = "FAILED"
    else:
        row["outcome"] = "no_solution"

    # critic consults
    cr = debug / "critic_reviews"
    row["critic_consults"] = len(list(cr.glob("*.md"))) if cr.exists() else 0

    # delegations + phases
    dlog = debug / "delegation_log.jsonl"
    if dlog.exists():
        recs = [json.loads(line) for line in dlog.read_text().splitlines()
                if line.strip()]
        row["delegations"] = len(recs)

    # ledger rows + mean wall_ms across the canonical store AND every design
    # namespace (Axis 3a). The canonical data dir is <run>/experiment_data/
    # experiment_data; each namespace store is a sibling <run>/experiment_data/
    # <ns> with its OWN /experiment_data data dir (different depths, so mirror
    # total_ledgered_evals's dir walk rather than a single glob).
    store_root = run_dir / "experiment_data"
    out_csvs = []
    _canon = store_root / "experiment_data" / "output.csv"
    if _canon.exists():
        out_csvs.append(_canon)
    if store_root.is_dir():
        for _sub in sorted(store_root.iterdir()):
            if _sub.is_dir() and _sub.name != "experiment_data":
                _f = _sub / "experiment_data" / "output.csv"
                if _f.exists():
                    out_csvs.append(_f)
    if out_csvs:
        rows = [r for f in out_csvs for r in csv.DictReader(f.open())]
        row["ledger_rows"] = len(rows)
        wm = [float(r["_wall_ms"]) for r in rows
              if r.get("_wall_ms") not in (None, "", "nan")
              and r.get("_source") != "precomputed_pool"]
        row["mean_wall_ms"] = round(sum(wm) / len(wm), 3) if wm else ""

    # tokens / cost / wall from solution.md or pipeline.ipynb metadata
    if sol.exists():
        t = sol.read_text()
        row["input_tokens"] = _solution_field(t, "input_tokens").replace(",", "")
        row["output_tokens"] = _solution_field(
            t, "output_tokens").replace(",", "")
        row["cost_usd"] = _solution_field(t, "estimated_cost").lstrip("$")
        row["time_used"] = _solution_field(t, "time_used")
    elif nb.exists():
        try:
            import nbformat as _nbf
            _nb = _nbf.read(str(nb), as_version=4)
            # Token usage cell is the last markdown cell appended by agent_runtime
            for _c in reversed(_nb.cells):
                if _c.cell_type == "markdown" and "## Token usage" in _c.source:
                    _t = _c.source
                    row["input_tokens"] = _solution_field(_t, "input_tokens").replace(",", "")
                    row["output_tokens"] = _solution_field(_t, "output_tokens").replace(",", "")
                    row["cost_usd"] = _solution_field(_t, "estimated_cost").lstrip("$")
                    row["time_used"] = _solution_field(_t, "time_used")
                    break
        except Exception:
            pass

    # milestones
    ms = debug / "strategizer_notes" / "milestones.json"
    if ms.exists():
        try:
            mvals = list(json.loads(ms.read_text()).values())
            for st in ("DONE", "SKIPPED", "PENDING"):
                row[f"milestones_{st.lower()}"] = sum(
                    1 for m in mvals if m.get("status") == st)
        except Exception:
            pass

    # diagnostics histogram
    diag = debug / "diagnostics.jsonl"
    if diag.exists():
        from collections import Counter
        rows = [json.loads(line) for line in diag.read_text().splitlines()
                if line.strip()]
        c = Counter(r.get("error_type") or r.get("type")
                    or r.get("intervention") or "other" for r in rows)
        row["diagnostics"] = json.dumps(dict(c))
    return row


def _delegation_proxies(run_dir: Path) -> list[dict]:
    """Per-delegation transcript PROXIES for possible trouble — 'look here'
    signals, never verdicts. Each is mechanical/semi-structured:

    - tool_errors: count of ``<tool_use_error>`` tags the SDK emits when a tool
      call fails (robust — not a free-text 'error' grep that matches code).
    - max_gap_s: the largest inter-event gap (a stall proxy).
    - errs: a few DISTINCT tool-error messages verbatim (the structured error
      text, like ERROR_RETURN — shows what tripped, not why the run failed).

    A high count or a long stall means READ that transcript; the proxy is
    allowed to over-fire (a false 'look here' costs one read) and says nothing
    about what actually happened — that judgement stays with the reader.
    """
    tdir = run_dir / "debug" / "transcripts"
    if not tdir.exists():
        return []
    out: list[dict] = []
    for f in sorted(tdir.glob("D*.jsonl")):
        errs: list[str] = []
        max_gap = 0.0
        try:
            for ln in f.read_text().splitlines():
                if not ln.strip():
                    continue
                r = json.loads(ln)
                g = r.get("gap_s") or 0
                max_gap = max(max_gap, float(g))
                if r.get("type") == "tool_result":
                    for item in (r.get("results") or []):
                        c = (item.get("content", "") if isinstance(item, dict)
                             else str(item))
                        if "<tool_use_error>" in c:
                            errs.append(c.replace("<tool_use_error>", "")
                                        .replace("</tool_use_error>", "").strip())
        except Exception:
            continue
        # distinct error messages, capped
        seen, distinct = set(), []
        for e in errs:
            key = e[:80]
            if key not in seen:
                seen.add(key)
                distinct.append(e)
        out.append({"id": f.stem, "tool_errors": len(errs),
                    "max_gap_s": round(max_gap, 1), "errs": distinct[:2]})
    # rank: most tool-errors first, then longest stall
    out.sort(key=lambda d: (-d["tool_errors"], -d["max_gap_s"]))
    return out


def analysis_brief(run_dir: Path) -> str:
    """Mechanical post-run digest for the CLAUDE.md run-analysis protocol.

    Offloads the MECHANICAL steps so the analyst reads ONE artifact instead of
    grepping several: the Step-5 KPI baseline (this run vs the previous ledger
    row for the same study), the Step-2 ScienceMonitor diagnostics tally, and
    the verbatim ERROR_RETURN events (the KPI whose target is 0).

    It deliberately does NOT interpret the prose artifacts — retrospectives,
    critic reviews, delegation transcripts. Failure modes live in prose that a
    keyword scan misses, so those are COUNTED and POINTED TO, never classified
    here. The retrospectives are surfaced verbatim (a relocation for one read,
    Step 1 / highest signal) — read and judge them; do not trust a proxy grep.
    """
    run_dir = Path(run_dir)
    debug = run_dir / "debug"
    row = extract(run_dir)

    # previous run of THIS study (read-only baseline; this run not yet appended)
    prev = None
    if LEDGER.exists():
        with LEDGER.open() as f:
            prior = [r for r in csv.DictReader(f)
                     if r.get("study") == row["study"]]
        if prior:
            prev = prior[-1]

    L: list[str] = []
    L.append(f"# Analysis brief — {row['study']}/{run_dir.name}")
    L.append("MECHANICAL facts below (read as given). Prose artifacts that need")
    L.append("JUDGEMENT are counted + pointed to, never scanned for proxies.")
    L.append("")
    L.append("## KPIs — this run vs previous (Step 5)")

    def _kv(k: str) -> str:
        cur = row.get(k, "")
        return f"- {k}: {cur}" + (f"   (prev {prev.get(k, '')})" if prev else "")

    for k in ("outcome", "critic_consults", "delegations", "ledger_rows",
              "mean_wall_ms", "time_used", "input_tokens", "output_tokens",
              "cost_usd", "milestones_done", "milestones_skipped",
              "milestones_pending"):
        L.append(_kv(k))
    L.append("  (critic_consults = gate attempts: 1 = clean, >2 = friction)")

    # Headline + hypothesis ledger — the science RESULT. Structured facts only:
    # min/max of the objective column (which end is 'good' is the analyst's call,
    # never asserted here), and each hypothesis' current status from its
    # status_log. A hypothesis still OPEN/INCONCLUSIVE at close is flagged as a
    # place to LOOK (a named failure mode), not classified as right or wrong.
    L.append("")
    L.append("## Headline & hypotheses (Step 5 — the science result)")
    oc = run_dir / "experiment_data" / "experiment_data" / "output.csv"
    if oc.exists():
        try:
            rws = list(csv.DictReader(oc.open()))
            outcols = [c for c in (rws[0] if rws else {})
                       if c and not c.startswith("_")]  # skip index + provenance
            if outcols and rws:
                obj = outcols[0]
                vals = [float(r[obj]) for r in rws
                        if r.get(obj) not in (None, "", "nan")]
                if vals:
                    L.append(f"- objective '{obj}': min={min(vals):.4g} "
                             f"max={max(vals):.4g} (n={len(vals)}; which end is "
                             "'good' is yours to judge)")
        except Exception:
            L.append("- (objective unreadable → open output.csv)")
    hf = debug / "strategizer_notes" / "hypotheses.json"
    if hf.exists():
        try:
            for hid, h in json.loads(hf.read_text()).items():
                slog = h.get("status_log") or []
                cur = slog[-1].get("status") if slog else "(no status)"
                open_at_close = cur in ("OPEN", "INCONCLUSIVE", None, "(no status)")
                L.append(f"- {hid}: {cur}"
                         + ("   ← still open at close (look here)"
                            if open_at_close else ""))
        except Exception:
            L.append("- (hypotheses.json unreadable → open it)")

    L.append("")
    L.append("## ScienceMonitor diagnostics tally (Step 2)")
    try:
        d = json.loads(row.get("diagnostics") or "{}")
    except Exception:
        d = {}
    L.extend([f"- {ev}: {n}" for ev, n in sorted(d.items(), key=lambda kv: -kv[1])]
             or ["- (none recorded)"])

    # verbatim ERROR_RETURN events (structured; Step-5 KPI — target is 0)
    errs: list[str] = []
    diagf = debug / "diagnostics.jsonl"
    if diagf.exists():
        for ln in diagf.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if "ERROR_RETURN" in (
                e.get("error_type"), e.get("type"), e.get("intervention")
            ):
                errs.append(ln)
    L.append("")
    L.append(f"## ERROR_RETURN events verbatim ({len(errs)}; target 0)")
    L.extend(errs or ["- (none)"])

    # prose artifacts — READ; do not trust a grep
    retro = debug / "retrospectives.jsonl"
    cr = debug / "critic_reviews"
    deleg = debug / "delegations"
    retro_lines = [l for l in (retro.read_text().splitlines()
                               if retro.exists() else []) if l.strip()]
    ncr = len(list(cr.glob("*.md"))) if cr.exists() else 0
    ndeleg = len([p for p in deleg.iterdir() if p.is_dir()]) if deleg.exists() else 0
    L.append("")
    L.append("## Prose artifacts — READ these (a scan cannot classify them)")
    L.append(f"- Step 1 retrospectives: {len(retro_lines)} entries → {retro} "
             "(read FIRST; verbatim below)")
    # verdict per call is a structured token — shows WHERE the gate bounced
    # (e.g. REJECT→PASS = one fix cycle) without summarising the critic's prose.
    verdicts: list[str] = []
    if cr.exists():
        for f in sorted(cr.glob("call_*.md")):
            m = re.search(r"verdict:\s*\**(PASS|REVISE|REJECT)",
                          f.read_text(), re.IGNORECASE)
            verdicts.append(f"{f.stem}={m.group(1).upper() if m else '?'}")
    seq = ", ".join(verdicts) if verdicts else "none"
    L.append(f"- Step 3 critic_reviews: {ncr} calls [{seq}] → {cr} "
             "(read in call order)")
    L.append(f"- Step 4 delegations: {ndeleg} → {deleg} "
             "(targeted only, on a hypothesis)")

    # Per-delegation transcript proxies — ranked 'look here', never verdicts.
    proxies = _delegation_proxies(run_dir)
    if proxies:
        L.append("")
        L.append("## Delegation transcript proxies (ranked; look here, not verdicts)")
        L.append("  (tool_errors = <tool_use_error> tags the SDK emitted; a high "
                 "count or long")
        L.append("   stall means READ that transcript — it does not say what went "
                 "wrong.)")
        for p in proxies:
            L.append(f"- {p['id']}: {p['tool_errors']} tool-errors, "
                     f"max stall {p['max_gap_s']}s → "
                     f"debug/transcripts/{p['id']}.jsonl")
            for e in p["errs"]:
                L.append(f"    e.g. {e[:120]}")

    if retro_lines:
        L.append("")
        L.append("## Retrospectives verbatim (Step 1 — judge, don't grep)")
        for ln in retro_lines:
            try:
                e = json.loads(ln)
            except Exception:
                L.append(ln)
                continue
            L.append(f"### {e.get('source_id', '?')} [{e.get('role', '?')}] "
                     f"flagged={e.get('flagged')}")
            L.append(str(e.get("text", "")).strip())
            L.append("")

    return "\n".join(L)


def main() -> None:
    args = sys.argv[1:]
    commit = None
    if "--commit" in args:
        i = args.index("--commit")
        commit = args[i + 1]
        args = args[:i] + args[i + 2:]
    run_dir = Path(args[0]).resolve()
    row = extract(run_dir)
    row["commit"] = commit or _git_short_sha()

    new = not LEDGER.exists()
    with LEDGER.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"appended row for {row['study']}/{row['run_id']} "
          f"(commit {row['commit']}, outcome {row['outcome']}) -> {LEDGER}")


if __name__ == "__main__":
    main()
