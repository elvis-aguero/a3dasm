"""Generate the prompt & gate provenance map.

Answers, for every role in the default graph, the only two questions a
reviewer of an agentic system actually asks:

  1. What EXACTLY does this agent see?   (the assembled system prompt,
     split into its real sections, each one traced to ``file:line``)
  2. What can STOP it?                   (every gate, hard or soft, its
     source, its escape hatch, its kill switch)

Everything here is READ OFF THE SYSTEM, never hand-typed:

* prompt text comes from the live ``Graph``/``Agent`` objects built by
  ``agents/_graphs.py`` — the same objects ``agent_runtime.py`` runs;
* section boundaries come from the prompts' own ``<tag>`` structure, not
  from an invented taxonomy layered on top;
* ``file:line`` for every section is resolved by locating that exact text
  in the source tree, so a moved prompt moves its citation with it;
* the Done() gate chain is parsed out of the literal tuple in
  ``feedback.py`` — its ORDER is the source's order, not a guess;
* every gate's summary is its own live docstring.

The one hand-maintained thing is ``GATES``: which symbols count as gates.
Each entry is RESOLVED against the AST at build time, so a renamed or
deleted gate fails the build loudly instead of leaving a stale card in a
map someone is about to present. That is the whole anti-drift contract —
the same one ``runtime/run_diagram.py`` adopted after this repo's
hand-authored class diagram rotted.

Usage::

    python internal/tools/promptmap.py                 # -> internal/promptmap.html
    python internal/tools/promptmap.py --json-only     # -> stdout
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
PKG = SRC / "a3dasm" / "_src"
TEMPLATE = Path(__file__).with_name("promptmap_template.html")
DEFAULT_OUT = REPO / "internal" / "promptmap.html"


# --------------------------------------------------------------------------
# source location
# --------------------------------------------------------------------------

_SOURCE_CACHE: dict[Path, str] = {}


def _read(path: Path) -> str:
    if path not in _SOURCE_CACHE:
        _SOURCE_CACHE[path] = path.read_text(encoding="utf-8")
    return _SOURCE_CACHE[path]


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def _py_files() -> list[Path]:
    return sorted(PKG.rglob("*.py"))


def locate(snippet: str, prefer: list[Path] | None = None) -> dict | None:
    """Find *snippet* verbatim in the source tree; return file + line span.

    Prompts are triple-quoted literals, so the assembled text appears
    byte-for-byte in the file that defines it — an exact search is both the
    simplest and the most honest citation available (no offset arithmetic
    over an AST node whose literal may have been concatenated). Falls back
    to matching the snippet's distinctive head when the tail was produced by
    concatenation across two literals.
    """
    snippet = snippet.strip("\n")
    if not snippet:
        return None
    candidates = list(prefer or []) + [p for p in _py_files() if p not in (prefer or [])]

    # Longest single line, as a last resort: a prompt built by concatenating
    # "...\n" fragments (notebook_deliverable_spec) never matches as a block,
    # but each of its LINES is still a verbatim literal in the source. So is a
    # section whose text contains an escape the source spells as two characters
    # (``\n`` inside <on_error>), which defeats a whole-block search.
    longest = max(
        (ln for ln in snippet.splitlines() if "\\" not in ln),
        key=len, default="",
    ).strip()

    probes = [(snippet, "exact"), (snippet[:240], "head")]
    if len(longest) >= 30:
        probes.append((longest, "line"))

    for probe, how in probes:
        if len(probe) < 24:
            continue
        for path in candidates:
            src = _read(path)
            idx = src.find(probe)
            if idx == -1:
                continue
            start = _line_of(src, idx)
            if how == "exact":
                end = start + snippet.count("\n")
            elif how == "head":
                end = start + probe.count("\n")
            else:
                end = start
            return {
                "file": _rel(path),
                "line": start,
                "line_end": end,
                "match": how,
            }
    return None


def resolve_symbol(module_rel: str, qualname: str) -> dict:
    """Resolve ``module_rel::qualname`` to file, line and live docstring.

    Raises loudly when the symbol is gone — the anti-drift contract.
    """
    path = PKG / module_rel
    if not path.exists():
        raise SystemExit(f"promptmap: module vanished: {module_rel}")
    tree = ast.parse(_read(path))
    parts = qualname.split(".")

    def walk(body, remaining):
        head, rest = remaining[0], remaining[1:]
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name != head:
                    continue
                if not rest:
                    return node
                return walk(node.body, rest)
        return None

    node = walk(tree.body, parts)
    if node is None:
        raise SystemExit(
            f"promptmap: symbol vanished: {module_rel}::{qualname}\n"
            "  The gate registry in internal/tools/promptmap.py is stale. Fix it\n"
            "  rather than presenting a map that cites code that no longer exists."
        )
    return {
        "file": _rel(path),
        "line": node.lineno,
        "doc": ast.get_docstring(node) or "",
        "signature": qualname,
    }


# --------------------------------------------------------------------------
# the Done() close sequence, parsed out of the source's own tuple
# --------------------------------------------------------------------------

def done_chain() -> list[str]:
    """The gate callables Done() iterates, in the order the source lists them."""
    path = PKG / "nodes" / "tools" / "routing" / "feedback.py"
    tree = ast.parse(_read(path))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "Done"):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.For) and isinstance(sub.iter, ast.Tuple):
                names = []
                for elt in sub.iter.elts:
                    if isinstance(elt, ast.Attribute):
                        names.append(elt.attr)
                if names:
                    return names
    raise SystemExit("promptmap: could not parse Done()'s gate tuple in feedback.py")


# --------------------------------------------------------------------------
# prompt assembly, mirroring agent_runtime.py
# --------------------------------------------------------------------------

#: Placeholder values for the run-scoped paths the runtime substitutes. Marked
#: as substituted in the UI so nobody mistakes them for literal prompt text.
_PATH_STUB = {
    "study_dir": "<study_dir>",
    "run_dir": "<study_dir>/runs/<run_id>",
    "debug_dir": "<study_dir>/runs/<run_id>/debug",
    "notes_dir": "<study_dir>/runs/<run_id>/debug/strategizer_notes",
    "experiment_data_dir": "<study_dir>/runs/<run_id>/experiment_data",
    "workspace_dir": "<study_dir>/runs/<run_id>/debug/delegations",
    "resources": "<resource stanza — resolved per run from the host/SLURM allocation>",
    "knowledge": "<handbook menu — resolved per role from knowledge/kb.py>",
}

_TAG_RE = re.compile(r"(?m)^<([a-z_0-9]+)>\n(.*?)\n</\1>", re.DOTALL)


def split_sections(prompt: str, prefer: list[Path]) -> list[dict]:
    """Split an assembled prompt at its own top-level ``<tag>`` boundaries.

    The tags are the prompt's real structure (they are what the model sees as
    section headers), so this is a reading of the document, not a taxonomy
    invented for the map. Text outside any tag is kept as an untagged section
    rather than dropped — the map must account for every character.
    """
    sections: list[dict] = []
    cursor = 0

    def emit(tag: str | None, body: str, start: int):
        body = body.strip("\n")
        if not body:
            return
        src = locate(body, prefer)
        if src is None and tag:
            src = locate(f"<{tag}>\n", prefer)
            if src:
                src["match"] = "tag"
        sections.append({
            "tag": tag,
            "chars": len(body),
            "text": body,
            "source": src,
        })

    for m in _TAG_RE.finditer(prompt):
        if m.start() > cursor:
            emit(None, prompt[cursor:m.start()], cursor)
        emit(m.group(1), m.group(2), m.start())
        cursor = m.end()
    if cursor < len(prompt):
        emit(None, prompt[cursor:], cursor)
    return sections


#: Text blocks that are DEFINED elsewhere and injected verbatim into a prompt.
#: Detected by substring containment so the citation points at the definition
#: (knowledge/charter.py) rather than the consuming agent module.
def shared_blocks() -> list[dict]:
    out = []
    for mod, name in (("knowledge/charter.py", "FALSIFICATION_CHARTER"),
                      ("knowledge/idioms.py", "F3DASM_CORE_IDIOMS")):
        path = PKG / mod
        tree = ast.parse(_read(path))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets
            ):
                value = ast.literal_eval(node.value)
                out.append({
                    "name": name,
                    "file": _rel(path),
                    "line": node.lineno,
                    "text": value,
                    "chars": len(value),
                })
    return out


def build_roles(shared: list[dict]) -> list[dict]:
    from a3dasm._src.agents import _graphs
    from a3dasm._src.evaluation.notebook_exec import notebook_deliverable_spec
    from a3dasm._src.prompts.agent_prompts import (
        RUN_PATHS_PREAMBLE_TEMPLATE,
        WORKSPACE_PREAMBLE_TEMPLATE,
    )

    graph = _graphs._default_graph()
    entry = graph.entry
    out_edges: dict[str, list[str]] = {}
    for edge in graph.edges:
        out_edges.setdefault(edge.source, []).append(edge.target)

    preamble_src = {
        "RUN_PATHS_PREAMBLE_TEMPLATE": locate(RUN_PATHS_PREAMBLE_TEMPLATE[:240]),
        "WORKSPACE_PREAMBLE_TEMPLATE": locate(WORKSPACE_PREAMBLE_TEMPLATE[:240]),
    }

    roles = []
    for name, agent in graph.nodes.items():
        module = sys.modules[type(agent).__module__]
        mod_path = Path(module.__file__).resolve()
        is_entry = name == entry

        layers = []

        # Layer 1 — the run-scoped preamble prepended in agent_runtime.py.
        tpl = RUN_PATHS_PREAMBLE_TEMPLATE if is_entry else WORKSPACE_PREAMBLE_TEMPLATE
        tpl_name = "RUN_PATHS_PREAMBLE_TEMPLATE" if is_entry else "WORKSPACE_PREAMBLE_TEMPLATE"
        rendered = tpl.format(**{
            k: v for k, v in _PATH_STUB.items()
            if "{" + k + "}" in tpl
        })
        layers.append({
            "kind": "preamble",
            "label": tpl_name,
            "note": "Prepended by the runtime. Braced values are substituted per run; "
                    "shown here as placeholders.",
            "assembled_at": locate("preamble = " + tpl_name),
            "definition": preamble_src[tpl_name],
            "chars": len(rendered),
            "sections": [{
                "tag": None, "chars": len(rendered), "text": rendered,
                "source": preamble_src[tpl_name], "substituted": True,
            }],
        })

        # Layer 2 — the agent's own system prompt.
        sp = agent.system_prompt or ""
        sections = split_sections(sp, prefer=[mod_path])
        for sec in sections:
            for block in shared:
                if block["text"].strip("\n") in sec["text"]:
                    sec["injects"] = {
                        "name": block["name"],
                        "file": block["file"],
                        "line": block["line"],
                        "chars": block["chars"],
                    }
        layers.append({
            "kind": "system",
            "label": f"{type(agent).__name__}.system_prompt",
            "note": "The role's own instructions, inlined in its agent module.",
            "definition": {"file": _rel(mod_path), "line": _class_line(mod_path, type(agent).__name__)},
            "chars": len(sp),
            "sections": sections,
        })

        # Layer 3 — the notebook deliverable contract, role-aware and settings-gated.
        role = getattr(agent, "role", None)
        if role in ("strategizer", "implementer", "critic"):
            spec = notebook_deliverable_spec(role)
            layers.append({
                "kind": "deliverable",
                "label": f"notebook_deliverable_spec({role!r})",
                "note": "Appended only for these three roles, and only while the "
                        "`pipeline_deliverable` knob is true.",
                "switch": "pipeline_deliverable",
                "definition": resolve_symbol("evaluation/notebook_exec.py",
                                             "notebook_deliverable_spec"),
                "assembled_at": locate("system_prompt = system_prompt + notebook_deliverable_spec"),
                "chars": len(spec),
                "sections": [{"tag": "deliverable_format", "chars": len(spec),
                              "text": spec, "generated": True,
                              "source": locate(spec, [PKG / "evaluation" / "notebook_exec.py"])}],
            })

        # Layer 4 — the <tools> catalog, generated from the live closure set.
        tools = [getattr(t, "__name__", str(t)) for t in (getattr(agent, "tools", ()) or ())]
        layers.append({
            "kind": "catalog",
            "label": "<tools> catalog",
            "note": "Rendered at dispatch from the LIVE closure dict, so it can never "
                    "name a tool the agent does not have. Each entry is the tool's own "
                    "docstring. The closures are bound per run (nodes/tools/routing/), "
                    "so the exact text is run-scoped — the tool names below are the "
                    "declared set.",
            "definition": resolve_symbol("prompts/tool_catalog.py", "render_tool_catalog"),
            "assembled_at": locate("system_prompt=system_prompt_with_catalog("),
            "chars": None,
            "tools": sorted(tools),
            "sections": [],
        })

        roles.append({
            "id": name,
            "class": type(agent).__name__,
            "role": role,
            "entry": is_entry,
            "module": _rel(mod_path),
            "delegates_to": sorted(out_edges.get(name, [])),
            "tools": sorted(tools),
            "tool_count": len(tools),
            "layers": layers,
            "static_chars": len(sp) + sum(
                lay["chars"] for lay in layers
                if lay["kind"] != "system" and lay["chars"]
            ),
        })
    return roles


def _class_line(path: Path, cls: str) -> int:
    tree = ast.parse(_read(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return node.lineno
    return 1


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------

#: The hand-maintained part: WHICH symbols are gates. Everything about each
#: one (file, line, wording) is resolved from source at build time.
GATES: list[dict] = [
    dict(id="pending", title="Delegations still in flight", kind="soft",
         phase="Done()", module="nodes/tools/routing/feedback.py",
         symbol="FeedbackTools._pending_refusal",
         effect="Done() returns a nudge, not an error; the run stays open."),
    dict(id="retrospective", title="Exit interview", kind="procedural",
         phase="Done()", module="nodes/tools/routing/feedback.py",
         symbol="FeedbackTools._capture_retrospective",
         effect="Holds the close for one turn to capture the retrospective."),
    dict(id="milestones", title="Milestone backlog", kind="hard",
         phase="Done() + pre-delegation", module="nodes/tools/routing/feedback.py",
         symbol="FeedbackTools._milestone_gate", switch="milestones_enabled",
         escape="MilestoneSkip(reason)",
         effect="Blocks the close while process milestones are open."),
    dict(id="milestone_block", title="Implementer delegation block", kind="hard",
         phase="Delegate()", module="nodes/tools/routing/delegation.py",
         symbol="DelegationTools._milestone_gate", switch="milestones_enabled",
         escape="MilestoneSkip(reason)",
         policy=("epistemics/milestones.py", "implementer_block"),
         effect="No delegation to the implementer until the backlog is resolved."),
    dict(id="falsification_checkpoint", title="Falsification checkpoint", kind="soft",
         phase="Delegate()", module="nodes/tools/routing/delegation.py",
         symbol="DelegationTools._falsification_checkpoint",
         effect="Prompts for a falsification attempt when the ledger has gone one-sided."),
    dict(id="verdict_advisory", title="Verdict advisory note", kind="soft",
         phase="HypothesisUpdate()", module="nodes/tools/routing/ledger.py",
         symbol="LedgerTools._verdict_advisory", switch="F3DASM_VERDICT_VALIDATOR",
         effect="The referee's ruling is returned as advice attached to the update."),
    dict(id="budget", title="Evaluation budget broadcast", kind="soft",
         phase="continuous", module="nodes/tools/routing/delegation.py",
         symbol="DelegationTools._budget_broadcast",
         effect="Budget pressure reaches the agent as an in-band notice; never blocks."),
    dict(id="first_call", title="Two-shot close", kind="procedural",
         phase="Done()", module="nodes/tools/routing/feedback.py",
         symbol="FeedbackTools._first_call_warning",
         effect="The first Done() warns and lists unmet conditions; only the second closes."),
    dict(id="reproduction", title="Reproduction gate", kind="hard",
         phase="Done() + CheckDeliverable()", module="nodes/reproduction_gate.py",
         symbol="ReproductionGateMixin._reproduction_gate",
         effect="pipeline.ipynb must execute cleanly, add zero oracle rows and rewrite none."),
    dict(id="must_reproduce", title="Pre-critic reproduction bounce", kind="hard",
         phase="Done()", module="nodes/tools/routing/feedback.py",
         symbol="FeedbackTools._must_reproduce",
         effect="A non-reproducing deliverable bounces back before a critic turn is spent. Bounded."),
    dict(id="deliverables", title="Required deliverables present", kind="hard",
         phase="Done() + CheckDeliverable()", module="nodes/reproduction_gate.py",
         symbol="ReproductionGateMixin._missing_deliverables",
         effect="Names any required deliverable that is absent."),
    dict(id="headline", title="Headline consistency", kind="soft",
         phase="reproduction gate", module="nodes/reproduction_gate.py",
         symbol="_headline_consistency",
         effect="Stated answer must match the computed one — lenient: silent if either marker is absent."),
    dict(id="critic", title="Adversarial critic gate", kind="hard",
         phase="Done()", module="nodes/tools/routing/feedback.py",
         symbol="FeedbackTools._critic_gate",
         effect="A run closes only on a critic PASS."),
    dict(id="verdict", title="Live verdict validator", kind="soft",
         phase="HypothesisUpdate()", module="epistemics/verdict_validator.py",
         symbol="build_judge_prompt", switch="F3DASM_VERDICT_VALIDATOR",
         effect="An independent referee judges a closing verdict against the same charter."),
    dict(id="supported", title="SUPPORTED needs an attempt", kind="hard",
         phase="HypothesisUpdate()", module="nodes/tools/routing/ledger.py",
         symbol="LedgerTools._supported_needs_attempt",
         effect="Refused synchronously at the data boundary."),
    dict(id="cited", title="Cited delegation must exist", kind="hard",
         phase="HypothesisUpdate()", module="nodes/tools/routing/ledger.py",
         symbol="LedgerTools._check_cited_delegation",
         effect="Evidence must name a delegation that actually ran."),
    dict(id="links", title="Delegation hypothesis links", kind="hard",
         phase="Delegate()", module="nodes/tools/routing/delegation.py",
         symbol="DelegationTools._check_hypothesis_links",
         effect="A delegation must be anchored to hypotheses that exist."),
    dict(id="unledgered", title="Unledgered evaluations", kind="soft",
         phase="continuous", module="epistemics/science_monitor.py",
         symbol="ScienceMonitor._check_unledgered",
         effect="Injected as an in-band notice; never blocks."),
    dict(id="duplicate", title="Duplicate evaluations", kind="soft",
         phase="continuous", module="epistemics/science_monitor.py",
         symbol="ScienceMonitor._check_duplicate_evaluations",
         effect="Injected as an in-band notice; never blocks."),
    dict(id="unstamped", title="Unstamped rows", kind="soft",
         phase="continuous", module="epistemics/science_monitor.py",
         symbol="ScienceMonitor._check_unstamped_rows",
         effect="Injected as an in-band notice; never blocks."),
    dict(id="reviewer", title="Problem-statement review", kind="advisory",
         phase="pre-run", module="epistemics/reviewer.py",
         symbol="review_gaps",
         effect="Always writes a report; never blocks a run."),
    dict(id="report_shape", title="Report shape retry", kind="soft",
         phase="worker return", module="nodes/parsing.py",
         symbol="_classify_response",
         effect="A malformed worker report is diagnosed and retried."),
]


def build_gates() -> list[dict]:
    chain = done_chain()
    out = []
    for spec in GATES:
        resolved = resolve_symbol(spec["module"], spec["symbol"])
        entry = dict(spec)
        entry.update(resolved)
        leaf = spec["symbol"].rsplit(".", 1)[-1]
        # Two different mixins define a ``_milestone_gate``; only the one in
        # feedback.py is a step of Done()'s own sequence.
        in_done = spec["module"].endswith("routing/feedback.py") and leaf in chain
        entry["done_order"] = chain.index(leaf) + 1 if in_done else None
        out.append(entry)
    out.sort(key=lambda g: (g["done_order"] is None, g["done_order"] or 0, g["title"]))
    return out


# --------------------------------------------------------------------------

def build() -> dict:
    shared = shared_blocks()
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        sha = "unknown"
    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": sha,
        "roles": build_roles(shared),
        "gates": build_gates(),
        "shared": [{k: v for k, v in b.items() if k != "text"} | {"text": b["text"]}
                   for b in shared],
        "done_chain": done_chain(),
        "switches": _switches(),
    }


def _switches() -> list[dict]:
    """Every knob a reviewer can point at, read from settings.KNOWN_KEYS plus
    the env-only kill switches actually consulted in the source."""
    from a3dasm._src.runtime import settings
    path = PKG / "runtime" / "settings.py"
    keys = [{"key": k, "kind": "config.yaml runtime:", "env": f"F3DASM_{k.upper()}",
             "file": _rel(path)} for k in sorted(settings.KNOWN_KEYS)]
    env_only: dict[str, list[str]] = {}
    for p in _py_files():
        for m in re.finditer(r'os\.environ\.get\(\s*"(F3DASM_[A-Z0-9_]+)"', _read(p)):
            name = m.group(1)
            if name == "F3DASM_" or name[len("F3DASM_"):].lower() in settings.KNOWN_KEYS:
                continue
            env_only.setdefault(name, []).append(
                f"{_rel(p)}:{_line_of(_read(p), m.start())}")
    keys += [{"key": n, "kind": "environment only", "env": n,
              "file": sites[0].rsplit(":", 1)[0], "sites": sorted(set(sites))}
             for n, sites in sorted(env_only.items())]
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    data = build()
    if args.json_only:
        json.dump(data, sys.stdout, indent=2)
        return

    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__PROMPTMAP_DATA__*/null",
        json.dumps(data, ensure_ascii=False),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    total = sum(r["static_chars"] for r in data["roles"])
    print(f"promptmap: {len(data['roles'])} roles, {len(data['gates'])} gates, "
          f"{total:,} chars of static prompt -> {args.out}")


if __name__ == "__main__":
    main()
