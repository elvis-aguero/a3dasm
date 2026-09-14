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
    """Find *snippet* in the source tree; return the truest span it can prove.

    Three probes, in descending order of what each may honestly claim:

    ``exact``
        The whole block is one verbatim run of characters in one file.
        ``line_end`` is its real end and ``span`` is true.
    ``lines``
        The block is a literal the source breaks across lines — a ``\\``
        continuation, or two literals concatenated — so no single search
        finds it, but its individual lines are still verbatim. The span runs
        from the first to the last of those lines, matched in order.
        ``span`` is true; the range is real, though the source lines between
        its ends may include the joins the prompt itself does not show.
    ``line``
        Only one distinctive line could be found. That is an ANCHOR, not a
        span: ``line_end`` is absent and ``span`` is false.

    A probe never reports a range it did not verify. The first version of
    this function searched a fixed 240-character head and then reported that
    head's line count as the block's end, which understated every template
    longer than seven lines while still labelling the citation ``exact``.
    """
    snippet = snippet.strip("\n")
    if not snippet:
        return None
    prefer = list(prefer or [])
    candidates = prefer + [p for p in _py_files() if p not in prefer]

    # A short snippet (a bare ``<tag>`` line) is not distinctive enough to
    # search the whole tree with — it may only be trusted inside the module
    # the caller already knows the prompt lives in.
    exact_in = candidates if len(snippet) >= 24 else prefer
    for path in exact_in:
        src = _read(path)
        idx = src.find(snippet)
        if idx != -1:
            start = _line_of(src, idx)
            return {
                "file": _rel(path), "line": start,
                "line_end": start + snippet.count("\n"),
                "match": "exact", "span": True,
            }

    # Lines free of the escapes a source literal spells as two characters
    # (``\n`` inside <on_error>). A line must be distinctive enough not to
    # match by luck: 24 characters across the tree, but only 8 inside a file
    # the caller already named, where a short line like ``<workspace>`` is
    # the block's real first line and dropping it would understate the span.
    lines_ = [ln.strip() for ln in snippet.splitlines()]
    lines_ = [ln for ln in lines_ if ln and "\\" not in ln]
    if not lines_:
        return None

    for path in candidates:
        src = _read(path)
        floor = 8 if path in prefer else 24
        clean = [ln for ln in lines_ if len(ln) >= floor]
        if not clean:
            continue
        hits: list[int] = []
        cursor = 0
        for ln in clean:
            idx = src.find(ln, cursor)
            if idx == -1:
                continue
            hits.append(_line_of(src, idx))
            cursor = idx + len(ln)
        # Half the lines, in order, is enough to call it this block and not a
        # coincidence; fewer and the file is more likely to merely share a
        # sentence with it.
        if len(hits) >= 2 and len(hits) * 2 >= len(clean):
            return {
                "file": _rel(path), "line": hits[0], "line_end": hits[-1],
                "match": "lines", "span": True,
                # True when some lines could not be matched — the literal runs
                # at least this far, and may run further through the joins.
                "partial": len(hits) < len(clean),
            }

    longest = max((ln for ln in lines_ if len(ln) >= 24), key=len, default="")
    if not longest:
        return None
    for path in candidates:
        src = _read(path)
        idx = src.find(longest)
        if idx != -1:
            return {
                "file": _rel(path), "line": _line_of(src, idx),
                "match": "line", "span": False,
            }
    return None


def assign_span(module_rel: str, name: str) -> dict:
    """The exact source span of a module-level string constant.

    A citation read off the AST cannot be understated: ``lineno``/``end_lineno``
    bound the whole literal however the source breaks it up. ``locate`` has to
    work from the text alone — this works from the syntax, so it is the right
    citation for anything the map can name.
    """
    path = PKG / module_rel
    tree = ast.parse(_read(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return {
                "file": _rel(path), "line": node.lineno,
                "line_end": node.end_lineno, "match": "ast", "span": True,
            }
    raise SystemExit(
        f"promptmap: constant vanished: {module_rel}::{name}\n"
        "  Fix the generator rather than shipping a map that cites code that\n"
        "  no longer exists."
    )


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
        # An anchor: where the code is written, not a span of prompt text.
        "match": "symbol",
        "span": False,
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

#: Stand-ins for the run-scoped PATHS the runtime substitutes. These are
#: inline values, not blocks — they sit inside a line of the template, so the
#: map keeps them inline and flags the section as substituted rather than
#: shattering a 26-line template into fragments around each one.
_PATH_STUB = {
    "study_dir": "<study_dir>",
    "run_dir": "<study_dir>/runs/<run_id>",
    "debug_dir": "<study_dir>/runs/<run_id>/debug",
    "notes_dir": "<study_dir>/runs/<run_id>/debug/strategizer_notes",
    "experiment_data_dir": "<study_dir>/runs/<run_id>/experiment_data",
    "workspace_dir": "<study_dir>/runs/<run_id>/debug/delegations",
}

#: The two ``.format()`` fields that are whole BLOCKS of prompt text produced
#: by other code. They get their own rows in the map, cited to the method that
#: builds them — never folded into the template's own citation, which is what
#: made an earlier version of this map attribute a resource stanza written in
#: ``runtime/agent_runtime.py`` to ``prompts/agent_prompts.py``.
_BLOCK_FIELDS = ("resources", "knowledge")

#: Volatile facts in the resource stanza: real per run, and per HOST at build
#: time. Normalised so the committed map is deterministic and so no reviewer
#: reads the build machine's core count as a constant in the prompt.
_VOLATILE = (
    (re.compile(r"~\d+ CPU cores"), "~<cores> CPU cores"),
    (re.compile(r"RAM cap [^ ]+(?: GB)? per"), "RAM cap <ram_cap> per"),
    (re.compile(r"disk free [^.]+\."), "disk free <disk_free>."),
)


def _resources_text(for_worker: bool) -> str:
    """The resource stanza, obtained by CALLING the code that emits it."""
    from a3dasm._src.runtime.agent_runtime import AgenticRun

    class _Stub:
        _mem_cap_bytes = None
        study_dir = str(REPO)

    text = AgenticRun._resource_stanza(_Stub(), None, for_worker=for_worker)
    for pattern, repl in _VOLATILE:
        text = pattern.sub(repl, text)
    return text


def _knowledge_text(role: str) -> str:
    """The handbook menu for *role*, obtained by CALLING the live KnowledgeBase."""
    from a3dasm._src.runtime.agent_runtime import AgenticRun

    class _Stub:
        _kb = None

    return AgenticRun._kb_menu(_Stub(), role)


#: Where each block field comes from: the method that builds it, and what the
#: reader needs to know about how much of it is fixed.
_BLOCK_SOURCE = {
    "resources": (
        "runtime/agent_runtime.py", "AgenticRun._resource_stanza",
        "Built per delegation from the host/SLURM allocation. The wording below "
        "is the live method's own output; its cores/RAM/disk figures are "
        "normalised to placeholders because they differ per run and per host. "
        "The second paragraph appears for the implementer only.",
    ),
    "knowledge": (
        "runtime/agent_runtime.py", "AgenticRun._kb_menu",
        "The handbook MENU, filtered to this role's audience, so the agent always "
        "sees what it can pull with ConsultHandbook instead of having to guess a "
        "chapter exists. The chapter titles below come from the knowledge base, "
        "not from Python source — edit them in knowledge/kb.py and its chapters.",
    ),
}


def preamble_sections(tpl: str, role: str, is_entry: bool,
                      prefer: list[Path]) -> list[dict]:
    """The preamble as ONE readable block, with every part traced separately.

    The runtime assembles this prompt with ``.format()``, so it has two kinds
    of text in it: literal template written in ``agent_prompts.py``, and whole
    stanzas computed in ``agent_runtime.py`` and formatted in. Both facts
    matter and they pull against each other — splitting the prompt into a row
    per fragment cites everything correctly but leaves the reader assembling a
    closing ``</run_paths>`` tag back onto the text it closes, while showing
    one filled-in block reads properly and quietly puts ``agent_runtime.py``'s
    words under ``agent_prompts.py``'s name.

    So: one section, read top to bottom in order, carrying ``parts`` that say
    which stretch of it came from where. The page renders it as a single
    prompt with the computed stretches marked in place.
    """
    paths = {k: v for k, v in _PATH_STUB.items() if "{" + k + "}" in tpl}
    chunks = re.split(r"(\{(?:" + "|".join(_BLOCK_FIELDS) + r")\})", tpl)
    parts: list[dict] = []

    for chunk in chunks:
        field = chunk[1:-1] if chunk[:1] == "{" and chunk[-1:] == "}" else None
        if field in _BLOCK_FIELDS:
            module, qualname, note = _BLOCK_SOURCE[field]
            text = (_resources_text(role == "implementer") if field == "resources"
                    else _knowledge_text(role))
            sym = resolve_symbol(module, qualname)
            parts.append({
                "field": "{" + field + "}",
                "note": note,
                "text": text or f"({field} is empty for {role} — nothing is injected)",
                "empty": not text,
                "source": {"file": sym["file"], "line": sym["line"],
                           "match": "symbol", "span": False},
            })
            continue
        if not chunk:
            continue
        parts.append({
            "text": chunk.format(**paths) if paths else chunk,
            "source": locate(chunk, prefer),
        })

    text = "".join(part["text"] for part in parts)
    opening = re.match(r"<([a-z_0-9]+)>\n", tpl)
    return [{
        "tag": opening.group(1) if opening else None,
        "chars": len(text),
        "text": text,
        "parts": parts,
        "source": assign_span(
            "prompts/agent_prompts.py",
            "RUN_PATHS_PREAMBLE_TEMPLATE" if is_entry else "WORKSPACE_PREAMBLE_TEMPLATE",
        ),
        "substituted": bool(paths),
    }]


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
                src["span"] = False
                src.pop("line_end", None)
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


#: Why a section cannot be edited from the page. A citation that is not a
#: verbatim span is not a patch target: the page would be offering to rewrite
#: text the generator cannot put back without guessing.
_NOT_EDITABLE = {
    "lines": "The source breaks this literal across lines, so an edit here has "
             "no single span to write back to. Edit the file directly.",
    "line": "Only one line of this block could be matched in the source, so "
            "there is no span to write an edit back to.",
    "tag": "Only this block's opening tag could be located, so there is no "
           "span to write an edit back to.",
    "symbol": "This text is computed at run time by the cited code. Editing "
              "what it produced would be editing a shadow — change the code.",
}


_TOOL_NAME = re.compile(r"[A-Z][A-Za-z0-9]+$")


def tool_docs() -> dict[str, dict]:
    """Every PascalCase tool definition in the package, with its docstring span.

    ``agent.tools`` on the graph spec is a list of NAMES — the closures are
    bound per node at dispatch, so there is no live object here to read a
    ``__doc__`` off. The text an agent sees is nonetheless written down: each
    tool is a PascalCase method, and ``render_tool_catalog`` renders exactly
    ``ast.get_docstring``'s cleaned form of it plus any ``@tool_examples``. So
    the catalog can be reproduced from the syntax tree, and each entry cited to
    the docstring literal it is rendered from.

    A name defined in more than one place is recorded as ambiguous rather than
    guessed at: the map would otherwise cite one of two candidate docstrings
    with no way for a reader to know it had a choice.
    """
    found: dict[str, list[dict]] = {}
    for path in _py_files():
        tree = ast.parse(_read(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _TOOL_NAME.match(node.name):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            examples: list[str] = []
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Call)
                        and getattr(dec.func, "id", getattr(dec.func, "attr", "")) == "tool_examples"):
                    examples = [a.value for a in dec.args if isinstance(a, ast.Constant)]
            lit = node.body[0]
            found.setdefault(node.name, []).append({
                "doc": doc, "examples": examples, "file": _rel(path),
                "line": lit.lineno, "line_end": lit.end_lineno,
                "def_line": node.lineno,
            })
    out: dict[str, dict] = {}
    for name, hits in found.items():
        out[name] = dict(hits[0], ambiguous=[
            f"{h['file']}:{h['def_line']}" for h in hits] if len(hits) > 1 else [])
    return out


#: The two tools whose docstring is rebuilt per node at dispatch (``with_doc``
#: in nodes/tools/routing/_binding.py), so what the agent reads is not what the
#: source says — Delegate embeds its node's connected targets, AskForFeedback
#: the connected critic's description.
_PER_NODE_DOC = ("Delegate", "AskForFeedback")


def catalog_sections(tool_names: list[str]) -> list[dict]:
    """The ``<tools>`` catalog as readable, citable sections — one per tool.

    Mirrors ``render_tool_catalog``'s own formatting so the text here is the
    text the model reads, and cites each entry to the docstring it comes from.
    """
    docs = tool_docs()
    header = (
        "Your available tools, generated from the live tool set (AUTHORITATIVE "
        "— these exact names are the ones you call; anything not listed here is "
        "not available):"
    )
    sections = [{
        "tag": "tools", "label": None, "chars": len(header), "text": header,
        "source": resolve_symbol("prompts/tool_catalog.py", "render_tool_catalog"),
        "edit": {"ok": False, "why": "The catalog's own framing sentence, built by "
                                     "render_tool_catalog rather than written as a "
                                     "prompt. Edit prompts/tool_catalog.py."},
    }]

    for name in sorted(tool_names):
        spec = docs.get(name)
        if spec is None:
            sections.append({
                "tag": None, "label": name, "chars": 0,
                "text": "(no definition in this repository — this tool is provided "
                        "by the backend SDK, and its description comes with it)",
                "source": None, "external": True,
                "edit": {"ok": False, "why": "Defined by the backend SDK, not in "
                                             "this repository."},
            })
            continue
        body = f"### {name}\n{spec['doc']}"
        if spec["examples"]:
            body += "\nExamples:\n" + "\n".join(f"  - {e}" for e in spec["examples"])
        source = {"file": spec["file"], "line": spec["line"],
                  "line_end": spec["line_end"], "match": "docstring", "span": True}
        if spec["ambiguous"]:
            edit = {"ok": False, "why": "This name is defined more than once — "
                                        + " and ".join(spec["ambiguous"])
                                        + " — and which one binds depends on the node, "
                                          "so the map will not guess which docstring "
                                          "this agent is actually reading."}
        elif name in _PER_NODE_DOC:
            edit = {"ok": False, "why": "This tool's description is rebuilt for each "
                                        "node at dispatch — it embeds that node's own "
                                        "connections — so what the agent reads is not "
                                        "what the source says. Edit the builder in "
                                        "nodes/tools/routing/, not this text."}
        else:
            edit = {
                "ok": True, "mode": "docstring",
                "key": "{}:{}-{}".format(spec["file"].replace("/", "~"),
                                         spec["line"], spec["line_end"]),
                "file": spec["file"], "line": spec["line"],
                "line_end": spec["line_end"],
            }
        sections.append({"tag": None, "label": name, "chars": len(body),
                         "text": body, "source": source, "edit": edit,
                         "doc": spec["doc"]})
    return sections


def annotate_edits(roles: list[dict]) -> None:
    """Mark which sections the page may offer to edit, and why not otherwise.

    The map already knows: a citation it resolved as ``exact`` or ``ast`` IS a
    verbatim span of one file, so an edit to it can be written back by exact
    replacement. Everything else was located by a weaker probe, and offering an
    edit box over text with no span to write it to would be the same failure as
    a citation that overstates itself — the page would look precise and be
    guessing. So editability is derived from the citation, never asserted.
    """
    # How many roles carry each injected block — editing one edits them all.
    reach: dict[str, set[str]] = {}
    for role in roles:
        for layer in role["layers"]:
            for section in layer.get("sections", []):
                if section.get("injects"):
                    reach.setdefault(section["injects"]["name"], set()).add(role["id"])

    for role in roles:
        for layer in role["layers"]:
            for section in layer.get("sections", []):
                if "edit" in section:      # the catalog decides its own
                    continue
                source = section.get("source") or {}
                if section.get("parts"):
                    section["edit"] = {"ok": False, "why":
                        "Assembled by .format() from a template plus stanzas "
                        "built elsewhere — the pieces have different homes. "
                        "Edit prompts/agent_prompts.py directly."}
                elif source.get("match") in ("exact", "ast"):
                    section["edit"] = {
                        "ok": True,
                        "key": "{}:{}-{}".format(
                            source["file"].replace("/", "~"),
                            source["line"], source["line_end"]),
                        "file": source["file"],
                        "line": source["line"],
                        "line_end": source["line_end"],
                    }
                    if section.get("injects"):
                        others = sorted(reach[section["injects"]["name"]] - {role["id"]})
                        if others:
                            section["edit"]["shared"] = {
                                "name": section["injects"]["name"],
                                "also": others,
                            }
                else:
                    section["edit"] = {"ok": False, "why": _NOT_EDITABLE.get(
                        source.get("match"), "This block could not be located "
                        "as a verbatim span of any file.")}


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

    prompts_py = [PKG / "prompts" / "agent_prompts.py"]
    preamble_src = {
        name: assign_span("prompts/agent_prompts.py", name)
        for name in ("RUN_PATHS_PREAMBLE_TEMPLATE", "WORKSPACE_PREAMBLE_TEMPLATE")
    }

    roles = []
    for name, agent in graph.nodes.items():
        module = sys.modules[type(agent).__module__]
        mod_path = Path(module.__file__).resolve()
        is_entry = name == entry

        layers = []

        # Layer 1 — the run-scoped preamble prepended in agent_runtime.py. It
        # is NOT one block: the template is literal text from agent_prompts.py
        # with two whole stanzas formatted into it from agent_runtime.py, so
        # the map splits it and cites each piece where it is actually written.
        tpl = RUN_PATHS_PREAMBLE_TEMPLATE if is_entry else WORKSPACE_PREAMBLE_TEMPLATE
        tpl_name = "RUN_PATHS_PREAMBLE_TEMPLATE" if is_entry else "WORKSPACE_PREAMBLE_TEMPLATE"
        sections = preamble_sections(tpl, name, is_entry, prompts_py)
        layers.append({
            "kind": "preamble",
            "label": tpl_name,
            "note": "Prepended by the runtime, in the order shown. Paths are substituted "
                    "per run and stand in as placeholders; the two highlighted stretches "
                    "are whole stanzas computed elsewhere, each cited to the code that "
                    "builds it rather than to the template that formats it in.",
            "assembled_at": locate("preamble = " + tpl_name),
            "definition": preamble_src[tpl_name],
            "chars": sum(sec["chars"] for sec in sections),
            "sections": sections,
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
            "definition": {"file": _rel(mod_path),
                           "line": _class_line(mod_path, type(agent).__name__),
                           "match": "symbol", "span": False},
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
                    "docstring, read off the source it is written in. The closures "
                    "bind per node at dispatch, so a tool whose description is built "
                    "there rather than written down says so instead of pretending.",
            "definition": resolve_symbol("prompts/tool_catalog.py", "render_tool_catalog"),
            "assembled_at": locate("system_prompt=system_prompt_with_catalog("),
            "chars": None,
            "tools": sorted(tools),
            "sections": catalog_sections(sorted(tools)),
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
    annotate_edits(roles)
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
