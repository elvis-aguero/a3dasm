"""Draft knowledge base: the canonical source of truth for agentic
conventions and practices, consultable on demand.

STATUS: DRAFT scaffold. This lays out what will become a worker-facing
"consult the standards" tool. v1 is a curated markdown corpus with a simple
keyword search; the search backend is intended to be swapped for semantic
(embedding) retrieval once the corpus is large enough to justify it — the
``KBEntry`` / ``KnowledgeBase`` API is designed to stay stable across that
change, so callers (and a future ``ConsultKnowledge`` tool) do not break.

WHAT BELONGS HERE (and what does not):
- YES: discretionary conventions, idioms, how-tos, and gotchas — the long
  tail of "how we do things" that helps a worker do its job well.
- NO: load-bearing invariants. Mandatory rules (ground truth MUST go through
  get_evaluator(); the headline MUST reproduce from the canonical store) are
  ENFORCED by the ScienceMonitor and the critic gate, NOT left to optional
  retrieval. The KB may restate an invariant as guidance, but enforcement
  never depends on an agent choosing to search.

Entries live as curated markdown files under ``entries/``; each file is one
self-contained chunk with frontmatter (id/title/tags/audience). Friction
surfaced by node retrospectives is the natural feeder for new entries — with
a human curation gate, never auto-ingest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .charter import FALSIFICATION_CHARTER

_ENTRIES_DIR = Path(__file__).parent / "entries"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass
class KBEntry:
    """One self-contained knowledge chunk."""

    id: str
    title: str
    tags: list[str] = field(default_factory=list)
    audience: list[str] = field(default_factory=list)
    body: str = ""

    @property
    def text(self) -> str:
        """Full searchable text (title + tags + body)."""
        return f"{self.title}\n{' '.join(self.tags)}\n{self.body}"

    @property
    def summary(self) -> str:
        """First non-empty body line, trimmed — the one-liner for the TOC."""
        for line in self.body.splitlines():
            s = line.strip()
            if s:
                return s if len(s) <= 88 else s[:85] + "..."
        return ""

    def render(self) -> str:
        """Human/agent-facing rendering of the entry."""
        tags = f"  [tags: {', '.join(self.tags)}]" if self.tags else ""
        return f"## {self.title}{tags}\n\n{self.body.strip()}\n"


def _parse_frontmatter(raw: str) -> dict:
    """Minimal frontmatter parser — `key: value`, with `[a, b]` lists.

    Dependency-free on purpose; the corpus frontmatter is intentionally
    simple. Unknown/malformed lines are skipped, never fatal.
    """
    meta: dict = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [v.strip() for v in value[1:-1].split(",")]
            meta[key] = [v for v in items if v]
        else:
            meta[key] = value
    return meta


def _load_entry(path: Path) -> KBEntry | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001
        return None
    m = _FRONTMATTER_RE.match(text)
    if m is None:
        # No frontmatter: treat the filename as id/title, whole file as body.
        return KBEntry(id=path.stem, title=path.stem, body=text)
    meta = _parse_frontmatter(m.group(1))
    _tags = meta.get("tags", [])
    _aud = meta.get("audience", [])
    return KBEntry(
        id=str(meta.get("id", path.stem)),
        title=str(meta.get("title", path.stem)),
        tags=_tags if isinstance(_tags, list) else [],
        audience=_aud if isinstance(_aud, list) else [],
        body=m.group(2),
    )


def _charter_entry() -> KBEntry:
    """The falsification charter exposed as a handbook chapter.

    Sourced from the single ``FALSIFICATION_CHARTER`` constant (the same text
    push-injected into the strategizer/critic prompts) — NOT a duplicated file,
    so there is one source and no drift. It is a canonical chapter, always
    present regardless of the on-disk corpus.
    """
    return KBEntry(
        id="falsification-charter",
        title="Popperian validation — the falsification charter",
        tags=["hypothesis", "falsification", "popper", "verdict", "charter"],
        audience=["strategizer", "critic", "implementer"],
        body=FALSIFICATION_CHARTER,
    )


class KnowledgeBase:
    """Curated, on-demand knowledge base of agentic conventions.

    DRAFT: ``search`` is a transparent keyword scorer — a stand-in for the
    eventual semantic (embedding) backend. The API surface (``load``,
    ``entries``, ``search``, ``get``) is what a future ``ConsultKnowledge``
    worker tool will call, so it is kept stable while the backend evolves.
    """

    def __init__(self, entries: list[KBEntry]) -> None:
        self._entries = entries
        self._by_id = {e.id: e for e in entries}

    @classmethod
    def load(cls, entries_dir: Path | None = None) -> KnowledgeBase:
        """Load all curated entries from the corpus directory."""
        d = entries_dir or _ENTRIES_DIR
        # The charter is a canonical chapter, always first — it is the
        # contract everything else is consistent with.
        entries: list[KBEntry] = [_charter_entry()]
        if d.is_dir():
            for path in sorted(d.glob("*.md")):
                entry = _load_entry(path)
                if entry is not None:
                    entries.append(entry)
        return cls(entries)

    @property
    def entries(self) -> list[KBEntry]:
        return list(self._entries)

    def get(self, entry_id: str) -> KBEntry | None:
        return self._by_id.get(entry_id)

    def toc(self) -> str:
        """The table of contents: one line per chapter (id — title + summary).

        Cheap and complete — the agent sees every chapter and drills into one
        by id, instead of guessing keywords that may match nothing.
        """
        lines = [
            'HANDBOOK — available chapters. Call ConsultHandbook("<id>") to '
            'read one in full, or ConsultHandbook("<keywords>") to search.',
            "",
        ]
        for e in self._entries:
            lines.append(f"- {e.id} — {e.title}")
            if e.summary:
                lines.append(f"    {e.summary}")
        return "\n".join(lines)

    def menu(self, audience: str | None = None) -> str:
        """A compact, audience-filtered MENU of available chapters for INJECTION
        into a system prompt — one line per entry (``id: title``), so an agent
        always sees the latent knowledge it can pull, the same way it always sees
        its tool list. This closes the discovery chicken-and-egg: ``toc()`` is only
        seen if the agent already thought to call ConsultHandbook.

        ``audience`` filters to entries relevant to that role (entries with no
        declared audience are shown to everyone). Empty string if nothing matches.
        Titles are the descriptors and are capped to one terse line by the
        ≤100-char invariant (enforced in tests)."""
        rows = [
            f"- {e.id}: {e.title}"
            for e in self._entries
            if not (audience and e.audience and audience not in e.audience)
        ]
        if not rows:
            return ""
        header = (
            "<knowledge_base>\n"
            "Latent expertise on tap — you are NOT expected to know these by "
            'heart. Pull a chapter with ConsultHandbook("<id>"), or search with '
            'ConsultHandbook("<keywords>"), whenever one is relevant:'
        )
        return header + "\n" + "\n".join(rows) + "\n</knowledge_base>\n"

    def search(self, query: str, k: int = 3) -> list[KBEntry]:
        """Return up to ``k`` entries most relevant to ``query``.

        DRAFT scorer: token overlap weighted title>tags>body. Returns only
        entries with a non-zero score. Replace with semantic retrieval when
        the corpus outgrows keyword matching — callers stay unchanged.
        """
        q_tokens = set(_TOKEN_RE.findall(query.lower()))
        if not q_tokens:
            return []
        scored: list[tuple[int, KBEntry]] = []
        for e in self._entries:
            title_t = set(_TOKEN_RE.findall(e.title.lower()))
            tag_t = set(_TOKEN_RE.findall(" ".join(e.tags).lower()))
            body_t = set(_TOKEN_RE.findall(e.body.lower()))
            score = (
                3 * len(q_tokens & title_t)
                + 2 * len(q_tokens & tag_t)
                + 1 * len(q_tokens & body_t)
            )
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [e for _, e in scored[:k]]
