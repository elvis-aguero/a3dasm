"""Notes tool closures: WriteNote, ReadNote -- the strategizer notes surface.
Built per-node via build_notes_closures(node) -- the node is passed in so
closures reach its state. Extracted verbatim from the former
build_routing_tools()."""
from __future__ import annotations

from pathlib import Path


def build_notes_closures(node) -> dict:
    study_dir = node._study_dir

    def WriteNote(path: str, body: str) -> str:
        """Write a Markdown (.md) note to strategizer_notes/ — free-form
        reasoning: why you chose each Delegate, interim findings, open issues.
        Do NOT write code in notes (embed it in Delegate().intent as plain
        text), and do NOT record priors/posteriors here — those live ONLY in
        the hypothesis ledger (HypothesisPropose/Update)."""
        prefix = node._drain_notifications()
        notes_dir = node._current_notes_dir
        if notes_dir is None:
            return "ERROR: notes_dir not set (run_dir missing from state)."
        bare = Path(path).name
        if bare.endswith(".ipynb"):
            if study_dir is None:
                return prefix + "ERROR: study_dir not set — cannot write .ipynb."
            target = Path(study_dir).resolve() / bare
        else:
            if not bare.endswith(".md"):
                bare = bare + ".md"
            target = Path(notes_dir) / bare
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
        return prefix + f"Written: {target}"


    def ReadNote(path: str) -> str:
        """Read a file — or LIST a directory — from the study directory. Use it
        to load PROBLEM_STATEMENT.md, review prior notes, and (importantly) to
        reuse the implementers' work: point it at a delegation workspace
        (workspace_dir/D###/) to LIST its files, then read the script you want
        to consolidate into pipeline.ipynb. Read what you need, not everything."""
        prefix = node._drain_notifications()
        if study_dir is None:
            return "ERROR: study_dir not set."
        # Contain to the study directory. An absolute or ..-escaping path — e.g.
        # ReadNote("/") — otherwise resolves OUTSIDE the study tree, and a
        # recursive listing of "/" walks the entire filesystem and HANGS the run
        # (the tool call never returns, so the strategizer turn never ends and
        # the time backstop, checked only between turns, never fires).
        study_root = Path(study_dir).resolve()
        target = (study_root / path).resolve()
        try:
            target.relative_to(study_root)
        except ValueError:
            return prefix + (
                f"ERROR: {path!r} resolves outside the study directory; "
                "use a path within it (e.g. 'PROBLEM_STATEMENT.md' or 'D001/')."
            )
        if not target.exists():
            return prefix + f"NOT FOUND: {target}"
        if target.is_dir():
            # List files recursively so the agent can discover what an
            # implementer wrote — but BOUND the walk: stop after _MAX files so a
            # large tree can never hang the run.
            _MAX = 300
            found: list[str] = []
            try:
                for p in target.rglob("*"):
                    if p.is_file():
                        found.append(str(p.relative_to(target)))
                        if len(found) > _MAX:
                            break
            except Exception:  # noqa: BLE001
                found = [p.name for p in target.iterdir()]
            truncated = len(found) > _MAX
            entries = sorted(found[:_MAX])
            listing = "\n".join(f"  {e}" for e in entries) or "  (empty)"
            if truncated:
                listing += "\n  … (more files — narrow the path)"
            return prefix + (
                f"{target} is a directory. Files (read one with "
                f"ReadNote('{path.rstrip('/')}/<file>')):\n{listing}"
            )
        return prefix + target.read_text(encoding="utf-8")


    return {
        "WriteNote": WriteNote,
        "ReadNote": ReadNote,
    }
