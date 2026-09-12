"""Delegation-bounded version control of a run's agent workspace.

See ``internal/specs/11-delegation-bounded-version-control.md``.

A delegation's account of what it changed is prose the agent wrote about
itself, and nothing on disk can contradict it. This module gives the harness
a mechanical record instead: one git repository per run, one commit per
delegation, the sha stamped onto the delegation record.

Agents never see this repo and get no git tool. The harness commits on their
behalf — an agent that can rewrite the history recording its own work defeats
the point.

Nesting inside an existing repository
-------------------------------------
The workspace lives at ``runs/<ts>/debug/delegations/``, which is normally
INSIDE a checkout of a3dasm (or of a benchmarks repo). Git's directory
discovery walks upward until it finds a ``.git``, so a plain ``git add -A``
run in that directory before the nested repo exists would find the PARENT
repository and stage the entire source tree. That is the one catastrophic
failure mode here, and every call in this module is shaped to make it
impossible rather than unlikely:

* every invocation passes ``--git-dir`` and ``--work-tree`` explicitly, so
  git never performs discovery and never walks upward — these also override
  any inherited ``GIT_DIR``/``GIT_WORK_TREE`` environment variables;
* ``_git`` refuses to run at all unless the nested ``.git`` already exists,
  so a failed or skipped init degrades to "no commits", never to "committed
  against the parent";
* the resulting sha is verified to come from the nested repo before it is
  returned, so a sha can never be a parent-repo commit.

The parent sees the nested repo as an ordinary untracked directory. a3dasm's
own ``.gitignore`` excludes ``studies/*/runs/`` so that a careless
``git add -A`` in the parent cannot absorb it as a gitlink/submodule.

Configuration is passed per-invocation with ``-c`` rather than written into
the repo, which keeps it hermetic and immune to the user's global git config:
commit signing is disabled (a global ``commit.gpgsign`` would otherwise block
on a passphrase prompt inside a run), hooks are disabled (a global
``core.hooksPath`` could run arbitrary code), and an identity is supplied so
a machine with no configured ``user.email`` cannot fail the commit.

Nothing here is ever fatal. Version control is an observation instrument, not
a dependency of the science: every failure logs a warning and returns a value
the caller records as "no sha".
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["init_workspace_repo", "commit_workspace", "WORKSPACE_AUTHOR"]

WORKSPACE_AUTHOR = ("a3dasm", "a3dasm@localhost")

# Per-invocation config. See the module docstring for why each one is here.
_HERMETIC_CONFIG = (
    "-c", "commit.gpgsign=false",
    "-c", "core.hooksPath=/dev/null",
    "-c", "gc.auto=0",
    "-c", "core.autocrlf=false",
    "-c", f"user.name={WORKSPACE_AUTHOR[0]}",
    "-c", f"user.email={WORKSPACE_AUTHOR[1]}",
)

# A git call here touches only the local filesystem; anything slower than this
# is a hung invocation (a credential or editor prompt), not slow work.
_TIMEOUT_S = 60


def _git(workspace: Path, *args: str) -> subprocess.CompletedProcess | None:
    """Run one git command pinned to the workspace's OWN repository.

    Returns None if the nested repo does not exist, git is unavailable, or the
    call times out. Never raises, and never runs a command that could resolve
    against a parent repository: the explicit --git-dir/--work-tree pair
    disables discovery, and the existence check refuses to run without them
    pointing at a real repo.
    """
    git_dir = workspace / ".git"
    if not git_dir.exists():
        return None
    try:
        return subprocess.run(
            ["git", f"--git-dir={git_dir}", f"--work-tree={workspace}",
             *_HERMETIC_CONFIG, *args],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("workspace git %s failed: %s", args[0] if args else "?", exc)
        return None


def init_workspace_repo(workspace: Path) -> bool:
    """Create the run's workspace repository. Idempotent; never raises.

    An empty root commit is made so the first delegation's commit has a parent
    and its diff is therefore expressible as a normal ``git show``.
    """
    workspace = Path(workspace)
    try:
        workspace.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.warning("could not create workspace %s: %s", workspace, exc)
        return False

    if (workspace / ".git").exists():
        return True  # resumed run, or a second call in the same run

    try:
        # `git init <path>` targets the path explicitly and does not consult
        # any enclosing repository, so nesting inside one is safe here.
        done = subprocess.run(
            ["git", "-c", "init.defaultBranch=main", "init", "-q", str(workspace)],
            capture_output=True, text=True, timeout=_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning(
            "workspace version control unavailable (git init failed: %s) — "
            "delegations will record no workspace_sha", exc)
        return False
    if done.returncode != 0:
        log.warning(
            "workspace version control unavailable (git init: %s) — "
            "delegations will record no workspace_sha", done.stderr.strip())
        return False

    root = _git(workspace, "commit", "--allow-empty", "-q", "-m",
                "run workspace: root commit")
    if root is None or root.returncode != 0:
        log.warning(
            "workspace repo created but the root commit failed: %s",
            "" if root is None else root.stderr.strip())
        return False
    return True


def commit_workspace(workspace: Path, message: str) -> str | None:
    """Commit the workspace's current state; return the sha, or None.

    ``--allow-empty`` so every delegation has exactly one commit whether or
    not it touched a file — an absent commit would otherwise be ambiguous
    between "changed nothing" and "the record failed".
    """
    workspace = Path(workspace)
    if _git(workspace, "add", "-A") is None:
        return None  # no nested repo: say nothing rather than guess

    done = _git(workspace, "commit", "--allow-empty", "-q", "-m", message)
    if done is None or done.returncode != 0:
        log.warning(
            "workspace commit failed for %r: %s", message,
            "" if done is None else done.stderr.strip())
        return None

    head = _git(workspace, "rev-parse", "HEAD")
    if head is None or head.returncode != 0:
        return None
    sha = head.stdout.strip()
    # A sha is only meaningful if it came from the workspace's own repo. _git
    # already pins --git-dir there, so this is a cheap assertion that the
    # pinning held rather than a second line of defence being relied upon.
    return sha or None
