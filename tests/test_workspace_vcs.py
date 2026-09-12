"""Spec 11 — delegation-bounded version control of the run workspace.

The load-bearing test in this file is
``test_commit_never_touches_an_enclosing_repository``: the workspace normally
lives inside a checkout of a3dasm, so a git call that performed directory
discovery would find the PARENT repo and stage the whole source tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from a3dasm._src.infra.workspace_vcs import commit_workspace, init_workspace_repo


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", f"--git-dir={repo / '.git'}", f"--work-tree={repo}", *args],
        capture_output=True, text=True, check=True,
    )
    return done.stdout.strip()


def _make_parent_repo(root: Path) -> Path:
    """A git repo standing in for the a3dasm checkout the run lives inside."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-c", "init.defaultBranch=main", "init", "-q", str(root)],
                   check=True, capture_output=True)
    (root / "source_file.py").write_text("# parent repo content\n")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t",
         "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t",
         "commit", "-q", "-m", "parent root")
    return root


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_workspace_repo_initialised_with_a_root_commit(tmp_path):
    ws = tmp_path / "runs" / "20260912T000000" / "debug" / "delegations"
    assert init_workspace_repo(ws) is True
    assert (ws / ".git").exists()
    assert _git(ws, "rev-list", "--count", "HEAD") == "1"


def test_init_is_idempotent_and_keeps_history(tmp_path):
    """A resumed run calls init again; it must not wipe or re-root the repo."""
    ws = tmp_path / "ws"
    init_workspace_repo(ws)
    commit_workspace(ws, "D001 a -> b [DONE]")
    before = _git(ws, "rev-parse", "HEAD")

    assert init_workspace_repo(ws) is True
    assert _git(ws, "rev-parse", "HEAD") == before
    assert _git(ws, "rev-list", "--count", "HEAD") == "2"


# ---------------------------------------------------------------------------
# the nesting guarantee
# ---------------------------------------------------------------------------


def test_commit_never_touches_an_enclosing_repository(tmp_path):
    """The whole point of pinning --git-dir/--work-tree.

    Without it, `git add -A` inside a not-yet-initialised workspace would walk
    upward, find the a3dasm checkout, and stage the entire source tree.
    """
    parent = _make_parent_repo(tmp_path / "a3dasm")
    ws = parent / "studies" / "s" / "runs" / "ts" / "debug" / "delegations"
    ws.mkdir(parents=True)

    # BEFORE init there is no workspace repo: the call must decline, NOT fall
    # through to the parent.
    (ws / "D001_scratch.py").write_text("x = 1\n")
    assert commit_workspace(ws, "D001 a -> b [DONE]") is None
    assert _git(parent, "rev-list", "--count", "HEAD") == "1"
    assert _git(parent, "status", "--porcelain") != ""  # still just untracked

    # AFTER init, commits land in the nested repo and the parent is untouched.
    init_workspace_repo(ws)
    sha = commit_workspace(ws, "D001 a -> b [DONE]")
    assert sha
    assert _git(parent, "rev-list", "--count", "HEAD") == "1"
    assert sha not in _git(parent, "rev-list", "HEAD")
    assert "D001_scratch.py" in _git(ws, "show", "--stat", "--name-only", sha)


def test_parent_gitignore_excludes_run_trees():
    """a3dasm's own .gitignore must exclude studies/*/runs/, or a `git add -A`
    in the repo absorbs a run's nested workspace repo as a gitlink."""
    root = Path(__file__).resolve().parent.parent
    done = subprocess.run(
        ["git", "check-ignore", "-q", "studies/example_study/runs/"],
        cwd=root, capture_output=True,
    )
    assert done.returncode == 0, "studies/*/runs/ is not gitignored"


# ---------------------------------------------------------------------------
# commits
# ---------------------------------------------------------------------------


def test_each_delegation_produces_exactly_one_commit(tmp_path):
    ws = tmp_path / "ws"
    init_workspace_repo(ws)

    (ws / "D001").mkdir()
    (ws / "D001" / "a.py").write_text("a\n")
    assert commit_workspace(ws, "D001 s -> impl [DONE]")

    # a delegation that writes NOTHING still gets a commit: an absent one is
    # ambiguous between "changed nothing" and "the record failed"
    assert commit_workspace(ws, "D002 s -> lit [DONE]")

    (ws / "D003").mkdir()
    (ws / "D003" / "c.py").write_text("c\n")
    assert commit_workspace(ws, "D003 s -> math [FAILED]")

    assert _git(ws, "rev-list", "--count", "HEAD") == "4"  # root + 3
    subjects = _git(ws, "log", "--format=%s").splitlines()
    assert subjects[0].startswith("D003 s -> math [FAILED]")


def test_commit_names_the_files_the_delegation_changed(tmp_path):
    """The point of the whole spec: 'which files changed' becomes evidence."""
    ws = tmp_path / "ws"
    init_workspace_repo(ws)
    (ws / "D001").mkdir()
    (ws / "D001" / "main.py").write_text("v1\n")
    commit_workspace(ws, "D001 s -> math [DONE]")

    (ws / "D002").mkdir()
    (ws / "D002" / "variant.py").write_text("v2\n")
    (ws / "D001" / "main.py").write_text("v1 revised\n")
    sha = commit_workspace(ws, "D002 s -> math [DONE]")

    changed = _git(ws, "show", "--pretty=", "--name-only", sha).split()
    assert sorted(changed) == ["D001/main.py", "D002/variant.py"]


def test_commit_without_a_repo_returns_none(tmp_path):
    """Version control is an instrument, not a dependency: no repo, no sha,
    no exception."""
    ws = tmp_path / "never_initialised"
    ws.mkdir()
    assert commit_workspace(ws, "D001 a -> b [DONE]") is None


def test_git_unavailable_does_not_fail_the_run(tmp_path, monkeypatch):
    """git missing from PATH degrades to 'no sha', never to a crashed run."""
    def _no_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", _no_git)
    ws = tmp_path / "ws"
    assert init_workspace_repo(ws) is False
    assert commit_workspace(ws, "D001 a -> b [DONE]") is None


def test_hostile_global_git_config_cannot_block_a_commit(tmp_path, monkeypatch):
    """A developer with commit.gpgsign=true globally would otherwise hang on a
    passphrase prompt, and one with no user.email would fail outright. Both are
    neutralised per-invocation."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    (fake_home / ".gitconfig").write_text(
        "[commit]\n\tgpgsign = true\n"
        "[user]\n\tsigningkey = DEADBEEF\n"
    )
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.delenv("GIT_AUTHOR_NAME", raising=False)
    monkeypatch.delenv("GIT_AUTHOR_EMAIL", raising=False)
    monkeypatch.delenv("GIT_COMMITTER_NAME", raising=False)
    monkeypatch.delenv("GIT_COMMITTER_EMAIL", raising=False)

    ws = tmp_path / "ws"
    assert init_workspace_repo(ws) is True
    (ws / "f.txt").write_text("x\n")
    assert commit_workspace(ws, "D001 a -> b [DONE]")


def test_inherited_git_dir_env_cannot_redirect_the_commit(tmp_path, monkeypatch):
    """GIT_DIR/GIT_WORK_TREE in the environment must not steer a commit into
    another repo; the explicit flags outrank them."""
    other = _make_parent_repo(tmp_path / "other")
    ws = tmp_path / "ws"
    init_workspace_repo(ws)

    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    (ws / "mine.txt").write_text("mine\n")
    sha = commit_workspace(ws, "D001 a -> b [DONE]")

    assert sha
    assert _git(other, "rev-list", "--count", "HEAD") == "1"
    assert "mine.txt" in _git(ws, "show", "--pretty=", "--name-only", sha)


@pytest.mark.parametrize("status", ["DONE", "FAILED"])
def test_message_records_route_and_outcome(tmp_path, status):
    ws = tmp_path / "ws"
    init_workspace_repo(ws)
    sha = commit_workspace(ws, f"D007 strategizer -> implementer [{status}]")
    assert _git(ws, "log", "-1", "--format=%s", sha) == (
        f"D007 strategizer -> implementer [{status}]"
    )
