"""ContainerRunner Claude auth precedence: subscription-preferred, API-key
fallback (warned), hard-fail when no credential is available."""
from __future__ import annotations

import pytest

from a3dasm._src.container_runner import ContainerRunner


def _runner(tmp_path):
    return ContainerRunner(study_dir=tmp_path, backend="claude")


def test_oauth_token_preferred_and_api_key_not_forwarded(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-oauth-xxx")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-be-ignored")
    # point config dir at an empty dir so the creds-mount branch can't fire
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "empty"))

    args = _runner(tmp_path)._claude_auth_args()

    assert args == ["-e", "CLAUDE_CODE_OAUTH_TOKEN"]
    # the API key must NOT be forwarded — it would outrank the subscription
    assert "ANTHROPIC_API_KEY" not in args


def test_host_credentials_mounted_when_present(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = tmp_path / "claude_cfg"
    cfg.mkdir()
    (cfg / ".credentials.json").write_text("{}")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(cfg))

    args = _runner(tmp_path)._claude_auth_args()

    assert "-v" in args
    assert f"{cfg}:/claude-config" in args
    assert "CLAUDE_CONFIG_DIR=/claude-config" in args


def test_api_key_is_a_warned_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fallback")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "empty"))  # no creds

    with pytest.warns(UserWarning, match="Subscription auth is preferred"):
        args = _runner(tmp_path)._claude_auth_args()

    assert args == ["-e", "ANTHROPIC_API_KEY"]


def test_no_credentials_hard_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "empty"))

    with pytest.raises(RuntimeError, match="No Claude credentials"):
        _runner(tmp_path)._claude_auth_args()
