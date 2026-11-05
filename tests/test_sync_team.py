"""Tests for per-team sync config + clone-dir isolation in millet.sync."""
from __future__ import annotations

import pytest

from millet import sync


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MEET_CONFIG_DIR", raising=False)
    return tmp_path


def test_per_team_config_roundtrip(home):
    sync.save_sync_config({"repo_url": "https://example.com/team-a.git"}, team="team-a")
    sync.save_sync_config(
        {"repo_url": "https://example.com/team-b.git"}, team="team-b"
    )

    team_a = sync.load_sync_config(team="team-a")
    team_b = sync.load_sync_config(team="team-b")
    assert team_a["repo_url"].endswith("team-a.git")
    assert team_b["repo_url"].endswith("team-b.git")

    # Files live in separate per-team dirs.
    assert (home / ".config" / "meet" / "team-a" / "sync_config.json").exists()
    assert (home / ".config" / "meet" / "team-b" / "sync_config.json").exists()
    # The global file is untouched.
    assert not (home / ".config" / "meet" / "sync_config.json").exists()


def test_is_sync_configured_per_team(home):
    assert sync.is_sync_configured(team="team-a") is False
    sync.save_sync_config({"repo_url": "https://example.com/team-a.git"}, team="team-a")
    assert sync.is_sync_configured(team="team-a") is True
    # A different team is still unconfigured.
    assert sync.is_sync_configured(team="team-b") is False


def test_clone_dir_namespaced_by_team(home):
    """Two teams pointing at same-named repos get distinct clone dirs."""
    url_a = "https://github.com/org-a/meetings.git"
    url_b = "https://github.com/org-b/meetings.git"
    dir_a = sync._clone_dir_for(url_a, "team-a")
    dir_b = sync._clone_dir_for(url_b, "team-b")
    assert dir_a != dir_b
    assert dir_a.parent.name == "team-a"
    assert dir_b.parent.name == "team-b"
    # Teamless keeps the flat layout.
    flat = sync._clone_dir_for(url_a, None)
    assert flat == sync.CLONE_BASE_DIR / "meetings"


def test_global_config_still_works(home, tmp_path):
    """No-team path is unchanged (back-compat).

    The global SYNC_CONFIG_PATH is a module-level constant frozen at
    import, so we route through an explicit config_path to exercise the
    teamless code path deterministically under a tmp HOME.
    """
    cfg = tmp_path / "global_sync_config.json"
    sync.save_sync_config(
        {"repo_url": "https://example.com/global.git"}, config_path=cfg
    )
    assert sync.is_sync_configured(config_path=cfg) is True
    assert cfg.exists()


# ── CLI exit-code propagation (sync failure must not exit 0) ─────────────────


def test_cli_sync_propagates_failure_exit_code(home, tmp_path, monkeypatch):
    """A push/sync failure exits non-zero so callers (e.g. vezir) don't have
    to scrape the log to notice a failed sync."""
    from click.testing import CliRunner

    from millet.cli.sync import sync as sync_cmd

    sdir = tmp_path / "meeting-20260101-000000"
    sdir.mkdir()

    # cli/sync.py imports these from millet.sync inside the function, so patch
    # the source module.
    monkeypatch.setattr("millet.sync.is_sync_configured", lambda team=None: True)
    from millet.sync import MeetingMatch
    monkeypatch.setattr(
        "millet.sync.detect_meeting_type",
        lambda path, team=None: MeetingMatch(name="Dev Sync", folder="dev-sync"),
    )

    def _boom(*a, **k):
        raise RuntimeError("Command failed: git push")

    monkeypatch.setattr("millet.sync.sync_session", _boom)

    result = CliRunner().invoke(sync_cmd, [str(sdir)])
    assert result.exit_code == 1
    assert "Command failed: git push" in result.output


def test_cli_sync_success_exits_zero(home, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from millet.cli.sync import sync as sync_cmd
    from millet.sync import MeetingMatch

    sdir = tmp_path / "meeting-20260101-000000"
    sdir.mkdir()
    monkeypatch.setattr("millet.sync.is_sync_configured", lambda team=None: True)
    monkeypatch.setattr(
        "millet.sync.detect_meeting_type",
        lambda path, team=None: MeetingMatch(name="Dev Sync", folder="dev-sync"),
    )
    monkeypatch.setattr(
        "millet.sync.sync_session", lambda *a, **k: [tmp_path / "summary.md"]
    )
    result = CliRunner().invoke(sync_cmd, [str(sdir)])
    assert result.exit_code == 0
