"""Tests for the central team-aware path resolver (millet.paths)."""
from __future__ import annotations

from pathlib import Path

import pytest

from millet import paths


def test_global_profiles_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MEET_PROFILES_PATH", raising=False)
    monkeypatch.delenv("MEET_CONFIG_DIR", raising=False)
    p = paths.profiles_path()
    assert p == tmp_path / ".config" / "meet" / "speaker_profiles.json"


def test_team_profiles_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MEET_PROFILES_PATH", raising=False)
    monkeypatch.delenv("MEET_CONFIG_DIR", raising=False)
    p = paths.profiles_path("team-a")
    assert p == tmp_path / ".config" / "meet" / "team-a" / "speaker_profiles.json"


def test_env_override_wins_when_no_team(monkeypatch, tmp_path):
    monkeypatch.setenv("MILLET_PROFILES_PATH", "/custom/profiles.json")
    assert paths.profiles_path() == Path("/custom/profiles.json")


def test_legacy_env_alias_still_works(monkeypatch):
    """MEET_PROFILES_PATH still honored for one release, with a warning."""
    import warnings
    monkeypatch.delenv("MILLET_PROFILES_PATH", raising=False)
    monkeypatch.setenv("MEET_PROFILES_PATH", "/legacy/p.json")
    # Reset the one-time-warning dedupe so this test sees the warning.
    paths._RENAME_WARNED.discard("MEET_PROFILES_PATH")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = paths.profiles_path()
    assert result == Path("/legacy/p.json")
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_new_env_name_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("MILLET_PROFILES_PATH", "/new/p.json")
    monkeypatch.setenv("MEET_PROFILES_PATH", "/old/p.json")
    assert paths.profiles_path() == Path("/new/p.json")


def test_team_ignores_env_override(monkeypatch, tmp_path):
    """An explicit team is more specific than the process-wide env override."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MEET_PROFILES_PATH", "/custom/profiles.json")
    monkeypatch.delenv("MEET_CONFIG_DIR", raising=False)
    p = paths.profiles_path("team-a")
    assert p == tmp_path / ".config" / "meet" / "team-a" / "speaker_profiles.json"


def test_sync_config_path_team(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MEET_CONFIG_DIR", raising=False)
    assert paths.sync_config_path() == (
        tmp_path / ".config" / "meet" / "sync_config.json"
    )
    assert paths.sync_config_path("team-b") == (
        tmp_path / ".config" / "meet" / "team-b" / "sync_config.json"
    )


def test_recordings_dir_team(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MEET_RECORDINGS_DIR", raising=False)
    assert paths.recordings_dir() == tmp_path / "meet-recordings"
    assert paths.recordings_dir("team-a") == tmp_path / "meet-recordings" / "team-a"


def test_recordings_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MILLET_RECORDINGS_DIR", str(tmp_path / "rec"))
    assert paths.recordings_dir() == tmp_path / "rec"
    assert paths.recordings_dir("team-a") == tmp_path / "rec" / "team-a"


@pytest.mark.parametrize("bad", ["AB", "a", "ab", "1team", "has space", "UPPER",
                                 "x" * 33, "../etc", "team/slug"])
def test_invalid_team_slug_rejected(bad):
    with pytest.raises(ValueError):
        paths.profiles_path(bad)


@pytest.mark.parametrize("good", ["team-a", "team-b", "team-a", "abc",
                                  "a1b2c3"])
def test_valid_team_slugs_accepted(good, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MEET_CONFIG_DIR", raising=False)
    # Should not raise.
    paths.profiles_path(good)


def test_none_and_empty_team_are_global(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MEET_PROFILES_PATH", raising=False)
    monkeypatch.delenv("MEET_CONFIG_DIR", raising=False)
    glob = tmp_path / ".config" / "meet" / "speaker_profiles.json"
    assert paths.profiles_path(None) == glob
    assert paths.profiles_path("") == glob
    assert paths.profiles_path("   ") == glob
