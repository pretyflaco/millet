"""Central path resolution for millet (a.k.a. meetscribe).

Historically every module hardcoded its own ``~/.config/meet/...`` and
``~/meet-recordings`` constants.  This module consolidates that logic in
one place and adds an optional ``team`` dimension so a scribe recording
for multiple teams can keep voiceprints, sync config, and recordings
separated locally.

Team-aware layout (when ``team`` is given)::

    ~/.config/meet/<team>/speaker_profiles.json
    ~/.config/meet/<team>/sync_config.json
    ~/meet-recordings/<team>/

Global layout (when ``team`` is None — unchanged, back-compatible)::

    ~/.config/meet/speaker_profiles.json
    ~/.config/meet/sync_config.json
    ~/meet-recordings/

Resolution is evaluated at call time (not import time) so callers that
override ``$HOME`` or the env vars between import and use get the right
path.

Environment overrides (no-team escape hatches, preserved for
back-compat):

* ``MEET_PROFILES_PATH`` — absolute path to the profiles DB.  When set
  AND no ``team`` is requested, it wins.  When a ``team`` is requested
  the env override is ignored (an explicit team is more specific than a
  process-wide override).
* ``MEET_CONFIG_DIR`` — root of the config tree (default
  ``~/.config/meet``).
* ``MEET_RECORDINGS_DIR`` — root of the recordings tree (default
  ``~/meet-recordings``).
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path

# ── env-var rename (meetscribe/meet → millet), v0.9.1 ────────────────────────
#
# The pipeline is being renamed meetscribe/meet → millet.  Env vars get a
# new ``MILLET_*`` spelling; the legacy ``MEETSCRIBE_*`` / ``MEET_*``
# names still work for one minor release with a one-time
# DeprecationWarning.  ``getenv_renamed`` centralizes that fallback.

_RENAME_WARNED: set[str] = set()


def getenv_renamed(
    new_name: str,
    *legacy_names: str,
    default: str | None = None,
) -> str | None:
    """Read ``new_name`` from the env, falling back to legacy aliases.

    The first set (in order: new, then each legacy) wins.  Reading via a
    legacy alias emits a one-time ``DeprecationWarning`` naming the new
    variable.  Returns ``default`` when nothing is set.
    """
    val = os.environ.get(new_name)
    if val is not None:
        return val
    for legacy in legacy_names:
        val = os.environ.get(legacy)
        if val is not None:
            if legacy not in _RENAME_WARNED:
                _RENAME_WARNED.add(legacy)
                warnings.warn(
                    f"environment variable {legacy} is deprecated; "
                    f"use {new_name} instead (the meetscribe→millet rename; "
                    "the legacy name is honored for one more release).",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return val
    return default

# A team slug mirrors vezir's contract: 3-32 chars, lowercase alnum +
# hyphen, must start with a letter.  We validate defensively so a bad
# value can never escape its directory.
_TEAM_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]{2,31}$")

_PROFILES_FILENAME = "speaker_profiles.json"
_SYNC_FILENAME = "sync_config.json"
_VOCABULARY_FILENAME = "vocabulary.json"


def _validate_team(team: str | None) -> str | None:
    """Return the team slug if valid, or None.  Raise on malformed input."""
    if team is None:
        return None
    team = team.strip()
    if not team:
        return None
    if not _TEAM_SLUG_RE.match(team):
        raise ValueError(
            f"invalid team slug {team!r}: must be 3-32 chars, lowercase "
            "letters/digits/hyphens, starting with a letter"
        )
    return team


def config_dir(team: str | None = None) -> Path:
    """Root config directory, optionally scoped to a team.

    ``~/.config/meet`` (or ``$MEET_CONFIG_DIR``), plus ``/<team>`` when a
    team is given.
    """
    team = _validate_team(team)
    root = getenv_renamed("MILLET_CONFIG_DIR", "MEET_CONFIG_DIR", default="")
    base = Path(root).expanduser() if root else Path.home() / ".config" / "meet"
    return base / team if team else base


def profiles_path(team: str | None = None) -> Path:
    """Path to the speaker-profiles DB.

    Precedence:
      1. ``team`` given → ``<config_dir>/<team>/speaker_profiles.json``
         (env override intentionally ignored — explicit team wins).
      2. ``$MEET_PROFILES_PATH`` if set.
      3. ``<config_dir>/speaker_profiles.json``.
    """
    team = _validate_team(team)
    if team:
        return config_dir(team) / _PROFILES_FILENAME
    env = getenv_renamed("MILLET_PROFILES_PATH", "MEET_PROFILES_PATH", default="")
    if env:
        return Path(env).expanduser()
    return config_dir() / _PROFILES_FILENAME


def sync_config_path(team: str | None = None) -> Path:
    """Path to the sync_config.json, optionally scoped to a team."""
    return config_dir(team) / _SYNC_FILENAME


def vocabulary_path(team: str | None = None) -> Path:
    """Path to the custom vocabulary/glossary file, optionally scoped to a team.

    Precedence:
      1. ``team`` given → ``<config_dir>/<team>/vocabulary.json``
         (env override intentionally ignored — explicit team wins).
      2. ``$MILLET_VOCABULARY_PATH`` if set.
      3. ``<config_dir>/vocabulary.json``.
    """
    team = _validate_team(team)
    if team:
        return config_dir(team) / _VOCABULARY_FILENAME
    env = getenv_renamed("MILLET_VOCABULARY_PATH", "MEET_VOCABULARY_PATH", default="")
    if env:
        return Path(env).expanduser()
    return config_dir() / _VOCABULARY_FILENAME


def recordings_dir(team: str | None = None) -> Path:
    """Root recordings directory, optionally scoped to a team.

    ``~/meet-recordings`` (or ``$MEET_RECORDINGS_DIR``), plus ``/<team>``
    when a team is given.
    """
    team = _validate_team(team)
    root = getenv_renamed("MILLET_RECORDINGS_DIR", "MEET_RECORDINGS_DIR", default="")
    base = Path(root).expanduser() if root else Path.home() / "meet-recordings"
    return base / team if team else base
