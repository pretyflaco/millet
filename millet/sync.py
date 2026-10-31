"""Meeting sync module — push meeting artifacts to a configured Git repository.

Detects whether a completed meeting matches a configured schedule based on
start time, then copies the relevant files into a local clone of the target
repository and pushes to GitHub via gh CLI.

Schedule and repo config: ~/.config/meet/sync_config.json
Repo clone:               ~/.local/share/meet/<repo-name>/
"""

from __future__ import annotations

import copy
import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ─── Credential redaction ────────────────────────────────────────────────────

# Matches the userinfo part of URLs like https://user:token@github.com/...
# so git credentials embedded in a repo_url never leak into exceptions,
# progress messages, or (via the vezir worker) job logs.
_URL_USERINFO_RE = re.compile(r"://[^/@\s]+@")


def _redact(text: str) -> str:
    """Strip credentials (``user:token@``) from URLs embedded in *text*."""
    return _URL_USERINFO_RE.sub("://***@", text)


# ─── Folder slug validation ──────────────────────────────────────────────────

# A meeting folder must be a single safe path segment: it is joined into
# the clone as ``meetings/<date>_<folder>``, and both the CLI
# (--meeting-type) and sync_config.json ("folder") feed it.  Without
# validation a value like ``../../foo`` copies meeting artifacts OUTSIDE
# the clone before git even sees them.  Mirrors the defensive-slug
# convention in millet.paths (_TEAM_SLUG_RE).
_FOLDER_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _validate_folder_slug(folder: str, source: str) -> str:
    """Return *folder* if it is a safe single path segment, else raise.

    ``source`` names where the value came from (CLI flag, config key) so
    the error is actionable.
    """
    if not _FOLDER_SLUG_RE.match(folder or ""):
        raise RuntimeError(
            f"Invalid meeting folder {folder!r} from {source}: must be a "
            "single path segment (letters, digits, '.', '_', '-'; must "
            "start with a letter or digit; max 64 chars)."
        )
    return folder


def _meetings_subdir(team: str | None = None) -> str:
    """Repo subdirectory sessions sync into (default ``"meetings"``).

    Overridable per team via the ``meetings_subdir`` key in
    sync_config.json — e.g. a team whose screenrecording iteration loops
    share a repo with regular meetings can keep a separate
    ``screenrecordings/`` tree instead of mixing into ``meetings/``.
    Validated like a folder slug: a hostile value (``../x``) must not
    escape the clone.
    """
    subdir = load_sync_config(team).get("meetings_subdir") or MEETINGS_SUBDIR
    return _validate_folder_slug(subdir, "meetings_subdir in sync_config.json")

# ─── Paths ────────────────────────────────────────────────────────────────────

SYNC_CONFIG_PATH = Path.home() / ".config" / "meet" / "sync_config.json"
CLONE_BASE_DIR = Path.home() / ".local" / "share" / "meet"
MEETINGS_SUBDIR = "meetings"
# Marker file written into each synced meeting folder recording which session
# produced it.  Used by the collision guard to detect when a different session
# would overwrite an existing folder.  Not pushed as a meeting artifact.
SESSION_ID_MARKER = ".session-id"


# A session id becomes part of a repo path (``<date>_<folder>-<id[-6:]>``).
# ``*.session.json`` is user-supplied on vezir's import path, so the value
# must be a single safe path component — never ``..``, a separator, or a
# control character (security audit 2026-10-30, M-5).
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _safe_session_id(value: object) -> str | None:
    """Return ``value`` if it is a safe single path component, else None."""
    if not isinstance(value, str):
        return None
    sid = value.strip()
    if not sid or sid in (".", "..") or not _SESSION_ID_RE.match(sid):
        return None
    return sid


def _session_id_for(session_dir: Path) -> str | None:
    """Best-effort session id for a session directory.

    Prefers ``session_id`` from ``*.session.json`` (vezir injects it), then
    falls back to the directory name (a bare ULID for vezir sessions).  A
    value that is not a safe single path component is ignored, not used.
    """
    sj = _find_session_json(session_dir)
    if sj is not None:
        try:
            data = json.loads(sj.read_text(encoding="utf-8"))
            sid = _safe_session_id(data.get("session_id"))
            if sid:
                return sid
            if data.get("session_id"):
                log.warning(
                    "%s: ignoring unsafe session_id %r", sj.name, data.get("session_id")
                )
        except Exception:
            pass
    return _safe_session_id(session_dir.name)


def _assert_inside(path: Path, root: Path, what: str) -> None:
    """Raise unless ``path`` resolves to ``root`` or somewhere beneath it.

    Called before every mkdir/copy into the clone so a crafted date, session
    id, or attachment name can never write outside the repository.
    """
    r = root.resolve()
    p = path.resolve()
    if p != r and r not in p.parents:
        raise RuntimeError(f"refusing to write {what} outside the sync repo: {p}")


def _resolve_target_dir(base_dir: Path, session_id: str | None, log_fn) -> Path:
    """Return the folder to sync into, avoiding clobbering a different session.

    If ``base_dir`` doesn't exist, or it already belongs to ``session_id``
    (matching ``.session-id`` marker), use it as-is.  Otherwise append a short
    suffix derived from the session id (``base_dir-<suffix>``) so two distinct
    meetings that resolve to the same folder coexist instead of overwriting.
    """
    if not base_dir.exists():
        return base_dir
    marker = base_dir / SESSION_ID_MARKER
    if session_id and marker.exists():
        try:
            if marker.read_text(encoding="utf-8").strip() == session_id:
                return base_dir  # same session re-syncing — overwrite is fine
        except Exception:
            pass
    # An existing folder with no marker, or a marker for a DIFFERENT session.
    if not session_id:
        return base_dir
    suffix = session_id[-6:]
    candidate = base_dir.parent / f"{base_dir.name}-{suffix}"
    # If that exact disambiguated folder is already ours, reuse it.
    cand_marker = candidate / SESSION_ID_MARKER
    if candidate.exists() and cand_marker.exists():
        try:
            if cand_marker.read_text(encoding="utf-8").strip() == session_id:
                return candidate
        except Exception:
            pass
    if candidate.exists():
        # Extremely unlikely collision on the suffix; fall back to a longer one.
        candidate = base_dir.parent / f"{base_dir.name}-{session_id[-12:]}"
    log_fn(
        f"  Note: {base_dir.name}/ already holds a different meeting; "
        f"using {candidate.name}/ to avoid overwriting it."
    )
    return candidate


def _resolve_sync_config_path(
    team: str | None = None,
    config_path: Path | None = None,
) -> Path:
    """Resolve which sync_config.json to use.

    Precedence:
      1. explicit ``config_path`` (operator/embedder override).
      2. ``team`` → ``~/.config/meet/<team>/sync_config.json`` via
         :mod:`millet.paths`.  Each team gets its own repo target — this
         is what lets one scribe sync different teams to different repos.
      3. the legacy global ``SYNC_CONFIG_PATH``.
    """
    if config_path is not None:
        return config_path
    if team:
        from . import paths
        return paths.sync_config_path(team)
    return SYNC_CONFIG_PATH

# ─── Files to push (by suffix) ───────────────────────────────────────────────

PUSH_SUFFIXES = {".md", ".txt", ".pdf", ".srt", ".json"}
# User-supplied meeting attachments live in this subdirectory of the session
# dir and are pushed verbatim into the same subdirectory of the meeting folder.
# See _collect_attachments.
ATTACHMENTS_SUBDIR = "attachments"
MAX_ATTACHMENTS = 50
MAX_ATTACHMENTS_BYTES = 100 * 1024 * 1024
# Exclude session metadata and large raw files
EXCLUDE_PATTERNS = {
    ".session.json",
    ".summary.meta.json",
    ".meta.json",  # summary/template meta sidecars (superset of the above)
    ".autoid.json",
    ".frontmatter.json",
    ".ffmpeg.log",
}


# ─── Default schedule config ─────────────────────────────────────────────────

DEFAULT_CONFIG: dict[str, Any] = {
    "repo_url": "",
    "meetings": [],
}

EXAMPLE_CONFIG: dict[str, Any] = {
    "repo_url": "https://github.com/yourorg/meetings.git",
    "meetings": [
        {
            "name": "Weekly Sync",
            "folder": "weekly-sync",
            "days": [0],
            "hour_utc": 14,
            "window_minutes": 60,
        },
        {
            "name": "Dev Standup",
            "folder": "dev-standup",
            "days": [0, 1, 2, 3, 4],
            "hour_utc": 8,
            "window_minutes": 60,
        },
    ],
}


def load_sync_config(
    team: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Load sync config from disk. Returns empty defaults if not found.

    When ``team`` is given the per-team config is read; otherwise the
    legacy global file.  ``config_path`` overrides both.
    """
    p = _resolve_sync_config_path(team, config_path)
    if not p.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load sync config: %s — using defaults", exc)
        return copy.deepcopy(DEFAULT_CONFIG)


def save_sync_config(
    config: dict[str, Any],
    team: str | None = None,
    config_path: Path | None = None,
) -> None:
    """Save sync config to disk (per-team when ``team`` is given)."""
    p = _resolve_sync_config_path(team, config_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def is_sync_configured(
    team: str | None = None,
    config_path: Path | None = None,
) -> bool:
    """Return True if a repo_url is configured for sync (for this team)."""
    config = load_sync_config(team, config_path)
    return bool(config.get("repo_url"))


def _normalize_remote(url: str) -> str:
    """Canonical form for comparing two git remote URLs.

    Case-folds, drops a trailing ``/`` and ``.git``, strips ``user:token@``
    credentials (a credential-helper rewrite must not look like a different
    remote), and maps scp-style ``git@host:org/repo`` to ``ssh://git@host/org/repo``.
    """
    u = (url or "").strip()
    m = re.match(r"^([A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+):(?!//)(.+)$", u)
    if m:
        u = f"ssh://{m.group(1)}/{m.group(2)}"
    u = re.sub(r"^([a-z][a-z0-9+.-]*://)[^@/]+@", r"\1", u, flags=re.IGNORECASE)
    u = u.rstrip("/")
    if u.lower().endswith(".git"):
        u = u[:-4]
    return u.rstrip("/").lower()


def _same_remote(actual: str, configured: str) -> bool:
    """True when two remote URLs denote the same repository."""
    a, c = _normalize_remote(actual), _normalize_remote(configured)
    return bool(a) and a == c


def _repo_name_from_url(repo_url: str) -> str:
    """Extract a safe directory name from a Git URL.

    e.g. 'https://github.com/org/my-repo.git' -> 'my-repo'

    The name is sanitized so a hostile ``repo_url`` (e.g. ending in ``..`` or
    containing path separators) cannot make the clone dir escape the clone
    base directory.
    """
    name = repo_url.rstrip("/").rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    # Strip anything that isn't a safe filename char; collapse traversal.
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    name = name.strip(".")  # no leading/trailing dots => no "." / ".."
    return name or "sync-repo"


# Allowed transports for `git clone`: standard network schemes, scp-style
# user@host:path, file:// URLs, and absolute local filesystem paths (used for
# local/NFS bare repos and in tests).  Anything else — notably git's
# option-capable / arbitrary-command transports like ``ext::sh -c ...`` — is
# rejected, as is a repo_url that git would parse as an option (leading '-').
# ``git://`` dropped in 0.21.8 (audit L-11): unauthenticated and unencrypted,
# retired by GitHub in 2022 — no legitimate sync target needs it.
_ALLOWED_REPO_URL_RE = re.compile(
    r"^(?:https://|ssh://|file://|/|[A-Za-z0-9._-]+@[A-Za-z0-9._-]+:)",
)


def _validate_repo_url(repo_url: str) -> None:
    """Reject repo URLs that could smuggle git options or unsafe transports.

    A ``repo_url`` from a provisioned/hostile ``sync_config.json`` is passed to
    ``git clone``; without validation a value like ``--upload-pack=<cmd>`` or
    ``ext::sh -c <cmd>`` yields arbitrary command execution.
    """
    if repo_url.startswith("-"):
        raise RuntimeError(
            f"Refusing repo_url that looks like a git option: {_redact(repo_url)!r}"
        )
    if any(ord(ch) < 0x20 or ch.isspace() for ch in repo_url):
        raise RuntimeError(
            f"Refusing repo_url with whitespace/control characters: {_redact(repo_url)!r}"
        )
    if not _ALLOWED_REPO_URL_RE.match(repo_url):
        raise RuntimeError(
            "Unsupported repo_url scheme. Use https://, ssh://, file://, an "
            "absolute path, or scp-style user@host:path — got: "
            f"{_redact(repo_url)!r}"
        )


def _host_from_url(url: str) -> str | None:
    """Best-effort remote host for the DNS gate; None for local paths."""
    u = (url or "").strip()
    m = re.match(r"^[a-z][a-z0-9+.-]*://(?:[^@/]+@)?([^/:]+)", u, flags=re.IGNORECASE)
    if m:
        host = m.group(1)
        return None if host in ("", "localhost") or u.lower().startswith("file://") else host
    m = re.match(r"^[A-Za-z0-9._-]+@([A-Za-z0-9._-]+):", u)
    if m:
        return m.group(1)
    return None


# Timeouts (audit L-11).  Neither helper had one, so a credential prompt or a
# stalled transfer hung until the vezir worker's 15-minute watchdog killed the
# whole job.  Local git ops are fast; a clone/pull/push of a meetings repo
# that takes longer than ten minutes is a stuck transfer, not a slow one.
_LOCAL_GIT_TIMEOUT = 120
_NETWORK_GIT_TIMEOUT = 600
_DNS_TIMEOUT = 10.0


def _getaddrinfo_with_timeout(host: str, port: int) -> None:
    """Resolve in a child that subprocess.run kills on timeout."""
    import socket
    import sys

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import socket,sys; socket.getaddrinfo(sys.argv[1], int(sys.argv[2]))",
                host,
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_DNS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"DNS resolution for {host} timed out after {_DNS_TIMEOUT}s"
        ) from exc
    if result.returncode:
        raise socket.gaierror(f"DNS resolution for {host} failed: {result.stderr[:200]}")


def _ssh_with_batch_mode(command: str) -> str:
    import shlex

    try:
        parts = shlex.split(command)
    except ValueError:
        return f"{command} -o BatchMode=yes"

    filtered: list[str] = []
    removed = False
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "-o" and i + 1 < len(parts):
            option = parts[i + 1]
            if re.match(r"^batchmode(?:\s|=|$)", option, flags=re.IGNORECASE):
                removed = True
                i += 2
                continue
        elif part.lower().startswith("-obatchmode"):
            removed = True
            i += 1
            continue
        filtered.append(part)
        i += 1

    if not removed:
        return f"{command} -o BatchMode=yes"
    return shlex.join([*filtered, "-o", "BatchMode=yes"])


def _git_env() -> dict[str, str]:
    """Environment for git subprocesses with prompting forcibly disabled."""
    import os
    import shlex

    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    command = env.get("GIT_SSH_COMMAND") or env.get("GIT_SSH") or "ssh"
    if "GIT_SSH_COMMAND" not in env and "GIT_SSH" in env:
        command = shlex.quote(command)
    env["GIT_SSH_COMMAND"] = _ssh_with_batch_mode(command)
    return env


# ─── Schedule matching ────────────────────────────────────────────────────────


@dataclass
class MeetingMatch:
    name: str
    folder: str


@dataclass
class SyncCandidate:
    """A meeting that passed schedule + team-member checks and is ready to sync."""

    match: MeetingMatch
    team_members_found: list[str]  # names of recognized team members in this meeting


def check_sync_candidate(
    session_dir: Path, team: str | None = None
) -> SyncCandidate | None:
    """Check whether a session should be offered for sync.

    Passes two gates:
    1. Schedule match (day + time window)
    2. At least min_team_members recognized team members are present

    Returns a SyncCandidate if both pass, None otherwise.
    """
    match = detect_meeting_type(session_dir, team=team)
    if match is None:
        return None

    config = load_sync_config(team)
    team_members = {m.lower() for m in config.get("team_members", [])}
    min_required = config.get("min_team_members", 2)

    if not team_members:
        # No team list configured — fall back to schedule-only check
        return SyncCandidate(match=match, team_members_found=[])

    # Read confirmed speaker labels from session.json
    session_json = _find_session_json(session_dir)
    if not session_json:
        return None

    try:
        meta = json.loads(session_json.read_text(encoding="utf-8"))
        labels = meta.get("speaker_labels", {})
        # labels values are the human names
        confirmed_names = list(labels.values())
    except Exception:
        return None

    found = [n for n in confirmed_names if n.lower() in team_members]
    if len(found) < min_required:
        log.debug(
            "Skipping sync for %s: only %d/%d team members found (%s)",
            session_dir.name,
            len(found),
            min_required,
            found,
        )
        return None

    return SyncCandidate(match=match, team_members_found=found)


def _meeting_slug(text: str) -> str:
    """Lowercase hyphen-slug for comparing a title against a schedule folder.

    "Dev Standup Daily" -> "dev-standup-daily"; "post-scrum" -> "post-scrum".
    """
    import re as _re
    slug = _re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug


def detect_meeting_type(
    session_dir: Path, team: str | None = None
) -> MeetingMatch | None:
    """Check if a session matches any configured scheduled meeting.

    Reads ``started_at`` (and optionally ``title``) from session.json and
    compares against the schedule.

    Title-aware matching: a session that carries a non-empty ``title`` only
    auto-matches a scheduled meeting whose name/folder slug equals the title's
    slug.  This stops an ad-hoc meeting recorded *during* a schedule window
    (e.g. a "post-scrum" at 09:03, inside the 06:30-09:30 standup window) from
    being misfiled as that scheduled meeting.  Untitled sessions keep the
    pure time-window behavior (back-compat).

    Returns a MeetingMatch if matched, None otherwise.
    """
    session_json = _find_session_json(session_dir)
    if not session_json:
        return None

    try:
        meta = json.loads(session_json.read_text(encoding="utf-8"))
        started_at_str = meta.get("started_at")
        if not started_at_str:
            return None
        title = (meta.get("title") or "").strip()
        title_slug = _meeting_slug(title) if title else ""
        # Parse ISO format — may or may not have timezone info.
        # Naive datetimes are treated as local time; astimezone() applies
        # the correct DST-aware local offset (the previous manual
        # time.timezone arithmetic used the non-DST offset year-round,
        # shifting every summer-time meeting by an hour and silently
        # missing the schedule window).
        started_at = datetime.fromisoformat(started_at_str)
        started_at = started_at.astimezone(timezone.utc)
    except Exception as exc:
        log.debug("Could not parse session start time: %s", exc)
        return None

    config = load_sync_config(team)
    weekday = started_at.weekday()  # 0=Monday
    hour = started_at.hour
    minute = started_at.minute
    meeting_minutes = hour * 60 + minute

    for m in config.get("meetings", []):
        if weekday not in m.get("days", []):
            continue
        scheduled_minutes = m["hour_utc"] * 60
        window = m.get("window_minutes", 60)
        # Wrap around midnight: a 23:40 UTC session is 20 minutes from an
        # hour_utc=0 schedule, not 1420.
        diff = abs(meeting_minutes - scheduled_minutes)
        diff = min(diff, 1440 - diff)
        if diff > window:
            continue
        # Title-aware guard: a titled session only matches a schedule whose
        # name/folder slug equals the title's slug.  An ad-hoc titled meeting
        # in the window does NOT inherit the scheduled folder.
        if title_slug:
            sched_slugs = {
                _meeting_slug(m.get("folder", "")),
                _meeting_slug(m.get("name", "")),
            }
            if title_slug not in sched_slugs:
                log.debug(
                    "session title '%s' (slug '%s') in window of '%s' but "
                    "does not match its slug; not auto-filing as scheduled",
                    title, title_slug, m.get("folder"),
                )
                continue
        return MeetingMatch(name=m["name"], folder=m["folder"])

    return None


def _find_session_json(session_dir: Path) -> Path | None:
    """Find the session.json file in a session directory."""
    matches = list(session_dir.glob("*.session.json"))
    return matches[0] if matches else None


# ─── Repo management ──────────────────────────────────────────────────────────


def _run(
    cmd: list[str], cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Run a command, raising on non-zero exit if check=True."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            timeout=_LOCAL_GIT_TIMEOUT, env=_git_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            _redact(f"Command timed out after {_LOCAL_GIT_TIMEOUT}s: {' '.join(cmd)}")
        ) from exc
    if check and result.returncode != 0:
        # _redact: never leak repo_url credentials into exception text
        # (which the vezir worker captures into job logs).
        raise RuntimeError(
            _redact(
                f"Command failed: {' '.join(cmd)}\n"
                f"stdout: {result.stdout.strip()}\n"
                f"stderr: {result.stderr.strip()}"
            )
        )
    return result


def _is_dns_error(result: subprocess.CompletedProcess) -> bool:
    """Return True if the failed command's stderr indicates a DNS resolution failure."""
    stderr = (result.stderr or "").lower()
    return any(
        needle in stderr
        for needle in ("could not resolve host", "name or service not known",
                       "temporary failure in name resolution")
    )


def _run_network(
    cmd: list[str], cwd: Path | None = None, retries: int = 5, delay: float = 4.0
) -> subprocess.CompletedProcess:
    """Run a git network command (clone/pull/push) with DNS-gated retries.

    Retries up to *retries* times when the failure looks like a transient DNS
    resolution error.  Raises RuntimeError on a non-DNS failure or after
    exhausting retries.
    """
    import socket
    import time

    # Gate on the remote's actual host (audit L-11: this used to be hard-coded
    # to github.com, so non-GitHub remotes got no gate and GitHub outages
    # gated unrelated remotes).  Clone carries the URL in argv; pull/push
    # ask the clone for its origin.  Local/unknown remotes skip the gate.
    host: str | None = None
    url_arg = next((a for a in cmd[2:] if "://" in a or re.match(r"^[^/]+@[^/]+:", a)), None)
    if url_arg:
        host = _host_from_url(url_arg)
    elif cwd is not None:
        try:
            origin = subprocess.run(
                ["git", "remote", "get-url", "origin"], cwd=cwd,
                capture_output=True, text=True, timeout=_LOCAL_GIT_TIMEOUT,
            ).stdout.strip()
            host = _host_from_url(origin)
        except Exception:
            host = None

    last_result: subprocess.CompletedProcess | None = None
    for attempt in range(1, retries + 1):
        # Gate on DNS: wait until the git host resolves before burning a try.
        if host:
            for _ in range(retries):
                try:
                    _getaddrinfo_with_timeout(host, 443)
                    break
                except (socket.gaierror, TimeoutError):
                    log.debug("DNS for %s not resolving, waiting %.0fs", host, delay)
                    time.sleep(delay)
            else:
                log.warning("DNS for %s did not resolve after %d waits", host, retries)

        try:
            result = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True,
                timeout=_NETWORK_GIT_TIMEOUT, env=_git_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                _redact(f"Command timed out after {_NETWORK_GIT_TIMEOUT}s: {' '.join(cmd)}")
            ) from exc
        if result.returncode == 0:
            return result
        last_result = result
        if _is_dns_error(result) and attempt < retries:
            log.warning(
                "Transient DNS error on attempt %d/%d for %s; retrying in %.0fs",
                attempt, retries, " ".join(cmd[:3]), delay,
            )
            time.sleep(delay)
            continue
        # Non-DNS failure or last attempt — raise immediately.
        break

    assert last_result is not None
    raise RuntimeError(
        _redact(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout: {last_result.stdout.strip()}\n"
            f"stderr: {last_result.stderr.strip()}"
        )
    )


def _current_branch_ahead_count(repo: Path) -> int:
    """Return how many local commits the current branch is ahead of upstream."""
    result = _run(
        ["git", "rev-list", "--count", "@{upstream}..HEAD"],
        cwd=repo,
        check=False,
    )
    if result.returncode != 0:
        return 0
    try:
        return int(result.stdout.strip() or "0")
    except ValueError:
        return 0


def _clone_dir_for(repo_url: str, team: str | None) -> Path:
    """Local clone dir for a repo, namespaced by team to avoid collisions.

    Two teams can legitimately point at different repos that share a
    name (``org-a/meetings`` vs ``org-b/meetings``); putting each team's
    clones under ``<base>/<team>/`` keeps them from clobbering one
    another.  Teamless (global) syncs keep the flat layout.
    """
    name = _repo_name_from_url(repo_url)
    return (CLONE_BASE_DIR / team / name) if team else (CLONE_BASE_DIR / name)


def _get_clone_dir(team: str | None = None) -> Path:
    """Return the local clone directory for the configured repo."""
    config = load_sync_config(team)
    repo_url = config.get("repo_url", "")
    if not repo_url:
        raise RuntimeError(
            "No repo_url configured for sync. "
            f"Set it in {_resolve_sync_config_path(team)} or run: "
            "meet sync --init-config"
        )
    return _clone_dir_for(repo_url, team)


def _ensure_local_gitignore(clone_dir: Path) -> None:
    """Add the collision marker to the clone's local-only ignore list.

    Writes ``.session-id`` into ``<clone>/.git/info/exclude`` rather than a
    tracked ``.gitignore`` so the markers stay strictly local: they are never
    committed/pushed and never show up in ``git status --porcelain`` (which
    would otherwise trip the uncommitted-changes guard).  Idempotent.
    """
    exclude = clone_dir / ".git" / "info" / "exclude"
    try:
        if exclude.exists():
            existing = exclude.read_text(encoding="utf-8")
            lines = {ln.strip() for ln in existing.splitlines()}
            if SESSION_ID_MARKER in lines:
                return
            sep = "" if existing.endswith("\n") or not existing else "\n"
            exclude.write_text(
                f"{existing}{sep}{SESSION_ID_MARKER}\n", encoding="utf-8"
            )
        else:
            exclude.parent.mkdir(parents=True, exist_ok=True)
            exclude.write_text(f"{SESSION_ID_MARKER}\n", encoding="utf-8")
    except Exception as exc:
        # Non-fatal: worst case the marker shows as untracked.  Log and move on.
        log.debug("Could not update local gitignore at %s: %s", exclude, exc)


def ensure_repo_cloned(progress_callback=None, team: str | None = None) -> Path:
    """Ensure the configured repo is cloned locally. Clone if not present, pull if it is.

    Returns the path to the local clone.
    """
    config = load_sync_config(team)
    repo_url = config.get("repo_url", "")
    if not repo_url:
        raise RuntimeError(
            "No repo_url configured for sync. "
            f"Set it in {_resolve_sync_config_path(team)} or run: "
            "meet sync --init-config"
        )
    _validate_repo_url(repo_url)

    clone_dir = _clone_dir_for(repo_url, team)

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            log.info(msg)

    if not clone_dir.exists():
        _log(f"Cloning {_redact(repo_url)}...")
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_network(["git", "clone", "--", repo_url, str(clone_dir)])
        _log("Clone complete.")
        # New clone: register the local-only ignore for collision markers.
        _ensure_local_gitignore(clone_dir)
    else:
        # The clone must belong to THIS remote.  Clone dirs are keyed by repo
        # basename (per team when --team is given), so ``orgA/meetings`` and
        # ``orgB/meetings`` — or an old and a new remote with the same name —
        # would otherwise share one clone and push each other's meetings to
        # the wrong repository (security audit 2026-10-30, H-3).  Refuse
        # rather than re-clone: a clone with the wrong origin may hold
        # unpushed commits an operator wants to inspect.
        actual = _run(
            ["git", "remote", "get-url", "origin"], cwd=clone_dir, check=False
        ).stdout.strip()
        if not _same_remote(actual, repo_url):
            raise RuntimeError(
                f"Sync clone at {clone_dir} points at {_redact(actual) or '<none>'}, "
                f"not the configured {_redact(repo_url)}.  Move or delete the "
                "clone (or pass --team so each team gets its own) before syncing."
            )

        # Keep the local-only collision markers out of git's sight so they
        # neither get committed nor trip the uncommitted-changes guard.
        _ensure_local_gitignore(clone_dir)

        status = _run(["git", "status", "--porcelain"], cwd=clone_dir)
        if status.stdout.strip():
            raise RuntimeError(
                f"Sync repo has uncommitted changes at {clone_dir}. "
                "Commit, stash, or clean them before syncing."
            )

        # Pull latest, rebasing any previously-created local meeting commits.
        # On failure, abort any half-applied rebase: a conflicting pull
        # leaves .git/rebase-merge behind, which would wedge every future
        # sync at the uncommitted-changes guard above with an error that
        # never mentions the rebase.
        try:
            _run_network(["git", "pull", "--rebase"], cwd=clone_dir)
        except Exception:
            _run(["git", "rebase", "--abort"], cwd=clone_dir, check=False)
            raise

    return clone_dir


def _collect_files(session_dir: Path) -> list[tuple[Path, str]]:
    """Return list of (source_path, dest_filename) pairs to push from a session directory.

    Files are renamed to descriptive names (summary.md, transcript.txt, etc.)
    since the date lives in the parent folder name.

    Collision-safe: if two source files map to the same descriptive name
    (e.g. a stray ``notes.md`` next to the real summary), later files keep
    their original names instead of silently clobbering the earlier one in
    the pushed repo.
    """
    result = []
    used_names: set[str] = set()
    for f in sorted(session_dir.iterdir()):
        if not f.is_file():
            continue
        if any(f.name.endswith(pat) for pat in EXCLUDE_PATTERNS):
            continue
        if f.suffix not in PUSH_SUFFIXES:
            continue

        # Map to a descriptive destination filename
        if f.name.endswith(".summary.md"):
            dest_name = "summary.md"
        elif ".summary." in f.name and f.name.endswith(".md"):
            # Additional per-language summary: <base>.summary.<lang>.md ->
            # summary.<lang>.md (kept distinct from the primary summary.md).
            stem = f.name[: -len(".md")]
            lang = stem.rsplit(".summary.", 1)[-1]
            dest_name = f"summary.{lang}.md" if lang else f.name
        elif f.name.endswith(".md") and "." in f.name[: -len(".md")]:
            # Template summary artifact: <base>.<template>.md ->
            # <template>.md (e.g. iteration-plan.md).  Dotted stray files
            # get the same treatment — a descriptive name beats summary.md,
            # which the real summary already owns.
            stem = f.name[: -len(".md")]
            tag = stem.rsplit(".", 1)[-1]
            dest_name = f"{tag}.md" if tag else f.name
        elif f.suffix == ".md":
            dest_name = "summary.md"
        elif f.suffix == ".txt":
            dest_name = "transcript.txt"
        elif f.suffix == ".pdf":
            dest_name = "transcript.pdf"
        elif f.suffix == ".srt":
            dest_name = "transcript.srt"
        elif f.suffix == ".json" and ".clips." in f.name:
            # video-clipping cut list: <base>.clips[.<lang>].json ->
            # clips[.<lang>].json.  Must precede the transcript mapping:
            # "<base>.clips.json" sorts before "<base>.json" and would
            # otherwise claim transcript.json.
            tail = f.name[: -len(".json")].split(".clips", 1)[1]  # "" or ".de"
            dest_name = f"clips{tail}.json"
        elif f.suffix == ".json":
            dest_name = "transcript.json"
        else:
            dest_name = f.name  # fallback: keep original name

        if dest_name in used_names:
            log.warning(
                "sync: %s also maps to %s — keeping its original name to "
                "avoid overwriting", f.name, dest_name,
            )
            dest_name = f.name
        used_names.add(dest_name)

        result.append((f, dest_name))

    result.extend(_collect_attachments(session_dir))
    return result


def _collect_attachments(session_dir: Path) -> list[tuple[Path, str]]:
    """Return (source, ``attachments/<name>``) pairs for user-supplied files.

    Files a user dropped into ``<session>/attachments/`` — slides, agendas,
    screenshots — are pushed verbatim: neither ``PUSH_SUFFIXES`` nor the
    descriptive-rename map above applies to them.  Both would be wrong here.
    The rename map keys off the suffix alone, so an attached ``slides.pdf``
    would land as ``transcript.pdf``, and the suffix allowlist would drop
    every image, office document and video.

    Symlinks are skipped: the pairs returned here are copied into a git
    clone, and a link could point anywhere on the host.  Count and total
    size are capped so a mis-aimed folder can't bloat the archive repo.
    """
    adir = session_dir / ATTACHMENTS_SUBDIR
    if not adir.is_dir():
        return []

    result: list[tuple[Path, str]] = []
    total = 0
    for f in sorted(adir.iterdir()):
        if f.is_symlink() or not f.is_file() or f.name.startswith("."):
            continue
        if len(result) >= MAX_ATTACHMENTS:
            log.warning(
                "sync: more than %d attachments in %s — pushing the first %d only",
                MAX_ATTACHMENTS, adir, MAX_ATTACHMENTS,
            )
            break
        size = f.stat().st_size
        if total + size > MAX_ATTACHMENTS_BYTES:
            log.warning(
                "sync: attachments in %s exceed %d bytes — skipping %s and the rest",
                adir, MAX_ATTACHMENTS_BYTES, f.name,
            )
            break
        total += size
        result.append((f, f"{ATTACHMENTS_SUBDIR}/{f.name}"))
    return result


def _date_from_session(session_dir: Path) -> str:
    """Extract YYYY-MM-DD from the session folder name or session.json."""
    # Try parsing from folder name: meeting-YYYYMMDD-HHMMSS_*
    name = session_dir.name
    parts = name.split("-")
    if len(parts) >= 2 and len(parts[1]) == 8 and parts[1].isdigit():
        d = parts[1]
        return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    # Fallback: read from session.json.  Parse, don't slice: the file is
    # user-supplied on vezir's import path and the value feeds a repo path
    # (``../../../x`` is also ten characters — audit M-5).
    sj = _find_session_json(session_dir)
    if sj:
        try:
            meta = json.loads(sj.read_text(encoding="utf-8"))
            started = meta.get("started_at")
            if isinstance(started, str) and started.strip():
                return datetime.fromisoformat(
                    started.strip().replace("Z", "+00:00")
                ).strftime("%Y-%m-%d")
        except Exception:
            log.warning("%s: ignoring unparseable started_at", sj.name)

    return datetime.now().strftime("%Y-%m-%d")


def _ensure_readme(meetings_dir: Path, team: str | None = None) -> None:
    """Create meetings/README.md from the sync config if it doesn't exist."""
    readme = meetings_dir / "README.md"
    if readme.exists():
        return

    config = load_sync_config(team)
    meetings = config.get("meetings", [])
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    lines = [
        "# Meeting Notes\n\n",
        "Automatically published meeting transcripts, summaries, and PDFs.\n\n",
    ]

    if meetings:
        lines.append("## Recurring Meetings\n\n")
        lines.append("| Folder | Meeting | Schedule |\n")
        lines.append("|--------|---------|----------|\n")
        for m in meetings:
            days = ", ".join(day_names[d] for d in m.get("days", []))
            lines.append(
                f"| `{m['folder']}/` | {m['name']} | {days} {m['hour_utc']:02d}:00 UTC |\n"
            )
        lines.append("\n")

    lines.append(
        "## File Types\n\n"
        "| Extension | Contents |\n"
        "|-----------|----------|\n"
        "| `.md` | AI-generated meeting summary |\n"
        "| `.txt` | Full transcript with timestamps |\n"
        "| `.pdf` | Formatted PDF |\n"
        "| `.srt` | Subtitle/caption file |\n"
        "| `.json` | Full transcript data (programmatic access) |\n"
    )

    readme.write_text("".join(lines), encoding="utf-8")


def sync_session(
    session_dir: Path,
    meeting_type: MeetingMatch,
    progress_callback=None,
    team: str | None = None,
) -> list[Path]:
    """Copy session files into the configured repo and push to GitHub.

    Args:
        session_dir: Path to the local meeting session directory.
        meeting_type: MeetingMatch identifying folder and name.
        progress_callback: Optional callable(str) for status messages.

    Returns:
        List of files that were pushed.

    Raises:
        RuntimeError: If git operations fail or sync is not configured.
    """

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            log.info(msg)

    # Defense in depth: the folder feeds a path join below; reject
    # traversal-capable values whether they came from --meeting-type or a
    # hostile/corrupt sync_config.json.
    _validate_folder_slug(meeting_type.folder, "meeting type/config folder")

    repo = ensure_repo_cloned(progress_callback=progress_callback, team=team)

    subdir = _meetings_subdir(team)
    sync_root = repo / subdir
    _assert_inside(sync_root, repo, "sync root")
    date_str = _date_from_session(session_dir)
    session_id = _session_id_for(session_dir)
    # Each meeting gets its own folder: date first so folders sort chronologically.
    # Collision guard: if a folder for this date+type already holds a DIFFERENT
    # session, don't overwrite it — disambiguate with a short suffix.  Prevents
    # two meetings that resolve to the same folder (e.g. two ad-hoc meetings in
    # one schedule window) from clobbering each other.
    base_dir = repo / subdir / f"{date_str}_{meeting_type.folder}"
    target_dir = _resolve_target_dir(base_dir, session_id, _log)
    # Containment is asserted BEFORE the first write.  Every component above
    # is validated, but this is the one check that holds regardless of how
    # the inputs were produced (audit M-5).
    _assert_inside(target_dir, sync_root, "meeting folder")
    target_dir.mkdir(parents=True, exist_ok=True)
    if session_id:
        (target_dir / SESSION_ID_MARKER).write_text(session_id + "\n", encoding="utf-8")

    # Ensure README exists
    _ensure_readme(sync_root, team=team)

    # Collect and copy files with descriptive names
    source_files = _collect_files(session_dir)
    if not source_files:
        raise RuntimeError(f"No pushable files found in {session_dir}")

    copied: list[Path] = []
    for src, dest_name in source_files:
        dest = target_dir / dest_name
        # dest_name carries an "attachments/" prefix for attached files.
        _assert_inside(dest, target_dir, f"artifact {dest_name!r}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(dest)
        _log(f"  Staged: {dest.relative_to(repo)}")

    # Also stage the README
    _run(
        ["git", "add", str((sync_root / "README.md").relative_to(repo))],
        cwd=repo,
    )

    # Stage all copied files
    rel_paths = [str(p.relative_to(repo)) for p in copied]
    _run(["git", "add", *rel_paths], cwd=repo)

    # Check if there's anything to commit
    status = _run(["git", "status", "--porcelain"], cwd=repo)
    if not status.stdout.strip():
        ahead_count = _current_branch_ahead_count(repo)
        if ahead_count:
            _log(f"  No new file changes, but {ahead_count} local commit(s) need pushing.")
            _log("  Pushing...")
            _run_network(["git", "push"], cwd=repo)
            _log(f"  Pushed {ahead_count} existing commit(s).")
            return copied

        _log("  Nothing to commit — already up to date.")
        return copied

    # Commit
    commit_msg = f"meetings: add {meeting_type.name} {date_str}"
    _run(["git", "commit", "-m", commit_msg], cwd=repo)
    _log(f"  Committed: {commit_msg}")

    # Push
    _log("  Pushing...")
    _run_network(["git", "push"], cwd=repo)
    _log(f"  Pushed {len(copied)} file(s).")

    return copied


def maybe_sync_session(
    session_dir: Path,
    progress_callback=None,
    team: str | None = None,
) -> MeetingMatch | None:
    """Detect, validate (team members), and sync a scheduled meeting.

    Used by the CLI `meet sync` command. The GUI uses check_sync_candidate +
    sync_session directly so it can interpose a confirmation dialog in between.

    Returns the MeetingMatch if synced, None if skipped.
    """
    if not is_sync_configured(team):
        return None

    candidate = check_sync_candidate(session_dir, team=team)
    if candidate is None:
        return None

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    _log(f"Scheduled meeting detected: {candidate.match.name} — syncing...")
    try:
        sync_session(
            session_dir, candidate.match,
            progress_callback=progress_callback, team=team,
        )
    except Exception as exc:
        _log(f"Sync failed (meeting saved locally): {exc}")
        log.exception("Sync failed for %s", session_dir)
        # A failed push must not be reported as synced: callers (GUI
        # auto-sync) treat a non-None return as success.
        return None

    return candidate.match
