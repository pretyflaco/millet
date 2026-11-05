"""millet sync command."""
from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.argument("session_dirs", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Push even if the meeting doesn't match a scheduled meeting",
)
@click.option(
    "--meeting-type",
    type=str,
    default=None,
    help="Override meeting type folder (e.g. 'weekly-sync', 'dev-standup')",
)
@click.option(
    "--list-schedule",
    is_flag=True,
    default=False,
    help="Show the current sync schedule and exit",
)
@click.option(
    "--init-config",
    is_flag=True,
    default=False,
    help="Create an example sync config and exit",
)
@click.option(
    "--team",
    type=str,
    default=None,
    help="Scope sync to a team: use ~/.config/meet/<team>/sync_config.json "
         "and a team-namespaced repo clone. Lets one scribe sync different "
         "teams to different repos.",
)
def sync(session_dirs, force, meeting_type, list_schedule, init_config, team):
    """Sync meeting artifacts to a configured Git repository.

    Detects whether each session matches a configured meeting schedule and
    pushes the transcript, summary, PDF, SRT, and JSON to the team repo.
    Audio files and internal metadata are excluded.

    \b
    Setup:
        millet sync --init-config        # create example config
        # edit ~/.config/meet/sync_config.json with your repo URL and schedule

    \b
    Per-team setup:
        millet sync --team myteam --init-config
        # edit ~/.config/meet/myteam/sync_config.json

    \b
    Examples:
        millet sync ~/meet-recordings/meeting-20260330-170216_WeeklySync
        millet sync --force --meeting-type weekly-sync ~/meet-recordings/meeting-20260330-*
        millet sync --list-schedule
        millet sync --team myteam ~/meet-recordings/myteam/meeting-*
    """
    from millet.sync import (
        EXAMPLE_CONFIG,
        MeetingMatch,
        _resolve_sync_config_path,
        _validate_folder_slug,
        detect_meeting_type,
        is_sync_configured,
        load_sync_config,
        save_sync_config,
        sync_session,
    )

    if meeting_type:
        # Fail fast with a friendly message before touching any session:
        # the folder becomes a path segment inside the sync repo, so
        # traversal-capable values ('../..', absolute paths) are rejected.
        try:
            _validate_folder_slug(meeting_type, "--meeting-type")
        except RuntimeError as exc:
            click.echo(f"Error: {exc}", err=True)
            raise SystemExit(1) from None

    config_path = _resolve_sync_config_path(team)

    if init_config:
        if config_path.exists():
            click.echo(f"Config already exists: {config_path}")
            click.echo("Edit it manually or delete it to regenerate.")
        else:
            save_sync_config(EXAMPLE_CONFIG, team)
            click.echo(f"Example config created: {config_path}")
            click.echo("Edit it with your repo URL and meeting schedule.")
        return

    if list_schedule:
        config = load_sync_config(team)
        repo_url = config.get("repo_url", "")
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        if repo_url:
            click.echo(f"Sync repo: {repo_url}")
        else:
            click.echo("Sync repo: (not configured)")
        click.echo()

        meetings = config.get("meetings", [])
        if not meetings:
            click.echo("No meetings configured.")
            click.echo(f"Edit {config_path} to add your schedule.")
        else:
            click.echo("Meeting schedule:")
            click.echo()
            for m in meetings:
                days = ", ".join(day_names[d] for d in m.get("days", []))
                click.echo(f"  {m['name']}")
                click.echo(f"    folder:  meetings/{m['folder']}/")
                click.echo(f"    days:    {days}")
                click.echo(
                    f"    time:    {m['hour_utc']:02d}:00 UTC ±{m.get('window_minutes', 60)} min"
                )
                click.echo()
        return

    if not session_dirs:
        click.echo("Error: provide at least one session directory", err=True)
        raise SystemExit(1)

    if not is_sync_configured(team):
        click.echo(
            "Error: sync not configured. Run 'millet sync --init-config' to get started.",
            err=True,
        )
        raise SystemExit(1)

    had_error = False
    for session_dir in session_dirs:
        session_path = Path(session_dir)
        click.echo(f"Syncing: {session_path.name}")

        if meeting_type:
            match = MeetingMatch(name=meeting_type, folder=meeting_type)
        else:
            match = detect_meeting_type(session_path, team=team)

        if match is None and not force:
            click.echo(
                "  Skipped: not a scheduled meeting "
                "(use --force to push anyway, --meeting-type to specify type)",
                err=True,
            )
            continue

        if match is None and force:
            click.echo("  Warning: no schedule match, using 'other' folder", err=True)
            match = MeetingMatch(name="Meeting", folder="other")

        try:
            files = sync_session(
                session_path,
                match,
                progress_callback=lambda msg: click.echo(msg),
                team=team,
            )
            click.echo(f"  Done: {len(files)} file(s) pushed as {match.folder}/")
        except Exception as exc:
            # A failure here (e.g. `git push` rejected) must NOT exit 0 — that
            # made vezir rely on brittle log-string scraping to notice failed
            # syncs.  Record it and propagate a non-zero exit after the loop.
            click.echo(f"  Error: {exc}", err=True)
            had_error = True
        click.echo()

    if had_error:
        raise SystemExit(1)
