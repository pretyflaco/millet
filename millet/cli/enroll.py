"""millet enroll command."""
from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.argument("session_dirs", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--list",
    "list_profiles",
    is_flag=True,
    default=False,
    help="List enrolled speaker profiles and exit",
)
@click.option(
    "--team",
    type=str,
    default=None,
    help="Enroll into this team's voiceprint DB "
         "(~/.config/meet/<team>/speaker_profiles.json) instead of the "
         "global one.",
)
def enroll(session_dirs, list_profiles, team):
    """Enroll speaker voice profiles from labeled session directories.

    Extracts voice embeddings from sessions that already have speaker labels
    (set via 'millet label') and stores them in ~/.config/meet/speaker_profiles.json.
    Future meetings will automatically recognize these speakers.

    \b
    Examples:
        millet enroll ~/meet-recordings/meeting-20260330-170216_WeeklySync
        millet enroll ~/meet-recordings/meeting-20260330-*
        millet enroll --list
        millet enroll --team myteam ~/meet-recordings/myteam/meeting-*
    """
    from millet import paths
    from millet.voiceprint import enroll_session, load_profiles

    # Resolve the team's DB path once; None => global default.
    team_profiles_path = paths.profiles_path(team) if team else None
    display_path = team_profiles_path or paths.profiles_path()

    if list_profiles:
        profiles = load_profiles(profiles_path=team_profiles_path)
        if not profiles:
            click.echo("No speaker profiles enrolled yet.")
            click.echo("  Run: millet enroll <session_dir>")
            return
        click.echo(f"Enrolled speaker profiles ({display_path}):")
        click.echo()
        click.echo(f"  {'Name':<20} {'Sessions'}")
        click.echo(f"  {'----':<20} {'--------'}")
        for name, profile in sorted(profiles.items()):
            click.echo(f"  {name:<20} {profile.n_sessions}")
        return

    if not session_dirs:
        click.echo(
            "Error: provide at least one session directory, or use --list", err=True
        )
        raise SystemExit(1)

    total_enrolled = 0

    for session_dir in session_dirs:
        session_path = Path(session_dir)
        click.echo(f"Enrolling: {session_path.name}")

        try:
            status = enroll_session(
                session_path,
                progress_callback=lambda msg: click.echo(msg),
                profiles_path=team_profiles_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            click.echo(f"  Skipped: {exc}", err=True)
            continue
        except Exception as exc:
            click.echo(f"  Error: {exc}", err=True)
            continue

        enrolled = sum(1 for ok in status.values() if ok)
        total_enrolled += enrolled
        click.echo(f"  Done: {enrolled} speaker(s) enrolled/updated")
        click.echo()

    # Final summary
    profiles = load_profiles(profiles_path=team_profiles_path)
    click.echo(f"Profile database now contains {len(profiles)} speaker(s):")
    for name, p in sorted(profiles.items()):
        click.echo(f"  {name} ({p.n_sessions} session(s))")
