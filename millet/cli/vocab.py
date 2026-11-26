"""CLI commands for domain vocabulary harvesting and dry-run diffing.

Usage:
    millet vocab harvest <sessions_dir_or_files...> --output report.json
    millet vocab dry-run vocabulary.json <sessions_dir_or_files...>
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click


def _parse_since(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise click.BadParameter(
            f"expected ISO date/timestamp (e.g. 2026-10-01), got {value!r}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _load_sessions_file(path: Path) -> list[Path]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        if isinstance(data, list):
            return [Path(str(x)) for x in data if str(x).strip()]
    return [Path(line.strip()) for line in raw.splitlines() if line.strip()]


@click.group()
def vocab():
    """Harvest candidate domain vocabulary and preview alias diffs."""


@vocab.command("harvest")
@click.argument("sources", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--sessions-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional JSON list or newline-separated file of session dirs/transcripts.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write harvest report JSON to this path (prints JSON to stdout if omitted).",
)
@click.option(
    "--anchors",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional plain-text, JSON, or Markdown file of canonical anchor terms.",
)
@click.option(
    "--existing",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Existing vocabulary.json (defaults to team/global vocabulary path).",
)
@click.option(
    "--rejected",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional rejected-pairs JSON file to suppress dismissed candidates.",
)
@click.option("--team", default=None, help="Team namespace for default vocab/voiceprints.")
@click.option("--since", default=None, help="Only scan sessions modified on/after YYYY-MM-DD.")
@click.option("--max-sessions", type=int, default=100, show_default=True)
@click.option("--min-occurrences", type=int, default=3, show_default=True)
@click.option("--min-sessions", type=int, default=2, show_default=True)
@click.option("--holdout-ratio", type=float, default=0.2, show_default=True)
@click.option(
    "--summary-backend",
    type=click.Choice(["tinfoil", "ollama"], case_sensitive=False),
    default=None,
    help="Private LLM backend to use (defaults to tinfoil -> ollama).",
)
@click.option("--summary-model", default=None, help="Override LLM model id.")
def harvest_cmd(
    sources: tuple[Path, ...],
    sessions_file: Path | None,
    output: Path | None,
    anchors: Path | None,
    existing: Path | None,
    rejected: Path | None,
    team: str | None,
    since: str | None,
    max_sessions: int,
    min_occurrences: int,
    min_sessions: int,
    holdout_ratio: float,
    summary_backend: str | None,
    summary_model: str | None,
):
    """Mine historical transcripts for recurring ASR mishearings (read-only)."""
    from millet import paths
    from millet.vocab_harvest import (
        harvest_vocabulary,
        load_anchors_file,
        load_rejected_pairs,
        write_report_atomic,
    )
    from millet.vocabulary import load_vocabulary

    all_sources = list(sources)
    if sessions_file is not None:
        all_sources.extend(_load_sessions_file(sessions_file))
    if not all_sources:
        raise click.UsageError("Provide at least one session path or --sessions-file.")

    since_dt = _parse_since(since)
    existing_vocab = (
        load_vocabulary(path=existing)
        if existing is not None
        else load_vocabulary(team=team)
    )
    profiles_path = paths.profiles_path(team) if team else None
    extra_anchors = load_anchors_file(anchors) if anchors is not None else []
    rejected_pairs = load_rejected_pairs(rejected)

    try:
        report = harvest_vocabulary(
            all_sources,
            existing_vocab=existing_vocab,
            profiles_path=profiles_path,
            extra_anchors=extra_anchors,
            rejected_pairs=rejected_pairs,
            since=since_dt,
            max_sessions=max_sessions,
            min_occurrences=min_occurrences,
            min_sessions=min_sessions,
            holdout_ratio=holdout_ratio,
            backend=summary_backend,
            model=summary_model,
            progress_callback=lambda msg: click.echo(msg, err=True),
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from None

    if output is not None:
        write_report_atomic(report, output)
        click.echo(
            f"Wrote {len(report['candidates'])} candidate(s) from "
            f"{report['sessions_scanned']} session(s) to {output}",
            err=True,
        )
    else:
        click.echo(json.dumps(report, indent=2, ensure_ascii=False))


@vocab.command("dry-run")
@click.argument("vocab_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("sources", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--sessions-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional JSON list or newline-separated file of session dirs/transcripts.",
)
@click.option("--since", default=None, help="Only scan sessions modified on/after YYYY-MM-DD.")
@click.option("--max-sessions", type=int, default=100, show_default=True)
@click.option("--limit", type=int, default=50, show_default=True, help="Max diff lines to show.")
@click.option("--json-output", "as_json", is_flag=True, help="Emit raw JSON diff report.")
def dry_run_cmd(
    vocab_file: Path,
    sources: tuple[Path, ...],
    sessions_file: Path | None,
    since: str | None,
    max_sessions: int,
    limit: int,
    as_json: bool,
):
    """Preview alias replacements across historical transcripts without modifying files."""
    from millet.vocab_harvest import dry_run_vocabulary_diff
    from millet.vocabulary import load_vocabulary

    all_sources = list(sources)
    if sessions_file is not None:
        all_sources.extend(_load_sessions_file(sessions_file))
    if not all_sources:
        raise click.UsageError("Provide at least one session path or --sessions-file.")

    v = load_vocabulary(path=vocab_file)
    result = dry_run_vocabulary_diff(
        v,
        all_sources,
        since=_parse_since(since),
        max_sessions=max_sessions,
        limit=limit,
    )
    if as_json:
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
        return

    click.echo(
        f"Scanned {result['sessions_scanned']} session(s): "
        f"{result['total_replacements']} replacement(s) across "
        f"{result['sessions_changed']} session(s)."
    )
    for d in result["diffs"]:
        click.echo(f"\n[{d['session_id']} @ {d['start']:.1f}s]")
        click.echo(f"  - {d['before']}")
        click.echo(f"  + {d['after']}")
