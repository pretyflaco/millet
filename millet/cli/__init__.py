"""CLI entrypoint for the millet tool.

Commands:
    millet record          - Record meeting audio (Ctrl+C to stop)
    millet transcribe FILE - Transcribe a recorded audio file
    millet run             - Record then transcribe when stopped
    millet gui             - Launch GUI widget for recording
    millet devices         - List available audio devices
    millet check           - Check system prerequisites
    millet download        - Download alignment models
    millet translate       - Translate a session's transcript
    millet label           - Assign real names to speakers in a session

v0.10.0: this was a single 1929-line ``cli.py`` module; it is now a
``cli/`` package with one module per command + a shared ``_helpers``
module.  The command symbols are re-exported here so the
``millet.subcommands`` / ``meet.subcommands`` entry points
(``millet.cli:transcribe`` etc.) keep resolving.
"""
from __future__ import annotations

import click

from ._helpers import _resolve_version
from .download import download
from .enroll import enroll
from .gui import gui
from .ingest import ingest
from .label import label
from .run import run
from .sync import sync
from .transcribe import transcribe
from .translate import translate
from .vocab import vocab

__all__ = [
    "download",
    "enroll",
    "gui",
    "ingest",
    "label",
    "main",
    "run",
    "sync",
    "transcribe",
    "translate",
    "vocab",
]


@click.group()
@click.version_option(
    version=_resolve_version(), prog_name="millet (millet-pipeline)"
)
def main():
    """Local meeting transcription with speaker diarization."""
    pass


# Register every command on the group.  (When invoked via the
# millet-record host CLI, the commands are discovered through the
# ``millet.subcommands`` entry-point group instead — see pyproject.toml.)
for _cmd in (transcribe, run, download, translate, label, enroll, sync, gui, ingest, vocab):
    main.add_command(_cmd)


if __name__ == "__main__":
    main()
