"""Shared helpers for the millet CLI command modules."""
from __future__ import annotations

import signal
import time

import click

from millet.capture import DRAIN_SECONDS
from millet.utils import fmt_elapsed, fmt_size


def _drain_countdown(session, seconds: int = DRAIN_SECONDS) -> None:
    """Keep recording for *seconds* more to let ffmpeg's delayed pipeline flush.

    During the countdown:
    - Additional Ctrl+C signals are ignored (SIGINT → SIG_IGN)
    - A single status line updates in-place each second showing remaining time,
      elapsed recording time, and file size
    After the countdown, default SIGINT handling is restored.
    """
    # Ignore further Ctrl+C during the drain window
    prev_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    try:
        for remaining in range(seconds, 0, -1):
            status = session.status()
            elapsed = fmt_elapsed(status.elapsed_seconds)
            size = fmt_size(status.file_size_bytes)
            click.echo(
                f"\r\033[K\033[1;33m⏳ Flushing audio buffer... {remaining}s\033[0m"
                f"  {elapsed}  {size}",
                nl=False,
            )
            time.sleep(1)
        # Final line
        status = session.status()
        elapsed = fmt_elapsed(status.elapsed_seconds)
        size = fmt_size(status.file_size_bytes)
        click.echo(f"\r\033[K\033[1;32m✔ Buffer flushed\033[0m  {elapsed}  {size}")
    finally:
        # Restore previous SIGINT handler
        signal.signal(signal.SIGINT, prev_handler)


def _generate_summary(
    transcript, out_dir, basename, summary_model, files, summary_backend=None,
    summary_preset=None, ollama_singlepass=False, summary_template=None,
    frames=None,
):
    """Generate an AI meeting summary. Returns MeetingSummary or None.

    Supports the private backends (tinfoil TEE, local ollama) via SummaryConfig.
    The fallback chain is handled inside summarize() — callers should not
    gate on is_backend_available().

    When ``ollama_singlepass`` is True, the legacy single-pass flow is used
    for the ollama backend.  By default the two-pass (extract+format) flow
    is used, which is more accurate on local 20B-class models at the cost
    of one extra LLM call.

    When ``summary_template`` names a summary template (e.g.
    "iteration-plan"), the template's prompt files are used and the output
    is written as ``<basename>.<template>.md`` instead of the default
    ``<basename>.summary.md``.
    """
    from millet.summarize import SummaryConfig
    from millet.summarize import summarize as do_summarize

    config_kwargs = {}
    if summary_preset:
        config_kwargs["preset"] = summary_preset
    if summary_backend:
        config_kwargs["backend"] = summary_backend
    if summary_model:
        config_kwargs["model"] = summary_model
    if ollama_singlepass:
        config_kwargs["ollama_singlepass"] = True
    if summary_template:
        config_kwargs["template"] = summary_template
    if frames:
        config_kwargs["frames"] = frames
    summary_config = SummaryConfig(**config_kwargs)
    if summary_config.frames:
        click.echo(f"  Including {len(summary_config.frames)} cue frame(s)")

    def _cli_progress(msg: str) -> None:
        click.echo(f"  {msg}")

    # The RESOLVED template (explicit arg > MILLET_SUMMARY_TEMPLATE, lowercased
    # by SummaryConfig) drives the output name and the clips decision -- the
    # raw argument is None when the template comes from the environment.
    template = summary_config.template
    artifact = template or "summary"
    click.echo(
        f"Generating meeting summary ({summary_config.model} via {summary_config.backend})..."
    )
    try:
        result = do_summarize(
            transcript.to_text(),
            summary_config,
            language=transcript.language,
            progress_callback=_cli_progress,
        )
        from millet.frontmatter import context_from_transcript

        fm_ctx = context_from_transcript(transcript, out_dir)
        path = result.save(out_dir, basename, frontmatter_context=fm_ctx, artifact=artifact)
        files[artifact] = path
        click.echo(f"  Summary generated in {result.elapsed_seconds:.1f}s")
        _write_clips_if_template(result, transcript, out_dir, basename, files, template)
        return result
    except Exception as exc:
        click.echo(f"  Summary failed: {exc}", err=True)
        if summary_preset:
            # When a preset was explicitly selected (e.g. "confidential"),
            # the user chose a specific privacy/quality level.  Silently
            # returning no summary would make the failure invisible to the
            # caller (vezir worker, GUI) and yield a misleading "success".
            # Re-raise so `meet transcribe` exits non-zero.
            raise
        return None


def _write_clips_if_template(
    summary_result, transcript, out_dir, basename, files, summary_template,
    lang_suffix=None,
):
    """Write ``<basename>.clips.json`` for a ``video-clipping`` summary.

    A failure here never fails the summary: the Markdown clip list already
    exists, and the clips file can be rebuilt by a summary retry.
    """
    try:
        from millet.clips import is_clips_template, write_clips_doc

        if not is_clips_template(summary_template):
            return None
        path, doc = write_clips_doc(summary_result, transcript, out_dir, basename,
                                    lang_suffix=lang_suffix)
        click.echo(
            f"  Clips: {len(doc['clips'])} kept, {len(doc['dropped'])} dropped "
            f"-> {path.name}"
        )
        if doc.get("error"):
            click.echo(f"  Clips: {doc['error']}", err=True)
    except Exception as exc:
        click.echo(f"  Clip extraction failed: {exc}", err=True)
        return None
    files["clips"] = path
    return path


def _generate_pdf(transcript, out_dir, basename, summary_result, files):
    """Generate a PDF transcript with optional summary."""
    from millet.pdf import generate_pdf

    pdf_path = out_dir / f"{basename}.pdf"
    try:
        generate_pdf(
            transcript,
            pdf_path,
            summary=summary_result,
            language=getattr(transcript, "language", "en"),
        )
        files["pdf"] = pdf_path
    except Exception as exc:
        click.echo(f"  PDF generation failed: {exc}", err=True)


def _recording_loop(session) -> None:
    """Run the live recording status display loop.

    Shows an updating single-line status indicator. Replaces signal.pause()
    with an active monitoring loop that displays:
        REC  00:07:23  14.2 MB  Ctrl+C to stop

    Immediately alerts if recording fails or restarts.
    """
    last_restart_count = 0
    warned_failed = False

    try:
        while True:
            status = session.status()

            elapsed = fmt_elapsed(status.elapsed_seconds)
            size = fmt_size(status.file_size_bytes)

            if status.failed and not warned_failed:
                # Recording failed and could not restart
                reason = status.fail_reason or "unknown error"
                click.echo(
                    f"\r\033[K\033[1;31m✖ RECORDING FAILED\033[0m  {elapsed}  {size}  — {reason}"
                )
                click.echo("  Press Ctrl+C to transcribe what was captured.")
                warned_failed = True
            elif status.restart_count > last_restart_count:
                # ffmpeg was restarted — show brief warning
                last_restart_count = status.restart_count
                click.echo(
                    f"\r\033[K\033[1;33m⚠ Recording restarted\033[0m (attempt {status.restart_count})  {elapsed}  {size}"
                )
            elif not warned_failed:
                # Normal status line — overwrite in place
                if status.is_alive:
                    line = f"\r\033[K\033[1;32m● REC\033[0m  {elapsed}  {size}  Ctrl+C to stop"
                else:
                    line = f"\r\033[K\033[1;33m● REC (starting...)\033[0m  {elapsed}  {size}"
                click.echo(line, nl=False)

            time.sleep(1)
    except KeyboardInterrupt:
        # Clear the status line before returning
        click.echo("\r\033[K", nl=False)
        raise


def _resolve_version() -> str:
    """Resolve the millet-pipeline (formerly meetscribe-offline) package
    version dynamically.

    Avoids the historical bug where `version="0.4.1"` was hardcoded and
    drifted from the real package version.
    """
    try:
        from importlib.metadata import version
        try:
            return version("millet-pipeline")
        except Exception:
            return version("meetscribe-offline")
    except Exception:
        # Source checkout with no installed metadata: fall back to the
        # package version literal.  __version__ lives in millet/__init__.py,
        # not millet.cli, so import it from the top-level package.
        try:
            from millet import __version__
            return __version__
        except Exception:
            return "unknown"
