"""millet run command."""
from __future__ import annotations

import sys

import click

from ._helpers import (
    _drain_countdown,
    _generate_pdf,
    _generate_summary,
    _recording_loop,
)


@click.command()
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Directory for recordings and transcripts",
)
@click.option(
    "--model",
    "-m",
    type=str,
    default="large-v3-turbo",
    help="Whisper model (default: large-v3-turbo)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=None,
    help="Device to run on (default: cpu on Apple Silicon, cuda elsewhere)",
)
@click.option(
    "--torch-device",
    type=click.Choice(["cuda", "cpu", "mps"]),
    default=None,
    help="PyTorch device for alignment/diarization (default: same as --device)",
)
@click.option(
    "--asr-backend",
    type=click.Choice(["auto", "whisperx", "mlx"]),
    default="auto",
    help="ASR backend: auto, whisperx, or mlx (default: auto)",
)
@click.option(
    "--mlx-model",
    type=str,
    default=None,
    help="MLX Whisper model path/repo (default: alias mapped from --model)",
)
@click.option("--compute-type", type=str, default="float16")
@click.option("--batch-size", "-b", type=int, default=16)
@click.option("--language", "-l", type=str, default="auto")
@click.option("--hf-token", type=str, default=None, envvar="HF_TOKEN")
@click.option("--min-speakers", type=int, default=None)
@click.option("--max-speakers", type=int, default=None)
@click.option("--virtual-sink", is_flag=True, default=False)
@click.option(
    "--summarize/--no-summarize",
    default=True,
    help="Generate AI meeting summary (default: on)",
)
@click.option(
    "--summary-preset",
    type=click.Choice(["high-quality", "confidential", "alternative"], case_sensitive=False),
    default=None,
    help="DEPRECATED (removed in 0.21.0): all presets now resolve to the default private backend. Kept so existing scripts keep working.",
)
@click.option(
    "--summary-backend",
    type=click.Choice(
        ["tinfoil", "ollama"], case_sensitive=False
    ),
    default=None,
    help="Summary backend (default: tinfoil, or MILLET_SUMMARY_BACKEND env var)",
)
@click.option(
    "--summary-model",
    type=str,
    default=None,
    help="Model for summary (default: per-backend, or MILLET_SUMMARY_MODEL env var)",
)
@click.option(
    "--summary-template",
    type=str,
    default=None,
    help="Summary prompt template name ('iteration-plan' or 'video-clipping'). Uses "
    "summarize_<template>_{system,user}.md from millet/prompts (dashes in "
    "the name become underscores) and writes "
    "<base>.<template>.md instead of <base>.summary.md. Also configurable "
    "via MILLET_SUMMARY_TEMPLATE.",
)
@click.option(
    "--ollama-singlepass",
    is_flag=True,
    default=False,
    help="Use the legacy single-pass Ollama flow instead of the default two-pass (extract+format) flow. The two-pass flow is more accurate on local 20B-class models but adds one extra LLM call. Also configurable via MILLET_OLLAMA_SINGLEPASS=1.",
)
@click.option(
    "--skip-alignment",
    is_flag=True,
    default=False,
    help="Skip word-level alignment (useful if alignment model is unavailable)",
)
@click.option(
    "--mixdown",
    type=click.Choice(["mono", "dual"]),
    default="mono",
    help="Stereo mixdown mode: mono=mic channel only, dual=transcribe both channels separately (default: mono)",
)
def run(
    output_dir,
    model,
    device,
    torch_device,
    asr_backend,
    mlx_model,
    compute_type,
    batch_size,
    language,
    hf_token,
    min_speakers,
    max_speakers,
    virtual_sink,
    summarize,
    summary_preset,
    summary_backend,
    summary_model,
    summary_template,
    ollama_singlepass,
    skip_alignment,
    mixdown,
):
    """Record a meeting, then transcribe when stopped with Ctrl+C."""
    from millet.capture import check_prerequisites, create_session
    from millet.transcribe import (
        AlignmentModelMissing,
        TranscriptionConfig,
        ensure_gpu_available,
    )
    from millet.transcribe import (
        transcribe as do_transcribe,
    )

    issues = check_prerequisites()
    if issues:
        click.echo("Prerequisites check failed:", err=True)
        for issue in issues:
            click.echo(f"  - {issue}", err=True)
        sys.exit(1)

    session = create_session(
        output_dir=output_dir,
        virtual_sink=virtual_sink,
    )

    config = TranscriptionConfig(
        model=model,
        device=device,
        torch_device=torch_device,
        asr_backend=asr_backend,
        mlx_model=mlx_model,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        skip_alignment=skip_alignment,
        mixdown=mixdown,
    )

    if not config.hf_token and mixdown != "dual":
        click.echo("Warning: No HF_TOKEN found. Diarization will be skipped.", err=True)
        click.echo("  Set HF_TOKEN env var or pass --hf-token", err=True)
        click.echo()

    click.echo(f"Recording to: {session.output_file}")
    click.echo(f"  Mic:     {session.mic_source}")
    click.echo(f"  Monitor: {session.monitor_source}")
    click.echo(f"  ASR:     {config.asr_backend}")
    click.echo(f"  Device:  {config.device}")
    click.echo(f"  Torch:   {config.torch_device}")
    click.echo(f"  Diarize: {bool(config.hf_token)}")
    click.echo()

    session.start()

    try:
        _recording_loop(session)
    except KeyboardInterrupt:
        _drain_countdown(session)
        click.echo("Stopping recording...")
        output = session.stop()

        if not output.exists() or output.stat().st_size == 0:
            click.echo("Error: No audio was recorded.", err=True)
            sys.exit(1)

        size_mb = output.stat().st_size / (1024 * 1024)
        rec_status = session.status()
        click.echo(f"Saved recording: {output} ({size_mb:.1f} MB)")
        if rec_status.restart_count > 0:
            click.echo(
                f"  Note: recording restarted {rec_status.restart_count} time(s)"
            )
        click.echo()
        click.echo("Starting transcription...")
        click.echo()

        # Free GPU memory from Ollama before transcription
        ensure_gpu_available()

        try:
            transcript = do_transcribe(output, config)
        except AlignmentModelMissing as exc:
            click.echo()
            click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
            click.echo(err=True)
            click.echo("  To download it, run:", err=True)
            click.echo(f"    meet download {exc.lang}", err=True)
            click.echo(err=True)
            click.echo("  Or re-run with --skip-alignment:", err=True)
            click.echo(
                f"    meet transcribe {output} --language {exc.lang} --skip-alignment",
                err=True,
            )
            click.echo(err=True)
            click.echo(f"  Your recording is saved at: {output}", err=True)
            sys.exit(1)
        files = transcript.save(output.parent, basename=output.stem)

        # ── Summary + PDF ──
        summary_result = None
        preset_summary_error: Exception | None = None
        if summarize:
            try:
                summary_result = _generate_summary(
                    transcript,
                    output.parent,
                    output.stem,
                    summary_model,
                    files,
                    summary_backend=summary_backend,
                    summary_preset=summary_preset,
                    ollama_singlepass=ollama_singlepass,
                    summary_template=summary_template,
                )
            except Exception as exc:
                # Only raised when summary_preset was set (preset guard).
                # Generate the PDF first so the recording's transcript
                # artifacts still exist, then fail non-zero below.
                preset_summary_error = exc

        _generate_pdf(transcript, output.parent, output.stem, summary_result, files)

        click.echo()
        click.echo("Done!")
        click.echo(
            f"  Duration: {transcript.duration:.0f}s" if transcript.duration else ""
        )
        click.echo(f"  Speakers: {len(transcript.speakers)}")
        click.echo(f"  Segments: {len(transcript.segments)}")
        click.echo()
        click.echo("Output files:")
        for fmt, path in files.items():
            click.echo(f"  {fmt}: {path}")

        click.echo()
        click.echo("--- Transcript ---")
        click.echo()
        click.echo(transcript.to_text())

        if preset_summary_error is not None:
            # Preset summary failure: surface as non-zero exit so callers
            # (vezir worker, CI) can detect the partial failure.  The
            # recording, transcript, and PDF are already on disk.
            click.echo(err=True)
            click.echo(
                click.style(
                    f"Error: summary failed for preset "
                    f"'{summary_preset}': {preset_summary_error}",
                    fg="red",
                ),
                err=True,
            )
            sys.exit(1)
        sys.exit(0)
