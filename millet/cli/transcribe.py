"""millet transcribe command."""
from __future__ import annotations

from pathlib import Path

import click

from ._helpers import (
    _generate_pdf,
    _generate_summary,
)


def _echo_offline_model_help(model: str, audio_file: str) -> None:
    """Print actionable guidance for an offline model-cache miss."""
    click.echo()
    click.echo(
        click.style(
            f"Error: model '{model}' is not in the local cache and Hugging Face "
            "downloads are disabled (HF_HUB_OFFLINE is set).",
            fg="red",
        ),
        err=True,
    )
    click.echo(err=True)
    click.echo("  Cache the model once with downloads enabled, then retry:", err=True)
    click.echo(f"    HF_HUB_OFFLINE=0 meet transcribe {audio_file}", err=True)
    click.echo(err=True)
    click.echo("  Or pre-download it, e.g.:", err=True)
    click.echo(
        f"    HF_HUB_OFFLINE=0 hf download {model}",
        err=True,
    )


def _cue_frames(audio_path, enabled: bool):
    """Cue frames sitting next to the session, or None when disabled/absent.

    Deliberately keyed off ``audio_path.parent`` (the true session dir) and
    not ``--output-dir``: the frames belong to the recording, wherever the
    outputs are written.
    """
    if not enabled:
        return None
    from millet.summarize import discover_cue_frames

    return discover_cue_frames(audio_path.parent) or None


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option(
    "--model",
    "-m",
    type=str,
    default="large-v3-turbo",
    help="Whisper model (default: large-v3-turbo). Also: base, medium, large-v2, or a local path.",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default=None,
    help=(
        "Device to run on (default: cpu on Apple Silicon, cuda elsewhere). "
        "Use --torch-device to set the PyTorch device separately."
    ),
)
@click.option(
    "--torch-device",
    type=click.Choice(["cuda", "cpu", "mps"]),
    default=None,
    help=(
        "PyTorch device for alignment/diarization "
        "(default: mps on Apple Silicon, otherwise same as --device)"
    ),
)
@click.option(
    "--asr-backend",
    type=click.Choice(["auto", "whisperx", "mlx", "parakeet"]),
    default="auto",
    help="ASR backend: auto, whisperx, mlx, or parakeet (default: auto). "
    "parakeet = NVIDIA Parakeet via onnx-asr (English; needs "
    "'millet download parakeet').",
)
@click.option(
    "--mlx-model",
    type=str,
    default=None,
    help="MLX Whisper model path/repo (default: alias mapped from --model)",
)
@click.option(
    "--parakeet-model",
    type=str,
    default=None,
    help="Parakeet onnx-asr model name (default: nemo-parakeet-tdt-0.6b-v2, English)",
)
@click.option(
    "--parakeet-keep-alignment",
    is_flag=True,
    default=False,
    help="With --asr-backend parakeet, run WhisperX word-level alignment on "
    "top of Parakeet (config 'C') instead of trusting Parakeet's native "
    "timestamps (config 'B', the default).",
)
@click.option(
    "--compute-type",
    type=str,
    default="float16",
    help="Compute type: float16, int8 (default: float16)",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=16,
    help="Batch size for transcription (default: 16)",
)
@click.option(
    "--language",
    "-l",
    type=str,
    default="auto",
    help="Language code or 'auto' to detect (default: auto). Examples: en, de, fr, es, tr, fa",
)
@click.option(
    "--hf-token",
    type=str,
    default=None,
    envvar="HF_TOKEN",
    help="HuggingFace token for diarization (or set HF_TOKEN env var)",
)
@click.option(
    "--min-speakers", type=int, default=None, help="Minimum number of speakers"
)
@click.option(
    "--max-speakers", type=int, default=None, help="Maximum number of speakers"
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=None,
    help="Output directory for transcripts (default: same as audio file)",
)
@click.option(
    "--no-diarize", is_flag=True, default=False, help="Skip speaker diarization"
)
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
    "--summary-frames/--no-summary-frames",
    default=True,
    help="Include cue frames (<session>/attachments/cue_*.png) in the summary "
    "when the model can accept image input, so a narrated screen recording is "
    "summarized from what is on screen as well as what was said. On by "
    "default; a no-op when no frames exist.",
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
    type=click.Choice(["mono", "dual", "dual-diarize"]),
    default="dual-diarize",
    help="Stereo mixdown mode: dual-diarize=transcribe channels separately "
    "+ diarize remotes (default; best accuracy), mono=mix+diarize (legacy), "
    "dual=transcribe channels separately (YOU/REMOTE only, no remote naming).",
)
@click.option(
    "--channel-correct/--no-channel-correct",
    default=True,
    help="Hybrid channel-energy correction for stereo (mono mixdown): reassign "
    "segments/words whose audio is strongly dominated by the opposite channel "
    "from their diarized YOU/REMOTE label. Fixes turn-boundary attribution "
    "leaks. On by default; --no-channel-correct to disable.",
)
@click.option(
    "--channel-correct-margin",
    type=float,
    default=0.30,
    help="Dominance margin for channel correction (default: 0.30). A segment's "
    "mic/(mic+sys) ratio must cross 0.5±margin to override its diarized side. "
    "Higher = more conservative (resists mic bleed from open speakers).",
)
@click.option(
    "--consolidate-remote-clusters/--no-consolidate-remote-clusters",
    default=True,
    help="Consolidate over-segmented remote speakers in the dual-diarize path: "
    "merge same-speaker clusters (voiceprint-guided) and absorb thin backchannel "
    "clusters into the dominant remote. Fixes phantom extra speakers. On by "
    "default; --no-consolidate-remote-clusters to disable.",
)
@click.option(
    "--single-source-fallback/--no-single-source-fallback",
    default=True,
    help="In the dual-diarize path, detect single-source stereo (system "
    "channel silent, or a duplicate of the mic — i.e. an in-room recording) "
    "and fall back to mono diarization so multiple in-room speakers on the "
    "mic channel are split instead of collapsing into one. On by default.",
)
@click.option(
    "--language-detection-segments",
    type=int,
    default=6,
    help="When auto-detecting language (whisperx), sample this many windows "
    "across each channel instead of only the first 30s (default: 6). Avoids "
    "mislabeling a channel from a misleading opener.",
)
@click.option(
    "--default-language",
    default=None,
    help="Soft team/operator default language (e.g. 'en'). When set and "
    "language is auto, the default wins UNLESS a channel confidently detects "
    "another language. Prevents drift for single-language teams.",
)
def transcribe(
    audio_file,
    model,
    device,
    torch_device,
    asr_backend,
    mlx_model,
    parakeet_model,
    parakeet_keep_alignment,
    compute_type,
    batch_size,
    language,
    hf_token,
    min_speakers,
    max_speakers,
    output_dir,
    no_diarize,
    summarize,
    summary_preset,
    summary_backend,
    summary_model,
    summary_template,
    summary_frames,
    ollama_singlepass,
    skip_alignment,
    mixdown,
    channel_correct,
    channel_correct_margin,
    consolidate_remote_clusters,
    single_source_fallback,
    language_detection_segments,
    default_language,
):
    """Transcribe a recorded audio file with speaker diarization."""
    from millet.transcribe import (
        AlignmentModelMissing,
        OfflineModelMissing,
        TranscriptionConfig,
        ensure_gpu_available,
    )
    from millet.transcribe import (
        transcribe as do_transcribe,
    )

    audio_path = Path(audio_file)

    # If user passed a session directory, find the audio file inside it.
    if audio_path.is_dir():
        wavs = sorted(audio_path.glob("*.wav"))
        oggs = sorted(audio_path.glob("*.ogg"))
        mp3s = sorted(audio_path.glob("*.mp3"))
        audio_files = wavs or oggs or mp3s
        if not audio_files:
            click.echo(
                f"Error: no audio file (.wav/.ogg/.mp3) found in {audio_path}", err=True
            )
            raise SystemExit(1)
        if len(audio_files) > 1:
            # Refuse to silently transcribe only the first of several files.
            # A session dir is expected to hold exactly one continuous
            # recording; multiple files almost always means the caller
            # intended them merged (see vezir multi-audio uploads, which
            # concatenate before invoking transcribe).  Fail loudly with the
            # list so the mistake is visible rather than producing a
            # transcript that silently drops most of the meeting.
            listing = "\n".join(f"  - {p.name}" for p in audio_files)
            click.echo(
                f"Error: {len(audio_files)} audio files found in {audio_path}; "
                "transcribe expects a single recording.\n"
                f"{listing}\n"
                "Merge them into one file first (e.g. ffmpeg concat) or pass "
                "the specific file you want.",
                err=True,
            )
            raise SystemExit(1)
        audio_path = audio_files[0]
        click.echo(f"  Resolved to: {audio_path}")

    config = TranscriptionConfig(
        model=model,
        device=device,
        torch_device=torch_device,
        asr_backend=asr_backend,
        mlx_model=mlx_model,
        parakeet_model=parakeet_model,
        parakeet_skip_alignment=not parakeet_keep_alignment,
        compute_type=compute_type,
        batch_size=batch_size,
        language=language,
        hf_token=hf_token if not no_diarize else None,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        skip_alignment=skip_alignment,
        mixdown=mixdown,
        channel_correct=channel_correct,
        channel_correct_margin=channel_correct_margin,
        consolidate_remote_clusters=consolidate_remote_clusters,
        single_source_fallback=single_source_fallback,
        language_detection_segments=language_detection_segments,
        default_language=default_language,
    )

    if not no_diarize and not config.hf_token and mixdown != "dual":
        click.echo("Warning: No HF_TOKEN found. Diarization will be skipped.", err=True)
        click.echo("  Set HF_TOKEN env var or pass --hf-token", err=True)
        click.echo("  Get a token at: https://huggingface.co/settings/tokens", err=True)
        click.echo(
            "  Accept model terms at: https://huggingface.co/pyannote/speaker-diarization-community-1",
            err=True,
        )
        click.echo()

    click.echo(f"Transcribing: {audio_path}")
    if config.asr_backend == "mlx":
        click.echo(f"  ASR:      mlx ({config.mlx_model})")
    elif config.asr_backend == "parakeet":
        _pk_model = config.parakeet_model or "nemo-parakeet-tdt-0.6b-v2"
        _align = "keep-alignment" if not config.skip_alignment else "native-timestamps"
        click.echo(f"  ASR:      parakeet ({_pk_model}, {_align})")
    else:
        click.echo(f"  ASR:      whisperx ({config.model}, {config.compute_type})")
    click.echo(f"  Device:   {config.device}")
    click.echo(f"  Torch:    {config.torch_device}")
    click.echo(f"  Language: {config.language}")
    click.echo(f"  Diarize:  {bool(config.hf_token)}")
    click.echo()

    # Free GPU memory from Ollama before transcription
    ensure_gpu_available()

    try:
        transcript = do_transcribe(audio_path, config)
    except AlignmentModelMissing as exc:
        click.echo()
        click.echo(click.style(f"Error: {exc}", fg="red"), err=True)
        click.echo(err=True)
        click.echo("  To download it, run:", err=True)
        click.echo(f"    meet download {exc.lang}", err=True)
        click.echo(err=True)
        click.echo(
            "  Or skip alignment (fewer segments, no word-level timestamps):", err=True
        )
        click.echo(
            f"    meet transcribe {audio_file} --language {exc.lang} --skip-alignment",
            err=True,
        )
        raise SystemExit(1) from None
    except OfflineModelMissing as exc:
        _echo_offline_model_help(exc.model, audio_file)
        raise SystemExit(1) from None
    except Exception as exc:
        # huggingface_hub raises LocalEntryNotFoundError when a model is not in
        # the local cache and downloads are disabled.  Catch it by name so we
        # avoid importing huggingface_hub at module load, and translate it into
        # the same actionable guidance rather than dumping a raw traceback.
        if type(exc).__name__ == "LocalEntryNotFoundError":
            _echo_offline_model_help(config.mlx_model or config.model, audio_file)
            raise SystemExit(1) from None
        raise

    # Determine output directory
    if output_dir is None:
        out_dir = audio_path.parent
    else:
        out_dir = Path(output_dir)

    files = transcript.save(out_dir, basename=audio_path.stem)

    # ── Summary + PDF ──
    summary_result = None
    preset_summary_error: Exception | None = None
    if summarize:
        try:
            summary_result = _generate_summary(
                transcript,
                out_dir,
                audio_path.stem,
                summary_model,
                files,
                summary_backend=summary_backend,
                summary_preset=summary_preset,
                ollama_singlepass=ollama_singlepass,
                summary_template=summary_template,
                frames=_cue_frames(audio_path, summary_frames),
            )
        except Exception as exc:
            # Only raised when summary_preset was set (preset guard).
            # Generate the PDF first so the transcript artifact still
            # exists, then surface the failure as a non-zero exit below.
            preset_summary_error = exc

    _generate_pdf(transcript, out_dir, audio_path.stem, summary_result, files)

    click.echo()
    click.echo("Transcription complete!")
    click.echo(f"  Duration: {transcript.duration:.0f}s" if transcript.duration else "")
    click.echo(f"  Speakers: {len(transcript.speakers)}")
    click.echo(f"  Segments: {len(transcript.segments)}")
    click.echo()
    click.echo("Output files:")
    for fmt, path in files.items():
        click.echo(f"  {fmt}: {path}")

    click.echo()
    click.echo("--- Transcript Preview ---")
    click.echo()
    # Show first 20 lines
    lines = transcript.to_text().split("\n")
    for line in lines[:20]:
        click.echo(line)
    if len(lines) > 20:
        click.echo(f"  ... ({len(lines) - 20} more lines, see {files['text']})")

    if preset_summary_error is not None:
        # Preset summary failure: surface as non-zero exit so callers
        # (vezir worker, CI) can detect the partial failure.  The transcript
        # and PDF are already on disk for inspection.
        click.echo(err=True)
        click.echo(
            click.style(
                f"Error: summary failed for preset "
                f"'{summary_preset}': {preset_summary_error}",
                fg="red",
            ),
            err=True,
        )
        raise SystemExit(1)
