"""millet gui command."""
from __future__ import annotations

import click


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
    type=click.Choice(["auto", "whisperx", "mlx", "parakeet"]),
    default="auto",
    help="ASR backend: auto, whisperx, mlx, or parakeet (default: auto). "
    "parakeet = NVIDIA Parakeet via onnx-asr (needs "
    "'millet download parakeet'; see --parakeet-model for languages).",
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
    help="Parakeet onnx-asr model name (default: nemo-parakeet-tdt-0.6b-v2, "
    "English only; nemo-parakeet-tdt-0.6b-v3 is multilingual)",
)
@click.option(
    "--parakeet-keep-alignment",
    is_flag=True,
    default=False,
    help="With --asr-backend parakeet, run WhisperX word-level alignment on "
    "top of Parakeet (config 'C') instead of trusting Parakeet's native "
    "timestamps (config 'B', the default).",
)
@click.option("--compute-type", type=str, default="float16")
@click.option("--batch-size", "-b", type=int, default=16)
@click.option("--language", "-l", type=str, default="auto")
@click.option("--hf-token", type=str, default=None, envvar="HF_TOKEN")
@click.option("--min-speakers", type=int, default=None)
@click.option("--max-speakers", type=int, default=None)
@click.option("--virtual-sink", is_flag=True, default=False)
@click.option(
    "--mic", type=str, default=None, help="Mic source name (default: system default)"
)
@click.option(
    "--monitor",
    type=str,
    default=None,
    help="Monitor source name (default: default sink monitor)",
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
    "--ollama-singlepass",
    is_flag=True,
    default=False,
    help="Use the legacy single-pass Ollama flow instead of the default two-pass (extract+format) flow. The two-pass flow is more accurate on local 20B-class models but adds one extra LLM call. Also configurable via MILLET_OLLAMA_SINGLEPASS=1.",
)
def gui(
    output_dir,
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
    virtual_sink,
    mic,
    monitor,
    summarize,
    summary_preset,
    summary_backend,
    summary_model,
    ollama_singlepass,
):
    """Launch the GUI recording widget."""
    from millet.gui import launch

    launch(
        output_dir=output_dir,
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
        hf_token=hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        virtual_sink=virtual_sink,
        mic=mic,
        monitor=monitor,
        summarize=summarize,
        summary_preset=summary_preset,
        summary_backend=summary_backend,
        summary_model=summary_model,
        ollama_singlepass=ollama_singlepass,
    )
