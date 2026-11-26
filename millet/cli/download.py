"""millet download command."""
from __future__ import annotations

import click


@click.command()
@click.argument("languages", nargs=-1)
@click.option(
    "--all",
    "download_all",
    is_flag=True,
    default=False,
    help="Download alignment models for all supported languages",
)
@click.option(
    "--parakeet-model",
    type=str,
    default=None,
    help="With 'parakeet': onnx-asr model to fetch (default: "
    "nemo-parakeet-tdt-0.6b-v2, English only; nemo-parakeet-tdt-0.6b-v3 "
    "is multilingual)",
)
def download(languages, download_all, parakeet_model):
    """Download alignment models for specified languages.

    \b
    Examples:
        meet download de tr fa    # download German, Turkish, Farsi
        meet download --all       # download all supported models
        meet download parakeet    # download the Parakeet ASR model (v2, English)
        meet download parakeet --parakeet-model nemo-parakeet-tdt-0.6b-v3
    """
    from millet.transcribe import (
        download_alignment_model,
        get_supported_alignment_languages,
    )

    # Special-case the Parakeet ASR model.  It is not a language alignment
    # model, but `millet download parakeet` is the natural place users look
    # for it (mirrors `meet download <lang>`).
    if "parakeet" in languages:
        from millet.parakeet import download_parakeet, ensure_parakeet_cached

        if ensure_parakeet_cached(parakeet_model):
            click.echo("  Parakeet model: already cached, skipping.")
        else:
            try:
                download_parakeet(
                    parakeet_model,
                    progress_callback=lambda msg: click.echo(f"  {msg}")
                )
            except Exception as exc:
                click.echo(f"  Error downloading Parakeet model: {exc}", err=True)
                raise SystemExit(1) from None
        languages = tuple(lang for lang in languages if lang != "parakeet")
        if not languages and not download_all:
            return

    info = get_supported_alignment_languages()

    if download_all:
        languages = tuple(info.keys())
    elif not languages:
        # No arguments — show status of all models
        click.echo("Alignment model status:")
        click.echo()
        click.echo(f"  {'Lang':<6} {'Name':<10} {'Model':<50} {'Size':<10} {'Status'}")
        click.echo(f"  {'----':<6} {'----':<10} {'-----':<50} {'----':<10} {'------'}")
        for lang, details in info.items():
            status = (
                click.style("cached", fg="green")
                if details["cached"]
                else click.style("missing", fg="red")
            )
            click.echo(
                f"  {lang:<6} {details['name']:<10} {details['model']:<50} {details['size']:<10} {status}"
            )
        click.echo()
        click.echo("To download: meet download <lang> [<lang> ...]")
        click.echo("To download all: meet download --all")
        return

    # Validate languages
    invalid = [lang for lang in languages if lang not in info]
    if invalid:
        supported = ", ".join(sorted(info.keys()))
        click.echo(f"Error: unsupported language(s): {', '.join(invalid)}", err=True)
        click.echo(f"  Supported: {supported}", err=True)
        raise SystemExit(1)

    # Download each model
    failed: list[str] = []
    for lang in languages:
        details = info[lang]
        if details["cached"]:
            click.echo(f"  {details['name']} ({lang}): already cached, skipping.")
            continue
        try:
            download_alignment_model(
                lang, progress_callback=lambda msg: click.echo(f"  {msg}")
            )
        except Exception as exc:
            click.echo(
                f"  Error downloading {details['name']} ({lang}): {exc}", err=True
            )
            failed.append(lang)

    if failed:
        # Non-zero exit so scripted/CI callers detect the failure instead of
        # assuming every requested model was fetched.
        click.echo(
            f"Error: failed to download: {', '.join(failed)}", err=True
        )
        raise SystemExit(1)
