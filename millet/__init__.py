"""millet-pipeline — meeting transcription pipeline (successor to meetscribe-offline).

Builds on millet-record (capture primitives) and adds transcription,
diarization, summarization, PDF, GUI, sync.  Named after the Ottoman
millet system — the legal framework of communal autonomy that, in 1493,
made it possible for two Sephardic Jewish brothers to establish
Istanbul's first printing press.  Part of the vezir ecosystem.
"""

# ── F3 fix (M8, reported by @patternn 2026-05-17): inject certifi's CA
# bundle as SSL_CERT_FILE if not already set. ──────────────────────────────
#
# `meet download <lang>` calls torchaudio's alignment-model fetcher
# (``torchaudio.pipelines.<bundle>.get_model()``), which under the hood
# uses raw ``urllib.request`` with the default SSL context. On macOS
# Python installs from python.org, no CA bundle ships with the
# interpreter, so verification fails with
# ``CERTIFICATE_VERIFY_FAILED — unable to get local issuer certificate``.
# (HuggingFace model downloads are unaffected because
# ``huggingface_hub`` / ``requests`` use ``certifi`` automatically.)
#
# Pointing ``SSL_CERT_FILE`` at ``certifi.where()`` makes ``urllib``
# (and any other urllib-based caller in the process) pick up a working
# CA bundle. Done at package import time so every entry point (CLI,
# GUI, library use) gets the fix for free. We use ``setdefault`` so
# anyone who set ``SSL_CERT_FILE`` explicitly keeps their choice.
#
# Repro pre-fix:
#   python3 -m venv /tmp/v && /tmp/v/bin/pip install meetscribe-offline
#   /tmp/v/bin/meet download es     # → CERTIFICATE_VERIFY_FAILED on python.org Python
import os as _os

if "SSL_CERT_FILE" not in _os.environ:
    try:
        import certifi as _certifi

        _os.environ["SSL_CERT_FILE"] = _certifi.where()
    except ImportError:
        # certifi is a transitive dependency via requests + huggingface_hub
        # and is also listed explicitly in pyproject.toml from 0.7.2 on.
        # If it is somehow missing, do nothing rather than crash — the
        # user will simply hit the original SSL error on `meet download`.
        pass

try:
    from importlib.metadata import version as _pkg_version
    try:
        __version__ = _pkg_version("millet-pipeline")
    except Exception:
        # During the 0.9.0 transition, the legacy distribution name
        # may still be the one installed.  Honor it as a fallback.
        __version__ = _pkg_version("meetscribe-offline")
except Exception:
    __version__ = "0.21.6"
