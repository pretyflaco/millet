"""Parakeet ASR backend via onnx-asr (NVIDIA Parakeet TDT, ONNX Runtime).

This module isolates all onnx-asr-specific logic so the rest of millet only
sees the same WhisperX-shaped result dict the other backends emit:

    {"segments": [{"start": float, "end": float, "text": str}, ...],
     "language": str,
     "text": str}

Key constraints handled here:

* **20-30s model input limit.** Parakeet (like most NeMo models) cannot
  ingest a whole meeting at once.  We wrap the model in onnx-asr's VAD
  (Silero) adapter, which segments long audio and yields per-segment
  results with *global* timestamps already stitched.  This is mandatory for
  our 45-105 minute recordings.

* **English by default.** We default to ``nemo-parakeet-tdt-0.6b-v2``
  (the English-only model, the benchmark target).  The multilingual
  ``nemo-parakeet-tdt-0.6b-v3`` is selected by name (``--parakeet-model``
  or the GUI's Parakeet model field).  Parakeet does not detect the
  language, so pass ``--language`` for non-English meetings.

* **Lazy, explicit model fetch.** ``millet download parakeet`` pre-fetches
  the ONNX weights to the HF cache; we never auto-download inside
  ``transcribe`` (matches the alignment-model policy).

The backend is *opt-in* via ``--asr-backend parakeet``; ``auto`` selection
is deliberately left unchanged until benchmark data justifies a default.
"""
from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

# Default English Parakeet model (onnx-asr name).  The multilingual variant
# is "nemo-parakeet-tdt-0.6b-v3"; v2 is English-only and our benchmark target.
DEFAULT_PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v2"

# VAD model used to chunk long audio into <=30s windows that Parakeet can
# ingest.  Silero is small, fast, and bundled support in onnx-asr.
DEFAULT_VAD_MODEL = "silero"

# Sample rate Parakeet (and the rest of our pipeline) expects.
SAMPLE_RATE = 16000

# Module-level cache so repeated calls in one process (e.g. dual-channel
# transcribes mic then system) don't reload the ~600MB model twice.
_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_VAD_CACHE: dict[str, Any] = {}


def parakeet_available() -> bool:
    """Return True if onnx-asr (and thus the Parakeet backend) is importable."""
    return importlib.util.find_spec("onnx_asr") is not None


_cuda_libs_prepared = False


def _prepare_cuda_libs() -> None:
    """Make CUDA/cuDNN shared libs from the torch nvidia-* pip packages
    discoverable to onnxruntime-gpu.

    onnxruntime-gpu's CUDAExecutionProvider needs ``libcudnn.so.9`` (and
    friends) on the loader path.  On hosts where CUDA is provided via pip
    wheels (nvidia-cudnn-cu*, nvidia-cublas-cu*, ...) rather than a system
    install, those libs live under ``site-packages/nvidia/*/lib`` and are not
    on the dynamic linker's search path.  Setting ``LD_LIBRARY_PATH`` from
    within the running process is too late (glibc cached it at startup), so we
    ``ctypes.CDLL`` the libraries directly to load them into the process
    address space before onnxruntime's CUDA provider dlopen()s them.  This is
    the same technique ``transcribe._preload_nvrtc_builtins`` uses.

    Best-effort and idempotent; safe to call when CUDA isn't present.
    """
    global _cuda_libs_prepared
    if _cuda_libs_prepared:
        return
    _cuda_libs_prepared = True

    import ctypes
    import os
    import sys
    from pathlib import Path

    site_dirs = [Path(p) for p in sys.path if "site-packages" in p or "dist-packages" in p]
    lib_dirs: list[Path] = []
    for sp in site_dirs:
        nvidia = sp / "nvidia"
        if not nvidia.is_dir():
            continue
        for sub in ("cublas", "cudnn", "cuda_runtime", "cufft", "curand"):
            lib = nvidia / sub / "lib"
            if lib.is_dir():
                lib_dirs.append(lib)

    if not lib_dirs:
        return

    # Also update LD_LIBRARY_PATH for any child processes / dependent dlopens.
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [str(d) for d in lib_dirs if str(d) not in existing]
    if parts:
        os.environ["LD_LIBRARY_PATH"] = (
            ":".join(parts) + (f":{existing}" if existing else "")
        )

    # Preload in dependency order: cuBLAS first, then cuDNN sub-libs, then the
    # cuDNN umbrella.  cuDNN's dlopen of its own sub-libraries (cnn/ops/graph/
    # engines) needs them already resident, so load those before libcudnn.so.9.
    def _try_load(path: Path) -> None:
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    for d in lib_dirs:
        if d.name == "lib" and d.parent.name == "cublas":
            for so in sorted(d.glob("libcublas*.so*")):
                _try_load(so)
    for d in lib_dirs:
        if d.parent.name == "cudnn":
            # sub-libraries first, umbrella (libcudnn.so.9) last
            subs = sorted(d.glob("libcudnn_*.so*"))
            umbrella = sorted(d.glob("libcudnn.so*"))
            for so in subs + umbrella:
                _try_load(so)


def _resolve_providers(device: str | None) -> list[str] | None:
    """Map a millet device string to onnxruntime execution providers.

    Returns None to let onnx-asr pick its default ordering when device is
    unknown / not CUDA.
    """
    if device == "cuda":
        # Prefer CUDA, fall back to CPU within the same session.
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cpu":
        return ["CPUExecutionProvider"]
    return None


def load_parakeet(model: str | None = None, device: str | None = None):
    """Load (and cache) the Parakeet onnx-asr model for the given device.

    Raises RuntimeError with actionable guidance if onnx-asr is missing.
    """
    if not parakeet_available():
        raise RuntimeError(
            "Parakeet backend requires the 'onnx-asr' package.\n"
            "  Install it with:  pip install 'millet-pipeline[parakeet]'\n"
            "  Then fetch the model:  millet download parakeet"
        )

    import onnx_asr

    if device == "cuda":
        _prepare_cuda_libs()

    model_name = model or DEFAULT_PARAKEET_MODEL
    providers = _resolve_providers(device)
    key = (model_name, device or "auto")
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    log.info("Loading Parakeet model %s (device=%s)", model_name, device)
    loaded = onnx_asr.load_model(model_name, providers=providers)
    _MODEL_CACHE[key] = loaded
    return loaded


def _load_vad(device: str | None = None):
    import onnx_asr

    key = device or "auto"
    cached = _VAD_CACHE.get(key)
    if cached is not None:
        return cached
    providers = _resolve_providers(device)
    vad = onnx_asr.load_vad(DEFAULT_VAD_MODEL, providers=providers)
    _VAD_CACHE[key] = vad
    return vad


def transcribe_parakeet(
    audio: np.ndarray,
    *,
    model: str | None = None,
    device: str | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """Transcribe a 16 kHz mono float32 waveform with Parakeet via onnx-asr.

    Long audio is automatically chunked through the Silero VAD adapter so we
    stay within Parakeet's per-utterance limit; the adapter returns segments
    with global timestamps which we map straight onto millet's segment shape.

    Args:
        audio: 1-D float32 numpy array at 16 kHz (the same array WhisperX /
            MLX backends receive from ``whisperx.load_audio``).
        model: onnx-asr model name (defaults to English Parakeet v2).
        device: "cuda" | "cpu" | None (auto).
        language: Hint only; Parakeet v2 is English.  Recorded into the
            returned dict for downstream alignment/labeling.

    Returns:
        WhisperX-compatible dict: {"segments": [...], "language": str, "text": str}
    """
    import numpy as np

    asr = load_parakeet(model=model, device=device)
    vad = _load_vad(device=device)

    # Ensure float32 mono at the expected sample rate.  Callers already pass
    # 16 kHz mono, but guard against accidental dtype drift.
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # VAD-segmented recognition: yields an iterator of SegmentResult
    # (start, end, text) with global timestamps already applied.
    seg_results = asr.with_vad(vad).recognize(audio, sample_rate=SAMPLE_RATE)

    segments: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for seg in seg_results:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        start = float(getattr(seg, "start", 0.0))
        end = float(getattr(seg, "end", start))
        segments.append({"start": start, "end": end, "text": text})
        text_parts.append(text)

    return {
        "segments": segments,
        "language": language or "en",
        "text": " ".join(text_parts),
    }


def ensure_parakeet_cached(model: str | None = None) -> bool:
    """Check whether the Parakeet model weights are fully present in the cache.

    Does NOT download.  Returns True only if a matching model dir contains at
    least one non-trivial ``.onnx`` weight file AND has no ``.incomplete``
    download stubs.  This guards against a partial/interrupted download (which
    leaves config.json + vocab.txt but zero-byte ``.onnx`` blobs) being
    mistaken for a finished install.
    """
    from pathlib import Path

    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.is_dir():
        return False

    # Match the specific model's cache dir when a model is given, so asking
    # about model B doesn't report True just because model A is cached.
    # HF cache dirs are named ``models--<org>--<name>``; match on the model's
    # final path component.
    model_name = model or DEFAULT_PARAKEET_MODEL
    leaf = model_name.rsplit("/", 1)[-1]
    glob_pat = f"models--*{leaf}*" if leaf else "models--*parakeet*"

    # Minimum plausible size for a real Parakeet ONNX weight file.  The
    # encoder graph is ~40 MB but its weights live in a sibling
    # ``encoder-model.onnx.data``; the decoder is ~35 MB.  Use a conservative
    # floor that rejects metadata-only stubs but accepts the real graphs.
    MIN_ONNX_BYTES = 1_000_000  # 1 MB

    # We deliberately inspect the *resolved snapshot* files (which follow the
    # symlink into blobs) rather than the blobs dir directly: stale 0-byte
    # ``*.incomplete`` stubs from interrupted earlier attempts can linger in
    # blobs even after a later attempt completes the real download.  A
    # resolved, non-trivial encoder + decoder is the authoritative signal.
    for child in hub.glob(glob_pat):
        snaps = child / "snapshots"
        if not snaps.is_dir():
            continue
        big_onnx = 0
        for onnx in snaps.rglob("*.onnx"):
            try:
                if onnx.stat().st_size >= MIN_ONNX_BYTES:  # stat follows symlink
                    big_onnx += 1
            except OSError:
                continue
        # Parakeet needs at least the encoder + decoder graphs.
        if big_onnx >= 2:
            return True
    return False


def download_parakeet(
    model: str | None = None,
    progress_callback=None,
) -> None:
    """Download the Parakeet ONNX model into the HF cache (explicit fetch).

    Loading the model through onnx-asr triggers the HF download as a side
    effect; we discard the handle afterwards.
    """
    model_name = model or DEFAULT_PARAKEET_MODEL

    def _status(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(f"  {msg}")

    if not parakeet_available():
        raise RuntimeError(
            "Cannot download Parakeet: the 'onnx-asr' package is not installed.\n"
            "  Install it with:  pip install 'millet-pipeline[parakeet]'"
        )

    import onnx_asr

    _status(f"Downloading Parakeet model {model_name} (~600 MB, first run only)...")
    onnx_asr.load_model(model_name)
    _status(f"Downloading VAD model {DEFAULT_VAD_MODEL}...")
    onnx_asr.load_vad(DEFAULT_VAD_MODEL)
    _status("Parakeet model ready.")
