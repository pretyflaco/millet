"""Transcription module using WhisperX with speaker diarization.

Pipeline:
1. Load audio (dual-channel WAV -> mono for transcription)
2. Transcribe with faster-whisper (batched, GPU-accelerated)
3. Align with wav2vec2 for word-level timestamps
4. Diarize with pyannote-audio for speaker labels
5. Merge diarization with transcription
6. Output formatted transcript
"""

from __future__ import annotations

import gc
import importlib.util
import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Fix for CUDA NVRTC JIT compilation: pyannote's wespeaker embedding model
# triggers torch.fft.rfft -> CUDA JIT -> NVRTC, which needs libnvrtc-builtins.so
# matching the CUDA version.  Setting LD_LIBRARY_PATH at runtime is too late
# (libnvrtc.so is already loaded), so we preload the library with ctypes.CDLL
# to make it available in the process address space before NVRTC needs it.


def _preload_nvrtc_builtins():
    """Preload libnvrtc-builtins.so so NVRTC JIT compilation can find it.

    When NVRTC's libnvrtc.so calls dlopen("libnvrtc-builtins.so.X.Y"), the
    dynamic linker checks already-loaded libraries first.  By loading the
    correct version early via ctypes.CDLL, we ensure it's found regardless
    of LD_LIBRARY_PATH at process startup time.
    """
    import ctypes
    import sys

    # Determine the CUDA major.minor that PyTorch / the driver expects.
    # We look for libnvrtc-builtins.so.<major>.<minor> in nvidia pip packages.
    search_dirs: list[Path] = []

    # 1. nvidia.cu<major> packages (e.g. nvidia-cuda-nvrtc 13.x installs here)
    site_dirs = [Path(p) for p in sys.path if "site-packages" in p]
    for sp in site_dirs:
        nvidia_dir = sp / "nvidia"
        if nvidia_dir.is_dir():
            for child in sorted(nvidia_dir.iterdir(), reverse=True):
                if child.name.startswith("cu") and child.name[2:].isdigit():
                    lib_dir = child / "lib"
                    if lib_dir.is_dir():
                        search_dirs.append(lib_dir)

    # 2. nvidia.cuda_nvrtc package (older layout)
    try:
        spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
        if spec and spec.origin:
            pkg_dir = Path(spec.origin).parent / "lib"
            if pkg_dir.is_dir():
                search_dirs.append(pkg_dir)
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    # 3. Common system paths
    search_dirs.extend(
        [
            Path("/usr/local/cuda/lib64"),
            Path("/usr/lib/x86_64-linux-gnu"),
        ]
    )

    # Find the highest-version libnvrtc-builtins.so.*
    candidates: list[Path] = []
    for d in search_dirs:
        if d.is_dir():
            found = sorted(d.glob("libnvrtc-builtins.so.*"))
            candidates.extend(c for c in found if "alt" not in c.name)

    if not candidates:
        return

    def _version_key(p: Path) -> tuple[int, ...]:
        """Extract numeric version tuple from libnvrtc-builtins.so.X.Y"""
        suffix = p.name.removeprefix("libnvrtc-builtins.so.")
        try:
            return tuple(int(x) for x in suffix.split("."))
        except ValueError:
            return (0,)

    # Pick the highest version by numeric comparison (13.0 > 12.8 > 12.4)
    best = max(candidates, key=_version_key)
    try:
        ctypes.CDLL(str(best))
    except OSError:
        pass  # Nothing more we can do

    # Also update LD_LIBRARY_PATH for any child processes
    lib_dir = str(best.parent)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{ld_path}" if ld_path else lib_dir


_preload_nvrtc_builtins()


# Local model aliases: map short names to local CTranslate2 model directories.
# These are populated by offline conversion from HuggingFace models.
_LOCAL_MODEL_ALIASES: dict[str, Path] = {}
_MLX_MODEL_ALIASES: dict[str, str] = {
    "base": "mlx-community/whisper-base",
    "medium": "mlx-community/whisper-medium",
    "large-v2": "mlx-community/whisper-large-v2",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}

_ct2_cache = Path.home() / ".cache"
for _candidate in [
    ("large-v3-turbo", _ct2_cache / "faster-whisper-large-v3-turbo-ct2"),
]:
    if _candidate[1].exists() and (_candidate[1] / "model.bin").exists():
        _LOCAL_MODEL_ALIASES[_candidate[0]] = _candidate[1]


def resolve_model(name: str) -> str:
    """Resolve a model name, checking local aliases first."""
    if name in _LOCAL_MODEL_ALIASES:
        return str(_LOCAL_MODEL_ALIASES[name])
    return name


def resolve_mlx_model(name: str) -> str:
    """Resolve a model alias to an MLX Whisper path or Hugging Face repo."""
    return _MLX_MODEL_ALIASES.get(name, name)


def _mlx_available() -> bool:
    return importlib.util.find_spec("mlx_whisper") is not None


def _parakeet_available() -> bool:
    return importlib.util.find_spec("onnx_asr") is not None


def _apple_silicon() -> bool:
    try:
        import platform
        return platform.system() == "Darwin" and platform.machine().lower() in {
            "arm64",
            "aarch64",
        }
    except Exception:
        return False


def _torch_device_available(device: str) -> bool | None:
    """Return whether the given torch device is available at runtime.

    Returns:
        True/False if torch can be imported and the answer is known.
        None if torch is not installed (caller should skip validation).
    """
    if device == "cpu":
        return True
    try:
        import torch
    except ImportError:
        return None
    if device == "cuda":
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False
    if device == "mps":
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        if mps is None:
            return False
        try:
            return bool(mps.is_available())
        except Exception:
            return False
    # Unknown device string — defer to caller's existing validation.
    return None


def _mps_available() -> bool:
    """Return True if PyTorch MPS backend is available at runtime.

    Returns False both when torch isn't installed and when MPS isn't built
    into the installed torch wheel.  Thin convenience wrapper over
    ``_torch_device_available('mps')`` that maps the None-when-torch-missing
    case to False (since callers picking platform defaults want a boolean).
    """
    return bool(_torch_device_available("mps"))


# ── Alignment model registry ───────────────────────────────────────────────
# Maps language codes to (model_name, model_type) where model_type is
# "torchaudio" or "huggingface".  Only languages we actively support are
# listed here; WhisperX supports ~39 but we only manage downloads for the
# ones the user cares about.

ALIGNMENT_MODELS: dict[str, tuple[str, str]] = {
    "en": ("WAV2VEC2_ASR_BASE_960H", "torchaudio"),
    "de": ("VOXPOPULI_ASR_BASE_10K_DE", "torchaudio"),
    "fr": ("VOXPOPULI_ASR_BASE_10K_FR", "torchaudio"),
    "es": ("VOXPOPULI_ASR_BASE_10K_ES", "torchaudio"),
    "tr": ("mpoyraz/wav2vec2-xls-r-300m-cv7-turkish", "huggingface"),
    "fa": ("jonatasgrosman/wav2vec2-large-xlsr-53-persian", "huggingface"),
}

# torchaudio pipeline names → local .pt filenames in ~/.cache/torch/hub/checkpoints/
_TORCHAUDIO_FILENAMES: dict[str, str] = {
    "WAV2VEC2_ASR_BASE_960H": "wav2vec2_fairseq_base_ls960_asr_ls960.pth",
    "VOXPOPULI_ASR_BASE_10K_DE": "wav2vec2_voxpopuli_base_10k_asr_de.pt",
    "VOXPOPULI_ASR_BASE_10K_FR": "wav2vec2_voxpopuli_base_10k_asr_fr.pt",
    "VOXPOPULI_ASR_BASE_10K_ES": "wav2vec2_voxpopuli_base_10k_asr_es.pt",
}

# Approximate download sizes for user display
_MODEL_SIZES: dict[str, str] = {
    "en": "~360 MB",
    "de": "~360 MB",
    "fr": "~360 MB",
    "es": "~360 MB",
    "tr": "~1.2 GB",
    "fa": "~1.2 GB",
}

from millet.languages import LANG_NAMES as _LANG_NAMES

MODEL_SIZES = _MODEL_SIZES  # public accessor


class AlignmentModelMissing(Exception):
    """Raised when an alignment model is not cached locally.

    Carries enough context for CLI/GUI to display actionable guidance.
    """

    def __init__(self, lang: str):
        model_name, model_type = ALIGNMENT_MODELS.get(lang, ("unknown", "unknown"))
        self.lang = lang
        self.lang_name = _LANG_NAMES.get(lang, lang)
        self.model_name = model_name
        self.model_type = model_type
        self.estimated_size = _MODEL_SIZES.get(lang, "unknown size")
        super().__init__(
            f"Alignment model for {self.lang_name} ({lang}) is not downloaded.\n"
            f"  Model: {model_name} ({model_type}, {self.estimated_size})\n"
            f"  Run:   meet download {lang}"
        )


class OfflineModelMissing(Exception):
    """Raised when an ASR model is not cached locally and downloads are disabled.

    Occurs when ``HF_HUB_OFFLINE`` (or ``TRANSFORMERS_OFFLINE``) is set but the
    requested Hugging Face model has never been cached, so huggingface_hub
    refuses to fetch it.  Carries the resolved model repo so CLI/GUI can show
    actionable guidance instead of a raw traceback.
    """

    def __init__(self, model: str, backend: str = "mlx"):
        self.model = model
        self.backend = backend
        super().__init__(
            f"{backend} model '{model}' is not cached locally and Hugging Face "
            f"downloads are disabled (HF_HUB_OFFLINE is set)."
        )


def _hf_offline() -> bool:
    """Return True if Hugging Face downloads are disabled via env vars."""
    return any(
        os.environ.get(var, "").strip().lower() in {"1", "true", "yes", "on"}
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _mlx_model_cached(repo: str) -> bool:
    """Best-effort check whether an MLX/HF model repo is in the local cache.

    Purely a filesystem check via huggingface_hub; never downloads.  Returns
    True when the check cannot be performed (so we fall through to the real
    load and let it raise its own error rather than blocking a valid run).
    """
    # A local path (not a "org/repo" HF id) is used directly by mlx_whisper.
    if os.path.sep in repo and Path(repo).exists():
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    except Exception:
        return True
    try:
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir(HUGGINGFACE_HUB_CACHE)
        return any(r.repo_id == repo for r in cache.repos)
    except Exception:
        # Fallback: probe for the model config file specifically.
        try:
            hit = try_to_load_from_cache(repo, "config.json")
            return isinstance(hit, str)
        except Exception:
            return True


def check_alignment_model_cached(lang: str) -> bool:
    """Check if the alignment model for *lang* is cached locally.

    Does NOT download anything — purely a filesystem check.

    Returns True if cached, False if missing.
    """
    if lang not in ALIGNMENT_MODELS:
        # Language not in our registry — WhisperX may still support it
        # (it has 39 languages).  We can't check, so assume it's fine
        # and let WhisperX handle the error at runtime.
        return True

    model_name, model_type = ALIGNMENT_MODELS[lang]

    if model_type == "torchaudio":
        filename = _TORCHAUDIO_FILENAMES.get(model_name)
        if not filename:
            return True  # Unknown filename — can't check
        cache_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / filename
        return cache_path.exists()

    elif model_type == "huggingface":
        # HuggingFace models are stored as models--<org>--<model>/
        # with a snapshots/ subdirectory containing the actual weights.
        safe_name = model_name.replace("/", "--")
        model_dir = (
            Path.home() / ".cache" / "huggingface" / "hub" / f"models--{safe_name}"
        )
        if not model_dir.exists():
            return False
        # Check that there's at least one snapshot (complete download)
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            return False
        # At least one non-empty snapshot directory
        return any(snapshots_dir.iterdir())

    return True


def download_alignment_model(
    lang: str,
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Download the alignment model for *lang*.

    Args:
        lang: Language code (e.g. "de", "tr").
        progress_callback: Optional callable(status_message) for progress updates.

    Raises:
        ValueError: If the language is not in our registry.
        RuntimeError: If the download fails.
    """
    if lang not in ALIGNMENT_MODELS:
        supported = ", ".join(sorted(ALIGNMENT_MODELS.keys()))
        raise ValueError(
            f"No alignment model registered for '{lang}'. Supported: {supported}"
        )

    model_name, model_type = ALIGNMENT_MODELS[lang]
    lang_name = _LANG_NAMES.get(lang, lang)
    size = _MODEL_SIZES.get(lang, "unknown size")

    def _status(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(f"  {msg}")

    _status(f"Downloading alignment model for {lang_name} ({model_name}, {size})...")

    if model_type == "torchaudio":
        import torchaudio

        # Load the pipeline — this triggers the download.
        bundle = getattr(torchaudio.pipelines, model_name)
        _status(f"Loading torchaudio bundle {model_name}...")
        bundle.get_model()
        _status(f"Alignment model for {lang_name} downloaded successfully.")

    elif model_type == "huggingface":
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        _status(f"Downloading HuggingFace model {model_name}...")
        Wav2Vec2Processor.from_pretrained(model_name)
        Wav2Vec2ForCTC.from_pretrained(model_name)
        _status(f"Alignment model for {lang_name} downloaded successfully.")


def get_supported_alignment_languages() -> dict[str, dict[str, str]]:
    """Return info about all supported alignment languages.

    Returns dict of lang_code -> {name, model, type, size, cached}.
    """
    result = {}
    for lang, (model_name, model_type) in ALIGNMENT_MODELS.items():
        result[lang] = {
            "name": _LANG_NAMES.get(lang, lang),
            "model": model_name,
            "type": model_type,
            "size": _MODEL_SIZES.get(lang, "unknown"),
            "cached": check_alignment_model_cached(lang),
        }
    return result


# ── GPU resource management ────────────────────────────────────────────────


def ensure_gpu_available(
    progress_callback: Callable[[str], None] | None = None,
) -> None:
    """Ensure GPU VRAM is available for transcription.

    Checks if Ollama has models loaded and unloads them to free VRAM.
    This is transparent — we tell the user what we're doing.
    """
    import json as _json

    def _status(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            print(f"  {msg}")

    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/ps"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return  # Ollama not running — nothing to do

        data = _json.loads(result.stdout)
        models = data.get("models", [])
        if not models:
            return  # No models loaded

        for model_info in models:
            model_name = model_info.get("name", "unknown")
            _status(
                f"Unloading Ollama model ({model_name}) to free GPU memory for transcription..."
            )
            subprocess.run(
                [
                    "curl",
                    "-s",
                    "http://localhost:11434/api/generate",
                    "-d",
                    _json.dumps({"model": model_name, "keep_alive": 0}),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

        # Brief pause to let VRAM actually free up
        import time

        time.sleep(2)
        _status("GPU memory freed.")

    except (subprocess.TimeoutExpired, _json.JSONDecodeError, FileNotFoundError):
        pass  # Ollama not installed or not responding — fine


@dataclass
class TranscriptionConfig:
    """Configuration for the transcription pipeline."""

    model: str = "large-v3-turbo"
    # device defaults to None so __post_init__ can platform-detect:
    # cpu on Apple Silicon (CUDA unavailable on macOS), cuda elsewhere.
    device: str | None = None
    torch_device: str | None = None
    asr_backend: str = "auto"
    mlx_model: str | None = None
    # Parakeet (onnx-asr) model name; None -> module default (English v2).
    parakeet_model: str | None = None
    # When using the parakeet backend, whether to skip WhisperX word-level
    # alignment and trust Parakeet's native (VAD-segment) timestamps.
    #   True  = config "B" (fast; native timestamps)
    #   False = config "C" (Parakeet text + wav2vec2-refined timestamps)
    # This flag exists so the ASR benchmark can measure B vs C and pick the
    # right default; it has no effect on whisperx/mlx backends.
    parakeet_skip_alignment: bool = True
    compute_type: str = "float16"
    batch_size: int = 16
    language: str = "auto"
    hf_token: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    # ── Language auto-detection tuning ──
    # whisperx detects the language from only the FIRST ~30s of each channel,
    # which a misleading opener (e.g. a one-word "Gracias" / cross-talk) can
    # skew — mislabeling an otherwise-English channel.  When language="auto"
    # we instead sample several windows spread across the channel and take the
    # most-confident majority.  Number of windows to sample (whisperx backend).
    language_detection_segments: int = 6
    # Soft team/operator default-language bias.  When set and language="auto",
    # the default wins UNLESS a channel confidently detects another language
    # (>= default_language_override_confidence).  Prevents drift to a
    # low-confidence minority detection for a team that meets in one language.
    default_language: str | None = None
    # Confidence a non-default detection must clear to override default_language.
    default_language_override_confidence: float = 0.70
    # Whether to use the dual-channel layout to improve diarization:
    # If True, the mic channel is labeled as SPEAKER_YOU and the system
    # channel helps confirm remote speaker segments.
    use_dual_channel: bool = True
    # VAD tuning: lower values = more sensitive to quiet/trailing speech.
    # Defaults are tuned below upstream (onset=0.5, offset=0.363) to avoid
    # cutting off the last few seconds of a recording.
    vad_onset: float = 0.40
    vad_offset: float = 0.25
    # Seconds of silence to pad at the end of audio before transcription.
    # Gives the VAD room to properly close the final speech segment.
    audio_pad_seconds: float = 3.0
    # If True, skip the alignment step entirely (no word-level timestamps,
    # fewer segments).  Used when alignment model is missing and the user
    # chose not to download it.
    skip_alignment: bool = False
    # Mixdown mode for stereo recordings:
    #   "dual-diarize" = transcribe channels separately + diarize system
    #                     channel (default; best accuracy for multi-speaker
    #                     stereo — preserves named remotes, no overlap leak)
    #   "mono"         = mix to mono + diarize (legacy; faster, but suffers
    #                     overlap-fragmentation at turn boundaries)
    #   "dual"         = transcribe channels separately, all remotes = REMOTE
    mixdown: str = "dual-diarize"
    # Hybrid channel-energy correction (mono path only): after diarization +
    # YOU/REMOTE labeling, reassign individual segments/words whose audio is
    # *strongly* dominated by the opposite channel from their label.  Fixes
    # turn-boundary leaks (e.g. a remote interjection glued onto the local
    # speaker) without the 2x cost of full dual transcription, and keeps
    # multi-speaker diarization/voiceprint naming intact.  On by default for
    # stereo; disable via --no-channel-correct.
    channel_correct: bool = True
    # Minimum margin by which the opposite channel must dominate a segment's
    # energy ratio before we override its diarized YOU/REMOTE side.  Higher =
    # more conservative (resists mic bleed from open speakers).  The ratio is
    # mic/(mic+sys) in [0,1]; a YOU segment is flipped to REMOTE only when its
    # ratio < (0.5 - margin), and a REMOTE segment to YOU only when its ratio
    # > (0.5 + margin).
    channel_correct_margin: float = 0.30

    # ── Remote-cluster consolidation (dual-diarize path) ──
    # pyannote sometimes over-segments a single remote stream into several
    # clusters (e.g. splitting short backchannel "yeah/cool/awesome" off the
    # main remote into a phantom speaker, which voiceprint matching then
    # mis-names).  After diarizing the system channel we consolidate clusters
    # that are almost certainly the same physical person:
    #   * voiceprint-guided merge: two clusters whose speaker embeddings are
    #     highly similar (cosine >= cluster_merge_similarity) are merged into
    #     the one with the most speech.
    #   * small-cluster absorb: a cluster with less than
    #     cluster_min_speech_seconds of embeddable (>= MIN_SEGMENT_DURATION)
    #     speech is absorbed into the dominant remote cluster.
    # On by default for stereo; disable via --no-consolidate-remote-clusters.
    consolidate_remote_clusters: bool = True
    # Cosine-similarity threshold above which two system clusters are treated
    # as the same speaker and merged.  Conservative so genuinely-distinct
    # remotes who merely sound alike are not collapsed.
    cluster_merge_similarity: float = 0.80
    # A system cluster with less than this many seconds of embeddable speech
    # is absorbed into the dominant remote cluster (phantom backchannel).
    cluster_min_speech_seconds: float = 8.0
    # Ultra-short system segments that diarization left unassigned (would
    # otherwise become a generic REMOTE speaker) are merged into the nearest
    # remote cluster when shorter than this many seconds.
    orphan_merge_max_seconds: float = 1.0
    # ── Isolated-backchannel guard (B1, 0.12.12) ──
    # pyannote sometimes lumps a few early sub-threshold backchannel utterances
    # ("yeah", "ok") into a cluster whose real speaker only appears much later
    # (e.g. someone who joins 25 min in).  Such fragments are below
    # MIN_SEGMENT_DURATION (so they contribute no embedding) yet inherit the
    # cluster's voiceprint-resolved name.  When a fragment is shorter than
    # ``backchannel_max_seconds`` AND is more than
    # ``cluster_isolation_gap_seconds`` from the nearest >=MIN_SEGMENT_DURATION
    # segment of its own cluster, reassign it to the generic REMOTE bucket (the
    # later REMOTE-rescue / labeling step re-homes it).  Conservative: only
    # touches non-embeddable, far-isolated fragments.
    strip_isolated_backchannel: bool = True
    backchannel_max_seconds: float = 1.5
    cluster_isolation_gap_seconds: float = 300.0

    # ── Single-source detection (dual-diarize path) ──
    # The dual-diarize path assumes the mic (left) channel carries exactly one
    # local speaker and only diarizes the system (right) channel.  That is
    # wrong for an in-room recording where multiple people share the mic and
    # the system channel is silent (or merely a copy of the mic): every mic
    # speaker would collapse into a single "YOU".  When the system channel is
    # inactive OR the two channels are near-duplicates, fall back to the mono
    # path (mix down + diarize the combined signal), which splits the in-room
    # speakers correctly.  Enable/disable via --[no-]single-source-fallback.
    single_source_fallback: bool = True
    # System (right) channel is considered inactive when its active-sample RMS
    # is below this fraction of the mic (left) channel's active-sample RMS.
    system_inactive_rms_ratio: float = 0.10
    # The two channels are considered near-duplicate (in-room dual-mono) when
    # their Pearson correlation is at or above this threshold.
    channel_duplicate_corr: float = 0.98

    # ── Per-segment speaker repair (mono / mixdown path) ──
    # After diarization, embed each ASR segment and split a cluster that
    # clearly holds two voices (see millet.speaker_repair).  Disable via
    # --no-speaker-repair.
    speaker_repair: bool = True

    # Internal: set to True by __post_init__ when `device` was auto-flipped
    # to 'cpu' because the requested accelerator was unavailable.  Used to
    # produce an honest annotation in the model-load log line — distinguishes
    # "user explicitly passed --device cpu" from "we fell back".
    _device_auto_fallback: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        if self.mixdown not in ("mono", "dual", "dual-diarize"):
            raise ValueError(
                f"Invalid mixdown mode '{self.mixdown}': must be "
                f"'mono', 'dual', or 'dual-diarize'"
            )
        if self.asr_backend not in ("auto", "whisperx", "mlx", "parakeet"):
            raise ValueError(
                f"Invalid ASR backend '{self.asr_backend}': must be 'auto', "
                f"'whisperx', 'mlx', or 'parakeet'"
            )
        if self.asr_backend == "auto":
            # Parakeet is intentionally NOT an auto candidate yet — it is
            # opt-in via --asr-backend parakeet until benchmark data justifies
            # promoting it (English-only, ONNX, separate timestamp behavior).
            self.asr_backend = "mlx" if _apple_silicon() and _mlx_available() else "whisperx"
        # Resolve model aliases for the selected backend.
        if self.asr_backend == "mlx":
            self.mlx_model = resolve_mlx_model(self.mlx_model or self.model)
        elif self.asr_backend == "parakeet":
            # Parakeet model names are onnx-asr identifiers, not whisper
            # aliases; leave self.model untouched and let the parakeet module
            # resolve its own default when parakeet_model is None.
            # Config "B": trust Parakeet's native VAD-segment timestamps and
            # skip the WhisperX wav2vec2 alignment pass.  Config "C" leaves
            # skip_alignment as the user set it so alignment still runs.
            if self.parakeet_skip_alignment:
                self.skip_alignment = True
        else:
            self.model = resolve_model(self.model)
        # Resolve device defaults.  CUDA is unavailable on macOS, so on Apple
        # Silicon we fall back to cpu for the ASR/whisperx device and prefer
        # mps for the torch device (alignment + diarization).
        if self.device is None:
            self.device = "cpu" if _apple_silicon() else "cuda"
        if self.torch_device is None:
            if _apple_silicon():
                self.torch_device = "mps" if _mps_available() else "cpu"
            else:
                self.torch_device = self.device

        # Validate device availability when torch is installed.  We deliberately
        # skip validation when torch can't be imported so that
        # `TranscriptionConfig` remains constructible in torch-less test
        # environments and lightweight CLI helpers.
        #
        # When the requested device is not available we automatically fall back
        # to CPU instead of raising — this handles the common case where CUDA
        # was requested but no GPU is present (e.g. running on a laptop or
        # inside a container without GPU passthrough).
        for field_name, value in (
            ("device", self.device),
            ("torch_device", self.torch_device),
        ):
            available = _torch_device_available(value)
            if available is None:
                continue
            if not available:
                fallback = "cpu"
                log.warning(
                    "%s='%s' is not available, falling back to '%s'",
                    field_name, value, fallback,
                )
                if field_name == "device":
                    self.device = fallback
                    self._device_auto_fallback = True
                    if self.compute_type == "float16":
                        log.warning(
                            "compute_type='float16' is unsupported on CPU, "
                            "downgrading to 'int8'"
                        )
                        self.compute_type = "int8"
                elif field_name == "torch_device":
                    self.torch_device = fallback

        if self.hf_token is None:
            self.hf_token = os.environ.get("HF_TOKEN")
        if self.hf_token is None:
            # Try reading from huggingface-cli cache
            token_path = Path.home() / ".cache" / "huggingface" / "token"
            if token_path.exists():
                self.hf_token = token_path.read_text().strip()


@dataclass
class Speaker:
    """A speaker in the transcript."""

    id: str
    label: str | None = None  # User-assigned name


@dataclass
class Segment:
    """A single segment of the transcript."""

    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[dict] | None = None


@dataclass
class Transcript:
    """Complete transcript with metadata."""

    segments: list[Segment]
    speakers: list[Speaker]
    language: str
    audio_file: str
    duration: float | None = None

    def to_text(self) -> str:
        """Plain text output with speaker labels."""
        lines = []
        for seg in self.segments:
            speaker = seg.speaker or "UNKNOWN"
            start = _fmt_time(seg.start)
            end = _fmt_time(seg.end)
            lines.append(f"[{start} --> {end}] {speaker}: {seg.text.strip()}")
        return "\n".join(lines)

    def to_srt(self) -> str:
        """SRT subtitle format with speaker labels."""
        lines = []
        for i, seg in enumerate(self.segments, 1):
            speaker = seg.speaker or "UNKNOWN"
            start = _fmt_srt_time(seg.start)
            end = _fmt_srt_time(seg.end)
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(f"[{speaker}] {seg.text.strip()}")
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        """JSON output with full detail."""
        data = {
            # Name only, never the absolute path: this JSON is pushed to the
            # team repo as transcript.json and leaked the server's
            # /home/<user>/vezir-data/sessions/<id>/ layout (audit L-7).
            # Every reader uses Path(audio_file).stem or the filename date.
            "audio_file": Path(self.audio_file).name if self.audio_file else "",
            "language": self.language,
            "duration": self.duration,
            "speakers": [{"id": s.id, "label": s.label} for s in self.speakers],
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "speaker": seg.speaker,
                    "words": seg.words,
                }
                for seg in self.segments
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def save(
        self, output_dir: str | Path, basename: str | None = None
    ) -> dict[str, Path]:
        """Save transcript in all formats. Returns dict of format -> filepath."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if basename is None:
            basename = Path(self.audio_file).stem

        files = {}
        for fmt, ext, content in [
            ("text", ".txt", self.to_text()),
            ("srt", ".srt", self.to_srt()),
            ("json", ".json", self.to_json()),
        ]:
            path = output_dir / f"{basename}{ext}"
            path.write_text(content, encoding="utf-8")
            files[fmt] = path

        return files


from millet.utils import fmt_srt_time as _fmt_srt_time
from millet.utils import fmt_time as _fmt_time


def _extract_mono(audio_file: Path, channel: int = 0) -> Path:
    """Extract a single channel from a stereo WAV file.

    Args:
        audio_file: Path to stereo WAV file.
        channel: 0 for left (mic), 1 for right (system).

    Returns:
        Path to temporary mono WAV file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    # Use ffmpeg to extract a single channel
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_file),
        "-filter_complex",
        f"[0:a]pan=mono|c0=c{channel}[out]",
        "-map",
        "[out]",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        tmp.name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Don't orphan the temp file when ffmpeg fails.
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise RuntimeError(f"Failed to extract channel {channel}: {result.stderr}") from None
    return Path(tmp.name)


def _load_stereo_int16(audio_file: Path):
    """Decode a stereo file to (left, right) int16 numpy arrays via ffmpeg.

    Returns ``(left, right)`` float32 arrays, or ``None`` if decoding fails or
    numpy is unavailable.  Works for any ffmpeg-readable format (wav, ogg, …).
    """
    try:
        import numpy as np
    except ImportError:
        return None
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(audio_file),
        "-ac", "2", "-ar", "16000", "-f", "s16le", "-c:a", "pcm_s16le", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
    except Exception:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    data = np.frombuffer(result.stdout, dtype=np.int16)
    if data.size < 2:
        return None
    data = data[: (data.size // 2) * 2].reshape(-1, 2).astype(np.float32)
    return data[:, 0], data[:, 1]


def _is_single_source_stereo(audio_file: Path, config) -> bool:
    """True if a stereo recording is effectively single-source.

    Two cases the dual-diarize path mishandles (it labels the whole mic
    channel as one "YOU" speaker):

    * **System (right) channel inactive** — its active-sample RMS is below
      ``system_inactive_rms_ratio`` of the mic channel's (an in-room
      recording: everyone is on the mic, nothing on system audio).
    * **Channels near-duplicate** — Pearson correlation >=
      ``channel_duplicate_corr`` (dual-mono: the same mixed signal on both
      channels, so the "system" channel adds no separate speaker).

    In either case the caller should fall back to the mono path so the
    combined-signal diarization can split multiple in-room speakers.

    Conservative on failure: returns False (keep dual-diarize) if the audio
    can't be analyzed, so genuine remote calls are never mis-routed.
    """
    chans = _load_stereo_int16(audio_file)
    if chans is None:
        return False
    try:
        import numpy as np
    except ImportError:
        return False
    left, right = chans

    # ── System-channel inactivity (per-channel active-sample RMS) ──
    silence_thr = 50.0  # ~-50 dBFS for 16-bit (matches _mixdown_to_mono)
    left_active = left[np.abs(left) > silence_thr]
    right_active = right[np.abs(right) > silence_thr]
    left_rms = float(np.sqrt(np.mean(left_active**2))) if left_active.size else 0.0
    right_rms = float(np.sqrt(np.mean(right_active**2))) if right_active.size else 0.0
    if left_rms > 0.0:
        if right_rms <= config.system_inactive_rms_ratio * left_rms:
            return True
    elif right_rms <= 0.0:
        # Both channels silent — nothing to diarize per-channel; mono is fine.
        return True

    # ── Near-duplicate channels (Pearson correlation) ──
    n = min(left.size, right.size)
    if n >= 16000:  # at least ~1 s at 16 kHz before trusting correlation
        a = left[:n]
        b = right[:n]
        if a.std() > 1e-6 and b.std() > 1e-6:
            corr = float(np.corrcoef(a, b)[0, 1])
            if corr >= config.channel_duplicate_corr:
                return True
    return False


def _mixdown_to_mono(audio_file: Path) -> Path:
    """Mix both channels to mono with RMS normalization for transcription.

    Normalizes each channel to equal RMS before averaging so that both
    your mic and the remote participants contribute equally to the mono
    signal.  This prevents the common failure mode where a quiet mic
    channel causes Whisper to miss all remote speech (which lives only
    on the system-audio channel).

    Falls back to a simple ffmpeg average if numpy is unavailable.
    """
    import wave

    try:
        import numpy as np
    except ImportError:
        # Fallback: simple ffmpeg average of both channels.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_file),
            "-filter_complex",
            "[0:a]pan=mono|c0=0.5*c0+0.5*c1[out]",
            "-map",
            "[out]",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Mono mixdown failed: {result.stderr}") from None
        return Path(tmp.name)

    # ── Read stereo WAV (only works for .wav files) ──
    try:
        with wave.open(str(audio_file), "r") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception:
        # Non-WAV format (e.g. OGG/Opus) — fall through to ffmpeg.
        n_channels = 0
        sampwidth = 0
        framerate = 0
        raw = b""

    if n_channels != 2 or sampwidth != 2:
        # Not the expected stereo 16-bit — fall through to ffmpeg.
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(audio_file),
            "-filter_complex",
            "[0:a]pan=mono|c0=0.5*c0+0.5*c1[out]",
            "-map",
            "[out]",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise RuntimeError(f"Mono mixdown failed: {result.stderr}")
        return Path(tmp.name)

    data = np.frombuffer(raw, dtype=np.int16).reshape(-1, 2)
    left = data[:, 0].astype(np.float32)  # mic (YOU)
    right = data[:, 1].astype(np.float32)  # system (REMOTE)

    # ── RMS of each channel (ignore near-silence) ──
    silence_thr = 50.0  # ~-50 dBFS for 16-bit
    left_active = left[np.abs(left) > silence_thr]
    right_active = right[np.abs(right) > silence_thr]

    left_rms = np.sqrt(np.mean(left_active**2)) if len(left_active) > 0 else 0.0
    right_rms = np.sqrt(np.mean(right_active**2)) if len(right_active) > 0 else 0.0

    # ── Normalize to equal RMS, then average ──
    if left_rms > 0 and right_rms > 0:
        # Scale the quieter channel up to match the louder one.
        target_rms = max(left_rms, right_rms)
        left_scaled = left * (target_rms / left_rms)
        right_scaled = right * (target_rms / right_rms)
        mono = (left_scaled + right_scaled) * 0.5
    elif right_rms > 0:
        # Mic is dead — use system channel only.
        mono = right
    elif left_rms > 0:
        # System channel is dead — use mic only.
        mono = left
    else:
        # Both silent — just average.
        mono = (left + right) * 0.5

    # Clip to int16 range and convert.
    mono = np.clip(mono, -32768, 32767).astype(np.int16)

    # ── Write mono WAV ──
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, "w") as wf_out:
        wf_out.setnchannels(1)
        wf_out.setsampwidth(2)
        wf_out.setframerate(framerate)
        wf_out.writeframes(mono.tobytes())

    return Path(tmp.name)


def get_audio_duration(audio_file: Path) -> float:
    """Get duration of an audio file in seconds."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "csv=p=0",
        str(audio_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _empty_torch_cache(torch_module, device: str | None) -> None:
    """Best-effort cache release for torch-backed devices."""
    try:
        if device == "cuda":
            torch_module.cuda.empty_cache()
        elif device == "mps" and hasattr(torch_module, "mps"):
            torch_module.mps.empty_cache()
    except Exception:
        log.debug("failed to empty torch cache for device %s", device, exc_info=True)


def _empty_torch_caches(torch_module, config: TranscriptionConfig) -> None:
    """Release cache once per configured torch-backed device."""
    _empty_torch_cache(torch_module, config.device)
    if config.torch_device != config.device:
        _empty_torch_cache(torch_module, config.torch_device)


def _load_whisperx_asr_model(config: TranscriptionConfig, language: str | None):
    import whisperx

    vad_options = {
        "vad_onset": config.vad_onset,
        "vad_offset": config.vad_offset,
    }
    device_note = ""
    if config.device == "cpu":
        if config._device_auto_fallback:
            device_note = " (fallback — no GPU)"
        else:
            device_note = " (forced)"
    print(
        f"  Loading model: {config.model} ({config.compute_type}) on {config.device}{device_note}"
    )
    return whisperx.load_model(
        config.model,
        config.device,
        compute_type=config.compute_type,
        language=language,
        vad_options=vad_options,
    )


def _run_whisperx_asr(
    model, audio, config: TranscriptionConfig, language: str | None = None
):
    print(
        f"  Transcribing (VAD onset={config.vad_onset}, offset={config.vad_offset}"
        + (f", language={language}" if language else "")
        + ")..."
    )
    # Passing ``language`` forces the decode tokenizer for this channel and
    # skips whisperx's first-30s auto-detect (which can drift to the wrong
    # language and transcribe e.g. English audio as Spanish).  When None,
    # whisperx auto-detects as before.
    return model.transcribe(
        audio, batch_size=config.batch_size, language=language
    )


_mlx_vad_note_logged = False


def _transcribe_asr(
    audio,
    config: TranscriptionConfig,
    language: str | None,
    whisperx_model=None,
):
    """Run the selected ASR backend and return a WhisperX-compatible result."""
    if config.asr_backend == "mlx":
        import mlx_whisper

        # MLX Whisper has its own internal VAD/segmentation and does not honor
        # WhisperX-style vad_onset/vad_offset.  Emit an info-level note so the
        # inert behavior is discoverable, regardless of whether the user passed
        # the defaults or non-default values.  Logged once per process to avoid
        # spamming logs on dual-channel runs.
        global _mlx_vad_note_logged
        if not _mlx_vad_note_logged:
            log.info(
                "MLX backend ignores VAD options (vad_onset/vad_offset); "
                "MLX uses its own internal segmentation."
            )
            _mlx_vad_note_logged = True

        print(f"  Loading MLX model: {config.mlx_model}")
        if _hf_offline() and not _mlx_model_cached(config.mlx_model):
            raise OfflineModelMissing(config.mlx_model, backend="mlx")
        decode_options = {}
        if language is not None:
            decode_options["language"] = language
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=config.mlx_model,
            verbose=False,
            word_timestamps=False,
            condition_on_previous_text=False,
            **decode_options,
        )
        segments = [
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": seg.get("text", ""),
            }
            for seg in result.get("segments", [])
        ]
        return {
            "segments": segments,
            "language": result.get("language") or language or "en",
            "text": result.get("text", ""),
        }

    if config.asr_backend == "parakeet":
        from millet.parakeet import transcribe_parakeet

        print(f"  Loading Parakeet model: {config.parakeet_model or 'parakeet-tdt-0.6b-v2 (en)'}")
        return transcribe_parakeet(
            audio,
            model=config.parakeet_model,
            device=config.device,
            language=language,
        )

    if whisperx_model is not None:
        return _run_whisperx_asr(whisperx_model, audio, config, language)

    model = _load_whisperx_asr_model(config, language)
    try:
        return _run_whisperx_asr(model, audio, config, language)
    finally:
        del model


def _transcribe_dual_channel(
    audio_file: Path, config: TranscriptionConfig, duration: float
) -> Transcript:
    """Transcribe each stereo channel separately and merge results.

    Used when mixdown="dual" — instead of mixing both channels into mono
    (which causes WhisperX to suppress the quieter voice), this transcribes
    the mic and system channels independently and merges by timestamp.

    Skips diarization entirely: channel identity = speaker identity.
    """
    import numpy as np
    import torch
    import whisperx

    mic_path = None
    sys_path = None
    asr_model = None

    try:
        mic_path = _extract_mono(audio_file, channel=0)
        sys_path = _extract_mono(audio_file, channel=1)

        whisper_lang = None if config.language == "auto" else config.language

        # Pre-compute padding
        pad_samples = (
            int(config.audio_pad_seconds * 16000) if config.audio_pad_seconds > 0 else 0
        )

        if config.asr_backend == "whisperx":
            asr_model = _load_whisperx_asr_model(config, whisper_lang)

        # ── Mic channel: detect → bias → transcribe with that language ──
        print("  Transcribing mic channel (left)...")
        mic_audio = whisperx.load_audio(str(mic_path))
        if pad_samples > 0:
            mic_audio = np.concatenate(
                [mic_audio, np.zeros(pad_samples, dtype=mic_audio.dtype)]
            )
        mic_lang, mic_conf, mic_decode_lang = _resolve_channel_language(
            asr_model, mic_audio, config, whisper_lang, "mic"
        )
        mic_result = _transcribe_asr(
            mic_audio, config, mic_decode_lang, whisperx_model=asr_model
        )

        # ── System channel: detect → bias → transcribe ──
        print("  Transcribing system channel (right)...")
        sys_audio = whisperx.load_audio(str(sys_path))
        if pad_samples > 0:
            sys_audio = np.concatenate(
                [sys_audio, np.zeros(pad_samples, dtype=sys_audio.dtype)]
            )
        sys_lang, sys_conf, sys_decode_lang = _resolve_channel_language(
            asr_model, sys_audio, config, whisper_lang, "system"
        )
        sys_result = _transcribe_asr(
            sys_audio, config, sys_decode_lang, whisperx_model=asr_model
        )

        if asr_model is not None:
            del asr_model
            asr_model = None

        # Free transcription model
        gc.collect()
        _empty_torch_caches(torch, config)

        # Transcript/summary language = the channel with the most speech
        # (the mic/local speaker may speak only a minority language).
        detected_language = _dominant_channel_language(
            mic_result.get("segments", []),
            sys_result.get("segments", []),
            mic_lang,
            sys_lang,
        )
        if config.language == "auto":
            dominant_conf = sys_conf if detected_language == sys_lang else mic_conf
            detected_language = _apply_default_language_bias(
                detected_language, dominant_conf, config
            )
            print(f"  Summary/transcript language: {detected_language}")

        # ── Align each channel with the language its text was DECODED in ──
        mic_result = _align_channel(
            mic_result, mic_audio, mic_decode_lang or mic_lang,
            config, torch, whisperx,
        )
        sys_result = _align_channel(
            sys_result, sys_audio, sys_decode_lang or sys_lang,
            config, torch, whisperx,
        )

        # ── Merge segments ──
        max_t = duration if duration and duration > 0 else float("inf")
        segments: list[Segment] = []

        for seg in mic_result["segments"]:
            seg_start = min(seg["start"], max_t)
            seg_end = min(seg["end"], max_t)
            if seg_end <= seg_start:
                continue
            segments.append(
                Segment(
                    start=seg_start,
                    end=seg_end,
                    text=seg["text"],
                    speaker="YOU",
                    words=seg.get("words"),
                )
            )

        for seg in sys_result["segments"]:
            seg_start = min(seg["start"], max_t)
            seg_end = min(seg["end"], max_t)
            if seg_end <= seg_start:
                continue
            segments.append(
                Segment(
                    start=seg_start,
                    end=seg_end,
                    text=seg["text"],
                    speaker="REMOTE",
                    words=seg.get("words"),
                )
            )

        segments.sort(key=lambda s: s.start)

        speakers = [
            Speaker(id="YOU", label="YOU"),
            Speaker(id="REMOTE", label="REMOTE"),
        ]

        return Transcript(
            segments=segments,
            speakers=speakers,
            language=detected_language,
            audio_file=str(audio_file),
            duration=duration,
        )

    finally:
        if asr_model is not None:
            del asr_model
        for p in (mic_path, sys_path):
            if p is not None:
                try:
                    p.unlink()
                except OSError:
                    pass


def _segments_total_seconds(segments: list) -> float:
    """Total speech duration across a list of segments (start/end dicts)."""
    return float(sum(max(0.0, float(s["end"]) - float(s["start"])) for s in segments))


def _segment_gap(segment, mid: float) -> float:
    """Temporal gap (seconds) between time ``mid`` and a segment's span.

    0.0 when ``mid`` falls inside ``[start, end]``; otherwise the distance to
    the nearer edge.
    """
    start = float(segment["start"])
    end = float(segment["end"])
    if mid < start:
        return start - mid
    if mid > end:
        return mid - end
    return 0.0


def _nearest_segment(segments: list, mid: float):
    """Return the segment whose time span is temporally nearest ``mid``.

    Returns ``None`` when ``segments`` is empty.  Defined at module scope (not
    as a per-iteration closure) so it neither captures a loop variable nor
    trips Ruff's late-binding guard (B023).
    """
    if not segments:
        return None
    return min(segments, key=lambda s: _segment_gap(s, mid))


def _detect_language_multiwindow(
    model, audio, config: TranscriptionConfig
) -> tuple[str, float]:
    """Detect language by sampling several windows across the audio.

    whisperx labels a channel from only the first ~30s, which a misleading
    opener can skew.  This asks the underlying faster-whisper model to sample
    ``config.language_detection_segments`` windows and returns
    ``(language, confidence)``.  Returns ``(None, 0.0)`` if the model doesn't
    expose ``detect_language`` (older whisperx) or detection fails, so the
    caller can let the backend auto-detect rather than force a wrong language.
    """
    fw = getattr(model, "model", None)
    detect = getattr(fw, "detect_language", None)
    if detect is None:
        return (None, 0.0)
    try:
        lang, prob, _all = detect(
            audio,
            vad_filter=True,
            language_detection_segments=max(1, config.language_detection_segments),
        )
        return (lang, float(prob))
    except Exception as exc:  # pragma: no cover - backend variance
        print(f"  Warning: multi-window language detection failed ({exc})")
        return (None, 0.0)


def _apply_default_language_bias(
    detected_lang: str,
    detected_conf: float,
    config: TranscriptionConfig,
) -> str:
    """Soft team/operator default-language bias.

    When ``config.default_language`` is set, it wins UNLESS the detected
    language differs from it AND the detection cleared
    ``config.default_language_override_confidence``.  Prevents drift to a
    low-confidence minority detection for a team that meets in one language.
    """
    default = config.default_language
    if not default:
        return detected_lang
    if detected_lang == default:
        return detected_lang
    if detected_conf >= config.default_language_override_confidence:
        return detected_lang
    print(
        f"  Language: keeping default '{default}' over low-confidence "
        f"'{detected_lang}' ({detected_conf:.2f} < "
        f"{config.default_language_override_confidence:.2f})"
    )
    return default


def _resolve_channel_language(
    asr_model,
    audio,
    config: TranscriptionConfig,
    whisper_lang: str | None,
    channel_name: str,
) -> tuple[str, float, str | None]:
    """Resolve the language to FORCE for a channel's ASR decode.

    Returns ``(reported_lang, confidence, decode_lang)`` where:

    * ``reported_lang`` is the language used for alignment + metadata (the
      multi-window detection result, or the explicit ``--language``).
    * ``confidence`` is the detection confidence (0.0 when not auto-detecting).
    * ``decode_lang`` is what to pass to the ASR ``transcribe(language=...)``
      call — the default-language-biased language so a low-confidence minority
      detection is overridden by the team/operator default, preventing whisperx
      from transcribing e.g. English audio as Spanish.

    When ``config.language`` is not ``"auto"`` (explicit), that language is used
    verbatim for both reporting and decoding.  For non-whisperx backends we
    cannot pre-detect, so we fall back to ``whisper_lang`` (which is the biased
    default when set) and let the backend auto-detect otherwise.
    """
    # Explicit language: force it everywhere.
    if config.language != "auto":
        return whisper_lang or "en", 0.0, whisper_lang

    if config.asr_backend != "whisperx" or asr_model is None:
        # Can't pre-detect; honor an explicit default as the decode language,
        # else let the backend auto-detect (decode_lang=None).
        default = config.default_language
        return (default or "en"), 0.0, default

    lang, conf = _detect_language_multiwindow(asr_model, audio, config)
    if lang is None:
        # Detection unavailable/failed.  Prefer an explicit operator default;
        # otherwise let the backend auto-detect (decode_lang=None) rather than
        # forcing English on a non-English meeting.
        default = config.default_language
        print(
            f"  Language detection unavailable for {channel_name}; "
            + (f"using default '{default}'." if default else "letting ASR auto-detect.")
        )
        return (default or "en"), 0.0, default
    decode_lang = _apply_default_language_bias(lang, conf, config)
    print(
        f"  Detected {channel_name} language: {lang} ({conf:.2f})"
        + (f" -> decoding as '{decode_lang}'" if decode_lang != lang else "")
    )
    return lang, conf, decode_lang


def _dominant_channel_language(
    mic_segments: list,
    sys_segments: list,
    mic_lang: str | None,
    sys_lang: str | None,
) -> str:
    """Pick the transcript/summary language from the channel with more speech.

    In the dual-channel paths each channel is transcribed (and its language
    detected) independently.  The mic channel is the local speaker, who may
    speak a *minority* language (e.g. a few Portuguese asides), while the bulk
    of the meeting is on the system channel in another language.  Choosing the
    language by total speech duration keeps the summary in the meeting's
    dominant language instead of the local speaker's incidental one.

    Ties (or a missing language) fall back to the mic language, then English.
    """
    mic_lang = mic_lang or "en"
    sys_lang = sys_lang or "en"
    if mic_lang == sys_lang:
        return mic_lang
    mic_s = _segments_total_seconds(mic_segments)
    sys_s = _segments_total_seconds(sys_segments)
    # Dominant by speech duration; mic wins exact ties (local speaker default).
    return sys_lang if sys_s > mic_s else mic_lang


def _align_channel(result, audio, lang, config, torch, whisperx):
    """Align one channel's segments with its OWN detected language model.

    Returns the aligned result (or the original on non-fatal failure).  Raises
    AlignmentModelMissing when the language has a known alignment model that
    isn't cached (so the caller can prompt a download) — matching the prior
    single-channel behavior.
    """
    if config.skip_alignment:
        return result
    if lang in ALIGNMENT_MODELS and not check_alignment_model_cached(lang):
        gc.collect()
        _empty_torch_cache(torch, config.torch_device)
        raise AlignmentModelMissing(lang)
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=lang,
            device=config.torch_device,
        )
        aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            config.torch_device,
            return_char_alignments=False,
        )
        del model_a
        gc.collect()
        _empty_torch_cache(torch, config.torch_device)
        return aligned
    except AlignmentModelMissing:
        raise
    except Exception as align_exc:
        if lang in ALIGNMENT_MODELS:
            raise AlignmentModelMissing(lang) from align_exc
        print(
            f"  Warning: alignment failed for '{lang}' ({align_exc}), "
            "continuing without word-level timestamps"
        )
        return result


def _consolidate_remote_clusters(
    sys_segments: list,
    config: TranscriptionConfig,
    embed_fn=None,
) -> dict[str, str]:
    """Compute a speaker-id remap that consolidates over-segmented remotes.

    pyannote occasionally splits one physical remote into several clusters
    (e.g. peeling short backchannel "yeah/cool" off the main speaker into a
    phantom).  This produces a ``{old_speaker_id: new_speaker_id}`` mapping
    that merges:

      1. **Voiceprint-guided**: two clusters whose embeddings are highly
         similar (cosine >= ``config.cluster_merge_similarity``) are merged
         into the one with more embeddable speech.  Requires ``embed_fn``.
      2. **Small-cluster absorb**: a cluster with less than
         ``config.cluster_min_speech_seconds`` of embeddable
         (>= MIN_SEGMENT_DURATION) speech is absorbed into the dominant
         remote cluster (the one with the most embeddable speech).

    Args:
        sys_segments: system-channel segments, each a dict-like with
            ``start``, ``end`` and ``speaker`` (None/missing = unassigned).
        config: transcription config (thresholds).
        embed_fn: optional ``speaker_id -> np.ndarray | None`` returning an
            L2-normalized embedding for that cluster.  When None, only the
            small-cluster absorb (rule 2) runs.

    Returns:
        ``{old_id: new_id}`` for ids that should be remapped.  Ids absent
        from the dict are unchanged.  Only real cluster ids (non-empty
        ``speaker``) participate; unassigned segments are handled separately
        by the orphan-merge step.
    """
    from millet.voiceprint import MIN_SEGMENT_DURATION

    # Embeddable speech per cluster (sum of >= MIN_SEGMENT_DURATION segments).
    speech: dict[str, float] = {}
    for seg in sys_segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        dur = float(seg["end"]) - float(seg["start"])
        if dur >= MIN_SEGMENT_DURATION:
            speech[spk] = speech.get(spk, 0.0) + dur
        else:
            speech.setdefault(spk, 0.0)

    cluster_ids = sorted(speech.keys())
    if len(cluster_ids) <= 1:
        return {}

    # Dominant cluster = most embeddable speech (tie-break: id order).
    dominant = max(cluster_ids, key=lambda c: (speech[c], c))

    # Union-find so chained merges resolve to a single representative.
    parent = {c: c for c in cluster_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        # Merge the cluster with less speech INTO the one with more.
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        keep, drop = (ra, rb) if (speech[ra], ra) >= (speech[rb], rb) else (rb, ra)
        parent[drop] = keep

    # ── Rule 1: voiceprint-guided merge ──
    if embed_fn is not None:
        import numpy as np

        embeddings: dict[str, Any] = {}
        for c in cluster_ids:
            try:
                emb = embed_fn(c)
            except Exception:
                emb = None
            if emb is not None:
                embeddings[c] = emb
        emb_ids = [c for c in cluster_ids if c in embeddings]
        for i in range(len(emb_ids)):
            for j in range(i + 1, len(emb_ids)):
                a, b = emb_ids[i], emb_ids[j]
                sim = float(np.dot(embeddings[a], embeddings[b]))
                if sim >= config.cluster_merge_similarity:
                    union(a, b)

    # ── Rule 2: small-cluster absorb ──
    # Absorb any cluster below the speech floor into the dominant cluster.
    for c in cluster_ids:
        if c == dominant:
            continue
        if speech[c] < config.cluster_min_speech_seconds:
            union(c, dominant)

    # Build the remap (only entries that actually change).
    remap: dict[str, str] = {}
    for c in cluster_ids:
        root = find(c)
        if root != c:
            remap[c] = root
    return remap


def _merge_orphan_system_segments(
    sys_segments: list,
    config: TranscriptionConfig,
) -> None:
    """Attach ultra-short unassigned system segments to the nearest remote.

    Diarization sometimes leaves a brief segment with no speaker; without
    this it would surface as a generic ``REMOTE`` speaker (e.g. a 0.4s
    "Should be." one-liner).  Such segments shorter than
    ``config.orphan_merge_max_seconds`` are reassigned in-place to the
    temporally-nearest assigned remote cluster.  Segments are mutated
    (``seg["speaker"]`` set).  No-op when there are no assigned clusters.
    """
    assigned = [s for s in sys_segments if s.get("speaker")]
    if not assigned:
        return
    for seg in sys_segments:
        if seg.get("speaker"):
            continue
        dur = float(seg["end"]) - float(seg["start"])
        if dur > config.orphan_merge_max_seconds:
            continue
        mid = (float(seg["start"]) + float(seg["end"])) / 2.0
        nearest = _nearest_segment(assigned, mid)
        if nearest is not None:
            seg["speaker"] = nearest["speaker"]


def _isolated_backchannel_outliers(
    sys_segments: list,
    config: TranscriptionConfig,
) -> list[int]:
    """Indices of sub-threshold segments isolated from their cluster's mass.

    See ``docs/spike-diarization-cluster-bleed.md`` (issue #2).  pyannote can
    lump a few early sub-``MIN_SEGMENT_DURATION`` backchannel utterances into a
    cluster whose real speaker only appears much later.  Those fragments carry
    no embedding weight but inherit the cluster's resolved name.

    A segment is flagged when:
      * its duration < ``config.backchannel_max_seconds``; AND
      * its cluster has at least one ``>= MIN_SEGMENT_DURATION`` ("real")
        segment; AND
      * the nearest such real segment of the SAME cluster is more than
        ``config.cluster_isolation_gap_seconds`` away in time.

    Returns indices into ``sys_segments`` (does not mutate).  Conservative by
    design — only isolated, non-embeddable fragments qualify.
    """
    from millet.voiceprint import MIN_SEGMENT_DURATION

    # Per cluster: list of (start, end) for "real" (embeddable) segments.
    real_by_cluster: dict[str, list[tuple[float, float]]] = {}
    for seg in sys_segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        s, e = float(seg["start"]), float(seg["end"])
        if (e - s) >= MIN_SEGMENT_DURATION:
            real_by_cluster.setdefault(spk, []).append((s, e))

    flagged: list[int] = []
    for i, seg in enumerate(sys_segments):
        spk = seg.get("speaker")
        if not spk:
            continue
        s, e = float(seg["start"]), float(seg["end"])
        if (e - s) >= config.backchannel_max_seconds:
            continue
        real = real_by_cluster.get(spk)
        if not real:
            continue  # cluster is ALL backchannel — leave it (not this case)
        mid = (s + e) / 2.0
        nearest_gap = min(
            (max(rs - mid, mid - re, 0.0) for rs, re in real),
            default=float("inf"),
        )
        if nearest_gap > config.cluster_isolation_gap_seconds:
            flagged.append(i)
    return flagged


def _consolidate_dual_diarize_speakers(
    sys_result: dict,
    sys_audio,
    config: TranscriptionConfig,
) -> None:
    """Apply remote-cluster consolidation + orphan-merge to ``sys_result``.

    Mutates ``sys_result["segments"]`` in place: remaps over-segmented
    speaker ids (voiceprint-guided merge + small-cluster absorb) and attaches
    ultra-short unassigned segments to the nearest remote.  Logs a summary of
    any merges performed.  Embedding failures degrade gracefully to the
    speech-duration heuristic only.
    """
    from millet import voiceprint as _vp

    sys_segments = sys_result.get("segments") or []
    if not sys_segments:
        return

    # Build an embedder over the system audio for voiceprint-guided merging.
    embed_fn = None
    try:
        inference = _vp._get_inference()

        def embed_fn(speaker_id: str):
            segs = [
                (float(s["start"]), float(s["end"]))
                for s in sys_segments
                if s.get("speaker") == speaker_id
                and (float(s["end"]) - float(s["start"])) >= _vp.MIN_SEGMENT_DURATION
            ]
            if not segs:
                return None
            segs.sort(key=lambda t: t[1] - t[0], reverse=True)
            return _vp._embed_segments(
                sys_audio, 16000, segs[: _vp.MAX_SEGMENTS_PER_SPEAKER], inference
            )
    except Exception as exc:
        print(f"  (embedding unavailable for consolidation: {exc}; "
              "using speech-duration heuristic only)")
        embed_fn = None

    remap = _consolidate_remote_clusters(sys_segments, config, embed_fn=embed_fn)
    if remap:
        # Apply to both segment-level speaker and word-level speaker tags.
        for seg in sys_segments:
            spk = seg.get("speaker")
            if spk in remap:
                seg["speaker"] = remap[spk]
            for w in seg.get("words") or []:
                wspk = w.get("speaker")
                if wspk in remap:
                    w["speaker"] = remap[wspk]
        merged_desc = ", ".join(f"{o}->{n}" for o, n in sorted(remap.items()))
        print(f"  Consolidated remote clusters: {merged_desc}")

    _merge_orphan_system_segments(sys_segments, config)

    # ── Strip isolated sub-threshold backchannel (B1, issue #2) ──
    # Reassign far-isolated, non-embeddable fragments out of a named cluster
    # into the generic REMOTE bucket so they don't inherit the wrong speaker.
    if config.strip_isolated_backchannel:
        try:
            outliers = _isolated_backchannel_outliers(sys_segments, config)
        except Exception as exc:  # never fail transcription over this
            print(f"  Warning: isolated-backchannel guard skipped ({exc})")
            outliers = []
        if outliers:
            for i in outliers:
                sys_segments[i]["speaker"] = "REMOTE"
                for w in sys_segments[i].get("words") or []:
                    w["speaker"] = "REMOTE"
            print(
                f"  Stripped {len(outliers)} isolated backchannel "
                f"segment(s) to REMOTE (likely mis-clustered)"
            )


def _transcribe_dual_diarize(
    audio_file: Path, config: TranscriptionConfig, duration: float
) -> Transcript:
    """Dual-channel transcribe + system-channel diarization.

    Best-of-both-worlds approach:

    * **Mic channel** → transcribed independently → all segments labeled YOU.
      The local speaker is captured continuously from their mic, immune to overlap
      with remote speakers.  No diarization needed (single local speaker).

    * **System channel** → transcribed independently → **pyannote diarization**
      splits the remote stream into distinct speakers (SPEAKER_00, _01, …)
      which downstream voiceprint matching names (Bob, Carol, etc.).

    * Both channels are **merged by timestamp** with overlaps preserved —
      when the local speaker and a remote speak simultaneously, both segments exist at
      that time, each from its own channel.

    This eliminates the overlap-fragmentation problem of mono+channel-correct
    (per-word channel sampling during overlap) while preserving multi-remote
    speaker naming that pure ``mixdown="dual"`` loses.
    """
    import numpy as np
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    mic_path = None
    sys_path = None
    asr_model = None

    try:
        mic_path = _extract_mono(audio_file, channel=0)
        sys_path = _extract_mono(audio_file, channel=1)

        whisper_lang = None if config.language == "auto" else config.language

        pad_samples = (
            int(config.audio_pad_seconds * 16000) if config.audio_pad_seconds > 0 else 0
        )

        if config.asr_backend == "whisperx":
            asr_model = _load_whisperx_asr_model(config, whisper_lang)

        # ── Mic channel (YOU): detect → bias → transcribe with that language ──
        # Detecting the channel language BEFORE the ASR and forcing it into the
        # decode prevents whisperx's first-30s auto-detect from drifting (which
        # would transcribe e.g. English audio as Spanish).  The team/operator
        # default-language bias is applied to the per-channel language so a
        # low-confidence minority detection is overridden by the default.
        print("  Transcribing mic channel (left / YOU)...")
        mic_audio = whisperx.load_audio(str(mic_path))
        if pad_samples > 0:
            mic_audio = np.concatenate(
                [mic_audio, np.zeros(pad_samples, dtype=mic_audio.dtype)]
            )
        mic_lang, mic_conf, mic_decode_lang = _resolve_channel_language(
            asr_model, mic_audio, config, whisper_lang, "mic"
        )
        mic_result = _transcribe_asr(
            mic_audio, config, mic_decode_lang, whisperx_model=asr_model
        )

        # ── System channel (remotes): detect → bias → transcribe ──
        print("  Transcribing system channel (right / remotes)...")
        sys_audio = whisperx.load_audio(str(sys_path))
        if pad_samples > 0:
            sys_audio = np.concatenate(
                [sys_audio, np.zeros(pad_samples, dtype=sys_audio.dtype)]
            )
        sys_lang, sys_conf, sys_decode_lang = _resolve_channel_language(
            asr_model, sys_audio, config, whisper_lang, "system"
        )
        sys_result = _transcribe_asr(
            sys_audio, config, sys_decode_lang, whisperx_model=asr_model
        )

        if asr_model is not None:
            del asr_model
            asr_model = None

        gc.collect()
        _empty_torch_caches(torch, config)

        # Transcript/summary language = the channel with the most speech.  The
        # mic (local speaker) may speak a minority language; the meeting's
        # dominant language usually lives on the system channel.
        detected_language = _dominant_channel_language(
            mic_result.get("segments", []),
            sys_result.get("segments", []),
            mic_lang,
            sys_lang,
        )
        # Soft default-language bias: keep the team default unless the chosen
        # language was detected with high confidence.
        if config.language == "auto":
            dominant_conf = sys_conf if detected_language == sys_lang else mic_conf
            detected_language = _apply_default_language_bias(
                detected_language, dominant_conf, config
            )
            print(f"  Summary/transcript language: {detected_language}")

        # ── Align each channel with the language its text was DECODED in ──
        # Alignment must match the transcribed text's language, which is the
        # (possibly default-language-biased) decode language, not the raw
        # detection.  When decode_lang is None (auto, non-whisperx) fall back to
        # the detected language.
        mic_result = _align_channel(
            mic_result, mic_audio, mic_decode_lang or mic_lang,
            config, torch, whisperx,
        )
        sys_result = _align_channel(
            sys_result, sys_audio, sys_decode_lang or sys_lang,
            config, torch, whisperx,
        )

        # ── Diarize system channel (split remotes) ──
        if config.hf_token:
            print("  Running speaker diarization on system channel...")
            diarize_model = DiarizationPipeline(
                token=config.hf_token,
                device=config.torch_device,
            )

            diarize_kwargs: dict[str, Any] = {}
            if config.min_speakers is not None:
                # The local speaker is on mic, not in sys_audio; remote
                # speaker count = total speakers - 1 (YOU).
                diarize_kwargs["min_speakers"] = max(1, config.min_speakers - 1)
            if config.max_speakers is not None:
                diarize_kwargs["max_speakers"] = max(1, config.max_speakers - 1)

            diarize_segments = diarize_model(sys_audio, **diarize_kwargs)
            sys_result = whisperx.assign_word_speakers(diarize_segments, sys_result)

            del diarize_model
            gc.collect()
            _empty_torch_cache(torch, config.torch_device)

            # ── Consolidate over-segmented remotes ──
            # pyannote can split one physical remote into several clusters
            # (e.g. backchannel peeled into a phantom speaker that voiceprint
            # matching then mis-names).  Merge same-speaker clusters and
            # absorb thin ones, then attach orphan one-liners to the nearest
            # remote so they don't surface as a generic REMOTE.
            if config.consolidate_remote_clusters:
                try:
                    _consolidate_dual_diarize_speakers(sys_result, sys_audio, config)
                except Exception as exc:  # never fail transcription over this
                    print(f"  Warning: remote-cluster consolidation skipped ({exc})")
        else:
            print("  Skipping remote diarization (no HF_TOKEN provided)")
            # Without diarization, all system segments become generic REMOTE.

        # ── Build segments ──
        max_t = duration if duration and duration > 0 else float("inf")
        mic_segments: list[Segment] = []
        sys_segments: list[Segment] = []
        remote_speaker_ids: set[str] = set()

        # Mic → all YOU
        for seg in mic_result["segments"]:
            seg_start = min(seg["start"], max_t)
            seg_end = min(seg["end"], max_t)
            if seg_end <= seg_start:
                continue
            mic_segments.append(
                Segment(
                    start=seg_start,
                    end=seg_end,
                    text=seg["text"],
                    speaker="YOU",
                    words=seg.get("words"),
                )
            )

        # System → diarized SPEAKER_XX (or REMOTE if no diarization)
        for seg in sys_result["segments"]:
            seg_start = min(seg["start"], max_t)
            seg_end = min(seg["end"], max_t)
            if seg_end <= seg_start:
                continue
            speaker = seg.get("speaker") or "REMOTE"
            remote_speaker_ids.add(speaker)
            sys_segments.append(
                Segment(
                    start=seg_start,
                    end=seg_end,
                    text=seg["text"],
                    speaker=speaker,
                    words=seg.get("words"),
                )
            )

        # ── Correct mic-channel crosstalk (no-headphones bleed) ──
        # Without echo cancellation the remote audio leaks into the mic, so a
        # large share of the "YOU" segments are echoes of remote speech.
        # Reassign those system-dominant / echo-duplicate mic segments to the
        # overlapping remote speaker so they aren't misattributed to the local
        # speaker.  Gated on channel_correct so it can be disabled with
        # --no-channel-correct.
        if config.channel_correct and mic_segments:
            try:
                mic_segments = _correct_mic_bleed_segments(
                    audio_file,
                    mic_segments,
                    sys_segments,
                    margin=config.channel_correct_margin,
                )
            except Exception as exc:  # never fail transcription over this
                print(f"  Warning: mic-bleed correction skipped ({exc})")

        segments: list[Segment] = mic_segments + sys_segments
        segments.sort(key=lambda s: s.start)
        # Reassignment may have introduced remote labels onto former mic
        # segments — make sure they're registered as speakers.
        remote_speaker_ids.update(
            s.speaker for s in mic_segments
            if s.speaker and s.speaker != "YOU"
        )

        speakers = [Speaker(id="YOU", label="YOU")]
        for sid in sorted(remote_speaker_ids):
            speakers.append(Speaker(id=sid))

        return Transcript(
            segments=segments,
            speakers=speakers,
            language=detected_language,
            audio_file=str(audio_file),
            duration=duration,
        )

    finally:
        if asr_model is not None:
            del asr_model
        for p in (mic_path, sys_path):
            if p is not None:
                try:
                    p.unlink()
                except OSError:
                    pass


def transcribe(
    audio_file: str | Path, config: TranscriptionConfig | None = None
) -> Transcript:
    """Run the full transcription + diarization pipeline.

    Args:
        audio_file: Path to the audio file (WAV preferred, any ffmpeg-supported format works).
        config: Transcription configuration. Uses defaults if not provided.

    Returns:
        Transcript object with diarized segments.
    """
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    if config is None:
        config = TranscriptionConfig()

    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    duration = get_audio_duration(audio_path)

    # If dual-channel, mixdown to mono for the main transcription pipeline
    # but keep the stereo file for channel-aware diarization hints
    is_stereo = _is_stereo(audio_path)

    if not is_stereo and config.mixdown in ("dual", "dual-diarize"):
        print(
            f"  Warning: --mixdown {config.mixdown} requires stereo audio, "
            "using standard mono pipeline"
        )

    # When True, this stereo file is effectively single-source (in-room):
    # the channels carry no real YOU-vs-REMOTE separation, so the mono path
    # must NOT remap diarized speakers onto YOU/REMOTE by channel energy
    # (that would collapse the genuine in-room speakers back into one).
    single_source = False

    if is_stereo and config.use_dual_channel and config.mixdown == "dual-diarize":
        # The dual-diarize path treats the mic channel as a single local
        # speaker.  For an in-room recording (multiple people share the mic,
        # system channel silent or a duplicate of the mic) that collapses
        # everyone into one speaker — fall back to the mono path, which
        # diarizes the combined signal and splits the in-room speakers.
        if config.single_source_fallback and _is_single_source_stereo(
            audio_path, config
        ):
            single_source = True
            print(
                "  Single-source stereo detected (system channel inactive or "
                "duplicate) — using mono diarization to split in-room speakers"
            )
        else:
            print("  Dual-channel detected: transcribing channels separately + diarizing remotes")
            return _transcribe_dual_diarize(audio_path, config, duration)

    if is_stereo and config.use_dual_channel and config.mixdown == "dual":
        print("  Dual-channel detected: transcribing channels separately")
        return _transcribe_dual_channel(audio_path, config, duration)

    if is_stereo and config.use_dual_channel:
        mono_path = _mixdown_to_mono(audio_path)
        print("  Dual-channel detected: mixing down to mono for transcription")
    else:
        mono_path = audio_path

    try:
        # ── Step 1: Transcribe with the selected ASR backend ──
        # "auto" means let WhisperX detect the language from the audio.
        whisper_lang = None if config.language == "auto" else config.language
        audio = whisperx.load_audio(str(mono_path))

        # Pad audio with silence at the end so the VAD properly closes the
        # final speech segment instead of cutting it off abruptly.
        if config.audio_pad_seconds > 0:
            import numpy as np

            pad_samples = int(config.audio_pad_seconds * 16000)
            audio = np.concatenate([audio, np.zeros(pad_samples, dtype=audio.dtype)])

        # Detect → default-language-bias → force the decode language, so a
        # low-confidence minority detection doesn't transcribe the whole meeting
        # in the wrong language.  Needs the model up front for multi-window
        # detection on the whisperx backend.
        mono_model = None
        if config.asr_backend == "whisperx":
            mono_model = _load_whisperx_asr_model(config, whisper_lang)
        detected_language, _mono_conf, decode_lang = _resolve_channel_language(
            mono_model, audio, config, whisper_lang, "audio"
        )
        result = _transcribe_asr(
            audio, config, decode_lang, whisperx_model=mono_model
        )
        if mono_model is not None:
            del mono_model
        # The decode language is authoritative for what the text is in; fall
        # back to the ASR-reported language only when we didn't force one.
        detected_language = decode_lang or result.get(
            "language", whisper_lang or "en"
        )
        if config.language == "auto":
            print(f"  Transcript language: {detected_language}")

        # Free transcription model memory
        gc.collect()
        _empty_torch_caches(torch, config)

        # ── Step 2: Align for word-level timestamps ──
        if config.skip_alignment:
            print("  Skipping alignment (--skip-alignment)")
        elif detected_language in ALIGNMENT_MODELS and not check_alignment_model_cached(
            detected_language
        ):
            # Model is in our registry but not downloaded — raise so
            # the caller (CLI/GUI) can show an actionable error.
            # Free VRAM first so the error handler can download if needed.
            gc.collect()
            _empty_torch_cache(torch, config.torch_device)
            raise AlignmentModelMissing(detected_language)
        else:
            print(f"  Aligning word timestamps ({detected_language})...")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=config.torch_device,
                )
                result = whisperx.align(
                    result["segments"],
                    model_a,
                    metadata,
                    audio,
                    config.torch_device,
                    return_char_alignments=False,
                )

                del model_a
                gc.collect()
                _empty_torch_cache(torch, config.torch_device)
            except Exception as align_exc:
                # For languages NOT in our registry (WhisperX supports ~39),
                # we can't pre-check the cache.  If the download fails at
                # runtime, fall back gracefully since there's no actionable
                # fix we can offer.
                if detected_language in ALIGNMENT_MODELS:
                    # This shouldn't happen (we checked cache above), but
                    # if it does, re-raise as AlignmentModelMissing.
                    raise AlignmentModelMissing(detected_language) from align_exc
                print(
                    f"  Warning: alignment failed ({align_exc}), continuing without word-level timestamps"
                )

        # ── Step 3: Speaker diarization ──
        if config.hf_token:
            print("  Running speaker diarization...")
            diarize_model = DiarizationPipeline(
                token=config.hf_token,
                device=config.torch_device,
            )

            diarize_kwargs: dict[str, Any] = {}
            if config.min_speakers is not None:
                diarize_kwargs["min_speakers"] = config.min_speakers
            if config.max_speakers is not None:
                diarize_kwargs["max_speakers"] = config.max_speakers

            diarize_segments = diarize_model(audio, **diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)

            del diarize_model
            gc.collect()
            _empty_torch_cache(torch, config.torch_device)
        else:
            print("  Skipping diarization (no HF_TOKEN provided)")

        # ── Step 4: Build Transcript object ──
        # Clamp segment timestamps to actual audio duration (we may have
        # padded silence at the end to help the VAD).
        max_t = duration if duration and duration > 0 else float("inf")

        segments = []
        for seg in result["segments"]:
            seg_start = min(seg["start"], max_t)
            seg_end = min(seg["end"], max_t)
            if seg_end <= seg_start:
                continue  # skip segments that fall entirely in the padding
            segments.append(
                Segment(
                    start=seg_start,
                    end=seg_end,
                    text=seg["text"],
                    speaker=seg.get("speaker"),
                    words=seg.get("words"),
                )
            )

        # ── Step 4b: Per-segment speaker repair ──
        # pyannote can put two people into one cluster (quick Q&A turns it
        # never cuts).  Re-check each cluster against per-segment voice
        # embeddings and move a clearly different voice out of it.
        if config.speaker_repair and config.hf_token and segments:
            from millet.speaker_repair import repair_speakers

            repair_speakers(audio, segments, device=config.torch_device)

        speaker_ids = {s.speaker for s in segments if s.speaker}
        speakers = [Speaker(id=sid) for sid in sorted(speaker_ids)]

        # ── Step 5: Dual-channel speaker labeling ──
        # Skip for single-source (in-room) recordings: the channels carry no
        # real YOU/REMOTE separation, so mapping diarized speakers onto
        # channel energy would collapse the genuine in-room speakers (all are
        # "mic-dominant") back into one.  Keep the pyannote diarization result
        # (SPEAKER_00/_01/…) and let voiceprint naming label them.
        if is_stereo and config.use_dual_channel and not single_source:
            if len(speakers) >= 2:
                # Pyannote found multiple speakers — map them to YOU/REMOTE
                # using channel energy ratios.
                print("  Labeling speakers from dual-channel audio...")
                segments, speakers = _label_speakers_from_channels(
                    audio_path,
                    segments,
                    speakers,
                )
            elif segments:
                # Pyannote found 0-1 speakers — fall back to per-segment
                # channel energy to split into YOU vs REMOTE.
                print(
                    f"  Diarization found {len(speakers)} speaker(s) in stereo"
                    f" audio — splitting by channel energy..."
                )
                segments, speakers = _split_by_channel(audio_path, segments)

            # ── Step 5b: Hybrid channel-energy correction ──
            # Diarization can mis-assign a segment whose audio clearly lives on
            # the *other* channel than its labeled speaker — most visibly when a
            # remote interjection ("thanks for that update") at a turn boundary
            # gets glued onto the local speaker's segment.  Correct such
            # segments (and words) by reassigning the YOU<->REMOTE channel side
            # when one channel *strongly* dominates, while leaving ambiguous /
            # mic-bleed segments on their diarized label.
            if config.channel_correct:
                segments, speakers = _channel_correct_segments(
                    audio_path, segments, speakers,
                    margin=config.channel_correct_margin,
                )

        return Transcript(
            segments=segments,
            speakers=speakers,
            language=detected_language,
            audio_file=str(audio_path),
            duration=duration,
        )

    finally:
        # Clean up temp files
        if is_stereo and config.use_dual_channel and mono_path != audio_path:
            try:
                mono_path.unlink()
            except OSError:
                pass


def _label_speakers_from_channels(
    stereo_file: Path,
    segments: list[Segment],
    speakers: list[Speaker],
    sample_rate: int = 16000,
) -> tuple[list[Segment], list[Speaker]]:
    """Use dual-channel stereo info to label speakers as YOU or REMOTE.

    Left channel = mic (your voice), right channel = system (remote participants).
    For each diarized speaker, compute RMS energy on each channel during their
    segments. The speaker with highest mic-channel energy ratio is labeled YOU;
    others are labeled REMOTE (or REMOTE_1, REMOTE_2, etc. if multiple).

    Args:
        stereo_file: Path to the original stereo WAV file.
        segments: Diarized segments with SPEAKER_XX labels.
        speakers: Speaker objects from diarization.
        sample_rate: Sample rate of the audio (default 16000).

    Returns:
        Updated (segments, speakers) with relabeled speaker IDs.
    """
    from millet.audio import compute_speaker_channel_energy, read_stereo_channels

    if not speakers:
        return segments, speakers

    stereo = read_stereo_channels(stereo_file)
    if stereo is None:
        print("  Channel labeling: skipping, not stereo or unreadable")
        return segments, speakers

    speaker_mic_ratio = compute_speaker_channel_energy(
        stereo.mic, stereo.system, segments, stereo.sample_rate
    )

    if not speaker_mic_ratio:
        return segments, speakers

    # Log the ratios for debugging
    print("  Channel analysis:")
    for spk, ratio in sorted(speaker_mic_ratio.items()):
        label = "mic-dominant" if ratio > 0.5 else "system-dominant"
        print(f"    {spk}: mic_ratio={ratio:.3f} ({label})")

    # The speaker with the highest mic ratio is YOU — but only if they
    # are actually mic-dominant.  When no speaker exceeds the threshold
    # (e.g. only system audio was captured), label everyone as REMOTE.
    #
    # We use both an absolute check (ratio > 0.5) and a relative margin
    # check.  Sensitive condenser mics (e.g. RODE NT-USB) pick up room
    # audio on the mic channel, which can push the local speaker's ratio
    # below 0.5 even though they are clearly the most mic-dominant.  The
    # margin check catches this case: if the top speaker's ratio is well
    # above the average of all other speakers, they are almost certainly
    # the local mic user.
    you_speaker = max(speaker_mic_ratio, key=lambda s: speaker_mic_ratio[s])
    you_ratio = speaker_mic_ratio[you_speaker]

    other_ratios = [r for s, r in speaker_mic_ratio.items() if s != you_speaker]
    avg_other = sum(other_ratios) / len(other_ratios) if other_ratios else 0.0
    margin = you_ratio - avg_other

    print(
        f"    Best candidate: {you_speaker} "
        f"(ratio={you_ratio:.3f}, margin={margin:.3f} over avg={avg_other:.3f})"
    )

    label_map: dict[str, str] = {}
    if you_ratio > 0.5 or (margin > 0.1 and you_ratio > 0.15):
        # At least one speaker is mic-dominant
        label_map[you_speaker] = "YOU"
        remote_speakers = [s for s in sorted(speaker_mic_ratio) if s != you_speaker]
        if len(remote_speakers) == 1:
            label_map[remote_speakers[0]] = "REMOTE"
        else:
            for i, spk in enumerate(remote_speakers):
                label_map[spk] = f"REMOTE_{i + 1}"
    else:
        # No speaker is mic-dominant — label all as REMOTE
        all_speakers = sorted(speaker_mic_ratio)
        if len(all_speakers) == 1:
            label_map[all_speakers[0]] = "REMOTE"
        else:
            for i, spk in enumerate(all_speakers):
                label_map[spk] = f"REMOTE_{i + 1}"

    print(f"  Speaker labels: {label_map}")

    # Relabel segments
    new_segments = []
    for seg in segments:
        new_speaker = (
            label_map.get(seg.speaker, seg.speaker) if seg.speaker else seg.speaker
        )
        new_segments.append(
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker=new_speaker,
                words=seg.words,
            )
        )

    # Relabel speakers
    new_speakers = []
    for spk in speakers:
        new_label = label_map.get(spk.id, spk.id)
        new_speakers.append(Speaker(id=new_label, label=new_label))

    return new_segments, new_speakers


def _seg_channel_ratio(mic, sys_, start_t, end_t, sr, n):
    """Return mic/(mic+sys) RMS ratio for a [start_t, end_t] window, or None.

    None means the window is empty or near-silent (no usable signal).
    """
    import numpy as np

    if start_t is None:
        # WhisperX alignment routinely emits words without timestamps
        # (numerals, non-alignable tokens). No window => no usable signal.
        return None
    start = max(0, min(int(start_t * sr), n))
    end = max(0, min(int((end_t if end_t is not None else start_t) * sr), n))
    if end <= start:
        return None
    mic_rms = float(np.sqrt(np.mean(mic[start:end] ** 2)))
    sys_rms = float(np.sqrt(np.mean(sys_[start:end] ** 2)))
    denom = mic_rms + sys_rms
    if denom < 1e-8:
        return None
    return mic_rms / denom


def _channel_correct_segments(
    stereo_file: Path,
    segments: list[Segment],
    speakers: list[Speaker],
    margin: float = 0.30,
) -> tuple[list[Segment], list[Speaker]]:
    """Reassign segments/words whose audio strongly contradicts their channel.

    Runs after diarization + YOU/REMOTE labeling on the mono path.  For each
    segment we compare its per-segment mic/(mic+sys) energy ratio against the
    channel *side* implied by its current label:

      * a YOU-side segment is flipped to REMOTE only when it is strongly
        system-dominant   (ratio < 0.5 - margin)
      * a REMOTE-side segment is flipped to YOU only when it is strongly
        mic-dominant       (ratio > 0.5 + margin)

    Ambiguous / near-balanced segments (the mic-bleed danger zone) are left
    untouched, so we don't *introduce* errors.  We only ever flip between the
    YOU side and the REMOTE side — we never invent or merge named remote
    speakers, so multi-remote diarization (Bob vs Carol vs Dave) and the
    downstream voiceprint naming are preserved.  Words inside a flipped or
    mixed segment are corrected individually, which splits a turn-boundary
    segment that contains both a local utterance and a remote interjection.

    Cost: one ffmpeg stereo decode (shared with the labeling step's reads) plus
    numpy RMS over in-memory slices — well under ~2s for a 1-hour meeting.
    """
    from millet.audio import read_stereo_channels

    if not segments:
        return segments, speakers

    # Determine the canonical mic-side ("YOU") and system-side labels present
    # after Step 5.  YOU is always the local mic speaker; any label starting
    # with REMOTE is system-side.  Correction only makes sense when there is a
    # YOU side to move leaks off of (or onto): a fully-remote recording has no
    # local channel to disambiguate against.
    labels = {s.speaker for s in segments if s.speaker}
    you_label = "YOU" if "YOU" in labels else None
    remote_labels = sorted(lbl for lbl in labels if lbl and lbl.startswith("REMOTE"))
    if you_label is None:
        # No local speaker to correct toward/away from.
        return segments, speakers
    # When a YOU-side segment is found to be system-dominant (a remote leak),
    # we move it to the generic "REMOTE" bucket — we cannot know *which*
    # specific remote it was, and inventing one would be wrong.  Prefer an
    # existing generic "REMOTE" label; else the first REMOTE_N; else "REMOTE".
    if "REMOTE" in remote_labels:
        remote_target = "REMOTE"
    elif remote_labels:
        remote_target = remote_labels[0]
    else:
        remote_target = "REMOTE"

    stereo = read_stereo_channels(stereo_file)
    if stereo is None:
        return segments, speakers

    import numpy as np

    mic = stereo.mic
    sys_ = stereo.system
    sr = stereo.sample_rate
    n = len(mic)
    lo = 0.5 - margin
    hi = 0.5 + margin

    flips = 0
    splits = 0
    out_segments: list[Segment] = []
    for seg in segments:
        if not seg.speaker:
            out_segments.append(seg)
            continue
        is_you = seg.speaker == you_label
        is_remote = seg.speaker.startswith("REMOTE")
        if not (is_you or is_remote):
            out_segments.append(seg)  # leave named-but-unlabeled speakers alone
            continue

        # ── Whole-segment flip when the segment (sans word detail) is strongly
        #    dominated by the opposite channel. ──
        # Only cross-side flips: YOU->remote (use generic bucket) or
        # remote->YOU.  A remote segment that stays system-dominant keeps its
        # specific REMOTE_N label (never collapsed).  We rely on the word-level
        # pass below to split mixed segments; the whole-segment flip handles the
        # case where a segment has no usable word timestamps.
        ratio = _seg_channel_ratio(mic, sys_, seg.start, seg.end, sr, n)
        if ratio is not None and not seg.words:
            if is_you and ratio < lo:
                seg.speaker = remote_target
                flips += 1
            elif is_remote and ratio > hi:
                seg.speaker = you_label
                flips += 1

        # ── Word-level correction + segment splitting. ──
        # When a segment's words split cleanly across the two channels (the
        # turn-boundary leak case), emit separate sub-segments per speaker so
        # the attribution is visible in the text/SRT output, not just the word
        # metadata.  Words in the mic-bleed grey zone inherit the (possibly
        # flipped) segment speaker.
        #
        # CRITICAL: we only override a word's speaker when its channel side
        # CONTRADICTS the diarized side.  A word that stays on its own side
        # keeps the diarized speaker (e.g. REMOTE_2 stays REMOTE_2) — we never
        # force same-side words into a single bucket, so multi-remote
        # diarization (REMOTE_1..N) and voiceprint naming are preserved.  Only
        # cross-side leaks are reassigned, and the only safe cross-side target
        # for a YOU->remote flip is the generic remote bucket.
        if not seg.words:
            out_segments.append(seg)
            continue

        seg_is_remote = seg.speaker.startswith("REMOTE")
        per_word_speaker: list[str] = []
        for word in seg.words:
            wr = _seg_channel_ratio(
                mic, sys_, word.get("start"), word.get("end"), sr, n
            )
            if wr is None:
                spk = seg.speaker
            elif wr > hi and seg_is_remote:
                # Remote-labeled word is strongly mic-dominant -> local leak.
                spk = you_label
            elif wr < lo and not seg_is_remote:
                # Local (YOU) word is strongly system-dominant -> remote leak.
                spk = remote_target
            else:
                # Same side as diarized (or ambiguous): KEEP the diarized
                # speaker, preserving the specific REMOTE_N identity.
                spk = seg.speaker
            word["speaker"] = spk
            per_word_speaker.append(spk)

        # Split the segment into maximal runs of identical word-speaker.
        runs = _split_segment_by_word_speaker(seg, per_word_speaker)
        if len(runs) > 1:
            splits += 1
        out_segments.extend(runs)

    if flips or splits:
        print(
            f"  Channel correction: flipped {flips} segment(s), "
            f"split {splits} mixed segment(s)"
        )

    # Rebuild speaker list from what's actually present now.
    present = {s.speaker for s in out_segments if s.speaker}
    new_speakers = [sp for sp in speakers if sp.id in present]
    for lbl in sorted(present):
        if lbl not in {sp.id for sp in new_speakers}:
            new_speakers.append(Speaker(id=lbl, label=lbl))
    _ = np  # numpy used via _seg_channel_ratio
    return out_segments, new_speakers


def _norm_dup_text(text: str | None) -> str:
    """Normalise text for the echo-duplicate comparison in bleed removal."""
    import re

    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).strip()


def _correct_mic_bleed_segments(
    stereo_file: Path,
    mic_segments: list[Segment],
    sys_segments: list[Segment],
    remote_fallback: str = "REMOTE",
    margin: float = 0.30,
    word_sys_frac: float = 0.60,
    dup_sim: float = 0.70,
) -> list[Segment]:
    """Reassign mic-channel ("YOU") segments that are actually remote bleed.

    The ``dual-diarize`` path labels the entire mic channel as the local
    speaker.  When the meeting was recorded WITHOUT headphones, the remote
    audio played through the room speakers leaks into the mic, so a large
    fraction of those "YOU" segments are echoes of remote speech.

    A mic segment is treated as remote bleed when ANY of:

    * its whole-segment mic/(mic+sys) energy ratio is strongly system-dominant
      (``< 0.5 - margin``) — unambiguous bleed;
    * at least ``word_sys_frac`` of its words are individually system-dominant;
    * its text closely matches (``>= dup_sim``) a temporally overlapping remote
      segment — a delayed echo of what a remote actually said.

    Bled segments are **reassigned to the diarized remote speaker that overlaps
    them in time** (or ``remote_fallback`` when none does), preserving the
    content with correct attribution instead of attributing it to YOU.  Empty /
    punctuation-only mic artifacts are dropped.  Genuine local speech
    (mic-dominant words, no remote twin) is left as YOU, so the correction is
    conservative and never invents or merges named remote speakers.

    Returns the corrected list of segments (YOU + reassigned), excluding empties.
    """
    import difflib

    from millet.audio import read_stereo_channels

    if not mic_segments:
        return mic_segments

    stereo = read_stereo_channels(stereo_file)
    if stereo is None:
        return mic_segments

    mic = stereo.mic
    sys_ = stereo.system
    sr = stereo.sample_rate
    n = len(mic)
    lo = 0.5 - margin

    # Pre-normalise remote text + keep timing for the echo check.
    remote_norm = [
        (s.start, s.end, _norm_dup_text(s.text)) for s in sys_segments
    ]

    def _is_echo(seg: Segment) -> bool:
        t0 = _norm_dup_text(seg.text)
        if len(t0) < 4:
            return False
        for r_start, r_end, r_text in remote_norm:
            if r_end < seg.start - 8.0 or r_start > seg.end + 8.0:
                continue
            if difflib.SequenceMatcher(None, t0, r_text).ratio() >= dup_sim:
                return True
        return False

    def _word_sys_fraction(seg: Segment) -> float | None:
        flags: list[bool] = []
        for w in seg.words or []:
            r = _seg_channel_ratio(mic, sys_, w.get("start"), w.get("end"), sr, n)
            if r is not None:
                flags.append(r < 0.5)
        if not flags:
            return None
        return sum(flags) / len(flags)

    def _overlapping_remote(seg: Segment) -> str | None:
        best = None
        best_ov = 0.3  # require > 0.3s overlap
        for s in sys_segments:
            ov = min(seg.end, s.end) - max(seg.start, s.start)
            if ov > best_ov and s.speaker:
                best_ov = ov
                best = s.speaker
        return best

    out: list[Segment] = []
    reassigned = 0
    dropped = 0
    for seg in mic_segments:
        if not _norm_dup_text(seg.text):
            dropped += 1
            continue
        seg_ratio = _seg_channel_ratio(mic, sys_, seg.start, seg.end, sr, n)
        wsf = _word_sys_fraction(seg)
        is_bleed = (
            (seg_ratio is not None and seg_ratio < lo)
            or (wsf is not None and wsf >= word_sys_frac)
            or _is_echo(seg)
        )
        if is_bleed:
            seg.speaker = _overlapping_remote(seg) or remote_fallback
            reassigned += 1
        out.append(seg)

    if reassigned or dropped:
        print(
            f"  Mic-bleed correction: reassigned {reassigned} of "
            f"{len(mic_segments)} local segment(s) to remote crosstalk, "
            f"dropped {dropped} empty"
        )
    return out


def _split_segment_by_word_speaker(
    seg: Segment, per_word_speaker: list[str]
) -> list[Segment]:
    """Split *seg* into consecutive runs of identical per-word speaker.

    Returns one Segment per maximal same-speaker run, preserving word lists and
    deriving each sub-segment's start/end from its words.  If all words share a
    speaker (or there are no usable word timestamps), returns ``[seg]`` with the
    single speaker applied.
    """
    words = seg.words or []
    if not words or len(per_word_speaker) != len(words):
        return [seg]

    # Identify contiguous runs.
    runs: list[tuple[str, list[dict]]] = []
    cur_spk = per_word_speaker[0]
    cur_words: list[dict] = []
    for w, spk in zip(words, per_word_speaker, strict=False):
        if spk != cur_spk:
            runs.append((cur_spk, cur_words))
            cur_spk = spk
            cur_words = []
        cur_words.append(w)
    runs.append((cur_spk, cur_words))

    if len(runs) == 1:
        seg.speaker = runs[0][0]
        return [seg]

    out: list[Segment] = []
    for spk, rwords in runs:
        text = " ".join((w.get("word") or "").strip() for w in rwords).strip()
        if not text:
            continue
        starts = [w["start"] for w in rwords if w.get("start") is not None]
        ends = [w["end"] for w in rwords if w.get("end") is not None]
        s_start = min(starts) if starts else seg.start
        s_end = max(ends) if ends else seg.end
        out.append(
            Segment(
                start=s_start,
                end=s_end,
                text=text,
                speaker=spk,
                words=rwords,
            )
        )
    return out or [seg]


def _split_by_channel(
    stereo_file: Path,
    segments: list[Segment],
) -> tuple[list[Segment], list[Speaker]]:
    """Split a single-speaker diarization into YOU/REMOTE using channel energy.

    This is a fallback for when pyannote diarization detects only one speaker
    in a stereo recording (e.g. due to short duration or GPU-dependent
    floating-point differences in speaker embeddings).

    For each segment, the mic (left) and system (right) channel RMS energy
    is compared.  Segments where the mic channel dominates are labeled YOU;
    segments where the system channel dominates are labeled REMOTE.

    Args:
        stereo_file: Path to the original stereo audio file (WAV or OGG).
        segments:    List of Segment objects (all assigned to a single speaker).

    Returns:
        Updated (segments, speakers) with YOU/REMOTE labels.
        If the stereo file can't be read or all segments land on one speaker,
        the original segments are returned unchanged with a single speaker.
    """
    from millet.audio import read_stereo_channels

    stereo = read_stereo_channels(stereo_file)
    if stereo is None:
        print("  Channel split: skipping, stereo data unreadable")
        speaker_ids = {s.speaker for s in segments if s.speaker}
        return segments, [Speaker(id=sid) for sid in sorted(speaker_ids)]

    sr = stereo.sample_rate
    n = len(stereo.mic)

    import numpy as np

    for seg in segments:
        start = max(0, min(int(seg.start * sr), n))
        end = max(0, min(int(seg.end * sr), n))
        if end <= start:
            continue

        mic_rms = float(np.sqrt(np.mean(stereo.mic[start:end] ** 2)))
        sys_rms = float(np.sqrt(np.mean(stereo.system[start:end] ** 2)))

        # Assign based on dominant channel
        if mic_rms + sys_rms < 1e-8:
            # Near-silence — keep existing label
            continue
        seg.speaker = "YOU" if mic_rms >= sys_rms else "REMOTE"

        # Update word-level labels too
        if seg.words:
            for word in seg.words:
                w_start_t = word.get("start")
                w_end_t = word.get("end")
                if w_start_t is None:
                    word["speaker"] = seg.speaker
                    continue

                ws = max(0, min(int(w_start_t * sr), n))
                we = max(0, min(int((w_end_t or w_start_t) * sr), n))
                if we <= ws:
                    word["speaker"] = seg.speaker
                    continue

                w_mic = float(np.sqrt(np.mean(stereo.mic[ws:we] ** 2)))
                w_sys = float(np.sqrt(np.mean(stereo.system[ws:we] ** 2)))
                if w_mic + w_sys < 1e-8:
                    word["speaker"] = seg.speaker
                else:
                    word["speaker"] = "YOU" if w_mic >= w_sys else "REMOTE"

    # Build speaker list from what was actually assigned
    speaker_ids = {s.speaker for s in segments if s.speaker}
    speakers = [Speaker(id=sid) for sid in sorted(speaker_ids)]

    # Log the split results
    from collections import Counter

    counts = Counter(s.speaker for s in segments if s.speaker)
    print(f"  Channel split result: {dict(counts)}")

    return segments, speakers


def _is_stereo(audio_file: Path) -> bool:
    """Check if an audio file has 2 channels."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "stream=channels",
        "-of",
        "csv=p=0",
        str(audio_file),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    try:
        return int(result.stdout.strip()) == 2
    except ValueError:
        return False


def post_process(
    transcript: Transcript,
    output_dir: Path,
    basename: str,
    *,
    summarize: bool = True,
    summary_backend: str | None = None,
    summary_model: str | None = None,
    summary_preset: str | None = None,
    ollama_singlepass: bool = False,
    progress_callback=None,
) -> dict:
    """Run summarization and PDF generation after transcription.

    This is the shared post-processing step used by both CLI and GUI.
    Summary generation is best-effort — failures are caught and logged via
    ``progress_callback`` but do not abort PDF generation.

    Args:
        transcript:       The completed Transcript object.
        output_dir:       Directory to write output files into.
        basename:         Stem for output filenames (e.g. "meeting-20260313-231509").
        summarize:        Whether to attempt AI summarization.
        summary_backend:  Backend override ("tinfoil" or "ollama"); None uses default.
        summary_model:    Model name override; None uses the per-backend default.
        progress_callback: Optional callable(str) for status/error messages.

    Returns:
        Dict with keys "summary" (Path or None) and "pdf" (Path or None).
    """

    def _log(msg: str) -> None:
        print(f"  {msg}")  # Always visible in terminal
        if progress_callback:
            progress_callback(msg)

    result: dict = {"summary": None, "pdf": None}
    summary_result = None
    # When a preset was explicitly selected, summary failure is fatal: the
    # caller (e.g. `meet transcribe` CLI, vezir worker) must learn about it
    # via a non-zero exit so it can mark the job as errored.  We still
    # generate the PDF (so the transcript artifact exists) before raising.
    preset_summary_error: Exception | None = None

    if summarize:
        try:
            from millet.summarize import SummaryConfig
            from millet.summarize import summarize as do_summarize

            cfg_kwargs: dict = {}
            if summary_preset:
                cfg_kwargs["preset"] = summary_preset
            if summary_backend:
                cfg_kwargs["backend"] = summary_backend
            if summary_model:
                cfg_kwargs["model"] = summary_model
            if ollama_singlepass:
                cfg_kwargs["ollama_singlepass"] = True
            summary_config = SummaryConfig(**cfg_kwargs)

            summary_result = do_summarize(
                transcript.to_text(),
                summary_config,
                language=transcript.language,
                progress_callback=progress_callback,
            )
            from millet.frontmatter import context_from_transcript

            fm_ctx = context_from_transcript(transcript, output_dir)
            path = summary_result.save(
                output_dir, basename, frontmatter_context=fm_ctx,
            )
            result["summary"] = path
            _log(f"Summary generated in {summary_result.elapsed_seconds:.1f}s")
        except Exception as exc:
            _log(f"Summary failed: {exc}")
            if summary_preset:
                # Preset was explicit — user chose a specific privacy/quality
                # level.  Remember the failure and raise after PDF is written.
                preset_summary_error = exc

    try:
        from millet.pdf import generate_pdf

        pdf_path = output_dir / f"{basename}.pdf"
        generate_pdf(
            transcript,
            pdf_path,
            summary=summary_result,
            language=getattr(transcript, "language", "en"),
        )
        result["pdf"] = pdf_path
    except Exception as exc:
        _log(f"PDF generation failed: {exc}")

    # ── Compress WAV to OGG/Opus ──
    wav_path = output_dir / f"{basename}.wav"
    if wav_path.exists():
        try:
            from millet.audio import compress_audio

            _log("Compressing audio to OGG/Opus...")
            ogg_path = compress_audio(wav_path)
            result["audio"] = ogg_path
            _log(f"Audio compressed: {ogg_path.name}")
        except Exception as exc:
            _log(f"Audio compression failed (WAV kept): {exc}")

    if preset_summary_error is not None:
        # Surface the preset-summary failure as a non-zero exit.  The
        # transcript, PDF, and audio artifacts are already on disk for
        # forensic / retry purposes.
        raise RuntimeError(
            f"summary failed for preset '{summary_preset}': "
            f"{preset_summary_error}"
        ) from preset_summary_error

    return result
