"""Speaker voiceprint database for cross-session speaker recognition.

Extracts speaker embeddings from labeled sessions using pyannote's bundled
WeSpeakerResNet34 model, stores averaged profiles per person, and identifies
speakers in new meetings by cosine similarity.

Default profile DB: ``~/.config/meet/speaker_profiles.json``.

Override with:
  - Pass ``profiles_path=`` to any public function (recommended for
    embedders like vezir that manage their own central DB).
  - Set ``MEET_PROFILES_PATH=/path/to/profiles.json`` in the environment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

from millet.crosstalk import is_reserved_label

log = logging.getLogger(__name__)


# ─── Profile path resolution ─────────────────────────────────────────────────

_DEFAULT_PROFILES_REL = Path(".config") / "meet" / "speaker_profiles.json"


def _default_profiles_path(team: str | None = None) -> Path:
    """Resolve the default profile DB path **at call time**.

    Delegates to :mod:`millet.paths`, which centralizes resolution:
      1. ``team`` given → ``~/.config/meet/<team>/speaker_profiles.json``.
      2. ``$MILLET_PROFILES_PATH`` (or legacy ``$MEET_PROFILES_PATH``).
      3. ``~/.config/meet/speaker_profiles.json``.

    ``Path.home()`` / env vars are read now (not at import), so callers
    that override them between import and call get the right path.
    """
    from . import paths
    return paths.profiles_path(team)


def __getattr__(name: str):
    """PEP 562 module-level attribute hook for back-compat.

    ``from meet.voiceprint import PROFILES_PATH`` still works but now
    resolves lazily at access time, not at import time.
    """
    if name == "PROFILES_PATH":
        return _default_profiles_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ─── Constants ────────────────────────────────────────────────────────────────
MATCH_THRESHOLD = 0.65  # cosine similarity — below this, don't auto-label
MIN_SEGMENT_DURATION = 1.5   # seconds — skip very short segments for embedding
MAX_SEGMENTS_PER_SPEAKER = 10  # how many segments to average per speaker
# Minimum embeddable speech (sum of >= MIN_SEGMENT_DURATION segments) a
# speaker cluster must have before a voiceprint match is trustworthy enough
# to AUTO-APPLY.  Thin clusters built from a few short backchannel utterances
# yield noisy embeddings that match unrelated profiles by chance (e.g. a
# 0.69 false positive); below this floor we keep the speaker raw rather than
# confidently mislabeling it.  The match is still computed/returned so
# diagnostics and interactive labeling can surface it.
MATCH_MIN_SPEECH_SECONDS = 4.0
# A match at or above MATCH_THRESHOLD is AUTO-APPLIED only if it is also
# *unambiguous*: either its absolute confidence clears a higher auto-apply
# floor, OR it beats the cluster's second-best profile by a clear margin.
# This rejects the false-positive signature observed on over-segmented
# backchannel clusters — a barely-over-threshold score (~0.69) with a tiny
# margin (~0.13) over the runner-up — while still auto-applying strong,
# well-separated matches (e.g. 0.93 with a 0.40 margin).
MATCH_AUTOAPPLY_CONFIDENCE = 0.72
MATCH_AUTOAPPLY_MARGIN = 0.15
# Pass-2 many-to-one folding (see identify_speakers) is HIGHER RISK than a
# pass-1 1:1 match: it attaches a still-unmatched cluster to a profile that is
# ALREADY claimed by another cluster, on the assumption that diarization
# over-segmented ONE person.  That assumption fails when two DIFFERENT people
# (e.g. two new founders on the same compressed system channel, speaking in
# non-overlapping turns) each resemble the same enrolled profile just enough to
# clear the ordinary auto-apply floor — the fold then silently merges two
# people into one name.  Folding therefore requires a STRICTER bar than a fresh
# 1:1 match: a higher absolute confidence AND a clear margin over the cluster's
# own runner-up profile (a small margin is the ambiguous-cluster signature).
# A genuine over-segmentation of one person clears both easily (the cluster
# looks overwhelmingly like that one profile); two similar-but-distinct voices
# do not.
MATCH_MANY_TO_ONE_CONFIDENCE = 0.80
# Empirically tuned floor for embedding extraction.
#
# The purpose of this gate is to reject TRUE silence — clips where the
# audio is essentially digital quantization noise (~1e-4 RMS) and the
# embedding model produces meaningless 256-dim vectors that would poison
# the central profile DB or match against unrelated profiles by chance.
#
# It must NOT reject quiet-but-real speech.  Validated case: a Linux
# thin-client recording with mic mean ~-49.7 dBFS (RMS ~0.003) and peaks
# at -4 dBFS — all 7 eligible segments had RMS in [0.0016, 0.0024].  The
# user's voice is matchable against the central 58-profile DB; the
# embedding extractor needs to see those segments.
#
# 0.0015 normalized (~ -56 dBFS) sits well above the silence floor
# (~ -90 dBFS) and below the lowest validated mic-recording RMS.  Earlier
# 0.005 (~ -46 dBFS) was too aggressive — it skipped every segment of
# quiet but real speech, leaving the auto-labeller blind to known
# speakers.
MIN_SEGMENT_RMS = 0.0015

# ─── What a profile may LEARN from (enroll / update-from-labels) ─────────────
# Matching is forgiving; learning must not be.  A profile is a running average
# with no memory of what went in, so one bad update can't be taken back.
#
# The trap: picking a cluster's LONGEST segments.  For a filler cluster those
# are the fillers Whisper stretched over silence ("Yeah." spanning 6.7 s) —
# audio that is mostly room tone and, on the system channel, the scribe's own
# voice echoing back through the call.  One real profile was built
# that way (cosine 0.86 to the scribe's echo, 0.10 to the person it was named
# after) and then matched filler buckets across every meeting at ~0.9.
#
# So only DENSE speech is learned from: segments carrying real words at a
# speaking rate, most-words-first, and only when enough of it exists.
LEARN_MIN_WORDS = 6                  # a real utterance, not "Okay."
LEARN_MIN_WORDS_PER_SECOND = 1.5     # speech, not a filler stretched over silence
LEARN_MIN_SPEECH_SECONDS = MATCH_MIN_SPEECH_SECONDS  # same floor as auto-apply


def is_dense_utterance(seg) -> bool:
    """True if ``seg`` is real speech: a sentence spoken at a speaking rate.

    >= :data:`MIN_SEGMENT_DURATION`, >= :data:`LEARN_MIN_WORDS` words and
    >= :data:`LEARN_MIN_WORDS_PER_SECOND`.  Fillers ("Okay.", "Yeah.") and
    fillers Whisper stretched over silence fail it.
    """
    dur = seg.end - seg.start
    words = len((getattr(seg, "text", "") or "").split())
    return (
        dur >= MIN_SEGMENT_DURATION
        and words >= LEARN_MIN_WORDS
        and words / dur >= LEARN_MIN_WORDS_PER_SECOND
    )


def learnable_segments(transcript_segments: list, speaker_id: str) -> list[tuple[float, float]]:
    """``(start, end)`` spans of ``speaker_id`` fit to learn a voiceprint from.

    Dense segments only (:func:`is_dense_utterance`), most words first, at
    most :data:`MAX_SEGMENTS_PER_SPEAKER`.  Returns ``[]`` when their total is
    under :data:`LEARN_MIN_SPEECH_SECONDS` — a filler / echo / silence cluster
    must never create or update a profile.
    """
    dense: list[tuple[int, float, float]] = []
    for seg in transcript_segments:
        if seg.speaker != speaker_id or not is_dense_utterance(seg):
            continue
        words = len((getattr(seg, "text", "") or "").split())
        dense.append((words, seg.start, seg.end))
    dense.sort(key=lambda d: d[0], reverse=True)
    selected = [(s, e) for _, s, e in dense[:MAX_SEGMENTS_PER_SPEAKER]]
    if sum(e - s for s, e in selected) < LEARN_MIN_SPEECH_SECONDS:
        return []
    return selected


# ─── Embedding model loading ──────────────────────────────────────────────────

_inference = None  # lazy singleton


def _get_inference():
    """Load and return the pyannote WeSpeaker embedding Inference object.

    Uses the embedding model bundled inside the cached
    pyannote/speaker-diarization-community-1 model, which is always present
    if diarization has been run at least once.
    """
    global _inference
    if _inference is not None:
        return _inference

    from pyannote.audio import Inference, Model

    hub = Path.home() / ".cache" / "huggingface" / "hub"
    candidates = sorted(hub.glob("models--pyannote--speaker-diarization*"))
    if not candidates:
        raise RuntimeError(
            "pyannote diarization model not found in HuggingFace cache. "
            "Run meet on a recording first to download it."
        )

    # Find the embedding model inside the first matching snapshot
    emb_path = None
    for model_dir in candidates:
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue
        for snap in sorted(snapshots_dir.iterdir()):
            candidate = snap / "embedding" / "pytorch_model.bin"
            if candidate.exists():
                emb_path = candidate
                break
        if emb_path:
            break

    if emb_path is None:
        raise RuntimeError(
            "Could not find embedding/pytorch_model.bin inside pyannote model cache."
        )

    log.debug("Loading embedding model from %s", emb_path)
    model = Model.from_pretrained(emb_path)
    _inference = Inference(model, window="whole")
    return _inference


# ─── Profile storage ──────────────────────────────────────────────────────────

class SpeakerProfile(NamedTuple):
    name: str
    embedding: np.ndarray  # 256-dim, L2-normalized
    n_sessions: int


def load_profiles(
    profiles_path: Path | None = None,
) -> dict[str, SpeakerProfile]:
    """Load speaker profiles from disk. Returns empty dict if not found.

    Args:
        profiles_path: Explicit path to the profile DB.  When ``None``,
            falls back to :func:`_default_profiles_path`.
    """
    p = profiles_path or _default_profiles_path()
    if not p.exists():
        return {}

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not load speaker profiles: %s", exc)
        return {}

    profiles: dict[str, SpeakerProfile] = {}
    for name, info in data.items():
        emb = np.array(info["embedding"], dtype=np.float32)
        profiles[name] = SpeakerProfile(
            name=name,
            embedding=emb,
            n_sessions=info.get("n_sessions", 1),
        )
    return profiles


def save_profiles(
    profiles: dict[str, SpeakerProfile],
    profiles_path: Path | None = None,
) -> None:
    """Save speaker profiles to disk.

    Args:
        profiles_path: Explicit path.  When ``None``, falls back to
            :func:`_default_profiles_path`.
    """
    import os
    import tempfile

    p = profiles_path or _default_profiles_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        name: {
            "embedding": prof.embedding.tolist(),
            "n_sessions": prof.n_sessions,
        }
        for name, prof in profiles.items()
    }
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    # Atomic replace (audit L-8).  The profile DB is a running average with
    # no provenance, and vezir SIGKILLs the whole millet process group on a
    # watchdog timeout — a plain write_text() interrupted mid-way left a
    # truncated, unrecoverable team voiceprint DB.  Write to a sibling temp
    # file (0600, same filesystem), fsync, then rename over the target.
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if p.exists():
            try:
                os.chmod(tmp, p.stat().st_mode & 0o777)
            except OSError:
                pass
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _merge_embedding(
    existing: SpeakerProfile,
    new_emb: np.ndarray,
) -> SpeakerProfile:
    """Weighted running average of speaker embeddings."""
    n = existing.n_sessions
    merged = (existing.embedding * n + new_emb) / (n + 1)
    merged = _l2_norm(merged)
    return SpeakerProfile(
        name=existing.name,
        embedding=merged,
        n_sessions=n + 1,
    )


def _l2_norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


# ─── Embedding extraction ─────────────────────────────────────────────────────

# Cache the decoded channels for the most recently accessed file only.  The
# identify/enroll/update loops call _extract_channel_audio once per speaker but
# there are only two distinct channels; without this each of N speakers
# triggered a full-file ffmpeg decode (hundreds of MB for an hour of audio).
# Scoped to one file so it never accumulates across sessions.
_CHANNEL_CACHE_KEY: tuple[str, float] | None = None
_CHANNEL_CACHE: dict[str, tuple[np.ndarray, int] | None] = {}


def _extract_channel_audio(audio_path: Path, channel: str) -> tuple[np.ndarray, int] | None:
    """Extract a single audio channel as float32 array.

    Decodes at most once per (file, channel); repeated calls for the same file
    and channel return the cached array.

    Args:
        audio_path: Path to stereo audio file.
        channel: 'mic' (left) or 'system' (right).

    Returns:
        (samples_float32, sample_rate) or None on failure.
    """
    global _CHANNEL_CACHE_KEY, _CHANNEL_CACHE

    try:
        mtime = Path(audio_path).stat().st_mtime
        key = (str(audio_path), mtime)
    except OSError:
        key = None

    if key is not None:
        if _CHANNEL_CACHE_KEY != key:
            # New file (or changed on disk): drop the previous file's channels.
            _CHANNEL_CACHE_KEY = key
            _CHANNEL_CACHE = {}
        elif channel in _CHANNEL_CACHE:
            return _CHANNEL_CACHE[channel]

    result = _decode_channel_audio(audio_path, channel)
    if key is not None:
        _CHANNEL_CACHE[channel] = result
    return result


def _decode_channel_audio(audio_path: Path, channel: str) -> tuple[np.ndarray, int] | None:
    """Decode a single audio channel via ffmpeg (uncached)."""
    import subprocess

    ch_idx = 0 if channel == "mic" else 1
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-i", str(audio_path),
        "-filter_complex", f"[0:a]pan=mono|c0=c{ch_idx}[out]",
        "-map", "[out]",
        "-ar", "16000",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 or not result.stdout:
            return None
        samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
        samples /= 32768.0  # normalize to [-1, 1] for pyannote
        return samples, 16000
    except Exception as exc:
        log.debug("Channel audio extraction failed: %s", exc)
        return None


def _embed_segments(
    samples: np.ndarray,
    sample_rate: int,
    segments: list,  # list of (start, end) float tuples
    inference,
) -> np.ndarray | None:
    """Extract and average embeddings from a list of time segments.

    Args:
        samples: float32 mono audio array (normalized to [-1, 1]).
        sample_rate: typically 16000.
        segments: list of (start_sec, end_sec) tuples.
        inference: pyannote Inference object (window='whole').

    Returns:
        Averaged, L2-normalized 256-dim embedding, or None if extraction fails.
    """
    import torch

    embeddings = []
    n_too_short = 0
    n_too_quiet = 0
    n_considered = 0
    for start, end in segments:
        n_considered += 1
        start_frame = int(start * sample_rate)
        end_frame = min(int(end * sample_rate), len(samples))
        clip = samples[start_frame:end_frame]

        if len(clip) < int(sample_rate * MIN_SEGMENT_DURATION):
            n_too_short += 1
            continue  # too short

        # Skip near-silent clips — the embedding model produces garbage
        # vectors from silence, which poison matching and profile updates.
        # Log at INFO so a user-visible diagnostic appears in the meet
        # job log when audio gain is too low.  See MIN_SEGMENT_RMS docstring.
        rms = float(np.sqrt(np.mean(clip ** 2)))
        if rms < MIN_SEGMENT_RMS:
            n_too_quiet += 1
            log.info(
                "Skipping near-silent segment %.1f-%.1f (rms=%.6f, floor=%.6f)",
                start, end, rms, MIN_SEGMENT_RMS,
            )
            continue

        try:
            # Pass the clip directly as a torch tensor — pyannote Inference
            # accepts {'waveform': Tensor(C, T), 'sample_rate': int}
            waveform = torch.from_numpy(clip).unsqueeze(0)  # (1, T)
            audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
            emb = inference(audio_dict)
            # emb is an np.ndarray of shape (dim,) when window='whole'
            if emb is not None:
                vec = np.array(emb).flatten().astype(np.float32)
                if len(vec) > 0:
                    embeddings.append(vec)
        except Exception as exc:
            log.debug("Embedding extraction failed for segment %.1f-%.1f: %s", start, end, exc)
            continue

    # Per-speaker summary — surfaces silent-gate skips at WARNING when
    # most segments were rejected, so auto-label failure has a visible
    # cause rather than a silent "no confident matches".
    if n_considered > 0 and n_too_quiet > 0:
        ratio = n_too_quiet / n_considered
        if ratio >= 0.8:
            log.warning(
                "Embedding: skipped %d/%d near-silent segments (floor=%.4f). "
                "Increase mic gain at record time to improve auto-label match.",
                n_too_quiet, n_considered, MIN_SEGMENT_RMS,
            )
        else:
            log.info(
                "Embedding: skipped %d/%d near-silent segments (floor=%.4f).",
                n_too_quiet, n_considered, MIN_SEGMENT_RMS,
            )

    if not embeddings:
        return None

    avg = np.mean(embeddings, axis=0).astype(np.float32)
    return _l2_norm(avg)


def extract_speaker_embeddings(
    audio_path: Path,
    transcript_segments: list,       # list of Segment objects
    speaker_labels: dict[str, str],  # {REMOTE_N: "Name"} or {speaker_id: name}
    channel_map: dict[str, str],     # {speaker_id: 'mic' | 'system'}
) -> dict[str, np.ndarray]:
    """Extract embeddings for all labeled speakers in a session.

    Args:
        audio_path: Path to the session audio file (OGG or WAV).
        transcript_segments: Segment objects with .speaker, .start, .end.
        speaker_labels: Map from speaker_id to human name.
        channel_map: Map from speaker_id to dominant channel ('mic' or 'system').

    Returns:
        Dict mapping human name to 256-dim embedding array.
        Speakers with insufficient audio are omitted.
    """
    inference = _get_inference()

    # For each speaker, only dense real speech is learned from (see
    # learnable_segments) — the same gate as update_profiles_from_confirmed_labels.
    result: dict[str, np.ndarray] = {}
    for speaker_id, name in speaker_labels.items():
        if is_reserved_label(name):
            continue
        selected = learnable_segments(transcript_segments, speaker_id)
        if not selected:
            log.debug("No learnable speech for speaker %s (%s)", speaker_id, name)
            continue

        channel = channel_map.get(speaker_id, "system")
        channel_data = _extract_channel_audio(audio_path, channel)
        if channel_data is None:
            log.warning("Could not extract %s channel for %s", channel, name)
            continue

        samples, sr = channel_data
        emb = _embed_segments(samples, sr, selected, inference)
        if emb is not None:
            result[name] = emb
            log.debug(
                "Extracted embedding for %s (%s) from %d segments",
                name, speaker_id, len(selected),
            )
        else:
            log.warning("Could not extract embedding for %s (%s)", name, speaker_id)

    return result


# ─── Session enrollment ───────────────────────────────────────────────────────

def enroll_session(
    session_dir: Path,
    progress_callback=None,
    *,
    profiles_path: Path | None = None,
) -> dict[str, bool]:
    """Enroll all labeled speakers from a session into the profile database.

    Args:
        session_dir: Path to a labeled session directory (must have
                     session.json with speaker_labels and a transcript JSON).
        progress_callback: Optional callable(str) for status messages.
        profiles_path: Explicit path to the profile DB.  When ``None``,
            falls back to :func:`_default_profiles_path`.

    Returns:
        Dict mapping speaker name to True (enrolled) or False (failed).
    """
    from millet.label import _find_session_files, _load_transcript

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)
        else:
            log.info(msg)

    session_dir = Path(session_dir)
    files = _find_session_files(session_dir)

    # Load speaker labels from session.json
    session_json = files.get("session")
    if not session_json or not session_json.exists():
        raise FileNotFoundError(f"No session.json found in {session_dir}")

    meta = json.loads(session_json.read_text(encoding="utf-8"))
    speaker_labels = meta.get("speaker_labels", {})
    if not speaker_labels:
        raise ValueError(f"No speaker_labels in {session_json} — label this session first")

    # Load transcript
    transcript_json = files.get("json")
    if not transcript_json or not transcript_json.exists():
        raise FileNotFoundError(f"No transcript JSON found in {session_dir}")

    transcript = _load_transcript(transcript_json)

    # Build a reverse map: relabeled_name -> original speaker_id
    # The transcript has already been relabeled, so segments use real names.
    # speaker_labels maps REMOTE_N -> Name, so we need to find segments by name.
    # Build a channel map: name -> channel based on whether it was YOU or REMOTE
    channel_map: dict[str, str] = {}
    for speaker_id, name in speaker_labels.items():
        if speaker_id == "YOU":
            channel_map[name] = "mic"
        else:
            channel_map[name] = "system"

    # Transcript segments already have real names as speaker IDs (post-relabel)
    # Build a pseudo speaker_labels where key == value (name -> name)
    # Reserved non-person labels (CROSSTALK) are never enrolled.
    name_to_name = {
        name: name for name in speaker_labels.values()
        if not is_reserved_label(name)
    }

    # Find audio file
    audio_path = files.get("wav")
    if not audio_path or not audio_path.exists():
        raise FileNotFoundError(f"No audio file found in {session_dir}")

    _log(f"  Extracting embeddings from {audio_path.name}...")

    embeddings = extract_speaker_embeddings(
        audio_path,
        transcript.segments,
        name_to_name,
        channel_map,
    )

    if not embeddings:
        _log("  No embeddings extracted — check audio file and transcript.")
        return {}

    # Merge into profiles
    profiles = load_profiles(profiles_path=profiles_path)
    status: dict[str, bool] = {}

    for name, emb in embeddings.items():
        if name in profiles:
            profiles[name] = _merge_embedding(profiles[name], emb)
            _log(f"  Updated profile: {name} (now {profiles[name].n_sessions} sessions)")
        else:
            profiles[name] = SpeakerProfile(
                name=name,
                embedding=emb,
                n_sessions=1,
            )
            _log(f"  New profile: {name}")
        status[name] = True

    save_profiles(profiles, profiles_path=profiles_path)
    return status


# ─── Speaker identification ───────────────────────────────────────────────────

class SpeakerMatch(NamedTuple):
    name: str
    confidence: float  # cosine similarity in [0, 1]
    # Total embeddable speech (sum of >= MIN_SEGMENT_DURATION segments) backing
    # this match, in seconds.  Callers use it to gate AUTO-APPLY: a high score
    # on a thin cluster (little real speech) is untrustworthy.  Defaults to 0.0
    # so older callers constructing SpeakerMatch by position keep working.
    evidence_seconds: float = 0.0
    # Margin = this cluster's top profile similarity minus its SECOND-best
    # profile similarity.  A small margin means the match is ambiguous (the
    # cluster looks nearly as much like another profile), which correlates
    # with false positives from noisy backchannel clusters.  Defaults to 1.0
    # ("unambiguous") so positionally-constructed matches aren't gated out.
    margin: float = 1.0


def identify_speakers(
    audio_path: Path,
    transcript_segments: list,
    speakers: list,              # list of Speaker objects (have .id attribute)
    channel_map: dict[str, str], # speaker_id -> 'mic' | 'system'
    *,
    profiles_path: Path | None = None,
) -> dict[str, SpeakerMatch]:
    """Identify speakers in a new meeting against the profile database.

    Uses cosine similarity with greedy 1:1 matching (best match first).
    Only returns matches above MATCH_THRESHOLD.

    Args:
        audio_path: Path to the session audio (OGG or WAV).
        transcript_segments: Segment objects from the new transcript.
        speakers: Speaker objects — their .id fields are used.
        channel_map: Map from speaker_id to dominant channel.
        profiles_path: Explicit path to the profile DB.  When ``None``,
            falls back to :func:`_default_profiles_path`.

    Returns:
        Dict mapping speaker_id to (matched_name, confidence).
        Speakers without a confident match are omitted.
    """
    profiles = load_profiles(profiles_path=profiles_path)
    # A reserved label must never be matched, even if an older build or a
    # hand-edited DB enrolled one.
    profiles = {n: p for n, p in profiles.items() if not is_reserved_label(n)}
    if not profiles:
        return {}

    # Build speaker_labels dict: all speaker IDs map to themselves
    # (we pass them through extract_speaker_embeddings)
    all_ids = {sp.id: sp.id for sp in speakers}

    try:
        inference = _get_inference()
    except Exception as exc:
        log.warning("Could not load embedding model: %s", exc)
        return {}

    # Extract embeddings for each speaker in the new transcript
    new_embeddings: dict[str, np.ndarray] = {}
    # Embeddable speech per speaker (sum of >= MIN_SEGMENT_DURATION segments),
    # used downstream to gate auto-apply on weak/thin clusters.
    evidence_seconds: dict[str, float] = {}
    for speaker_id in all_ids:
        segs = [
            (seg.start, seg.end)
            for seg in transcript_segments
            if seg.speaker == speaker_id and (seg.end - seg.start) >= MIN_SEGMENT_DURATION
        ]
        if not segs:
            continue
        evidence_seconds[speaker_id] = float(sum(e - s for s, e in segs))
        segs.sort(key=lambda s: s[1] - s[0], reverse=True)
        selected = segs[:MAX_SEGMENTS_PER_SPEAKER]

        channel = channel_map.get(speaker_id, "system")
        channel_data = _extract_channel_audio(audio_path, channel)
        if channel_data is None:
            continue

        samples, sr = channel_data

        # Warn when the whole channel is near-silent — individual segment
        # checks in _embed_segments handle fine-grained rejection, but a
        # channel-level check gives a more informative diagnostic.
        channel_rms = float(np.sqrt(np.mean(samples ** 2)))
        if channel_rms < MIN_SEGMENT_RMS:
            log.warning(
                "Channel '%s' for speaker %s has very low energy "
                "(rms=%.6f) — embedding quality may be poor",
                channel, speaker_id, channel_rms,
            )

        emb = _embed_segments(samples, sr, selected, inference)
        if emb is not None:
            new_embeddings[speaker_id] = emb

    if not new_embeddings:
        return {}

    # Compute cosine similarity matrix (new speakers × profiles)
    profile_names = list(profiles.keys())
    profile_matrix = np.stack([profiles[n].embedding for n in profile_names])  # (P, 256)

    speaker_ids = list(new_embeddings.keys())
    new_matrix = np.stack([new_embeddings[sid] for sid in speaker_ids])  # (S, 256)

    # Cosine similarity: since both are L2-normalized, dot product = cosine similarity
    sim_matrix = new_matrix @ profile_matrix.T  # (S, P)

    # Per-speaker match margin = top profile similarity − 2nd-best profile
    # similarity (for that speaker's row).  A small margin flags an ambiguous
    # cluster (looks nearly as much like another profile) — a common
    # false-positive signature for noisy backchannel clusters.  With a single
    # profile there is no runner-up, so margin is the top score itself.
    row_margin: dict[int, float] = {}
    for s_idx in range(sim_matrix.shape[0]):
        row = np.sort(sim_matrix[s_idx])[::-1]
        top = float(row[0])
        second = float(row[1]) if row.shape[0] > 1 else 0.0
        row_margin[s_idx] = top - second

    # Matching is two-pass:
    #
    #  Pass 1 — greedy 1:1: assign each profile to its single best-scoring
    #    cluster (highest similarity first).  This guarantees the strongest
    #    cluster for a person wins that person's name.
    #
    #  Pass 2 — many-to-one (added 0.12.11): diarization frequently
    #    over-segments ONE physical person into several clusters (volume/mic
    #    changes, cross-channel bleed, backchannel splits).  The 1:1 lock used
    #    to leave every cluster after the first as a raw placeholder, forcing a
    #    needs_labeling hop and producing duplicate "Alice ×3" entries after a
    #    human named them.  Instead, allow an already-used profile to ALSO claim
    #    additional clusters — but only when that profile is the cluster's OWN
    #    top match AND the match is independently confident (clears the
    #    auto-apply bar), so we don't fold a noisy phantom onto a real person.
    #    The later apply/relabel step collapses same-name clusters into one
    #    speaker.
    matches: dict[str, SpeakerMatch] = {}
    used_profiles: set[int] = set()

    # ── Pass 1: greedy 1:1 ──
    indices = np.argsort(-sim_matrix, axis=None)  # flat indices, sorted desc
    for flat_idx in indices:
        s_idx, p_idx = np.unravel_index(flat_idx, sim_matrix.shape)
        score = float(sim_matrix[s_idx, p_idx])
        if score < MATCH_THRESHOLD:
            break

        speaker_id = speaker_ids[s_idx]
        profile_name = profile_names[p_idx]

        if speaker_id in matches or p_idx in used_profiles:
            continue

        matches[speaker_id] = SpeakerMatch(
            name=profile_name,
            confidence=score,
            evidence_seconds=evidence_seconds.get(speaker_id, 0.0),
            margin=row_margin.get(int(s_idx), 1.0),
        )
        used_profiles.add(p_idx)

    # ── Pass 2: many-to-one for confident leftover clusters ──
    # A still-unmatched cluster may join an already-claimed profile if that
    # profile is its top match AND the fold clears the STRICTER many-to-one bar
    # (a higher absolute confidence AND a clear runner-up margin — see
    # MATCH_MANY_TO_ONE_CONFIDENCE).  This never auto-merges an ambiguous
    # cluster or two distinct-but-similar voices; those stay raw and route to
    # human review.  Brand-new identities are left to pass 1 / human review.
    for s_idx in range(sim_matrix.shape[0]):
        speaker_id = speaker_ids[s_idx]
        if speaker_id in matches:
            continue
        p_idx = int(np.argmax(sim_matrix[s_idx]))
        score = float(sim_matrix[s_idx, p_idx])
        margin = row_margin.get(int(s_idx), 1.0)
        # Only fold onto an EXISTING (already-claimed) identity.
        if p_idx not in used_profiles:
            continue
        # Higher-confidence floor for the fold: attaching to a person who is
        # already named is riskier than a fresh 1:1 match, so require a genuine
        # over-segmentation signature (very high score) rather than a merely
        # auto-appliable one.
        if score < MATCH_MANY_TO_ONE_CONFIDENCE:
            continue
        # Clear separation from the cluster's OWN runner-up profile.  A small
        # margin means the cluster resembles another profile nearly as much —
        # the ambiguous / two-different-people signature — so don't fold.
        if margin < MATCH_AUTOAPPLY_MARGIN:
            continue
        matches[speaker_id] = SpeakerMatch(
            name=profile_names[p_idx],
            confidence=score,
            evidence_seconds=evidence_seconds.get(speaker_id, 0.0),
            margin=margin,
        )

    return matches


def update_profiles_from_confirmed_labels(
    audio_path: Path,
    transcript_segments: list,
    confirmed_label_map: dict[str, str],  # speaker_id -> confirmed name
    channel_map: dict[str, str],
    *,
    profiles_path: Path | None = None,
) -> None:
    """Update profiles with confirmed labels from a just-completed meeting.

    Called automatically after the GUI's label dialog is accepted, so that
    the database improves over time without explicit `meet enroll` runs.

    Args:
        audio_path: Session audio file.
        transcript_segments: Segments from the transcript.
        confirmed_label_map: Map from speaker_id to confirmed human name.
        channel_map: Map from speaker_id to 'mic' | 'system'.
        profiles_path: Explicit path to the profile DB.  When ``None``,
            falls back to :func:`_default_profiles_path`.
    """
    # Reserved non-person labels (CROSSTALK) are mixed audio from several
    # people — a profile built from them would later "match" real speakers.
    confirmed_label_map = {
        sid: name for sid, name in confirmed_label_map.items()
        if not is_reserved_label(name)
    }
    if not confirmed_label_map:
        return

    # Only clusters with enough real speech are learned from (see
    # learnable_segments); decide before loading the model.
    selections = {
        sid: learnable_segments(transcript_segments, sid)
        for sid in confirmed_label_map
    }
    for sid, name in confirmed_label_map.items():
        if not selections[sid]:
            log.info(
                "Not learning '%s' from %s: under %.0fs of dense speech "
                "(fillers / echo / silence would pollute the profile)",
                name, sid, LEARN_MIN_SPEECH_SECONDS,
            )
    if not any(selections.values()):
        return

    try:
        inference = _get_inference()
    except Exception as exc:
        log.warning("Could not load embedding model for profile update: %s", exc)
        return

    profiles = load_profiles(profiles_path=profiles_path)
    updated = []

    for speaker_id, name in confirmed_label_map.items():
        selected = selections[speaker_id]
        if not selected:
            continue

        channel = channel_map.get(speaker_id, "system")
        channel_data = _extract_channel_audio(audio_path, channel)
        if channel_data is None:
            continue

        samples, sr = channel_data

        # Skip profile update when channel audio is near-silent — the
        # resulting embedding would be meaningless and pollute the profile.
        channel_rms = float(np.sqrt(np.mean(samples ** 2)))
        if channel_rms < MIN_SEGMENT_RMS:
            log.warning(
                "Skipping profile update for '%s' — channel '%s' is near-silent (rms=%.6f)",
                name, channel, channel_rms,
            )
            continue

        emb = _embed_segments(samples, sr, selected, inference)
        if emb is None:
            continue

        if name in profiles:
            profiles[name] = _merge_embedding(profiles[name], emb)
        else:
            profiles[name] = SpeakerProfile(name=name, embedding=emb, n_sessions=1)
        updated.append(name)

    if updated:
        save_profiles(profiles, profiles_path=profiles_path)
        log.info("Updated voice profiles for: %s", ", ".join(updated))
