"""Speaker labeling module for millet.

Provides functions to:
  - List speakers in a transcribed session
  - Extract short audio clips per speaker for voice identification
  - Play audio clips via ffplay
  - Apply user-assigned speaker names and regenerate all outputs
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from millet.audio import compute_speaker_channel_energy, read_stereo_channels
from millet.crosstalk import CROSSTALK, ghost_speakers, is_reserved_label
from millet.transcribe import Segment, Speaker, Transcript

# ─── Data ───────────────────────────────────────────────────────────────────

@dataclass
class SpeakerInfo:
    """Information about a speaker for the labeling UI."""

    id: str
    """Current speaker label (e.g. 'YOU', 'REMOTE', 'REMOTE_1')."""

    channel: str
    """Which audio channel this speaker is dominant on: 'mic' or 'system'."""

    sample_text: str
    """A short text excerpt from this speaker's longest segment."""

    sample_start: float
    """Start time (seconds) of the representative segment."""

    sample_end: float
    """End time (seconds) of the representative segment."""

    segment_count: int
    """Number of segments attributed to this speaker."""


# ─── Load transcript from session directory ─────────────────────────────────

def _find_session_files(session_dir: Path) -> dict[str, Path]:
    """Locate key files in a session directory. Returns dict of type -> path."""
    files: dict[str, Path] = {}

    # Find the JSON transcript.  Exclude sidecar/metadata JSONs that are NOT
    # the transcript: session metadata (both ``<stem>.session.json`` and a bare
    # ``session.json`` written by vezir on pull), frontmatter, translations,
    # summary meta, and the auto-id sidecar.  Selection is then deterministic
    # (not "last sorted wins"): prefer the canonical ``transcript.json`` (the
    # vezir-pulled friendly name), then ``<dirname>.json`` (the worker's stem
    # convention), else the first remaining candidate — so a lexicographically
    # later sidecar (e.g. ``session.json``) can never shadow the real transcript.
    json_candidates: list[Path] = []
    for p in sorted(session_dir.glob("*.json")):
        if p.name == "session.json" or ".session." in p.name:
            files["session"] = p
            continue
        if (
            ".frontmatter." in p.name
            or ".translation." in p.name
            or ".summary." in p.name
            or ".autoid." in p.name
            or ".clips." in p.name
            or ".meta." in p.name
        ):
            continue
        json_candidates.append(p)

    if json_candidates:
        by_name = {p.name: p for p in json_candidates}
        files["json"] = (
            by_name.get("transcript.json")
            or by_name.get(f"{session_dir.name}.json")
            or json_candidates[0]
        )

    # Find audio file (prefer WAV if still present, fall back to OGG then MP3)
    wavs = sorted(session_dir.glob("*.wav"))
    oggs = sorted(session_dir.glob("*.ogg"))
    mp3s = sorted(session_dir.glob("*.mp3"))
    if wavs:
        files["wav"] = wavs[0]
    elif oggs:
        files["wav"] = oggs[0]  # key stays "wav" for backward compat
    elif mp3s:
        files["wav"] = mp3s[0]  # key stays "wav" for backward compat

    # Find summary
    summaries = sorted(session_dir.glob("*.summary.md"))
    if summaries:
        files["summary"] = summaries[0]

    # Find PDF
    pdfs = sorted(session_dir.glob("*.pdf"))
    if pdfs:
        files["pdf"] = pdfs[0]

    return files



find_session_files = _find_session_files  # public accessor

def _load_transcript(json_path: Path) -> Transcript:
    """Reconstruct a Transcript object from a saved JSON file."""
    data = json.loads(json_path.read_text(encoding="utf-8"))

    segments = []
    for s in data["segments"]:
        segments.append(Segment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            speaker=s.get("speaker"),
            words=s.get("words"),
        ))

    speakers = []
    for sp in data.get("speakers", []):
        speakers.append(Speaker(id=sp["id"], label=sp.get("label")))

    return Transcript(
        segments=segments,
        speakers=speakers,
        language=data.get("language", "en"),
        audio_file=data.get("audio_file", ""),
        duration=data.get("duration"),
    )


# ─── Speaker analysis ──────────────────────────────────────────────────────

def _detect_speaker_channels(
    wav_path: Path,
    segments: list[Segment],
    speakers: list[Speaker],
) -> dict[str, str]:
    """Determine which audio channel each speaker is dominant on.

    Returns a dict mapping speaker ID -> 'mic' or 'system'.
    Uses the same RMS energy + margin analysis as
    ``_label_speakers_from_channels`` in transcribe.py so that the
    channel picked here matches the one used to assign the speaker
    label at transcription time.
    """
    stereo = read_stereo_channels(wav_path)
    if stereo is None:
        return {sp.id: "mic" for sp in speakers}

    mic_ratio = compute_speaker_channel_energy(
        stereo.mic, stereo.system, segments, stereo.sample_rate
    )

    if not mic_ratio:
        return {sp.id: "mic" for sp in speakers}

    best_speaker = max(mic_ratio, key=lambda s: mic_ratio[s])
    best_ratio = mic_ratio[best_speaker]
    other_ratios = [r for s, r in mic_ratio.items() if s != best_speaker]
    avg_other = sum(other_ratios) / len(other_ratios) if other_ratios else 0.0
    margin = best_ratio - avg_other

    channel_map: dict[str, str] = {}
    for sp in speakers:
        ratio = mic_ratio.get(sp.id, 0.5)
        if ratio > 0.5:
            channel_map[sp.id] = "mic"
        elif sp.id == best_speaker and margin > 0.1 and best_ratio > 0.15:
            channel_map[sp.id] = "mic"
        else:
            channel_map[sp.id] = "system"
    return channel_map


def get_speakers(session_dir: str | Path) -> list[SpeakerInfo]:
    """Get information about all speakers in a session.

    Args:
        session_dir: Path to the session directory.

    Returns:
        List of SpeakerInfo objects, one per speaker.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    session_dir = Path(session_dir)
    files = _find_session_files(session_dir)

    if "json" not in files:
        raise FileNotFoundError(f"No transcript JSON found in {session_dir}")

    transcript = _load_transcript(files["json"])

    if not transcript.speakers:
        return []

    # Detect which channel each speaker is on
    wav_path = files.get("wav")
    if wav_path and wav_path.exists():
        channel_map = _detect_speaker_channels(
            wav_path, transcript.segments, transcript.speakers,
        )
    else:
        channel_map = {sp.id: "mic" for sp in transcript.speakers}

    # Find best sample segment per speaker (longest, capped at ~15s for display)
    speaker_segments: dict[str, list[Segment]] = {}
    for seg in transcript.segments:
        if seg.speaker:
            speaker_segments.setdefault(seg.speaker, []).append(seg)

    result: list[SpeakerInfo] = []
    for sp in transcript.speakers:
        segs = speaker_segments.get(sp.id, [])
        if not segs:
            continue

        # Pick the longest segment for the sample
        best = max(segs, key=lambda s: s.end - s.start)
        sample_text = best.text.strip()[:120]
        if len(best.text.strip()) > 120:
            sample_text += "..."

        result.append(SpeakerInfo(
            id=sp.id,
            channel=channel_map.get(sp.id, "mic"),
            sample_text=sample_text,
            sample_start=best.start,
            sample_end=best.end,
            segment_count=len(segs),
        ))

    return result


# ─── Audio clip extraction ──────────────────────────────────────────────────

def extract_speaker_clip(
    wav_path: str | Path,
    speaker_info: SpeakerInfo,
    max_duration: float = 8.0,
) -> Path:
    """Extract a short audio clip for a speaker from the session audio.

    Reads the appropriate channel (mic for YOU-like speakers, system for
    REMOTE-like speakers) and writes a temporary mono WAV file.

    Args:
        wav_path: Path to the stereo session audio file (WAV or OGG).
        speaker_info: SpeakerInfo with channel hint and sample timestamps.
        max_duration: Maximum clip duration in seconds.

    Returns:
        Path to a temporary mono WAV file containing the clip.
        Caller is responsible for cleanup.
    """
    wav_path = Path(wav_path)

    # Seek-decode ONLY the clip window instead of decoding the whole file.
    # For an hour-long recording the old full-file decode (per speaker, and on
    # the GTK main thread) blocked the UI for seconds and allocated hundreds of
    # MB; -ss/-t makes this near-instant.
    sampwidth = 2  # we always request pcm_s16le below
    file_sr = 16000
    start = max(0.0, float(speaker_info.sample_start))
    duration = min(speaker_info.sample_end - speaker_info.sample_start, max_duration)
    if duration <= 0:
        raise RuntimeError(f"Empty audio clip for speaker {speaker_info.id}")

    # Probe channel count so we know whether we can select a channel.
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=channels",
         "-of", "csv=p=0", str(wav_path)],
        capture_output=True, text=True,
    )
    try:
        n_channels = int(probe.stdout.strip().split(",")[0])
    except (ValueError, IndexError):
        n_channels = 1

    cmd = [
        "ffmpeg", "-v", "quiet",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
        "-i", str(wav_path),
    ]
    if n_channels >= 2:
        # mic = channel 0 (left), system = channel 1 (right)
        ch = 0 if speaker_info.channel == "mic" else 1
        cmd += ["-filter_complex", f"[0:a]pan=mono|c0=c{ch}[out]", "-map", "[out]"]
    else:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(file_sr), "-f", "s16le", "-acodec", "pcm_s16le", "-"]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not result.stdout:
        raise RuntimeError(f"Cannot decode audio clip: {wav_path}")
    clip = np.frombuffer(result.stdout, dtype=np.int16).copy()

    if len(clip) == 0:
        raise RuntimeError(f"Empty audio clip for speaker {speaker_info.id}")

    # Normalize volume so quiet mic channels are audible during playback.
    # Target peak at ~70 % of full-scale for the sample width.
    peak = int(np.max(np.abs(clip)))
    if peak > 0:
        max_val = (1 << (sampwidth * 8 - 1)) - 1  # 32767 for int16
        target_peak = int(0.7 * max_val)
        if peak < target_peak:
            scale = target_peak / peak
            clip = np.clip(
                clip.astype(np.float64) * scale, -max_val, max_val,
            ).astype(clip.dtype)

    # Write to temp file (close the fd immediately; wave reopens by name).
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    with wave.open(tmp.name, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(sampwidth)
        out.setframerate(file_sr)
        out.writeframes(clip.tobytes())

    return Path(tmp.name)


def play_clip(clip_path: str | Path) -> subprocess.Popen:
    """Play an audio clip using ffplay.

    Args:
        clip_path: Path to a WAV file to play.

    Returns:
        The subprocess.Popen object. Call .wait() to block until playback
        finishes, or .kill() to stop early.
    """
    return subprocess.Popen(
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(clip_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ─── Leftover REMOTE bucket → CROSSTALK; tiny noise → dominant speaker ──────

import re as _re  # noqa: E402

# Raw, auto-generated speaker ids that mean "not yet identified".
_RAW_SPEAKER_RE = _re.compile(r"^(REMOTE(?:_\d+)?|SPEAKER_\d+)$")

# A "tiny" raw cluster is spurious noise — a one-liner backchannel or heavily
# distorted blip on the system channel that diarization split off and that
# voiceprint can't match.  It is folded into the DOMINANT speaker by speech
# time even when that dominant speaker is itself still raw/unmatched, so a
# single noise blip no longer forces an otherwise-clean session into review.
TINY_SPEAKER_MAX_SECONDS = 5.0
TINY_SPEAKER_MAX_SEGMENTS = 3


def _is_raw_speaker(sid: str) -> bool:
    return bool(_RAW_SPEAKER_RE.match(sid or ""))


def crosstalk_label_map(
    transcript: Transcript,
    resolved_ids: set[str],
) -> dict[str, str]:
    """Map every ghost ``REMOTE``/``REMOTE_N`` bucket to :data:`CROSSTALK`.

    The dual-diarize path creates a literal ``REMOTE`` bucket *after* cluster
    consolidation, so it never merges; it collects sub-second fillers from
    several people that no voiceprint can match.  Rather than guess an owner,
    a bucket that looks like filler crosstalk (see
    :func:`millet.crosstalk.is_ghost_speaker`) is labeled ``CROSSTALK`` —
    honest, human-readable, and no longer an unresolved speaker.  Several
    ghost buckets collapse into one ``CROSSTALK`` speaker.  A substantial
    ``REMOTE`` is left raw for human review.

    ``resolved_ids`` are the *raw* ids already mapped to a name (the keys of
    the label map) — they are never touched.  Pure/deterministic.
    """
    return {
        sid: CROSSTALK
        for sid in ghost_speakers(transcript.segments, resolved_ids)
    }


def absorb_tiny_speakers(
    transcript: Transcript,
    resolved_ids: set[str],
) -> dict[str, str]:
    """Fold spurious *tiny* raw clusters into the dominant speaker.

    Runs after :func:`crosstalk_label_map`, so ghost ``REMOTE`` buckets are
    already ``CROSSTALK``; what remains here is a 1-3 segment / few-second raw
    cluster (a backchannel one-liner or a distorted noise blip) — typically a
    ``SPEAKER_n`` split off by pyannote.  That single tiny cluster otherwise
    forces the whole session into ``needs_labeling`` even though there is
    nothing meaningful to label.

    ``resolved_ids`` are the *raw* ids already mapped to a name (label-map
    keys, not the names): a short cluster that voiceprint confidently matched
    is not noise and must keep its match.

    For each raw cluster that is TINY (``<= TINY_SPEAKER_MAX_SECONDS`` of speech
    AND ``<= TINY_SPEAKER_MAX_SEGMENTS`` segments) and not already resolved,
    fold it into the speaker with the most total speech time (preferring the
    speaker it overlaps most, falling back to the overall dominant speaker).
    The target may itself be a raw id — the point is to collapse noise into the
    real conversation, not to name it.

    Returns ``{tiny_id: target_id}`` (possibly empty).  Pure/deterministic; the
    caller folds the result into the label_map.  A tiny cluster is never folded
    into another tiny cluster, and if *every* speaker is tiny nothing is folded
    (the whole session is noise — let the worker decide its fate).
    """
    speech: dict[str, float] = {}
    count: dict[str, int] = {}
    segs_by: dict[str, list[tuple[float, float]]] = {}
    for seg in transcript.segments:
        sid = seg.speaker or ""
        if not sid:
            continue
        s, e = float(seg.start), float(seg.end)
        speech[sid] = speech.get(sid, 0.0) + max(0.0, e - s)
        count[sid] = count.get(sid, 0) + 1
        segs_by.setdefault(sid, []).append((s, e))

    if not speech:
        return {}

    def _tiny(sid: str) -> bool:
        return (
            speech.get(sid, 0.0) <= TINY_SPEAKER_MAX_SECONDS
            and count.get(sid, 0) <= TINY_SPEAKER_MAX_SEGMENTS
        )

    tiny = {
        sid for sid in speech
        if _is_raw_speaker(sid) and sid not in resolved_ids and _tiny(sid)
    }
    if not tiny:
        return {}

    # Candidate targets: any non-tiny speaker (named or raw).  If all speakers
    # are tiny, there's no real conversation to fold into — bail.
    targets = [sid for sid in speech if sid not in tiny]
    if not targets:
        return {}

    def _overlap_score(a: tuple[float, float], sid: str) -> float:
        s0, e0 = a
        total = 0.0
        nearest = float("inf")
        for s1, e1 in segs_by.get(sid, []):
            ov = min(e0, e1) - max(s0, s1)
            if ov > 0:
                total += ov
            else:
                nearest = min(nearest, max(s1 - e0, s0 - e1))
        if total > 0:
            return total
        return -nearest if nearest != float("inf") else -1e9

    absorb: dict[str, str] = {}
    for tid in tiny:
        scores: dict[str, float] = {}
        for s, e in segs_by[tid]:
            for sid in targets:
                scores[sid] = scores.get(sid, 0.0) + _overlap_score((s, e), sid)
        # Tie-break / no-overlap fallback: the overall dominant speaker by speech.
        if not scores or max(scores.values()) <= 0:
            best = max(targets, key=lambda s: speech.get(s, 0.0))
        else:
            best = max(scores, key=lambda s: scores[s])
        absorb[tid] = best
    return absorb


# ─── Apply labels ──────────────────────────────────────────────────────────

def _write_clips_best_effort(
    summary_result, transcript, session_dir: Path, basename: str,
    template: str | None, lang_suffix: str | None,
    result_files: dict, log_fn,
) -> None:
    """Write the ``video-clipping`` cut list; never fails the summary.

    Clip boundaries come from the FULL transcript (CROSSTALK included): the
    cut must match the video, not the filtered summary input.
    """
    try:
        from millet.clips import is_clips_template, write_clips_doc

        if not is_clips_template(template):
            return
        path, doc = write_clips_doc(
            summary_result, transcript, session_dir, basename, lang_suffix=lang_suffix,
        )
        result_files["clips"] = path
        log_fn(
            f"Clips written: {path.name} ({len(doc['clips'])} kept, "
            f"{len(doc['dropped'])} dropped)"
        )
    except Exception as exc:
        log_fn(f"Clip extraction failed: {exc}")


def apply_labels(
    session_dir: str | Path,
    label_map: dict[str, str],
    regenerate_summary: bool = True,
    summary_preset: str | None = None,
    summary_backend: str | None = None,
    summary_model: str | None = None,
    ollama_singlepass: bool = False,
    summary_language: str | None = None,
    summary_template: str | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Path]:
    """Apply user-assigned speaker names to a session's outputs.

    Relabels the transcript and regenerates all output files (txt, srt, json,
    summary, pdf).

    Args:
        session_dir: Path to the session directory.
        label_map: Mapping of current speaker ID to new name.
            E.g. {"YOU": "Kasita", "REMOTE_1": "Ahmad"}.
            Only speakers present in the map are relabeled.
        regenerate_summary: If True, re-run the LLM to generate a new summary
            with updated speaker names. If False, do a best-effort
            find-and-replace on the existing summary.
        summary_preset: Optional summarization preset id
            ("high-quality" | "confidential" | "alternative").  Forwarded
            to SummaryConfig when regenerate_summary is True so the same
            preset guard semantics apply during relabel-driven summary
            regeneration as during the initial transcribe pass.
        summary_backend: Backend override ("tinfoil" or "ollama"); None uses default.
        summary_model: Model name override; None uses the per-backend default.
        summary_language: Optional language override for the regenerated
            summary.  When set (e.g. "de"), the summary is generated in that
            language and saved as an ADDITIONAL ``<basename>.summary.<lang>.md``
            file, leaving the primary auto-detected ``<basename>.summary.md``
            intact.  When None, the transcript's own language is used and the
            primary summary file is (re)written.
        summary_template: Optional summary prompt template name (e.g.
            "iteration-plan").  Uses the template's prompt files and writes
            ``<basename>.<template>.md`` instead of ``<basename>.summary.md``.
        progress_callback: Optional callable(str) for status messages.

    Returns:
        Dict of format name -> file path for all updated files.

    Raises:
        FileNotFoundError: If required session files are missing.
    """
    session_dir = Path(session_dir)
    files = _find_session_files(session_dir)

    if "json" not in files:
        raise FileNotFoundError(f"No transcript JSON found in {session_dir}")

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    transcript = _load_transcript(files["json"])
    basename = files["json"].stem  # e.g. meeting-20260313-231509

    # ── Relabel transcript in-memory ──
    _log("Relabeling transcript...")
    transcript = relabel_transcript_in_memory(transcript, label_map)

    # ── Save updated transcript files ──
    _log("Saving updated transcript files...")
    result_files = transcript.save(session_dir, basename=basename)

    # ── Summary ──
    summary_result = None

    if regenerate_summary:
        _log("Regenerating summary with updated speaker names...")
        try:
            from millet.summarize import (
                SummaryConfig,
                _backend_not_available_message,
                discover_cue_frames,
                is_backend_available,
            )
            from millet.summarize import (
                summarize as do_summarize,
            )
            from millet.transcribe import ensure_gpu_available

            cfg_kwargs: dict = {}
            if summary_preset:
                cfg_kwargs["preset"] = summary_preset
            if summary_backend:
                cfg_kwargs["backend"] = summary_backend
            if summary_model:
                cfg_kwargs["model"] = summary_model
            if ollama_singlepass:
                cfg_kwargs["ollama_singlepass"] = True
            if summary_template:
                cfg_kwargs["template"] = summary_template
            # Narrated screen recordings: show the model the screen at each
            # cue, not just the narration.  Frames are written next to the
            # session by the caller (vezir) *after* transcription, so they
            # only exist on a re-summary like this one -- which is exactly
            # the path that regenerates an iteration plan.  Missing frames
            # are normal (audio meetings) and simply summarize as before.
            frames = discover_cue_frames(session_dir)
            if frames:
                cfg_kwargs["frames"] = frames
            summary_config = SummaryConfig(**cfg_kwargs)
            if frames:
                _log(
                    f"Found {len(frames)} cue frame(s) alongside the session; "
                    "including them in the summary"
                )

            if is_backend_available(summary_config):
                # Only free GPU for Ollama backend
                if summary_config.backend == "ollama":
                    ensure_gpu_available()
                # Language override → generate in that language and save as an
                # ADDITIONAL <basename>.summary.<lang>.md (don't clobber the
                # primary auto-detected summary).
                effective_language = summary_language or transcript.language
                lang_suffix = summary_language or None
                # CROSSTALK lines are unattributable fillers ("Bye.", "Yeah.")
                # — no content, and the model would list "CROSSTALK" as a
                # participant.  They stay in txt/srt/json/PDF, not the summary.
                summary_input = replace(
                    transcript,
                    segments=[
                        s for s in transcript.segments
                        if not is_reserved_label(s.speaker)
                    ],
                )
                summary_result = do_summarize(
                    summary_input.to_text(), summary_config,
                    language=effective_language,
                )
                from millet.frontmatter import context_from_transcript

                fm_ctx = context_from_transcript(transcript, session_dir)
                # The RESOLVED template (explicit arg > MILLET_SUMMARY_TEMPLATE,
                # lowercased by SummaryConfig): the raw argument is None when
                # the template comes from the environment.
                template = summary_config.template
                artifact = template or "summary"
                path = summary_result.save(
                    session_dir, basename, frontmatter_context=fm_ctx,
                    lang_suffix=lang_suffix,
                    artifact=artifact,
                )
                result_files[artifact] = path
                _write_clips_best_effort(
                    summary_result, transcript, session_dir, basename,
                    template, lang_suffix, result_files, _log,
                )
                if lang_suffix:
                    _log(
                        f"Additional '{lang_suffix}' summary generated in "
                        f"{summary_result.elapsed_seconds:.1f}s"
                    )
                else:
                    _log(
                        f"Summary regenerated in "
                        f"{summary_result.elapsed_seconds:.1f}s"
                    )
            else:
                _log(_backend_not_available_message(summary_config))
                # Fall back to find-and-replace
                regenerate_summary = False
        except Exception as exc:
            _log(f"Summary regeneration failed: {exc}")
            if summary_preset:
                # Preset was explicitly requested — user chose a specific
                # privacy/quality level.  Mirror the post_process() contract
                # in transcribe.py: surface the failure as an exception so
                # callers (vezir, automation) can react.  Standalone callers
                # that don't pass summary_preset keep the silent-skip behavior.
                raise
            regenerate_summary = False

    if not regenerate_summary and files.get("summary") and files["summary"].exists():
        # Best-effort find-and-replace on existing summary.
        # The summary may carry a YAML frontmatter block — split it off so
        # we don't accidentally rename inside structural keys, and so the
        # PDF gets a clean Markdown body.
        _log("Updating speaker names in existing summary...")
        from millet.frontmatter import (
            parse_frontmatter_block,
            render_frontmatter_block,
            write_frontmatter_sidecar,
        )

        raw = files["summary"].read_text(encoding="utf-8")
        fm_dict, body = parse_frontmatter_block(raw)

        def _replace_all(s: str) -> str:
            # Single-pass alternation so each source position is rewritten
            # exactly once from the ORIGINAL text.  Sequential per-key
            # re.sub chained on the same string, which corrupted a SWAP:
            # {"Alaaddin":"Kemal","Kemal":"Alaaddin"} first turned every
            # "Alaaddin" into "Kemal", then the second pass re-caught those
            # just-written "Kemal"s and collapsed BOTH speakers to one name.
            # Alternation is sorted longest-first so overlapping labels match
            # the longer one (REMOTE_1 before REMOTE; SPEAKER_10 before
            # SPEAKER_1).  \b guards short labels like "YOU" from prose.
            import re as _re

            if not label_map:
                return s
            keys = sorted(label_map, key=len, reverse=True)
            pattern = _re.compile(
                r"\b(?:" + "|".join(_re.escape(k) for k in keys) + r")\b"
            )
            # Callable repl: value inserted literally (no backreference
            # interpretation).  Each match resolves against the original map,
            # never against a previously-substituted value.
            return pattern.sub(lambda m: label_map[m.group(0)], s)

        body = _replace_all(body)

        if fm_dict is not None:
            # Walk the structured frontmatter and apply the same renames
            # to participant names, action_item.assignee, etc.  We only
            # touch string fields the schema knows about, leaving anything
            # exotic alone.
            for p in fm_dict.get("participants") or []:
                if isinstance(p, dict) and isinstance(p.get("name"), str):
                    p["name"] = _replace_all(p["name"])
            # A participant relabeled to CROSSTALK is not a person — drop it.
            if isinstance(fm_dict.get("participants"), list):
                fm_dict["participants"] = [
                    p for p in fm_dict["participants"]
                    if not (isinstance(p, dict) and is_reserved_label(p.get("name")))
                ]
            for ai in fm_dict.get("action_items") or []:
                if isinstance(ai, dict):
                    if isinstance(ai.get("assignee"), str):
                        ai["assignee"] = _replace_all(ai["assignee"])
                    if isinstance(ai.get("task"), str):
                        ai["task"] = _replace_all(ai["task"])
            for d in fm_dict.get("decisions") or []:
                if isinstance(d, dict) and isinstance(d.get("text"), str):
                    d["text"] = _replace_all(d["text"])
            new_text = render_frontmatter_block(fm_dict) + body
            write_frontmatter_sidecar(session_dir, basename, fm_dict)
        else:
            new_text = body

        files["summary"].write_text(new_text, encoding="utf-8")
        result_files["summary"] = files["summary"]

        # Load body-only summary for PDF embedding.
        # Read the original summary meta to preserve model name and backend
        # (needed for CONFIDENTIAL watermark on TEE-backed summaries).
        from millet.summarize import MeetingSummary
        orig_model = "(relabeled)"
        orig_backend = ""
        orig_elapsed = 0.0
        meta_path = session_dir / f"{basename}.summary.meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                orig_model = meta.get("model", orig_model)
                orig_backend = meta.get("backend", orig_backend)
                orig_elapsed = meta.get("elapsed_seconds", orig_elapsed)
            except Exception:
                pass
        summary_result = MeetingSummary(
            markdown=body, model=orig_model, elapsed_seconds=orig_elapsed,
            backend=orig_backend,
        )

    if not regenerate_summary and label_map:
        # video-clipping: clip boundaries don't move on a relabel, only the
        # speaker names inside each clip do.
        from millet.clips import relabel_clips_file

        clips_files = sorted(
            p for p in session_dir.iterdir()
            if p.is_file() and p.name.startswith(f"{basename}.clips.")
            and p.name.endswith(".json")
        )
        for clips_file in clips_files:
            try:
                if relabel_clips_file(clips_file, label_map):
                    result_files.setdefault("clips", clips_file)
            except Exception as exc:
                _log(f"Clip relabel failed for {clips_file.name}: {exc}")

    # ── PDF ──
    _log("Regenerating PDF...")
    try:
        from millet.pdf import generate_pdf

        pdf_path = session_dir / f"{basename}.pdf"
        generate_pdf(
            transcript, pdf_path,
            summary=summary_result,
            language=transcript.language,
        )
        result_files["pdf"] = pdf_path
    except Exception as exc:
        _log(f"PDF regeneration failed: {exc}")

    # ── Update session.json with label mapping ──
    if files.get("session") and files["session"].exists():
        _log("Updating session metadata...")
        try:
            meta = json.loads(files["session"].read_text(encoding="utf-8"))
            # Merge, don't overwrite: a later labeling pass (e.g. naming a
            # remaining REMOTE_2) must not discard names confirmed earlier,
            # which enroll_session depends on being complete.
            existing = meta.get("speaker_labels")
            if isinstance(existing, dict):
                existing.update(label_map)
                meta["speaker_labels"] = existing
            else:
                meta["speaker_labels"] = label_map
            files["session"].write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
            )
        except Exception:
            pass

    _log("Done!")
    return result_files


def relabel_transcript_in_memory(
    transcript: Transcript,
    label_map: dict[str, str],
) -> Transcript:
    """Apply label_map to a Transcript object and return a new one.

    This is used by the GUI to relabel in-memory before the first save,
    avoiding the need to regenerate outputs.

    Also collapses speakers that resolve to the same id into one entry, so an
    empty ``label_map`` is still meaningful: it de-duplicates a transcript that
    already carries duplicate-named speakers (e.g. an older session where one
    person was labeled across several clusters before 0.12.11).
    """
    # Nothing to relabel AND no duplicate speakers → return as-is.
    resolved_ids = [label_map.get(sp.id, sp.id) for sp in transcript.speakers]
    has_dupes = len(resolved_ids) != len(set(resolved_ids))
    if not label_map and not has_dupes:
        return transcript

    new_segments = []
    for seg in transcript.segments:
        new_speaker = label_map.get(seg.speaker, seg.speaker) if seg.speaker else seg.speaker
        new_segments.append(Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            speaker=new_speaker,
            words=seg.words,
        ))

    # De-duplicate speakers by resolved id (added 0.12.11).  Diarization can
    # over-segment one person into several clusters; once they resolve to the
    # same name (via many-to-one voiceprint matching or a human labeling each
    # cluster), they must collapse into a SINGLE speaker entry — otherwise the
    # transcript/summary/PDF show e.g. "Destiny, Destiny, Destiny".  Segments
    # already point at the merged id above, so this only needs to dedupe the
    # speakers list while preserving first-seen order.
    new_speakers = []
    seen_ids: set[str] = set()
    for sp in transcript.speakers:
        new_id = label_map.get(sp.id, sp.id)
        if new_id in seen_ids:
            continue
        seen_ids.add(new_id)
        new_speakers.append(Speaker(id=new_id, label=new_id))

    return Transcript(
        segments=new_segments,
        speakers=new_speakers,
        language=transcript.language,
        audio_file=transcript.audio_file,
        duration=transcript.duration,
    )
