"""Clip extraction for the ``video-clipping`` summary template.

The LLM proposes clip-worthy moments with second-granular timestamps copied
from the transcript text (``[HH:MM:SS --> HH:MM:SS]``).  Those are cues, not
cut points: they are floored to the second, and a model can mis-copy or
invent one.  This module turns them into cut points that are safe to hand to
ffmpeg by snapping every boundary onto a real segment boundary from the
transcript, padding into the silence around it, and enforcing the duration
window of each clip kind.  Nothing is guessed: a clip whose boundaries can't
be matched to the transcript is dropped *with a recorded reason*, never
silently.

The result is written as ``<basename>.clips.json`` (``.clips.<lang>.json``
for an additional-language summary) next to the template summary.  It is
the machine-readable artifact an agent uses to cut the clips; the Markdown
summary is for humans.

Timestamps here are on the TRANSCRIPT timeline (seconds from the start of
the audio).  A caller that knows the source video's timeline offset (vezir
probes ``start_time`` with ffprobe) adds it under ``source`` and rebases the
ffmpeg arguments; millet never sees the video.

Pure stdlib, no LLM, no I/O beyond the final write.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CLIPS_TEMPLATE = "video-clipping"
SCHEMA_VERSION = 1

# Target windows from the prompt (short 15-60 s, long 60-180 s), widened to
# tolerate the padding and segment snapping.  Outside these a clip is
# dropped, not silently re-classified.
KIND_BOUNDS: dict[str, tuple[float, float]] = {
    "short": (10.0, 75.0),
    "long": (45.0, 240.0),
}

# Lead-in / tail padding, clamped so a clip never bleeds into the
# neighbouring segment's speech.
PAD_BEFORE_S = 0.25
PAD_AFTER_S = 0.5

# How far (seconds) a requested timestamp may sit from the segment boundary
# it snaps to.  Timestamps in the transcript text are floored to the second,
# so an exact copy is always within [0, 1); anything beyond this tolerance
# matches no line and the clip is dropped.
SNAP_TOLERANCE_S = 5.0

# Clips of the same kind may touch through their padding, not overlap.
MAX_SAME_KIND_OVERLAP_S = 1.0

MAX_CLIPS = {"short": 15, "long": 8}

_TS_RE = re.compile(r"^\[?\s*(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*\]?$")


def is_clips_template(template: str | None) -> bool:
    return (template or "").strip().lower() == CLIPS_TEMPLATE


def parse_timestamp(value: Any) -> float | None:
    """Parse ``HH:MM:SS``, ``MM:SS``, ``[HH:MM:SS]`` or plain seconds.

    Returns ``None`` for anything that is not a finite, non-negative time —
    including values that overflow a float: JSON integers are arbitrary
    precision (``1`` followed by 400 zeros parses fine), and so is a huge
    hour field inside an ``HH:MM:SS`` string.
    """
    try:
        return _parse_timestamp(value)
    except (OverflowError, ValueError):
        # OverflowError: int too large for a float.  ValueError: an int
        # string beyond Python's int_max_str_digits limit (3.11+).
        return None


def _parse_timestamp(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        # JSON accepts 1e309 (-> inf); a non-finite cue matches nothing.
        return float(value) if math.isfinite(value) and value >= 0 else None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        f = float(s)  # also parses "inf" / "nan"
    except ValueError:
        pass
    else:
        return f if math.isfinite(f) and f >= 0 else None
    m = _TS_RE.match(s)
    if not m:
        return None
    h = int(m.group(1) or 0)
    mn = int(m.group(2))
    sec = float(m.group(3))
    if mn >= 60 or sec >= 60:
        return None
    total = h * 3600 + mn * 60 + sec
    return total if math.isfinite(total) else None


def fmt_ts(seconds: float) -> str:
    """``HH:MM:SS.mmm``."""
    ms = max(0, round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def ffmpeg_args(seek_s: float, duration_s: float) -> list[str]:
    """Frame-accurate re-encoding cut.  ``{input}``/``{output}`` are placeholders.

    Stream copy (``-c copy``) cuts on keyframes and would start a clip up to
    a few seconds early, so the suggestion re-encodes.
    """
    return [
        "-ss", f"{seek_s:.3f}", "-i", "{input}", "-t", f"{duration_s:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "{output}",
    ]


def _seg_field(seg: Any, name: str) -> Any:
    if isinstance(seg, dict):
        return seg.get(name)
    return getattr(seg, name, None)


def _normalize_segments(segments: list[Any]) -> list[dict[str, Any]]:
    out = []
    for seg in segments or []:
        try:
            start = float(_seg_field(seg, "start"))
            end = float(_seg_field(seg, "end"))
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        out.append({
            "start": start,
            "end": end,
            "text": str(_seg_field(seg, "text") or "").strip(),
            "speaker": _seg_field(seg, "speaker"),
            "words": _seg_field(seg, "words") or [],
        })
    out.sort(key=lambda s: s["start"])
    return out


def _snap(t: float, segs: list[dict[str, Any]], edge: str) -> int | None:
    """Index of the segment whose ``edge`` ("start"/"end") best matches ``t``.

    Preference: a boundary in ``[t - 0.5, t + 1)`` (an exact copy of a
    floored timestamp), then the segment containing ``t``, then the nearest
    boundary within :data:`SNAP_TOLERANCE_S`.
    """
    best = None
    best_d = None
    for i, s in enumerate(segs):
        d = s[edge] - t
        if -0.5 <= d < 1.0 and (best_d is None or abs(d) < best_d):
            best, best_d = i, abs(d)
    if best is not None:
        return best
    for i, s in enumerate(segs):
        if s["start"] <= t <= s["end"]:
            return i
    for i, s in enumerate(segs):
        d = abs(s[edge] - t)
        if d <= SNAP_TOLERANCE_S and (best_d is None or d < best_d):
            best, best_d = i, d
    return best


def _coerce_score(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):  # Overflow: a huge JSON int
        return None
    if not math.isfinite(f):  # round(inf) raises OverflowError
        return None
    return max(1, min(10, round(f)))


_MAX_ECHO_CHARS = 200
# Integers are always finite and JSON-serializable, but a model can emit an
# arbitrarily long one; echo only what fits in 64 bits.
_MAX_ECHO_INT = 2**63 - 1


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Echo model-supplied data into ``requested``/``dropped`` safely.

    Non-finite floats and integers beyond 64 bits become ``None``
    (``json.dumps`` would otherwise emit the non-standard ``Infinity``/``NaN``,
    and ``math.isfinite`` on a huge int raises), long strings are truncated,
    and anything nested or exotic becomes a short ``repr``.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if abs(value) <= _MAX_ECHO_INT else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:_MAX_ECHO_CHARS]
    if depth < 2 and isinstance(value, list):
        return [_json_safe(v, depth + 1) for v in value[:20]]
    if depth < 2 and isinstance(value, dict):
        return {str(k)[:50]: _json_safe(v, depth + 1) for k, v in list(value.items())[:20]}
    return repr(value)[:_MAX_ECHO_CHARS]


def _dumps(doc: dict[str, Any]) -> str:
    # allow_nan=False: the file is a machine contract; never emit Infinity.
    return json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _str(v: Any) -> str:
    return v.strip() if isinstance(v, str) else ""


def _clip_words(
    segs: list[dict[str, Any]], first: int, last: int, start_s: float, end_s: float,
) -> list[dict[str, Any]]:
    """Word timings inside the clip, rebased so 0.0 is the clip's first frame."""
    words = []
    for s in segs[first:last + 1]:
        for w in s["words"]:
            if not isinstance(w, dict):
                continue
            ws, we = w.get("start"), w.get("end")
            text = _str(w.get("word"))
            if not text or not isinstance(ws, (int, float)) or not isinstance(we, (int, float)):
                continue
            if we < start_s or ws > end_s:
                continue
            words.append({
                "word": text,
                "start": round(max(0.0, ws - start_s), 3),
                "end": round(max(0.0, we - start_s), 3),
            })
    return words


def build_clips(
    data: dict[str, Any] | None,
    segments: list[Any],
    *,
    duration: float | None = None,
    data_error: str | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``clips.json`` document from the LLM's structured data.

    ``segments`` are transcript segments (``Segment`` objects or dicts with
    ``start``/``end``/``text``/``speaker``/``words``).  Always returns a
    document: when the model emitted no usable clips, ``clips`` is empty and
    ``error`` says why, so a consumer can tell "nothing worth clipping" from
    "the model broke the contract".
    """
    segs = _normalize_segments(segments)
    if duration is None or duration <= 0:
        duration = segs[-1]["end"] if segs else 0.0

    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "template": CLIPS_TEMPLATE,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "timeline": "transcript",
        "source": dict(source or {}),
        "transcript_duration_s": round(float(duration), 3),
        "padding": {"before_s": PAD_BEFORE_S, "after_s": PAD_AFTER_S},
        "clips": [],
        "dropped": [],
        "error": None,
    }

    raw_clips = data.get("clips") if isinstance(data, dict) else None
    if not isinstance(data, dict):
        doc["error"] = data_error or "model emitted no structured data"
        return doc
    if not isinstance(raw_clips, list):
        doc["error"] = "model output has no 'clips' list"
        return doc
    if not segs:
        doc["error"] = "transcript has no timed segments"
        return doc

    candidates: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = doc["dropped"]

    for raw in raw_clips:
        if not isinstance(raw, dict):
            dropped.append({"requested": _json_safe(raw),
                            "reason": "clip entry is not an object"})
            continue
        req = {"start": _json_safe(raw.get("start")), "end": _json_safe(raw.get("end"))}
        title = _str(raw.get("title"))
        kind = _str(raw.get("kind")).lower()

        def _drop(reason: str, *, req=req, title=title, kind=kind) -> None:
            dropped.append({"kind": kind or None, "title": title or None,
                            "requested": req, "reason": reason})

        if kind not in KIND_BOUNDS:
            _drop(f"unknown kind {_json_safe(raw.get('kind'))!r} (expected short|long)")
            continue
        t0 = parse_timestamp(raw.get("start"))
        t1 = parse_timestamp(raw.get("end"))
        if t0 is None or t1 is None:
            _drop("unparseable start/end timestamp")
            continue
        first = _snap(t0, segs, "start")
        last = _snap(t1, segs, "end")
        if first is None or last is None:
            _drop("timestamp matches no transcript line")
            continue
        if last < first:
            _drop("end is before start")
            continue

        seg_start, seg_end = segs[first]["start"], segs[last]["end"]
        prev_end = segs[first - 1]["end"] if first > 0 else 0.0
        next_start = segs[last + 1]["start"] if last + 1 < len(segs) else duration
        # Pad into the surrounding silence, never into a neighbour's speech.
        start_s = min(seg_start, max(0.0, seg_start - PAD_BEFORE_S, prev_end))
        # ``duration`` is always positive here (it falls back to the last
        # segment end above), so the clamp never collapses a clip to zero.
        end_s = min(max(seg_end, min(seg_end + PAD_AFTER_S, next_start)), float(duration))
        length = end_s - start_s
        lo, hi = KIND_BOUNDS[kind]
        if length < lo or length > hi:
            _drop(f"{length:.1f}s is outside the {kind} window ({lo:.0f}-{hi:.0f}s)")
            continue

        lines = [
            f"{s['speaker'] or 'UNKNOWN'}: {s['text']}" for s in segs[first:last + 1]
        ]
        speakers = []
        for s in segs[first:last + 1]:
            if s["speaker"] and s["speaker"] not in speakers:
                speakers.append(s["speaker"])
        tags = raw.get("tags")
        candidates.append({
            "kind": kind,
            "title": title or None,
            "hook": _str(raw.get("hook")) or None,
            "reason": _str(raw.get("reason")) or None,
            "score": _coerce_score(raw.get("score")),
            "tags": [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]
            if isinstance(tags, list) else [],
            "requested": req,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "duration_s": round(length, 3),
            "start_ts": fmt_ts(start_s),
            "end_ts": fmt_ts(end_s),
            "segments": [first, last],
            "speakers": speakers,
            "text": "\n".join(lines),
            "words": _clip_words(segs, first, last, start_s, end_s),
        })

    # Same-kind overlap: keep the higher-scored clip (earlier one on a tie).
    kept: list[dict[str, Any]] = []
    order = sorted(
        enumerate(candidates),
        key=lambda ic: (-(ic[1]["score"] or 0), ic[0]),
    )
    counts = {k: 0 for k in KIND_BOUNDS}
    for _, c in order:
        clash = next(
            (k for k in kept if k["kind"] == c["kind"]
             and min(k["end_s"], c["end_s"]) - max(k["start_s"], c["start_s"])
             > MAX_SAME_KIND_OVERLAP_S),
            None,
        )
        if clash is not None:
            dropped.append({"kind": c["kind"], "title": c["title"],
                            "requested": c["requested"],
                            "reason": f"overlaps {clash['kind']} clip "
                                      f"{clash['title']!r} ({clash['start_ts']})"})
            continue
        if counts[c["kind"]] >= MAX_CLIPS[c["kind"]]:
            dropped.append({"kind": c["kind"], "title": c["title"],
                            "requested": c["requested"],
                            "reason": f"more than {MAX_CLIPS[c['kind']]} {c['kind']} clips"})
            continue
        counts[c["kind"]] += 1
        kept.append(c)

    kept.sort(key=lambda c: (c["start_s"], c["kind"]))
    seq = {k: 0 for k in KIND_BOUNDS}
    for c in kept:
        seq[c["kind"]] += 1
        c["id"] = f"{c['kind']}-{seq[c['kind']]:02d}"
        c["ffmpeg_args"] = ffmpeg_args(c["start_s"], c["duration_s"])
    # Stable key order: id first.
    doc["clips"] = [{"id": c.pop("id"), **c} for c in kept]
    if raw_clips and not kept and not doc["error"]:
        doc["error"] = "every proposed clip was dropped (see 'dropped')"
    return doc


def relabel_clips_file(path: str | Path, label_map: dict[str, str]) -> bool:
    """Apply a speaker label map to an existing clips file in place.

    Only exact speaker labels change (``speakers`` entries and the
    ``SPEAKER: `` prefix of each ``text`` line); boundaries never move.
    Returns True when the file was rewritten.
    """
    if not label_map:
        return False
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("clips"), list):
        return False
    changed = False
    for clip in doc["clips"]:
        if not isinstance(clip, dict):
            continue
        speakers = clip.get("speakers")
        if isinstance(speakers, list):
            renamed: list[Any] = []
            for s in speakers:
                new = label_map.get(s, s) if isinstance(s, str) else s
                if new not in renamed:
                    renamed.append(new)
            if renamed != speakers:
                clip["speakers"] = renamed
                changed = True
        text = clip.get("text")
        if isinstance(text, str):
            lines = []
            for line in text.split("\n"):
                label, sep, rest = line.partition(": ")
                if sep and label in label_map:
                    line = f"{label_map[label]}: {rest}"
                lines.append(line)
            new_text = "\n".join(lines)
            if new_text != text:
                clip["text"] = new_text
                changed = True
    if changed:
        text = _dumps(doc)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    return changed


def clips_path(output_dir: str | Path, basename: str, lang_suffix: str | None = None) -> Path:
    suffix = f".{lang_suffix}" if lang_suffix else ""
    return Path(output_dir) / f"{basename}.clips{suffix}.json"


def write_clips(
    summary_result: Any,
    transcript: Any,
    output_dir: str | Path,
    basename: str,
    *,
    lang_suffix: str | None = None,
) -> Path:
    """Build and write ``<basename>.clips[.<lang>].json``; returns its path."""
    return write_clips_doc(
        summary_result, transcript, output_dir, basename, lang_suffix=lang_suffix,
    )[0]


def write_clips_doc(
    summary_result: Any,
    transcript: Any,
    output_dir: str | Path,
    basename: str,
    *,
    lang_suffix: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Build and write ``<basename>.clips[.<lang>].json`` for a template summary.

    ``summary_result`` is a ``MeetingSummary`` (its ``data``/``data_error``
    carry the parsed JSON block); ``transcript`` is the FULL transcript
    (CROSSTALK included — the clip text should match what the video says).
    Returns ``(path, document)`` so callers can report without reading back.
    """
    audio = getattr(transcript, "audio_file", "") or ""
    doc = build_clips(
        getattr(summary_result, "data", None),
        list(getattr(transcript, "segments", []) or []),
        duration=getattr(transcript, "duration", None),
        data_error=getattr(summary_result, "data_error", None),
        source={
            "transcript": f"{basename}.json",
            "audio": Path(audio).name if audio else None,
            "language": getattr(transcript, "language", None),
        },
    )
    doc["summary"] = {
        "backend": getattr(summary_result, "backend", None),
        "model": getattr(summary_result, "model", None),
    }
    path = clips_path(output_dir, basename, lang_suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _dumps(doc)  # serialize BEFORE touching the filesystem
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path, doc
