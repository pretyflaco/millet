"""CROSSTALK: the honest label for speech that cannot be attributed.

The dual-diarize path leaves a literal ``REMOTE`` bucket (system segments
pyannote left unassigned, plus mic-bleed with no overlapping diarized remote).
In practice it is almost always a *ghost*: a handful of sub-second fillers
("Bye.", "Yeah.", "Hmm.") scattered across the meeting, from several people,
that no voiceprint can match.  Left raw it forces an otherwise fully identified
session into manual labeling for nothing.

Instead of guessing an owner, ``label --auto`` renames a ghost bucket to
:data:`CROSSTALK` — the technical truth, readable by a human: these words were
heard but could not be assigned to a speaker.

``CROSSTALK`` is a *reserved* label: it is never a participant, never gets a
voiceprint, and never matches one.  Kept dependency-free (no numpy/torch) so
frontmatter, summary and voiceprint code can import it cheaply.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterable
from typing import Any

CROSSTALK = "CROSSTALK"

# Labels that name no person.  Compared case-insensitively.
RESERVED_LABELS = frozenset({CROSSTALK})

# Only the catch-all remote buckets are ghost candidates.  A raw SPEAKER_n is a
# real pyannote cluster (more likely a person) and keeps the stricter
# tiny-noise rule in millet.label.absorb_tiny_speakers.
_GHOST_CANDIDATE_RE = re.compile(r"^REMOTE(?:_\d+)?$")

# Ghost thresholds — relative to the meeting, not fixed counts, so a 3-hour
# call with 40 scattered "yeah"s is recognized as well as a 20-minute one.
#
# Measured in WORDS, not seconds: Whisper stretches filler timestamps ("Yeah."
# 6.7 s, "No, no, no." 22.6 s), so duration overstates how much was said.
#
# Calibrated by replaying 257 real transcripts:
#   * 49/55 REMOTE buckets a human had left unnamed become CROSSTALK; the
#     rest are 1-segment blips (folded by the tiny-noise rule) or real speech
#     that should reach a human (11.8% share, full German sentences, ...).
#   * 1/120 buckets a human had named a real person would have been labeled
#     CROSSTALK: two greetings ("Hi, Kofi.", 1.9 s).  Harmless.
GHOST_MAX_WORD_SHARE = 0.02             # of the transcript's words
GHOST_MAX_WORDS = 150                   # absolute ceiling regardless of length
GHOST_MAX_MEDIAN_SEGMENT_SECONDS = 1.0  # backchannel shape (robust to outliers)
# A "substantive" segment is a real utterance, not a filler.  A ghost may
# carry a couple (a line leaking from another speaker), never more.
GHOST_SUBSTANTIVE_SEGMENT_WORDS = 6
GHOST_MAX_SUBSTANTIVE_SEGMENTS = 2


def is_reserved_label(name: str | None) -> bool:
    """True if ``name`` is a reserved non-person label (e.g. CROSSTALK)."""
    return name is not None and name.strip().upper() in RESERVED_LABELS


def _field(seg: Any, key: str) -> Any:
    return seg.get(key) if isinstance(seg, dict) else getattr(seg, key, None)


def _duration(seg: Any) -> float:
    try:
        return max(0.0, float(_field(seg, "end")) - float(_field(seg, "start")))
    except (TypeError, ValueError):
        return 0.0


def _words(seg: Any) -> int:
    return len((_field(seg, "text") or "").split())


def is_ghost_speaker(speaker_id: str, segments: Iterable[Any]) -> bool:
    """True if ``speaker_id`` is a ghost REMOTE bucket (filler crosstalk only).

    ``segments`` is the whole transcript (``Segment`` objects or plain dicts
    with ``start``/``end``/``text``/``speaker``).  A ghost must satisfy all of:

    * it is a raw ``REMOTE``/``REMOTE_N`` bucket;
    * its words are at most :data:`GHOST_MAX_WORD_SHARE` of the meeting's
      words (and at most :data:`GHOST_MAX_WORDS`);
    * its median segment is at most :data:`GHOST_MAX_MEDIAN_SEGMENT_SECONDS`;
    * at most :data:`GHOST_MAX_SUBSTANTIVE_SEGMENTS` of its segments are real
      utterances (more than :data:`GHOST_SUBSTANTIVE_SEGMENT_WORDS` words).

    Pure and deterministic.
    """
    if not _GHOST_CANDIDATE_RE.match(speaker_id or ""):
        return False

    total_words = 0
    own: list[Any] = []
    for seg in segments:
        total_words += _words(seg)
        if _field(seg, "speaker") == speaker_id:
            own.append(seg)

    if not own or total_words <= 0:
        return False

    words = [_words(s) for s in own]
    if sum(words) > GHOST_MAX_WORDS:
        return False
    if sum(words) > GHOST_MAX_WORD_SHARE * total_words:
        return False
    if statistics.median(_duration(s) for s in own) > GHOST_MAX_MEDIAN_SEGMENT_SECONDS:
        return False
    substantive = sum(1 for n in words if n > GHOST_SUBSTANTIVE_SEGMENT_WORDS)
    return substantive <= GHOST_MAX_SUBSTANTIVE_SEGMENTS


def ghost_speakers(
    segments: Iterable[Any], resolved_ids: set[str] | None = None
) -> list[str]:
    """Return the ghost bucket ids in ``segments`` (sorted), skipping resolved ones."""
    segs = list(segments)
    resolved = resolved_ids or set()
    ids = sorted({_field(s, "speaker") or "" for s in segs} - resolved - {""})
    return [sid for sid in ids if is_ghost_speaker(sid, segs)]
