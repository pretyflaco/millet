"""Per-segment speaker repair for the mono diarization path.

pyannote sometimes puts two people into one cluster.  Incident 2026-11-02: in
a 58 s phone interview with quick question/answer turns it produced one turn
for the first 43 s, and the interviewee's answers all landed in the
interviewer's cluster.  The two voices were easy to tell apart (centroid
cosine 0.17 between the two halves); pyannote just never cut the turn.

This pass embeds every ASR segment on its own and asks each diarized cluster
one question: does it hold two clearly different voices?  It runs one
average-linkage 2-way split per cluster and accepts it only when both halves
are coherent and far apart.  The smaller half then either joins an existing
cluster it clearly belongs to, or becomes a new ``SPEAKER_NN`` for a human to
name.  A wrong merge cannot be fixed by labeling; an extra speaker can.

Replayed on saray (2026-11-03) over the 100 diarized mono sessions there, with
the raw ids rebuilt from word-level speakers:

* the incident session: 9 segments (8.3 s) move from the interviewer to the
  interviewee's one-segment cluster (split cosine 0.17, join cosine 0.62);
* 56 single-speaker sessions: no move;
* 15 group meetings (10-100 min): 11 moves of 4-13 segments into an existing
  speaker (join cosine 0.61 to 0.82) and 8 new speakers of 4-7 segments.
  These have not been checked by ear.

Kept torch-free at import time: the planner is pure numpy so it can be unit
tested without the model stack.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

# Segments shorter than this are not embedded (and keep their label).  Short
# answers ("I'm doing well.", 0.7 s) carry the second voice in a quick
# interview, so this is lower than voiceprint's 1.5 s learning floor.
REPAIR_MIN_SEGMENT_SECONDS = 0.5
# Same silence floor as voiceprint embedding (millet.voiceprint.MIN_SEGMENT_RMS).
REPAIR_MIN_SEGMENT_RMS = 0.0015
# A cluster needs this many embedded segments before a split is considered.
REPAIR_MIN_CLUSTER_SEGMENTS = 6
# The smaller half must be at least this big.  Rules out single outliers
# (a cough, a hallucinated line) and short noise runs.
REPAIR_MIN_PART_SEGMENTS = 3
REPAIR_MIN_PART_SECONDS = 4.0
# The two halves' centroids must be at most this similar.  One person across
# a long recording stays well above it; the incident halves were at 0.17.  A
# single-speaker kids' game recording split at 0.46.
REPAIR_MAX_CENTROID_COS = 0.35
# Mean cosine of the smaller half's segments to their own centroid.
REPAIR_MIN_PART_COHESION = 0.5
# The smaller half joins an existing cluster whose centroid it matches at
# least this well.  That match is evidence of its own.
REPAIR_JOIN_MIN_COS = 0.6
# Inventing a new speaker needs more evidence than moving segments to one who
# is already there: more speech and a wider gap.  On the saray replay this
# halved the new speakers (16 -> 8), all in 30-100 min group meetings.
REPAIR_NEW_SPEAKER_MIN_SECONDS = 8.0
REPAIR_NEW_SPEAKER_MAX_CENTROID_COS = 0.30
# Above this many embedded segments, the split is computed on an evenly spaced
# subsample and the rest are assigned to the nearer half.  Keeps the O(n^2)
# linkage bounded for long meetings.
REPAIR_MAX_POINTS = 500

_SPEAKER_NUM_RE = re.compile(r"^SPEAKER_(\d+)$")


@dataclass(frozen=True)
class RepairItem:
    """One embedded segment: its index in the transcript, label and voice."""

    index: int
    speaker: str
    duration: float
    embedding: np.ndarray  # L2-normalized


@dataclass(frozen=True)
class RepairMove:
    """Segments moved out of ``source`` into ``target``."""

    source: str
    target: str
    indices: tuple[int, ...]
    seconds: float
    split_cos: float
    join_cos: float | None  # None when ``target`` is a new speaker

    @property
    def new_speaker(self) -> bool:
        return self.join_cos is None


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def average_linkage_bipartition(emb: np.ndarray) -> np.ndarray:
    """Split L2-normalized rows into two groups by average linkage (cosine).

    Equivalent to ``scipy.cluster.hierarchy.linkage(emb, "average",
    metric="cosine")`` cut at two clusters, without depending on scipy (CI
    and the label path only have numpy).  Returns a 0/1 label per row.
    Requires at least two rows.  Deterministic.
    """
    n = len(emb)
    if n < 2:
        raise ValueError("need at least two rows to split")
    sim = (emb @ emb.T).astype(np.float64)
    np.fill_diagonal(sim, -np.inf)
    size = np.ones(n)
    root = np.arange(n)
    for _ in range(n - 2):
        i, j = np.unravel_index(int(np.argmax(sim)), sim.shape)
        merged = (size[i] * sim[i] + size[j] * sim[j]) / (size[i] + size[j])
        sim[i, :] = merged
        sim[:, i] = merged
        sim[i, i] = -np.inf
        sim[j, :] = -np.inf
        sim[:, j] = -np.inf
        size[i] += size[j]
        root[root == j] = i
    first = root[0]
    return (root != first).astype(int)


def _split_cluster(items: Sequence[RepairItem]) -> tuple[np.ndarray, float] | None:
    """Labels (0 = major, 1 = minor) and centroid cosine of an accepted split."""
    emb = np.stack([it.embedding for it in items])
    dur = np.array([it.duration for it in items])
    if len(items) > REPAIR_MAX_POINTS:
        pick = np.linspace(0, len(items) - 1, REPAIR_MAX_POINTS).round().astype(int)
        sub = average_linkage_bipartition(emb[pick])
        c = [_unit(emb[pick][sub == k].mean(axis=0)) for k in (0, 1)]
        labels = (emb @ c[1] > emb @ c[0]).astype(int)
    else:
        labels = average_linkage_bipartition(emb)

    if labels.min() == labels.max():
        return None
    # Minor = the half with less speech.
    if dur[labels == 1].sum() > dur[labels == 0].sum():
        labels = 1 - labels
    minor = labels == 1
    if minor.sum() < REPAIR_MIN_PART_SEGMENTS or dur[minor].sum() < REPAIR_MIN_PART_SECONDS:
        return None
    c_major = _unit(emb[~minor].mean(axis=0))
    c_minor = _unit(emb[minor].mean(axis=0))
    split_cos = float(c_major @ c_minor)
    if split_cos > REPAIR_MAX_CENTROID_COS:
        return None
    if float(np.mean(emb[minor] @ c_minor)) < REPAIR_MIN_PART_COHESION:
        return None
    return labels, split_cos


def _next_speaker_ids(existing: Iterable[str]) -> Callable[[], str]:
    taken = set(existing)
    nums = [int(m.group(1)) for s in taken if (m := _SPEAKER_NUM_RE.match(s))]
    counter = [max(nums) + 1 if nums else 0]

    def take() -> str:
        while True:
            sid = f"SPEAKER_{counter[0]:02d}"
            counter[0] += 1
            if sid not in taken:
                taken.add(sid)
                return sid

    return take


def plan_speaker_repair(
    items: Sequence[RepairItem], existing_speakers: Iterable[str] = ()
) -> list[RepairMove]:
    """Decide which segments move to which speaker.  Pure and deterministic.

    ``items`` are the embedded segments of the whole transcript.
    ``existing_speakers`` are all speaker ids in the transcript (including
    those with no embedded segment), so a new id never collides.
    """
    by_speaker: dict[str, list[RepairItem]] = {}
    for it in items:
        by_speaker.setdefault(it.speaker, []).append(it)

    # Centroids from diarization's own assignment, before any move.
    centroids = {
        sid: _unit(np.mean([it.embedding for it in its], axis=0))
        for sid, its in by_speaker.items()
    }
    new_id = _next_speaker_ids(set(existing_speakers) | set(by_speaker))

    moves: list[RepairMove] = []
    for sid in sorted(by_speaker):
        its = by_speaker[sid]
        if len(its) < REPAIR_MIN_CLUSTER_SEGMENTS:
            continue
        split = _split_cluster(its)
        if split is None:
            continue
        labels, split_cos = split
        minor = [it for it, lab in zip(its, labels, strict=True) if lab == 1]
        c_minor = _unit(np.mean([it.embedding for it in minor], axis=0))

        best_sid, best_cos = None, -1.0
        for other, c in centroids.items():
            if other == sid:
                continue
            cos = float(c_minor @ c)
            if cos > best_cos:
                best_sid, best_cos = other, cos

        seconds = float(sum(it.duration for it in minor))
        if best_sid is not None and best_cos >= REPAIR_JOIN_MIN_COS:
            target, join_cos = best_sid, best_cos
        elif (seconds >= REPAIR_NEW_SPEAKER_MIN_SECONDS
              and split_cos <= REPAIR_NEW_SPEAKER_MAX_CENTROID_COS):
            target, join_cos = new_id(), None
        else:
            continue
        moves.append(RepairMove(
            source=sid,
            target=target,
            indices=tuple(it.index for it in minor),
            seconds=seconds,
            split_cos=split_cos,
            join_cos=join_cos,
        ))
    return moves


def apply_moves(segments: list, moves: Sequence[RepairMove]) -> None:
    """Relabel the moved segments (and their words) in place."""
    for mv in moves:
        for i in mv.indices:
            seg = segments[i]
            seg.speaker = mv.target
            for w in seg.words or []:
                if isinstance(w, dict) and "speaker" in w:
                    w["speaker"] = mv.target


def describe_move(mv: RepairMove) -> str:
    where = (
        f"new speaker {mv.target}" if mv.new_speaker
        else f"{mv.target} (cos {mv.join_cos:.2f})"
    )
    return (
        f"{len(mv.indices)} segment(s), {mv.seconds:.1f}s from {mv.source} -> "
        f"{where}; voices differ (cos {mv.split_cos:.2f})"
    )


def embed_segments(
    audio: np.ndarray,
    segments: Sequence,
    device: str | None = None,
    sample_rate: int = 16000,
) -> list[RepairItem]:
    """Embed every labeled segment long and loud enough (needs torch/pyannote)."""
    import torch

    from millet import voiceprint as _vp

    inference = _vp._get_inference()
    if device:
        try:
            inference.to(torch.device(device))
        except Exception:
            pass  # stay on whatever device it is on

    items: list[RepairItem] = []
    for i, seg in enumerate(segments):
        if not seg.speaker:
            continue
        dur = float(seg.end) - float(seg.start)
        if dur < REPAIR_MIN_SEGMENT_SECONDS:
            continue
        clip = audio[int(seg.start * sample_rate):int(seg.end * sample_rate)]
        if len(clip) < int(REPAIR_MIN_SEGMENT_SECONDS * sample_rate):
            continue
        if float(np.sqrt(np.mean(clip.astype(np.float64) ** 2))) < REPAIR_MIN_SEGMENT_RMS:
            continue
        waveform = torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32)).unsqueeze(0)
        vec = np.asarray(
            inference({"waveform": waveform, "sample_rate": sample_rate})
        ).ravel().astype(np.float32)
        if vec.size == 0 or not np.all(np.isfinite(vec)):
            continue
        items.append(RepairItem(i, seg.speaker, dur, _unit(vec)))
    return items


def repair_speakers(
    audio: np.ndarray,
    segments: list,
    device: str | None = None,
    log: Callable[[str], None] = print,
) -> list[RepairMove]:
    """Embed, plan and apply.  Never raises: a failure leaves ``segments`` as is."""
    try:
        items = embed_segments(audio, segments, device=device)
        moves = plan_speaker_repair(
            items, existing_speakers={s.speaker for s in segments if s.speaker}
        )
    except Exception as exc:
        log(f"  Warning: speaker repair skipped ({exc})")
        return []
    if moves:
        apply_moves(segments, moves)
        log("  Speaker repair (one cluster held two voices):")
        for mv in moves:
            log(f"    {describe_move(mv)}")
    return moves
