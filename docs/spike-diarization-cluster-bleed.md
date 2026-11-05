# Spike: cross-speaker diarization cluster bleed (issue #2)

Status: **investigation** (2026-06-18). A minimal, low-risk guard is implemented
and unit-tested (`_isolated_backchannel_outliers` in `millet/transcribe.py`),
gated by `TranscriptionConfig.strip_isolated_backchannel` (default **on**, very
conservative thresholds). This doc records the analysis and the decision space.

## Symptom

In a dev standup, Bob joined ~27:32, yet the
transcript attributed early backchannel to Bob:

```
Bob 00:10  Thank you. Yeah. Yeah.
Bob 00:53  OK.
Bob 00:58  I had something.
```

## Root cause

pyannote diarization **under-segmented**: it placed 5 early sub-1.5 s
backchannel utterances (from whoever was present at the start) into the *same*
cluster (`SPEAKER_02`) that later held Bob's real speech. Voiceprint matching
then named the whole cluster "Bob" (0.92, dominated by Bob's 84 real segments),
so the 5 early fragments inherited the wrong name.

This is upstream of voiceprint matching, which only ever sees/labels **whole
clusters**. It is the *opposite* of the over-segmentation in issue #1.

### Measured signature (this session)

| Property | Early outliers (5) | Bob main mass (84) |
|---|---|---|
| Time span | 11–59 s | 1588 s (26.5 min) → 64.7 min |
| Gap to main mass | **1529 s (25.5 min)** | — |
| Segment duration | all **< 1.5 s** (0.22–0.52 s) | median 2.36 s, max 26.8 s |
| Content | "Thank you", "Yeah", "OK" | substantive turns |

The outliers are: (a) all **below `MIN_SEGMENT_DURATION` (1.5 s)** — i.e. already
excluded from the cluster's *embedding* (so the match itself is unaffected);
(b) separated from the cluster's main time-mass by a **very large gap**.

## Options considered

- **B1 (chosen, minimal):** detect segments that are both sub-threshold AND
  temporally isolated from their cluster's main mass by a large gap, and strip
  them out of the named cluster (reassign to `REMOTE`). The A1 REMOTE-rescue
  then re-absorbs them into the speaker they actually overlap, or they surface
  as a tiny unknown. Low risk because it only touches sub-1.5 s fragments that
  contribute no embedding evidence and are far from the cluster's real speech.
- **B2 (rejected for now):** pass pyannote clustering/min-duration knobs to bias
  against merges. Uncertain payoff; affects all sessions; needs broad eval.
- **B3 (rejected):** per-segment re-embed + nearest-cluster reassign for every
  segment. Most accurate but expensive and a much larger change; revisit if B1
  proves insufficient.

## B1 thresholds (conservative)

- `BACKCHANNEL_MAX_SECONDS = 1.5` — only fragments below the embedding floor.
- `CLUSTER_ISOLATION_GAP_SECONDS = 300.0` (5 min) — the fragment must be ≥ 5 min
  from the *nearest* same-cluster segment that is itself ≥ `MIN_SEGMENT_DURATION`
  (the cluster's "real" speech). 5 min is far beyond normal turn spacing, so a
  genuine quick "yeah" near one's own speech is never stripped.
- Only fires when the cluster HAS a real main mass (≥ 1 embeddable segment).

On this session B1 reassigns exactly the 5 outliers to `REMOTE`; A1 then absorbs
them onto the temporally-overlapping early speaker, and Bob's cluster contains
only her real (≥26.5 min) speech.

## Limitations / follow-ups

- B1 only catches the *isolated sub-threshold* signature. A mis-clustered
  *substantive* segment (≥1.5 s) is left alone (it carries embedding weight; a
  wrong strip would be worse than the current behavior). Those remain a known
  diarization-quality limitation.
- True fix is better diarization (B2/B3) or per-segment voiceprint correction in
  the dual-diarize path (already on the deferred roadmap).
