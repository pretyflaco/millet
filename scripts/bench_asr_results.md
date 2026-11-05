# ASR Backend Benchmark — Results (Phase 2)

Host: a dev box (RTX 3090, 24 GB). Date: 2026-05-30.
Model: Parakeet TDT 0.6b v2 (English, ONNX via onnx-asr), Silero VAD.
WhisperX baseline: large-v3-turbo.

## Primary file: a recorded meeting (stereo .wav, 1371 s / 22.8 min)

| Config | Backend | Alignment | RTFx | wall (s) | speakers | segments |
|--------|---------|-----------|------|----------|----------|----------|
| A | whisperx | n/a | **54.49** | 25.2 | 2 | 264 |
| B | parakeet | native (skip) | 21.28 | 64.4 | **3** | 562 |
| C | parakeet | whisperx align | 18.51 | 74.0 | 2 | 602 |

## Findings

### Speed
- **WhisperX is FASTER on this GPU (RTFx 54 vs 21)**, opposite of meetily's
  "4x faster" marketing. WhisperX's batched faster-whisper is highly tuned on
  a 3090. The "4x" claim appears to be CPU- or model-specific, not universal.
  Parakeet at RTFx 21 is still ~21x real-time — fast enough for batch use.

### Quality / speaker attribution (the decisive axis)
Same conversational window (~03:12–03:41), two people in rapid Q&A:

- **Config A (whisperx)** collapsed the exchange into ONE 29-second
  `REMOTE` mega-segment, mashing both speakers' turns together —
  the cross-attribution failure mode we set out to test.
- **Config B (parakeet, native timestamps)** correctly separated
  `REMOTE_1` vs `REMOTE_2` turn-by-turn, preserving the Q&A structure and
  finding a 3rd speaker WhisperX missed. Finer VAD segmentation + cleaner
  per-segment audio gave diarization a real chance.
- **Config C (parakeet + whisperx alignment)** had fine segments (602) but
  diarization collapsed back to 2 speakers all labelled `REMOTE` — the
  alignment path LOST the speaker separation that B preserved.

## Implications for the open decisions

- **#2 (alignment default for Parakeet):** prefer **B (skip alignment /
  native timestamps)**. C is both slower AND worse on attribution here.
  Keep `parakeet_skip_alignment=True` as the default.
- **#5 (auto / promote Parakeet):** Parakeet is NOT a speed win on this GPU,
  but IS an attribution-quality win on overlapping speech. Recommendation:
  keep it opt-in for now; consider exposing it as a "high-accuracy /
  multi-speaker" mode rather than the default. Needs confirmation on the
  overlap-heavy .ogg meetings (UXUI, PROSPERA, BLINKBUS) and ideally with
  --mixdown dual before any auto change.

## Second file: a recorded meeting (.ogg, 37 min, overlap-heavy)

| Config | Backend | RTFx | wall (s) | speakers | segments |
|--------|---------|------|----------|----------|----------|
| A | whisperx | 51.12 | 43.0 | 2 | 337 |
| B | parakeet | 28.83 | 76.3 | 2 | 827 |

- Both kept most speech as `REMOTE`. UXUI is a multi-participant screen-share
  where the remote speakers share ONE system-audio channel, so the
  dual-channel YOU/REMOTE labeler cannot separate them regardless of ASR, and
  pyannote did not split them either.
- **Key caveat learned:** Parakeet's attribution advantage is realized when
  speakers are *separable* — i.e. on different channels (the first file's mic-vs-
  system case) or cleanly diarizable. When multiple remote speakers are mixed
  on one channel, finer segmentation alone doesn't fix attribution.

## Third file: a recorded meeting (stereo .wav, 51 min)

A KNOWN misattribution: production transcript attributed Bob's "thanks for
that update" to Alice. Energy analysis confirms the speakers are cleanly
channel-separated — Alice on mic/L (RODE), Bob on system/R (Bluetooth) —
so attribution is fully achievable. Tested a 150 s clip (25:00–27:30) across
a 2x2 matrix (whisperx/parakeet x mono/dual) on CUDA.

| Config | mixdown | RTFx | "thanks for that update" attributed to | Correct? |
|--------|---------|------|----------------------------------------|----------|
| A whisperx | mono | 17.65 | absorbed into Alice/YOU block (lost)   | NO (reproduces bug) |
| B parakeet | mono | 17.45 | absorbed (lost)                        | NO |
| A whisperx | dual | 16.96 | **REMOTE (Bob)**                       | **YES** |
| B parakeet | dual | 13.68 | **REMOTE (Bob)**                       | **YES** |

### DECISIVE FINDING

**The fix is `--mixdown dual`, NOT the ASR backend.** Both whisperx AND
parakeet attribute the phrase correctly in dual mode and both fail in mono
mode. Mono mixdown averages the two channels *before* ASR, destroying the
channel-separation signal that identifies who spoke — no ASR backend can
recover speaker identity from a pre-mixed mono signal.

Root cause in production: `vezir/server/meet_runner.py:build_transcribe_args`
never passes `--mixdown`, so every transcription uses millet's default
`mixdown="mono"`. For channel-separated recordings (local mic vs remote
system audio — the standard vezir setup), `dual` is strictly better for
attribution.

### Revised recommendations

1. **Biggest win is mixdown, not Parakeet.** Consider making `dual` the
   default (or auto-selecting it when the recording is stereo with separable
   channels). This fixes the class of bug independent of ASR choice.
2. **Parakeet remains a secondary lever** — its finer segmentation helps when
   speakers are NOT channel-separable (multiple remotes on one channel), but
   that case isn't resolved by either backend here.
3. Parakeet's value proposition is therefore narrower than first thought:
   not speed (whisperx is faster on GPU), and the headline attribution bug is
   actually a mixdown-mode issue.

## Caveats
- Single meeting so far (the one with a retained .wav). The overlap-heavy
  shortlist (UXUI etc.) is .ogg-only; transcode + rerun pending.
- No hand-labeled ground truth yet, so attribution judgement is qualitative
  (reading transcripts), not a computed metric.
- Diarization quality depends on pyannote + the dual-channel labeler, which
  sit downstream of ASR; B's advantage comes from feeding them better
  segments, not from Parakeet "doing diarization".
