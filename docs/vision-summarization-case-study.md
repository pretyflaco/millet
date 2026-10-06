# Vision Summarization — a case study, n = 1

Created: 2026-09-13
Scope: does sending cue frames to a vision-capable TEE model surface defects
that a transcript-only summary cannot contain?

> **This is a case study, not an evaluation.** One session, four generated
> summaries. It is enough to show that the vision path *can* surface
> screen-only defects, and enough to show that *which* defects it surfaces is
> not stable run to run. It is not enough to support a general quality claim.
> Compare [`tee-summarization-evaluation.md`](tee-summarization-evaluation.md),
> which is a real evaluation: 10 meetings, blind, two independent judges,
> mechanically applied pass/fail bar.
>
> Metrics and defect *categories* only. No transcript or summary content is
> reproduced here.

---

## Why this exists

A narrated screen recording carries one PNG per transcript cue. Until millet
0.20.0 those frames were stored and never used: the iteration plan was written
from the transcript alone, by a model that had never seen the screen. So the
plan could only ever report what the narrator happened to say out loud.

The obvious question is whether showing the model the screen actually buys
anything, or whether a good narrator already says everything that matters.

## Method

**Session** — one narrated screen recording of a mobile wallet build.
3,506 characters of transcript, 31 cue frames (median 182 KB, 6.07 MB raw,
8.09 MB as base64 in the request). Well under the `MAX_FRAMES = 45` cap.

**Generation** — `millet.summarize.summarize()` with
`template="iteration-plan"`, the same entry point production uses. Model
`glm-5-3-flash` in a Tinfoil TEE for every run — the only model on millet's
`VISION_MODELS` allowlist. No fallbacks triggered.

**Runs** — 1 text-only (`frames=None`) and 3 vision (`frames=`31 frames), so
the vision findings can be checked for run-to-run stability rather than
reported from a single lucky sample.

**Comparison** — a finding counts as *screen-only* if it states a fact that is
visible on screen and absent from the transcript (a label, a rendered value, a
layout relationship). Findings restated from narration are excluded.

## Results

### Screen-only findings, by run

| Screen-only finding | text-only | v1 | v2 | v3 |
|---|---|---|---|---|
| Announced version differs from the version string rendered in the settings footer | — | yes | yes | yes |
| Three fee tiers (Slow/Medium/Fast) all render an identical rate | — | yes | yes | yes |
| A purchase action is grouped under a display-preferences section | — | yes | — | yes |
| Address-field placeholder shows a different address format than the one pasted | — | yes | — | yes |
| Toggle title and its subtitle use inconsistent currency terminology | — | — | — | yes |

The text-only run surfaced **none** of them, which is the expected result
rather than a surprising one: none of these facts are in the transcript.

**Two findings reproduced in 3/3 vision runs.** Two more in 2/3, one in 1/3.

### Latency

| run | elapsed |
|---|---|
| text-only | 53.7 s |
| vision v1 | 77.7 s |
| vision v2 | 92.5 s |
| vision v3 | 79.4 s |

Median vision run 79.4 s against a 53.7 s text-only baseline — roughly +50%
for 31 frames. An earlier development run of the same session measured 126.9 s;
enclave latency varies enough between runs that any single timing should be
read as indicative, not as a benchmark.

### Output size

3,246 bytes text-only; 3,393 / 3,668 / 3,678 for the vision runs — between
+5% and +13%. The screen-only findings are not bought by a proportional
increase in length: the vision summaries carry several additional concrete
findings for roughly a tenth more text, largely by displacing bullets that
restate the narration.

## What this does and does not establish

**Supported by this data:**

1. The vision path surfaces defects that are absent from the transcript, and
   the text-only path cannot surface them at all. For a screen recording,
   that gap is structural, not a quality difference.
2. Two findings were stable across all three vision runs, so the effect is not
   a single fortunate sample.
3. The cost is a ~50% latency increase and an 8 MB request, with no meaningful
   growth in output length.

**Not supported by this data:**

1. **Any general claim.** n = 1 session, one app, one narrator, English only.
2. **That the finding set is reliable.** Three of five screen-only findings
   appeared in some runs and not others. A user who runs this once gets a
   sample of the model's attention, not an exhaustive defect list. If a
   comprehensive list ever matters, that needs multiple runs or a different
   design.
3. **Severity calibration.** The same defect was labelled `high` in one run
   and `medium` in another. Severities here are suggestions.
4. **That every screen-only finding is correct.** These were checked for
   plausibility against the frames, not adjudicated by an independent judge as
   in the TEE evaluation. A confident-sounding claim about a rendered value
   can still be wrong.

### A correction worth recording

An earlier, undocumented development run of this session was described
internally as having caught a destination field accepting a mainnet address
while the wallet was on regtest — a funds-loss bug. **That did not reproduce
in any of the three runs here**, and the closest stable finding is far more
mundane: the address-field *placeholder* advertises one address format while
another is pasted, which is a hint-text nit, not a funds-loss bug.

The original claim was recorded from memory rather than from a retained
artifact. It is exactly the sort of overstatement an n=1 anecdote invites, and
the reason this document re-ran the session instead of writing down what was
remembered.

## Operational findings

Two things surfaced while building this path that are not about summary
quality:

1. **The vendor's `multimodal` flag cannot be trusted.** Tinfoil's model
   catalog advertises `multimodal: true` for `deepseek-v4-1-flash`, but its
   vision endpoint returned 502 on every attempt. Since that model is
   millet's sibling fallback, believing the catalog would have converted a
   drained enclave pool into a hard failure. `VISION_MODELS` is therefore a
   hand-maintained allowlist (`millet/summarize.py:143`), not a catalog read.
   Frames are dropped with a warning for any model not on it, so a fallback
   degrades a vision run to text-only instead of failing it.
   *Update 2026-10-06:* DeepSeek's vision endpoint now works (10 synthetic
   880×1920 frames, every on-screen code read correctly, 2/2 runs), and it
   was allowlisted as the new default in millet 0.21.6. It bills ~10k image
   tokens where `glm-5-3-flash` billed ~22k for the same frames, so it
   likely downscales harder; this case study has not been re-run on it.
2. **Enclave attestation fails intermittently.** Malformed SEV attestation
   reports were observed at roughly 25% (9 of 12 plain-text attempts
   succeeded). These fail *before* generation, so they are cheap to retry, and
   they are given their own retry budget
   (`_TINFOIL_ATTEST_MAX_ATTEMPTS = 6`) separate from the network-error budget
   (`_TINFOIL_MAX_ATTEMPTS = 3`). Retrying never accepts an unverified
   response; exhausting the budget raises an error that says so.

## If this were to become an evaluation

The honest next step, in rough order of value:

1. **More sessions** — 8–10 screen recordings across different apps and
   narrators. One app's UI quirks are not a sample.
2. **A per-session answer key**, built by a judge reading *the frames* and
   enumerating the visible defects a correct plan must contain. That converts
   "found interesting things" into recall against a fixed denominator, the
   same method the TEE evaluation used.
3. **Multiple runs per session**, since this study already shows the finding
   set is unstable — a single run per condition would measure sampling noise.
4. **Precision scoring**, to catch confident claims about rendered values that
   are simply wrong. This study did not measure it at all.

## Reproducing

```python
from millet.summarize import SummaryConfig, summarize, discover_cue_frames

frames = discover_cue_frames(session_dir)          # cue_HH-MM-SS.png
cfg    = SummaryConfig(template="iteration-plan", frames=frames)
result = summarize(transcript_text, cfg, language="en")
print(result.frames_used)                          # 0 if the model lacks vision
```

CLI equivalent: `millet transcribe … --summary-template iteration-plan
--summary-frames`. Vezir runs this automatically for `video` sessions using
the `iteration-plan` template — deferring the summary until after frame
extraction, since the frames are sampled from transcript timestamps and do not
exist while `millet transcribe` is running.
