# millet

[![CI](https://github.com/pretyflaco/millet/actions/workflows/ci.yml/badge.svg)](https://github.com/pretyflaco/millet/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/millet-pipeline.svg)](https://pypi.org/project/millet-pipeline/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/millet-pipeline?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/millet-pipeline)

> **Millet** is a meeting transcription, summarization, and PDF output
> tool.  It's named after the Ottoman *millet system* — the legal
> framework of communal autonomy that, in 1493, made it possible for
> two Sephardic Jewish brothers to establish Istanbul's first printing
> press, just one year after their expulsion from Spain.  Like the
> millets it's named after, this tool operates under its own rules,
> on your own machine, within the broader
> [vezir](https://github.com/pretyflaco/vezir) ecosystem.

Formerly known as **meetscribe**.  PyPI distribution name:
`millet-pipeline` (the bare `millet` slot on PyPI is held by an
unrelated dormant 2021 package).  See
[CHANGELOG.md](CHANGELOG.md) for the rename details and full release
history.

Meeting transcription with speaker diarization, AI-generated summaries,
and professional PDF output.

Records dual-channel audio (your mic + system audio) from **any** meeting
app and produces diarized transcripts using WhisperX + pyannote-audio.
Works fully offline with local models, or summarizes inside a
hardware-attested Tinfoil TEE where the prompts never leave the secure
enclave.  **Every supported summary backend is private** — as of 0.19.0
the cloud backends that could read your meetings (Claude Max, OpenRouter,
generic OpenAI endpoints) are gone, because a TEE model measurably
out-summarized Sonnet 4.6 on grounded precision and recall across EN/DE/TR
(see [the evaluation](docs/tee-summarization-evaluation.md)).  There is no
longer a privacy/quality tradeoff to configure.

## Status

Alpha (**0.21.6**).  Full history in [CHANGELOG.md](CHANGELOG.md).

Recent highlights:

- **0.21.6** -- TEE default moved to `deepseek-v4-1-flash` (Tinfoil
  deprecated `glm-5-3-flash` on 2026-10-09); sibling fallback is now
  `glm-5-3`.  DeepSeek was the runner-up in the TEE evaluation and its
  vision endpoint was re-verified before it took over screen recordings.
- **0.20.1** -- the default backend's SDK (`tinfoil`) is a base
  dependency, not an optional extra.  A plain `pip install
  millet-pipeline` can now summarize on its own default path.
- **0.20.0** -- vision summarization: cue frames from a screen recording
  are sent to the model with the transcript.
- **0.19.0** -- the cloud summary backends were removed; every remaining
  backend is private.  PDF gained a quiet attestation footer in place of
  the blanket CONFIDENTIAL watermark.
- **0.18.1** -- TEE default moved to `glm-5-3-flash` after the vendor
  retired `glm-5-2`, with a sibling-model retry for a drained enclave.

## Evaluations

Two write-ups, deliberately labelled by how much they actually support:

| Question | Evidence | Result |
|---|---|---|
| Can an in-TEE model replace Sonnet 4.6 as the default summarizer? | **Evaluation** -- 10 meetings (EN/DE/TR), blind relabeling, two independent in-TEE judges, 80 verdicts, mechanical 16-check bar | **Yes.** `glm-5-3-flash` beat Sonnet on precision *and* recall under both judges in every language; 16/16. Runner-up `deepseek-v4-1-flash` (the default since 0.21.6, after Tinfoil deprecated GLM-5.3 Flash) matched its precision at ~4 pp lower recall; 15/16. [Full evaluation](docs/tee-summarization-evaluation.md) |
| Does showing the model the screen surface defects the transcript cannot? | **Case study, n = 1** -- one session, 1 text-only + 3 vision runs | Screen-only defects appear that the text-only run surfaced none of; 2 of 5 reproduced in 3/3 runs. Not a general claim. [Case study](docs/vision-summarization-case-study.md) |

The first is worth quoting; the second is worth reading before believing.
Note what the evaluation found about *method*: all four models scored
10/10 on format and language compliance, so structural metrics were a
four-way tie and every real difference was in content.

## Works with any meeting app

Because millet captures system audio at the OS level, it works with
every voice/video call application:

- **Zoom**
- **Google Meet**
- **Microsoft Teams**
- **Slack** (huddles and calls)
- **Discord**
- **Signal** (voice and video calls)
- **Telegram** (voice and video calls)
- **WhatsApp** (desktop voice and video calls)
- **Keet** (P2P calls)
- **Jitsi Meet**
- **Webex**
- **Skype**
- **FaceTime** (via browser)
- **GoTo Meeting**
- **RingCentral**
- **Amazon Chime**
- **BlueJeans**

Any app that plays audio through your system speakers will work --
including browser-based meetings and standalone desktop clients.

## Features

- **Dual-channel audio capture** -- records your mic (left channel) and remote
  participants (right channel) simultaneously via PipeWire/PulseAudio + ffmpeg
- **WhisperX transcription** -- fast batched inference with
  `openai/whisper-large-v3-turbo`, word-level timestamps via wav2vec2 alignment
- **Multilingual** -- auto-detects language or manually set it; supports
  English, German, Turkish, French, Spanish, Farsi, and 90+ other languages
- **Speaker diarization** -- pyannote-audio identifies who said what, with
  automatic YOU/REMOTE labeling from the dual-channel signal
- **AI meeting summaries, private by default** -- a hardware-attested
  Tinfoil TEE (`deepseek-v4-1-flash`, the default) or a fully local Ollama
  model.  Both are private, so the `tinfoil -> ollama` fallback chain can
  degrade *quality* but never *confidentiality*
- **Summarize from the screen** -- for a narrated screen recording, the
  cue frames sampled from transcript timestamps are sent to the model
  alongside the transcript, so an iteration plan can report what is
  visibly wrong and not only what the narrator said aloud
  (`--summary-template iteration-plan --summary-frames`)
- **Attestation footer on the PDF** -- a TEE-backed summary carries a
  quiet "Summarized in a hardware-attested TEE" footer recording the
  backend and model
- **Voiceprint speaker recognition** -- automatically identifies speakers
  across meetings using voice embedding profiles
- **Meeting sync** -- push transcripts and summaries to any Git repository
  on a configurable schedule
- **Professional PDF output** -- summary + full transcript in a clean,
  page-numbered PDF with full Unicode support (DejaVu Sans) and RTL for Farsi
- **Multiple output formats** -- `.txt`, `.srt`, `.json`, `.summary.md`, `.pdf`
- **Structured YAML frontmatter** -- every `.summary.md` carries a typed
  schema (action items, decisions, participants, topics, language,
  duration) plus a matching `.frontmatter.json` sidecar, ready for
  indexers and downstream tooling like
  [vezir](https://github.com/pretyflaco/vezir)
- **GTK3 GUI widget** -- small always-on-top window with record/stop, timer,
  and one-click access to results
- **CLI** -- `millet record`, `millet transcribe`, `millet run`, `millet gui`,
  `millet label`, `millet enroll`, `millet sync`, `millet ingest`,
  `millet download`, `millet translate`, `millet devices`, `millet check`
- **Per-session folders** -- each recording gets its own organized directory
- **Offline-first** -- after initial model download, transcription,
  diarization and PDF output work without internet; summarization needs
  either the TEE (network, private) or a local Ollama model (no network)

## Quick start

```bash
# Install from PyPI
pip install millet-pipeline

# Set your HuggingFace token (required for speaker diarization)
export HF_TOKEN=hf_your_token_here

# Record a meeting, then auto-transcribe + summarize when you stop
millet run
# Press Ctrl+C when the meeting ends
```

## Requirements

millet runs in two configurations:

**Linux desktop** (full pipeline: record + transcribe + label + sync)

- Linux with PipeWire or PulseAudio (for system-audio capture)
- NVIDIA GPU with CUDA (8 GB+ VRAM recommended; CPU mode available but slower)
- Python 3.10+, ffmpeg
- HuggingFace token (free) for the diarization model
- Ollama (optional) for local AI summaries

**macOS Apple Silicon** (post-capture pipeline: transcribe + label + sync)

- M1 / M2 / M3 Mac running macOS
- Python 3.10+, ffmpeg
- `pip install 'millet-pipeline[mlx]'` to auto-select MLX Whisper for ASR
- HuggingFace token, Ollama as above
- Note: `millet record` / `millet run` (audio capture) require Linux. On a Mac,
  feed in audio captured elsewhere via `millet transcribe <file.wav>`, or use
  [vezir](https://github.com/pretyflaco/vezir) to run a Mac as a server with
  Linux/Android thin clients providing the recordings.

See [REQUIREMENTS.md](REQUIREMENTS.md) for full hardware/software details.

## Installation

### 1. System dependencies

```bash
# Ubuntu / Pop!_OS / Debian
sudo apt install ffmpeg pulseaudio-utils

# Fedora
sudo dnf install ffmpeg pulseaudio-utils
```

### 2. Install millet

```bash
# From PyPI (recommended)
pip install millet-pipeline

# From source
git clone https://github.com/pretyflaco/millet
cd millet
pip install -e .
```

This creates the `millet` command in your PATH (the older `meet` command is
kept as a deprecated alias).

The `tinfoil` SDK behind the default TEE backend is a **base dependency**
since 0.20.1 — no extra to remember.  Set `TINFOIL_API_KEY` to summarize in
the enclave, or run fully locally with `--summary-backend ollama` and no key
at all.  (`millet-pipeline[tee]` still resolves, as an empty no-op extra, so
older install commands don't break.)

### 3. HuggingFace token (for speaker diarization)

1. Create a free account at https://huggingface.co
2. Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-community-1
3. Create a read token at https://huggingface.co/settings/tokens
4. Set it:

```bash
export HF_TOKEN=hf_your_token_here
# Add to ~/.bashrc for persistence:
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc
```

### 4. Ollama (optional, for AI summaries)

Install from https://ollama.com, then pull the default summary model:

```bash
ollama pull qwen3.5:9b
```

### 5. Verify setup

```bash
millet check
```

## Usage

### Check audio devices

```bash
millet devices
```

### Record a meeting

Start recording before or during your meeting:

```bash
millet record
```

Press Ctrl+C when the meeting ends. A 10-second drain buffer ensures all audio
is captured. Recordings are saved to `~/meet-recordings/`.

Options:
- `-o /path` -- save recordings elsewhere
- `--virtual-sink` -- create isolated virtual sink (avoids capturing notification sounds)
- `--mic <source>` -- specify mic source (use `millet devices` to find names)
- `--monitor <source>` -- specify monitor source

### Transcribe a recording

```bash
millet transcribe ~/meet-recordings/meeting-20260312-140000/meeting-20260312-140000.wav
```

Options:
- `-m large-v3-turbo` -- Whisper model (default: `large-v3-turbo`; also: `base`, `medium`, `large-v2`)
- `-l auto` -- language code or `auto` to auto-detect (default: `auto`; e.g. `en`, `de`, `tr`, `fa`)
- `--asr-backend auto` -- ASR backend: `auto`, `whisperx`, `mlx`, or
  `parakeet`. On Apple Silicon with `mlx-whisper` installed, `auto` uses MLX
  Whisper for ASR. MLX only replaces the transcription step; millet still
  requires WhisperX for audio loading, alignment, and diarization. `parakeet`
  uses NVIDIA Parakeet ONNX (English; install with `pip install
  'millet-pipeline[parakeet]'`); see `--parakeet-model` /
  `--parakeet-keep-alignment`.
- `--mlx-model <repo-or-path>` -- MLX Whisper model path/repo (default: maps
  `large-v3-turbo` to `mlx-community/whisper-large-v3-turbo`)
- `--device cuda` -- `cuda` or `cpu`. Default: auto-detected — `cpu` on
  Apple Silicon (since macOS has no CUDA), `cuda` elsewhere.
- `--torch-device mps` -- optional PyTorch device for alignment/diarization;
  useful with MLX ASR or CPU ASR on Apple Silicon.
- `--compute-type float16` -- `float16` or `int8` for lower VRAM (default: `float16`)
- `-b 16` -- batch size, reduce if running low on VRAM (default: `16`)
- `--min-speakers 2` / `--max-speakers 6` -- hint for number of speakers
- `--no-diarize` -- skip speaker diarization
- `--no-summarize` -- skip AI summary generation
- `--summary-backend ollama` -- summary backend (`tinfoil` default, or `ollama`)
- `--summary-model <model>` -- model for summary (default: per-backend)
- `--summary-template <name>` -- use a named summary template instead of the
  default meeting summary (e.g. `iteration-plan` for a narrated walkthrough
  of a build)
- `--summary-frames` / `--no-summary-frames` -- send the session's cue frames
  (`attachments/cue_HH-MM-SS.png`) to the model alongside the transcript, so
  the summary can describe what is on screen.  Requires a vision-capable
  model: `deepseek-v4-1-flash` (the default) and `glm-5-3-flash` are on the
  allowlist, and frames are dropped with a warning for anything else (up to
  `MAX_FRAMES = 10`, the endpoint's per-request image cap)
- `--skip-alignment` -- skip word-level alignment (useful if alignment model is unavailable)
- `--mixdown mono|dual|dual-diarize` -- stereo mixdown mode (default:
  `dual-diarize`). See *Dual-channel modes* below.
- `--default-language <code>` -- team/operator default language that biases
  low-confidence auto-detection (e.g. `en`); pairs with
  `--language-detection-segments N` (default `6`).
- Dual-diarize tuning: `--channel-correct/--no-channel-correct` (+
  `--channel-correct-margin`, default `0.30`) for mic-bleed/echo correction,
  `--consolidate-remote-clusters/--no-...` to merge phantom remote speakers,
  and `--single-source-fallback/--no-...` for in-room (non-headphone)
  recordings detected as single-source.

#### Dual-channel modes

Stereo recordings carry your mic on the left channel and system audio on the
right. `transcribe` defaults to **`--mixdown dual-diarize`**, which
transcribes each channel independently for best accuracy and then diarizes
the remote (system) channel to separate multiple remote participants. It
labels the local speaker as YOU (mic) and diarized remotes as REMOTE_1,
REMOTE_2, … The three modes:

- `dual-diarize` (default) -- per-channel transcription + diarization of
  remotes. Best accuracy for headphone setups where mic and system audio
  don't bleed into each other.
- `dual` -- per-channel transcription with no remote diarization (channel
  identity = speaker identity: YOU vs. REMOTE).
- `mono` (legacy) -- mix both channels to mono, then diarize. Use it when
  your speakers play into the room and both voices appear on both channels.

```bash
millet transcribe --mixdown dual-diarize ~/meet-recordings/meeting-20260312-140000/
```

Note: `millet run` still defaults to `--mixdown mono` and accepts `mono` or
`dual`; pass `--mixdown dual` there for headphone recordings.

### Record + transcribe in one shot

```bash
millet run
```

Records until Ctrl+C, then automatically transcribes, generates a summary,
and produces a PDF. Takes options from both `record` and `transcribe`. Note
`run`'s `--mixdown` defaults to `mono` and accepts only `mono` or `dual`
(unlike `transcribe`, which defaults to `dual-diarize`); use `millet
transcribe` on the saved recording if you need `dual-diarize`.

### Launch the GUI widget

```bash
millet gui
```

A small always-on-top window with:
- Record / Stop button
- Live timer and file size
- Status indicator (Recording, Flushing, Transcribing, Summarizing, Done)
- "Open PDF" and "Open Folder" buttons after completion

When 2 or more speakers are detected, a **speaker labeling dialog** appears
before the results are saved. Each speaker is shown with their channel and a
sample line of text. If voice profiles exist, confident matches are shown
automatically. Enter a real name or leave blank to keep the auto-assigned
label (YOU, REMOTE_1, etc.).

If meeting sync is configured and the recording matches a scheduled meeting,
a **sync confirmation prompt** appears with Push / Skip buttons.

![millet GUI](screenshot.png)

### Label speakers after the fact

```bash
millet label ~/meet-recordings/meeting-20260313-214133
```

For each speaker in the recording, `millet label`:
1. Shows a table of all speakers (label, channel, segment count, sample text)
2. Plays a short audio clip from that speaker's channel (requires `ffplay`)
3. Prompts you to enter a real name (press Enter to keep the existing label)
4. Regenerates all outputs (`.txt`, `.srt`, `.json`, `.summary.md`, `.pdf`) with the new names

With `--auto`, voice profiles are used to automatically identify known speakers.
Confident matches are applied without prompting; only unrecognized speakers get
the interactive prompt:

```bash
millet label --auto ~/meet-recordings/meeting-20260313-214133
```

Options:
- `--auto` -- auto-label using voice profiles (see [Voiceprint speaker recognition](#voiceprint-speaker-recognition))
- `--no-audio` -- skip audio playback, just show text samples
- `--no-summary` -- use find-and-replace instead of re-running the summarizer
- `--summary-backend` / `--summary-model` -- override summary backend and model for regeneration
- `--summary-template <name>` -- regenerate using a named template
- `--apply-json FILE` (or `-` for stdin) -- non-interactive labeling: apply a
  `{"OLD_ID": "Name", ...}` map (or `{"labels": {...}}` envelope) and exit.
  An **empty** map re-runs just the summary + PDF step, which is how vezir
  produces a screen recording's summary after cue frames exist
- `--update-profiles` -- with `--apply-json`, update voiceprint profiles from
  the confirmed labels
- `--summary-language <code>` -- also emit a translated `.summary.<lang>.md`
- `--team <name>` -- use a team-scoped profile database (see below)

When run without a TTY (e.g. from a worker), `--auto` applies confident
matches and skips the interactive prompt automatically, writing an
`.autoid.json` sidecar with per-speaker match confidence.

## Output

Each recording gets its own session directory:

```
~/meet-recordings/meeting-20260312-140000/
    meeting-20260312-140000.wav                 # Stereo audio (16kHz)
    meeting-20260312-140000.session.json        # Recording metadata
    meeting-20260312-140000.ffmpeg.log          # ffmpeg capture log
    meeting-20260312-140000.txt                 # Plain text transcript
    meeting-20260312-140000.srt                 # Subtitle format
    meeting-20260312-140000.json                # Full detail (word-level timestamps)
    meeting-20260312-140000.summary.md          # AI meeting summary with YAML frontmatter
    meeting-20260312-140000.summary.meta.json   # Summary backend/model + timing metadata
    meeting-20260312-140000.frontmatter.json    # Structured frontmatter (schema_version 1)
    meeting-20260312-140000.pdf                 # Professional PDF (summary + transcript)
```

Example `.txt` output:

```
[00:00:12 --> 00:00:18] YOU: So the main issue we're seeing is with the API rate limiting.
[00:00:19 --> 00:00:25] REMOTE_1: Right, I think we should implement exponential backoff.
[00:00:26 --> 00:00:31] YOU: Agreed. Can you also look at caching the responses?
```

### Structured frontmatter

Every `.summary.md` ships with a typed YAML frontmatter block plus a
matching `.frontmatter.json` sidecar.  The schema is intentionally
small in v1 so downstream consumers can rely on it:

```yaml
---
schema_version: 1
type: meeting
title: Q2 Pricing Discussion
date: "2026-03-17T14:00:00+00:00"
duration: PT42M17S
language: en
participants:
  - name: YOU
    role: null
    channel: mic
  - name: Alice
    role: null
    channel: system
topics:
  - pricing
  - onboarding
action_items:
  - assignee: Alice
    task: Send pricing doc
    due: Friday
    status: open
decisions:
  - text: Run pricing experiment at $99/mo
    topic: pricing
source:
  session_id: meeting-20260312-140000
  audio_sha256: null
---
## Meeting Overview
...

## Key Topics Discussed
...
```

`schema_version: 1` is what every consumer should pin against.
The JSON sidecar contains the exact same dict for tools that don't
want to parse YAML.  `[vezir](https://github.com/pretyflaco/vezir)
0.2.0+` reads these files directly to build a queryable index over
your meetings.

### Backfilling existing sessions

Sessions recorded before meetscribe 0.7.0 / millet-pipeline 0.9.0 don't
carry frontmatter. Re-extract it for one or more sessions with:

```bash
# Re-run the LLM to produce frontmatter; idempotent (skips sessions
# whose .summary.meta.json already records data_extracted=true).
millet ingest ~/meet-recordings/meeting-2026*

# Force re-extraction even when frontmatter is already present:
millet ingest --force ~/meet-recordings/meeting-20260312-140000

# Preview without invoking the LLM:
millet ingest --dry-run ~/meet-recordings/meeting-2026*
```

`millet ingest` accepts the same `--summary-backend` /
`--summary-model` / `--ollama-singlepass` flags as
`millet transcribe` and regenerates the PDF by default
(`--no-pdf` to skip).

## AI summary

millet generates a structured meeting summary with:
- Overview
- Key topics discussed
- Action items (with owners when mentioned)
- Decisions made
- Open questions / follow-ups

### Supported models

| Model | Size | Speed | Notes |
|-------|------|-------|-------|
| `qwen3.5:9b` | 6.6 GB | ~18-35s | **Default** -- best balance of quality and speed |
| `gemma3:12b` | 8.1 GB | ~15s | Fastest |
| `qwen3:14b` | 9.3 GB | ~39s | Good quality |
| `glm-4.7-flash` | 19 GB | ~37s | Must use thinking-off mode (handled automatically) |

Change the model:

```bash
millet run --summary-model gemma3:12b
```

Disable summaries:

```bash
millet run --no-summarize
```

### Summary backends

millet supports two backends, both private, with automatic fallback:

| Backend | Setup | Cost | Quality | Privacy |
|---------|-------|------|---------|---------|
| `tinfoil` (default) | `pip install 'millet-pipeline[tee]'`, set `TINFOIL_API_KEY` (or drop a key file at `~/models/tinfoil/tinfoil.txt`) | ~$0.03/meeting | Excellent (DeepSeek V4.1 Flash) | **Hardware-attested TEE — prompts not visible to provider/operator** |
| `ollama` | `ollama serve` + `ollama pull qwen3.5:9b` | Free | Good | Fully local |

`claudemax`, `openrouter` and the generic `openai` backend were **removed in
0.19.0**.  They were the only way meeting content reached a party that could
read it, and they no longer bought any quality: in a blind, judged evaluation
the TEE model beat Sonnet 4.6 on both precision and recall in every language
tested.  A stale `MILLET_SUMMARY_BACKEND` naming one of them degrades to the
default with a warning rather than failing your jobs.

The `tinfoil` backend runs inference inside a hardware-attested TEE (AMD
SEV-SNP or Intel TDX, depending on the model).  The model provider can't
see the prompts, the cloud operator can't see the prompts, and the
integrity is checked against an attestation report on every request.
~$0.03 per meeting; median latency ~90 s (max ~115 s in the evaluation)
on DeepSeek V4.1 Flash.  Cost scales with the model's reasoning tokens,
not just transcript length — the non-Flash GLM-5.3 (now only the sibling
fallback) spends ~9× more for no measurable gain in summary coverage (see
CHANGELOG v0.18.1).

```bash
# Default: TEE. Nothing to configure beyond the API key.
export TINFOIL_API_KEY=tk_...

# Fully local instead (no API key, no network)
export MILLET_SUMMARY_BACKEND=ollama
export MILLET_SUMMARY_MODEL=qwen3.8:27b
```

> Environment variables use the `MILLET_` prefix. The older `MEETSCRIBE_`
> (and `MEET_`) spellings still work for one more release but emit a
> `DeprecationWarning`.

If the configured backend is unavailable, millet automatically tries the
next one in the fallback chain: **tinfoil → ollama**.  Both destinations are
private, so unlike the pre-0.19.0 chain this cannot silently downgrade
confidentiality — only quality.

Two narrower safety nets sit inside that:

- **Sibling TEE model.**  If the primary model's enclave pool is drained
  (HTTP 503) or retired (404), millet retries once on a different TEE model
  family before giving up.  Still attested; never silent (`fallback_used` is
  recorded in the `.meta.json` sidecar).
- **Requested presets ride the same chain** (since 0.21.2).  Every
  destination is private — attested TEEs plus fully-local Ollama — so a
  `confidential` summary produced anywhere in the chain is still
  confidential: content never reaches a third party.  A local fallback is
  recorded as `ollama/<model>` provenance, never labelled a TEE, and every
  switch is marked in the `.meta.json` sidecar.  (Until 0.21.2 a requested
  preset failed loud instead of falling back — a guard from the era when
  presets selected between backends of different privacy.)

### Summarization presets (deprecated)

Presets used to select between backends with different privacy and quality.
With only private backends left there is nothing to trade off, so the three
historical names are now **aliases for the default** and will be removed in
0.21.0.  They keep working meanwhile so existing scripts and stored jobs
don't break.

| Preset | Resolves to | Status |
|---|---|---|
| `confidential` | `tinfoil` / `deepseek-v4-1-flash` | Deprecated alias (the default) |
| `high-quality` | `tinfoil` / `deepseek-v4-1-flash` | Deprecated alias |
| `alternative` | `tinfoil` / `deepseek-v4-1-flash` | Deprecated alias |

```bash
# Quick check of which preset is in effect
millet transcribe ~/meet-recordings/today/today.wav --summary-preset confidential

# Or set per-session via env
export MILLET_SUMMARY_PRESET=high-quality
millet run
```

When a preset is set, `--summary-backend` and `--summary-model` overrides
are honored within that preset (e.g. `--summary-preset confidential
--summary-model gpt-oss-120b` swaps the model but keeps the TEE backend).

### Two-pass local summarization

When the **ollama** backend is selected, millet runs two
LLM calls instead of one:

1. **Pass 1 (extraction)** — pulls topics, actions, decisions, and open
   questions out of the transcript as plain numbered lists, using a
   context window sized to the full transcript.
2. **Pass 2 (formatting)** — takes the much smaller extracted data and
   organizes it into the canonical Markdown structure with a fixed 8K
   context window.

This dramatically improves format compliance and reduces hallucinations
on 20B-class local models (`gpt-oss:20b`, `qwen3.6:27b`) compared to a
single-pass call, at the cost of one additional LLM call (~30–90s extra).
The Tinfoil backend remains single-pass — it
already produce well-structured output in one shot.

To opt out and use the previous single-pass behavior:

```bash
millet run --ollama-singlepass
# Or via environment:
export MILLET_OLLAMA_SINGLEPASS=1
```

The `.summary.meta.json` sidecar records per-pass timings
(`pass1_seconds`, `pass2_seconds`, `pass1_chars`) when two-pass was used.

See [docs/local-model-evaluation.md](docs/local-model-evaluation.md) for
the full evaluation that motivated this design, including known failure
modes of local 20B-class models.

### Customizing the prompt

The summarization prompts live in `millet/prompts/` (e.g.
`summarize_system.md`, plus the `summarize_extract_*` / `summarize_format_*`
pairs used by the two-pass Ollama flow). Edit them to change the summary
format, add domain-specific instructions, or tune for your preferred model.
No Python changes needed.

## Voiceprint speaker recognition

millet can automatically identify speakers across meetings using voice
embeddings. After you label speakers in one meeting, their voice profiles are
stored and matched against future recordings.

```bash
# Build profiles from already-labeled sessions
millet enroll ~/meet-recordings/meeting-20260330-*

# Auto-label speakers in future meetings using voice profiles
millet label --auto ~/meet-recordings/meeting-20260401-093000
```

Profiles are stored in `~/.config/meet/speaker_profiles.json` and improve
with each labeled session (running average of embeddings).

Use `millet enroll --list` to list enrolled speakers. Team-scoped profiles
are supported via `--team <name>` on `enroll` and `label`, stored under
`~/.config/meet/<team>/speaker_profiles.json`, so each team keeps its own
voiceprint database.

## Meeting sync

Push meeting artifacts to a Git repository on a configurable schedule.

```bash
# Create an example config
millet sync --init-config
# Edit ~/.config/meet/sync_config.json with your repo URL and schedule

# Push a session manually
millet sync ~/meet-recordings/meeting-20260331-110038_STANDUP

# View configured schedule
millet sync --list-schedule
```

When the GUI detects a matching scheduled meeting, it prompts for confirmation
before syncing. Sessions that don't match the schedule are skipped. The CLI
uses `--force` to sync unmatched sessions.

You can also configure a `team_members` list and `min_team_members` threshold
in `sync_config.json` to require that a minimum number of recognized speakers
are present before offering to sync.

`millet sync` supports `--team <name>` (per-team config + clone) and
`--meeting-type <slug>` for routing. Push failures exit non-zero so scheduled
sync jobs can detect and retry them.

## Multilingual support

millet auto-detects the spoken language by default (Whisper large-v3-turbo
supports 99 languages). You can also set it explicitly:

```bash
millet run --language de       # German
millet run --language tr       # Turkish
millet run --language fr       # French
millet run --language es       # Spanish
millet run --language fa       # Farsi (Persian)
millet run --language auto     # Auto-detect (default)
```

### How it works

- **Transcription**: The same Whisper model handles all languages -- no extra
  download or VRAM cost. When set to `auto`, the detected language is used for
  alignment and all downstream steps.
- **Speaker diarization**: Completely language-agnostic (based on voice
  characteristics, not speech content).
- **AI summary**: When a non-English language is detected, the summary prompt
  instructs the LLM to write the summary in the same language as the transcript.
- **PDF output**: Uses DejaVu Sans for full Unicode coverage (Latin, Cyrillic,
  Greek, Turkish special characters, etc.). Farsi uses Noto Naskh Arabic with
  RTL text reshaping.

### Tested languages

| Language | Code | Alignment model | PDF font | Notes |
|----------|------|----------------|----------|-------|
| English  | `en` | wav2vec2 (torchaudio) | DejaVu Sans | |
| German   | `de` | VoxPopuli (torchaudio) | DejaVu Sans | |
| French   | `fr` | VoxPopuli (torchaudio) | DejaVu Sans | |
| Spanish  | `es` | VoxPopuli (torchaudio) | DejaVu Sans | |
| Turkish  | `tr` | wav2vec2 (HuggingFace) | DejaVu Sans | ~1.2 GB alignment model download |
| Farsi    | `fa` | wav2vec2 (HuggingFace) | Noto Naskh Arabic | ~1.2 GB alignment model download, RTL |

### Downloading alignment models

Alignment models download automatically on first use, but you can pre-fetch
them (e.g. before an offline session) with `millet download`:

```bash
millet download            # show cache status for all supported languages
millet download de tr fa   # download German, Turkish, Farsi alignment models
millet download --all      # download every supported alignment model
millet download parakeet   # download the Parakeet ASR model (English)
```

It exits non-zero if any requested download fails, so it's safe to script.

### Translating a transcript

`millet translate` renders an existing session transcript into another
language via Ollama, preserving timestamps and speaker labels. The result is
saved as `<basename>.translation.<lang>.txt` in the session directory.

```bash
millet translate ~/meet-recordings/meeting-20260313-231509           # to English
millet translate ~/meet-recordings/meeting-20260313-231509 --to de   # to German
```

### Farsi RTL requirements

Farsi uses right-to-left text. For proper PDF rendering, install the optional
RTL dependencies:

```bash
pip install arabic-reshaper python-bidi
# Or with the optional extra:
pip install "millet-pipeline[rtl]"
```

Without these libraries, Farsi text will appear in the PDF but glyphs may not
be joined correctly and reading order may be wrong.

## Virtual sink mode

By default, `millet record` captures all system audio (including notification
sounds, music, etc.). For cleaner recordings, use `--virtual-sink`:

```bash
millet record --virtual-sink
```

This creates an isolated audio sink. Route your meeting app's audio to it:

1. Open `pavucontrol` (PulseAudio Volume Control)
2. Go to the "Playback" tab
3. Find your browser or meeting app
4. Change its output to "Meet-Capture"

You'll still hear the meeting through your normal speakers via automatic loopback.

## Configuration & environment variables

Environment variables use the `MILLET_` prefix (legacy `MEETSCRIBE_` / `MEET_`
spellings are honored for one more release with a `DeprecationWarning`):

| Variable | Purpose |
|----------|---------|
| `MILLET_SUMMARY_BACKEND` | Default summary backend (`tinfoil` default, or `ollama`) |
| `MILLET_SUMMARY_MODEL` | Default summary model for the chosen backend |
| `MILLET_SUMMARY_PRESET` | Deprecated. All three names resolve to the default; removed in 0.21.0 |
| `MILLET_OLLAMA_SINGLEPASS` | Set to `1` to disable two-pass Ollama summarization |
| `TINFOIL_API_KEY` | Required for the `tinfoil` backend (or a key file at `~/models/tinfoil/tinfoil.txt`) |
| `HF_TOKEN` | HuggingFace token for pyannote diarization |
| `MILLET_CONFIG_DIR` | Override the config dir (default `~/.config/meet`) |
| `MILLET_PROFILES_PATH` | Override the voiceprint profile DB path |
| `MILLET_RECORDINGS_DIR` | Override the recordings directory (default `~/meet-recordings`) |

### Offline mode

Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to run without any network
access. millet honors these and, on a cache miss, prints actionable guidance
naming the exact model to pre-fetch (via `millet download`) rather than
failing with an opaque network error.

## VRAM usage

With an NVIDIA GPU (12 GB VRAM):

| Model | Transcription | + Diarization | Recommended batch_size |
|-------|--------------|---------------|----------------------|
| large-v3-turbo | ~4 GB | ~7 GB total | 16 |
| medium | ~3 GB | ~6 GB total | 16 |
| base | ~1 GB | ~4 GB total | 16 |

If you hit OOM errors:
1. Reduce `--batch-size` to 4 or 8
2. Use `--compute-type int8`
3. Use a smaller model (`--model medium` or `--model base`)
4. Use `--device cpu` as a last resort

## How it works

```
[Meeting App] --> [PipeWire/PulseAudio] --> [ffmpeg dual-channel capture] --> meeting.wav
                                                                                  |
                  [WhisperX: faster-whisper + wav2vec2 alignment + pyannote diarization]
                                                                                  |
                                      [ LLM summary ]          [Diarized transcript]
                                              |                         |
                                        .summary.md          .txt / .srt / .json
                                              |                         |
                                              +--------> .pdf <---------+
```

**Capture**: Records your mic (left channel) and system audio (right channel)
simultaneously into a single stereo WAV file at 16 kHz.

**Transcribe**: Runs the WhisperX pipeline -- batched Whisper transcription,
wav2vec2 forced alignment for word-level timestamps, and pyannote speaker
diarization. Dual-channel energy analysis maps speakers to YOU or REMOTE.

**Summarize**: Sends the transcript to the summary backend (TEE by
default, or a local Ollama model) which extracts
a structured summary.

**PDF**: Combines the summary and full transcript into a professional
page-numbered PDF document.

## CUDA NVRTC note

The pyannote diarization model requires CUDA NVRTC for JIT compilation. If your
CUDA driver version doesn't match the installed libnvrtc-builtins version,
millet automatically creates a compatibility symlink. This happens
transparently on first use.

If you still see NVRTC errors:

```bash
export LD_LIBRARY_PATH=$HOME/.local/lib/cuda:$LD_LIBRARY_PATH
```

## Limitations

- Overlapping speech is not handled well (Whisper limitation)
- Speaker labels default to role-based (YOU, REMOTE_1, REMOTE_2) — use `millet label` or the GUI dialog to assign real names
- Diarization accuracy varies with audio quality and number of speakers
- Audio capture (`millet record`, `millet run`) requires Linux with PulseAudio
  or PipeWire. Transcription, labeling, summarization, and sync work on both
  Linux (CUDA) and macOS Apple Silicon (MLX Whisper + MPS) as of v0.6.0.
- Windows is not supported.
- Local 20B-class summary models (e.g. `gpt-oss:20b`) can hallucinate on
  transcripts dominated by very short low-information utterances ("yes",
  "okay") and may exceed the default 600s timeout on very large
  (>100 KB) non-English transcripts. For these cases configure a cloud
  backend (tinfoil) — the fallback chain takes over
  automatically. See [docs/local-model-evaluation.md](docs/local-model-evaluation.md).

## FAQ

**Is there a GUI?** Yes — run `millet gui` for a small always-on-top GTK3
widget with Record/Stop, live timer, status indicator, and one-click
access to the resulting PDF and session folder. See
[Launch the GUI widget](#launch-the-gui-widget) for details.

**Does it work on Windows / macOS?** System-audio recording requires Linux
(PulseAudio / PipeWire). The post-capture pipeline (`millet transcribe`,
`millet label`, `millet sync`, etc.) works on macOS Apple Silicon as of v0.6.0
— install with `pip install 'millet-pipeline[mlx]'`. Windows is not
supported.

**Can I run a Mac as a transcription server?** Yes — see
[vezir](https://github.com/pretyflaco/vezir), the team-scale wrapper around
millet. A Mac can act as the GPU server with Linux laptops or the
[Android client](https://github.com/pretyflaco/vezir-android) providing
the audio.

**Can I use it without a GPU?** Yes, with `--device cpu`, but
transcription will be 5–20× slower depending on the Whisper model.
See [VRAM usage](#vram-usage).

## Contributing

```bash
git clone https://github.com/pretyflaco/millet
cd millet
pip install -e .[dev]
/usr/bin/python3 -m pytest tests/
```

Pull requests welcome. Please run the test suite before submitting.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

[GPL-3.0](LICENSE)
