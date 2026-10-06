# Changelog

## v0.21.6 — TEE default → DeepSeek V4.1 Flash (Tinfoil deprecates GLM-5.3 Flash)

Tinfoil announced on 2026-10-06 that `glm-5-3-flash` — the default summary
model since 0.18.1 — is deprecated **on 2026-10-09**, to shift capacity to
`deepseek-v4-1-flash` and `glm-5-3`.  Unlike `glm-5-2`'s silent retirement,
this one came with notice; this release lands before the cutoff.

### Changed

* **`DEFAULT_TINFOIL_MODEL` → `deepseek-v4-1-flash`.**  It was the
  runner-up in the grounded TEE evaluation
  (`docs/tee-summarization-evaluation.md`), so this is a switch on existing
  evidence, not a new bet:

  | | `glm-5-3-flash` (old) | `deepseek-v4-1-flash` (new) |
  |---|---|---|
  | precision | 98.3 / 98.7% | 98.2 / 98.6% |
  | recall | 91.2 / 94.2% | 87.2 / 89.8% |
  | hallucination | 0.1 / 0.2% | 0.3 / 0.1% |
  | owner attribution | 94.6 / 95.5% | 90.6 / 93.4% |
  | latency median / max | 111 s / 353 s | 88 s / 115 s |
  | gate vs Sonnet 4.6 | 16/16 | 15/16 (TR recall −2.1 pp, one judge) |

  Expect slightly less topic coverage (recall gap is mostly *topics*,
  87 vs 94; actions/decisions are within ~1 pp) and steadier latency.
  Cost rises from ~$0.02 to ~$0.03 per meeting.
* **`DEFAULT_TINFOIL_FALLBACK_MODEL` → `glm-5-3`** (was
  `deepseek-v4-1-flash`).  The sibling must stay a different family from
  the primary.  `glm-5-3` is text-only, ~9× the cost and ~3× slower, which
  is acceptable on a path that only runs when the primary pool is down; a
  frames run that lands on it degrades to text-only, never fails.
* **`deepseek-v4-1-flash` added to `VISION_MODELS`.**  It was excluded in
  0.20.0 because its vision endpoint answered 502 on every request
  (2026-09-12).  Re-verified 2026-10-06 through the attested SDK: 10
  synthetic 880×1920 frames, every on-screen code read correctly in 2/2
  runs (5–7 s).  It bills ~10k image tokens where `glm-5-3-flash` billed
  ~22k for the same frames, so it likely downscales harder — small UI text
  in real screen recordings is not yet verified.  `glm-5-3-flash` stays on
  the allowlist for anyone pinning it until Tinfoil removes it.
* The three deprecated presets resolve to the new default; GUI label,
  README, REQUIREMENTS updated.

### Operator note

A `MILLET_SUMMARY_MODEL=glm-5-3-flash` pin in a deployment's environment
overrides this default and will start failing on 2026-10-09 — remove it.

### Tests

Vision-allowlist and sibling-fallback tests now use the constants instead
of the literal `glm-5-3-flash`, the env-override test uses a model distinct
from the default (it had become a tautology), and a new test pins that the
`glm-5-3-flash` prefix cannot admit text-only `glm-5-3` to the vision path.

## v0.21.5 — voiceprints learn only from real speech; CROSSTALK beats a voiceprint match

Incident 2026-09-30 (vezir, blink team): a profile named "Pattern" (a
teammate's handle) appeared in dev standups, sales calls and interviews,
always on the same shape of text — "Okay. | Yeah. | Bye." at ~1 word/s — and
once on a colleague's self-introduction.  Measured on the live DB it sat
**0.10** from the voice of the person it was named after and **0.86** from
the *scribe's own voice echoing back through the call* on the system
channel.  It had been built from filler, then reinforced on every labeling
submit that confirmed its (pre-filled) match — five running-average merges,
none reversible.  It also defeated 0.21.4: the filler bucket matched
"Pattern" at ~0.9 *before* the CROSSTALK rule could see it.

### Fixed

* **Profiles learned from the longest segments — for filler, the worst
  ones.**  Enroll and update-from-labels picked a cluster's longest
  segments; for a filler cluster those are fillers Whisper stretched over
  silence ("Yeah." across 6.7 s): room tone and echo.  Learning now uses
  `learnable_segments()`: dense speech only (≥ 1.5 s, ≥ 6 words, ≥ 1.5
  words/s), most words first, and nothing at all under 4 s of it (the same
  floor that gates auto-apply).  A filler/echo cluster can no longer create
  or update a profile, however often its name is confirmed.  Matching is
  unchanged.
* **A ghost REMOTE bucket now becomes CROSSTALK even when a voiceprint
  matches it.**  Filler from several people has no single voice to match; a
  confident match on it means a polluted profile, not a person.  The match
  is logged ("Ignoring voiceprint match … filler crosstalk") and not offered
  as an auto-id suggestion.  Replay data: 1 in 120 ghost-shaped buckets was
  ever one real person.

## v0.21.4 — label ghost REMOTE buckets as CROSSTALK

A leftover `REMOTE` bucket of sub-second fillers ("Bye.", "Yeah.", "Hmm.")
forced about half of all team meetings into `needs_labeling` even when every
real participant had been identified by voiceprint; the scribe then opened
the session only to *not* label it.  Measured on 257 real transcripts: 55
such buckets had been left unnamed by a human.

### Changed

* **`label --auto` labels a ghost `REMOTE`/`REMOTE_N` bucket `CROSSTALK`.**
  The honest technical reason, readable by a human: these words were heard
  but cannot be assigned to a speaker.  No owner is guessed.  A bucket is a
  ghost (`millet.crosstalk.is_ghost_speaker`) when its words are ≤ 2% of the
  meeting's (≤ 150), its median segment is ≤ 1 s, and at most 2 of its
  segments exceed 6 words.  Judged in words, not seconds — Whisper stretches
  filler timestamps ("No, no, no." over 22.6 s).  Several ghost buckets
  collapse into one `CROSSTALK` speaker; a substantial `REMOTE` stays raw
  for a human.  Replay: 49/55 unnamed buckets → `CROSSTALK` (the other 6
  are single blips the tiny-noise fold already handles, or real speech);
  1/120 buckets humans had named a real person would have been labeled
  `CROSSTALK` (two greetings, 1.9 s).
* **`CROSSTALK` is reserved.**  It is never enrolled or updated as a
  voiceprint and never matched (a stored profile of that name is ignored),
  never listed as a participant (frontmatter, PDF header), and its lines are
  left out of the summary input.  They stay in the txt/srt/json/PDF
  transcript under that label.
* Replaces the 0.12.12 overlap-absorb of a small `REMOTE` into the nearest
  named speaker (`absorb_unresolved_remote`, removed) — see below.

### Fixed

* **The 0.12.12 REMOTE rescue never ran in production.**  `label --auto`
  passed the label map's *names* as the resolved set while the transcript
  still carried *raw* ids, so no named target was ever found.  Its tests
  used pre-named transcripts, which hid it.  The new tests start from raw
  ids, as production does.
* **The tiny-noise fold could overwrite a confident voiceprint match.**  Same
  names-vs-ids mix-up: a short cluster voiceprint had matched (e.g. 2 s →
  Andrej) still counted as unresolved noise and was folded into the dominant
  speaker.  Resolved raw ids are now passed.
* **`label --auto` fed derived labels into the voiceprint DB.**  Labels it
  derives itself (tiny-noise folds, `CROSSTALK`) counted as "manually
  confirmed" and updated profiles — with no voiceprint matches at all, the
  whole label map did.  Only labels a human typed update profiles now.
* **CI red since millet-record 0.6.0** (test-only).  Its new `.recorder.json`
  marker `json.dumps` the recorder's pid; the `test_capture` fake Popen was a
  bare `MagicMock` whose `pid` is a truthy mock.  The fake now has
  `pid = None`, millet-record's documented skip path for test doubles.

## v0.21.3 — cap summary frames at the endpoint's 10-image request limit

Incident 2026-09-17 (vezir/saray): the first screen recording long enough
to narrate more than ten cues failed its summary on every backend.  Session
`01M2P6FTRG4WAKKE5T7TV6HNFM` (18 cue frames, `iteration-plan` template)
hit venice answering 400 `At most 10 image(s) may be provided in one
request` — a request-shape validation, not a transient fault, so retry
budgets burned and the vision gate (correctly) refused to drop the frames
onto a text-only tier.  `millet label --apply-json` exited 1 and vezir
recorded `summary_error`.

`MAX_FRAMES` was 45, chosen for token budget and attachment-sync headroom
on the assumption that vezir's extraction cap was the binding constraint;
nobody had checked what the vision endpoints actually accept per request.
Both attested vision paths (tinfoil's `_user_content`, the shared
`_encode_frames_content` for venice/near) attach every normalized frame to
a single message, so any session with >10 cues was guaranteed to fail —
short demos only ever worked because they were short.

- `MAX_FRAMES` 45 → 10: now the endpoint's hard limit, documented where it
  is enforced (verified live 2026-09-17), not a budget heuristic.
- `_usable_frames` over-cap selection changed from head-truncation
  (`out[:MAX_FRAMES]`, which would summarize only the opening minutes) to
  even sampling across the timeline — first + last frames kept, mirroring
  vezir's `_even_sample`.  Frames on disk are untouched: they remain full
  artifacts for the TUI/Android/git sync; only the model request is
  sampled.  A 10-frame request is also ~4.5x cheaper (~22k vs ~100k
  frame-tokens).
- Tests: even sampling keeps first + last and never duplicates through
  index rounding; a regression test pins the production failure shape —
  `MAX_FRAMES + 8` frames in, exactly `MAX_FRAMES` image parts on the
  wire and `frames_used == MAX_FRAMES`.  Suite 418 → 420.

## v0.21.2 — requested presets ride the private fallback chain

Until now an explicitly requested preset (`--summary-preset confidential`,
which vezir sends on every upload) re-raised the primary backend's failure
instead of falling back: no venice, no near, no ollama — a hard job error
on any full Tinfoil outage.  That guard predates the private-only era: it
existed to prevent a requested preset from silently routing meeting content
to a cloud backend that could read it.  Since 0.19.0 every backend is
private — three hardware-attested TEEs plus fully-local Ollama — so the
guard protected nothing while converting recoverable provider outages into
hard failures on the one path production always sends.

The confidential contract is "no third party can read the content".  Every
destination in the chain satisfies it — a local model as strictly as a TEE
(content never leaves the box).  A requested preset therefore follows the
same chain as the default (`tinfoil → venice → near → ollama`, order
unchanged): a Tinfoil outage degrades quality, never confidentiality, and
never silently — every switch records `fallback_used` and `<backend>/<model>`
provenance in `.summary.meta.json` (vezir surfaces it as `jobs.summary_fallback`
+ `· fallback`, and a local fallback is `ollama/<model>`, never labelled a
TEE).  The preset-unavailable pre-flight raise is gone too: a missing
primary key now chains (with the skip reason kept for the all-unavailable
error) instead of hard-failing.

- `summarize()` module/function docstrings updated to the chain semantics;
  README's "explicit preset pins the backend" section rewritten.
- Tests: `TestPresetFallback` replaces `TestPresetNeverFallsBack` — preset
  chains across the tier (TEE tier preferred when available, local marked
  never-TEE, unavailable primary chains, all-failing still fails loud);
  the tinfoil-SDK-missing end-to-end test now pins the chained path.
  Suite 410 → 418.

## v0.21.1 — hard wall-clock deadline on every Tinfoil attempt

Incident 2026-09-15 (vezir/saray): a `confidential` summary pinned a job in
`transcribing` for 30+ minutes with the GPU idle.  faulthandler stack:
`openai → httpx → tinfoil custom transport → httpcore _receive_event →
ssl.read` — blocked forever.  Two compounding causes:

- The Tinfoil SDK's custom httpx transport does not reliably honour the
  per-request `timeout` (passed since 0.13.0), so no per-phase timeout
  bounds the call.
- The openai SDK silently retries a timed-out request internally
  (`max_retries=2` by default), so one millet "attempt" could legitimately
  block for 600s × 3 = 30 minutes before millet's own retry ladder ever saw
  an exception — and a transport-level stall never surfaced at all.

Every `_summarize_tinfoil` attempt (client init + completion, primary and
sibling model alike) now runs under an external wall-clock deadline of
`config.timeout + 120s` (default 720s), enforced in a daemon thread.  A
stalled attempt is abandoned (the thread dies with the process — a raw
daemon thread, not a `ThreadPoolExecutor`, whose atexit join would hang
interpreter exit on an abandoned worker) and raises `_TinfoilAttemptStuck`,
a `TimeoutError` subclass, so the existing transient retry ladder applies
unchanged.  After the budget the job fails loudly — or, on 0.21.0's chain,
falls through to the decorrelated TEE providers.

Verified live: a 42KB-prompt summary that repeatedly hung vezir's worker
completed in 153.6s on the same code path, and a forced-stall test confirms
all 3 attempts hit the wall-clock deadline and fail loud.

## v0.21.0 — attested TEE fallbacks (Venice, NEAR) so Tinfoil is not a SPOF

Summarization had exactly one attested provider: Tinfoil.  If its enclave
router was unreachable the chain fell straight through to local Ollama —
private, but a quality cliff and no attested network alternative in between.

Two decorrelated TEE providers now sit between them.  The default fallback
chain is `tinfoil → venice → near → ollama`.  Venice and NEAR are both
Intel TDX + NVIDIA H100/H200 Confidential-Computing, OpenAI-compatible and
self-serve, and run on different hardware/clouds than Tinfoil, so a
Tinfoil-side outage no longer takes summarization off attested hardware.

- **First-party attestation** (`millet/attestation.py`), verified against the
  live providers 2026-09-15.  Both use the same Phala/dstack two-step flow: a
  GET returns a document with an opaque `nvidia_payload`, which millet POSTs to
  NVIDIA's Remote Attestation Service (NRAS v4).  millet then requires, from the
  NVIDIA-signed EAT, `x-nvidia-overall-att-result == true` **and** that
  `eat_nonce` equals the fresh nonce we sent (anti-replay).  Confirmed PASS on
  `z-ai/glm-5.3-flash` (NEAR) and `e2ee-glm-5-3-flash` (Venice).  A failed
  attestation is loud — it falls through to the next backend, never saves an
  unverified result as attested.  Full Intel TDX quote / DCAP-chain validation
  is a deliberate ceiling delegated to provider reference verifiers, not
  hand-rolled here (see the module).
- **TLS against a private CA.**  The openai SDK uses httpx, which honors
  `SSL_CERT_FILE`; on server hosts that points at a private CA (saray's Caddy
  internal CA), so every Venice/NEAR call failed with a misleading "Connection
  error".  These are public endpoints, so the client is pinned to certifi's
  public CA bundle explicitly.  (requests never hit this — it always uses
  certifi.)
- **Billing errors fail loud, not silent.**  Unpaid keys return HTTP 402
  (NEAR `no_limit_configured`, Venice "Insufficient USD or Diem balance").  That
  is classified as a billing error and raised with an "add credits" message —
  not retried as a transient (which would just burn the budget), and clearly
  distinct from an attestation fault.
- **Vision-gating.**  A screen-recording (frames) job never falls back to a
  text-only tier (NEAR, Ollama) — that would silently drop the images.
  Such tiers are skipped as fallbacks for a frames job; it stays on a vision
  tier (Tinfoil, or Venice's `e2ee-qwen3-vl-30b-a3b-p`) or fails loud.  Model
  ids verified live: Venice attested models carry an `e2ee-` prefix
  (`e2ee-glm-5-3-flash`); NEAR uses `z-ai/glm-5.3-flash`.
- **Generic backend** `_summarize_attested_oai` drives both providers over
  one OpenAI-compatible path; they differ only by the `ATTESTED_BACKENDS`
  table (base URL, key env/file, models, attestation + catalog URLs).
- **Catalog pre-flight** generalized (`verify_tinfoil_model` →
  `verify_model(backend, model)`) so the model-check timer covers the new
  providers too.
- **Local tier (Nemotron), benchmarked.**  The Ollama tier is the zero-network
  privacy floor.  `nemotron-3-nano:4b` (2.8 GB) was benchmarked on an RTX 3090
  against the current default `qwen3.5:9b`: it fits the saray target (RTX 5070,
  12 GB) with ~3.2 GB of weights, captured 6/6 key facts in a real summary, at
  ~17 s vs qwen's ~12.6 s — an easy fit and fine quality for a last-resort tier.
  Point the tier at it with `MILLET_SUMMARY_BACKEND=ollama
  MILLET_SUMMARY_MODEL=nemotron-3-nano:4b` (no code change).  Note the 24 GB
  `nemotron-3-nano:latest`/`:30b` Omni tags do NOT fit 12 GB — use `:4b`.  The
  tier is text-only: Ollama cannot load the multimodal Nemotrons' mmproj vision
  weights, so frames never route here.

Keys: `VENICE_API_KEY` / `NEAR_AI_API_KEY` (env, then a 0600 key file,
mirroring the Tinfoil key pattern).  Tests +9 (`test_attested_fallback.py`).

## v0.20.1 — fix: the default backend's SDK ships by default

`pip install millet-pipeline` could not summarize on its own default path.

Two defects compounded.  `tinfoil` — the SDK behind
`DEFAULT_SUMMARY_BACKEND` since 0.19.0 — was still declared only in the
optional `[tee]` extra, a leftover from when `ollama` was the default and
the TEE was the opt-in `confidential` preset.  And availability was decided
by "is an API key resolvable?" alone, never checking that the package
imports.

So with a key set but no SDK the backend reported itself **available**, the
run walked past the guard, and execution reached a bare
`from tinfoil import TinfoilAI`:

| situation | before | now |
|---|---|---|
| key + no SDK, no preset | `ModuleNotFoundError` caught, fell to ollama | skipped cleanly, falls to ollama |
| key + no SDK + **preset** | **`ModuleNotFoundError` re-raised — hard crash** | readable error naming the fix |
| no SDK, no ollama | `All summary backends failed. Last error: None` | names both causes |

The preset row was the live path: `vezir[server]` pins `millet-pipeline`
without the `[tee]` extra, and the Android client sends a preset on every
upload.

### Fixed

- **`tinfoil>=0.12` moved from the `[tee]` extra into base dependencies.**
  The default backend's SDK is no longer optional.  `[tee]` is kept as an
  empty no-op extra so existing `millet-pipeline[tee]` install commands, CI
  jobs and deploy scripts keep resolving instead of erroring on an unknown
  extra.  ~700 KB plus its tree, negligible beside whisperx/torch, and only
  server-side installs pull this package at all.
- **`is_backend_available()` now also requires the SDK to import**
  (`tinfoil_sdk_installed()`).  Availability must not report a backend as
  usable when the import that drives it would fail.  Kept even though the
  dependency is no longer optional: constraint files, partial upgrades and
  pre-0.20.1 environments can all still lack it.
- **`_backend_not_available_message()` distinguishes a missing SDK from a
  missing key**, and names the command that fixes it.
- **"No summary backend is available" replaces "Last error: None".**  When
  every backend is skipped as unavailable nothing is ever dispatched, so
  `last_error` stayed `None` and the final error named no cause at all.  It
  now reports why each backend was skipped.

### Tests

527 passing (9 new in `tests/test_tinfoil_sdk_missing.py`: availability with
and without SDK/key, message content for each cause, ollama fallback when
the SDK is absent, the preset path failing readably rather than with an
`ImportError`, and the all-unavailable message naming both causes).

## v0.20.0 — feat: summarize from what's on screen, not just what was said

A narrated screen recording carries one PNG per transcript cue.  Those
frames can now be sent to a vision-capable TEE model alongside the
transcript, so an iteration plan reports what is **visibly** wrong rather
than only what the narrator happened to say out loud.

On a real 31-frame session the vision run surfaced five issues that are
not derivable from the transcript at any quality of language model:

* a "Save Changes" bar overlapping the "Start Unilateral Exit" row,
* a destination field accepting a **mainnet** `bc1p…` address while the
  wallet was on **regtest** (it read both addresses off-screen and
  reasoned about the mismatch) — a funds-loss bug,
* Slow/Medium/Fast fee tiers all rendering an identical "1 sat/vB",
* narration saying "1.1.0" while the footer read "Glow v1.2.0 (dev)",
* "Buy Bitcoin" filed under the "Display" settings section.

### Added

* **`SummaryConfig.frames`** — optional list of still images sent with the
  transcript.  Normalized on construction (readable image files only,
  de-duplicated, ordered, capped at `MAX_FRAMES = 45`); anything unusable
  is dropped quietly, because frames are an enhancement and must never
  fail a summary the caller asked for.
* **`discover_cue_frames(session_dir)`** — finds
  `<session>/attachments/cue_*.png`, whose names sort into narration
  order.  Wired into `millet label` (so a re-summary of an existing
  session picks frames up automatically — the path vezir uses) and into
  `millet transcribe` via `--summary-frames/--no-summary-frames`.
* **`MeetingSummary.frames_used`**, recorded in the `.meta.json` sidecar.
  0 means the summary was text-only — either no frames were supplied, or
  the model that served it cannot see.
* **Vision allowlist** (`VISION_MODELS`), deliberately *not* Tinfoil's
  advertised `multimodal` flag.  The catalog marks
  `deepseek-v4-1-flash` multimodal, but its vision endpoint answers 502 on
  every request (verified 2026-09-12 over three consecutive attempts).
  Since that model is the sibling fallback, trusting the flag would turn a
  drained primary pool into a hard failure.  A model that cannot see gets
  the prompt as plain text instead.

### Fixed

* **Enclave attestation failures are now retried.**  Tinfoil intermittently
  serves a malformed SEV attestation report (`current_tcb not correctly
  formed`) and the SDK correctly refuses the response.  This was classified
  non-transient, so it failed the job outright with zero retries — on the
  `confidential` path, which by design has no fallback.  Measured failure
  rate on 2026-09-12: **9/12 requests succeeded (~25% failure)**, on plain
  text requests, i.e. this was silently costing real jobs.

  Attestation errors get their own larger budget
  (`_TINFOIL_ATTEST_MAX_ATTEMPTS = 6`, flat 1.5 s backoff) and do **not**
  consume the attempts reserved for genuine network faults, because they
  fail fast — during enclave verification, before any tokens are generated.
  Retrying does not weaken the guarantee: an unverified response is never
  accepted, we simply ask again, and exhausting the budget raises an error
  that says so explicitly.  They are classified transient but *not* a pool
  error, so the retry stays on the same model — a sibling runs on the same
  enclave infrastructure and would fail identically.

* **Sibling-fallback provenance was being erased.**  `summarize()` assigned
  `result.fallback_used = backend != config.backend`, overwriting the flag
  the tinfoil backend had already set for a sibling-model fallback (which
  keeps `backend == config.backend`).  Every 0.18.1 sibling rescue was
  therefore recorded as "no fallback" in the sidecar.  The existing test
  missed it by calling `_summarize_tinfoil` directly rather than going
  through `summarize()`.

### Changed

* The `iteration-plan` prompt no longer merely *claims* a screenshot
  exists for every cue — it explains how to use them when present, demands
  that visible defects be described in the actual on-screen wording,
  forbids claiming to see anything not in a frame, and still works from the
  transcript alone when no frames are supplied.

### Cost

~2.2k tokens per 880×1920 frame, linear.  A 31-frame session ran 126.9 s
and cost roughly $0.03 — against 43.2 s text-only.

### Tests

385 (up from 351): 34 new across frame normalization, discovery, vision
gating, message construction, degradation to text-only, frames surviving
the dispatch rebuild, attestation retry/budget-isolation, and the
fallback-provenance regression.

## v0.19.0 — BREAKING: private-only summarization (cloud backends removed)

Meeting content no longer reaches any party that can read it.  The three
non-private summary backends — `claudemax`, `openrouter`, and the generic
`openai` endpoint — are **removed**.  What remains is a hardware-attested
TEE (`tinfoil`, now the default) and fully local `ollama`.

This is not a privacy-over-quality trade.  A blind, judged evaluation over
10 real meetings found the TEE model **beat Sonnet 4.6 on both precision
and recall, in every language tested** — so routing transcripts through a
provider that can read them had stopped buying anything.  Full method and
numbers in `docs/tee-summarization-evaluation.md`.

| | `claude-sonnet-4-6` | `glm-5-3-flash` |
|---|---|---|
| precision | 94.5 / 95.3% | **98.3 / 98.7%** |
| recall | 73.9 / 79.0% | **91.2 / 94.2%** |
| hallucination | 1.1 / 0.6% | **0.1 / 0.2%** |
| distortion | 4.3 / 4.1% | **1.6 / 1.1%** |

(two figures = the two independent judges; both ranked the field identically)

### Removed

* **Backends `claudemax`, `openrouter`, `openai`** — along with
  `_summarize_claudemax`, `_summarize_openrouter`, `_summarize_openai`,
  `is_claudemax_available`, `_effective_temperature`, `_KIMI_KSERIES_RE`,
  and the `DEFAULT_OPENROUTER_MODEL` / `DEFAULT_CLAUDEMAX_MODEL` /
  `DEFAULT_OPENAI_COMPAT_MODEL` / `CLAUDEMAX_BASE_URL` /
  `CLAUDEMAX_HEALTH_URL` / `OPENROUTER_BASE_URL` constants.  `millet/` now
  has zero references to the `openai` Python package.
* **`MILLET_SUMMARY_PRESET_FALLBACK`** — the opt-in existed to let a preset
  reach a *cloud* backend when the primary was exhausted.  Those backends
  are gone, so an explicitly requested preset now always fails loud.
* **Env vars** `MILLET_OPENAI_BASE_URL`, `MILLET_OPENAI_API_KEY`,
  `MILLET_OPENAI_MODEL`, `OPENROUTER_API_KEY` are no longer read.

### Changed

* **Default backend is now `tinfoil`** (was `ollama`), model
  `glm-5-3-flash`.
* **Fallback chain is `tinfoil → ollama`** (was
  `claudemax → tinfoil → openrouter → ollama`).  Both destinations are
  private, so the chain can degrade *quality* but never *confidentiality* —
  which was the whole hazard of the old chain.
* **Presets are deprecated, not removed.**  `high-quality`, `confidential`
  and `alternative` all now resolve to the same default and will be deleted
  in 0.21.0.  They must keep working meanwhile: vezir passes
  `--summary-preset` on every job and ~580 stored jobs carry the names.
  Using one logs an informational notice (once per name per process).
* **PDF: attestation footer replaces the blanket CONFIDENTIAL watermark.**
  Previously any TEE-backed summary stamped every page `CONFIDENTIAL` in
  red.  Now that *every* summary is TEE-backed that would be wallpaper, so
  a TEE summary gets a quiet grey "Summarized in a hardware-attested TEE"
  footer, and the red banner is reserved for callers passing
  `confidential=True`.
* GUI preset dropdown offers only the default; CLI `--summary-backend`
  choices are `tinfoil` / `ollama`.

### Fixed (upgrade safety)

Two ways this release could otherwise have broken every job on an existing
deployment, both caught before shipping:

* **Stale `MILLET_SUMMARY_BACKEND` / `MEETSCRIBE_SUMMARY_BACKEND` naming a
  removed backend** no longer raises `ValueError` on every summarization.
  Deployments keep these in systemd env files that outlive an upgrade
  (saray had `MEETSCRIBE_SUMMARY_BACKEND=claudemax`), so it now degrades to
  the default with a loud warning.  An *explicit* `backend=` argument still
  raises — that is a caller bug, not stale config.
* **Stale `MILLET_SUMMARY_MODEL` holding an Ollama tag** is no longer sent
  to the enclave.  Before 0.19.0 the default backend was `ollama`, so a
  bare `MILLET_SUMMARY_MODEL=qwen3.8:27b` was normal; with `tinfoil` as the
  default that value would be rejected as a nonexistent model at request
  time.  Ollama tags contain `:` and Tinfoil ids never do, so it is ignored
  with a warning pointing at `MILLET_SUMMARY_BACKEND=ollama`.

### Tests

348 total (up from 344): the preset/fallback suite was rewritten around the
new semantics (backend registry, stale-env downgrade, stale-model guard,
private-only chain, preset aliasing, preset-never-falls-back), plus 4 new
PDF tests covering attestation-vs-watermark.

## v0.18.1 — fix: `confidential` preset model migration (GLM-5.2 → GLM-5.3 Flash)

Tinfoil retired `glm-5-2` — the model behind the `confidential` TEE
summarization preset — **with no deprecation notice**.  It disappeared
from `/v1/models` and every request began returning HTTP 503 *"The engine
is currently overloaded"*.  Because 503 was not classified as transient,
and because `confidential` by design never falls back, every confidential
job failed outright.  Last known-good run was 2026-09-10 21:57; the break
was detected 2026-09-12.

The preset now targets **GLM-5.3 Flash** (`glm-5-3-flash`), chosen over
the higher-tier `glm-5-3` on evidence, not on vendor tier (see below).
Three independent resilience fixes ship alongside so the next silent
retirement degrades instead of failing.

### Changed

* **`confidential` preset → `glm-5-3-flash`.**  `DEFAULT_TINFOIL_MODEL`
  and `SUMMARY_PRESETS["confidential"]` in `millet/summarize.py`, the GUI
  preset dropdown label, and the docs (README, REQUIREMENTS) now
  reference GLM-5.3 Flash.  No API, CLI-flag, or config change.

  Evaluated via millet's real two-pass code path on 6 real meetings
  (3 EN, 1 DE, 1 TR, 1 narrated screen recording; 3.5 KB–165 KB
  transcripts), scored against the on-disk `glm-5-2` and
  `claude-sonnet-4-6` baselines.  Metrics only — no transcript content.

  | metric | `glm-5-2` (old) | `glm-5-3-flash` (new) | `glm-5-3` |
  |---|---|---|---|
  | format compliance | 5/5 | 5/5 | 5/5 |
  | DE/TR localized headers | yes | yes | yes |
  | coverage vs baseline | — | **equal or better** | equal or better |
  | cost / meeting (17.8K-token input) | ~$0.009 | **~$0.02** | ~$0.18 |
  | latency (same input) | ~17 s | **64 s** | 193 s |
  | reasoning tokens (same input) | — | 8,422 | 23,011 |
  | image input | no | **yes** | no |

  `glm-5-3` scores higher on Tinfoil's published intelligence metric
  (45 vs 42), but that did not translate into better meeting summaries:
  on head-to-head runs GLM-5.3 Flash matched or beat it on bullet
  coverage (77 vs 76, 22 vs 20) while spending 8.9× less and running 3×
  faster.  The extra intelligence is spent on reasoning tokens we pay
  for.  Flash also leaves 3.2× timeout headroom against the 600 s cap
  versus 1.8× — which matters on a preset that cannot fall back.

* **README cost figures corrected.**  The documented "~$0.009/meeting"
  predated reasoning-token billing and was ~2× low for Flash (and would
  have been ~20× low for `glm-5-3`).

### Fixed

* **HTTP 503 is now classified as transient** (`_is_transient_network_error`).
  A drained enclave pool is a capacity problem and retries with backoff
  (3 attempts).  A 404 *"The model does not exist"* is deliberately **not**
  retried — a retired model never comes back.

* **Sibling TEE fallback.**  When the primary model's pool is unusable
  (503/404) after exhausting retries, summarization retries once on
  `DEFAULT_TINFOIL_FALLBACK_MODEL` (`deepseek-v4-1-flash`) — a different
  model *family*, so a drained GLM pool cannot take out both.  This is
  **not** a privacy fallback: both models run inside the TEE, so the
  `confidential` contract holds exactly as before.  It is never silent —
  the switch sets `fallback_used` and the actual model is recorded in
  `.summary.meta.json`.  A DNS/router-discovery failure does *not* trigger
  it (the whole service is unreachable; a sibling would fail identically).

* **Catalog pre-flight probe** (`verify_tinfoil_model`).  Before the first
  request for a given model, millet checks `/v1/models` and logs a loud
  warning if the model is absent (retired) or carries `deprecated` /
  `deprecationDate`.  Advisory only — it never raises and never blocks
  summarization, and each model is checked at most once per process.
  This probe reproduces the `glm-5-2` failure signature exactly and would
  have caught this outage before a single job died.

### Tests

11 new tests (344 total, up from 333): 503/404 classification, pool-error
vs service-error discrimination, sibling fallback on persistent 503,
sibling *not* tried on DNS failure, sibling-family divergence guard, and
6 catalog-probe cases (retired, deprecated, healthy, network failure,
once-per-process caching).

## v0.18.0 — feat: `meetings_subdir` per-team sync target subdirectory

### Added

* **`millet/sync.py`** — teams can now sync into a custom subdirectory of
  the configured repo via the `"meetings_subdir"` key in
  `sync_config.json` (default `"meetings"`, unchanged).  A team whose
  screenrecording iteration loops share a repo with regular meetings
  keeps a separate tree — e.g. `screenrecordings/<date>_<folder>/` —
  instead of mixing into the general `meetings/` archive.  The value is
  validated like a meeting folder slug (single safe path segment), so a
  hostile or corrupt config cannot escape the clone.  4 new tests
  (default, custom, traversal rejection, end-to-end push into the custom
  tree against a local bare repo).

## v0.17.0 — feat: summary templates (`--summary-template`) + `iteration-plan` prompt

### Added

* **`millet/summarize.py`** — summary templates: a named template
  (e.g. `iteration-plan`) selects its own prompt files
  (`summarize_<template>_{system,user}.md` under `millet/prompts/`, dashes
  in the name become underscores) while presets keep selecting
  backend/model only — the two compose.  Resolution order: explicit
  `--summary-template` > `MILLET_SUMMARY_TEMPLATE` env > default meeting
  summary.  Unknown templates fall back to the default prompts; template
  names are validated (`[a-z0-9][a-z0-9_-]*`) so no path traversal is
  possible.  A template run forces the single-pass Ollama flow (the
  two-pass extract/format prompts are meeting-summary-specific) and the
  template survives fallback-backend config rebuilds.  The
  `.summary.meta.json`-style sidecar now records `"template"`.
* **`millet/prompts/summarize_iteration_plan_{system,user}.md`** — new
  `iteration-plan` template for narrated screen recordings: timestamped
  Issues (severity + suggested fix), UX Notes, and Went Well, keyed to
  `[HH:MM:SS]` transcript cues so each item points at a video frame.
  Keeps the mandatory fenced-JSON contract, so frontmatter parsing is
  unchanged.
* **`MeetingSummary.save(..., artifact=)`** — template output is written
  as `<base>.<template>.md` (plus `.<template>.meta.json` /
  `.<template>.frontmatter.json` sidecars) instead of clobbering
  `<base>.summary.md`.
* **CLI** — `--summary-template` on `millet transcribe`, `millet run`,
  and `millet label` (including `--apply-json`, the vezir subprocess
  boundary, so a session uploaded without a template can get the plan
  later via retry-summary).
* **`millet/sync.py`** — template artifacts push to the team repo under
  their own descriptive name (`<base>.iteration-plan.md` →
  `iteration-plan.md`, previously mis-mapped to `summary.md`), and all
  `*.meta.json` sidecars are excluded from push (previously only
  `.summary.meta.json`).

### Tests

* 23 new tests: template prompt resolution + fallback, config validation
  and env precedence, dispatch routing (template forces single-pass,
  survives fallback configs), artifact/sidecar naming, end-to-end
  provenance, `apply_labels` + `--apply-json` forwarding, sync
  collection.

## v0.16.1 — fix: Kimi K-series temperature clamp on the openai backend

### Fixed

* **`millet/summarize.py`** — calling a Kimi K-series reasoning model
  (`kimi-k2*`, `kimi-k3`, `kimi-for-coding`) through the generic
  OpenAI-compatible backend failed with HTTP 400 "invalid temperature:
  only 1 is allowed for this model", because the default summary
  temperature is 0.3.  New `_effective_temperature()` clamps the request
  temperature to 1.0 for `kimi-(k\d+|for-coding)` model names; all other
  models keep the configured temperature.  This unblocks the
  Claude-Max→Kimi K3 fallback path (`api.kimi.com/coding/v1`).
  8 new parametrized tests.

## v0.16.0 — feat: opt-in preset fallback (e.g. Claude Max exhausted → Kimi)

### Added

* **`millet/summarize.py`** — three operator-level env knobs that let an
  explicitly requested summarization preset degrade gracefully instead of
  hard-failing, aimed at self-hosted servers whose primary backend has
  occasional outages (e.g. a Claude Max subscription running out):

  * `MILLET_SUMMARY_PRESET_FALLBACK=1` — when a non-`confidential` preset's
    backend fails or is unavailable, continue down the fallback chain
    instead of re-raising. Every fallback attempt is logged via the
    progress callback. The **`confidential` preset never falls back** —
    a silent tinfoil→cloud fallback would defeat the privacy contract,
    so it stays fail-loud regardless of this setting.
  * `MILLET_SUMMARY_FALLBACK_ORDER` — comma-separated override of the
    fallback chain (default unchanged: `claudemax,tinfoil,openrouter,
    ollama`). This is how the generic `openai` backend joins the chain,
    e.g. `MILLET_SUMMARY_FALLBACK_ORDER=openai`. Availability gating
    still applies, so listing a backend that isn't configured is a no-op.
  * `MILLET_OPENAI_MODEL` — per-backend default model for the generic
    OpenAI-compatible backend (e.g. `kimi-k3` for
    `https://api.moonshot.ai/v1`). Previously a fallback into `openai`
    would have used the hardcoded `gpt-4o-mini` default. The chain-wide
    `MILLET_SUMMARY_MODEL` is still deliberately ignored for fallback
    backends.

* **`.summary.meta.json` sidecar** now records provenance: `"preset"`
  (the requested preset, or `null`) and `"fallback_used"` (true when the
  winning backend differs from the configured one). Downstream consumers
  (vezir) read this to display "summarized via fallback" instead of
  silently presenting a fallback summary as if the requested preset ran.

### Tests

* New `tests/test_summarize_preset_fallback.py` (24 tests): fallback-order
  parsing, `MILLET_OPENAI_MODEL` resolution (and continued
  `MILLET_SUMMARY_MODEL` isolation), the preset-fallback gate truth table,
  dispatch behavior with mocked backends (fires when enabled, raises by
  default, `confidential` never falls back, primary-unavailable path,
  all-backends-failed path, result tagging), and meta sidecar provenance.

## v0.15.1 — fix: two-pass extraction truncated on thinking-heavy models

### Fixed — correctness

* **`millet/summarize.py`** — the Ollama two-pass flow reserved only 4096
  output tokens when auto-sizing Pass 1's context window (`_dynamic_num_ctx`).
  Thinking-heavy models that emit long exhaustive extractions — notably
  `qwen3.8:27b` — exhausted the window mid-list and were silently cut off
  (`done_reason: length`), roughly halving topic/action/question coverage on
  larger meetings. Pass 1 now reserves 16384 output tokens; Pass 2 (small
  input) is unchanged. `_call_ollama_chat` gained an `output_reserve`
  parameter. Measured on a 44 KB English meeting, Pass 1 extraction grew from
  ~1.8 KB to ~5.6 KB and topic coverage rose from 8 to 20 (matching the
  cloud baseline); verified on English, Turkish, and German meetings.

## v0.15.0 — feat: sync pushes user attachments verbatim

### Added

* **`millet/sync.py`** — files a user drops into `<session>/attachments/` are
  now pushed into an `attachments/` subdirectory of the meeting folder, with
  their **names intact**. `_collect_files()` gained `_collect_attachments()`,
  which deliberately bypasses both `PUSH_SUFFIXES` and the descriptive-rename
  map: the map keys off the suffix alone, so an attached `slides.pdf` would
  land as `transcript.pdf` (or force the real transcript to keep its raw
  name), and the allowlist dropped every image, office document and video —
  most of what people attach to a meeting. The copy loop in `sync_session()`
  now creates `dest.parent` so the subdirectory prefix works; `git add`
  needed no change.

  Guards: symlinks are skipped (the collected pairs are copied into a git
  clone, and a link could point anywhere on the host), as are dotfiles and
  nested directories. `MAX_ATTACHMENTS` (50) and `MAX_ATTACHMENTS_BYTES`
  (100 MB) cap what a mis-aimed folder can push into the archive repo.
  Sessions without an `attachments/` directory collect exactly what they did
  before.

  This is the upstream half of vezir's attachment-folder workflow: vezir
  stores attachments in the session directory it hands to `millet sync`, so
  nothing further is needed on that side to get them into the team repo.

## v0.14.1 — fix: speaker relabel swap collapsed both speakers to one name

### Fixed — correctness

* **`millet/label.py`** — `apply_labels(..., regenerate_summary=False)` (the
  fast find-and-replace path used by vezir's TUI relabel) now applies the
  label map in a **single atomic pass**. The old `_replace_all` chained one
  `re.sub` per key over the same string, so a **swap**
  `{"Alaaddin":"Kemal","Kemal":"Alaaddin"}` first turned every "Alaaddin"
  into "Kemal", then the second pass re-caught those just-written "Kemal"s
  and collapsed **both** speakers to a single name — in the summary body, the
  structured frontmatter sidecar (participants/assignees/decisions), and the
  regenerated PDF's summary section. Replacement is now a single
  longest-match-first alternation regex resolved against the original map, so
  a cyclic swap `{A:B,B:A}` and a chain `{A:B,B:C}` both apply correctly. The
  transcript path (`relabel_transcript_in_memory`) was already single-pass and
  is unchanged. Focused CI suite grows to 401 passing tests.

## v0.14.0 — codebase review: correctness, security, and efficiency fixes

A broad review pass across the pipeline fixing correctness bugs (several of
which silently produced wrong output), hardening the git-based sync path, and
cutting redundant full-file audio decodes on the labeling hot path. Also a
large README accuracy sweep. Focused CI suite grows to 399 passing tests.

### Fixed — correctness

* **`millet/transcribe.py`** — `_seg_channel_ratio` now guards `start_t is
  None`. WhisperX alignment routinely emits words without timestamps; the mono
  channel-correction path had no try/except, so a single such word crashed the
  entire transcription *after* ASR completed, and on the dual-diarize path it
  silently disabled mic-bleed correction.
* **`millet/cli/label.py`, `millet/gui.py`** — voiceprint profile updates now
  match against the **pre-relabel** segments (snapshotted before `apply_labels`
  rewrites the transcript). Previously the id-keyed update matched nothing
  after relabeling and silently no-op'd while printing "Voice profiles
  updated." The GUI path also no longer races the job thread's re-save.
* **`millet/summarize.py`** — the fallback chain no longer forces the user's
  `MILLET_SUMMARY_MODEL` onto a *different* fallback backend (which guaranteed
  a failed chain); each fallback backend uses its own default model.
  Availability probes now carry the caller's `ollama_url`, so a custom Ollama
  server is no longer reported unavailable.
* **`millet/transcribe.py`** — a failed/absent language detection now yields
  `decode_lang=None` (backend auto-detects) or the operator default, instead of
  forcing an English decode of a non-English meeting.
* **`millet/gui.py`** — closed a TOCTOU in `_ensure_job_thread`/`_job_consumer`
  that could strand a queued recording when the consumer exited between
  `put()` and the `is_alive()` check.
* **`millet/cli/run.py`** — a preset summary failure now writes the PDF first
  and exits non-zero, matching `transcribe` (previously a raw traceback with no
  PDF).
* **`millet/cli/download.py`** — exits non-zero when any requested download
  fails (was exit 0).
* **`millet/cli/translate.py`** — sizes `num_ctx` to fit the full transcript
  plus translation output; the fixed `num_ctx=8192` silently truncated typical
  45–105 min meetings.
* Smaller fixes: version-fallback import in `cli/_helpers.py`; `sync.py`
  `load_sync_config` returns a copy (not the shared mutable default);
  `apply_labels` merges `speaker_labels` instead of overwriting; tempfile
  cleanup on ffmpeg failure and a clip fd leak; `parakeet` cache check keyed by
  the requested model; RTL reshaping applied before XML-escaping in PDFs
  (fixes scrambled entities in Farsi output); a rewritten trailing-JSON scan in
  `frontmatter.py` that no longer discards valid data when the body contains an
  earlier `{`.

### Fixed — security

* **`millet/sync.py`** — `git clone` now validates `repo_url`: rejects
  option-looking values (leading `-`, e.g. `--upload-pack=`) and unsafe git
  transports (`ext::` …), allows only https/ssh/git/file/local paths, and uses
  a `--` separator. `_repo_name_from_url` is sanitized so a hostile URL cannot
  traverse out of the clone base directory.

### Changed — efficiency

* **`millet/voiceprint.py`** — a per-file channel-decode cache eliminates the N
  redundant full-file ffmpeg decodes (one per speaker) in the identify / enroll
  / profile-update loops.
* **`millet/label.py`** — `extract_speaker_clip` seek-decodes only the ~8 s clip
  window (`ffmpeg -ss/-t`) instead of decoding the whole file per speaker.

### Documentation

* README accuracy sweep: corrected the `--mixdown` default (`dual-diarize`),
  the recordings directory (`~/meet-recordings`), the `meet` → `millet` command
  rename, `MEETSCRIBE_*` → `MILLET_*` env vars, the prompt path, and the
  `--asr-backend` / `--summary-backend` choice lists. Added docs for the
  `download` and `translate` commands, the Parakeet backend, dual-diarize
  tuning flags, `label` power flags, team-scoped enroll/sync, a
  configuration/env-var reference, and offline mode. Fixed the stale
  `summarize.py` module docstring.

### Tests

* New regressions: `_seg_channel_ratio` None-guard and the
  detection-unavailable language path (`tests/test_transcribe.py`); fallback
  model resolution (`tests/test_summarize_twopass.py`); trailing-JSON recovery
  despite an earlier brace line (`tests/test_frontmatter.py`).

## v0.13.3 — voiceprint many-to-one fold no longer merges two people

The `--auto` labeler could silently attribute one person's turns to a
DIFFERENT, already-named person.  Field case: a new founder ("Humphrey")
whose diarization cluster was correctly separated got labeled as an
existing, enrolled participant ("Bright"), because both spoke on the same
compressed system (Bluetooth) channel in non-overlapping turns and their
voice embeddings landed just over the auto-apply floor.

Root cause is the **Pass-2 "many-to-one" fold** in `identify_speakers`
(added in 0.12.11 to reunite ONE over-segmented person's clusters).  It
folded any still-unmatched cluster onto an already-claimed profile as long
as the cross-voice cosine cleared `MATCH_AUTOAPPLY_CONFIDENCE` (0.72) —
too low to distinguish "same person, split by mic/volume drift" from "two
similar-but-distinct voices on a lossy channel".  A new, unenrolled speaker
with no profile of their own was the exact trigger: Pass 1 left them
unmatched, then Pass 2 absorbed them into the nearest enrolled name.

### Fixed

* **`millet/voiceprint.py`** — Pass 2 now requires a STRICTER bar before
  folding onto an already-named identity: a new
  `MATCH_MANY_TO_ONE_CONFIDENCE = 0.80` absolute floor **and** a clear
  runner-up margin (`>= MATCH_AUTOAPPLY_MARGIN`, 0.15) over the cluster's
  own second-best profile.  A genuine over-segmentation of one person
  clears both trivially (the cluster looks overwhelmingly like that one
  profile); two distinct voices do not, so the unmatched cluster stays raw
  and routes to `needs_labeling` for a human — no silent mis-merge.  Pass 1
  (fresh 1:1 matching) is unchanged.

### Tests

* Three new regressions in `tests/test_voiceprint_gate.py`: a
  distinct-but-similar voice (~0.74 cosine) is NOT folded onto an enrolled
  profile; a high-confidence-but-ambiguous (small-margin) leftover is NOT
  folded; and the existing genuine over-segmentation case still folds both
  clusters into one name.

## v0.13.2 — actionable error for offline MLX model-cache miss

When `HF_HUB_OFFLINE` (or `TRANSFORMERS_OFFLINE`) is set and the MLX
Whisper model has never been cached, `mlx_whisper` bubbled a raw
`huggingface_hub.errors.LocalEntryNotFoundError` traceback and
`millet transcribe` exited 1 with no guidance.  This was hit in the
field on a macOS/Apple-Silicon client (offline env + empty MLX cache).
The transcribe path now detects the offline cache-miss and prints
actionable remediation instead of a traceback.  Focused CI suite grows
284 → 288 tests.

### Added

* **`OfflineModelMissing`** exception (`millet/transcribe.py`), plus
  `_hf_offline()` and `_mlx_model_cached()` helpers.  The MLX branch of
  `_transcribe_asr` pre-flights the Hugging Face cache when offline mode
  is enabled and raises `OfflineModelMissing` (carrying the resolved
  model repo) before invoking `mlx_whisper`, mirroring the existing
  `AlignmentModelMissing` pattern.
* **CLI guidance** (`millet/cli/transcribe.py`): `millet transcribe` now
  catches `OfflineModelMissing` — and, defensively, any
  `LocalEntryNotFoundError` by name — and prints how to warm the cache
  (`HF_HUB_OFFLINE=0 meet transcribe …` or `hf download <model>`) before
  exiting 1.  Online runs are unaffected: the pre-flight only runs when
  offline mode is set, so normal first-run downloads still work.

## v0.13.1 — `confidential` preset model migration (DeepSeek V4 Pro → GLM-5.2)

Tinfoil deprecated `deepseek-v4-pro` (the model behind the
`confidential` TEE summarization preset), effective 2026-07-03.  The
preset now targets **GLM-5.2** (`glm-5-2`), Tinfoil's recommended
successor.  Verified equal-or-better summarization quality via a
side-by-side TEE eval on two real meetings — GLM-5.2 matched or beat
DeepSeek V4 Pro on topic coverage, action-item extraction, and the
hard-to-catch **Open Questions** section (8 vs 7), with no
hallucinations.  Same Tinfoil price tier; no fallback-behavior change
(the `confidential` preset still fails loud, never silently).  Test
count unchanged (284).

### Changed

* **`confidential` preset → `glm-5-2`.**  `DEFAULT_TINFOIL_MODEL` and
  `SUMMARY_PRESETS["confidential"]` in `millet/summarize.py`, the GUI
  preset dropdown label, and the docs (README, REQUIREMENTS) now
  reference GLM-5.2 instead of DeepSeek V4 Pro.  No API, CLI-flag, or
  preset-name change — existing `--summary-preset confidential`
  invocations are unaffected.

## v0.13.0 — sync security/correctness fixes, `label --apply-json` embedder mode

Fixes from the 2026-07 ecosystem review, plus a new non-interactive
label-apply mode that gives embedders (vezir ≥ 0.11.0) a proper
subprocess boundary.  Focused CI suite grows 251 → 284 tests and now
runs on a 3.10/3.11/3.12 matrix with a pyproject↔`__init__` version
gate.

### Added

* **`millet label SESSION --apply-json FILE`** — non-interactive apply
  mode: applies a speaker→name map from JSON (`{"labels": {...}}` or a
  plain object; `-` reads stdin) with no prompting, no audio, no
  terminal dependence.  An empty map with summary regeneration enabled
  just re-runs the summary+PDF step (vezir's retry-summary path).  New
  companions: `--summary-language LANG` (additional-language summary)
  and `--update-profiles` (voiceprint update from the applied labels;
  non-fatal on failure).  This is the sanctioned boundary for vezir,
  which previously imported `millet.label` in-process and mutated
  `os.environ["HOME"]` around the call — racing every other thread in
  the server process.

### Fixed

* **`maybe_sync_session` no longer reports a failed push as synced.**
  The exception branch logged the failure and then returned the match
  anyway; GUI auto-sync treated failed pushes as success.  Now returns
  `None`.
* **Path traversal via `--meeting-type` / config `folder` closed.**
  The folder becomes a path segment in `meetings/<date>_<folder>`; a
  value like `../../foo` copied meeting artifacts *outside* the clone
  before git ever saw them.  New `_validate_folder_slug` (mirroring the
  `paths.py` team-slug convention) is enforced in `sync_session` and at
  the CLI with an early, friendly error.
* **Git credentials no longer leak into errors and logs.**  A
  `repo_url` like `https://user:TOKEN@github.com/...` appeared verbatim
  in `Command failed:` RuntimeErrors (which the vezir worker captures
  into job logs) and in the `Cloning ...` progress line.  URL userinfo
  is now redacted (`://***@`) everywhere failures are formatted.
* **A failed `git pull --rebase` no longer wedges the clone.**  A
  conflicting rebase left `.git/rebase-merge` behind, so every later
  sync died at the uncommitted-changes guard with an error that never
  mentioned the rebase.  The pull path now runs `git rebase --abort`
  before re-raising.
* **DST-aware schedule matching.**  Naive `started_at` timestamps were
  converted to UTC with the non-DST `time.timezone` offset year-round,
  shifting every summer-time meeting by an hour and silently missing
  the schedule window.  Now uses `astimezone()` (correct local offset).
* **Schedule window wraps midnight.**  A 23:40 UTC session is 20
  minutes from an `hour_utc: 0` schedule, not 1420.
* **`_collect_files` no longer silently overwrites artifacts.**  Any
  extra `.md`/`.txt`/`.pdf` in a session dir mapped onto `summary.md` /
  `transcript.txt` / `transcript.pdf` and clobbered the real artifact
  in the pushed repo.  Colliding files now keep their original names.
* **Tinfoil completion call gets a timeout.**  It was the only backend
  call omitting `timeout=config.timeout`; a stalled TLS connection to
  the enclave hung the pipeline forever — worst for the
  `confidential` preset, which by design has no fallback.
* **Speaker relabel is substring-safe.**  Summary find-and-replace now
  applies the label map longest-key-first with word-boundary regexes;
  naive dict-order replacement corrupted overlapping labels (`REMOTE`
  before `REMOTE_1` produced `Alice_1`) and matched short labels like
  `YOU` inside uppercase prose.
* **`tests/test_cli.py` had been silently broken since the 0.10.0
  `cli/` package split** (patched helpers at their pre-split location;
  all 3 tests failed with AttributeError while excluded from CI).
  Fixed and added to CI, along with `test_parakeet.py` (mocked, no
  torch needed) and the new `test_sync_core.py`.

## v0.12.16 — transcribe: fail loudly on multiple files in a directory

### Changed

* **`millet transcribe <dir>` no longer silently transcribes only the first
  audio file when several are present.**  Previously a directory with multiple
  `.wav`/`.ogg`/`.mp3` files resolved to `audio_files[0]` (alphabetically
  first), silently dropping the rest — producing a transcript that quietly
  omitted most of a meeting.  It now errors with the full file listing and
  guidance to merge the files first (or pass the specific file).  A single
  file in a directory resolves exactly as before, so existing single-recording
  session dirs (and Vezir's post-merge session dirs) are unaffected.

  This pairs with Vezir 0.9.0's multi-audio meetings, where several uploaded
  files are concatenated into one recording *before* `transcribe` runs.

## v0.12.15 — Fold spurious tiny noise speakers into the dominant speaker

### Fixed

* **A single tiny `REMOTE`/`SPEAKER_n` noise blip forced needs_labeling even
  when there was no real participant to absorb it into (A2).**  `0.12.12`'s
  `absorb_unresolved_remote()` only folds a leftover raw cluster into a
  *named* speaker.  In the common case the dominant speaker is itself an
  unmatched `SPEAKER_n`, so a 1–3 segment / few-second backchannel one-liner
  or a heavily distorted blip had no absorb target and survived as its own
  speaker — sending the whole session to manual labeling for nothing.

  `label --auto` now also runs `absorb_tiny_speakers()` after the existing
  rescue: a **tiny** unresolved raw cluster (≤ `TINY_SPEAKER_MAX_SECONDS`
  = 5.0 s of speech **and** ≤ `TINY_SPEAKER_MAX_SEGMENTS` = 3 segments) is
  folded into the speaker with the most overlapping/total speech time — even
  when that dominant speaker is itself still raw.  Tiny clusters are never
  folded into another tiny cluster, and if *every* speaker is tiny nothing is
  folded (the whole recording is noise — the caller decides its fate).  This
  runs in auto mode even when no confident voiceprint match was applied.

## v0.12.14 — Discover MP3 audio in session directories (release fix)

Re-release of 0.12.13: the 0.12.13 tag built the wrong version because the
static `version` in `pyproject.toml` was not bumped (only `__init__.py` was),
so the release workflow rebuilt 0.12.12 and PyPI rejected it as a duplicate.
No code changes versus 0.12.13 — the MP3 discovery feature below is unchanged.

## v0.12.13 — Discover MP3 audio in session directories

### Added

* **MP3 session-dir discovery.**  `millet transcribe <session_dir>` and
  `_find_session_files` (used by `label`, `ingest`, `enroll`, `sync`) now
  discover `*.mp3` audio, in addition to `*.wav` and `*.ogg`.  Preference
  order is WAV → OGG → MP3; the discovered audio is still surfaced under the
  `"wav"` key for backward compatibility.  Decoding already worked for any
  ffmpeg-readable format (`whisperx.load_audio` is ffmpeg-backed); only the
  directory-discovery globs were restricted.  An explicit `millet transcribe
  path/to/file.mp3` already worked and is unchanged.

## v0.12.12 — Rescue the leftover REMOTE bucket; strip mis-clustered backchannel

### Fixed

* **A single unidentified `REMOTE` forced needs_labeling even when every real
  participant was matched (A1).**  The dual-diarize path creates a literal
  `REMOTE` bucket — system segments pyannote left unassigned, plus mic-bleed
  segments with no temporally-overlapping diarized remote — *after* cluster
  consolidation runs, so it never merges, and its mixed/thin backchannel audio
  rarely voiceprint-matches.  That one leftover then sent the whole session to
  manual labeling (and, once a human named it, produced a duplicate speaker).

  `label --auto` now runs `absorb_unresolved_remote()` after applying confident
  matches: a **small** unresolved raw cluster (≤ 30 s of speech and ≤ 25
  segments) is absorbed into the *named* speaker it overlaps most in time.
  Large unknowns are still left raw for human review.

* **Backchannel mis-attributed to a late-joining speaker (B1, issue #2).**
  pyannote sometimes lumps a few early sub-1.5 s utterances ("yeah", "ok") into
  a cluster whose real speaker only appears much later (e.g. someone who joins
  25 min in); voiceprint then names the whole cluster, so those early fragments
  inherit the wrong name.  A new guard
  (`TranscriptionConfig.strip_isolated_backchannel`, default on) reassigns
  fragments that are both below `backchannel_max_seconds` (1.5 s) **and** more
  than `cluster_isolation_gap_seconds` (5 min) from the nearest real
  (≥ `MIN_SEGMENT_DURATION`) segment of their own cluster to the generic
  `REMOTE` bucket — which the A1 rescue then re-homes.  Conservative: only
  touches non-embeddable, far-isolated fragments.  See
  `docs/spike-diarization-cluster-bleed.md` for the analysis.

## v0.12.11 — One speaker per person: many-to-one voiceprint matching + speaker de-dup

### Fixed

* **One physical person came back as several speakers, forcing a
  needs_labeling hop and "Destiny, Destiny, Destiny" in the notes.**
  Diarization (pyannote) routinely over-segments a single voice into multiple
  clusters — from volume/mic changes, cross-channel bleed (the scribe's voice
  on both the mic and system channels in the dual-diarize path), or peeled-off
  backchannel utterances.  Voiceprint auto-identification used **greedy 1:1
  matching**: once a profile (say "Destiny") was claimed by its best-scoring
  cluster, every *other* Destiny cluster was barred from that name and left as
  a raw `SPEAKER_n`, which (a) routed the session to manual labeling and
  (b) — once a human named each cluster — produced duplicate same-name speaker
  entries that nothing collapsed.

  Two changes fix this:

  1. **Many-to-one matching** (`voiceprint.identify_speakers`): after the
     greedy 1:1 pass assigns each profile to its strongest cluster, a second
     pass lets an already-claimed profile **also** claim additional clusters —
     but only when that profile is the cluster's own top match **and** the
     similarity is confident on its own (clears `MATCH_AUTOAPPLY_CONFIDENCE`).
     Ambiguous/weak leftover clusters are still left raw for human review, so
     this never folds a noisy phantom onto a real person.

  2. **Speaker de-duplication** (`label.relabel_transcript_in_memory`, used by
     `apply_labels` and the GUI): speakers that resolve to the same id collapse
     into a single entry (segments already point at the merged id).  This also
     runs for an **empty** label_map, so an older transcript that already
     carries duplicate names is de-duped in place (enables backfill).

  Net effect: a meeting with one over-segmented known speaker now auto-resolves
  to a single speaker and completes without manual intervention.

### CI / release

* New `release.yml`: tag-triggered (`v*`) build + publish to PyPI via Trusted
  Publishing (OIDC), matching vezir and millet-record.  (Requires the
  `millet-pipeline` PyPI project to register this workflow as a trusted
  publisher, environment `pypi`.)
* `test_voiceprint_gate.py` added to the focused CI test set.

## v0.12.10 — Robust transcript-JSON discovery in `label`

### Fixed

* **`millet label` could pick the wrong JSON and crash with
  `KeyError: 'segments'`.**  `_find_session_files` chose the transcript via
  "last sorted `*.json` wins" with an incomplete exclusion list: it did not
  exclude `*.frontmatter.json`, nor a **bare `session.json`** (vezir writes
  one on pull; its name lacks the `.session.` substring the old check looked
  for).  On a vezir-pulled directory (`transcript.json` + `session.json` +
  `frontmatter.json`), `session.json` sorts last and was selected as the
  transcript → crash.  Selection is now deterministic: frontmatter,
  translation, summary-meta, auto-id, and both session-metadata forms are
  excluded, then the canonical `transcript.json` is preferred, then
  `<dirname>.json`, then the first remaining candidate.  A bare `session.json`
  is also routed to `files["session"]`.

  Note: the vezir **server worker** was never affected (it transcribes/labels
  with ULID-stem names where `<id>.frontmatter.json` sorts before `<id>.json`,
  and never writes a bare `session.json`); this only bit manual `millet label`
  runs on pulled/client directories.

## v0.12.9 — Force the default-language-biased decode language per channel

### Fixed

* **English audio transcribed as Spanish (and other per-channel language
  drift).**  In the dual-channel / dual-diarize paths each channel was
  transcribed with `language=None`, so whisperx auto-detected the decode
  language from only the first ~30 s of that channel.  When that guess was
  wrong (e.g. "es" on an English meeting), the entire channel's English
  speech was decoded with the wrong tokenizer → Spanish-looking
  hallucinations.  The `--default-language` bias previously only relabeled
  the transcript *metadata* language; it never reached the ASR, so it could
  not prevent the mistranscription.

  Each channel now **detects → applies the default-language bias → forces
  that resolved language into the ASR decode** (`model.transcribe(...,
  language=...)`) via the new `_resolve_channel_language` helper.  A
  low-confidence minority detection is overridden by the team/operator
  default (`default_language`, gated by `default_language_override_confidence`,
  default 0.70) *before* decoding, so single-language teams no longer drift.
  Alignment uses the same decode language.  The mono path got the same
  detect→bias→force treatment.  Applies to whisperx (`--language auto`);
  an explicit `--language <code>` still forces that language as before.

## v0.12.8 — Correct no-headphones mic crosstalk in the dual-diarize path

### Fixed

* **Remote speech bled into the mic channel was attributed to the local
  speaker.**  When a meeting is recorded WITHOUT headphones, the remote
  participants' audio played through the room speakers leaks into the
  microphone (left) channel.  The default `dual-diarize` path labels the
  entire mic channel as the local speaker (`YOU`), so a large share of those
  segments were echoes of remote speech misattributed to the local speaker.
  New `_correct_mic_bleed_segments` (invoked from `_transcribe_dual_diarize`,
  gated on `channel_correct`, on by default) detects bled segments by
  per-segment / per-word channel energy and near-duplicate echo matching, and
  **reassigns them to the diarized remote speaker that overlaps them in time**
  (or a generic `REMOTE` fallback), dropping empty/punctuation-only mic
  artifacts.  Genuine local speech (mic-dominant, no remote twin) is left as
  `YOU`, so the correction is conservative and never invents or merges named
  remote speakers.  Disable with `--no-channel-correct`.

## v0.12.7 — Single-source path: keep diarized in-room speakers

### Fixed

* **Second collapse in the single-source fallback.**  v0.12.6 routed in-room
  recordings to the mono path, which correctly diarized two in-room speakers —
  but the mono path then remapped the diarized speakers onto YOU/REMOTE by
  channel energy.  On dual-mono audio every speaker is equally "mic-dominant",
  so that remap collapsed the genuine speakers back into one.  The mono path
  now **skips** the channel-energy YOU/REMOTE relabeling (and the channel
  correction) when the recording was detected as single-source, keeping the
  pyannote diarization result (`SPEAKER_00`/`SPEAKER_01`/…) so voiceprint
  naming can label each in-room speaker.

## v0.12.6 — Fix in-room multi-speaker collapse in the dual-diarize path

### Fixed

* **Multiple in-room speakers on the mic channel collapsed into one.**  The
  default `dual-diarize` path assumes the mic (left) channel carries a single
  local speaker (labeled `YOU`) and only diarizes the system (right) channel.
  For an **in-room recording** — several people sharing one mic, with the
  system channel silent or merely a duplicate of the mic — every mic speaker
  was therefore merged into a single speaker.  The pipeline now detects this
  single-source case and falls back to the mono path (mix down + diarize the
  combined signal), which splits the in-room speakers correctly.  Genuine
  remote calls (an active, distinct system channel) are unaffected and keep
  using `dual-diarize`.

### Added

* `--single-source-fallback` / `--no-single-source-fallback` (default on) and
  `TranscriptionConfig.single_source_fallback`.  Detection thresholds are
  tunable via `system_inactive_rms_ratio` (default 0.10) and
  `channel_duplicate_corr` (default 0.98).  New helpers
  `_is_single_source_stereo` + `_load_stereo_int16`.

### Notes

* Single-source is detected when the system channel's active-sample RMS is
  below `system_inactive_rms_ratio` of the mic channel's, OR the two channels'
  Pearson correlation is at/above `channel_duplicate_corr`.  Conservative on
  analysis failure (keeps `dual-diarize`), so remote calls are never
  mis-routed.

## v0.12.5 — Title-aware schedule matching + collision guard for sync

### Fixed

* **Title-aware schedule matching.**  `detect_meeting_type` now considers the
  session `title` (when present in `*.session.json`).  A *titled* session only
  auto-matches a scheduled meeting whose `name`/`folder` slug equals the
  title's slug; otherwise it returns `None` so the caller files it under its
  own folder.  This stops an ad-hoc meeting recorded *inside* a schedule
  window (e.g. a "post-scrum" at 09:03 inside the 06:30–09:30 standup window)
  from being misfiled as the scheduled meeting.  **Untitled sessions keep the
  prior pure time-window behavior** (back-compat — existing scheduled-meeting
  workflows are unchanged).

* **Collision guard: never silently overwrite a different meeting.**
  `sync_session` writes a small local-only `.session-id` marker into each
  synced folder and, before reusing a dated folder, checks it.  If an existing
  folder belongs to a *different* session, the new meeting is filed into a
  disambiguated folder (`<folder>-<sessionid-suffix>`) instead of clobbering
  the existing one.  Previously two meetings that resolved to the same folder
  (e.g. two ad-hoc meetings in one schedule window) overwrote each other.

### Notes

* The `.session-id` marker is kept strictly local: it is registered in the
  clone's `.git/info/exclude`, so it is never committed/pushed and never trips
  the "uncommitted changes" sync guard.
* Pairs with vezir v0.7.16, which injects the session `title` into
  `*.session.json` (so the title-aware matching above can engage) and adds an
  explicit "sync as" folder override.

## v0.12.4 — Robust language detection + sync exit-code

### Changed

* **Multi-window language detection.**  whisperx detected language from only
  the first ~30 s of each channel, so a misleading opener (e.g. an opening
  "Gracias") mislabeled an English meeting as Spanish even after the
  dominant-channel fix.  Now samples N windows across each channel via
  faster-whisper's `detect_language(language_detection_segments=N)`
  (whisperx backend; `--language-detection-segments`, default 6).
* **Soft default-language bias.**  `--default-language <lang>` keeps the team
  default unless a channel confidently detects another language
  (≥ `default_language_override_confidence`, default 0.70); fed into
  dominant-channel selection.

### Fixed

* **`millet sync` exit code.**  `cli/sync.py` now raises `SystemExit(1)` when
  any session fails (e.g. git push rejected) instead of exiting 0 — callers
  no longer have to scrape the log to notice a failed sync.

## v0.12.3 — Summary language from the dominant channel + per-language summaries

### Fixed

* **Summary/transcript language now follows the channel with the most
  speech** (`_dominant_channel_language`; mic wins exact ties).  Previously
  the dual-channel paths took the language from the mic channel only, so a
  local speaker's minority-language asides made the whole summary that
  language.  Each channel is also word-aligned with its OWN detected language
  (`_align_channel`) instead of sharing the mic's model.

### Added

* `apply_labels` gains `summary_language`: regenerate the summary in a chosen
  language and save it as an ADDITIONAL `<base>.summary.<lang>.md` (with
  suffixed meta/frontmatter sidecars), preserving the primary auto-detected
  summary.  `MeetingSummary.save` gains `lang_suffix`.
* `sync`: `<base>.summary.<lang>.md` syncs as a distinct `summary.<lang>.md`;
  `.frontmatter.json` is excluded (also fixes a latent collision).

## v0.12.2 — Suppress phantom remote speakers in dual-diarize

### Fixed

* pyannote can over-segment a single remote stream into multiple clusters
  (peeling short backchannel "yeah/cool" off the main speaker into a
  phantom), which voiceprint matching then mis-named from a weak,
  barely-over-threshold match.
  * **Voiceprint auto-apply gate**: a match at/above `MATCH_THRESHOLD` is
    applied only if it has enough embeddable speech AND is unambiguous
    (strong absolute confidence OR a clear margin over the runner-up).
    `SpeakerMatch` gains `evidence_seconds` + `margin`; weak/ambiguous
    matches stay raw and route to `needs_labeling` instead of confidently
    mislabeling.  The sidecar records only applied matches.
  * **Remote-cluster consolidation** (dual-diarize): merge same-speaker
    clusters (cosine ≥ `cluster_merge_similarity`) and absorb thin clusters
    (< `cluster_min_speech_seconds`) into the dominant remote.

## v0.12.1 — Fix auto-label discarding matches in non-interactive runs

### Fixed

* **`label --auto` aborted (and discarded all matches) when any speaker was
  unmatched in a non-interactive context.**  After auto-applying confident
  voiceprint matches, the command unconditionally prompted for unrecognized
  speakers; in a worker/batch context (no TTY) `click.prompt` hit EOF and
  raised `Abort`, so the already-collected confident matches were never
  written.  Now: when stdin is not a TTY, skip the prompt — apply the
  auto-matches and leave unmatched speakers as their raw `SPEAKER_N` ids
  (the documented "unknowns remain as REMOTE_N" behavior).  This is the bug
  that left fully-recognizable meetings stuck in `needs_labeling` with raw
  speaker ids.

### Added

* **`*.autoid.json` sidecar** written by `label --auto`: records each
  auto-matched speaker's name + confidence, keyed by the final transcript
  speaker id.  Lets downstream UIs (vezir's labeling screen) pre-fill
  recognized names and show match confidence.  Excluded from `millet sync`
  pushes and from transcript-file resolution.

## v0.12.0 — Dual-diarize: per-channel transcription + remote speaker diarization

New default for stereo recordings (`--mixdown dual-diarize`).  Transcribes the
mic and system channels **separately** — Kemal's mic stream is captured as a
continuous "YOU" source immune to overlap with remote speakers — then runs
**pyannote diarization on the system channel only** to split distinct remote
speakers (Openoms, Jonas, Max, …).  Downstream voiceprint naming maps each
SPEAKER_N to a real name, exactly as before.

This eliminates the **overlap-fragmentation** problem of the mono path, where
WhisperX's word timestamps during overlapping speech caused words to flicker
between speakers ("This year" → Openoms, "they" → Kemal, "rented the" →
Openoms, "whole island" → Kemal — when Kemal said the entire sentence).
Overlapping segments from different channels are **preserved**, not
serialized.

### Added

* **`--mixdown dual-diarize`** (now the **default** for stereo): dual-channel
  ASR + system-channel diarization.  `mono` and `dual` remain available via
  the flag.
* **Channel-energy correction** (mono path, `--channel-correct`): per-segment
  and per-word mic/(mic+sys) RMS reassignment for turn-boundary leaks.  On by
  default when `--mixdown mono` is used; includes `--channel-correct-margin`
  (default 0.30) for tuning.  Mixed segments are split at word-speaker
  boundaries.  11 new tests.
* **DNS-retry hardening** for `millet sync` git operations (clone/pull/push):
  transient DNS failures auto-retry up to 5× with backoff instead of aborting.

### Notes

* 2× ASR cost (both channels transcribed); still fast on GPU — a 62-minute
  meeting completes in ~3 minutes on a 3090.
* Diarization on the isolated system channel produces a **cleaner speaker
  signal** (no local mic bleed), improving remote-speaker clustering.
* Minor over-segmentation of single-remote meetings (pyannote may split one
  person into 2–3 clusters); voiceprint matching merges them.
* Validated on DEVSTANDUP (5 speakers), LUKAS_2 (2 speakers), AB_BOARD (4
  speakers, .ogg): the known overlap-fragmentation is eliminated and all
  distinct remote speakers are preserved.

## v0.11.0 — Parakeet ASR backend (opt-in, English, ONNX)

Adds a third ASR backend alongside `whisperx` and `mlx`: **NVIDIA Parakeet
TDT** via [onnx-asr](https://github.com/istupakov/onnx-asr) (ONNX Runtime,
pure-Python — no extra torch/transformers).  Opt-in only; `auto` selection is
unchanged.  Intended for benchmarking against the WhisperX default on English
meetings before any default change.

### Added

* **`--asr-backend parakeet`** (`millet transcribe`).  Uses the English
  `nemo-parakeet-tdt-0.6b-v2` model by default; override with
  `--parakeet-model`.  Long audio is automatically chunked through onnx-asr's
  Silero VAD adapter (Parakeet's per-utterance limit is ~20-30 s), with
  global timestamps stitched back.  Emits the same WhisperX-shaped result
  dict the other backends produce, so alignment / diarization / dual-channel
  labeling downstream are unchanged.
* **Alignment toggle for Parakeet** — `--parakeet-keep-alignment`:
  * default (config "B"): trust Parakeet's native VAD-segment timestamps
    (skips WhisperX wav2vec2 alignment; faster).
  * with the flag (config "C"): run WhisperX alignment on top of Parakeet
    text for word-level timestamps.
  This exists so the ASR benchmark can measure B vs C and pick a default.
* **`millet download parakeet`** — explicit, lazy fetch of the Parakeet +
  Silero VAD ONNX weights into the HF cache (mirrors `millet download <lang>`
  for alignment models).  Never auto-downloads inside `transcribe`.
* **`millet-pipeline[parakeet]` optional extra** — pulls `onnx-asr[hub]`
  (numpy + onnxruntime + huggingface-hub only).  For CUDA, install
  `onnxruntime-gpu` on the GPU host.
* **`scripts/bench_asr.py`** — benchmark harness comparing configs A
  (whisperx) / B (parakeet native ts) / C (parakeet + alignment): reports
  RTFx, wall time, segment/speaker counts, and dumps transcripts for
  side-by-side human comparison.  WER-vs-stored-transcripts is intentionally
  not computed (those are themselves Whisper output).
* `millet/parakeet.py` module + `tests/test_parakeet.py` (12 tests: contract
  shape, config B/C wiring, backend validation, dispatch, availability guard).

### Notes

* `auto` deliberately never selects Parakeet; it remains opt-in pending
  benchmark data (English-only, separate timestamp behavior).
* Parakeet v2 is English-only; multilingual v3 (`nemo-parakeet-tdt-0.6b-v3`)
  is reachable via `--parakeet-model` but is not the benchmark target.

## v0.10.0 — tech-debt sweep: import-bug fix, CI, ruff, cli.py split

Code-health release.  No user-facing behavior change; minor bump because
the internal `cli.py` module became a `cli/` package.

### Fixed

* **Latent clean-install crash**: `millet/{capture,audio,utils,languages}.py`
  shims imported `from meet_record.*` (the pre-rename package).  On a
  clean `pip install millet-pipeline` (which depends on `millet-record`,
  providing `millet_record`) every shim raised
  `ModuleNotFoundError: meet_record`.  It only worked where the legacy
  `meetscribe-record` happened to be co-installed.  Now they import
  `from millet_record.*`.  `voiceprint.py` likewise switched its lazy
  `from meet.* import` to `from millet.*` (and dropped two unused
  imports).  New `tests/test_shim_imports.py` guards against regression.

### Changed

* **`cli.py` (1929 lines) split into a `cli/` package**: one module per
  command (`transcribe`, `run`, `download`, `translate`, `label`,
  `enroll`, `sync`, `gui`, `ingest`) + a shared `cli/_helpers.py`, with
  `cli/__init__.py` defining the `main` group and re-exporting every
  command symbol so the `millet.subcommands` / `meet.subcommands` entry
  points (`millet.cli:transcribe` etc.) keep resolving.
* **CI fixed**: the workflow linted/tested dead `meet/` paths (package is
  `millet/`) and installed `meetscribe-record` (old PyPI name).  Now
  lints the full `millet/` + `tests/` tree, installs `millet-record`,
  and runs the no-torch/no-GTK test suites.
* **Ruff config added** (`[tool.ruff]`, mirroring vezir's
  `E,F,W,I,B,UP,RUF` ruleset) and the tree cleaned up (≈120 findings:
  unused imports, `raise ... from`, import sorting, etc.).
* Legacy `meet.*` references in the test suite rewritten to `millet.*`
  (recovered ~29 previously-erroring tests).

### Notes

* `test_gui.py` (needs GTK/`gi`) and the torch-device-detection cases in
  `test_transcribe.py` / `test_cli.py` still require a GPU/display
  environment; they are excluded from the CI no-torch run.

## v0.9.2 — resilient Tinfoil (confidential) summarization

The `confidential` summary preset (Tinfoil TEE backend) could hard-fail
on a single transient DNS/network blip: the Tinfoil SDK does a network
fetch at client construction (router discovery,
`GET https://atc.tinfoil.sh/routers`) and the client init was outside
the retry path, so one flaky lookup aborted the whole summarization.

### Fixed

* **`_summarize_tinfoil` now retries transient network/DNS errors** with
  exponential backoff (3 attempts, ~2s/4s/8s).  Both the client
  construction (router discovery) and the completion call are inside the
  retry.  Genuine auth/model errors still fail fast (no retry), and a
  persistent outage surfaces a clear "Tinfoil TEE unreachable after N
  attempts" message naming the likely cause.
* New `_is_transient_network_error()` classifier walks the exception
  cause chain (the SDK wraps `URLError` in `ValueError("Failed to fetch
  router addresses…")`) and matches common DNS/connection failure text.

## v0.9.1 — team-aware paths (`--team`) + millet env-var aliases

Adds an optional team dimension so a scribe recording for multiple
teams can keep voiceprints, sync config, and recordings separated
locally, and introduces `MILLET_*` env-var names alongside the legacy
`MEETSCRIBE_*` / `MEET_*` ones.  Fully back-compatible: with no
`--team` flag and the old env vars, behavior is unchanged.

### Added

- **`millet.paths` module** — central, call-time path resolver with an
  optional `team` argument:
  - `profiles_path(team)` → `~/.config/meet/<team>/speaker_profiles.json`
  - `sync_config_path(team)` → `~/.config/meet/<team>/sync_config.json`
  - `recordings_dir(team)` → `~/meet-recordings/<team>/`
  - Team slugs validated (`[a-z][a-z0-9-]{2,31}`) so a bad value can
    never escape its directory.
- **`--team <slug>` flag** on `millet sync`, `millet enroll`,
  `millet label`.
  - `sync --team` reads `~/.config/meet/<team>/sync_config.json` and
    clones into a team-namespaced dir
    (`~/.local/share/meet/<team>/<repo>/`), so two teams can sync to
    different repos that share a name without colliding.
  - `enroll`/`label --team` use the team's voiceprint DB for matching
    and profile updates.
- **`MILLET_*` environment variables** with one-release fallback to the
  legacy names (one-time `DeprecationWarning` on legacy use), via
  `millet.paths.getenv_renamed`:
  - `MILLET_SUMMARY_BACKEND` ← `MEETSCRIBE_SUMMARY_BACKEND`
  - `MILLET_SUMMARY_MODEL` ← `MEETSCRIBE_SUMMARY_MODEL`
  - `MILLET_SUMMARY_PRESET` ← `MEETSCRIBE_SUMMARY_PRESET`
  - `MILLET_OLLAMA_SINGLEPASS` ← `MEETSCRIBE_OLLAMA_SINGLEPASS`
  - `MILLET_OPENAI_BASE_URL` ← `MEETSCRIBE_OPENAI_BASE_URL`
  - `MILLET_OPENAI_API_KEY` ← `MEETSCRIBE_OPENAI_API_KEY`
  - `MILLET_PROFILES_PATH` ← `MEET_PROFILES_PATH`
  - `MILLET_CONFIG_DIR` ← `MEET_CONFIG_DIR`
  - `MILLET_RECORDINGS_DIR` ← `MEET_RECORDINGS_DIR`

### Changed

- `millet.sync` config accessors and entry points
  (`load_sync_config`, `save_sync_config`, `is_sync_configured`,
  `detect_meeting_type`, `check_sync_candidate`, `sync_session`,
  `maybe_sync_session`, `ensure_repo_cloned`) now accept an optional
  `team` (and `config_path` override).  Teamless callers unaffected.
- `millet.voiceprint._default_profiles_path` now delegates to
  `millet.paths`.

### Notes

- On-disk paths remain `~/.config/meet/` and `~/meet-recordings/` (the
  `meet` spelling); only the env-var names gained `MILLET_*` aliases.
  Migrating the on-disk paths is deferred to a future release with a
  data-move step.

### Tests

- `tests/test_paths.py` (NEW), `tests/test_sync_team.py` (NEW).

## v0.9.0 — 2026-05-24 — rename to `millet-pipeline`

The package formerly known as `meetscribe-offline` is now
**`millet-pipeline`**.  Named after the Ottoman *millet system* — the
legal framework of communal autonomy that, in 1493, made it possible
for two Sephardic Jewish brothers to establish Istanbul's first
printing press, just one year after their expulsion from Spain.  Part
of the [vezir](https://github.com/pretyflaco/vezir) ecosystem.

This release is **all rename, no feature change.**  Functional
behavior is identical to 0.8.3.  Companion: `millet-record 0.4.0`
(formerly `meetscribe-record`); the `meet` console script is retained
as a deprecation-warning alias of the new `millet` command for two
minor versions before removal.

### Why not `millet`?

The bare `millet` name on PyPI is held by an unrelated dialogue-
framework package (last upload 2021).  We've opened a PEP 541 takeover
petition; if it succeeds in the future, a simpler `millet` name may
follow in a later major release.  `-pipeline` is more honest than
`-offline` (which was no longer accurate — the Tinfoil TEE and
OpenRouter summary backends are network-attached).

### Migration

```bash
# Out:
pip uninstall meetscribe-offline meetscribe-record
# In:
pip install millet-pipeline           # full pipeline (pulls millet-record transitively)
# or just the capture-only sibling:
pip install millet-record
```

CLI: `meet` keeps working in `millet-record 0.4.0` and `0.5.0`, with a
deprecation warning forwarding to `millet`.  Removed in `millet-record
0.6.0`.

### Changed

* **Distribution name**: `meetscribe-offline` → `millet-pipeline`.
* **Import name**: `from meet.X` → `from millet.X` (e.g. `from
  millet.label import apply_labels`, `from millet.transcribe import
  TranscriptionConfig`, `from millet.summarize import summarize`).
* **Entry-point group**: `meet.subcommands` → `millet.subcommands`.
  The legacy `meet.subcommands` group is also published for one
  deprecation cycle so a transitional `meet` CLI from `millet-record
  < 0.4.0` continues to load these subcommands.
* **Compatibility shims** (`millet/audio.py`, `millet/utils.py`,
  `millet/languages.py`, `millet/capture.py`) re-export from
  `millet_record.*` (formerly `meet_record.*`).  Both import paths
  work via the alias module shipped in `millet-record 0.4.0`.
* **CLI prog_name**: `meet (meetscribe-offline)` →
  `millet (millet-pipeline)` in `--version` output.
* **PDF model attribution line**: "AI transcription (meetscribe)" →
  "AI transcription (millet)".

### Compatibility

* **vezir 0.4.0** is the first vezir release pinning `millet-pipeline
  >= 0.9.0`.  Vezir 0.3.x continues to work against `meetscribe-offline
  0.8.3` (the last release under the old name); both old names stay on
  PyPI as historical artifacts.
* **No wire-format changes.**  Sessions produced by 0.8.3 are read
  unchanged.  All artifact paths and formats unchanged.

### What did NOT change

* Module-internal class names, function names, function signatures.
* `~/.config/meet/speaker_profiles.json` and other runtime paths
  (these live in `vezir-data` for the vezir-managed deployments; the
  standalone CLI's user-config path is the same).
* The `meet-record-mac` Swift sidecar binary name.  (Renaming would
  require macOS code-signing bundle-path changes.)

### Reserved names (future submodules)

`hattat`, `nahmias`, `basmahane`, `amire` — see the project's
RENAMING handoff for the reasoning behind reserving each.

---

## v0.8.3 — 2026-05-23

### Fixed

- **`apply_labels()` now re-raises summary failures when `summary_preset`
  is set.**  Previously, all summary exceptions were silently caught and
  `regenerate_summary` was set to `False`, making it impossible for
  callers to detect that the summary was not generated.  When
  `summary_preset` is explicitly provided (e.g. by vezir's retry-summary
  flow), the exception now propagates so the caller can surface it.
  Callers that don't pass `summary_preset` (the default `meet label`
  CLI flow) keep the existing silent-skip behavior.

## v0.8.2 — 2026-05-23

### Fixed

- **Voiceprint profile path was frozen at module-import time.**
  `PROFILES_PATH = Path.home() / ".config/meet/speaker_profiles.json"`
  was computed once at `import meet.voiceprint` time.  When an embedder
  (such as vezir) overrode `$HOME` after import and called voiceprint
  functions in-process, reads and writes went to the original home
  directory instead of the overridden one.  This caused the central
  voiceprint DB to go stale while a shadow copy at the real `$HOME`
  accumulated updates silently.

### Changed

- **All public voiceprint functions accept an optional `profiles_path`
  keyword argument**: `load_profiles()`, `save_profiles()`,
  `identify_speakers()`, `update_profiles_from_confirmed_labels()`,
  `enroll_session()`.  When `None` (the default), the path is resolved
  at **call time** (not import time) via `_default_profiles_path()`.
  Existing callers are unaffected — the default behavior matches 0.8.1.

- **New `MEET_PROFILES_PATH` environment variable** overrides the
  default profile path.  Useful for shell users or service managers
  that want a non-default location without modifying code.

- **`PROFILES_PATH` module constant** is now a lazy attribute via
  PEP 562 `__getattr__`.  `from meet.voiceprint import PROFILES_PATH`
  still works and resolves at access time (respecting current `$HOME`
  and `$MEET_PROFILES_PATH`).

## v0.8.1 — 2026-05-22

### Fixes

- **`apply_labels()` accepts `summary_preset` kwarg.**  `meet label`'s
  CLI was passing `summary_preset=...` to `apply_labels()` since the
  preset feature landed in 0.8.0, but the parameter wasn't in the
  function signature.  Every `meet label --auto` that found a
  confident voiceprint match crashed with `TypeError`.  The bug was
  latent because the auto-label path itself was failing on the
  `MIN_SEGMENT_RMS` regression below; with both fixed, the auto-label
  contract works end-to-end.  Regression test added.
- **Voiceprint matcher: lower per-segment RMS floor 0.005 → 0.0015.**
  The 0.005 floor introduced as a silence guard turned out to be too
  aggressive for real-world recordings with quiet mic gain
  (mean ~-49 dBFS, peaks at -4 dBFS).  Per-segment RMS clustered in
  [0.0016, 0.0024] — all silently skipped (`log.debug`) → no embedding
  → no match → every session needed manual labeling against an already-
  populated profile DB.  New floor at ~ -56 dBFS is well above true
  silence (-90) but below the lowest validated mic-recording RMS.
  Skip log promoted to `log.info`; a `log.warning` fires when ≥80% of
  segments were skipped.
- **Relabeled PDF preserves model attribution + backend.**
  `apply_labels(regenerate_summary=False)` used to hardcode
  `model="(relabeled)"` and lose the original `backend`, breaking the
  CONFIDENTIAL watermark on relabel-driven PDF regeneration.  Now reads
  `.summary.meta.json` and threads model + backend through.

## v0.8.0 — 2026-05-22

### New features

- **Summarization preset selector** — `--summary-preset
  {high-quality,confidential,alternative}` on `transcribe`, `run`,
  `label`, `gui`, and `ingest`.  Resolves to a concrete
  `(backend, model)` pair via `meet.summarize.SUMMARY_PRESETS`.  GUI
  gains a preset dropdown above the Advanced panel.
- **Tinfoil TEE backend** — `pip install 'meetscribe-offline[tee]'`
  pulls in the `tinfoil` SDK.  Inference runs inside a hardware-
  attested TEE (AMD SEV-SNP or Intel TDX); prompts are not visible to
  the model provider or the cloud operator.  ~$0.009 per meeting,
  ~66 s for a 30-min recording on DeepSeek V4 Pro.  Set
  `TINFOIL_API_KEY` or drop a key file at `~/models/tinfoil/tinfoil.txt`.
- **CONFIDENTIAL PDF watermark** — sessions summarized via the
  `tinfoil` backend get a red CONFIDENTIAL watermark on every page
  header and footer.  Auto-detected from `summary.backend`.

### Behavior changes

- **Preset guard semantics**: when a preset is explicitly chosen,
  summarization failures are NOT silently absorbed into the fallback
  chain.  A silent tinfoil → claudemax fallback would defeat the
  entire point of the Confidential preset.  Preset failures now raise
  `RuntimeError` from `summarize()`; the CLI catches it, finishes
  writing the transcript artifact, then exits non-zero so downstream
  tooling (vezir, CI) can detect the partial failure.
- **Default Ollama model** changed from `gpt-oss:20b` to `qwen3.5:9b`
  (better quality and no hallucinations on unseen transcripts).
- **Fallback chain** now: `claudemax → tinfoil → openrouter → ollama`
  (was `claudemax → openrouter → ollama`).
- **PDF: strip trailing JSON block from summary body**, so the
  rendered PDF shows clean Markdown rather than the structured data
  block contract introduced in 0.7.0.

### Internals

- New `tee` optional-dependency group in `pyproject.toml`.
- `SUMMARY_PRESETS` table at `meet/summarize.py:77`.
- `_resolve_tinfoil_api_key()` helper reads from env var or fallback
  file at `~/models/tinfoil/tinfoil.txt`.
- PDF generator auto-enables `confidential=True` when
  `summary.backend in ("tinfoil", "tinfoil-tee")`.

## v0.7.2 — 2026-05-17

### Fixes

- **`meet download <lang>` no longer fails with
  `CERTIFICATE_VERIFY_FAILED` on python.org Python builds (macOS)**
  (reported by @patternn in the M8 retrospective). `torchaudio`'s
  alignment-model fetcher uses raw `urllib`, which inherits the
  interpreter's default SSL context — empty on python.org Python
  installs that ship without a CA bundle. `meet/__init__.py` now
  injects `certifi`'s CA bundle as `SSL_CERT_FILE` at package
  import time (only if not already set), so every `urllib` caller
  in the process picks up a working store. HuggingFace downloads
  were never affected (they use `requests` + `certifi` directly).

### Internals

- `certifi` is now listed explicitly in `dependencies` (it was
  already a transitive dep via `requests`).

## v0.7.1 — 2026-05-14

### Fixes

- **Transcription auto-falls back to CPU + `int8` when CUDA is
  unavailable** (#19, thanks @fadenb) — running `meet run` on a machine
  without a GPU (laptop, container without passthrough, CI runner)
  previously crashed with `ValueError: device='cuda' but CUDA is not
  available`. `TranscriptionConfig` now warns and falls back to
  `device=cpu`, downgrading `compute_type=float16` to `int8` (float16
  is unsupported on CPU). The model-load log line annotates whether
  CPU was forced (`--device cpu`) or auto-selected because no GPU was
  found, so the diagnostic distinction is preserved.

### Internals

- `TranscriptionConfig` gains an internal `_device_auto_fallback`
  flag set in `__post_init__` when the device is auto-flipped, so
  `_load_whisperx_asr_model` can label the load line accurately
  without re-sniffing torch at print time.

## v0.7.0 — 2026-05-08

### Features

- **Structured YAML frontmatter on every summary (schema_version 1)** —
  `.summary.md` now begins with a typed YAML frontmatter block carrying
  `participants`, `topics`, `action_items` (with assignee, task, due,
  status), `decisions` (text, topic), `language`, `duration`, and a
  `source` pointer. A matching `.frontmatter.json` sidecar is written
  next to it for tools that don't want to parse YAML. The schema is
  intentionally small in v1; downstream tools (e.g. the
  [vezir](https://github.com/pretyflaco/vezir) 0.2.0+ indexer) build
  richer derived views over this stable surface. See the README's
  "Structured frontmatter" section for the schema.
- **`meet ingest` subcommand** — re-extract structured frontmatter for
  one or more existing session directories. Idempotent: skips sessions
  whose `.summary.meta.json` already records `data_extracted: true`
  unless `--force` is passed. Accepts the standard summary-backend
  flags. `--dry-run` previews without invoking the LLM. `--no-pdf`
  skips PDF regeneration.
- **LLM contract: fenced JSON data block** — every summarization prompt
  (single-pass, two-pass formatter, and inline fallbacks) instructs
  the model to append exactly one fenced ```json block at the end of
  its output with the structured fields. Single source of truth: the
  Markdown body still drives the PDF, the JSON block populates the
  frontmatter. JSON is required to be in English even when the body is
  in another language so cross-language indexing works.

### Internals

- New module `meet/frontmatter.py`: schema, build/parse/validate, YAML
  render and read-back, and a `context_from_transcript()` helper that
  pulls `started_at` / `title` from the session's `*.session.json`.
  No PyYAML dependency added; the writer is small enough to maintain
  in-tree and the reader prefers PyYAML when installed but falls back
  to a tightly-scoped subset parser otherwise.
- `MeetingSummary.save(out_dir, basename, *, frontmatter_context=...)` —
  new keyword argument. When provided, the saved Markdown is prefixed
  with the YAML block and a `.frontmatter.json` sidecar is written.
  When omitted, behavior is unchanged from 0.6.x for backward
  compatibility.
- `_dispatch()` in `summarize.py` strips the trailing JSON block off
  every backend's output once, so PDF rendering keeps using a clean
  Markdown body and `MeetingSummary.data` exposes the parsed dict to
  callers.
- `meet label` (find-and-replace fallback) splits, replaces, and
  re-renders both the YAML frontmatter and the JSON sidecar in step
  with the body, so renames stay consistent across all four artifacts.
- The summary `.summary.meta.json` sidecar now records
  `data_extracted: true` on success or `data_error: "<reason>"` on
  failure to extract.

### Backwards compatibility

- All callers that don't pass `frontmatter_context=` to
  `MeetingSummary.save()` continue to produce the legacy artifacts.
- Sessions recorded before 0.7.0 work unchanged; run `meet ingest` to
  upgrade them to schema_version 1.

### Tests

- 33 new tests across `tests/test_frontmatter.py` (24) and
  `tests/test_ingest.py` (9), plus 3 new assertions in
  `tests/test_summarize.py` confirming both the on-disk and inline
  fallback prompts carry the JSON contract.

---

## v0.6.1 — 2026-05-05

### GUI

- **Language dropdown** in the GTK Advanced expander to mitigate
  Whisper's auto-detect failure on sparse / quiet audio (long opening
  silences, short initial utterances).  Without an explicit language,
  Whisper's classifier can mis-identify English audio as Japanese /
  Chinese / Korean and produce pages of CJK hallucinations interspersed
  with correctly-transcribed English fragments.  A real Blink dev-sync
  recorded on 2026-05-05 was rendered useless this way: the meeting
  was English but every artifact came out in Japanese hallucinations.
- New fourth row in **Advanced**: a Language combobox with values
  `auto` (default) plus `en`, `de`, `fr`, `es`, `tr`, `fa`, `it`, `pt`,
  `nl`, `ja`, `zh`, `ko`, `ar`, `ru`.  Selection writes back into
  `transcribe_kwargs["language"]` immediately so the next recording's
  transcription picks it up.

### Compatibility

- CLI `--language` plumbing unchanged (in place since PR #4).
- vezir 0.1.3 unaffected — vezir doesn't invoke the GUI.

### PRs

- #16 feat(gui): add Language dropdown to Advanced settings panel

---

## v0.6.0 — 2026-05-04

### Apple Silicon (first-class support)

- **New `--asr-backend [auto|whisperx|mlx]` flag.** `auto` selects
  MLX Whisper on Apple Silicon when `mlx-whisper` is installed.
  Contributed by @openoms in #4 (first external contribution).
- **New `--torch-device [cuda|cpu|mps]` flag** for splitting
  alignment / diarization device from the ASR device.
- **New `--mlx-model` flag** to override the MLX repo (defaults to
  alias-mapped variants of `--model`).
- **`--device` and `--torch-device` auto-detect platform-appropriate
  defaults**: `cpu` / `mps` on Apple Silicon, `cuda` elsewhere.  Mac
  users no longer need to pass flags manually.
- New collapsible **Advanced** settings panel in the GTK recorder
  widget exposes the three new options without restarting.

Install with `pip install 'meetscribe-offline[mlx]'` on Apple Silicon
to pull in the MLX backend.

### Robustness

- `TranscriptionConfig` now validates `device` and `torch_device`
  against runtime availability with clear `ValueError` messages.
  Bad device combinations fail fast instead of deep inside whisperx.
- MLX backend always logs a one-time info note that VAD options
  (`--vad-onset`, `--vad-offset`) are inert under MLX.

### Infrastructure

- New GitHub Actions workflow runs `ruff` + focused `pytest` on
  push and PR.  First green CI on the repo.

### Compatibility

- **vezir 0.1.2+** detects the new flags via help-parsing
  (`meet_supports_option`); existing deployments work without
  configuration changes.
- **vezir-android** unchanged — communicates with vezir's HTTP API,
  not meetscribe directly.
- **`meetscribe-record`** dependency unchanged (still `>=0.1.0`).

### PRs

- #4 Add support for Apple Silicon and PyTorch device configuration (@openoms)
- #10 Validate torch device availability in TranscriptionConfig (closes #7)
- #11 Expose ASR backend, torch device, and MLX model in GUI (closes #5)
- #12 Auto-default device on Apple Silicon (closes #8)
- #13 Add minimal CI workflow with ruff and pytest (closes #9)
- #14 Always note that MLX backend ignores VAD options (closes #6)

---

## v0.5.0 — 2026-04-25

### Package split

The capture-only modules (`capture`, `audio`, `utils`, `languages`)
and the four capture-only subcommands (`record`, `devices`, `check`,
`archive`) move into a new sibling package
[`meetscribe-record`](https://github.com/pretyflaco/meetscribe-record).
`meetscribe-offline` now depends on it, keeps the heavy
transcription / diarization / summarization stack, and registers its
remaining subcommands via Click plugin entry-points
(`meet.subcommands`) so they appear under the same single `meet`
console script that `meetscribe-record` provides.

### Why

A meetscribe install pulls ~3 GB of transitive deps (whisperx +
torch + pyannote + ctranslate2 + reportlab).  For thin clients that
only need to record audio (e.g. [vezir](https://github.com/pretyflaco/vezir)
scribe widgets on teammate laptops), that's wasteful and slow to
install.  With the split:

| Profile | Install command | Footprint |
|---|---|---|
| Capture-only client | `pip install meetscribe-record` | ~30 MB |
| Full pipeline | `pip install meetscribe-offline` | ~3 GB (pulls -record transitively) |

### User-facing changes

- `meet` console script is now provided by `meetscribe-record`.
  When `meetscribe-offline` is also installed, the same `meet`
  command exposes all 12 subcommands (4 capture + 8 offline) via
  Click entry-point plugin discovery.
- `meet --version` reports both packages: e.g.
  `meet, version 0.5.0 (meetscribe-offline 0.5.0; meetscribe-record 0.1.0)`.
- `meetscribe-offline` 0.5.0 no longer registers a `meet` console
  script of its own; this avoids pip's last-installed-wins
  entry-point conflict.

### Internal changes

- `meet/capture.py`, `meet/audio.py`, `meet/utils.py`,
  `meet/languages.py` become thin compat shims that
  `from meet_record.X import *`, so any existing
  `from meet.capture import ...` continues to work.
- `meet/cli.py` loses the 4 capture commands (~286 lines deleted).
  The remaining 8 subcommands keep their `@main.command()`
  decorators and are exposed by name to the `meet.subcommands`
  entry-point group.
- `meet/__init__.py` now exposes `__version__` resolved via
  `importlib.metadata`, fixing the historical hardcoded `0.4.1`
  literal in `cli.py`.

### Compatibility

Existing installs continue to work after `pip install --upgrade
meetscribe-offline` — the dependency on `meetscribe-record>=0.1.0`
pulls it in transparently.  No flag or config changes required.

### Fixes

- `cli.py` no longer reports a stale `0.4.1` from `python -m
  meet.cli --version`; the version is resolved dynamically from
  package metadata.

---

## v0.4.2 — 2026-04-24

### Improvements

- **Two-pass Ollama summarization (default for local LLMs)** — the local
  Ollama backend now runs a separate extraction pass (Pass 1: pull topics,
  actions, decisions, questions out of the transcript with a wide context
  window) followed by a formatting pass (Pass 2: organize the extracted
  data into the canonical Markdown structure with a small 8K context).
  This dramatically improves format compliance and reduces hallucinations
  on 20B-class local models like `gpt-oss:20b`. Cloud backends
  (claudemax, openrouter, openai) are unchanged — they remain single-pass.
- **Improved cloud-summary prompt** — the system prompt used by the
  cloud backends has been rewritten based on A/B-tested results: more
  topics extracted, ~20% faster on Sonnet, no regressions.
- **`--ollama-singlepass` opt-out flag** — added to `transcribe`, `run`,
  `gui`, and `label` commands for users who want the previous single-pass
  behavior. Also configurable via `MEETSCRIBE_OLLAMA_SINGLEPASS=1`.
- **Per-pass timing in summary sidecar** — the `.summary.meta.json`
  sidecar now records `mode: "two_pass"`, `pass1_seconds`,
  `pass2_seconds`, and `pass1_chars` when two-pass was used.

### Documentation

- Added `docs/local-model-evaluation.md` — full evaluation of local
  20B-class models on 4 reference transcripts, including known
  failure modes (gpt-oss:20b unreliability on low-information short
  transcripts, qwen3.6:27b reasoning-mode bottleneck, and the
  rationale for the two-pass design).

### Testing

- Added 27 new tests covering env-var resolution, two-pass system
  prompts (en + de), two-pass call flow, dispatcher routing, and
  sidecar serialization. All 127 tests pass.

### Known limitations

- `gpt-oss:20b` may hallucinate on transcripts dominated by very
  short low-information utterances ("yes", "okay"). For such meetings
  the cloud backends produce more reliable summaries — the fallback
  chain (claudemax → openrouter → ollama) handles this automatically
  if a cloud backend is configured.
- `gpt-oss:20b` may exceed the default 600s timeout on very large
  (>100 KB) non-English transcripts during Pass 1. The fallback chain
  catches this; alternatively pass `--summary-timeout 1200` or use a
  cloud backend.

---

## v0.4.1 — 2026-04-13

### Improvements

- **Speaker labeling and sync prompts no longer deferred during recording** —
  previously, the speaker labeling dialog and sync confirmation prompt would
  wait until the user stopped recording before appearing. They now appear
  immediately, allowing users to label speakers from a previous meeting while
  the next one records.

---

## v0.4.0 — 2026-04-13

### New features

- **Background post-processing for back-to-back meetings** — after stopping a
  recording, the GUI returns to idle within seconds (drain time) so you can
  immediately start recording the next meeting. Transcription, speaker labeling,
  summarization, PDF generation, and sync all run in a background job queue.
  A small status line at the bottom of the window shows background progress
  (e.g., "Transcribing: meeting-20260413-143453..."). Interactive dialogs
  (speaker labeling, alignment model prompts, sync confirmation) are deferred
  until the user is not actively recording.

### Improvements

- Simplified GUI state machine: removed 8 post-processing states that blocked
  the recording controls. Primary states are now: idle, recording, paused,
  draining, done, error.
- Background jobs process sequentially via a FIFO queue, ensuring GPU resources
  are not contended between concurrent transcriptions.
- Clean shutdown: closing the window unblocks any background threads waiting
  for user input.

### Testing

- All 100 tests pass (99 + 1 pre-existing environment-dependent skip).

---

## v0.3.3 — 2026-04-13

### New features

- **GUI Pause/Resume** — the recording widget now shows side-by-side Pause and
  Stop buttons while recording. Pressing Pause stops the current ffmpeg chunk
  and freezes the timer; pressing Resume starts a new chunk. Stopping from
  either recording or paused state works seamlessly — chunks are stitched
  together automatically. The idle/done/error states still show a single
  centered Record button as before.

### Improvements

- `RecordingSession` in `capture.py` gained `pause()` and `resume()` methods
  and a `paused` field on `RecordingStatus`, making pause/resume available to
  any future consumer (CLI, scripts, etc.) without GUI dependency.
- The watchdog thread now skips health checks while paused, preventing false
  stall-restart triggers.
- Stopping from the paused state skips the 10-second drain buffer since there
  is no active ffmpeg pipeline to flush.

### Bug fixes

- **CLI version string** — `meet --version` now reports the correct version
  (`0.3.3`) instead of the stale `0.1.0` it has shown since the initial release.

### Testing

- 13 new tests for pause/resume functionality (`tests/test_capture.py`):
  pause flag, ffmpeg stop, error cases, resume chunk creation, status reporting,
  elapsed-time freezing, stop-from-paused, and watchdog behaviour.
- All 100 tests pass.

---

## v0.3.2 — 2026-04-10

### New features

- **`--mixdown dual` mode for headphone users** — new CLI flag on `meet transcribe`
  and `meet run` that transcribes each stereo channel independently (mic → YOU,
  system → REMOTE) instead of mixing to mono. This fixes transcription for
  headphone setups where the ~20× energy difference between mic and system
  channels causes WhisperX to suppress the quieter voice. Diarization is skipped
  in dual mode since channel identity equals speaker identity. Default behavior
  (`--mixdown mono`) is unchanged.
  *(Contributed by [@Rolloniel](https://github.com/Rolloniel) in [#1](https://github.com/pretyflaco/meetscribe/pull/1))*

### Bug fixes

- **Speaker labeling threshold** — `_label_speakers_from_channels()` now requires
  `mic_ratio > 0.5` before labeling a speaker as YOU. Previously, the speaker
  with the highest mic ratio was always labeled YOU even when no speaker was
  actually mic-dominant (e.g. system-only audio capture). When no speaker exceeds
  the threshold, all speakers are labeled REMOTE.
  *(Contributed by [@Rolloniel](https://github.com/Rolloniel) in [#1](https://github.com/pretyflaco/meetscribe/pull/1))*

---

## v0.3.1 — 2026-04-10

### Bug fixes

- **CUDA NVRTC JIT fix** — replaced `_ensure_nvrtc_compat()` symlink approach
  with `_preload_nvrtc_builtins()` using `ctypes.CDLL`. The old method created
  a wrong-version symlink and set `LD_LIBRARY_PATH` too late (after
  `libnvrtc.so` was already loaded). The new approach preloads the correct
  `libnvrtc-builtins.so` into the process address space before NVRTC needs it,
  with automatic version detection across `nvidia-cuda-nvrtc` pip packages.

- **Channel-based diarization fallback** — added `_split_by_channel()` for
  stereo recordings where pyannote detects only 0–1 speakers. This can happen
  on short recordings or when GPU-dependent floating-point differences in
  WeSpeaker speaker embeddings cause VBx clustering to collapse multiple
  speakers into one. The fallback uses per-segment and per-word mic vs system
  channel RMS energy to assign YOU/REMOTE labels, which is hardware-independent
  and reliable when stereo channels are cleanly separated.

---

## v0.3.0 — 2026-04-01

### New features

- **Multi-backend summarization** — supports four backends with automatic
  fallback: `claudemax` (Claude Max API Proxy), `openrouter` (OpenRouter API),
  `openai` (any OpenAI-compatible endpoint), and `ollama` (local). If the
  configured backend is unavailable, meetscribe automatically tries the next
  one. Use `--summary-backend` and `--summary-model` flags, or set
  `MEETSCRIBE_SUMMARY_BACKEND` / `MEETSCRIBE_SUMMARY_MODEL` env vars.

- **Generic OpenAI-compatible backend** — use any OpenAI-compatible API for
  summarization (Lemonade, LiteLLM, vLLM, LocalAI, self-hosted endpoints).
  Set `MEETSCRIBE_OPENAI_BASE_URL` and optionally `MEETSCRIBE_OPENAI_API_KEY`.

- **Voiceprint speaker recognition** — automatically identifies speakers across
  meetings using voice embeddings. After labeling a meeting, speaker profiles
  are stored in `~/.config/meet/speaker_profiles.json`. Future meetings match
  voices against the database using cosine similarity. Use `meet enroll` to
  build profiles from past sessions, or let the GUI update profiles
  automatically after each labeling.

- **Meeting sync** — push meeting artifacts (transcript, summary, PDF, SRT) to
  any configured Git repository on a schedule. Configure your repo URL and
  meeting schedule in `~/.config/meet/sync_config.json`. Use `meet sync` to
  push manually or let the GUI auto-sync after recording. Run
  `meet sync --init-config` to generate an example config.

- **Improved summarization prompts** — prompt templates extracted to standalone
  markdown files (`meet/prompts/summarize_system.md`, etc.) for easy iteration
  without touching Python code. Prompt rewritten for better results with
  local/open-source models: more information-dense, preserves technical
  specificity, captures implied action items, provides format guidance.

### Improvements

- Dynamic context window sizing for ollama — automatically sizes `num_ctx` to
  fit long transcripts (up to 64K tokens) instead of truncating.
- Response validation catches upstream API errors (expired tokens, rate limits)
  that would otherwise be silently saved as the meeting summary.
- Thinking mode explicitly disabled for ollama models (`think: false`) to avoid
  wasting tokens on hidden reasoning with models like GLM-4.7-flash and Qwen 3.5.
- GUI auto-sync guarded by `is_sync_configured()` — silently skips if no repo
  is configured.

### Testing

- All 81 existing tests pass with the new prompt loading system.

---

## v0.2.0 — 2026-03-14

### New features

- **Multilingual support** — Whisper large-v3-turbo supports 99 languages.
  meetscribe now passes language hints through the full pipeline: transcription,
  wav2vec2 alignment, Ollama summary (prompted in the source language), and PDF.
  Use `--language auto` (default) or specify a code: `en`, `de`, `tr`, `fr`,
  `es`, `fa`.

- **Farsi / RTL support** — Farsi transcripts render correctly in PDF using
  Noto Naskh Arabic with arabic-reshaper + python-bidi for right-to-left layout.
  Install optional deps with `pip install "meetscribe-offline[rtl]"`.

- **`meet label` CLI command** — assign real names to speakers after the fact.
  For each speaker: shows a summary table, plays a short audio clip from the
  correct stereo channel (via ffplay), prompts for a name. Regenerates all
  outputs (txt, srt, json, summary.md, pdf) with the new names. Options:
  `--no-audio`, `--no-summary`.

- **GUI speaker labeling dialog** — when 2+ speakers are detected, a dialog
  appears before results are saved. Shows each speaker's channel and a sample
  line. Labels are applied before writing any output files.

### Improvements

- PDF now uses DejaVu Sans for full Unicode coverage (replaces previous
  Latin-only font). Handles Cyrillic, Greek, Turkish special characters, etc.
- Ollama summary prompts are now language-aware: when a non-English language is
  detected, the prompt instructs the LLM to write the summary in that language.
- `post_process()` function centralises all output generation (txt, srt, json,
  pdf) so that `meet label` and the GUI dialog share the same code path.
- Shared utilities extracted to `meet/audio.py`, `meet/languages.py`,
  `meet/utils.py`, `meet/label.py` for cleaner architecture.

### Testing

- 81-test suite added covering `label`, `pdf`, `summarize`, `transcribe`, and
  `utils` modules.

### Package

- PyPI package renamed to `meetscribe-offline` to distinguish from an unrelated
  squatted project. Install with `pip install meetscribe-offline`.

---

## v0.1.0 — 2026-03-01

Initial release.

- Dual-channel audio capture (mic left, system audio right) via
  PipeWire/PulseAudio + ffmpeg
- WhisperX transcription (faster-whisper + wav2vec2 alignment)
- pyannote-audio speaker diarization with YOU/REMOTE channel mapping
- Ollama AI meeting summaries (qwen3.5:9b default)
- PDF output (summary + full transcript)
- Output formats: `.txt`, `.srt`, `.json`, `.summary.md`, `.pdf`
- GTK3 GUI widget (always-on-top, record/stop, live timer, open results)
- CLI: `meet run`, `meet record`, `meet transcribe`, `meet gui`, `meet devices`,
  `meet check`
