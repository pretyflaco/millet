# TEE Summarization Evaluation — can in-enclave models replace Sonnet 4.6?

Created: 2026-09-12
Scope: does a hardware-attested TEE model match or beat `claude-sonnet-4-6`
(via claudemax proxy) as the default meeting summarizer?

> **Result:** `glm-5-3-flash` is the only candidate that passed the agreed
> bar — it beats Sonnet on precision *and* recall under both judges, in
> every language tested, with no regression. `deepseek-v4-1-flash` failed on
> a single cell (Turkish recall, −2.1 pp under one judge).
> `gpt-oss-120b` failed all 16 checks.
>
> Metrics only. No transcript or summary content is reproduced here.
>
> **Update 2026-10-06 (millet 0.21.6):** Tinfoil deprecated `glm-5-3-flash`
> effective 2026-10-09, so the runner-up `deepseek-v4-1-flash` is now the
> default and `glm-5-3` the sibling fallback. No new evaluation was run; the
> numbers below are the basis for the switch.

---

## Why this evaluation exists

The earlier `glm-5-2` → `glm-5-3-flash` migration (v0.18.1) was scored with
**structural metrics** — section compliance, bullet counts, header
localization. That was adequate for its question ("don't regress against a
model we're forced off anyway"), but it cannot distinguish *recall* from
*padding*: a model that invents ten plausible bullets scores the same as one
that captures ten real ones.

Retiring Sonnet is a much larger decision, so this evaluation measures
**grounded** precision and recall against a per-meeting answer key.

Note how little the structural metrics would have told us: **all four
models scored 10/10 on both format compliance and language compliance.**
Structure was a four-way tie. Every real difference was in content.

## Method

**Corpus** — 10 meetings with an existing on-disk Sonnet baseline:
6 EN (20 KB–164 KB), 2 DE (24 KB, 166 KB), 2 TR (34 KB, 44 KB).
Speaker counts 2–14.

**Generation** — via `millet.summarize.summarize()`, the same entry point
production uses, so prompts, language handling and two-pass logic are
identical to a real job. 30 candidate summaries, no fallbacks triggered.

**Answer keys** — each judge read *only the transcript* (never any summary)
and enumerated every topic, action, decision and open question a correct
summary must contain. Recall is scored against these keys. 20 keys built
(10 meetings × 2 judges).

**Blinding** — for each meeting the 3 candidates plus the Sonnet baseline
were stripped of identifying scaffolding (YAML frontmatter, trailing JSON
data blocks) and relabelled A–D under a per-meeting permutation. The
label→model mapping was never shown to a judge.

**Scoring** — each judge graded **one summary at a time** (independent, not
comparative, so there is no positional or spread-the-grades effect) against
its own answer key plus the transcript: claim-level precision, recall
against key IDs, and owner-attribution accuracy. 80 verdicts, 0 errors.

**Judges** — `glm-5-3` and `kimi-k3`, both in-TEE, so no meeting content
left the enclave at any point. `deepseek-v4-1-flash` was excluded as a judge
because it is a candidate. `kimi-k3` (Moonshot) is independent of all three
candidate families.

## Results

### Overall (10 meetings pooled)

| model | judge | precision | halluc | distort | recall | owner |
|---|---|---|---|---|---|---|
| `claude-sonnet-4-6` | glm-5-3 | 94.5% | 1.1% | 4.3% | 73.9% | 92.4% |
| | kimi-k3 | 95.3% | 0.6% | 4.1% | 79.0% | 96.2% |
| `glm-5-3-flash` | glm-5-3 | 98.3% | 0.1% | 1.6% | 91.2% | 94.6% |
| | kimi-k3 | 98.7% | 0.2% | 1.1% | 94.2% | 95.5% |
| `deepseek-v4-1-flash` | glm-5-3 | 98.2% | 0.3% | 1.5% | 87.2% | 90.6% |
| | kimi-k3 | 98.6% | 0.1% | 1.3% | 89.8% | 93.4% |
| `gpt-oss-120b` | glm-5-3 | 86.3% | 2.9% | 10.9% | 52.8% | 78.9% |
| | kimi-k3 | 86.2% | 5.1% | 8.7% | 56.4% | 78.9% |

Format compliance and language compliance: **10/10 for all four models.**

### Recall by category (pooled across judges)

| model | topics | actions | decisions | questions |
|---|---|---|---|---|
| `claude-sonnet-4-6` | 76.7 | 76.3 | 77.7 | 73.3 |
| `glm-5-3-flash` | 94.1 | 93.2 | 92.2 | 85.3 |
| `deepseek-v4-1-flash` | 87.2 | 92.1 | 91.3 | 82.8 |
| `gpt-oss-120b` | 55.3 | 53.4 | 62.1 | 47.4 |

Open questions are the weakest category for every model — the hardest thing
to catch is what was raised and left unresolved.

### Generation latency

| model | median | max |
|---|---|---|
| `glm-5-3-flash` | 110.8 s | 352.7 s |
| `deepseek-v4-1-flash` | 87.5 s | 114.6 s |
| `gpt-oss-120b` | 32.1 s | 45.7 s |
| `claude-sonnet-4-6` (historical, n=534) | 52.0 s | 703.9 s |

### Cross-judge agreement

- **Recall ranking: both judges agree** —
  `glm-5-3-flash` > `deepseek-v4-1-flash` > `claude-sonnet-4-6` > `gpt-oss-120b`.
- **Hallucination ranking: judges disagree**, but only in swapping the top
  two, which sit at 0.1–0.3% — the noise floor. Sonnet's and `gpt-oss`'s
  positions agree under both.
- **No self-preference bias observed, and the effect ran backwards:**
  `glm-5-3` scored its own family's `glm-5-3-flash` *lower* (91.2%) than the
  independent `kimi-k3` did (94.2%).
- Per-meeting disagreements reach 15 pp; pooled figures are far steadier
  than any individual cell.

## The bar, and the verdict

Agreed rule: a candidate replaces Sonnet only if it (1) beats Sonnet on
precision under both judges, (2) beats Sonnet on recall under both judges,
and (3) shows no language regression — precision and recall at least
Sonnet's in every language, under both judges. 16 checks per candidate,
applied mechanically (`gate.py`) rather than by eye.

| candidate | verdict | detail |
|---|---|---|
| `glm-5-3-flash` | **PASS** | 16/16 |
| `deepseek-v4-1-flash` | FAIL | 15/16 — TR recall 87.5% vs Sonnet 89.6% (−2.1 pp, kimi-k3) |
| `gpt-oss-120b` | FAIL | 0/16 — e.g. DE recall 30.9% vs 68.1% |

`deepseek-v4-1-flash` failing by 2.1 pp on one of two Turkish meetings is
within the noise this corpus can resolve. It remains a sound **sibling
fallback** (v0.18.1) — where the alternative is no summary at all — but it
did not clear the bar to become the default.

## Limitations

1. **n = 10**, with only 2 DE and 2 TR meetings. Per-language numbers are
   directional.
2. **No Farsi/RTL evidence.** The only FA session on record is 1.7 KB of
   test audio. RTL rendering remains unvalidated by this evaluation.
3. **Baselines are historical**, not fresh re-runs: same prompts and
   temperature, but generated between 2026-05 and 2026-09 against whatever
   `claude-sonnet-4-6` resolved to then.
4. **One baseline predates a prompt change.** `01KQZ561CR` (TR) was
   summarized 2026-05-06, before the 2026-05-24 prompt edit. The other 9
   used prompts identical to the candidate runs; the default summary
   prompts have not changed since (v0.17.0 only *added* iteration-plan
   files).
5. **Owner attribution was not part of the bar**, and is the one place
   Sonnet still edges the winner: in English, Sonnet leads `glm-5-3-flash`
   by 0.0–1.1 pp. Worth watching, not decisive.
6. **Judges are LLMs.** Cross-family agreement is a control, not a proof.
   The 15 pp per-meeting spread is the honest error bar.

## Cost

| stage | calls | cost |
|---|---|---|
| Generation | 30 | ~$0.60 |
| Answer keys | 20 | $6.39 |
| Scoring | 80 | $22.10 |
| **Total** | **130** | **~$29.09** |

At the observed rate of ~120 meetings/month, `glm-5-3-flash` in production
costs roughly **$2.40/month**.
