# MeetScribe Local Model Improvement: Double-Pass Approach & Prompt Engineering

Created: 2026-04-24
Last updated: 2026-09-15 — added a `nemotron-3-nano:4b` evaluation for the zero-network **fallback floor** (millet 0.21.0's `tinfoil → venice → near → ollama` chain); positioned it below the standing `qwen3.8:27b` recommendation.
Earlier: 2026-08-15 — added Qwen3.8-27B evaluation (EN/TR/DE, via millet's real two-pass code path); reconciled recommendations.
Context: Research session evaluating local models as potential replacements for Sonnet 4.6 in meetscribe's meeting transcript summarization pipeline.

> **TL;DR (2026-08-15):** With the two-pass extraction-truncation fix in
> v0.15.1, **`qwen3.8:27b` is the strongest local model** for this task —
> it matches or exceeds the Sonnet baseline on topic/action/question
> coverage across English, Turkish, and German meetings, with 5/5 format
> compliance and localized section headers, at ~55–100 s/meeting on an
> RTX 3090. It supersedes the earlier "gpt-oss:20b is the best local"
> conclusion below. Caveats: ~2× slower than Sonnet; a mild tendency to
> over-count "decisions"; validated on 3 EN + 1 TR + 1 DE meetings.

> **TL;DR (2026-09-15) — the fallback floor is a different, lower bar.**
> millet 0.21.0 added a `tinfoil → venice → near → ollama` summary chain; the
> `ollama` tier is the zero-network floor that runs only when every attested
> TEE provider is unreachable. `nemotron-3-nano:4b` (2.8 GB, the one Nemotron
> that fits a 12 GB RTX 5070) was evaluated for it and is **fit as a last
> resort only.** On one hard 1h18m ~28-speaker all-hands it covered **≤ 43 %**
> of topics vs a `glm-5-3-flash (TEE)` baseline's 100 %: two-pass caught only
> the meeting's opening (and fabricated a "decision"), single-pass only the
> closing (and broke format), `:4b-q8_0` looped. Context fit was **not** the
> limiter (num_ctx ≈ 19.5k, well under ceiling) — 4B attention capacity was.
> It does **not** approach `qwen3.8:27b`; it exists so a network outage yields
> *some* summary rather than none. See "Nemotron-3-Nano-4B Floor Evaluation
> (2026-09-15)" below.

---

## Background

MeetScribe uses an LLM to summarize meeting transcripts into a structured 5-section Markdown format (Overview, Topics, Actions, Decisions, Open Questions). The current production backend is **Sonnet 4.6** via claudemax proxy, with **Ollama** as fallback. The hardcoded Ollama default is `qwen3.5:9b`; per-machine model choice is set via the `MILLET_SUMMARY_MODEL` env var (e.g. `qwen3.8:27b` on GPUs with ≥18 GB VRAM). The default is deliberately conservative so it runs on small GPUs — the hardware, not the codebase, dictates which model to prefer.

The goal is to determine whether local models can match Sonnet 4.6 quality for fully offline summarization.

### Relevant files

- **System prompt:** `meet/prompts/summarize_system.md`
- **User prompt:** `meet/prompts/summarize_user.md`
- **User prompt (non-English):** `meet/prompts/summarize_user_lang.md`
- **Summarization code:** `meet/summarize.py` (4 backends: claudemax, openrouter, ollama, openai)
- **Language headers:** `meet/languages.py` (section header translations for DE, FR, ES, TR, FA)
- **Default Ollama model:** `qwen3.5:9b` (hardcoded floor; override per-machine with `MILLET_SUMMARY_MODEL`)

### Typical meeting characteristics

| Metric | Range |
|--------|-------|
| Transcript size | 15-120 KB (plain text) |
| Token count | 4K-30K tokens |
| Duration | 15-90 min |
| Speakers | 2-8 |
| Languages | English (primary), German, Turkish, Farsi |
| Output size (Sonnet 4.6) | 3-9 KB Markdown |

### Test meetings used in this evaluation

| Meeting | Size | Tokens | Lang | Speakers | Complexity |
|---------|------|--------|------|----------|------------|
| Meeting A | 26 KB | ~7K | EN | 4 | Medium — short-form discussion with side topics |
| Meeting B | 67 KB | ~17K | EN | 6 | High — multi-topic technical discussion |
| Meeting C | 118 KB | ~30K | DE | 8+ | Very high — 18+ topics, German |
| Meeting D | 79 KB | ~20K | EN | 4 | High — 11+ distinct topics |

---

## Models Tested

| Model | Ollama ID | Size | Context | Architecture | Speed (tg128) |
|-------|-----------|------|---------|--------------|---------------|
| Sonnet 4.6 | N/A (cloud) | N/A | 200K | Anthropic | N/A |
| **Qwen3.8 27B** | `qwen3.8:27b` | 18 GB | 262K | Qwen3.5/DeltaNet hybrid + vision | 41.9 t/s |
| GPT-OSS 20B | `gpt-oss:20b` | 13 GB | 128K | OpenAI MoE, MXFP4 | ~50 t/s |
| Qwen3.6 27B | `qwen3.6:27b` | 17 GB | 262K | Qwen3.5/DeltaNet hybrid | 36 t/s |
| Qwen3.5 27B | `qwen3.5:27b` | 17 GB | 262K | Qwen3.5 | ~8 t/s (partial CPU) |
| Gemma4 26B | `gemma4-26b-16k` | 17 GB | 16K | Google Gemma4 | 137 t/s |

Hardware: RTX 3090 (24 GB VRAM), 64 GB RAM.

---

## Current System Prompt

File: `meet/prompts/summarize_system.md`

```markdown
You are a professional meeting assistant. Analyze the meeting transcript below and produce a structured summary in exactly the Markdown format specified.

## OUTPUT FORMAT (use exactly this structure):

## {overview}
2-3 sentences covering: what the meeting type was, who was involved (by role/function if unclear), and the main themes discussed.

## {topics}
* **Topic name:** 1-2 sentence description of what was discussed, including any key technical details, tools, or product names mentioned.
(List every substantive topic — aim for 6-10 bullet points for a 60-90 min meeting. Scale proportionally for shorter/longer meetings.)

## {actions}
* Action item in imperative form, with enough context to act on it — **Owner** (use exact name from transcript)
(Capture both explicitly assigned AND clearly implied items, e.g. "I'll look into that" = action for that speaker. If owner is unclear, write **Owner Unknown**. If none, write "{none_stated}".)

## {decisions}
* Concrete decision reached, stated as a fact (e.g. "X was chosen over Y", "Z is deprioritized")
(Only include things actually agreed upon, not merely suggested or explored. If none, write "{none_stated}".)

## {questions}
* Unresolved question, open dependency, or follow-up item — include enough context to understand why it matters
(Include unresolved debates, blockers waiting on third parties, and things flagged as "we need to figure out". If none, write "{none_stated}".)

## RULES:
1. Use speaker labels EXACTLY as they appear in the transcript. Do not rename, merge, or invent speakers.
2. Do NOT hallucinate. Every item must be traceable to something said in the transcript.
3. Be concise but information-dense. Avoid filler phrases like "the team discussed..." — state the substance directly.
4. For technical topics, preserve specificity: name the exact tools, frameworks, APIs, error types, or architectural patterns mentioned.
5. Actions: include items that were explicitly assigned AND items clearly implied (e.g. "I'll look into that" = action for that speaker).
6. Decisions: only include things actually agreed upon, not things merely suggested or explored.
7. Questions: include unresolved debates, blockers waiting on third parties, and things flagged as "we need to figure out".
8. Keep the summary professional and objective.
{lang_instruction}
```

The `{overview}`, `{topics}`, `{actions}`, `{decisions}`, `{questions}`, `{none_stated}` placeholders are replaced by language-specific headers at runtime via `_build_system_prompt()` in `summarize.py`. The `{lang_instruction}` placeholder is replaced with a language directive for non-English transcripts.

---

## Single-Pass Results (Current Approach)

### Test: Meeting A (26 KB, ~7K tokens, English)

| Dimension | Sonnet 4.6 | gpt-oss:20b | qwen3.6:27b | qwen3.5:27b | gemma4-26b-16k |
|-----------|:-:|:-:|:-:|:-:|:-:|
| 5-section format | 5/5 | **5/5** | 0/5 | 0/5 | 0/5 |
| Topics covered | 10 | 10 | 1 (tunneled) | 1 (tunneled) | 3 |
| Action items | 5 with owners | 10 (some generic) | 0 | 0 | Generic, no names |
| Decisions | 3 | 0 | 0 | 0 | 0 |
| Open questions | 5 | 5 | 0 | 0 | 0 |
| Time | 40s | 36s | 132s | 98s | 23s |

### Test: Meeting D (79 KB, ~20K tokens, English)

| Dimension | Sonnet 4.6 | gpt-oss:20b | qwen3.6:27b |
|-----------|:-:|:-:|:-:|
| 5-section format | 5/5 | 4/5 (tables, extra sections) | 0/5 (offered a menu) |
| Topics covered | 11 | 5 (grouped) | 0 |
| Action items | 13 with owners | 4 vague | 0 |
| Decisions | 6 | 5 (some fabricated) | 0 |
| Open questions | 8 | 0 | 0 |
| Hallucinations | None | 3 (wrong date, wrong names) | N/A |
| Output size | 8.4 KB | 3.1 KB | 657 chars |
| Time | 64s | 65s | 147s |

### Failure modes identified

1. **Format non-compliance** — All local models except gpt-oss deviate from the specified 5-section Markdown structure. They invent their own formats (tables, numbered sections, "Executive Summary" layouts).
2. **Content tunneling** — Local models fixate on 1-3 topics (usually the most prominent or the last one discussed) and ignore the rest of the meeting.
3. **System prompt ignorance** — qwen3.6 and qwen3.5 treat the transcript as a conversation to respond to, not a document to summarize. qwen3.6 literally offered a menu of things it *could* do instead of following the system prompt.
4. **Hallucination** — gpt-oss invents dates, misnames entities (close-but-wrong proper nouns — e.g. swapping a syllable in a company name, mistranslating part of a foreign-language brand), and fabricates organization names that were never mentioned in the transcript.
5. **Context limitation** — gemma4-26b-16k has a 16K context limit, truncating 2 of 3 test meetings.

---

## Improved Prompt (Single-Pass)

Tested on Meeting D with gpt-oss:20b. This prompt was designed to address format non-compliance, hallucination, and content tunneling.

```markdown
You are a meeting summarizer. Read the transcript and output EXACTLY the Markdown structure shown below — no other sections, no tables, no preamble, no closing remarks.

## {overview}
2-3 sentences: meeting type, participants (by name as they appear in the transcript), and main themes.

## {topics}
* **Topic name:** 1-2 sentence description with specific details (tools, product names, numbers, URLs mentioned).
(Cover EVERY substantive topic. Aim for 6-10 bullets for a 60-90 min meeting. Do NOT group multiple topics into one bullet. Do NOT skip topics from the beginning or middle of the meeting.)

## {actions}
* Action description with enough context to act on it — **Owner** (exact speaker name from transcript)
(Include both explicitly stated AND implied commitments, e.g. "I'll look into that" = action for that speaker. If owner is unclear, write **Owner Unknown**. If none exist, write "{none_stated}".)

## {decisions}
* Decision stated as a fact (e.g. "X was chosen over Y")
(ONLY include things explicitly agreed upon by participants. If someone merely suggested something or expressed interest without commitment, it is NOT a decision — put it in Open Questions instead. If none, write "{none_stated}".)

## {questions}
* Unresolved question or open dependency — include context for why it matters
(Include: unresolved debates, items waiting on third parties, things flagged as "we need to figure out", and suggestions that were NOT confirmed as decisions. If none, write "{none_stated}".)

RULES:
- Output ONLY the 5 sections above. Do NOT add any other sections, headers, metadata, dates, participant lists, "Next Steps", "Prepared by", or sign-off lines.
- Do NOT use tables. Use bullet lists only.
- Do NOT invent or guess dates, company names, or product names. Use ONLY names and terms that appear verbatim in the transcript.
- Use speaker labels EXACTLY as they appear in the transcript. Do not rename, abbreviate, or invent speakers.
- Every item must be directly traceable to something said in the transcript. Do not infer or fabricate.
- Be concise but information-dense. State substance directly — avoid filler like "the team discussed..." or "there was a conversation about...".
- Preserve technical specificity: exact tool names, frameworks, URLs, version numbers, error messages.
- Scan the ENTIRE transcript from start to finish. Do not focus only on the most recent or most prominent portion.
{lang_instruction}
```

### Key changes from original prompt

| Change | Rationale |
|--------|-----------|
| Removed `## OUTPUT FORMAT (use exactly this structure):` header | Local models confused this instructional header with an output header |
| Added explicit "no tables, no preamble, no closing remarks" | gpt-oss added tables, "Prepared by" footer, date metadata |
| `Do NOT use tables. Use bullet lists only.` | gpt-oss reformatted into tables, breaking expected structure |
| `Do NOT invent or guess dates, company names, or product names` | Addresses specific hallucination pattern (dates, entity names) |
| `Do NOT group multiple topics into one bullet` | Prevents the content compression that local models default to |
| `Do NOT skip topics from the beginning or middle` | Addresses attention/tunneling bias |
| Decisions → Open Questions cross-reference | "if someone merely suggested something... it is NOT a decision — put it in Open Questions" |
| `Scan the ENTIRE transcript from start to finish` | Directly targets the content tunneling problem |
| Removed `## RULES:` heading | The `##` prefix made it look like part of the output format |
| Removed redundant rules 5, 6, 7 | Already stated in section descriptions; duplication adds noise |

### Result: Improved prompt made things WORSE

On Meeting D with gpt-oss:20b:
- Format compliance dropped from 4/5 to 0/5 (reverted to table layout)
- Topic coverage went from 5 to 6 but still grouped
- Hallucinations persisted
- Output size decreased from 3.1 KB to 2.3 KB

**Conclusion:** Prompt engineering alone cannot fix the core issues with 20B-class models on this task. The model has a strong prior toward table formatting that overrides explicit prohibitions. Content tunneling is a fundamental attention/capability limitation, not a prompt problem.

**However, the improved prompt should still be evaluated with Sonnet 4.6** to ensure no regression. The cleaner structure and explicit anti-hallucination rules may benefit the cloud model even if they don't help local models.

---

## Double-Pass Approach

### Design

Split the summarization into two sequential LLM calls:

**Pass 1 — Extraction:** Ask the model to scan the full transcript and produce raw numbered lists of topics, actions, decisions, and questions. No formatting pressure.

**Pass 2 — Formatting:** Take the Pass 1 output (~2-3K tokens) and ask the model to organize it into the exact 5-section Markdown structure.

### Why it helps

- **Content tunneling:** Pass 1 only asks the model to extract (simpler task). The explicit instruction to "be exhaustive, cover entire transcript" is more effective when the model isn't simultaneously worrying about format.
- **Format compliance:** Pass 2 receives a short input (~2K tokens) and only needs to reformat. Formatting a short list is much easier than extracting + formatting from 20K tokens simultaneously.
- **Hallucination:** Pass 2 can only work with what Pass 1 extracted — it can't invent new facts from a transcript it never saw.

### Pass 1 Prompt (Extraction)

```
You are a meeting transcript analyzer. Your job is to extract information — NOT to summarize or format.

Read the entire transcript below from the very first line to the very last line.
Extract ALL of the following and output them as simple numbered lists.

TOPICS:
(Every distinct topic discussed — one line per topic. Include key details: names, tools, products, URLs, numbers mentioned. Do NOT group related topics — list each one separately. Aim for 10+ topics for a meeting over 60 minutes.)

ACTIONS:
(Every action item or commitment made by any speaker — state WHAT needs to be done and WHO said they would do it. Include both explicit "I'll do X" and implied commitments. Use speaker names exactly as they appear in the transcript.)

DECISIONS:
(Only things explicitly agreed upon by participants — NOT suggestions, interests, or proposals. A decision requires confirmation from the relevant parties.)

QUESTIONS:
(Unresolved items, open dependencies, things someone said "we need to figure out", proposals that were NOT confirmed, and blockers waiting on external factors.)

RULES:
- Be exhaustive. Cover the ENTIRE transcript from start to finish.
- Use speaker names EXACTLY as they appear in the transcript. Do NOT rename or invent names.
- Do NOT invent dates, company names, or product names not in the transcript.
- Output plain numbered lists only. No Markdown formatting, no tables, no headers beyond the four category labels above.
```

### Pass 2 Prompt (Formatting)

```
You are a meeting summary formatter. You will receive pre-extracted meeting data. Your ONLY job is to organize it into the EXACT Markdown structure below.

Do NOT add any information that is not in the extracted data.
Do NOT remove any information from the extracted data.
Do NOT use tables. Use bullet lists only.
Do NOT add preamble, metadata, dates, participant lists, or closing remarks.
Start your output with "## Meeting Overview" — nothing before it.

## Meeting Overview
2-3 sentences: meeting type, who was involved (use names from the data), and the main themes.

## Key Topics Discussed
* **Topic name:** 1-2 sentence description with specifics from the extracted data.

## Action Items
* Action description with context — **Owner**

## Decisions Made
* Decision stated as a fact.
(If no decisions were extracted, write "None explicitly stated".)

## Open Questions / Follow-ups
* Unresolved question with context for why it matters.
(If none were extracted, write "None explicitly stated".)
```

**Note:** The Pass 2 prompt uses hardcoded English headers. For non-English meetings, the headers need to be language-specific (same as the current prompt system). The `{lang_instruction}` should be appended to Pass 1 to ensure extraction happens in the target language, and Pass 2 headers should use the language-specific section headers from `languages.py`.

### User prompts

**Pass 1:**
```
Extract all topics, actions, decisions, and questions from this transcript:

---
{transcript}
---
```

**Pass 2:**
```
Organize the following extracted meeting data into the required format:

---
{pass1_output}
---
```

### Configuration

- Temperature: 0.2 (slightly lower than 0.3 for more deterministic extraction)
- Pass 1 context: dynamically sized based on transcript length + 4K headroom
- Pass 2 context: dynamically sized based on Pass 1 output length + 4K headroom (typically ~5-8K)
- `think: False` for qwen models (prevents hidden reasoning that inflates latency)

---

## Double-Pass Results

### Test: Meeting D (79 KB, ~20K tokens, English)

#### gpt-oss:20b

| Dimension | Single-pass (original prompt) | Single-pass (improved prompt) | **Double-pass** | Sonnet 4.6 |
|-----------|:-:|:-:|:-:|:-:|
| 5-section format | 4/5 | 0/5 | **5/5** | 5/5 |
| Topics covered | 5 | 6 | **5** | 11 |
| Action items | 4 vague | 6 vague | **6 with owners** | 13 with owners |
| Decisions | 5 (some fabricated) | 0 | **4 (plausible)** | 6 |
| Open questions | 0 | 0 | **0 ("None stated")** | 8 |
| Hallucinations | 3 | 2 | **1** | 0 |
| Output size | 3.1 KB | 2.3 KB | **2.4 KB** | 8.4 KB |
| Time | 65s | 24s | **94s** | 64s |

**Pass 1 behavior:** Still produced its characteristic table+preamble format despite being told "plain numbered lists only." However, it DID extract more content than the single-pass approach. The extraction included topics, actions, decisions, and next steps (though not in the requested numbered list format).

**Pass 2 behavior:** Successfully reformatted the Pass 1 output into perfect 5-section Markdown structure. No tables, correct headers, bullet lists only, no preamble or closing remarks.

**Remaining issues:**
- Content coverage still ~50% of Sonnet 4.6 (5 topics vs 11)
- Open questions section empty despite 8 genuine open items in the meeting
- 1 hallucination persisted (an invented store name not in transcript)
- Pass 1 does not follow its own format instructions (tables instead of numbered lists) — but Pass 2 compensates

#### qwen3.6:27b

| Dimension | Single-pass (original prompt) | **Double-pass** | Sonnet 4.6 |
|-----------|:-:|:-:|:-:|
| 5-section format | 0/5 (refused to summarize) | **5/5** | 5/5 |
| Topics covered | 0 | **5** | 11 |
| Action items | 0 | **5 with owners** | 13 with owners |
| Decisions | 0 | **3 (reasonable)** | 6 |
| Open questions | 0 | **0 ("None stated")** | 8 |
| Hallucinations | N/A | **2** | 0 |
| Output size | 657 chars | **2.8 KB** | 8.4 KB |
| Time | 147s | **353s** | 64s |

**Pass 1 behavior:** Still produced its characteristic "let me help you" preamble with tables and emojis. However, it DID extract meaningful content (topics, action items, strategic notes) — far more than its single-pass attempt where it simply offered a menu.

**Pass 2 behavior:** Successfully reformatted Pass 1 output into perfect 5-section Markdown. The extraction-first approach completely broke qwen3.6's "conversation mode" refusal pattern.

**Remaining issues:**
- Very slow (353s total — 236s extraction + 118s formatting)
- Content coverage ~50% of Sonnet 4.6
- Hallucinated proper nouns: misspelled a real participant's first name and corrupted a company name (same close-but-wrong pattern as gpt-oss)
- Open questions section empty

---

## Summary of All Approaches

### On Meeting D (the hardest test — 79 KB, ~20K tokens, 11 topics)

| Approach | Format | Content | Hallucination | Speed | Verdict |
|----------|:-:|:-:|:-:|:-:|:--|
| Sonnet 4.6 single-pass | 5/5 | 11 topics, 13 actions, 6 decisions, 8 questions | 0 | 64s | Gold standard |
| gpt-oss single-pass (original prompt) | 4/5 | 5 topics, 4 actions | 3 | 65s | Usable but poor |
| gpt-oss single-pass (improved prompt) | 0/5 | 6 topics | 2 | 24s | Regression |
| **gpt-oss double-pass** | **5/5** | **5 topics, 6 actions, 4 decisions** | **1** | **94s** | **Best local (Apr 2026)** |
| qwen3.6 single-pass | 0/5 | 0 (refused) | N/A | 147s | Unusable |
| **qwen3.6 double-pass** | **5/5** | **5 topics, 5 actions, 3 decisions** | **2** | **353s** | **Good but slow** |
| qwen3.5 single-pass | 0/5 | 1 topic (tunneled) | 0 | 98s | Unusable |
| gemma4 single-pass | 0/5 | 3 topics (context-limited) | 0 | 23s | Context-limited |

> **Note (2026-08-15):** Meeting D predates Qwen3.8 and wasn't re-run on it.
> On a comparable set (3 EN + 1 TR + 1 DE), **Qwen3.8-27B double-pass reached
> full cloud-baseline coverage** — a clear step above gpt-oss's ~50% here.
> See "Qwen3.8-27B Evaluation (2026-08-15)" below for the current best local.

### Remaining quality gap vs Sonnet 4.6 (even with double-pass)

> These gaps describe the **20B-class models (gpt-oss, qwen3.6)** as of
> April 2026. **Qwen3.8-27B (2026-08-15) closes most of them** — see its
> section below: it reaches baseline coverage and produces non-empty
> open-questions sections. The items below still hold for the older models.

1. **Content coverage (~50%):** Local models extract ~5 topics where Sonnet finds 11. This is a fundamental attention/capability limitation of 20B-class models processing 20K tokens.
2. **Open questions (0 vs 8):** No local model identified any open questions. This requires understanding the difference between "agreed upon" and "discussed but unresolved" — a nuanced distinction that smaller models miss.
3. **Action item specificity:** Sonnet extracts 13 specific action items with exact owners and context. Local models find 4-6, often vague.
4. **Hallucination resistance:** gpt-oss consistently hallucinates 1-3 proper nouns per summary despite explicit anti-hallucination instructions.

---

## Implementation Plan

### Phase 1: Implement double-pass in summarize.py ✅ DONE

**Shipped.** `_summarize_ollama_twopass()` is implemented and is the default
for the ollama backend (opt out with `MILLET_OLLAMA_SINGLEPASS=1`). The
v0.15.1 fix widened Pass-1's output reserve so thinking-heavy models
(qwen3.8) don't truncate mid-extraction. The original design notes follow.

Add a `_summarize_ollama_twopass()` function to `summarize.py` that:

1. Runs Pass 1 (extraction) with the full transcript
2. Runs Pass 2 (formatting) with the Pass 1 output
3. Returns the Pass 2 output as the summary

**Key implementation details:**
- Only used by the `ollama` backend — claudemax/openrouter/openai continue using single-pass (they don't need it)
- Pass 1 context: dynamically sized via existing `_dynamic_num_ctx()` function
- Pass 2 context: ~8192 (Pass 1 output is typically 2-4K tokens)
- Temperature: 0.2 for both passes
- `think: False` option for qwen models
- Timeout: Pass 1 gets 80% of the 600s total timeout, Pass 2 gets 20%
- Metadata: `summary.meta.json` should record both pass timings
- Language support: Pass 1 gets `{lang_instruction}` appended, Pass 2 headers use language-specific section headers from `languages.py`

**Estimated effort:** ~50-80 lines of new code, isolated to `summarize.py`. No changes to CLI, GUI, or other modules.

### Phase 2: Evaluate improved prompt for Sonnet 4.6 (optional)

The improved prompt (see above) was designed for local models but contains useful clarifications that might benefit Sonnet 4.6:
- Explicit decision vs. suggestion distinction
- Anti-hallucination specifics
- "Scan entire transcript" instruction

Test by running Sonnet 4.6 with the improved prompt on 2-3 meetings and comparing against existing Sonnet summaries. Only adopt if quality is maintained or improved.

### Phase 3: Explore larger local models (future)

The content coverage gap (~50%) is a model capability issue. Options to close it:
- **Qwen3.6 70B+ (quantized):** Would require significant CPU offload or a second GPU. Not practical on current hardware.
- **Future model releases:** As 27B-class models improve in instruction following and long-context attention, the gap will narrow. Re-evaluate in 3-6 months.
- **Hybrid approach:** Use local model for Pass 1 (extraction) and a cloud model for Pass 2 (formatting). This keeps the transcript local but uses the cloud model only for formatting — the extracted data contains no secrets. This is a privacy compromise but a quality improvement.

---

## Qwen3.8-27B Evaluation (2026-08-15)

Qwen3.8-27B was released 2026-08-14. Same hybrid DeltaNet architecture as
Qwen3.6 (`qwen3_5`, 64 layers) plus native vision. This evaluation was run
**through millet's real two-pass code path** (`summarize()` →
`_summarize_ollama_twopass()`), not an ad-hoc harness — the numbers reflect
what production actually produces.

### Requirements & setup

- **Ollama ≥ 0.32.12** — older versions cannot run the architecture
  (`qwen3next: layer 64 missing attn projections`).
- Official model: `ollama pull qwen3.8:27b` (18 GB, 256K context, vision +
  tools + thinking). Do **not** hand-register the Unsloth GGUF — Ollama's
  engine can't run it; only the official pull works.
- Runs fully on a 24 GB GPU (17 GB resident). **Will not fit a 12 GB GPU**
  (e.g. RTX 5070) — pick a smaller model there via `MILLET_SUMMARY_MODEL`.

### Prerequisite fix: Pass-1 truncation (v0.15.1)

The first run through millet **under-performed a throwaway harness** by ~2×.
Root cause: `_dynamic_num_ctx` reserved only 4096 output tokens, so
Qwen3.8's long exhaustive Pass-1 extraction hit the context window and was
silently cut off (`done_reason: length`). Raising the Pass-1 reserve to
16384 (v0.15.1) fixed it — extraction on a 44 KB meeting grew ~1.8 KB →
~5.6 KB, and `done_reason` flipped `length` → `stop`. **All results below
require v0.15.1 or later.** `think: False` is already set by the two-pass
flow and prevents the separate "empty content" failure mode.

### Throughput (llama-bench, pp512/tg128, RTX 3090)

| Model | Prompt (pp512) | Generation (tg128) |
|-------|:-:|:-:|
| Qwen3.8-27B Q4_K_M | 1405 t/s | 41.9 t/s |
| Qwen3.6-27B Q4_K_M | 1421 t/s | 41.3 t/s |
| SuperGemma4-26B Q4_K_M | 4646 t/s | 161.8 t/s |

Marginally faster than Qwen3.6 on generation; still ~4× slower than
SuperGemma4 (which is unusable here for other reasons — format + context).

### Double-pass results vs cloud baseline (via millet, v0.15.1)

Counts are section item counts (**T**opics / **A**ctions / **D**ecisions /
open **Q**uestions). Baselines are the existing production summaries
(claudemax / Sonnet-class). Meetings are anonymized; only metrics are shown.

| Meeting | Lang | ~Tokens | Qwen3.8 (T/A/D/Q) | Baseline (T/A/D) | Format | Time |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|
| E | EN | 11K | 20 / 8 / 4 / 7 | 11 / 6 / 3 | 5/5 | 84 s |
| F | EN | 8K | 10 / 24 / 5 / 10 | 18 / 20 / 4 | 5/5 | 102 s |
| G | EN | 11K | 20 / 16 / 6 / 10 | 10 / 12 / 4 | 5/5 | 102 s |
| H | **TR** | 1.5K | 10 / 1 / 3 / 7 | 6 / 1 / 0 | 5/5 | 53 s |
| I | **DE** | 17K | 15 / 4 / 3 / 8 | 15 / 3 / 0 | 5/5 | 93 s |

**Findings:**

- **Coverage matches or exceeds the baseline** on every meeting. The model
  is more granular (often more topics/questions than the baseline) without
  dropping content.
- **Format compliance 5/5** everywhere, including a genuine open-questions
  section — historically the section all other local models left empty.
- **Non-English works.** Turkish and German summaries are written *in the
  target language* with **localized section headers** (`Toplantı Özeti` /
  `Besprechungsübersicht`, etc.) via `languages.py`. No language leakage.
- **No hallucinated speakers.** Names in the TR/DE outputs trace to the
  transcript (proper nouns that look unusual — e.g. a vendor founder's
  name — were genuinely present).
- **Known weakness — decisions over-inclusion.** Qwen3.8 tends to list
  more "decisions" than the baseline (e.g. 4 vs 3), occasionally promoting
  a strong suggestion or an action to a decision. Not fabrication, but
  looser than Sonnet's stricter "explicitly agreed" bar.
- **Speed:** ~55–100 s/meeting (pass1 + pass2), roughly 2× the cloud
  latency but fine for a batch/offline job.

### Verdict

**Qwen3.8-27B (double-pass, v0.15.1+) is the best local model for meetscribe
summarization to date** — the first to reach cloud-baseline coverage across
English *and* non-English (TR/DE) meetings with correct format and no
hallucinated speakers. It supersedes gpt-oss:20b as the recommended local
model. Sonnet remains more concise/polished and stricter on decisions, and
is ~2× faster, so it stays the production default; but for a fully offline
deployment on a ≥18 GB GPU, Qwen3.8 is now a credible replacement.

Remaining gaps to close before a stronger claim: larger non-English sample
(1 TR + 1 DE so far), and tightening the decisions-vs-suggestions boundary
(prompt-level).

---

## Nemotron-3-Nano-4B Floor Evaluation (2026-09-15)

A different question from the rest of this doc, at a lower bar. millet 0.21.0
added a summary fallback chain — `tinfoil → venice → near → ollama` (two
attested TEE providers between the primary enclave and the local model). The
`ollama` tier is the **zero-network privacy floor**: it runs only when every
attested provider is unreachable, so "degrade quality, never confidentiality"
already applies. NVIDIA's Jetson-optimized `nemotron-3-nano` was proposed for
it because it is the one Nemotron that fits the saray target (RTX 5070, 12 GB).
The question here is narrow: **on a real, hard meeting, is the 4B floor an
acceptable last resort?**

### Method

One anonymized meeting **J** with an existing on-disk production summary:
EN, ~15.2K tokens (60,652 chars / 10,378 words), ~28 speakers, 1h18m — a dense
all-hands, the hardest case in this doc. Baseline is the production **primary**
`glm-5-3-flash (TEE)` summary (not a cloud model). Generated via
`millet.summarize.summarize()` — the same entry point production uses (same
prompts, two-pass logic, dynamic `num_ctx`) — on the real target hardware
(RTX 5070, 12 GB) through the local Ollama tier. Meetings are anonymized;
only metrics are shown.

Dynamic `num_ctx` resolved to ≈ 19,456 — well under the 65,536 ceiling — so
**context fit was NOT the limiter; model capacity was.** Coverage is scored as
topic-presence against 28 distinct meeting-wide subjects the baseline captured
(**T** = topics/28); other columns are worst duplicate-line count
(degeneration), a hallucination spot-check, format + JSON-frontmatter
compliance, output length, and time / peak VRAM.

### Results

| Config | Cov (T/28) | Words | Worst dup | Hallucination | Fmt / JSON | Time / VRAM |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| baseline `glm-5-3-flash (TEE)` | 28/28 | 3854 | 1× | none | ok | — |
| `nemotron-3-nano:4b` two-pass | 6/28 | 380 | 1× | fabricated decision | no JSON | 59 s / 3.1 GB |
| `nemotron-3-nano:4b` single-pass | 12/28 | 603 | — | format-broken | no JSON | 10 s / 3.4 GB |
| `nemotron-3-nano:4b-q8_0` two-pass | 10/28 | 885 | 6× | invented speaker | no JSON | 39 s / 4.3 GB |

- **Complementary blindness.** Two-pass captured the meeting's *front* segment;
  single-pass captured its *back*; the entire *middle* (roughly ten distinct
  topics) was absent from all three. The model fixates on one contiguous span
  and drops the rest — small-model attention collapse on a long input, not
  truncation (context was sized to fit). This is why the two passes almost
  never overlap.
- **Precision fails, not just recall.** The two-pass run emitted a confident,
  correctly-formatted *decision* that never happened in the meeting (a
  fabricated pricing figure; not reproduced here per the metrics-only policy).
  Structural metrics would have scored this run as passing — correct 5 sections
  and 22 topic bullets — which is exactly the trap the grounded TEE evaluation
  was built to avoid.
- **q8 is not the fix.** The higher-precision quant of the *same 4B model*
  covered no more of the meeting and degenerated into repetition (one action
  item repeated 6×). The limiter is parameter count / attention, which
  quantization does not change.
- **It does fit and it is fast.** ≤ 4.3 GB peak VRAM on a 12 GB card, ≤ 59 s.
  The 24 GB `nemotron-3-nano:latest` / `:30b` (Omni) tags do **not** fit 12 GB;
  only the `:4b` family does.

### Verdict

`nemotron-3-nano:4b` is **fit as a last-resort, zero-network floor and nothing
more.** When it runs at all, some summary beats no summary, and the tier only
activates when every attested provider is down. It does **not** approach the
standing `qwen3.8:27b` recommendation (which needs ≥18 GB and so will not fit
the 12 GB floor target anyway), and it should not be promoted toward the
primary/attested path. Operators should know that a network-outage summary of
a *large* meeting will be partial and may contain a fabricated line.

Follow-ups if the floor ever needs to be better (not done here): (1) chunked /
map-reduce summarization so a long transcript is summarized in segments the 4B
model can hold, then merged — this addresses the attention collapse directly,
which context sizing does not; (2) a head-to-head against the prior default
`qwen3.5:9b` (~6.6 GB, also fits 12 GB) on this same meeting before committing
the floor's default model.

> **Env note (millet 0.21.0):** the fallback floor's model is now set with
> `MILLET_OLLAMA_MODEL` (e.g. `nemotron-3-nano:4b`), which overrides **only**
> the `ollama` tier — distinct from `MILLET_SUMMARY_MODEL` (used elsewhere in
> this doc), which targets the user's *chosen* backend. This keeps the floor's
> per-host model choice from leaking into the primary/attested tiers. On a
> 12 GB GPU, prefer a `:4b`-family Nemotron or `qwen3.5:9b`; the ≥18 GB
> `qwen3.8:27b` recommendation above does not apply to the floor.

---

## 12 GB Floor Bake-Off (2026-09-15) — the floor should be `qwen3.5:9b`, not Nemotron

The Nemotron floor eval above left ~7 GB of the 12 GB card unused, so a natural
question was whether a bigger model that still fits would fix the coverage
collapse. **No larger Nemotron fits** — the Nemotron 3 Nano family jumps 4B →
31.6B (24 GB) with nothing between, and `:4b-bf16` (8 GB) is the same 4B
capacity. So the headroom was spent on two higher-capacity **non-Nemotron**
models that fit 12 GB, run on the same meeting **J** through the same
`millet.summarize()` two-pass path on the RTX 5070.

| Model | Size | Cov (T/24) | Words | Worst dup | Halluc. | Fit | Time / VRAM |
|-------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| baseline `glm-5-3-flash (TEE)` | — | 16/24 | 3854 | 1× | none | — | — |
| `nemotron-3-nano:4b` | 2.8 GB | 6/24 | 380 | 1× | fabricated decision | GPU | 59 s / 3.1 GB |
| **`qwen3.5:9b`** | 6.6 GB | **15/24** | 1699 | 1× | none | **GPU** | **100 s / 6.7 GB** |
| `qwen3:14b` | 9.3 GB | 13/24 | 2337 | 1× | none | **CPU-spill** | 626 s / 10 GB |

(Coverage here is scored against 24 topic *groups* on this meeting, so the
numbers are internally comparable but not identical to the /28 scale in the
Nemotron section above; the relative ordering is the point.)

- **`qwen3.5:9b` is the clear winner.** It reaches near-baseline coverage
  (15/24 vs 16/24) — capturing the whole meeting including the middle segment
  that *every* Nemotron config dropped — with correct 5-section format, no
  looping, and no hallucination, while fitting the GPU fully at 6.7 GB and
  running in 100 s. The limiter for the floor was model *capacity*; 9B clears
  the bar where 4B does not.
- **`qwen3:14b` does not fit and is not better.** At 32k context Ollama reported
  `26%/74% CPU/GPU` (needs ~14 GB), so it **spilled to CPU** — 626 s (6× slower)
  for *lower* coverage than 9B (13/24). A floor that takes >10 min when the
  network is down is barely usable; 14B is the wrong choice on a 12 GB card.
- **This makes the Nemotron floor a regression for large meetings.**
  `qwen3.5:9b` is the *prior* default that `nemotron-3-nano:4b` replaced. On
  short meetings the 4B is fine and faster; on a dense all-hands it drops ~⅔ of
  the content. The floor's default should therefore be **`qwen3.5:9b`** on a
  12 GB host — set via `MILLET_OLLAMA_MODEL=qwen3.5:9b`. Nemotron `:4b` stays a
  valid choice only where VRAM is tighter than 6.6 GB or speed matters more than
  completeness.

**Action taken (saray, RTX 5070):** `MILLET_OLLAMA_MODEL` switched from
`nemotron-3-nano:4b` to `qwen3.5:9b`.

---

## Appendix: Model-Specific Notes

### qwen3.8:27b (recommended local model, 2026-08-15)
- 18 GB, fits a 24 GB GPU; needs Ollama ≥ 0.32.12; vision + tools + thinking.
- Best local results to date: matches/exceeds cloud baseline coverage on
  EN/TR/DE via the two-pass flow; 5/5 format; localized non-English headers.
- **Requires the v0.15.1 Pass-1 reserve fix** or it silently truncates.
- Weaknesses: ~2× Sonnet latency; mild over-inclusion of "decisions".
- Won't fit a 12 GB GPU — choose a smaller model there.

### nemotron-3-nano:4b (zero-network fallback floor only, 2026-09-15)
- 2.8 GB (`:4b`) / 4.2 GB (`:4b-q8_0`); fits a 12 GB GPU (RTX 5070) with headroom.
  The 24 GB `:latest` / `:30b` (Omni) tags do **not** fit 12 GB.
- Text-only in Ollama: it cannot load the multimodal Nemotrons' mmproj vision
  weights, so the frames (screen-recording) path never routes to the floor.
- For the `ollama` fallback tier of millet 0.21.0's `tinfoil → venice → near →
  ollama` chain — set via `MILLET_OLLAMA_MODEL`. Fast (≤ 59 s, ≤ 4.3 GB VRAM).
- **Weak on large meetings:** ≤ 43 % topic coverage vs the primary baseline on
  a hard 1h18m all-hands (attention collapse, not context truncation); a
  fabricated "decision" in two-pass; looping in `:4b-q8_0`. See the
  "Nemotron-3-Nano-4B Floor Evaluation (2026-09-15)" section above. Acceptable
  as a last resort (some summary beats none), not as a general summarizer.

### gpt-oss:20b (prior best local; now a fallback)
- 13 GB, fits comfortably in 24 GB VRAM
- 128K context — handles all meeting sizes without truncation
- MoE architecture with MXFP4 native quantization (OpenAI)
- Was the best local model here until Qwen3.8 (2026-08-15); follows format well in double-pass, reasonable speed. Now a fallback / small-GPU option (fits 13 GB where Qwen3.8's 18 GB does not).
- Persistent weakness: hallucinated proper nouns (entity names, dates)
- Ollama ID: `gpt-oss:20b` (9M pulls, well-tested)

### qwen3.6:27b
- 17 GB, fits in 24 GB VRAM
- 262K context
- In single-pass mode: completely ignores system prompt and offers a menu instead of summarizing
- In double-pass mode: produces good results but very slow (353s)
- Hallucinated speaker names (close-but-wrong proper nouns)
- Not recommended for meetscribe due to speed, but viable as a fallback

### qwen3.5:27b
- 17 GB, requires partial CPU offload at 16K+ context (23 GB total with KV cache)
- Very slow when partially offloaded (~8 t/s)
- Same architecture as qwen3.6, similar failure modes
- Not recommended for meetscribe

### gemma4-26b-16k
- 17 GB, 16K context limit — truncates most meetings
- Fast (137 t/s) but context-limited
- Does not follow the 5-section format
- Not suitable for meetscribe summarization (context too small)

### Sonnet 4.6 (current production model)
- Consistently perfect format compliance
- Comprehensive topic coverage (11+ topics on complex meetings)
- No hallucinations observed
- 40-90s per meeting via claudemax proxy
- Most concise/polished output and the strictest "explicitly agreed" bar for decisions; ~2× faster than local Qwen3.8. Remains the production default. As of 2026-08-15, **Qwen3.8-27B (double-pass, v0.15.1+) matches it on coverage** and is a credible fully-offline replacement on a ≥18 GB GPU (see the Qwen3.8 section above).
