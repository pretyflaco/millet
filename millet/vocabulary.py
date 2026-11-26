"""Custom domain vocabulary, ASR prompting, and post-ASR alias normalization.

Improves recognition of proper nouns, project names, acronyms, and domain
jargon across three layers:

1. **ASR decoding bias** — builds a compact, language-neutral ``initial_prompt``
   (WhisperX / MLX / Tinfoil ASR) and ``hotwords`` string (WhisperX /
   faster-whisper CTranslate2 beam search) so the model prefers the canonical
   spelling during decoding.
2. **Deterministic post-ASR normalization** — applies explicit ``aliases``
   (known recurring mishearings such as ``vzr`` / ``vizier`` -> ``Vezir``) to
   both segment text and word-level timestamps, keeping ``.txt``, ``.srt``,
   ``.json``, and burned clip captions consistent.  Works on every ASR backend,
   including backends without prompting (Parakeet).
3. **Summary glossary** — formats a compact terminology block for the summary
   LLM so it resolves phonetic ASR errors in summaries and frontmatter without
   hallucinating unmentioned terms.

Default file: ``~/.config/meet/vocabulary.json`` (or per-team via
``~/.config/meet/<team>/vocabulary.json`` / ``$MILLET_VOCABULARY_PATH``).

Kept strictly stdlib-only (no torch/numpy) so lightweight callers and unit
tests can import it without loading the ML stack.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from millet.crosstalk import is_reserved_label

log = logging.getLogger(__name__)

# Whisper's decoder prompt is capped at 224 tokens (~700-900 chars for Latin
# scripts, less for CJK/Arabic).  Stay well under that ceiling so leading terms
# are never silently truncated.
DEFAULT_MAX_PROMPT_CHARS = 700
DEFAULT_MAX_GLOSSARY_TERMS = 80

_RAW_SPEAKER_RE = re.compile(r"^(?:YOU|REMOTE(?:_\d+)?|SPEAKER_\d+|UNKNOWN)$", re.IGNORECASE)
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
_EDGE_PUNCT_RE = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.UNICODE)


def _clean_str(value: object) -> str:
    """Strip whitespace and control characters; collapse internal whitespace."""
    if not isinstance(value, str):
        return ""
    cleaned = _CTRL_RE.sub(" ", value)
    return " ".join(cleaned.split())


@dataclass(frozen=True)
class Vocabulary:
    """Team or user custom vocabulary for ASR and summarization."""

    terms: tuple[str, ...] = ()
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    glossary: dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (self.terms or self.aliases or self.glossary)

    @property
    def all_terms(self) -> tuple[str, ...]:
        """Deduplicated canonical terms in priority order.

        Explicit ``terms`` come first, followed by canonical keys from
        ``aliases`` and ``glossary``, deduplicated case-insensitively.
        """
        out: list[str] = []
        seen: set[str] = set()
        for source in (self.terms, tuple(self.aliases), tuple(self.glossary)):
            for raw in source:
                term = _clean_str(raw)
                if not term:
                    continue
                key = term.casefold()
                if key not in seen:
                    seen.add(key)
                    out.append(term)
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": list(self.terms),
            "aliases": {k: list(v) for k, v in self.aliases.items()},
            "glossary": dict(self.glossary),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Vocabulary:
        """Build a normalized Vocabulary from a parsed JSON mapping."""
        terms: list[str] = []
        seen_terms: set[str] = set()
        raw_terms = data.get("terms")
        if isinstance(raw_terms, list):
            for item in raw_terms:
                t = _clean_str(item)
                if t and t.casefold() not in seen_terms:
                    seen_terms.add(t.casefold())
                    terms.append(t)

        aliases: dict[str, tuple[str, ...]] = {}
        raw_aliases = data.get("aliases")
        if isinstance(raw_aliases, Mapping):
            for raw_canon, raw_variants in raw_aliases.items():
                canon = _clean_str(raw_canon)
                if not canon:
                    continue
                if isinstance(raw_variants, str):
                    variants_in: Sequence[object] = [raw_variants]
                elif isinstance(raw_variants, list):
                    variants_in = raw_variants
                else:
                    continue
                cleaned_variants: list[str] = []
                seen_v: set[str] = {canon.casefold()}
                for v in variants_in:
                    cv = _clean_str(v)
                    if cv and cv.casefold() not in seen_v:
                        seen_v.add(cv.casefold())
                        cleaned_variants.append(cv)
                if cleaned_variants:
                    aliases[canon] = tuple(cleaned_variants)

        glossary: dict[str, str] = {}
        raw_glossary = data.get("glossary")
        if isinstance(raw_glossary, Mapping):
            for raw_term, raw_desc in raw_glossary.items():
                term = _clean_str(raw_term)
                desc = _clean_str(raw_desc)
                if term and desc:
                    glossary[term] = desc

        return cls(terms=tuple(terms), aliases=aliases, glossary=glossary)


def merge_vocabularies(*vocabs: Vocabulary) -> Vocabulary:
    """Merge multiple vocabularies, earlier entries winning on conflicts."""
    terms: list[str] = []
    seen_terms: set[str] = set()
    aliases: dict[str, list[str]] = {}
    alias_keys: dict[str, str] = {}
    glossary: dict[str, str] = {}
    glossary_keys: dict[str, str] = {}

    for v in vocabs:
        if v is None or v.is_empty:
            continue
        for t in v.terms:
            ct = _clean_str(t)
            if ct and ct.casefold() not in seen_terms:
                seen_terms.add(ct.casefold())
                terms.append(ct)
        for canon, variants in v.aliases.items():
            cc = _clean_str(canon)
            if not cc:
                continue
            key = cc.casefold()
            target = alias_keys.setdefault(key, cc)
            bucket = aliases.setdefault(target, [])
            existing = {x.casefold() for x in bucket} | {target.casefold()}
            for var in variants:
                cv = _clean_str(var)
                if cv and cv.casefold() not in existing:
                    existing.add(cv.casefold())
                    bucket.append(cv)
        for term, desc in v.glossary.items():
            ct = _clean_str(term)
            cd = _clean_str(desc)
            if ct and cd:
                key = ct.casefold()
                if key not in glossary_keys:
                    glossary_keys[key] = ct
                    glossary[ct] = cd

    return Vocabulary(
        terms=tuple(terms),
        aliases={k: tuple(v) for k, v in aliases.items() if v},
        glossary=glossary,
    )


def load_vocabulary(
    path: Path | None = None,
    team: str | None = None,
) -> Vocabulary:
    """Load vocabulary from disk.  Returns an empty Vocabulary if missing/invalid.

    Accepts canonical JSON (``{"terms": [...], "aliases": {...}, "glossary": {...}}``)
    as well as plain newline/comma-separated text word lists when an explicit
    ``.txt`` file is passed.  Never raises on unreadable or malformed files so
    a bad vocabulary entry can never fail a transcription.
    """
    if path is None:
        from . import paths

        p = paths.vocabulary_path(team)
    else:
        p = Path(path)

    if not p.is_file():
        return Vocabulary()

    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Could not read vocabulary file %s: %s", p, exc)
        return Vocabulary()

    stripped = raw.strip()
    if not stripped:
        return Vocabulary()

    # Plain-text fallback for ad-hoc `--vocabulary words.txt` files.
    if not stripped.startswith("{") and p.suffix.lower() != ".json":
        plain_terms: list[str] = []
        for line in stripped.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            for part in line.split(","):
                t = _clean_str(part)
                if t:
                    plain_terms.append(t)
        return Vocabulary.from_dict({"terms": plain_terms})

    try:
        data = json.loads(stripped)
    except ValueError as exc:
        log.warning("Could not parse vocabulary JSON %s: %s", p, exc)
        return Vocabulary()

    if not isinstance(data, Mapping):
        log.warning("Vocabulary file %s must contain a JSON object", p)
        return Vocabulary()

    return Vocabulary.from_dict(data)


def save_vocabulary(
    vocab: Vocabulary,
    path: Path | None = None,
    team: str | None = None,
) -> Path:
    """Atomically write *vocab* to disk as formatted JSON."""
    if path is None:
        from . import paths

        p = paths.vocabulary_path(team)
    else:
        p = Path(path)

    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(vocab.to_dict(), indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if p.exists():
            try:
                os.chmod(tmp, p.stat().st_mode & 0o777)
            except OSError:
                pass
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def speaker_names_from_profiles(
    profiles_path: Path | None = None,
    team: str | None = None,
) -> tuple[str, ...]:
    """Read enrolled speaker names from ``speaker_profiles.json`` (stdlib-only).

    Used to seed ASR prompting with known team participant names without
    importing torch/pyannote/numpy.  Filters out raw placeholders and reserved
    labels (YOU, REMOTE, SPEAKER_00, CROSSTALK).
    """
    if profiles_path is None:
        from . import paths

        p = paths.profiles_path(team)
    else:
        p = Path(profiles_path)

    if not p.is_file():
        return ()

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()

    if not isinstance(data, Mapping):
        return ()

    names: list[str] = []
    seen: set[str] = set()
    for raw_name in data:
        name = _clean_str(raw_name)
        if not name or _RAW_SPEAKER_RE.match(name) or is_reserved_label(name):
            continue
        key = name.casefold()
        if key not in seen:
            seen.add(key)
            names.append(name)
    return tuple(names)


# ─── Layer 1: ASR prompting & hotwords ──────────────────────────────────────


def _collect_bounded_terms(
    vocab: Vocabulary,
    extra_terms: Sequence[str],
    sep: str,
    max_chars: int,
    suffix: str = "",
) -> list[str]:
    """Collect deduplicated terms in priority order fitting within *max_chars*."""
    selected: list[str] = []
    seen: set[str] = set()
    used = len(suffix)

    for source in (vocab.all_terms, extra_terms):
        for raw in source:
            term = _clean_str(raw)
            if not term:
                continue
            key = term.casefold()
            if key in seen:
                continue
            added = len(term) + (len(sep) if selected else 0)
            if used + added > max_chars:
                continue
            seen.add(key)
            selected.append(term)
            used += added
    return selected


def build_initial_prompt(
    vocab: Vocabulary,
    *,
    extra_terms: Sequence[str] = (),
    base_prompt: str | None = None,
    max_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> str | None:
    """Build a language-neutral Whisper ``initial_prompt`` from *vocab*.

    Deliberately avoids an English prefix like ``"Vocabulary:"`` so non-English
    recordings are not biased toward English or prone to hallucinating a label
    during silence.  Returns ``None`` when no prompt text is available.
    """
    cleaned_base = _clean_str(base_prompt) if base_prompt else ""
    if cleaned_base:
        remaining = max_chars - len(cleaned_base) - 2
        if remaining <= 0:
            return cleaned_base[:max_chars]
        terms = _collect_bounded_terms(vocab, extra_terms, ", ", remaining, suffix=".")
        if not terms:
            return cleaned_base
        sep = " " if cleaned_base.endswith((".", "!", "?", ";", ":")) else ". "
        return f"{cleaned_base}{sep}{', '.join(terms)}."

    terms = _collect_bounded_terms(vocab, extra_terms, ", ", max_chars, suffix=".")
    if not terms:
        return None
    return ", ".join(terms) + "."


def build_hotwords(
    vocab: Vocabulary,
    *,
    extra_terms: Sequence[str] = (),
    base_hotwords: str | None = None,
    max_chars: int = DEFAULT_MAX_PROMPT_CHARS,
) -> str | None:
    """Build a space-separated ``hotwords`` string for faster-whisper / CTranslate2.

    Space-separated (not comma-separated) so the tokenizer does not boost
    spurious comma tokens during beam search.  Returns ``None`` when empty.
    """
    cleaned_base = _clean_str(base_hotwords) if base_hotwords else ""
    if cleaned_base:
        remaining = max_chars - len(cleaned_base) - 1
        if remaining <= 0:
            return cleaned_base[:max_chars]
        terms = _collect_bounded_terms(vocab, extra_terms, " ", remaining)
        if not terms:
            return cleaned_base
        return f"{cleaned_base} {' '.join(terms)}"

    terms = _collect_bounded_terms(vocab, extra_terms, " ", max_chars)
    if not terms:
        return None
    return " ".join(terms)


# ─── Layer 2: Deterministic post-ASR alias normalization ──────────────────────


def _build_alias_lookup(
    aliases: Mapping[str, Sequence[str]],
) -> tuple[re.Pattern[str] | None, dict[str, str]]:
    """Compile a single-pass longest-first regex and casefolded replacement map."""
    pairs: dict[str, str] = {}
    for raw_canon, variants in aliases.items():
        canon = _clean_str(raw_canon)
        if not canon:
            continue
        for raw_var in variants:
            var = _clean_str(raw_var)
            if not var or var.casefold() == canon.casefold():
                continue
            pairs.setdefault(var.casefold(), canon)

    if not pairs:
        return None, {}

    # Longest aliases first so multi-word forms win over shorter substrings.
    ordered = sorted(pairs.keys(), key=len, reverse=True)
    # Collapse flexible whitespace in multi-word aliases (e.g. "pie  a note").
    escaped = [r"\s+".join(re.escape(part) for part in k.split()) for k in ordered]
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)", re.IGNORECASE)
    return pattern, pairs


def _match_canonical_casing(matched: str, canonical: str) -> str:
    """Return *canonical*, preserving sentence-initial capitalization when safe.

    Proper nouns / acronyms (``Vezir``, ``NIP-46``, ``iOS``, ``PyTorch``) keep
    their exact canonical casing.  All-lowercase terms (``millet``) capitalize
    their first character when the matched ASR token started uppercase.
    """
    if (
        matched
        and canonical
        and matched[0].isupper()
        and canonical[0].islower()
        and canonical == canonical.lower()
    ):
        return canonical[0].upper() + canonical[1:]
    return canonical


def apply_aliases_to_text(
    text: str,
    aliases: Mapping[str, Sequence[str]],
) -> tuple[str, int]:
    """Replace known mishearings in *text* in a single non-cascading pass.

    Returns ``(updated_text, replacement_count)``.
    """
    if not text or not aliases:
        return text, 0
    pattern, lookup = _build_alias_lookup(aliases)
    if pattern is None:
        return text, 0

    count = 0

    def _repl(match: re.Match[str]) -> str:
        nonlocal count
        matched = match.group(0)
        key = " ".join(matched.split()).casefold()
        canon = lookup.get(key)
        if canon is None:
            return matched
        count += 1
        return _match_canonical_casing(matched, canon)

    return pattern.sub(_repl, text), count


def _split_edge_punct(token: str) -> tuple[str, str, str]:
    m = _EDGE_PUNCT_RE.match(token)
    if not m:
        return "", token, ""
    return m.group(1), m.group(2), m.group(3)


def apply_aliases_to_words(
    words: list[dict[str, Any]],
    aliases: Mapping[str, Sequence[str]],
) -> int:
    """Apply single- and multi-word alias replacements to WhisperX word dicts.

    Mutates *words* in place so burned clip captions and word-level JSON stay
    consistent with ``seg.text``.  Multi-word aliases merge consecutive tokens,
    keeping ``start`` from the first token, ``end`` from the last, and
    preserving leading/trailing punctuation.

    Returns the number of replacements applied.
    """
    if not words or not aliases:
        return 0

    # Index aliases by tokenized length (longest first).
    by_tokens: dict[tuple[str, ...], str] = {}
    max_len = 0
    for raw_canon, variants in aliases.items():
        canon = _clean_str(raw_canon)
        if not canon:
            continue
        for raw_var in variants:
            var = _clean_str(raw_var)
            if not var or var.casefold() == canon.casefold():
                continue
            toks = tuple(var.casefold().split())
            if toks:
                by_tokens.setdefault(toks, canon)
                if len(toks) > max_len:
                    max_len = len(toks)

    if not by_tokens:
        return 0

    # Pre-split edge punctuation for each word dict.
    parsed = [_split_edge_punct(str(w.get("word") or "")) for w in words]
    out: list[dict[str, Any]] = []
    replacements = 0
    i = 0
    n = len(words)

    while i < n:
        matched_span = 0
        matched_canon = ""
        # Try longest span first.
        span_limit = min(max_len, n - i)
        for span in range(span_limit, 0, -1):
            # Internal tokens in a multi-word match must not have sentence
            # punctuation breaking the phrase.
            if span > 1:
                broken = False
                for k in range(i, i + span - 1):
                    if parsed[k][2].strip():
                        broken = True
                        break
                for k in range(i + 1, i + span):
                    if parsed[k][0].strip():
                        broken = True
                        break
                if broken:
                    continue
            cores = tuple(parsed[i + k][1].casefold() for k in range(span))
            if not all(cores):
                continue
            canon = by_tokens.get(cores)
            if canon is not None:
                matched_span = span
                matched_canon = canon
                break

        if matched_span == 0:
            out.append(words[i])
            i += 1
            continue

        lead = parsed[i][0]
        trail = parsed[i + matched_span - 1][2]
        first_core = parsed[i][1]
        replacement_word = f"{lead}{_match_canonical_casing(first_core, matched_canon)}{trail}"

        if matched_span == 1:
            updated = dict(words[i])
            updated["word"] = replacement_word
            out.append(updated)
        else:
            first = words[i]
            last = words[i + matched_span - 1]
            merged = dict(first)
            merged["word"] = replacement_word
            if "end" in last:
                merged["end"] = last["end"]
            scores = [
                float(words[i + k]["score"])
                for k in range(matched_span)
                if isinstance(words[i + k].get("score"), (int, float))
            ]
            if scores:
                merged["score"] = round(sum(scores) / len(scores), 4)
            out.append(merged)

        replacements += 1
        i += matched_span

    if replacements:
        words[:] = out
    return replacements


def apply_vocabulary_to_transcript(transcript: Any, vocab: Vocabulary) -> int:
    """Apply *vocab*'s deterministic aliases across all transcript segments.

    Updates both ``seg.text`` and ``seg.words`` in place.  Returns total
    text replacements applied.
    """
    if vocab is None or not vocab.aliases:
        return 0
    total = 0
    for seg in getattr(transcript, "segments", ()):
        if getattr(seg, "text", None):
            new_text, n = apply_aliases_to_text(seg.text, vocab.aliases)
            if n:
                seg.text = new_text
                total += n
        words = getattr(seg, "words", None)
        if isinstance(words, list) and words:
            apply_aliases_to_words(words, vocab.aliases)
    return total


# ─── Layer 3: Summary LLM glossary block ────────────────────────────────────


def build_summary_glossary_block(
    vocab: Vocabulary,
    *,
    max_terms: int = DEFAULT_MAX_GLOSSARY_TERMS,
) -> str:
    """Format a compact terminology block to append to summary system prompts.

    Returns an empty string when *vocab* is empty so prompts remain byte-for-byte
    unchanged when no vocabulary is configured.
    """
    if vocab is None or vocab.is_empty:
        return ""

    # Index aliases and glossary by casefolded canonical term.
    aliases_by_key = {k.casefold(): v for k, v in vocab.aliases.items()}
    glossary_by_key = {k.casefold(): v for k, v in vocab.glossary.items()}

    lines: list[str] = []
    for term in vocab.all_terms[:max_terms]:
        key = term.casefold()
        entry = f"- {term}"
        variants = aliases_by_key.get(key)
        if variants:
            entry += f" (often misheard by ASR as: {', '.join(variants)})"
        desc = glossary_by_key.get(key)
        if desc:
            entry += f" — {desc}"
        lines.append(entry)

    if not lines:
        return ""

    return (
        "KNOWN TERMINOLOGY (team vocabulary — ASR may phonetically mishear these):\n"
        + "\n".join(lines)
        + "\n"
        "When the transcript clearly contains a phonetic mishearing of one of these terms, "
        "use the canonical spelling above. Do NOT introduce any term from this list unless "
        "it was actually discussed in the meeting."
    )
