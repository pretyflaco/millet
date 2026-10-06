"""Meeting summary generation using LLMs.

Every supported backend keeps meeting content private:
  - tinfoil: Hardware-attested TEE inference (requires TINFOIL_API_KEY).
             Prompts are encrypted into the enclave; neither the model
             provider nor the cloud operator can read them.  Default.
  - venice / near: Decorrelated hardware-attested TEE fallbacks (Intel
             TDX + NVIDIA H100/H200 Confidential Computing; millet verifies
             each enclave's NRAS attestation + freshness nonce itself).
  - ollama:  Local Ollama server (free, fully local, never leaves the box).

The non-private cloud backends (claudemax, openrouter, openai) were removed
in 0.19.0.  A TEE model measurably out-summarized Sonnet 4.6 on grounded
precision and recall across EN/DE/TR, so routing meeting content through
a provider that can read it no longer bought anything -- see
docs/tee-summarization-evaluation.md.

Fallback chain: tinfoil -> venice -> near -> ollama (DEFAULT_FALLBACK_ORDER,
overridable via MILLET_SUMMARY_FALLBACK_ORDER).  Every destination is
private — three attested TEEs plus the local box — so the chain can never
leak content to a third party; it can only trade quality, and it always
says so (fallback_used + backend/model in the summary meta sidecar; a local
fallback is recorded as ollama/<model>, never labelled a TEE).

Since 0.21.2 an explicitly requested preset follows the same chain.  The
preset names are pure aliases of the default (0.19.0), so pinning them to
one provider converted recoverable outages into hard failures while
protecting nothing: "confidential" means no third party can read the
content, and every backend in the chain satisfies that — a local model as
strictly as a TEE.  Within the tinfoil backend, a drained enclave pool
falls back once to a sibling TEE model (DEFAULT_TINFOIL_FALLBACK_MODEL)
before giving up.

Configuration precedence (highest to lowest):
  1. Explicit keyword arguments / CLI flags (--summary-backend, --summary-model)
  2. Environment variables (MILLET_SUMMARY_BACKEND, MILLET_SUMMARY_MODEL)
  3. Hardcoded defaults (tinfoil / deepseek-v4-1-flash)
"""

from __future__ import annotations

import base64
import importlib.util
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from millet.frontmatter import FrontmatterContext

import re

import requests

# ─── Constants ──────────────────────────────────────────────────────────────

# Ollama defaults.
#
# The local backend is the zero-network privacy floor of the fallback chain:
# when every attested TEE provider is unreachable it still produces a summary
# on the operator's own hardware, so confidentiality never degrades (only
# quality can).  To point it at a lighter/faster local model — e.g. NVIDIA's
# Jetson-optimized Nemotron, which fits ~12 GB VRAM — set MILLET_OLLAMA_MODEL
# (e.g. nemotron-3-nano:4b) to point JUST the fallback floor at it without
# touching the primary backend or this global default.  The global default here
# stays qwen3.5:9b (broadly pulled) and only moves once a model is benchmarked
# to win everywhere.  NOTE the local tier is TEXT-ONLY: Ollama cannot load the
# separate mmproj vision weights the multimodal Nemotrons need, so it is
# registered supports_vision=False and the frames (screen-recording) path is
# never routed here — it stays on a vision-capable attested tier.
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 600  # 10 minutes max

# Tinfoil TEE defaults (hardware-enforced prompt privacy)
# Tinfoil deprecated glm-5-3-flash on 2026-10-09 (announced 2026-10-06).
# deepseek-v4-1-flash was the runner-up in the grounded TEE evaluation
# (docs/tee-summarization-evaluation.md): equal precision, ~4 pp lower
# recall, faster and far steadier latency (max 115 s vs 353 s).
DEFAULT_TINFOIL_MODEL = "deepseek-v4-1-flash"
# Sibling TEE model tried when the primary model's enclave pool is
# unavailable.  Deliberately a different model family (GLM, not DeepSeek) so
# a drained pool of one family — exactly what retired glm-5-2 — does not take
# out both.  glm-5-3 is text-only (a frames run degrades to text-only on the
# sibling, never fails), ~9x the cost and ~3x slower, which is acceptable
# for a path that only runs when the primary is down.
# This is NOT a privacy fallback: both models run inside the TEE, so the
# `confidential` contract holds.  It is never silent (see fallback_used).
DEFAULT_TINFOIL_FALLBACK_MODEL = "glm-5-3"
TINFOIL_API_KEY_ENV = "TINFOIL_API_KEY"
_TINFOIL_KEY_FILE = Path.home() / "models" / "tinfoil" / "tinfoil.txt"


def _resolve_tinfoil_api_key() -> str | None:
    """Resolve Tinfoil API key from env var or standard key file."""
    return _resolve_api_key(TINFOIL_API_KEY_ENV, _TINFOIL_KEY_FILE)


def _resolve_api_key(env_var: str, key_file: Path) -> str | None:
    """Resolve an API key from an env var, falling back to a 0600 key file."""
    key = os.environ.get(env_var)
    if key:
        return key.strip()
    if key_file.exists():
        try:
            return key_file.read_text().strip()
        except OSError:
            pass
    return None


# ─── Attested OpenAI-compatible fallback backends (venice, near) ────────────
#
# Decorrelated TEE providers so Tinfoil is not a single point of failure.
# Both speak the OpenAI API, so one generic path (_summarize_attested_oai)
# drives them; they differ only in this table.  Before a summary from one is
# badged "attested", millet.attestation.verify_attestation checks the
# enclave's NVIDIA NRAS evidence + our freshness nonce itself (see that
# module for the deliberate ceiling on full TDX-quote validation).
#
#   base_url        OpenAI-compatible endpoint
#   key_env/file    API key (env var, then 0600 key file — Tinfoil pattern)
#   model           default text model (GLM 5.3 flash equivalent)
#   vision_model    model to use when frames are present; None = no vision
#   attestation_url per-request attestation document endpoint ({model} filled)
#   catalog_url     OpenAI /models list, for the model pre-flight
#   confidentiality short provenance note on the trust model (recorded, honest)
# Values below were verified against the live APIs on 2026-09-15 (see the
# Phase-0 probe): base URLs, the exact model ids each provider serves, and the
# attestation-endpoint shapes.  Venice's attested models carry an ``e2ee-``
# prefix — the plain id 404s the attestation endpoint.  NEAR uses ``z-ai/``
# (slash), Venice uses ``z-ai-`` / ``e2ee-`` (dashes); they are NOT the same
# string as Tinfoil's ``glm-5-3-flash``, so no id collision to worry about.
ATTESTED_BACKENDS: dict[str, dict[str, object]] = {
    "venice": {
        "base_url": "https://api.venice.ai/api/v1",
        "key_env": "VENICE_API_KEY",
        "key_file": Path.home() / "models" / "venice" / "api-key.txt",
        "model": "e2ee-glm-5-3-flash",
        "vision_model": "e2ee-qwen3-vl-30b-a3b-p",
        "attestation_url": "https://api.venice.ai/api/v1/tee/attestation?model={model}",
        "catalog_url": "https://api.venice.ai/api/v1/models",
        "confidentiality": "e2ee-gateway-tls",  # gateway TLS terminates outside TEE; app-layer E2EE
    },
    "near": {
        "base_url": "https://cloud-api.near.ai/v1",
        "key_env": "NEAR_AI_API_KEY",
        "key_file": Path.home() / "models" / "near" / "api-key.txt",
        "model": "z-ai/glm-5.3-flash",
        "vision_model": None,  # NEAR gateway advertises no vision here — text-only tier
        "attestation_url": (
            "https://cloud-api.near.ai/v1/attestation/report?model={model}&signing_algo=ecdsa"
        ),
        "catalog_url": "https://cloud-api.near.ai/v1/models",
        "confidentiality": "tee-terminated-tls",  # direct-completions terminate TLS in the TEE
    },
}


def _resolve_attested_api_key(backend: str) -> str | None:
    """Resolve the API key for an attested OpenAI-compatible backend."""
    cfg = ATTESTED_BACKENDS[backend]
    return _resolve_api_key(str(cfg["key_env"]), cfg["key_file"])  # type: ignore[arg-type]


# Catalog pre-flight: Tinfoil retires models out from under us (deepseek-v4-pro
# in 2026-07, glm-5-2 in 2026-09 — the latter with no deprecation notice, it
# simply stopped having an engine and started answering 503).  The catalog
# endpoint self-documents both states, so check it once per process and warn
# loudly rather than discovering the problem when a user's job dies.
TINFOIL_MODELS_URL = "https://inference.tinfoil.sh/v1/models"
_TINFOIL_CATALOG_CHECKED: set[str] = set()
logger = logging.getLogger("millet.summarize")


def verify_tinfoil_model(model: str, *, timeout: int = 10) -> str | None:
    """Warn if ``model`` is missing from or deprecated in Tinfoil's catalog.

    Returns a human-readable problem description, or None when the model
    looks healthy.  Never raises and never blocks summarization: the
    catalog is advisory, and a probe failure (offline, API blip) must not
    take down a pipeline that would otherwise work.  Each model is checked
    at most once per process.
    """
    if not model or model in _TINFOIL_CATALOG_CHECKED:
        return None
    _TINFOIL_CATALOG_CHECKED.add(model)

    api_key = _resolve_tinfoil_api_key()
    if not api_key:
        return None
    try:
        resp = requests.get(
            TINFOIL_MODELS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        resp.raise_for_status()
        catalog = {m.get("id"): m for m in resp.json().get("data", [])}
    except Exception as e:  # advisory probe, never fatal
        logger.debug("Tinfoil catalog probe failed (ignored): %s", e)
        return None

    entry = catalog.get(model)
    if entry is None:
        problem = (
            f"Tinfoil model {model!r} is not in the catalog "
            f"({', '.join(sorted(k for k in catalog if k))}). "
            "It has most likely been retired — requests will fail."
        )
    elif entry.get("deprecated"):
        when = entry.get("deprecationDate") or "an unannounced date"
        problem = (
            f"Tinfoil model {model!r} is deprecated and will be removed on "
            f"{when}. Migrate before then."
        )
    else:
        return None

    logger.warning("%s", problem)
    return problem


_ATTESTED_CATALOG_CHECKED: set[str] = set()


def verify_model(backend: str, model: str, *, timeout: int = 10) -> str | None:
    """Advisory model pre-flight for any backend. Never raises, never blocks.

    Delegates to the Tinfoil catalog check for tinfoil; for the attested
    OpenAI-compatible backends it probes the provider's /models list once per
    (backend, model) and warns if the model is absent — the same "retired out
    from under us" guard the vezir-model-check timer relies on.  ollama has no
    remote catalog, so it is a no-op there.
    """
    if backend == "tinfoil":
        return verify_tinfoil_model(model, timeout=timeout)
    if backend not in ATTESTED_BACKENDS or not model:
        return None
    key = f"{backend}:{model}"
    if key in _ATTESTED_CATALOG_CHECKED:
        return None
    _ATTESTED_CATALOG_CHECKED.add(key)

    cfg = ATTESTED_BACKENDS[backend]
    api_key = _resolve_attested_api_key(backend)
    if not api_key:
        return None
    try:
        resp = requests.get(
            str(cfg["catalog_url"]),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        resp.raise_for_status()
        ids = {m.get("id") for m in resp.json().get("data", [])}
    except Exception as e:  # advisory probe, never fatal
        logger.debug("%s catalog probe failed (ignored): %s", backend, e)
        return None

    if model not in ids:
        problem = (
            f"{backend} model {model!r} is not in the catalog — it may have "
            "been retired; requests will fail."
        )
        logger.warning("%s", problem)
        return problem
    return None


# ─── Vision (still frames alongside the transcript) ─────────────────────────

# Models that accept image input.  Deliberately an allowlist of models we
# have actually exercised, NOT Tinfoil's advertised `multimodal` flag: that
# flag was set for deepseek-v4-1-flash while its vision endpoint answered 502
# on every request (2026-09-12).  Re-verified working 2026-10-06 (10 frames
# at 880x1920, every on-screen code read correctly) before it was added here.
# Note it bills ~half the image tokens glm-5-3-flash did, so it likely
# downscales harder; small UI text is untested.  glm-5-3-flash stays listed
# for anyone pinning it via MILLET_SUMMARY_MODEL until Tinfoil removes it.
VISION_MODELS = ("deepseek-v4-1-flash", "glm-5-3-flash", "qwen3-vl", "e2ee-qwen3-vl")

# Hard endpoint limit, not a budget choice: the attested vision endpoints
# reject a request carrying more than 10 images outright — verified live
# 2026-09-17 (session 01M2P6FTRG4WAKKE5T7TV6HNFM, 18 cue frames → venice
# 400 "At most 10 image(s) may be provided in one request").  The job log
# only carries the chain's last error, so tinfoil's own failure text went
# uncaptured — the cap is applied to every vision backend so whichever one
# serves the request is safe.  Over-cap frames are even-sampled (first +
# last kept) so coverage still spans the whole recording; the full frame
# set remains on disk as synced artifacts.
MAX_FRAMES = 10
_FRAME_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_MAX_FRAME_BYTES = 8 * 1024 * 1024


def _usable_frames(frames: list[Path]) -> list[Path]:
    """Filter to readable image files, de-duplicated, ordered, capped."""
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in frames:
        p = Path(raw)
        if p in seen:
            continue
        seen.add(p)
        if p.suffix.lower() not in _FRAME_SUFFIXES:
            continue
        try:
            if not p.is_file() or p.stat().st_size > _MAX_FRAME_BYTES:
                continue
        except OSError:
            continue
        out.append(p)
    if len(out) > MAX_FRAMES:
        logger.info(
            "frames: %d supplied, evenly sampling %d (endpoint image cap)",
            len(out), MAX_FRAMES,
        )
        step = (len(out) - 1) / (MAX_FRAMES - 1)
        out = [out[round(i * step)] for i in range(MAX_FRAMES)]
    return out


def model_supports_vision(model: str | None) -> bool:
    """True when ``model`` is known to accept image input."""
    return bool(model) and any(model.startswith(m) for m in VISION_MODELS)


def backend_supports_vision(backend: str) -> bool:
    """True when a backend has any vision-capable model for the frames path.

    The gate for routing screen-recording frames: text-only tiers (near,
    ollama) return False, so a frames job is never silently downgraded to a
    text-only summary on fallback — it stays on a vision tier or fails loud.
    """
    if backend == "tinfoil":
        return True  # primary vision tier (deepseek-v4-1-flash)
    if backend in ATTESTED_BACKENDS:
        return ATTESTED_BACKENDS[backend]["vision_model"] is not None
    return False  # ollama local tier is text-only (mmproj vision unsupported)


def discover_cue_frames(session_dir: Path) -> list[Path]:
    """Cue frames written next to a session, oldest cue first.

    vezir extracts one PNG per narrated transcript cue into
    ``<session>/attachments/cue_HH-MM-SS.png``.  Sorting by name sorts by
    timestamp, which is the order the narration happened in.
    """
    try:
        return sorted((Path(session_dir) / "attachments").glob("cue_*.png"))
    except OSError:
        return []


# Supported backends.  All are private: tinfoil/venice/near run inside a
# hardware-attested TEE, ollama runs on your own machine.
BACKENDS = ("ollama", "tinfoil", "venice", "near")

# Removed in 0.19.0 because the provider could read meeting content.  Kept
# as a named set so stale environment config degrades with a warning instead
# of crashing every job (see _resolve_backend).
RETIRED_BACKENDS = ("claudemax", "openrouter", "openai")
_WARNED_BACKENDS: set[str] = set()
_WARNED_MODELS: set[str] = set()

# Fallback order.  Every destination is private, so the chain cannot silently
# downgrade confidentiality the way the old claudemax/openrouter chain could.
# tinfoil (attested, vision) -> venice (attested, vision) -> near (attested,
# text) -> ollama (local, text).  Vision-gating (see _dispatch) keeps a frames
# job off the text-only tiers rather than silently dropping the images.
DEFAULT_FALLBACK_ORDER = ("tinfoil", "venice", "near", "ollama")
# Backward-compatible alias for the historical default order.
FALLBACK_ORDER = DEFAULT_FALLBACK_ORDER


def _resolve_fallback_order() -> tuple[str, ...]:
    """Resolve the fallback chain from MILLET_SUMMARY_FALLBACK_ORDER.

    Comma-separated backend names (e.g. "tinfoil,ollama").  Unknown or
    duplicate names are dropped; an empty/unset/invalid value falls back
    to the default order.  Availability gating still applies per backend,
    so listing "tinfoil" is a no-op without a TINFOIL_API_KEY.
    """
    raw = os.environ.get("MILLET_SUMMARY_FALLBACK_ORDER", "").strip()
    if not raw:
        return DEFAULT_FALLBACK_ORDER
    order: list[str] = []
    for name in raw.split(","):
        name = name.strip().lower()
        if name in BACKENDS and name not in order:
            order.append(name)
    return tuple(order) or DEFAULT_FALLBACK_ORDER


# ─── Summarization presets (deprecated) ─────────────────────────────────────
# Presets used to select between backends of differing privacy and quality.
# With only private backends left there is nothing to choose between, so the
# three historical names are kept purely as aliases for the default and are
# scheduled for removal in 0.21.0.  They must keep working meanwhile: vezir
# passes --summary-preset on every job, and ~580 stored jobs carry the names.

DEFAULT_SUMMARY_BACKEND = "tinfoil"

SUMMARY_PRESETS = {
    "confidential": {"backend": DEFAULT_SUMMARY_BACKEND, "model": DEFAULT_TINFOIL_MODEL},
    # Deprecated aliases -> identical config.
    "high-quality": {"backend": DEFAULT_SUMMARY_BACKEND, "model": DEFAULT_TINFOIL_MODEL},
    "alternative": {"backend": DEFAULT_SUMMARY_BACKEND, "model": DEFAULT_TINFOIL_MODEL},
}
DEPRECATED_PRESETS = ("high-quality", "alternative", "confidential")
DEFAULT_PRESET = "confidential"


def _warn_deprecated_preset(preset: str | None) -> None:
    """Log once-per-name when a caller uses a retired preset name."""
    if preset in DEPRECATED_PRESETS and preset not in _WARNED_PRESETS:
        _WARNED_PRESETS.add(preset)
        logger.info(
            "summary preset %r is deprecated and now resolves to the default "
            "(%s/%s); every backend is private, so the preset axis no longer "
            "selects anything. It will be removed in 0.21.0.",
            preset, DEFAULT_SUMMARY_BACKEND, DEFAULT_TINFOIL_MODEL,
        )


_WARNED_PRESETS: set[str] = set()

# Backward-compatible aliases (referenced by translate command, etc.)
DEFAULT_MODEL = DEFAULT_OLLAMA_MODEL

from millet.languages import LANG_NAMES as _LANGUAGE_NAMES
from millet.languages import SECTION_HEADERS as _SECTION_HEADERS

# ─── Prompt loading ────────────────────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(filename: str) -> str | None:
    """Load a prompt template from the prompts directory. Returns None if missing."""
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return None


def _lang_instruction(language: str | None) -> str:
    """Return the appended CRITICAL language instruction line for a non-English language.

    Returns an empty string for English / unknown so prompts stay clean.
    """
    if not language or language == "en":
        return ""
    lang_name = _LANGUAGE_NAMES.get(language, language)
    return (
        f"\n- CRITICAL: Write the ENTIRE summary in {lang_name}, "
        f"including ALL section headers. Do NOT use any English text."
    )


def _build_system_prompt(language: str | None = None, template: str | None = None) -> str:
    """Build the system prompt with section headers in the target language.

    When ``template`` names a summary template (e.g. ``"iteration-plan"``),
    ``summarize_<template>_system.md`` is preferred, falling back to the
    default ``summarize_system.md`` when no such file exists.
    """
    lang = language or "en"
    h = _SECTION_HEADERS.get(lang, _SECTION_HEADERS["en"])

    lang_instruction = _lang_instruction(lang)

    template_text = None
    if template:
        template_text = _load_prompt(f"summarize_{template.replace('-', '_')}_system.md")
    if template_text is None:
        template_text = _load_prompt("summarize_system.md")
    if template_text is not None:
        return template_text.format(
            overview=h["overview"],
            topics=h["topics"],
            actions=h["actions"],
            decisions=h["decisions"],
            questions=h["questions"],
            none_stated=h["none_stated"],
            lang_instruction=lang_instruction,
        )

    # Inline fallback if prompt file is missing
    return f"""\
You are a professional meeting assistant. Analyze the meeting transcript \
and produce a structured summary.

## {h["overview"]}
2-3 sentences covering: what the meeting was about, who was involved, and the main themes.

## {h["topics"]}
* **Topic name:** 1-2 sentence description with key technical details.

## {h["actions"]}
* Action item — **Owner**
(If none, write "{h["none_stated"]}".)

## {h["decisions"]}
* Concrete decision stated as a fact.
(If none, write "{h["none_stated"]}".)

## {h["questions"]}
* Unresolved question or follow-up item.
(If none, write "{h["none_stated"]}".)

After the Markdown sections, append exactly ONE fenced JSON block with the same content as structured data:

```json
{{
  "participants": ["Alice", "Bob"],
  "topics": ["Topic name"],
  "action_items": [
    {{"assignee": "Alice", "task": "Send doc", "due": null, "status": "open"}}
  ],
  "decisions": [
    {{"text": "Use X over Y", "topic": null}}
  ]
}}
```

Rules:
- Use speaker labels exactly as they appear — do not rename or invent names
- Do not hallucinate — every item must be traceable to the transcript
- Be concise but information-dense
- Preserve technical specificity: name exact tools, APIs, frameworks mentioned
- Keep the summary professional and objective
- The JSON block: every field is REQUIRED. Use [] for empty lists, null for unknown assignee/due/topic. action_items.status must be one of "open", "closed", "blocked" — default to "open". The JSON content must be in English even when the body is in another language.{lang_instruction}"""


def _load_user_prompt_template(template: str | None = None) -> str:
    """Load the user prompt template.

    When ``template`` names a summary template (e.g. ``"iteration-plan"``),
    ``summarize_<template>_user.md`` is preferred, falling back to the
    default ``summarize_user.md`` when no such file exists.
    """
    if template:
        text = _load_prompt(f"summarize_{template.replace('-', '_')}_user.md")
        if text is not None:
            return text
    text = _load_prompt("summarize_user.md")
    if text is not None:
        return text
    return "Please summarize the following meeting transcript:\n\n---\n{transcript}\n---"


def _load_user_prompt_template_lang() -> str:
    """Load the language-specific user prompt template."""
    template = _load_prompt("summarize_user_lang.md")
    if template is not None:
        return template
    return (
        "The following meeting transcript is in {language}. "
        "Please summarize it in {language}.\n\n---\n{transcript}\n---"
    )


# ─── Two-pass (extract + format) prompts ──────────────────────────────────


def _extract_lang_instruction(language: str | None) -> str:
    """Lang instruction appended to Pass 1 (extraction) system prompt."""
    if not language or language == "en":
        return ""
    lang_name = _LANGUAGE_NAMES.get(language, language)
    return f"\n- CRITICAL: Output the extracted lists in {lang_name}."


def _format_lang_instruction(language: str | None) -> str:
    """Lang instruction appended to Pass 2 (formatting) system prompt."""
    if not language or language == "en":
        return ""
    lang_name = _LANGUAGE_NAMES.get(language, language)
    return (
        f"\n- CRITICAL: Output everything in {lang_name}, including section headers. "
        "Do NOT use any English text."
    )


def _build_extract_system_prompt(language: str | None = None) -> str:
    """Build the Pass 1 (extraction) system prompt for the two-pass Ollama flow."""
    lang_instruction = _extract_lang_instruction(language)
    template = _load_prompt("summarize_extract_system.md")
    if template is not None:
        return template.format(lang_instruction=lang_instruction)
    # Inline fallback
    return (
        "You are a meeting transcript analyzer. Extract topics, actions, "
        "decisions, and questions from the transcript as plain numbered "
        f"lists.{lang_instruction}"
    )


def _build_format_system_prompt(language: str | None = None) -> str:
    """Build the Pass 2 (formatting) system prompt for the two-pass Ollama flow."""
    lang = language or "en"
    h = _SECTION_HEADERS.get(lang, _SECTION_HEADERS["en"])
    lang_instruction = _format_lang_instruction(lang)
    template = _load_prompt("summarize_format_system.md")
    if template is not None:
        return template.format(
            overview=h["overview"],
            topics=h["topics"],
            actions=h["actions"],
            decisions=h["decisions"],
            questions=h["questions"],
            none_stated=h["none_stated"],
            lang_instruction=lang_instruction,
        )
    # Inline fallback
    return (
        f"Format the extracted meeting data into Markdown with sections: "
        f"## {h['overview']}, ## {h['topics']}, ## {h['actions']}, "
        f"## {h['decisions']}, ## {h['questions']}.\n\n"
        "After the Markdown sections, append exactly ONE fenced ```json block "
        'with keys "participants", "topics", "action_items", "decisions". '
        "Every field is REQUIRED — use [] for empty lists, null for unknown "
        'assignee/due/topic. action_items.status must be one of "open", '
        '"closed", or "blocked". JSON must be in English even when the body '
        f"is in another language.{lang_instruction}"
    )


def _load_extract_user_template() -> str:
    template = _load_prompt("summarize_extract_user.md")
    if template is not None:
        return template
    return (
        "Extract all topics, actions, decisions, and questions from this "
        "transcript:\n\n---\n{transcript}\n---"
    )


def _load_format_user_template() -> str:
    template = _load_prompt("summarize_format_user.md")
    if template is not None:
        return template
    return (
        "Organize the following extracted meeting data into the required "
        "format:\n\n---\n{extracted}\n---"
    )


USER_PROMPT_TEMPLATE = _load_user_prompt_template()

USER_PROMPT_TEMPLATE_LANG = _load_user_prompt_template_lang()


# ─── Data classes ───────────────────────────────────────────────────────────


def _resolve_backend() -> str:
    """Resolve the default backend from env var or hardcoded default.

    A retired backend name in the environment is downgraded to a warning
    rather than an error.  Deployments carry ``MILLET_SUMMARY_BACKEND`` /
    ``MEETSCRIBE_SUMMARY_BACKEND`` in systemd env files that outlive an
    upgrade, and hard-failing every job on a stale env var would turn a
    routine version bump into an outage.  An *explicit* ``backend=`` still
    raises (see ``SummaryConfig.__post_init__``) -- that is a caller bug,
    not stale config.
    """
    from .paths import getenv_renamed

    backend = getenv_renamed(
        "MILLET_SUMMARY_BACKEND",
        "MEETSCRIBE_SUMMARY_BACKEND",
        default=DEFAULT_SUMMARY_BACKEND,
    ).lower()
    if backend in RETIRED_BACKENDS:
        if backend not in _WARNED_BACKENDS:
            _WARNED_BACKENDS.add(backend)
            logger.warning(
                "summary backend %r was removed in 0.19.0 (it could read your "
                "meeting content); using %r instead. Drop the stale "
                "MILLET_SUMMARY_BACKEND/MEETSCRIBE_SUMMARY_BACKEND setting.",
                backend, DEFAULT_SUMMARY_BACKEND,
            )
        return DEFAULT_SUMMARY_BACKEND
    return backend


def _default_model_for_backend(backend: str) -> str:
    """Default model for a backend, ignoring the MILLET_SUMMARY_MODEL override.

    ``MILLET_SUMMARY_MODEL`` targets the user's *chosen* backend and must not
    leak into a different fallback backend.

    The ollama *fallback* tier is special: it must name a model that is actually
    pulled on the host, which differs per deployment.  ``MILLET_OLLAMA_MODEL``
    overrides just this tier (e.g. saray sets it to ``nemotron-3-nano:4b``)
    without touching the global default or leaking into the primary backend the
    way ``MILLET_SUMMARY_MODEL`` would.
    """
    if backend == "tinfoil":
        return DEFAULT_TINFOIL_MODEL
    if backend in ATTESTED_BACKENDS:
        return str(ATTESTED_BACKENDS[backend]["model"])
    return os.environ.get("MILLET_OLLAMA_MODEL", "").strip() or DEFAULT_OLLAMA_MODEL


def _resolve_model(backend: str) -> str:
    """Resolve the default model for a backend from env var or hardcoded default.

    The ``MILLET_SUMMARY_MODEL`` env override applies to the user's *chosen*
    backend only.  For fallback backends (see :func:`_default_model_for_backend`)
    the env model must be ignored — an Ollama model name would otherwise be
    forced onto Tinfoil and fail the whole chain.
    """
    from .paths import getenv_renamed

    env_model = getenv_renamed("MILLET_SUMMARY_MODEL", "MEETSCRIBE_SUMMARY_MODEL")
    if env_model:
        # Upgrade guard: before 0.19.0 the default backend was ollama, so a
        # bare MILLET_SUMMARY_MODEL almost always names an Ollama model.  Now
        # that tinfoil is the default, that stale value would be sent to the
        # enclave and rejected with 404 "model does not exist" at request
        # time.  Ollama tags carry a ":" (qwen3.8:27b); Tinfoil ids never do.
        if backend == "tinfoil" and ":" in env_model:
            if env_model not in _WARNED_MODELS:
                _WARNED_MODELS.add(env_model)
                logger.warning(
                    "MILLET_SUMMARY_MODEL=%r looks like an Ollama tag but the "
                    "backend is %r; ignoring it and using %r. Set "
                    "MILLET_SUMMARY_BACKEND=ollama to keep using that model.",
                    env_model, backend, DEFAULT_TINFOIL_MODEL,
                )
            return DEFAULT_TINFOIL_MODEL
        return env_model
    return _default_model_for_backend(backend)


def _resolve_ollama_singlepass() -> bool:
    """Resolve the default for the ollama single-pass opt-out from the env var."""
    from .paths import getenv_renamed

    raw = (
        (
            getenv_renamed(
                "MILLET_OLLAMA_SINGLEPASS",
                "MEETSCRIBE_OLLAMA_SINGLEPASS",
                default="",
            )
            or ""
        )
        .strip()
        .lower()
    )
    return raw in ("1", "true", "yes", "on")


@dataclass
class SummaryConfig:
    """Configuration for meeting summary generation.

    Supports two private backends. The ``backend`` and ``model`` fields
    respect environment variables when left at their sentinel values:

        MILLET_SUMMARY_BACKEND      -> backend  (default: "tinfoil")
        MILLET_SUMMARY_MODEL        -> model    (default: per-backend)
        MILLET_SUMMARY_TEMPLATE     -> template (default: None = meeting summary)
        TINFOIL_API_KEY             -> required for the tinfoil backend

    Operator-level knob (read at dispatch time, not stored here):
        MILLET_SUMMARY_FALLBACK_ORDER  -> comma-separated fallback chain
            override (default: tinfoil,ollama)
    """

    backend: str | None = None  # None = resolve from env/default
    model: str | None = None  # None = resolve from env/default per backend
    preset: str | None = None  # None = no preset; deprecated legacy names alias the default
    template: str | None = None  # None = default meeting-summary prompts
    ollama_url: str = OLLAMA_BASE_URL
    timeout: int = DEFAULT_TIMEOUT
    temperature: float = 0.3
    num_ctx: int = 8192  # Ollama-specific context window
    ollama_singlepass: bool | None = None  # None = resolve from env (default: two-pass)
    # Optional still frames (PNG/JPEG) shown to the model alongside the
    # transcript.  Used by the narrated screen-recording flow, where each
    # frame is the screen at a transcript cue, so the model can report what
    # it can SEE rather than only what was said.  Ignored by backends and
    # models that cannot accept image input -- never a hard failure.
    frames: list[Path] | None = None

    def __post_init__(self):
        # Resolve preset: explicit arg > env var > None
        if self.preset is None:
            from .paths import getenv_renamed

            self.preset = getenv_renamed(
                "MILLET_SUMMARY_PRESET",
                "MEETSCRIBE_SUMMARY_PRESET",
            )
        # Resolve template: explicit arg > env var > None.  Templates select
        # the prompt files (summarize_<template>_{system,user}.md under
        # millet/prompts/); presets select backend/model.  They compose.
        if self.template is None:
            from .paths import getenv_renamed

            self.template = getenv_renamed(
                "MILLET_SUMMARY_TEMPLATE",
                "MEETSCRIBE_SUMMARY_TEMPLATE",
            )
        if self.template:
            self.template = self.template.lower().strip()
            # Template names become prompt filenames — restrict to a safe
            # charset so no path traversal is possible.
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.template):
                raise ValueError(
                    f"Invalid summary template name {self.template!r}. "
                    "Use lowercase letters, digits, '-' and '_'."
                )
        if self.preset:
            self.preset = self.preset.lower().strip()
            if self.preset in SUMMARY_PRESETS:
                p = SUMMARY_PRESETS[self.preset]
                # Preset sets backend/model only if not explicitly provided
                if self.backend is None:
                    self.backend = p["backend"]
                if self.model is None:
                    self.model = p["model"]

        # Resolve backend: explicit arg > env var > "ollama"
        if self.backend is None:
            self.backend = _resolve_backend()
        self.backend = self.backend.lower()

        if self.backend not in BACKENDS:
            raise ValueError(
                f"Unknown summary backend '{self.backend}'. Supported: {', '.join(BACKENDS)}"
            )

        # Resolve model: explicit arg > env var > per-backend default
        if self.model is None:
            self.model = _resolve_model(self.backend)

        # Resolve ollama two-pass opt-out: explicit arg > env > False (two-pass on)
        if self.ollama_singlepass is None:
            self.ollama_singlepass = _resolve_ollama_singlepass()

        # Normalize frames: keep only readable image files, in a stable
        # order, capped.  Frames are an enhancement, so anything unusable is
        # dropped quietly rather than failing a summary the caller wants.
        if self.frames:
            self.frames = _usable_frames(self.frames)


@dataclass
class MeetingSummary:
    """Result of a meeting summary generation.

    ``markdown`` always holds the human-readable Markdown body suitable for
    PDF rendering — never the trailing JSON data block.  When the LLM
    emitted a structured data block (the contract since schema_version 1),
    the parsed dict is stashed in ``data`` and the body is stripped before
    storage.  See ``meet.frontmatter`` for the schema.
    """

    markdown: str
    model: str
    elapsed_seconds: float
    backend: str = ""
    # Provenance for downstream consumers (e.g. vezir reads the meta
    # sidecar): which preset was requested (if any) and whether the summary
    # was actually produced by a fallback backend instead of the configured
    # one.  Set by summarize() on the returned result before saving.
    preset: str | None = None
    template: str | None = None
    fallback_used: bool = False
    # Number of still frames actually sent to the model.  0 means the
    # summary was text-only, either because no frames were supplied or
    # because the model that served the request cannot see images.
    frames_used: int = 0
    # Optional fields populated by the two-pass Ollama flow
    pass1_seconds: float | None = None
    pass2_seconds: float | None = None
    pass1_chars: int | None = None
    extraction: str | None = None  # Raw Pass 1 output (kept in-memory only)
    # Structured data parsed out of the LLM completion (schema_version 1).
    # ``None`` means the model didn't emit a JSON block or it failed to
    # parse; ``data_error`` records why so the indexer can flag it.
    data: dict[str, Any] | None = None
    data_error: str | None = None

    def save(
        self,
        output_dir: str | Path,
        basename: str,
        *,
        frontmatter_context: FrontmatterContext | None = None,
        lang_suffix: str | None = None,
        artifact: str = "summary",
    ) -> Path:
        """Save the summary as a ``.summary.md`` file plus sidecars.

        When ``frontmatter_context`` is provided, the saved Markdown is
        prefixed with a YAML frontmatter block built from the LLM's
        structured data + the session-level context.  A
        ``.frontmatter.json`` sidecar is also written so consumers that
        don't want to parse YAML can read the same data verbatim.

        When ``frontmatter_context`` is ``None`` we fall back to the
        legacy behavior (raw Markdown body, no frontmatter) for callers
        that haven't been migrated yet.

        The ``.summary.meta.json`` sidecar continues to record which
        backend/model produced the summary, plus per-pass timings for
        the two-pass Ollama flow.

        ``lang_suffix`` (e.g. ``"de"``) writes an ADDITIONAL, language-tagged
        summary — ``<basename>.summary.de.md`` (with matching
        ``.summary.de.meta.json`` and ``<basename>.de.frontmatter.json``
        sidecars) — without clobbering the primary auto-detected
        ``<basename>.summary.md``.  When ``None`` the primary filename is used.

        ``artifact`` names the output artifact family (default ``"summary"``).
        A summary generated from a named template (e.g. ``"iteration-plan"``)
        passes its template name here so the output is written as
        ``<basename>.iteration-plan.md`` (with matching
        ``.iteration-plan.meta.json`` and ``.iteration-plan.frontmatter.json``
        sidecars) instead of clobbering the regular meeting summary.

        Returns the path to the saved ``.<artifact>[.<lang>].md`` file.
        """
        import datetime

        # Local import to avoid a circular import at module load time.
        from millet.frontmatter import (
            build_frontmatter,
            render_frontmatter_block,
            write_frontmatter_sidecar,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # A language-tagged additional summary uses a suffixed name so it
        # coexists with the primary <basename>.summary.md.  A non-default
        # artifact family (template output) likewise gets its own name so it
        # coexists with the meeting summary for the same session.
        suffix = f".{lang_suffix}" if lang_suffix else ""
        stem = artifact or "summary"
        md_path = output_dir / f"{basename}.{stem}{suffix}.md"
        # Sidecar basename carries the suffix too (frontmatter writer appends
        # ".frontmatter.json"), so additional-language/artifact sidecars don't
        # clobber the primary's.
        sidecar_basename = (
            f"{basename}{suffix}" if stem == "summary" else f"{basename}.{stem}{suffix}"
        )

        if frontmatter_context is not None:
            fm = build_frontmatter(
                self.data,
                frontmatter_context,
                extraction_error=self.data_error,
            )
            md_path.write_text(
                render_frontmatter_block(fm) + self.markdown,
                encoding="utf-8",
            )
            write_frontmatter_sidecar(output_dir, sidecar_basename, fm)
        else:
            md_path.write_text(self.markdown, encoding="utf-8")

        meta: dict[str, Any] = {
            "backend": self.backend,
            "model": self.model,
            "preset": self.preset,
            "template": self.template,
            "fallback_used": self.fallback_used,
            "frames_used": self.frames_used,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if self.pass1_seconds is not None:
            meta["mode"] = "two_pass"
            meta["pass1_seconds"] = round(self.pass1_seconds, 2)
            meta["pass2_seconds"] = round(self.pass2_seconds or 0.0, 2)
            if self.pass1_chars is not None:
                meta["pass1_chars"] = self.pass1_chars
        if self.data_error:
            meta["data_error"] = self.data_error
        elif self.data is not None:
            meta["data_extracted"] = True
        meta_path = output_dir / f"{basename}.{stem}{suffix}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        return md_path


# ─── Ollama availability check ─────────────────────────────────────────────


def is_ollama_available(url: str = OLLAMA_BASE_URL) -> bool:
    """Check if Ollama is running and reachable."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=5)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def list_models(url: str = OLLAMA_BASE_URL) -> list[str]:
    """List available Ollama models."""
    try:
        resp = requests.get(f"{url}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


# ─── Backend availability checks ───────────────────────────────────────────


def tinfoil_sdk_installed() -> bool:
    """Is the `tinfoil` SDK importable?

    Base dependency since 0.20.1, so normally yes.  Still checked because an
    environment can lack it — a constraints file that excludes it, a partial
    upgrade, or a pre-0.20.1 install where it lived in the `[tee]` extra.
    Availability must not report a backend as usable when the import that
    drives it would fail (that turned a missing SDK into a ModuleNotFoundError
    traceback mid-job instead of a readable message).
    """
    return importlib.util.find_spec("tinfoil") is not None


def is_backend_available(config: SummaryConfig | None = None) -> bool:
    """Check if the configured summary backend is reachable.

    For tinfoil: the SDK must be importable *and* an API key resolvable.
    For ollama: checks the local server.
    """
    if config is None:
        config = SummaryConfig()

    if config.backend == "tinfoil":
        return tinfoil_sdk_installed() and bool(_resolve_tinfoil_api_key())
    if config.backend in ATTESTED_BACKENDS:
        # OpenAI-compatible over HTTP: needs the openai SDK importable and a key.
        return (
            importlib.util.find_spec("openai") is not None
            and bool(_resolve_attested_api_key(config.backend))
        )
    return is_ollama_available(config.ollama_url)


def _backend_not_available_message(config: SummaryConfig) -> str:
    """Return a user-friendly message when the backend is unavailable."""
    if config.backend == "tinfoil":
        if not tinfoil_sdk_installed():
            return (
                "the 'tinfoil' SDK is not installed, so the default "
                "hardware-attested TEE backend cannot be used. Install it with "
                "`pip install --upgrade millet-pipeline`, or run fully locally "
                "with --summary-backend ollama."
            )
        return (
            f"TINFOIL_API_KEY is not set and key file {_TINFOIL_KEY_FILE} "
            "not found. Get an API key at https://tinfoil.sh, or run fully "
            "locally with --summary-backend ollama."
        )
    if config.backend in ATTESTED_BACKENDS:
        cfg = ATTESTED_BACKENDS[config.backend]
        if importlib.util.find_spec("openai") is None:
            return (
                f"the 'openai' SDK is not installed, so the {config.backend!r} "
                "attested backend cannot be used."
            )
        return (
            f"{cfg['key_env']} is not set and key file {cfg['key_file']} not "
            f"found; the {config.backend!r} attested backend is unavailable."
        )
    return f"Ollama is not running at {config.ollama_url}. Start it with: ollama serve"


# ─── Ollama backend ───────────────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1 token per 4 characters for English text.

    This is a conservative heuristic.  Real tokenizers vary by model,
    but 4 chars/token is a safe lower bound (i.e. overestimates tokens)
    which is what we want when sizing the context window.
    """
    return len(text) // 4


def _dynamic_num_ctx(
    system_prompt: str,
    user_prompt: str,
    floor: int = 8192,
    ceiling: int = 65536,
    output_reserve: int = 4096,
) -> int:
    """Calculate a context window size that fits the full prompt.

    Returns a value between *floor* and *ceiling* (inclusive).  The
    calculation adds an *output_reserve* buffer so the model has room
    to generate the summary without truncating its own output.
    """
    prompt_tokens = _estimate_tokens(system_prompt + user_prompt)
    needed = prompt_tokens + output_reserve
    # Round up to nearest 1024 for tidiness
    needed = ((needed + 1023) // 1024) * 1024
    return max(floor, min(needed, ceiling))


def _call_ollama_chat(
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
    *,
    num_ctx: int | None = None,
    output_reserve: int = 4096,
    timeout: int | None = None,
    temperature: float | None = None,
) -> tuple[str, float]:
    """Single Ollama /api/chat call. Returns (content, elapsed_seconds).

    Raises ConnectionError if Ollama is unreachable, RuntimeError on API
    error / empty response.  Used by both the single-pass and two-pass flows.

    ``output_reserve`` is the token budget reserved for the model's own
    output when ``num_ctx`` is auto-sized.  The two-pass extraction (Pass 1)
    overrides the 4096 default: thinking-heavy models like qwen3.8:27b emit
    long exhaustive lists and, with only 4096 reserved, hit the context
    window mid-extraction (``done_reason: length``) and silently truncate.
    """
    import time

    if not is_ollama_available(config.ollama_url):
        raise ConnectionError(
            f"Ollama is not running at {config.ollama_url}. Start it with: ollama serve"
        )

    if num_ctx is None:
        num_ctx = _dynamic_num_ctx(
            system_prompt,
            user_prompt,
            floor=config.num_ctx,
            output_reserve=output_reserve,
        )
    if timeout is None:
        timeout = config.timeout
    if temperature is None:
        temperature = config.temperature

    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "think": False,  # Disable thinking/reasoning for speed
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    url = f"{config.ollama_url}/api/chat"
    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.Timeout as e:
        raise RuntimeError(
            f"Ollama timed out after {timeout}s. "
            f"The model '{config.model}' may be too large or slow. "
            "Try a smaller model with --summary-model."
        ) from e
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama API error: {e}") from e
    elapsed = time.time() - t0
    data = resp.json()
    content = (data.get("message", {}).get("content") or "").strip()
    if not content:
        raise RuntimeError(
            f"Ollama returned an empty response. Model '{config.model}' may "
            "not be available. Check with: ollama list"
        )
    return content, elapsed


def _summarize_ollama(
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
) -> MeetingSummary:
    """Send a single-pass summarization request to local Ollama."""
    content, elapsed = _call_ollama_chat(system_prompt, user_prompt, config)
    return MeetingSummary(
        markdown=content,
        model=config.model,
        elapsed_seconds=elapsed,
        backend="ollama",
    )


def _summarize_ollama_twopass(
    transcript_text: str,
    config: SummaryConfig,
    language: str | None = None,
) -> MeetingSummary:
    """Two-pass Ollama summarization: extract (Pass 1) then format (Pass 2).

    Pass 1 uses a wide context window sized to the transcript and a long
    timeout to extract topics/actions/decisions/questions as plain numbered
    lists.  Pass 2 takes the much smaller extracted data and formats it into
    the canonical Markdown structure with a fixed 8K context window and a
    shorter timeout.

    This dramatically improves format compliance and reduces hallucinations
    on local 20B-class models like gpt-oss:20b and qwen3.6:27b, at the cost
    of one additional LLM call (typically ~30-90s extra).
    """
    # ── Pass 1: extraction ────────────────────────────────────────────────
    extract_sys = _build_extract_system_prompt(language)
    extract_user_tmpl = _load_extract_user_template()
    extract_user = extract_user_tmpl.format(transcript=transcript_text)
    extracted, t1 = _call_ollama_chat(
        extract_sys,
        extract_user,
        config,
        # Pass 1 needs the full transcript to fit AND room for a long
        # exhaustive extraction. Reserve 16K output tokens (not the 4K
        # default) so thinking-heavy models don't truncate mid-list.
        num_ctx=None,
        output_reserve=16384,
        timeout=config.timeout,
        temperature=config.temperature,
    )

    # ── Pass 2: formatting ────────────────────────────────────────────────
    format_sys = _build_format_system_prompt(language)
    format_user_tmpl = _load_format_user_template()
    format_user = format_user_tmpl.format(extracted=extracted)
    # Pass 2 input is small (the extracted lists). Cap context at 8K and use a
    # shorter timeout so we fail fast if something goes wrong.
    pass2_timeout = min(config.timeout, 240)
    formatted, t2 = _call_ollama_chat(
        format_sys,
        format_user,
        config,
        num_ctx=8192,
        timeout=pass2_timeout,
        temperature=config.temperature,
    )

    return MeetingSummary(
        markdown=formatted,
        model=config.model,
        elapsed_seconds=t1 + t2,
        backend="ollama",
        pass1_seconds=t1,
        pass2_seconds=t2,
        pass1_chars=len(extracted),
        extraction=extracted,
    )


# ─── Tinfoil TEE backend ──────────────────────────────────────────────────

# Number of attempts + base backoff for the Tinfoil path.  The SDK does
# a network fetch at client construction (GET https://atc.tinfoil.sh/routers)
# AND for the inference call; on hosts with flaky DNS a single transient
# lookup failure used to hard-fail the whole summarization.  Retry both.
_TINFOIL_MAX_ATTEMPTS = 3
_TINFOIL_BACKOFF_BASE = 2.0  # seconds: ~2s, 4s, 8s

# Attestation-verification blips get their own, larger budget.  Measured
# 2026-09-12: 9/12 plain requests succeeded, i.e. a ~25% failure rate,
# randomly distributed — three attempts leaves a real chance of losing a
# job to pure luck.  These are safe to retry generously because they fail
# *fast*, during enclave verification and before any tokens are generated,
# unlike a timeout.  Retrying never weakens the guarantee: an unverified
# response is still never accepted.
_TINFOIL_ATTEST_MAX_ATTEMPTS = 6
_TINFOIL_ATTEST_BACKOFF = 1.5  # seconds, flat — the blip clears quickly

# Headroom on top of ``config.timeout`` for one attempt's non-inference
# work (router discovery, TLS, enclave attestation).  The hard deadline for
# one attempt is ``config.timeout + _TINFOIL_ATTEMPT_GRACE_SECONDS``.
_TINFOIL_ATTEMPT_GRACE_SECONDS = 120


class _TinfoilAttemptStuck(TimeoutError):
    """One Tinfoil attempt exceeded its wall-clock deadline.

    Subclasses :class:`TimeoutError` so the transient-error classifier
    treats it like any other timeout and the normal retry ladder applies.
    """


def _run_under_deadline(fn, *, seconds: float):
    """Run ``fn()`` under a hard wall-clock deadline in a daemon thread.

    Incident 2026-09-15: the Tinfoil SDK's custom httpx transport can block
    in ``ssl.read()`` with no effective timeout — a summary attempt pinned a
    vezir job in ``transcribing`` for 30+ minutes while millet's per-request
    ``timeout=600`` never fired (faulthandler stack: openai -> httpx ->
    tinfoil transport -> httpcore ``_receive_event`` -> ``ssl.read``).  The
    transport drops the per-request deadline on the floor, so no per-phase
    timeout can bound the call; only an external timer can.

    When the deadline passes, the worker thread is *abandoned* (it cannot be
    interrupted mid-``ssl.read``) and ``_TinfoilAttemptStuck`` is raised.
    The thread is a daemon, so at most ``_TINFOIL_MAX_ATTEMPTS`` blocked
    threads are leaked and they die with the process — millet runs as a
    short-lived per-job subprocess.  (This is why a raw thread is used
    instead of :class:`concurrent.futures.ThreadPoolExecutor`: the executor
    joins its workers at interpreter exit via atexit, so an abandoned
    blocked worker would hang process exit.)

    Exceptions raised by ``fn`` itself propagate unchanged.
    """
    import queue
    import threading

    result: queue.Queue = queue.Queue(maxsize=1)

    def _worker():
        try:
            result.put(("ok", fn()))
        except BaseException as exc:
            result.put(("err", exc))

    threading.Thread(
        target=_worker, name="tinfoil-attempt", daemon=True
    ).start()
    try:
        kind, value = result.get(timeout=seconds)
    except queue.Empty:
        raise _TinfoilAttemptStuck(
            f"attempt exceeded the hard wall-clock deadline of {seconds:.0f}s "
            "(the Tinfoil transport stalled without honouring its "
            "per-request timeout)"
        ) from None
    if kind == "err":
        raise value
    return value


def _is_transient_network_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a transient connectivity/DNS blip worth
    retrying (as opposed to a genuine auth/model/config error that won't
    improve on retry).

    The Tinfoil SDK surfaces a DNS failure during router discovery as
    ``ValueError("Failed to fetch router addresses: <urlopen error ...>")``;
    other backends raise socket/urllib/httpx connection errors.
    """
    import socket
    import urllib.error

    transient_types: tuple[type[BaseException], ...] = (
        socket.gaierror,
        socket.timeout,
        ConnectionError,
        TimeoutError,
        urllib.error.URLError,
    )
    try:
        import httpx

        transient_types += (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.RemoteProtocolError,
        )
    except Exception:
        pass

    if isinstance(exc, transient_types):
        return True
    # Walk the cause/context chain (the SDK wraps URLError in ValueError).
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, transient_types):
            return True
        cur = cur.__cause__ or cur.__context__
    # Last resort: match the SDK's router-discovery message + common DNS text.
    #
    # Server-side capacity errors (HTTP 503) are included: Tinfoil returns
    # "The engine is currently overloaded, please try again later." when an
    # enclave pool has no free capacity.  That is genuinely transient and
    # worth a backoff retry — unlike a 404 "The model does not exist.",
    # which means the model is gone and will never come back.
    msg = str(exc).lower()
    markers = (
        "failed to fetch router addresses",
        "name or service not known",
        "temporary failure in name resolution",
        "no address associated with hostname",
        "connection reset",
        "connection refused",
        "timed out",
        "engine is currently overloaded",
        "error code: 503",
        "service unavailable",
        # Tinfoil occasionally serves a malformed SEV attestation report
        # ("current_tcb not correctly formed"), and the SDK refuses the
        # response.  Observed intermittently on 2026-09-12: two failures
        # then success, on plain text requests.  Retrying is safe and does
        # NOT weaken the guarantee -- an unverified response is never
        # accepted; we simply ask again and either get a report that
        # verifies or fail loudly.  Without this the confidential path
        # fails a job outright on a blip that clears in seconds.
        "attestation verification failed",
        "failed to parse report",
    )
    return any(m in msg for m in markers)


def _is_attestation_error(exc: BaseException) -> bool:
    """True when the TEE's attestation report could not be verified.

    The SDK refuses such a response, which is the correct behaviour — we
    never accept unverified inference.  It is worth distinguishing because
    it is provider-side, frequent (~25% of requests on 2026-09-12), fails
    fast, and clears on its own, so it earns a larger retry budget than a
    genuine network fault without weakening any guarantee.
    """
    msg = str(exc).lower()
    return "attestation verification failed" in msg or "failed to parse report" in msg


def _is_model_pool_error(exc: BaseException) -> bool:
    """True if ``exc`` implicates one specific model rather than Tinfoil itself.

    A drained/removed enclave pool answers 503 ("engine is currently
    overloaded") or 404 ("the model does not exist").  Those are worth
    retrying on a *sibling* TEE model.  A DNS/router-discovery failure, by
    contrast, means the whole service is unreachable — a sibling would fail
    identically, so it is not worth the extra round trip.
    """
    msg = str(exc).lower()
    return any(
        m in msg
        for m in (
            "engine is currently overloaded",
            "error code: 503",
            "service unavailable",
            "does not exist",
            "error code: 404",
        )
    )


def _summarize_tinfoil(
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
) -> MeetingSummary:
    """Send a summarization request to Tinfoil TEE (hardware-private inference).

    Prompts are encrypted into the secure enclave and processed inside
    NVIDIA H100 Confidential Computing.  The provider cannot see the data.

    Resilience (v0.9.2): both the client construction (which fetches the
    enclave router list over the network) and the completion call are
    retried on transient connectivity/DNS errors with exponential
    backoff.  Genuine auth/model errors fail fast (no retry).
    """
    import time

    api_key = _resolve_tinfoil_api_key()
    if not api_key:
        raise RuntimeError(
            f"{TINFOIL_API_KEY_ENV} environment variable is not set and "
            f"key file {_TINFOIL_KEY_FILE} not found. "
            "Get an API key at https://tinfoil.sh"
        )

    # Advisory catalog check (once per model per process).  Surfaces a
    # retired/deprecated model in the log before we spend a request on it.
    verify_tinfoil_model(config.model)

    from tinfoil import TinfoilAI

    frames = config.frames or []
    frames_sent = 0

    def _user_content(model: str):
        """User message body: plain string, or multi-part when sending frames.

        Frames are dropped for a model not on the vision allowlist -- notably
        the sibling fallback -- so a drained enclave degrades a vision run to
        text-only instead of failing it outright.
        """
        nonlocal frames_sent
        if not frames:
            frames_sent = 0
            return user_prompt
        if not model_supports_vision(model):
            frames_sent = 0
            logger.warning(
                "model %r cannot accept image input; summarizing %d frame(s) "
                "as text-only", model, len(frames),
            )
            return user_prompt
        parts: list[dict] = [{"type": "text", "text": user_prompt}]
        for p in frames:
            try:
                encoded = base64.b64encode(p.read_bytes()).decode("ascii")
            except OSError as exc:
                logger.warning("frames: skipping unreadable %s (%s)", p.name, exc)
                continue
            mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            })
        frames_sent = len(parts) - 1
        return parts

    def _attempt(model: str):
        """Call one TEE model with the retry/backoff budget. Raises on failure.

        Attestation failures consume a separate budget: they are frequent,
        fail fast, and are unrelated to the model or the network, so they
        should not burn the attempts reserved for genuine transients.
        """
        def _call():
            # Client init does a network fetch (router discovery) — keep it
            # inside the retry so a DNS blip here doesn't hard-fail.
            client = TinfoilAI(api_key=api_key)
            # timeout: passed per-request since 0.13.0 — but the SDK's custom
            # transport does not reliably honour it (see _run_under_deadline),
            # so the wall-clock deadline below is the actual guarantee.
            return client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _user_content(model)},
                ],
                temperature=config.temperature,
                timeout=config.timeout,
            )

        attempt = 0
        attest_failures = 0
        while True:
            attempt += 1
            try:
                deadline = config.timeout + _TINFOIL_ATTEMPT_GRACE_SECONDS
                return _run_under_deadline(_call, seconds=deadline)
            except Exception as e:
                if _is_attestation_error(e):
                    attest_failures += 1
                    attempt -= 1  # doesn't consume the transient budget
                    if attest_failures < _TINFOIL_ATTEST_MAX_ATTEMPTS:
                        logger.warning(
                            "Tinfoil enclave attestation failed (%d/%d) for "
                            "%r; re-verifying in %.1fs: %s",
                            attest_failures,
                            _TINFOIL_ATTEST_MAX_ATTEMPTS,
                            model,
                            _TINFOIL_ATTEST_BACKOFF,
                            e,
                        )
                        time.sleep(_TINFOIL_ATTEST_BACKOFF)
                        continue
                    raise RuntimeError(
                        f"Tinfoil enclave attestation could not be verified after "
                        f"{_TINFOIL_ATTEST_MAX_ATTEMPTS} attempts. The response was "
                        f"never accepted unverified — this is a provider-side "
                        f"attestation fault, not a downgrade: {e}"
                    ) from e
                if attempt < _TINFOIL_MAX_ATTEMPTS and _is_transient_network_error(e):
                    wait = _TINFOIL_BACKOFF_BASE**attempt
                    logger.warning(
                        "Tinfoil attempt %d/%d for %r hit a transient "
                        "network/capacity error; retrying in %.0fs: %s",
                        attempt,
                        _TINFOIL_MAX_ATTEMPTS,
                        model,
                        wait,
                        e,
                    )
                    time.sleep(wait)
                    continue
                raise

    t0 = time.time()
    used_model = config.model
    sibling_used = False
    sibling = DEFAULT_TINFOIL_FALLBACK_MODEL

    try:
        response = _attempt(config.model)
    except Exception as e:
        # The primary model's enclave pool is unreachable or gone.  Try the
        # sibling TEE model before giving up: still inside the TEE, so the
        # privacy contract is intact, and it survives a single-pool outage.
        if not _is_model_pool_error(e) or not sibling or sibling == config.model:
            if _is_transient_network_error(e):
                raise RuntimeError(
                    "Tinfoil TEE unreachable after "
                    f"{_TINFOIL_MAX_ATTEMPTS} attempts (transient network/"
                    f"timeout/capacity errors — last error: {e})"
                ) from e
            raise RuntimeError(f"Tinfoil TEE API error: {e}") from e

        logger.warning(
            "Tinfoil model %r unusable (%s); falling back to sibling TEE "
            "model %r. Still hardware-attested — privacy contract intact.",
            config.model,
            e,
            sibling,
        )
        try:
            response = _attempt(sibling)
        except Exception as e2:
            raise RuntimeError(
                f"Tinfoil TEE API error: primary model {config.model!r} failed "
                f"({e}) and sibling {sibling!r} also failed ({e2})"
            ) from e2
        used_model = sibling
        sibling_used = True

    elapsed = time.time() - t0
    content = (response.choices[0].message.content or "").strip()

    if not content:
        raise RuntimeError(f"Tinfoil returned an empty response for model '{used_model}'.")

    return MeetingSummary(
        markdown=content,
        model=f"{used_model} (TEE)",
        elapsed_seconds=elapsed,
        backend="tinfoil",
        fallback_used=sibling_used,
        frames_used=frames_sent,
    )


# ─── Generic attested OpenAI-compatible backend (venice, near) ─────────────


def _encode_frames_content(user_prompt: str, frames: list[Path]) -> tuple[object, int]:
    """Build an OpenAI multi-part user message with base64 image_url frames.

    Returns (content, frames_sent).  Caller has already decided the model can
    see (vision gating happens at dispatch), so this always attaches frames.
    """
    parts: list[dict] = [{"type": "text", "text": user_prompt}]
    for p in frames:
        try:
            encoded = base64.b64encode(p.read_bytes()).decode("ascii")
        except OSError as exc:
            logger.warning("frames: skipping unreadable %s (%s)", p.name, exc)
            continue
        mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        parts.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
        )
    return parts, len(parts) - 1


def _is_billing_error(exc: Exception) -> bool:
    """True if an API error is an out-of-credits / billing rejection.

    Verified live 2026-09-15: unpaid keys return HTTP 402 — NEAR
    ``{"type":"no_limit_configured"}``, Venice ``"Insufficient USD or Diem
    balance"``.  This must fail LOUD with cause (the operator has to add
    credits), never be mistaken for a transient blip and silently retried.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 402:
        return True
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "402",
            "insufficient",
            "no_limit_configured",
            "no spending limit",
            "add credits",
            "quota",
            "billing",
        )
    )


def _summarize_attested_oai(
    backend: str,
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
) -> MeetingSummary:
    """Summarize via an OpenAI-compatible attested TEE backend (venice/near).

    Verifies the enclave's attestation (NVIDIA NRAS evidence + freshness nonce)
    *before* sending any prompt, so meeting content only leaves the box once the
    hardware is confirmed.  A failed attestation raises AttestationError and is
    treated by summarize() like the Tinfoil attestation fault: fall through to
    the next backend, never save an unverified result as attested.

    Transient network errors are retried with the same budget/backoff as the
    Tinfoil path; genuine auth/model errors fail fast.
    """
    import time

    from millet.attestation import AttestationError, new_nonce, verify_attestation

    cfg = ATTESTED_BACKENDS[backend]
    api_key = _resolve_attested_api_key(backend)
    if not api_key:
        raise RuntimeError(
            f"{cfg['key_env']} is not set and key file {cfg['key_file']} not "
            f"found; cannot use the {backend!r} attested backend."
        )

    # Pick the vision model when frames are present (dispatch guarantees this
    # backend has one before routing frames here); else the text model.
    frames = config.frames or []
    model = config.model
    if frames and cfg["vision_model"]:
        model = str(cfg["vision_model"])

    # Advisory catalog pre-flight (once per model per process).
    verify_model(backend, model)

    # Attestation gate — verify the enclave before the prompt leaves.  Failure
    # is loud and non-silent (AttestationError propagates to the fallback loop).
    nonce = new_nonce()
    attestation_url = str(cfg["attestation_url"]).format(model=model)
    verify_attestation(attestation_url, api_key=api_key, nonce=nonce)

    if frames:
        content, frames_sent = _encode_frames_content(user_prompt, frames)
    else:
        content, frames_sent = user_prompt, 0

    # TLS: these are PUBLIC endpoints, so verify against certifi's CA bundle.
    # The openai SDK uses httpx, which honors SSL_CERT_FILE — and on server
    # hosts that env var points at a private CA (e.g. saray's Caddy internal
    # CA, verified 2026-09-15), which lacks public roots and makes every
    # Venice/NEAR call fail with a misleading "Connection error" instead of
    # the real response.  Pin the public bundle explicitly. (requests dodged
    # this by always using certifi; httpx does not.)
    import certifi
    import httpx
    from openai import OpenAI

    client = OpenAI(
        base_url=str(cfg["base_url"]),
        api_key=api_key,
        http_client=httpx.Client(verify=certifi.where()),
    )

    t0 = time.time()
    attempt = 0
    while True:
        attempt += 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=config.temperature,
                timeout=config.timeout,
            )
            break
        except AttestationError:
            raise
        except Exception as e:
            # Billing/credits: fail fast and LOUD with the real cause — retrying
            # an out-of-credits key just wastes the budget, and the operator
            # needs to know to add credits (not a transient blip).
            if _is_billing_error(e):
                raise RuntimeError(
                    f"{backend} rejected the request for billing reasons "
                    f"(add credits for this provider): {e}"
                ) from e
            if attempt < _TINFOIL_MAX_ATTEMPTS and _is_transient_network_error(e):
                wait = _TINFOIL_BACKOFF_BASE**attempt
                logger.warning(
                    "%s attempt %d/%d hit a transient network error; retrying "
                    "in %.0fs: %s",
                    backend, attempt, _TINFOIL_MAX_ATTEMPTS, wait, e,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"{backend} TEE API error: {e}") from e

    elapsed = time.time() - t0
    content_out = (response.choices[0].message.content or "").strip()
    if not content_out:
        raise RuntimeError(f"{backend} returned an empty response for model {model!r}.")

    return MeetingSummary(
        markdown=content_out,
        model=f"{model} (TEE)",
        elapsed_seconds=elapsed,
        backend=backend,
        fallback_used=False,
        frames_used=frames_sent,
    )


# ─── Response validation ──────────────────────────────────────────────────

# Patterns that indicate the "summary" is actually an error response from
# an upstream API, not real meeting content.  These are checked as a
# defense-in-depth measure so that even if a backend proxy returns error
# text as a 200/valid completion, we catch it and trigger the fallback.
_ERROR_PATTERNS = re.compile(
    r'"type"\s*:\s*"error"'  # JSON error envelope
    r"|authentication_error"  # Anthropic auth failure
    r"|Invalid\s+(authentication\s+)?credentials"
    r"|Failed\s+to\s+authenticate"
    r"|rate_limit_error"
    r"|overloaded_error",
    re.IGNORECASE,
)


def _validate_summary_content(content: str, backend: str) -> None:
    """Raise RuntimeError if *content* looks like an error message, not a summary.

    This prevents upstream API errors (e.g. expired OAuth tokens returning
    401 error JSON) from being silently saved as the meeting summary.
    """
    # Short responses that match known error patterns are almost certainly
    # not real summaries (real summaries are typically 500+ chars).
    if len(content) < 400 and _ERROR_PATTERNS.search(content):
        raise RuntimeError(f"{backend} returned an error instead of a summary: {content[:200]}")


# ─── Core summarization (dispatcher with fallback chain) ──────────────────


def _dispatch(
    backend: str,
    system_prompt: str,
    user_prompt: str,
    config: SummaryConfig,
    *,
    transcript_text: str | None = None,
    language: str | None = None,
) -> MeetingSummary:
    """Dispatch to a specific backend's summarization function.

    Creates a temporary config with the correct backend and model if
    falling back from the originally configured backend.

    For the ollama backend, uses the two-pass (extract+format) flow by
    default unless ``config.ollama_singlepass`` is True.  Two-pass requires
    ``transcript_text`` and ``language`` to be passed through.
    """
    if backend != config.backend:
        # Build a new config for the fallback backend with its own default
        # model.  Use the hardcoded default (not _resolve_model) so a user's
        # MILLET_SUMMARY_MODEL set for the primary backend does not leak into
        # a different fallback backend and fail the entire chain.
        fallback_config = SummaryConfig(
            backend=backend,
            model=_default_model_for_backend(backend),
            ollama_url=config.ollama_url,
            timeout=config.timeout,
            temperature=config.temperature,
            num_ctx=config.num_ctx,
            ollama_singlepass=config.ollama_singlepass,
            template=config.template,
            # Carry frames too: a rebuilt config that drops them would
            # silently downgrade a vision run to text-only on fallback.
            frames=config.frames,
        )
    else:
        fallback_config = config

    if backend == "tinfoil":
        result = _summarize_tinfoil(system_prompt, user_prompt, fallback_config)
    elif backend in ATTESTED_BACKENDS:
        result = _summarize_attested_oai(
            backend, system_prompt, user_prompt, fallback_config
        )
    else:
        # Ollama: prefer the two-pass flow unless explicitly opted out.  A
        # named summary template always uses the single-pass flow — the
        # two-pass extract/format prompts are specific to the default
        # meeting-summary template.
        if (
            not fallback_config.ollama_singlepass
            and transcript_text is not None
            and not fallback_config.template
        ):
            result = _summarize_ollama_twopass(
                transcript_text,
                fallback_config,
                language=language,
            )
        else:
            result = _summarize_ollama(system_prompt, user_prompt, fallback_config)

    # Split the trailing JSON data block off the markdown body so that
    # PDF rendering keeps using the body and frontmatter writers can
    # consume the parsed data.  Done once here so every backend benefits.
    from millet.frontmatter import split_body_and_data

    body, data, data_error = split_body_and_data(result.markdown)
    result.markdown = body
    result.data = data
    # Only record an error if extraction failed AND the prompt should
    # have produced data; "no JSON block found" on a model that ignored
    # the contract is still useful diagnostic info, so keep it.
    if data is None:
        result.data_error = data_error

    # Defense-in-depth: catch error text masquerading as a valid summary
    _validate_summary_content(result.markdown, backend)
    return result


def summarize(
    transcript_text: str,
    config: SummaryConfig | None = None,
    language: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> MeetingSummary:
    """Generate a structured meeting summary from transcript text.

    Dispatches to the appropriate backend based on ``config.backend``.
    If the configured backend is unavailable or fails, automatically tries
    the next backend in the fallback order (default:
    tinfoil -> venice -> near -> ollama; overridable via
    MILLET_SUMMARY_FALLBACK_ORDER).  Every backend in the chain is private —
    three hardware-attested TEEs plus fully-local ollama — so the chain can
    never downgrade confidentiality, only quality, and every fallback is
    recorded (fallback_used + backend/model provenance, never silent).
    Since 0.21.2 this applies to explicitly requested presets too: the
    preset names are aliases of the default, so a "confidential" summary
    produced by the local fallback is still confidential — it is marked
    ollama/<model>, never labelled a TEE.

    Args:
        transcript_text: The plain-text transcript (as produced by
            ``Transcript.to_text()``).
        config: Summary configuration. Uses defaults if not provided.
        language: Language code of the transcript (e.g. "de", "fa").
            When provided (and not "en") the LLM is instructed to
            write the summary in that language.
        progress_callback: Optional callable(str) -> None for status
            messages (e.g. reporting fallback attempts to the GUI/CLI).

    Returns:
        MeetingSummary with the Markdown summary, model used, and timing.

    Raises:
        ConnectionError: If no backend is reachable.
        RuntimeError: If all backends fail to generate a response.
    """
    if config is None:
        config = SummaryConfig()

    def _log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    # Build prompts with language-aware section headers.  A named template
    # swaps in its own prompt files (loaded per call — the module-level
    # USER_PROMPT_TEMPLATE caches only the default template).
    system_prompt = _build_system_prompt(language, template=config.template)

    if config.template:
        user_prompt = _load_user_prompt_template(config.template).format(
            transcript=transcript_text
        )
    elif language and language != "en":
        lang_name = _LANGUAGE_NAMES.get(language, language)
        user_prompt = USER_PROMPT_TEMPLATE_LANG.format(
            language=lang_name,
            transcript=transcript_text,
        )
    else:
        user_prompt = USER_PROMPT_TEMPLATE.format(transcript=transcript_text)

    # An explicit preset is a deprecated alias of the default (0.19.0):
    # warn once, then let it ride the same fallback chain as everyone
    # else.  Until 0.21.2 a requested preset re-raised the primary
    # backend's failure instead — a guard from when presets selected
    # backends with different privacy properties.  That axis is gone:
    # every chain destination is private (attested TEEs + local), so the
    # guard only converted recoverable provider outages into hard job
    # failures.  "Confidential" remains guaranteed by construction — the
    # chain has no third-party-readable destination — and any fallback is
    # recorded, never silent.
    if config.preset and config.preset in SUMMARY_PRESETS:
        _warn_deprecated_preset(config.preset)

    # Build the list of backends to try: configured first, then fallback order
    backends_to_try = [config.backend]
    for fb in _resolve_fallback_order():
        if fb not in backends_to_try:
            backends_to_try.append(fb)

    last_error = None
    unavailable: list[str] = []
    for backend in backends_to_try:
        # Check availability before attempting.  Carry the caller's ollama_url
        # so a custom Ollama server isn't reported unavailable just because the
        # probe defaulted to localhost.
        avail_config = SummaryConfig(backend=backend, ollama_url=config.ollama_url)
        if not is_backend_available(avail_config):
            reason = _backend_not_available_message(avail_config)
            # Keep the reason: if every backend is skipped this way nothing is
            # ever dispatched, so last_error stays None and the final error
            # would otherwise read "Last error: None" and name no cause.
            unavailable.append(f"{backend}: {reason}")
            if backend == config.backend:
                _log(f"{backend} is unavailable: {reason}")
            else:
                _log(f"Fallback {backend} also unavailable, skipping...")
            continue

        # Vision gate: a screen-recording (frames) job must not fall back to a
        # text-only tier — that would silently drop the images and summarize
        # blind.  Skip such a backend as a *fallback* only; the primary backend
        # is left to its own text-only degradation (unchanged behavior).
        if (
            config.frames
            and backend != config.backend
            and not backend_supports_vision(backend)
        ):
            unavailable.append(f"{backend}: text-only, skipped for a frames job")
            _log(f"Fallback {backend} is text-only; skipping (session has frames)...")
            continue

        # If this is a fallback, log it (with the model actually used)
        if backend != config.backend:
            _log(f"Falling back to {backend} ({_default_model_for_backend(backend)})...")

        # Inform the user when the local two-pass flow is about to run, since
        # it takes noticeably longer than a single LLM call.
        if backend == "ollama" and not config.ollama_singlepass and not config.template:
            _log("Running Ollama two-pass summarization (extract + format)...")

        try:
            result = _dispatch(
                backend,
                system_prompt,
                user_prompt,
                config,
                transcript_text=transcript_text,
                language=language,
            )
            result.preset = config.preset
            result.template = config.template
            # OR rather than assign: a backend may already have recorded its
            # own internal fallback.  The tinfoil sibling-model fallback
            # (0.18.1) keeps backend == config.backend, so a plain assignment
            # here silently erased it from the meta sidecar.
            result.fallback_used = (backend != config.backend) or result.fallback_used
            if backend != config.backend:
                _log(f"Summary generated via fallback backend {backend}")
            return result
        except Exception as exc:
            last_error = exc
            _log(f"{backend} failed: {exc}")
            continue

    # All backends failed.  Distinguish "never reachable" from "tried and
    # errored": if nothing was dispatched, the useful information is *why*
    # each backend was skipped, not a null last error.
    if last_error is None and unavailable:
        raise RuntimeError(
            "No summary backend is available. " + " | ".join(unavailable)
        )
    raise RuntimeError(f"All summary backends failed. Last error: {last_error}")
