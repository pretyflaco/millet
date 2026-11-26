"""Tests for custom domain vocabulary (ASR prompting, alias cleanup, summary glossary)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from millet import paths
from millet.transcribe import Segment, Speaker, Transcript, TranscriptionConfig
from millet.vocabulary import (
    Vocabulary,
    apply_aliases_to_text,
    apply_aliases_to_words,
    apply_vocabulary_to_transcript,
    build_hotwords,
    build_initial_prompt,
    build_summary_glossary_block,
    load_vocabulary,
    merge_vocabularies,
    save_vocabulary,
    speaker_names_from_profiles,
)


def test_vocabulary_roundtrip_and_normalization(tmp_path: Path):
    vocab = Vocabulary.from_dict(
        {
            "terms": [" Vezir ", "millet", "vezir", "", "NIP-46"],
            "aliases": {
                "Vezir": ["vzr", "vizier", "Vezir", "vzr"],
                "pyannote": ["pie a note"],
            },
            "glossary": {
                "Vezir": "self-hosted\nmeeting   service",
            },
        }
    )

    assert vocab.terms == ("Vezir", "millet", "NIP-46")
    assert vocab.aliases == {"Vezir": ("vzr", "vizier"), "pyannote": ("pie a note",)}
    assert vocab.glossary == {"Vezir": "self-hosted meeting service"}
    # all_terms includes canonical keys from aliases/glossary without duplicates
    assert vocab.all_terms == ("Vezir", "millet", "NIP-46", "pyannote")

    dest = tmp_path / "vocabulary.json"
    save_vocabulary(vocab, dest)
    loaded = load_vocabulary(dest)
    assert loaded == vocab


def test_load_vocabulary_missing_corrupt_and_plain_txt(tmp_path: Path):
    assert load_vocabulary(tmp_path / "missing.json").is_empty

    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    assert load_vocabulary(bad).is_empty

    txt = tmp_path / "words.txt"
    txt.write_text("# comment\nVezir, millet\npyannote\n", encoding="utf-8")
    loaded = load_vocabulary(txt)
    assert loaded.terms == ("Vezir", "millet", "pyannote")


def test_vocabulary_path_resolution(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MILLET_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("MILLET_VOCABULARY_PATH", raising=False)
    monkeypatch.delenv("MEET_VOCABULARY_PATH", raising=False)

    assert paths.vocabulary_path() == tmp_path / "cfg" / "vocabulary.json"
    assert paths.vocabulary_path("team-a") == tmp_path / "cfg" / "team-a" / "vocabulary.json"

    override = tmp_path / "custom.json"
    monkeypatch.setenv("MILLET_VOCABULARY_PATH", str(override))
    assert paths.vocabulary_path() == override
    # Explicit team wins over process-wide env var
    assert paths.vocabulary_path("team-a") == tmp_path / "cfg" / "team-a" / "vocabulary.json"


def test_merge_vocabularies_preserves_priority():
    v1 = Vocabulary.from_dict(
        {
            "terms": ["Vezir"],
            "aliases": {"Vezir": ["vzr"]},
            "glossary": {"Vezir": "primary description"},
        }
    )
    v2 = Vocabulary.from_dict(
        {
            "terms": ["vezir", "millet"],
            "aliases": {"vezir": ["vizier", "vzr"]},
            "glossary": {"vezir": "secondary", "millet": "ASR pipeline"},
        }
    )
    merged = merge_vocabularies(v1, v2)
    assert merged.terms == ("Vezir", "millet")
    assert merged.aliases == {"Vezir": ("vzr", "vizier")}
    assert merged.glossary == {"Vezir": "primary description", "millet": "ASR pipeline"}


def test_speaker_names_from_profiles_filters_placeholders(tmp_path: Path):
    p = tmp_path / "speaker_profiles.json"
    p.write_text(
        json.dumps(
            {
                "Alice": {"embedding": [0.1], "n_sessions": 2},
                "Bob": {"embedding": [0.2], "n_sessions": 1},
                "YOU": {"embedding": [0.3], "n_sessions": 1},
                "REMOTE_1": {"embedding": [0.4], "n_sessions": 1},
                "SPEAKER_00": {"embedding": [0.5], "n_sessions": 1},
                "CROSSTALK": {"embedding": [0.6], "n_sessions": 1},
            }
        ),
        encoding="utf-8",
    )
    assert speaker_names_from_profiles(p) == ("Alice", "Bob")


def test_build_initial_prompt_and_hotwords_respect_budget():
    vocab = Vocabulary.from_dict({"terms": ["Vezir", "millet", "NIP-46"]})
    prompt = build_initial_prompt(vocab, extra_terms=["Alice", "vezir"])
    assert prompt == "Vezir, millet, NIP-46, Alice."
    assert not prompt.startswith("Vocabulary:")

    hw = build_hotwords(vocab, extra_terms=["Alice"])
    assert hw == "Vezir millet NIP-46 Alice"

    # Length cap drops overflow cleanly without cutting a word in half
    short = build_initial_prompt(vocab, max_chars=16)
    assert short == "Vezir, millet."
    assert len(short) <= 16


def test_apply_aliases_to_text_unicode_casing_and_multiword():
    aliases = {
        "Vezir": ("vzr", "vizier", "the zeer"),
        "millet": ("milet",),
        "pyannote": ("pie a note",),
    }
    text, n = apply_aliases_to_text(
        "We deployed vzr and the zeer with pie a note. Milet works, vizier's UI too.",
        aliases,
    )
    assert n == 5
    assert text == "We deployed Vezir and Vezir with pyannote. Millet works, Vezir's UI too."

    # Word boundary: must not replace inside unrelated words
    untouched, n0 = apply_aliases_to_text("subvzrx grandvizier", aliases)
    assert n0 == 0
    assert untouched == "subvzrx grandvizier"


def test_apply_aliases_to_words_single_and_multiword():
    aliases = {
        "Vezir": ("vzr", "vizier"),
        "pyannote": ("pie a note",),
    }
    words = [
        {"word": "Using", "start": 0.0, "end": 0.4, "score": 0.9},
        {"word": "vzr,", "start": 0.4, "end": 0.8, "score": 0.8},
        {"word": "with", "start": 0.8, "end": 1.0, "score": 0.9},
        {"word": "pie", "start": 1.0, "end": 1.2, "score": 0.7},
        {"word": "a", "start": 1.2, "end": 1.3, "score": 0.8},
        {"word": "note.", "start": 1.3, "end": 1.6, "score": 0.9},
    ]
    n = apply_aliases_to_words(words, aliases)
    assert n == 2
    assert [w["word"] for w in words] == ["Using", "Vezir,", "with", "pyannote."]
    # Merged multi-word keeps start of first and end of last
    assert words[3]["start"] == 1.0
    assert words[3]["end"] == 1.6
    assert words[3]["score"] == 0.8


def test_apply_vocabulary_to_transcript():
    vocab = Vocabulary.from_dict({"aliases": {"Vezir": ["vzr", "vizier"]}})
    tr = Transcript(
        segments=[
            Segment(
                start=0.0,
                end=2.0,
                text="Let's check vzr and vizier.",
                speaker="Alice",
                words=[
                    {"word": "Let's", "start": 0.0, "end": 0.3},
                    {"word": "check", "start": 0.3, "end": 0.6},
                    {"word": "vzr", "start": 0.6, "end": 1.0},
                    {"word": "and", "start": 1.0, "end": 1.2},
                    {"word": "vizier.", "start": 1.2, "end": 2.0},
                ],
            )
        ],
        speakers=[Speaker(id="Alice", label="Alice")],
        language="en",
        audio_file="meeting.ogg",
    )
    n = apply_vocabulary_to_transcript(tr, vocab)
    assert n == 2
    assert tr.segments[0].text == "Let's check Vezir and Vezir."
    assert [w["word"] for w in tr.segments[0].words] == [
        "Let's",
        "check",
        "Vezir",
        "and",
        "Vezir.",
    ]


def test_summary_glossary_block_and_injection(tmp_path: Path, monkeypatch):
    from millet import summarize as sum_mod

    # Empty vocabulary leaves prompt completely untouched
    assert build_summary_glossary_block(Vocabulary()) == ""

    vocab = Vocabulary.from_dict(
        {
            "terms": ["Vezir", "millet"],
            "aliases": {"Vezir": ["vzr", "vizier"]},
            "glossary": {"Vezir": "team meeting service"},
        }
    )
    block = build_summary_glossary_block(vocab)
    assert "KNOWN TERMINOLOGY" in block
    assert "- Vezir (often misheard by ASR as: vzr, vizier) — team meeting service" in block
    assert "- millet" in block
    assert "Do NOT introduce any term" in block

    captured: dict[str, str] = {}

    def _fake_tinfoil(system_prompt, user_prompt, cfg):
        captured["system_prompt"] = system_prompt
        return sum_mod.MeetingSummary(
            markdown="## Overview\nTest summary.",
            model="test-model",
            elapsed_seconds=0.1,
            backend="tinfoil",
        )

    monkeypatch.setattr(sum_mod, "is_backend_available", lambda _cfg: True)
    monkeypatch.setattr(sum_mod, "_summarize_tinfoil", _fake_tinfoil)

    vocab_file = tmp_path / "vocabulary.json"
    save_vocabulary(vocab, vocab_file)
    cfg = sum_mod.SummaryConfig(backend="tinfoil", vocabulary_path=vocab_file)
    sum_mod.summarize("Alice: We updated vzr.", cfg)
    assert "KNOWN TERMINOLOGY" in captured["system_prompt"]
    assert "Vezir" in captured["system_prompt"]


def test_whisperx_load_model_passes_asr_options_and_falls_back_on_hotwords(
    tmp_path: Path, monkeypatch
):
    from millet import transcribe as tr_mod

    vocab_file = tmp_path / "vocabulary.json"
    save_vocabulary(Vocabulary.from_dict({"terms": ["Vezir", "millet"]}), vocab_file)

    calls: list[dict] = []

    def _fake_load_model(model, device, **kwargs):
        calls.append(kwargs)
        if "hotwords" in (kwargs.get("asr_options") or {}):
            raise TypeError("unexpected keyword argument 'hotwords'")
        return "FAKE_MODEL"

    fake_wx = types.ModuleType("whisperx")
    fake_wx.load_model = _fake_load_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "whisperx", fake_wx)

    cfg = TranscriptionConfig(
        device="cpu",
        compute_type="int8",
        vocabulary_path=vocab_file,
        auto_vocabulary=False,
    )
    result = tr_mod._load_whisperx_asr_model(cfg, "en")
    assert result == "FAKE_MODEL"
    assert len(calls) == 2
    assert calls[0]["asr_options"]["initial_prompt"] == "Vezir, millet."
    assert calls[0]["asr_options"]["hotwords"] == "Vezir millet"
    assert calls[1]["asr_options"] == {"initial_prompt": "Vezir, millet."}
