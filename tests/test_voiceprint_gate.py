"""Tests for the voiceprint auto-apply gate (weak_match_reason).

Regression: over-segmented backchannel clusters produce a barely-confident,
ambiguous voiceprint match (e.g. 0.69 with a 0.13 margin) that mis-names a
phantom speaker.  The gate suppresses such matches while still auto-applying
strong, well-separated ones (e.g. 0.93 with a 0.40 margin).
"""
from __future__ import annotations

from millet.cli.label import weak_match_reason
from millet.voiceprint import (
    MATCH_AUTOAPPLY_CONFIDENCE,
    MATCH_AUTOAPPLY_MARGIN,
    MATCH_MIN_SPEECH_SECONDS,
    SpeakerMatch,
)


def test_strong_well_separated_match_applies():
    """0.93 confidence / 0.40 margin (a real Lukas-style match) → applies."""
    m = SpeakerMatch("Lukas", 0.93, evidence_seconds=958.0, margin=0.40)
    assert weak_match_reason(m) is None


def test_ambiguous_low_conf_low_margin_gated():
    """0.69 / 0.13 margin (the false 'Roark' signature) → gated."""
    m = SpeakerMatch("Roark", 0.686, evidence_seconds=62.4, margin=0.128)
    reason = weak_match_reason(m)
    assert reason is not None
    assert "ambiguous" in reason


def test_strong_confidence_low_margin_still_applies():
    """High absolute confidence rescues a low margin (two similar real profiles)."""
    m = SpeakerMatch("X", 0.80, evidence_seconds=30.0, margin=0.05)
    assert m.confidence >= MATCH_AUTOAPPLY_CONFIDENCE
    assert weak_match_reason(m) is None


def test_clear_margin_rescues_modest_confidence():
    """A clear margin applies even when below the absolute auto-apply floor."""
    m = SpeakerMatch("Y", 0.70, evidence_seconds=30.0, margin=0.25)
    assert m.confidence < MATCH_AUTOAPPLY_CONFIDENCE
    assert m.margin >= MATCH_AUTOAPPLY_MARGIN
    assert weak_match_reason(m) is None


def test_thin_cluster_gated_even_if_unambiguous():
    """Too little embeddable speech → gated regardless of confidence/margin."""
    m = SpeakerMatch("Z", 0.95, evidence_seconds=2.0, margin=0.50)
    assert m.evidence_seconds < MATCH_MIN_SPEECH_SECONDS
    reason = weak_match_reason(m)
    assert reason is not None
    assert "speech" in reason


def test_positional_backcompat_not_gated():
    """Matches built positionally (no evidence/margin) use trustworthy
    sentinels and are never gated."""
    m = SpeakerMatch("W", 0.70)
    assert m.evidence_seconds == 0.0
    assert m.margin == 1.0
    assert weak_match_reason(m) is None


# ─── many-to-one matching (0.12.11) ──────────────────────────────────────────
# Regression: diarization over-segments ONE person into several clusters
# (volume/mic changes, cross-channel bleed).  Greedy 1:1 matching used to name
# only the first cluster and leave the rest raw → needs_labeling + "Destiny ×3".
# identify_speakers now lets an already-claimed profile ALSO claim additional
# clusters when each is confident on its own.

import numpy as np  # noqa: E402

from millet import voiceprint as _vp  # noqa: E402
from millet.transcribe import Segment, Speaker  # noqa: E402
from millet.voiceprint import (  # noqa: E402
    MATCH_MANY_TO_ONE_CONFIDENCE,
    SpeakerProfile,
    identify_speakers,
)


def _unit(*vals: float) -> np.ndarray:
    v = np.array(vals, dtype=np.float32)
    return v / np.linalg.norm(v)


def _run_identify(monkeypatch, *, cluster_embeddings, profiles):
    """Drive identify_speakers with mocked embedding extraction.

    ``cluster_embeddings``: {speaker_id -> unit vector}.  Each cluster gets one
    long (well over the 4s evidence floor) segment so the auto-apply gate never
    trips for confident matches.
    """
    speakers = [Speaker(id=sid, label=sid) for sid in cluster_embeddings]
    segments = [
        Segment(start=float(i * 100), end=float(i * 100 + 60), text="x", speaker=sid)
        for i, sid in enumerate(cluster_embeddings)
    ]
    channel_map = {sid: "system" for sid in cluster_embeddings}

    monkeypatch.setattr(_vp, "load_profiles", lambda profiles_path=None: profiles)
    monkeypatch.setattr(_vp, "_get_inference", lambda: object())
    monkeypatch.setattr(
        _vp, "_extract_channel_audio",
        lambda audio_path, channel: (np.ones(16000, dtype=np.float32), 16000),
    )

    # _embed_segments is called once per cluster, in the loop order of all_ids.
    # Map by the (sorted-by-duration) selected segments isn't needed: we key off
    # which speaker's audio is being embedded.  Use a counter over speaker ids.
    order = list(cluster_embeddings)
    state = {"i": 0}

    def fake_embed(samples, sr, selected, inference):
        sid = order[state["i"]]
        state["i"] += 1
        return cluster_embeddings[sid]

    monkeypatch.setattr(_vp, "_embed_segments", fake_embed)

    from pathlib import Path
    return identify_speakers(
        Path("/tmp/fake.ogg"), segments, speakers, channel_map,
        profiles_path=Path("/tmp/profiles.json"),
    )


def test_two_clusters_same_person_both_matched(monkeypatch):
    """Two near-identical Destiny clusters → BOTH named Destiny (many-to-one)."""
    destiny = _unit(1.0, 0.0, 0.0)
    andrej = _unit(0.0, 1.0, 0.0)
    matches = _run_identify(
        monkeypatch,
        cluster_embeddings={
            "SPEAKER_00": _unit(1.0, 0.02, 0.0),   # Destiny (best)
            "SPEAKER_01": andrej,                   # Andrej
            "SPEAKER_02": _unit(0.98, 0.05, 0.0),  # Destiny again (high conf)
        },
        profiles={
            "Destiny": SpeakerProfile("Destiny", destiny, 3),
            "Andrej": SpeakerProfile("Andrej", andrej, 3),
        },
    )
    assert matches["SPEAKER_00"].name == "Destiny"
    assert matches["SPEAKER_02"].name == "Destiny"  # would be raw under 1:1
    assert matches["SPEAKER_01"].name == "Andrej"


def test_secondary_cluster_below_autoapply_stays_unmatched(monkeypatch):
    """A 2nd cluster that only weakly resembles the claimed profile is NOT
    folded on (stays raw → human review), guarding against over-merge."""
    destiny = _unit(1.0, 0.0, 0.0)
    # Second 'Destiny-ish' cluster at ~0.66 cosine: above MATCH_THRESHOLD (0.65)
    # but below the many-to-one fold floor → not auto-folded.
    weak = _unit(0.66, 0.751, 0.0)
    assert float(np.dot(weak, destiny)) < MATCH_MANY_TO_ONE_CONFIDENCE
    matches = _run_identify(
        monkeypatch,
        cluster_embeddings={
            "SPEAKER_00": destiny,        # strong Destiny
            "SPEAKER_01": weak,           # weak resemblance
        },
        profiles={"Destiny": SpeakerProfile("Destiny", destiny, 3)},
    )
    assert matches["SPEAKER_00"].name == "Destiny"
    assert "SPEAKER_01" not in matches


# ─── many-to-one "different person" guard (0.13.3) ───────────────────────────
# Regression: two DIFFERENT people on the same compressed system channel,
# speaking in non-overlapping turns, where only ONE is enrolled.  Pass 1 names
# the enrolled person's cluster; the unenrolled person's cluster (a NEW founder
# with no profile) is left unmatched, then Pass 2 used to FOLD it onto the
# enrolled profile whenever the cross-voice cosine merely cleared 0.72 — mixing
# two people into one name (field case: Humphrey mislabeled "Bright").  Pass 2
# now requires a higher absolute floor (MATCH_MANY_TO_ONE_CONFIDENCE) AND a
# clear runner-up margin, so a similar-but-distinct voice is NOT folded.


def test_distinct_similar_voice_not_folded_onto_named_profile(monkeypatch):
    """New person resembling an enrolled profile at ~0.74 (no runner-up margin)
    is NOT folded — stays raw for human review (the Humphrey→Bright case)."""
    bright = _unit(1.0, 0.0, 0.0)
    # Humphrey: a distinct voice that lands at ~0.74 cosine to Bright — above
    # the ordinary auto-apply floor (0.72) but below the stricter fold floor.
    humphrey = _unit(0.74, 0.6726, 0.0)
    score = float(np.dot(humphrey, bright))
    assert 0.72 <= score < MATCH_MANY_TO_ONE_CONFIDENCE
    matches = _run_identify(
        monkeypatch,
        cluster_embeddings={
            "SPEAKER_06": bright,      # enrolled Bright
            "SPEAKER_08": humphrey,    # new, unenrolled — must NOT become Bright
        },
        profiles={"Bright": SpeakerProfile("Bright", bright, 3)},
    )
    assert matches["SPEAKER_06"].name == "Bright"
    assert "SPEAKER_08" not in matches


def test_high_confidence_fold_needs_margin(monkeypatch):
    """Even a high absolute score does not fold when the cluster resembles a
    SECOND profile nearly as much (small runner-up margin = ambiguous)."""
    bright = _unit(1.0, 0.0, 0.0)
    other = _unit(0.85, 0.5268, 0.0)   # a second enrolled profile, close to Bright
    # A leftover cluster that scores ~0.90 to Bright but ~0.88 to `other`:
    # high absolute confidence, tiny margin → ambiguous, must stay raw.
    amb = _unit(0.94, 0.32, 0.0)
    top = max(float(np.dot(amb, bright)), float(np.dot(amb, other)))
    second = min(float(np.dot(amb, bright)), float(np.dot(amb, other)))
    assert top >= MATCH_MANY_TO_ONE_CONFIDENCE
    assert (top - second) < 0.15  # below MATCH_AUTOAPPLY_MARGIN
    matches = _run_identify(
        monkeypatch,
        cluster_embeddings={
            "SPEAKER_00": bright,      # claims Bright (exact)
            "SPEAKER_01": other,       # claims Other (exact)
            "SPEAKER_02": amb,         # ambiguous leftover
        },
        profiles={
            "Bright": SpeakerProfile("Bright", bright, 3),
            "Other": SpeakerProfile("Other", other, 3),
        },
    )
    assert matches["SPEAKER_00"].name == "Bright"
    assert matches["SPEAKER_01"].name == "Other"
    assert "SPEAKER_02" not in matches


def test_single_cluster_per_profile_unchanged(monkeypatch):
    """Baseline: distinct people still map 1:1."""
    destiny = _unit(1.0, 0.0, 0.0)
    andrej = _unit(0.0, 1.0, 0.0)
    matches = _run_identify(
        monkeypatch,
        cluster_embeddings={"SPEAKER_00": destiny, "SPEAKER_01": andrej},
        profiles={
            "Destiny": SpeakerProfile("Destiny", destiny, 3),
            "Andrej": SpeakerProfile("Andrej", andrej, 3),
        },
    )
    assert matches["SPEAKER_00"].name == "Destiny"
    assert matches["SPEAKER_01"].name == "Andrej"


# ─── CROSSTALK is reserved: never enrolled, never matched ────────────────────
# CROSSTALK is mixed audio from several people.  A profile built from it would
# later "match" real speakers, so no path may create or use one.


def test_crosstalk_profile_is_never_matched(monkeypatch):
    """Even if an older build enrolled a CROSSTALK profile, identify ignores it."""
    crosstalk = _unit(1.0, 0.0, 0.0)
    kemal = _unit(0.0, 1.0, 0.0)
    matches = _run_identify(
        monkeypatch,
        cluster_embeddings={"SPEAKER_00": crosstalk, "SPEAKER_01": kemal},
        profiles={
            "CROSSTALK": SpeakerProfile("CROSSTALK", crosstalk, 5),
            "Kemal": SpeakerProfile("Kemal", kemal, 5),
        },
    )
    assert "SPEAKER_00" not in matches
    assert matches["SPEAKER_01"].name == "Kemal"


def test_crosstalk_label_never_updates_profiles(monkeypatch):
    """A reserved label short-circuits before any embedding work."""
    from pathlib import Path

    def boom():
        raise AssertionError("embedding model must not load for CROSSTALK")

    monkeypatch.setattr(_vp, "_get_inference", boom)
    _vp.update_profiles_from_confirmed_labels(
        Path("/tmp/fake.ogg"), [], {"REMOTE": "CROSSTALK"}, {},
        profiles_path=Path("/tmp/profiles.json"),
    )


# ─── Learning hygiene: only dense real speech builds a profile ───────────────
# The "Pattern" incident (vezir, 2026-09): profile updates took a cluster's
# LONGEST segments — for a filler cluster, fillers Whisper stretched over
# silence, i.e. room tone + the scribe's echo.  The profile ended up 0.86 to
# the scribe's echo and 0.10 to the person it was named after, then matched
# filler buckets in every meeting at ~0.9.

from millet.voiceprint import (  # noqa: E402
    LEARN_MIN_SPEECH_SECONDS,
    learnable_segments,
)


def _pattern_cluster(sid="Pattern"):
    """Filler cluster as it appeared on 01M3QN8F: stretched 'Yeah.'s."""
    return [
        Segment(start=10.0, end=16.7, text="Yeah.", speaker=sid),
        Segment(start=40.0, end=45.2, text="Okay.", speaker=sid),
        Segment(start=80.0, end=82.3, text="Mm-hmm.", speaker=sid),
        Segment(start=95.0, end=97.1, text="So it's a bot management interface.", speaker=sid),
        Segment(start=120.0, end=120.4, text="Bye.", speaker=sid),
    ]


def test_filler_cluster_is_not_learnable():
    assert learnable_segments(_pattern_cluster(), "Pattern") == []


def test_dense_speech_is_learnable_most_words_first():
    segs = [
        Segment(start=0.0, end=4.0, text="we shipped the migration fix this morning finally", speaker="G"),
        Segment(start=10.0, end=16.7, text="Yeah.", speaker="G"),  # stretched filler: excluded
        Segment(start=20.0, end=23.0, text="and the payouts look right on staging now", speaker="G"),
        Segment(start=30.0, end=31.0, text="one two three four five six", speaker="G"),  # < 1.5 s
    ]
    assert learnable_segments(segs, "G") == [(0.0, 4.0), (20.0, 23.0)]
    assert LEARN_MIN_SPEECH_SECONDS == 4.0


def test_is_dense_utterance():
    from millet.voiceprint import is_dense_utterance

    def seg(s, e, text):
        return Segment(start=s, end=e, text=text, speaker="G")

    assert is_dense_utterance(seg(0.0, 4.0, "we shipped the migration fix this morning"))
    assert not is_dense_utterance(seg(0.0, 6.7, "Yeah."))                   # stretched filler
    assert not is_dense_utterance(seg(0.0, 1.0, "one two three four five six"))  # < 1.5 s
    assert not is_dense_utterance(seg(0.0, 5.0, "one two three four five six"))  # 1.2 words/s
    assert not is_dense_utterance(seg(0.0, 0.0, ""))


def test_too_little_dense_speech_is_not_learnable():
    segs = [Segment(start=0.0, end=2.0, text="we shipped the migration fix today", speaker="G")]
    assert learnable_segments(segs, "G") == []


def test_filler_cluster_never_updates_a_profile(monkeypatch):
    """Confirming a filler cluster's (pre-filled) name must not touch the DB."""
    from pathlib import Path

    def boom(*a, **k):
        raise AssertionError("must not embed or write for a filler cluster")

    monkeypatch.setattr(_vp, "_get_inference", boom)
    monkeypatch.setattr(_vp, "save_profiles", boom)
    _vp.update_profiles_from_confirmed_labels(
        Path("/tmp/fake.ogg"), _pattern_cluster(), {"Pattern": "Pattern"},
        {"Pattern": "system"}, profiles_path=Path("/tmp/profiles.json"),
    )


def test_update_learns_from_dense_spans_only(monkeypatch):
    from pathlib import Path

    seen: dict = {}
    monkeypatch.setattr(_vp, "_get_inference", lambda: object())
    monkeypatch.setattr(_vp, "load_profiles", lambda profiles_path=None: {})
    monkeypatch.setattr(
        _vp, "_extract_channel_audio",
        lambda audio_path, channel: (np.ones(16000, dtype=np.float32), 16000),
    )

    def fake_embed(samples, sr, selected, inference):
        seen["spans"] = list(selected)
        return _unit(1.0, 0.0, 0.0)

    monkeypatch.setattr(_vp, "_embed_segments", fake_embed)
    monkeypatch.setattr(_vp, "save_profiles", lambda p, profiles_path=None: seen.update(saved=p))
    segs = _pattern_cluster("SPEAKER_01") + [
        Segment(start=200.0, end=204.0, text="we shipped the migration fix this morning finally",
                speaker="SPEAKER_01"),
        Segment(start=210.0, end=213.0, text="and the payouts look right on staging now",
                speaker="SPEAKER_01"),
    ]
    _vp.update_profiles_from_confirmed_labels(
        Path("/tmp/fake.ogg"), segs, {"SPEAKER_01": "Gustavo"}, {"SPEAKER_01": "system"},
        profiles_path=Path("/tmp/profiles.json"),
    )
    # Neither the stretched "Yeah." (6.7 s) nor "Okay." (5.2 s) — the longest
    # segments, which the old selection preferred — reach the embedding; the
    # one real line in the filler part does (6 words in 2.1 s), most words first.
    assert seen["spans"] == [(200.0, 204.0), (210.0, 213.0), (95.0, 97.1)]
    assert "Gustavo" in seen["saved"]
