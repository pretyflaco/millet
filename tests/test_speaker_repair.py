"""Per-segment speaker repair (millet.speaker_repair).  numpy only, no torch."""

from __future__ import annotations

import numpy as np
import pytest

from millet import speaker_repair as sr
from millet.speaker_repair import (
    RepairItem,
    apply_moves,
    average_linkage_bipartition,
    plan_speaker_repair,
    repair_speakers,
)
from millet.transcribe import Segment

DIM = 256


def _voice(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM)
    return v / np.linalg.norm(v)


def _utterance(voice: np.ndarray, rng: np.random.Generator, noise: float = 1.0) -> np.ndarray:
    """One segment's embedding: ~0.7 cosine to its voice at noise=1.0."""
    v = voice + rng.normal(scale=noise / np.sqrt(DIM), size=DIM)
    return v / np.linalg.norm(v)


def _items(spec, seed=0):
    """spec: list of (diarized speaker, true voice, duration)."""
    rng = np.random.default_rng(seed)
    return [
        RepairItem(i, spk, dur, _utterance(voice, rng))
        for i, (spk, voice, dur) in enumerate(spec)
    ]


KEMAL, ROB, THIRD = _voice(1), _voice(2), _voice(3)


def _interview(with_rob_cluster: bool):
    """Shape of 01M6EV6Z: everything in SPEAKER_01, Rob's answers included."""
    spec = []
    for k in range(17):
        spec.append(("SPEAKER_01", KEMAL, 1.5))
        if k < 9:
            spec.append(("SPEAKER_01", ROB, 0.95))  # 9 answers, 8.55 s
    if with_rob_cluster:
        spec.append(("SPEAKER_00", ROB, 4.4))
    return spec


class TestBipartition:
    def test_separates_two_voices(self):
        items = _items([("x", KEMAL, 1.0)] * 10 + [("x", ROB, 1.0)] * 5)
        labels = average_linkage_bipartition(np.stack([it.embedding for it in items]))
        assert len(set(labels[:10])) == 1 and len(set(labels[10:])) == 1
        assert labels[0] != labels[10]

    def test_deterministic(self):
        emb = np.stack([it.embedding for it in _items([("x", KEMAL, 1.0)] * 20)])
        assert np.array_equal(average_linkage_bipartition(emb), average_linkage_bipartition(emb))

    def test_needs_two_rows(self):
        with pytest.raises(ValueError):
            average_linkage_bipartition(np.ones((1, 4)))


class TestPlan:
    def test_interview_answers_join_the_interviewee(self):
        items = _items(_interview(with_rob_cluster=True))
        moves = plan_speaker_repair(items, {"SPEAKER_00", "SPEAKER_01"})
        assert len(moves) == 1
        mv = moves[0]
        assert (mv.source, mv.target, mv.new_speaker) == ("SPEAKER_01", "SPEAKER_00", False)
        rob = {it.index for it in items if it.speaker == "SPEAKER_01"
               and it.duration == 0.95}
        assert set(mv.indices) == rob
        assert mv.split_cos < sr.REPAIR_MAX_CENTROID_COS
        assert mv.join_cos >= sr.REPAIR_JOIN_MIN_COS

    def test_interview_without_a_second_cluster_makes_a_new_speaker(self):
        items = _items(_interview(with_rob_cluster=False))
        moves = plan_speaker_repair(items, {"SPEAKER_01"})
        assert [(m.source, m.target, m.new_speaker) for m in moves] == [
            ("SPEAKER_01", "SPEAKER_02", True),
        ]

    def test_new_speaker_needs_more_speech_than_a_join(self):
        # Same split, but only 5 x 0.95 s of the second voice and nowhere to
        # join: too little evidence to invent a person.
        spec = [("SPEAKER_00", KEMAL, 1.5)] * 17 + [("SPEAKER_00", ROB, 0.95)] * 5
        assert plan_speaker_repair(_items(spec), {"SPEAKER_00"}) == []

    def test_single_speaker_is_left_alone(self):
        spec = [("SPEAKER_00", KEMAL, 2.0)] * 60
        assert plan_speaker_repair(_items(spec), {"SPEAKER_00"}) == []

    def test_one_outlier_is_not_a_speaker(self):
        spec = [("SPEAKER_00", KEMAL, 2.0)] * 30 + [("SPEAKER_00", ROB, 6.0)]
        assert plan_speaker_repair(_items(spec), {"SPEAKER_00"}) == []

    def test_similar_halves_are_not_split(self):
        # Two "voices" at ~0.6 cosine: one person in two conditions, not two.
        near = KEMAL + 0.8 * THIRD
        near /= np.linalg.norm(near)
        rng = np.random.default_rng(5)
        items = [RepairItem(i, "SPEAKER_00", 2.0, _utterance(KEMAL, rng, 0.3))
                 for i in range(20)]
        items += [RepairItem(20 + i, "SPEAKER_00", 2.0, _utterance(near, rng, 0.3))
                  for i in range(10)]
        assert float(KEMAL @ near) > sr.REPAIR_MAX_CENTROID_COS
        assert plan_speaker_repair(items, {"SPEAKER_00"}) == []

    def test_small_clusters_are_not_examined(self):
        spec = [("SPEAKER_00", KEMAL, 3.0)] * 3 + [("SPEAKER_00", ROB, 3.0)] * 2
        assert plan_speaker_repair(_items(spec), {"SPEAKER_00"}) == []

    def test_new_ids_skip_existing_ones(self):
        items = _items(_interview(with_rob_cluster=False))
        moves = plan_speaker_repair(items, {"SPEAKER_01", "SPEAKER_02", "SPEAKER_07"})
        assert moves[0].target == "SPEAKER_08"

    def test_large_cluster_uses_a_subsample(self, monkeypatch):
        monkeypatch.setattr(sr, "REPAIR_MAX_POINTS", 20)
        spec = [("SPEAKER_00", KEMAL, 1.5)] * 40 + [("SPEAKER_00", ROB, 1.5)] * 12
        moves = plan_speaker_repair(_items(spec), {"SPEAKER_00"})
        assert len(moves) == 1
        assert set(moves[0].indices) == set(range(40, 52))


class TestApply:
    def test_moves_segments_and_their_words(self):
        segs = [
            Segment(0.0, 1.0, "hi", "SPEAKER_01", [{"word": "hi", "speaker": "SPEAKER_01"}]),
            Segment(1.0, 2.0, "yo", "SPEAKER_01", [{"word": "yo", "speaker": "SPEAKER_01"},
                                                     {"word": "x"}]),
        ]
        mv = sr.RepairMove("SPEAKER_01", "SPEAKER_00", (1,), 1.0, 0.1, 0.7)
        apply_moves(segs, [mv])
        assert segs[0].speaker == "SPEAKER_01"
        assert segs[1].speaker == "SPEAKER_00"
        assert segs[1].words == [{"word": "yo", "speaker": "SPEAKER_00"}, {"word": "x"}]


class TestMonoTranscribeWiring:
    """transcribe()'s mono path runs the repair after diarization and builds
    the speaker list from the repaired segments."""

    def _run(self, monkeypatch, tmp_path, **config_kwargs):
        import sys
        import wave
        from types import SimpleNamespace

        from millet import transcribe as tr

        wav = tmp_path / "mono.wav"
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(np.zeros(16000 * 3, dtype=np.int16).tobytes())

        asr = {"language": "en", "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hey Rob, how are you?"},
            {"start": 1.0, "end": 2.0, "text": "I'm doing well."},
        ]}

        def assign(diarized, result):
            for seg in result["segments"]:
                seg["speaker"] = "SPEAKER_01"
            return result

        class FakePipeline:
            def __init__(self, **kwargs):
                pass

            def __call__(self, audio, **kwargs):
                return "diarized"

        whisperx = SimpleNamespace(load_audio=lambda p: np.zeros(48000, dtype=np.float32),
                                   assign_word_speakers=assign)
        monkeypatch.setitem(sys.modules, "whisperx", whisperx)
        monkeypatch.setitem(sys.modules, "whisperx.diarize",
                            SimpleNamespace(DiarizationPipeline=FakePipeline))
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
        monkeypatch.setattr(tr, "_load_whisperx_asr_model", lambda config, lang: None)
        monkeypatch.setattr(tr, "_resolve_channel_language",
                            lambda model, audio, config, lang, label: ("en", 0.99, "en"))
        monkeypatch.setattr(tr, "_transcribe_asr", lambda *a, **k: asr)

        calls = []

        def fake_repair(audio, segments, device=None):
            calls.append(device)
            segments[1].speaker = "SPEAKER_00"
            return []

        monkeypatch.setattr(sr, "repair_speakers", fake_repair)
        config = tr.TranscriptionConfig(
            asr_backend="whisperx", device="cpu", compute_type="int8",
            skip_alignment=True, audio_pad_seconds=0, hf_token="hf_test",
            **config_kwargs,
        )
        return tr.transcribe(wav, config), calls

    def test_repair_runs_and_speakers_follow_it(self, monkeypatch, tmp_path):
        transcript, calls = self._run(monkeypatch, tmp_path)
        assert calls == ["cpu"]
        assert [s.speaker for s in transcript.segments] == ["SPEAKER_01", "SPEAKER_00"]
        assert [s.id for s in transcript.speakers] == ["SPEAKER_00", "SPEAKER_01"]

    def test_can_be_disabled(self, monkeypatch, tmp_path):
        transcript, calls = self._run(monkeypatch, tmp_path, speaker_repair=False)
        assert calls == []
        assert [s.id for s in transcript.speakers] == ["SPEAKER_01"]


class TestRepairSpeakers:
    def _segments(self, items):
        return [Segment(float(it.index), float(it.index) + it.duration, "t", it.speaker, None)
                for it in items]

    def test_end_to_end_with_stubbed_embedder(self, monkeypatch):
        items = _items(_interview(with_rob_cluster=True))
        segs = self._segments(items)
        monkeypatch.setattr(sr, "embed_segments", lambda audio, segments, device=None: items)
        logged: list[str] = []
        moves = repair_speakers(np.zeros(16000), segs, log=logged.append)
        assert len(moves) == 1
        assert {s.speaker for s in segs} == {"SPEAKER_00", "SPEAKER_01"}
        assert sum(s.speaker == "SPEAKER_00" for s in segs) == 10
        assert any("SPEAKER_01 -> SPEAKER_00" in line for line in logged)

    def test_embedding_failure_leaves_segments_alone(self, monkeypatch):
        segs = self._segments(_items(_interview(with_rob_cluster=True)))
        before = [s.speaker for s in segs]

        def boom(*a, **k):
            raise RuntimeError("no model")

        monkeypatch.setattr(sr, "embed_segments", boom)
        logged: list[str] = []
        assert repair_speakers(np.zeros(16000), segs, log=logged.append) == []
        assert [s.speaker for s in segs] == before
        assert "speaker repair skipped (no model)" in logged[0]
