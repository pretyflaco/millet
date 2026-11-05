"""Tests for millet.transcribe — Transcript dataclass methods and speaker labeling."""

from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from millet.transcribe import (
    Segment,
    Speaker,
    Transcript,
    TranscriptionConfig,
    _apply_default_language_bias,
    _channel_correct_segments,
    _consolidate_remote_clusters,
    _correct_mic_bleed_segments,
    _dominant_channel_language,
    _isolated_backchannel_outliers,
    _merge_orphan_system_segments,
    _nearest_segment,
    _resolve_channel_language,
    _segment_gap,
    _segments_total_seconds,
    _split_segment_by_word_speaker,
    _transcribe_asr,
    _transcribe_dual_channel,
)
from millet.transcribe import transcribe as do_transcribe

# ─── Transcript.to_text() ──────────────────────────────────────────────────

class TestToText:
    def test_basic_format(self, transcript):
        text = transcript.to_text()
        lines = text.strip().split("\n")
        assert len(lines) == 6

    def test_speaker_labels_present(self, transcript):
        text = transcript.to_text()
        assert "YOU:" in text
        assert "REMOTE_1:" in text
        assert "REMOTE_2:" in text

    def test_timestamp_format(self, transcript):
        text = transcript.to_text()
        # First line: [00:00:00 --> 00:00:05]
        assert "[00:00:00 --> 00:00:05]" in text

    def test_missing_speaker(self):
        t = Transcript(
            segments=[Segment(start=0, end=1, text="hello", speaker=None)],
            speakers=[], language="en", audio_file="test.wav",
        )
        assert "UNKNOWN:" in t.to_text()


# ─── Transcript.to_srt() ───────────────────────────────────────────────────

class TestToSrt:
    def test_srt_numbering(self, transcript):
        srt = transcript.to_srt()
        # SRT entries are numbered 1..6
        assert srt.startswith("1\n")
        assert "\n6\n" in srt

    def test_srt_timestamp_format(self, transcript):
        srt = transcript.to_srt()
        # First timestamp
        assert "00:00:00,000 --> 00:00:05,500" in srt

    def test_srt_speaker_brackets(self, transcript):
        srt = transcript.to_srt()
        assert "[YOU]" in srt
        assert "[REMOTE_1]" in srt


# ─── Transcript.to_json() ──────────────────────────────────────────────────

class TestToJson:
    def test_round_trip(self, transcript):
        """JSON output should parse back to the same data."""
        data = json.loads(transcript.to_json())
        assert data["language"] == "en"
        assert data["duration"] == 42.0
        assert len(data["segments"]) == 6
        assert len(data["speakers"]) == 3

    def test_segment_fields(self, transcript):
        data = json.loads(transcript.to_json())
        seg = data["segments"][0]
        assert seg["start"] == 0.0
        assert seg["end"] == 5.5
        assert seg["speaker"] == "YOU"
        assert "Hello everyone" in seg["text"]

    def test_speaker_fields(self, transcript):
        data = json.loads(transcript.to_json())
        sp = data["speakers"][0]
        assert sp["id"] == "YOU"
        assert sp["label"] == "YOU"


# ─── Transcript.save() ─────────────────────────────────────────────────────

class TestSave:
    def test_creates_all_files(self, transcript, tmp_path):
        files = transcript.save(tmp_path, basename="test")
        assert (tmp_path / "test.txt").exists()
        assert (tmp_path / "test.srt").exists()
        assert (tmp_path / "test.json").exists()
        assert "text" in files
        assert "srt" in files
        assert "json" in files

    def test_json_content_matches(self, transcript, tmp_path):
        transcript.save(tmp_path, basename="test")
        data = json.loads((tmp_path / "test.json").read_text())
        assert len(data["segments"]) == 6

    def test_creates_output_dir(self, transcript, tmp_path):
        subdir = tmp_path / "deep" / "nested"
        transcript.save(subdir, basename="test")
        assert (subdir / "test.txt").exists()


# ─── _label_speakers_from_channels() ───────────────────────────────────────

class TestLabelSpeakersFromChannels:
    def test_assigns_you_to_mic_dominant(self, transcript, stereo_wav_with_speakers):
        """The speaker with highest mic energy should be labeled YOU."""
        from millet.transcribe import _label_speakers_from_channels

        # Use raw SPEAKER_XX labels for the input
        raw_segments = [
            Segment(start=s.start, end=s.end, text=s.text, speaker=f"SPEAKER_{i:02d}")
            for i, s in enumerate(transcript.segments)
        ]
        # Map: segments 0,3 are YOU (SPEAKER_00, SPEAKER_03)
        #       segments 1,4 are REMOTE_1 (SPEAKER_01, SPEAKER_04)
        #       segments 2,5 are REMOTE_2 (SPEAKER_02, SPEAKER_05)
        # But diarization would group by speaker, not by segment index.
        # Let's use consistent speaker IDs:
        raw_segments[0].speaker = "SPEAKER_00"  # YOU
        raw_segments[3].speaker = "SPEAKER_00"  # YOU
        raw_segments[1].speaker = "SPEAKER_01"  # REMOTE_1
        raw_segments[4].speaker = "SPEAKER_01"  # REMOTE_1
        raw_segments[2].speaker = "SPEAKER_02"  # REMOTE_2
        raw_segments[5].speaker = "SPEAKER_02"  # REMOTE_2

        raw_speakers = [
            Speaker(id="SPEAKER_00"), Speaker(id="SPEAKER_01"), Speaker(id="SPEAKER_02"),
        ]

        new_segs, new_spks = _label_speakers_from_channels(
            stereo_wav_with_speakers, raw_segments, raw_speakers,
        )

        # Find which raw speaker became YOU
        you_segs = [s for s in new_segs if s.speaker == "YOU"]
        assert len(you_segs) == 2
        # The YOU segments should correspond to our mic-loud segments (0.0-5.5 and 20.3-28.0)
        you_starts = sorted(s.start for s in you_segs)
        assert you_starts[0] == 0.0
        assert you_starts[1] == 20.3

    def test_remote_speakers_labeled(self, transcript, stereo_wav_with_speakers):
        """Non-YOU speakers should get REMOTE labels."""
        from millet.transcribe import _label_speakers_from_channels

        raw_segments = []
        speaker_map = {0: "SPEAKER_00", 1: "SPEAKER_01", 2: "SPEAKER_02",
                       3: "SPEAKER_00", 4: "SPEAKER_01", 5: "SPEAKER_02"}
        for i, s in enumerate(transcript.segments):
            raw_segments.append(
                Segment(start=s.start, end=s.end, text=s.text, speaker=speaker_map[i])
            )

        raw_speakers = [
            Speaker(id="SPEAKER_00"), Speaker(id="SPEAKER_01"), Speaker(id="SPEAKER_02"),
        ]

        new_segs, new_spks = _label_speakers_from_channels(
            stereo_wav_with_speakers, raw_segments, raw_speakers,
        )

        remote_labels = {s.speaker for s in new_segs if s.speaker != "YOU"}
        # With 2 remote speakers, labels should be REMOTE_1 and REMOTE_2
        assert "REMOTE_1" in remote_labels or "REMOTE_2" in remote_labels

    def test_empty_speakers(self, stereo_wav_with_speakers):
        """Empty speaker list should return inputs unchanged."""
        from millet.transcribe import _label_speakers_from_channels

        segs, spks = _label_speakers_from_channels(
            stereo_wav_with_speakers, [], [],
        )
        assert segs == []
        assert spks == []

    def test_no_mic_dominant_speaker_all_remote(self, tmp_path):
        """When no speaker has mic_ratio > 0.5, all should be labeled REMOTE."""
        import wave as wave_mod

        import numpy as np

        from millet.transcribe import _label_speakers_from_channels

        # Create a stereo WAV where system channel is always louder
        sr = 16000
        duration = 10.0
        n_frames = int(duration * sr)
        t = np.linspace(0, duration, n_frames, dtype=np.float32)

        # Mic: very quiet, System: loud
        mic = (500 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        system = (20000 * np.sin(2 * np.pi * 880 * t)).astype(np.int16)
        stereo = np.column_stack((mic, system)).flatten()

        wav_path = tmp_path / "system-only.wav"
        with wave_mod.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo.tobytes())

        segments = [
            Segment(start=0.0, end=5.0, text="Hello", speaker="SPEAKER_00"),
            Segment(start=5.0, end=10.0, text="World", speaker="SPEAKER_01"),
        ]
        speakers = [Speaker(id="SPEAKER_00"), Speaker(id="SPEAKER_01")]

        new_segs, new_spks = _label_speakers_from_channels(
            wav_path, segments, speakers,
        )

        # No speaker should be labeled YOU
        labels = {s.speaker for s in new_segs}
        assert "YOU" not in labels
        # All should be REMOTE variants
        assert all("REMOTE" in label for label in labels)

    def test_sensitive_condenser_mic_assigns_you_via_margin(self, tmp_path):
        """Sensitive condenser mics (e.g. RODE NT-USB) pick up enough room
        audio that the local speaker's mic_ratio sits below 0.5, even though
        they are clearly the most mic-dominant.  The margin check (top
        candidate >0.1 above the average of others, with absolute >0.15)
        should still assign YOU in this case."""
        import wave as wave_mod

        import numpy as np

        from millet.transcribe import _label_speakers_from_channels

        sr = 16000
        # 9s of audio = three 3s segments back-to-back.
        n_frames_each = int(3.0 * sr)
        t_each = np.linspace(0, 3.0, n_frames_each, dtype=np.float32)

        # Speaker 0 (the local user, talks 0-3s).  The mic pickup is only
        # somewhat louder than the system channel because the condenser mic
        # picks up the speakers' own bleed.  ratio ~0.4.
        mic_seg0 = (8000 * np.sin(2 * np.pi * 440 * t_each)).astype(np.int16)
        sys_seg0 = (5500 * np.sin(2 * np.pi * 880 * t_each)).astype(np.int16)

        # Speaker 1 (remote, talks 3-6s).  System channel dominant.
        mic_seg1 = (500 * np.sin(2 * np.pi * 220 * t_each)).astype(np.int16)
        sys_seg1 = (20000 * np.sin(2 * np.pi * 1100 * t_each)).astype(np.int16)

        # Speaker 2 (remote, talks 6-9s).  System channel dominant.
        mic_seg2 = (500 * np.sin(2 * np.pi * 330 * t_each)).astype(np.int16)
        sys_seg2 = (20000 * np.sin(2 * np.pi * 1320 * t_each)).astype(np.int16)

        mic = np.concatenate([mic_seg0, mic_seg1, mic_seg2])
        system = np.concatenate([sys_seg0, sys_seg1, sys_seg2])
        stereo = np.column_stack((mic, system)).flatten()

        wav_path = tmp_path / "condenser.wav"
        with wave_mod.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo.tobytes())

        segments = [
            Segment(start=0.0, end=3.0, text="local user", speaker="SPEAKER_00"),
            Segment(start=3.0, end=6.0, text="remote a",   speaker="SPEAKER_01"),
            Segment(start=6.0, end=9.0, text="remote b",   speaker="SPEAKER_02"),
        ]
        speakers = [
            Speaker(id="SPEAKER_00"),
            Speaker(id="SPEAKER_01"),
            Speaker(id="SPEAKER_02"),
        ]

        new_segs, _ = _label_speakers_from_channels(
            wav_path, segments, speakers,
        )

        labels = {s.speaker for s in new_segs}
        # SPEAKER_00 should be labeled YOU even though its absolute ratio
        # is below 0.5 — the margin over the average of the other two
        # speakers' ratios is large enough.
        assert "YOU" in labels, (
            f"expected YOU label via margin check, got labels={labels}"
        )
    def test_mac_sidecar_g4_default_assigns_you_via_margin(self, tmp_path):
        """Mac sidecar (meetscribe-record on macOS, M4.5 g4 production
        default) produces stereo where the local user's per-speaker
        ``mic_ratio`` lands around 0.24 — well below the absolute 0.5
        gate but comfortably above the 0.15 floor and with a wide
        margin over remote speakers (whose ratio sits near 0.03).

        This codifies the M4.5 ``MicCapture.defaultGain = 4.0`` decision
        as a labeler contract: future tuning of either side (sidecar
        gain, labeler thresholds) must keep the Mac path working.

        Calibration source: peggy's M6c.ii.b sign-off run
        (2026-05-14, meetscribe-record 0.2.0a1, Apple M1 / macOS
        26.4.1) reported ``you_ratio = 0.242``.  The synthetic
        amplitudes here reproduce that ratio: a sine-of-amplitude
        ``A`` has RMS ``A/sqrt(2)``, and ``mic_ratio`` reduces to
        ``amp_mic / (amp_mic + amp_sys)`` over a single speaker's
        segments, so amp_mic ≈ 0.316 × amp_sys yields ratio ≈ 0.24.

        Three speakers (one YOU, two REMOTE) so the relative-margin
        branch of the gate (``margin > 0.1 AND you_ratio > 0.15``) is
        the path under test, not the absolute ``> 0.5`` shortcut.

        See: meetscribe-record commit 7b4a6fd (M4.5), epic
        pretyflaco/meetscribe-record#1, M7 sign-off."""
        import wave as wave_mod

        import numpy as np

        from millet.transcribe import _label_speakers_from_channels

        sr = 16000

        def _seg(secs, freq_mic, amp_mic, freq_sys, amp_sys):
            n = int(secs * sr)
            t = np.linspace(0.0, secs, n, dtype=np.float32, endpoint=False)
            mic = (amp_mic * np.sin(2 * np.pi * freq_mic * t)).astype(np.int16)
            sysc = (amp_sys * np.sin(2 * np.pi * freq_sys * t)).astype(np.int16)
            return mic, sysc

        # YOU window 1: 0..15s.  Mic / system amplitude ratio 3800/12000
        # ≈ 0.317 ⇒ mic_ratio ≈ 0.241 (verified by local prototype).
        m0a, s0a = _seg(15.0, 220.0, 3800, 660.0, 12000)
        # Silence 15..20s (warmup-like padding between speakers).
        sil1_m = np.zeros(int(5.0 * sr), dtype=np.int16)
        sil1_s = np.zeros(int(5.0 * sr), dtype=np.int16)
        # REMOTE_A window: 20..35s.  Mic ≈ ambient bleed (amp 600),
        # system loud (amp 18000) ⇒ mic_ratio ≈ 0.032.
        m1, s1 = _seg(15.0, 110.0, 600, 880.0, 18000)
        # Silence 35..55s.
        sil2_m = np.zeros(int(20.0 * sr), dtype=np.int16)
        sil2_s = np.zeros(int(20.0 * sr), dtype=np.int16)
        # YOU window 2: 55..70s.
        m0b, s0b = _seg(15.0, 220.0, 3800, 660.0, 12000)
        # Silence 70..75s.
        sil3_m = np.zeros(int(5.0 * sr), dtype=np.int16)
        sil3_s = np.zeros(int(5.0 * sr), dtype=np.int16)
        # REMOTE_B window: 75..90s.
        m2, s2 = _seg(15.0, 165.0, 600, 990.0, 18000)

        mic = np.concatenate([m0a, sil1_m, m1, sil2_m, m0b, sil3_m, m2])
        system = np.concatenate([s0a, sil1_s, s1, sil2_s, s0b, sil3_s, s2])
        stereo = np.column_stack((mic, system)).flatten()

        wav_path = tmp_path / "mac-sidecar-g4.wav"
        with wave_mod.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo.tobytes())

        segments = [
            Segment(start=0.0,  end=15.0, text="me talking 1",     speaker="SPEAKER_00"),
            Segment(start=20.0, end=35.0, text="remote a talking", speaker="SPEAKER_01"),
            Segment(start=55.0, end=70.0, text="me talking 2",     speaker="SPEAKER_00"),
            Segment(start=75.0, end=90.0, text="remote b talking", speaker="SPEAKER_02"),
        ]
        speakers = [
            Speaker(id="SPEAKER_00"),
            Speaker(id="SPEAKER_01"),
            Speaker(id="SPEAKER_02"),
        ]

        new_segs, new_spks = _label_speakers_from_channels(
            wav_path, segments, speakers,
        )

        # SPEAKER_00 must become YOU via the margin branch.
        you_segs = [s for s in new_segs if s.speaker == "YOU"]
        assert len(you_segs) == 2, (
            f"expected exactly 2 YOU segments (the local user's two "
            f"windows), got {[(s.speaker, s.start, s.end) for s in new_segs]}"
        )
        you_starts = sorted(s.start for s in you_segs)
        assert you_starts == [0.0, 55.0]

        # The other two raw speakers must become REMOTE_1 / REMOTE_2.
        remote_labels = sorted(s.speaker for s in new_segs if s.speaker != "YOU")
        assert remote_labels == ["REMOTE_1", "REMOTE_2"], (
            f"expected REMOTE_1 + REMOTE_2 for the two remote speakers, "
            f"got {remote_labels}"
        )

        # Speaker objects must be relabeled too.
        new_ids = sorted(s.id for s in new_spks)
        assert new_ids == ["REMOTE_1", "REMOTE_2", "YOU"]


# ─── TranscriptionConfig validation ──────────────────────────────────────

class TestTranscriptionConfig:
    def test_default_mixdown_is_dual_diarize(self):
        config = TranscriptionConfig()
        assert config.mixdown == "dual-diarize"

    def test_valid_mixdown_dual(self):
        config = TranscriptionConfig(mixdown="dual")
        assert config.mixdown == "dual"

    def test_torch_device_defaults_to_device(self):
        config = TranscriptionConfig(device="cpu")
        assert config.torch_device == "cpu"

    def test_torch_device_can_split_from_asr_device(self, monkeypatch):
        # Pretend MPS is available (otherwise validation rejects mps on
        # non-Mac CI).
        monkeypatch.setattr(
            "millet.transcribe._torch_device_available",
            lambda d: True,
        )
        config = TranscriptionConfig(device="cpu", torch_device="mps")
        assert config.device == "cpu"
        assert config.torch_device == "mps"

    def test_invalid_torch_device_cuda_raises(self, monkeypatch):
        # PR #19 changed this from raising to auto-falling-back.  Renamed test
        # kept here as a thin alias so failure trail still searches the old
        # name; full coverage lives in test_cuda_unavailable_falls_back_*.
        def fake_avail(d):
            return False if d == "cuda" else True
        monkeypatch.setattr("millet.transcribe._torch_device_available", fake_avail)
        config = TranscriptionConfig(device="cuda", torch_device="cuda")
        # Both 'device' and 'torch_device' fall back to cpu when cuda is
        # unavailable; compute_type downgrades because device flipped.
        assert config.device == "cpu"
        assert config.torch_device == "cpu"
        assert config.compute_type == "int8"
        assert config._device_auto_fallback is True

    def test_invalid_torch_device_mps_raises(self, monkeypatch):
        # PR #19: mps unavailability falls back to cpu instead of raising.
        # Only torch_device is affected; device/compute_type are untouched
        # (compute_type only flips when *device* falls back).
        def fake_avail(d):
            return False if d == "mps" else True
        monkeypatch.setattr("millet.transcribe._torch_device_available", fake_avail)
        config = TranscriptionConfig(
            device="cpu", torch_device="mps", compute_type="float16"
        )
        assert config.device == "cpu"
        assert config.torch_device == "cpu"
        # device was already cpu (not auto-flipped), so compute_type stays.
        assert config.compute_type == "float16"
        assert config._device_auto_fallback is False

    def test_cuda_unavailable_logs_both_warnings(self, monkeypatch, caplog):
        def fake_avail(d):
            return False if d == "cuda" else True
        monkeypatch.setattr("millet.transcribe._torch_device_available", fake_avail)
        with caplog.at_level(logging.WARNING, logger="millet.transcribe"):
            TranscriptionConfig(device="cuda", torch_device="cuda",
                                compute_type="float16")
        messages = [r.getMessage() for r in caplog.records]
        # Device fallback warning (formatted via %-args)
        assert any("device='cuda'" in m and "falling back to 'cpu'" in m
                   for m in messages), messages
        # compute_type downgrade warning
        assert any("compute_type='float16'" in m and "int8" in m
                   for m in messages), messages

    def test_cuda_unavailable_with_int8_does_not_log_compute_type_change(
        self, monkeypatch, caplog
    ):
        def fake_avail(d):
            return False if d == "cuda" else True
        monkeypatch.setattr("millet.transcribe._torch_device_available", fake_avail)
        with caplog.at_level(logging.WARNING, logger="millet.transcribe"):
            config = TranscriptionConfig(device="cuda", torch_device="cuda",
                                         compute_type="int8")
        assert config.compute_type == "int8"
        messages = [r.getMessage() for r in caplog.records]
        assert not any("compute_type" in m for m in messages), messages

    def test_explicit_cpu_is_not_marked_as_auto_fallback(self, monkeypatch):
        # User passing --device cpu on a no-GPU machine must NOT be flagged
        # as a fallback (guards _load_whisperx_asr_model's "(forced)" vs
        # "(fallback — no GPU)" annotation).
        monkeypatch.setattr(
            "millet.transcribe._torch_device_available", lambda d: True
        )
        config = TranscriptionConfig(device="cpu", torch_device="cpu",
                                     compute_type="int8")
        assert config._device_auto_fallback is False

    def test_validation_skipped_when_torch_missing(self, monkeypatch):
        # When torch is not installed, the helper returns None; validation
        # must not raise.  This preserves the invariant that the package is
        # importable / configurable without torch.
        monkeypatch.setattr("millet.transcribe._torch_device_available", lambda d: None)
        # cuda would normally fail validation — but with torch missing, this
        # should construct silently.
        config = TranscriptionConfig(device="cuda", torch_device="cuda")
        assert config.device == "cuda"
        assert config.torch_device == "cuda"

    def test_device_defaults_to_cuda_on_linux(self, monkeypatch):
        monkeypatch.setattr("millet.transcribe._apple_silicon", lambda: False)
        # Stub validation (#7) so 'cuda' passes regardless of host GPU.
        monkeypatch.setattr(
            "millet.transcribe._torch_device_available", lambda d: True
        )
        config = TranscriptionConfig()
        assert config.device == "cuda"
        assert config.torch_device == "cuda"

    def test_device_defaults_to_cpu_with_mps_on_apple_silicon(self, monkeypatch):
        monkeypatch.setattr("millet.transcribe._apple_silicon", lambda: True)
        monkeypatch.setattr("millet.transcribe._mps_available", lambda: True)
        # Disable MLX auto-selection to keep this test focused on device defaults.
        monkeypatch.setattr("millet.transcribe._mlx_available", lambda: False)
        # _mps_available is the platform-default helper; the device-validation
        # helper (_torch_device_available) is independent.  Stub both so the
        # config picks 'mps' as the default AND passes validation under #7.
        monkeypatch.setattr(
            "millet.transcribe._torch_device_available", lambda d: True
        )
        config = TranscriptionConfig()
        assert config.device == "cpu"
        assert config.torch_device == "mps"

    def test_apple_silicon_without_mps_falls_back_to_cpu_torch(self, monkeypatch):
        monkeypatch.setattr("millet.transcribe._apple_silicon", lambda: True)
        monkeypatch.setattr("millet.transcribe._mps_available", lambda: False)
        monkeypatch.setattr("millet.transcribe._mlx_available", lambda: False)
        config = TranscriptionConfig()
        assert config.device == "cpu"
        assert config.torch_device == "cpu"

    def test_explicit_device_overrides_apple_silicon_default(self, monkeypatch):
        monkeypatch.setattr("millet.transcribe._apple_silicon", lambda: True)
        monkeypatch.setattr("millet.transcribe._mps_available", lambda: True)
        monkeypatch.setattr("millet.transcribe._mlx_available", lambda: False)
        config = TranscriptionConfig(device="cpu", torch_device="cpu")
        assert config.device == "cpu"
        assert config.torch_device == "cpu"


    def test_asr_backend_auto_uses_whisperx_without_mlx(self, monkeypatch):
        monkeypatch.setattr("millet.transcribe._apple_silicon", lambda: True)
        monkeypatch.setattr("millet.transcribe._mlx_available", lambda: False)

        config = TranscriptionConfig(asr_backend="auto")

        assert config.asr_backend == "whisperx"

    def test_asr_backend_auto_uses_mlx_on_apple_silicon(self, monkeypatch):
        monkeypatch.setattr("millet.transcribe._apple_silicon", lambda: True)
        monkeypatch.setattr("millet.transcribe._mlx_available", lambda: True)

        config = TranscriptionConfig(asr_backend="auto", model="large-v3-turbo")

        assert config.asr_backend == "mlx"
        assert config.mlx_model == "mlx-community/whisper-large-v3-turbo"

    def test_invalid_asr_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid ASR backend"):
            TranscriptionConfig(asr_backend="bogus")

    def test_invalid_mixdown_raises(self):
        with pytest.raises(ValueError, match="Invalid mixdown mode"):
            TranscriptionConfig(mixdown="stereo")


class TestMlxAsrBackend:
    def test_transcribe_asr_normalizes_mlx_result(self, monkeypatch):
        class FakeMlxWhisper:
            @staticmethod
            def transcribe(audio, **kwargs):
                return {
                    "text": " hello",
                    "language": "en",
                    "segments": [
                        {"start": 0, "end": 1.25, "text": " hello"},
                    ],
                }

        monkeypatch.setitem(sys.modules, "mlx_whisper", FakeMlxWhisper)
        config = TranscriptionConfig(
            asr_backend="mlx",
            model="large-v3-turbo",
            mlx_model="test/model",
        )

        result = _transcribe_asr("audio.wav", config, "en")

        assert result == {
            "segments": [{"start": 0.0, "end": 1.25, "text": " hello"}],
            "language": "en",
            "text": " hello",
        }

    def test_transcribe_asr_passes_mlx_array_input_through(self, monkeypatch):
        captured = {}

        class FakeMlxWhisper:
            @staticmethod
            def transcribe(audio, **kwargs):
                captured["audio"] = audio
                return {
                    "text": " hello",
                    "language": "en",
                    "segments": [
                        {"start": 0, "end": 1.25, "text": " hello"},
                    ],
                }

        monkeypatch.setitem(sys.modules, "mlx_whisper", FakeMlxWhisper)
        config = TranscriptionConfig(
            asr_backend="mlx",
            model="large-v3-turbo",
            mlx_model="test/model",
        )
        audio = np.zeros(16000, dtype=np.float32)

        _transcribe_asr(audio, config, "en")

        assert captured["audio"] is audio

    def test_transcribe_asr_notes_mlx_vad_inert_with_default_values(
        self, monkeypatch, caplog
    ):
        """The MLX VAD note must fire even when the user passes the defaults,
        since the values are still inert under MLX."""
        # Reset the module-level once-per-process flag so the note fires.
        monkeypatch.setattr("millet.transcribe._mlx_vad_note_logged", False)

        class FakeMlxWhisper:
            @staticmethod
            def transcribe(audio, **kwargs):
                return {
                    "text": " hello",
                    "language": "en",
                    "segments": [
                        {"start": 0, "end": 1.25, "text": " hello"},
                    ],
                }

        monkeypatch.setitem(sys.modules, "mlx_whisper", FakeMlxWhisper)
        # Construct with explicit defaults — pre-#6 behavior would NOT warn.
        config = TranscriptionConfig(
            asr_backend="mlx",
            model="large-v3-turbo",
            mlx_model="test/model",
            vad_onset=TranscriptionConfig.vad_onset,
            vad_offset=TranscriptionConfig.vad_offset,
        )

        with caplog.at_level(logging.INFO, logger="millet.transcribe"):
            _transcribe_asr(np.zeros(16000, dtype=np.float32), config, "en")

        assert "MLX backend ignores VAD options" in caplog.text

    def test_transcribe_asr_mlx_vad_note_logged_once_per_process(
        self, monkeypatch, caplog
    ):
        """Two MLX calls in the same process should produce only one VAD note."""
        monkeypatch.setattr("millet.transcribe._mlx_vad_note_logged", False)

        class FakeMlxWhisper:
            @staticmethod
            def transcribe(audio, **kwargs):
                return {
                    "text": " hello",
                    "language": "en",
                    "segments": [
                        {"start": 0, "end": 1.25, "text": " hello"},
                    ],
                }

        monkeypatch.setitem(sys.modules, "mlx_whisper", FakeMlxWhisper)
        config = TranscriptionConfig(
            asr_backend="mlx",
            model="large-v3-turbo",
            mlx_model="test/model",
        )

        with caplog.at_level(logging.INFO, logger="millet.transcribe"):
            _transcribe_asr(np.zeros(16000, dtype=np.float32), config, "en")
            _transcribe_asr(np.zeros(16000, dtype=np.float32), config, "en")

        note_count = caplog.text.count("MLX backend ignores VAD options")
        assert note_count == 1, (
            f"Expected exactly one VAD-inert note, saw {note_count}"
        )


class TestOfflineModelMissing:
    def test_hf_offline_detects_env(self, monkeypatch):
        from millet.transcribe import _hf_offline

        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        assert _hf_offline() is False

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        assert _hf_offline() is True

        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "true")
        assert _hf_offline() is True

    def test_offline_uncached_raises_offline_model_missing(self, monkeypatch):
        from millet.transcribe import OfflineModelMissing

        called = {"transcribe": False}

        class FakeMlxWhisper:
            @staticmethod
            def transcribe(audio, **kwargs):
                called["transcribe"] = True
                return {"text": "", "language": "en", "segments": []}

        monkeypatch.setitem(sys.modules, "mlx_whisper", FakeMlxWhisper)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setattr("millet.transcribe._mlx_model_cached", lambda repo: False)

        config = TranscriptionConfig(
            asr_backend="mlx",
            model="large-v3-turbo",
            mlx_model="mlx-community/whisper-large-v3-turbo",
        )

        with pytest.raises(OfflineModelMissing) as exc_info:
            _transcribe_asr("audio.wav", config, "en")

        assert exc_info.value.model == "mlx-community/whisper-large-v3-turbo"
        assert exc_info.value.backend == "mlx"
        # The pre-flight must short-circuit before invoking mlx_whisper.
        assert called["transcribe"] is False

    def test_offline_cached_proceeds_to_transcribe(self, monkeypatch):
        class FakeMlxWhisper:
            @staticmethod
            def transcribe(audio, **kwargs):
                return {
                    "text": " hi",
                    "language": "en",
                    "segments": [{"start": 0, "end": 1.0, "text": " hi"}],
                }

        monkeypatch.setitem(sys.modules, "mlx_whisper", FakeMlxWhisper)
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setattr("millet.transcribe._mlx_model_cached", lambda repo: True)

        config = TranscriptionConfig(
            asr_backend="mlx",
            model="large-v3-turbo",
            mlx_model="test/model",
        )

        result = _transcribe_asr("audio.wav", config, "en")
        assert result["text"] == " hi"

    def test_online_skips_cache_preflight(self, monkeypatch):
        class FakeMlxWhisper:
            @staticmethod
            def transcribe(audio, **kwargs):
                return {"text": " ok", "language": "en", "segments": []}

        monkeypatch.setitem(sys.modules, "mlx_whisper", FakeMlxWhisper)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        # Even if the model is not cached, online mode must not pre-empt: it
        # should proceed and let mlx_whisper download as usual.
        def _fail_if_called(repo):
            raise AssertionError("cache preflight ran while online")

        monkeypatch.setattr("millet.transcribe._mlx_model_cached", _fail_if_called)

        config = TranscriptionConfig(
            asr_backend="mlx",
            model="large-v3-turbo",
            mlx_model="test/model",
        )

        result = _transcribe_asr("audio.wav", config, "en")
        assert result["text"] == " ok"


class TestWhisperXAsrBackend:
    def test_dual_channel_reuses_whisperx_model(self, monkeypatch, tmp_path):
        mic_path = tmp_path / "mic.wav"
        sys_path = tmp_path / "sys.wav"
        mic_path.write_bytes(b"")
        sys_path.write_bytes(b"")
        model_loads = []
        audio_loads = []

        class FakeModel:
            def __init__(self):
                self.calls = 0
                self.languages = []

            def transcribe(self, audio, batch_size, language=None):
                self.calls += 1
                self.languages.append(language)
                return {
                    "language": "en",
                    "segments": [
                        {
                            "start": float(self.calls - 1),
                            "end": float(self.calls),
                            "text": f"channel {self.calls}",
                        }
                    ],
                }

        fake_model = FakeModel()

        def fake_load_model(*args, **kwargs):
            model_loads.append((args, kwargs))
            return fake_model

        def fake_load_audio(path):
            audio_loads.append(path)
            return np.zeros(16000, dtype=np.float32)

        def fake_extract_mono(audio_file, channel):
            return mic_path if channel == 0 else sys_path

        monkeypatch.setitem(
            sys.modules,
            "whisperx",
            SimpleNamespace(load_model=fake_load_model, load_audio=fake_load_audio),
        )
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
        monkeypatch.setattr("millet.transcribe._extract_mono", fake_extract_mono)
        # The per-channel language resolution runs multi-window detection on the
        # whisperx backend; stub it so the test stays focused on model reuse.
        monkeypatch.setattr(
            "millet.transcribe._detect_language_multiwindow",
            lambda model, audio, config: ("en", 0.99),
        )

        config = TranscriptionConfig(
            asr_backend="whisperx",
            device="cpu",
            compute_type="int8",
            skip_alignment=True,
            audio_pad_seconds=0,
        )

        transcript = _transcribe_dual_channel(tmp_path / "stereo.wav", config, 10.0)

        assert len(model_loads) == 1
        assert fake_model.calls == 2
        assert len(audio_loads) == 2
        # Both channels decoded with the resolved language (en).
        assert fake_model.languages == ["en", "en"]
        assert [segment.speaker for segment in transcript.segments] == [
            "YOU",
            "REMOTE",
        ]


# ─── Dual-channel dispatch (mocked — full pipeline requires GPU) ──────────

class TestDualChannelDispatch:
    def test_dual_mixdown_dispatches_to_dual_channel(self, stereo_wav):
        """Stereo audio with mixdown='dual' should call _transcribe_dual_channel."""
        dummy = Transcript(
            segments=[], speakers=[], language="en",
            audio_file=str(stereo_wav), duration=5.0,
        )
        with patch("millet.transcribe._transcribe_dual_channel", return_value=dummy) as mock_dual:
            config = TranscriptionConfig(mixdown="dual")
            result = do_transcribe(str(stereo_wav), config)
            mock_dual.assert_called_once()
            assert result is dummy

    def test_mono_mixdown_does_not_dispatch_to_dual_channel(self, stereo_wav):
        """Stereo audio with mixdown='mono' should NOT call _transcribe_dual_channel."""
        with (
            patch("millet.transcribe._transcribe_dual_channel") as mock_dual,
            patch(
                "millet.transcribe._mixdown_to_mono",
                side_effect=RuntimeError("reached mono path"),
            ) as mock_mono,
        ):
            config = TranscriptionConfig(mixdown="mono")
            with pytest.raises(RuntimeError, match="reached mono path"):
                do_transcribe(str(stereo_wav), config)
            mock_mono.assert_called_once_with(stereo_wav)
            mock_dual.assert_not_called()


def _write_stereo_lr(path, left, right, sr=16000):
    """Write a stereo WAV from explicit left/right int16 arrays."""
    import wave
    n = min(len(left), len(right))
    stereo = np.column_stack(
        (
            np.clip(left[:n], -32768, 32767).astype(np.int16),
            np.clip(right[:n], -32768, 32767).astype(np.int16),
        )
    ).flatten()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())
    return path


class TestSingleSourceStereo:
    """Detect in-room recordings (single source) so the dual-diarize path
    falls back to mono diarization instead of collapsing the mic channel."""

    def _t(self, dur=3.0, sr=16000):
        return np.arange(int(dur * sr), dtype=np.float32) / sr

    def test_silent_system_channel_is_single_source(self, tmp_path):
        from millet.transcribe import _is_single_source_stereo
        t = self._t()
        left = (16000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
        right = np.zeros_like(left)  # system channel silent (in-room)
        wav = _write_stereo_lr(tmp_path / "silent-sys.wav", left, right)
        assert _is_single_source_stereo(wav, TranscriptionConfig()) is True

    def test_duplicate_channels_are_single_source(self, tmp_path):
        from millet.transcribe import _is_single_source_stereo
        t = self._t()
        # Both channels carry the same mixed signal (dual-mono).
        sig = (12000 * np.sin(2 * np.pi * 200 * t)
               + 8000 * np.sin(2 * np.pi * 330 * t)).astype(np.int16)
        wav = _write_stereo_lr(tmp_path / "dual-mono.wav", sig, sig.copy())
        assert _is_single_source_stereo(wav, TranscriptionConfig()) is True

    def test_decorrelated_active_channels_not_single_source(self, tmp_path):
        from millet.transcribe import _is_single_source_stereo
        t = self._t()
        # Genuine remote call: both channels active, different content.
        left = (15000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
        right = (15000 * np.sin(2 * np.pi * 510 * t)).astype(np.int16)
        wav = _write_stereo_lr(tmp_path / "real-call.wav", left, right)
        assert _is_single_source_stereo(wav, TranscriptionConfig()) is False

    def test_dispatch_falls_back_to_mono_for_single_source(self, tmp_path):
        """dual-diarize + single-source stereo must NOT call the dual-diarize
        path; it falls through to the mono pipeline (which reaches
        _mixdown_to_mono)."""
        t = self._t()
        sig = (12000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
        wav = _write_stereo_lr(tmp_path / "inroom.wav", sig, sig.copy())
        # Stop the pipeline right at the mono entry point so we don't depend on
        # real ASR/diarization: a sentinel exception proves the mono path was
        # taken AND the dual-diarize path was not.
        sentinel = RuntimeError("reached mono path")
        with patch(
            "millet.transcribe._transcribe_dual_diarize"
        ) as mock_dd, patch(
            "millet.transcribe._mixdown_to_mono", side_effect=sentinel
        ) as mock_mono:
            config = TranscriptionConfig(mixdown="dual-diarize")
            with pytest.raises(RuntimeError, match="reached mono path"):
                do_transcribe(str(wav), config)
            mock_dd.assert_not_called()
            mock_mono.assert_called_once()

    def test_dispatch_uses_dual_diarize_for_real_call(self, tmp_path):
        """dual-diarize + decorrelated active channels must dispatch to the
        dual-diarize path (no regression for genuine remote calls)."""
        t = self._t()
        left = (15000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
        right = (15000 * np.sin(2 * np.pi * 510 * t)).astype(np.int16)
        wav = _write_stereo_lr(tmp_path / "real.wav", left, right)
        dummy = Transcript(
            segments=[], speakers=[], language="en",
            audio_file=str(wav), duration=3.0,
        )
        with patch(
            "millet.transcribe._transcribe_dual_diarize", return_value=dummy
        ) as mock_dd:
            config = TranscriptionConfig(mixdown="dual-diarize")
            result = do_transcribe(str(wav), config)
            mock_dd.assert_called_once()
            assert result is dummy

    def test_fallback_disabled_keeps_dual_diarize(self, tmp_path):
        """--no-single-source-fallback keeps the old behavior."""
        t = self._t()
        sig = (12000 * np.sin(2 * np.pi * 200 * t)).astype(np.int16)
        wav = _write_stereo_lr(tmp_path / "inroom2.wav", sig, sig.copy())
        dummy = Transcript(
            segments=[], speakers=[], language="en",
            audio_file=str(wav), duration=3.0,
        )
        with patch(
            "millet.transcribe._transcribe_dual_diarize", return_value=dummy
        ) as mock_dd:
            config = TranscriptionConfig(
                mixdown="dual-diarize", single_source_fallback=False,
            )
            result = do_transcribe(str(wav), config)
            mock_dd.assert_called_once()
            assert result is dummy


# ─── Hybrid channel-energy correction ──────────────────────────────────────


def _write_stereo(path, placements, sr=16000):
    """Write a stereo WAV from (start, end, channel) placements.

    channel: 'mic' -> loud left/quiet right; 'sys' -> loud right/quiet left.
    """
    import wave

    duration = max(e for _, e, _ in placements) + 0.5
    n = int(duration * sr)
    mic = np.zeros(n, dtype=np.float32)
    system = np.zeros(n, dtype=np.float32)
    for start_t, end_t, ch in placements:
        s = int(start_t * sr)
        e = min(int(end_t * sr), n)
        t = np.arange(e - s, dtype=np.float32) / sr
        if ch == "mic":
            mic[s:e] += 18000 * np.sin(2 * np.pi * 440 * t)
            system[s:e] += 800 * np.sin(2 * np.pi * 880 * t)
        else:  # sys
            system[s:e] += 18000 * np.sin(2 * np.pi * 880 * t)
            mic[s:e] += 800 * np.sin(2 * np.pi * 440 * t)
    stereo = np.column_stack(
        (
            np.clip(mic, -32768, 32767).astype(np.int16),
            np.clip(system, -32768, 32767).astype(np.int16),
        )
    ).flatten()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())
    return path


class TestChannelCorrect:
    def test_you_segment_on_system_channel_flips_to_remote(self, tmp_path):
        """A YOU segment whose audio is on the system channel is reassigned."""
        wav = _write_stereo(
            tmp_path / "m.wav",
            [(0.0, 2.0, "mic"), (2.0, 4.0, "sys")],  # 2nd is actually remote
        )
        segs = [
            Segment(0.0, 2.0, "hello", speaker="YOU"),
            Segment(2.0, 4.0, "leaked remote", speaker="YOU"),  # mislabeled
        ]
        speakers = [Speaker(id="YOU"), Speaker(id="REMOTE")]
        out, _ = _channel_correct_segments(wav, segs, speakers, margin=0.30)
        by_text = {s.text: s.speaker for s in out}
        assert by_text["hello"] == "YOU"
        assert by_text["leaked remote"] == "REMOTE"

    def test_multi_remote_identities_preserved(self, tmp_path):
        """Same-side remote words keep their specific REMOTE_N label."""
        wav = _write_stereo(
            tmp_path / "m.wav",
            [(0.0, 2.0, "mic"), (2.0, 4.0, "sys"), (4.0, 6.0, "sys")],
        )
        segs = [
            Segment(0.0, 2.0, "you talk", speaker="YOU"),
            Segment(2.0, 4.0, "remote one", speaker="REMOTE_1"),
            Segment(4.0, 6.0, "remote two", speaker="REMOTE_2"),
        ]
        speakers = [Speaker(id="YOU"), Speaker(id="REMOTE_1"), Speaker(id="REMOTE_2")]
        out, out_speakers = _channel_correct_segments(wav, segs, speakers, margin=0.30)
        by_text = {s.text: s.speaker for s in out}
        # The two distinct remotes must NOT be collapsed.
        assert by_text["remote one"] == "REMOTE_1"
        assert by_text["remote two"] == "REMOTE_2"
        assert by_text["you talk"] == "YOU"

    def test_word_level_split_of_mixed_segment(self, tmp_path):
        """A segment whose words span both channels is split per speaker."""
        wav = _write_stereo(
            tmp_path / "m.wav",
            [(0.0, 1.0, "mic"), (1.0, 2.0, "sys")],
        )
        seg = Segment(
            0.0, 2.0, "local part remote part", speaker="YOU",
            words=[
                {"word": "local", "start": 0.0, "end": 0.4},
                {"word": "part", "start": 0.4, "end": 1.0},
                {"word": "remote", "start": 1.0, "end": 1.5},
                {"word": "part", "start": 1.5, "end": 2.0},
            ],
        )
        out, _ = _channel_correct_segments(
            wav, [seg], [Speaker(id="YOU"), Speaker(id="REMOTE")], margin=0.30
        )
        # Should split into a YOU run and a REMOTE run.
        speakers_seen = [s.speaker for s in out]
        assert "YOU" in speakers_seen
        assert "REMOTE" in speakers_seen
        you_seg = next(s for s in out if s.speaker == "YOU")
        rem_seg = next(s for s in out if s.speaker == "REMOTE")
        assert "local" in you_seg.text
        assert "remote" in rem_seg.text

    def test_ambiguous_segment_not_flipped(self, tmp_path):
        """A balanced (mic≈sys) segment stays on its diarized label."""
        # Place equal energy on both channels for the segment.
        import wave

        sr = 16000
        n = int(2.5 * sr)
        t = np.arange(n, dtype=np.float32) / sr
        both = (10000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        stereo = np.column_stack((both, both)).flatten()
        wav = tmp_path / "amb.wav"
        with wave.open(str(wav), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(stereo.tobytes())
        segs = [Segment(0.0, 2.0, "balanced", speaker="YOU")]
        out, _ = _channel_correct_segments(
            wav, segs, [Speaker(id="YOU"), Speaker(id="REMOTE")], margin=0.30
        )
        assert out[0].speaker == "YOU"  # not flipped

    def test_noop_without_you_or_remote(self, tmp_path):
        """Correction is a no-op when there is no YOU+REMOTE pair."""
        wav = _write_stereo(tmp_path / "m.wav", [(0.0, 2.0, "mic")])
        segs = [Segment(0.0, 2.0, "named", speaker="Nora")]
        out, sp = _channel_correct_segments(
            wav, segs, [Speaker(id="Nora")], margin=0.30
        )
        assert out[0].speaker == "Nora"


class TestCorrectMicBleedSegments:
    """No-headphones bleed correction for the dual-diarize mic channel.

    Bled 'YOU' segments are reassigned to the overlapping remote speaker
    (or a generic fallback), not dropped; empties are dropped.
    """

    def test_system_dominant_mic_segment_reassigned_to_overlapping_remote(
        self, tmp_path
    ):
        """A 'YOU' segment on the system channel is reassigned to that remote."""
        wav = _write_stereo(
            tmp_path / "m.wav",
            [(0.0, 2.0, "mic"), (2.0, 4.0, "sys")],  # 2nd is bled remote
        )
        mic_segs = [
            Segment(0.0, 2.0, "genuine local", speaker="YOU"),
            Segment(2.0, 4.0, "bled remote", speaker="YOU"),
        ]
        sys_segs = [Segment(2.0, 4.0, "remote speech", speaker="Sam")]
        out = _correct_mic_bleed_segments(wav, mic_segs, sys_segs, margin=0.30)
        by_text = {s.text: s.speaker for s in out}
        assert by_text["genuine local"] == "YOU"
        assert by_text["bled remote"] == "Sam"  # reassigned, not dropped

    def test_bled_segment_without_overlap_uses_fallback(self, tmp_path):
        """A bled 'YOU' segment with no overlapping remote -> generic fallback."""
        wav = _write_stereo(tmp_path / "m.wav", [(0.0, 2.0, "sys")])
        mic_segs = [Segment(0.0, 2.0, "bled remote", speaker="YOU")]
        out = _correct_mic_bleed_segments(
            wav, mic_segs, sys_segments=[], remote_fallback="REMOTE", margin=0.30
        )
        assert [(s.text, s.speaker) for s in out] == [("bled remote", "REMOTE")]

    def test_genuine_local_segment_kept(self, tmp_path):
        """A clearly mic-dominant local segment stays YOU."""
        wav = _write_stereo(tmp_path / "m.wav", [(0.0, 2.0, "mic")])
        mic_segs = [Segment(0.0, 2.0, "hello team", speaker="YOU")]
        out = _correct_mic_bleed_segments(wav, mic_segs, sys_segments=[], margin=0.30)
        assert [(s.text, s.speaker) for s in out] == [("hello team", "YOU")]

    def test_echo_duplicate_of_remote_reassigned(self, tmp_path):
        """A mic segment echoing an overlapping remote segment is reassigned."""
        # Mic-dominant audio (energy alone would keep it), but the text is a
        # near-duplicate of a remote segment at the same time -> echo.
        wav = _write_stereo(tmp_path / "m.wav", [(0.0, 3.0, "mic")])
        mic_segs = [
            Segment(0.0, 3.0, "the servers now to control", speaker="YOU"),
        ]
        sys_segs = [
            Segment(0.2, 3.2, "the servers now to control", speaker="Sam"),
        ]
        out = _correct_mic_bleed_segments(wav, mic_segs, sys_segs, margin=0.30)
        assert [(s.text, s.speaker) for s in out] == [
            ("the servers now to control", "Sam")
        ]

    def test_empty_segment_dropped(self, tmp_path):
        """Punctuation-only mic artifacts are removed."""
        wav = _write_stereo(tmp_path / "m.wav", [(0.0, 1.0, "mic")])
        mic_segs = [Segment(0.0, 1.0, " . ", speaker="YOU")]
        out = _correct_mic_bleed_segments(wav, mic_segs, sys_segments=[], margin=0.30)
        assert out == []

    def test_no_segments_noop(self, tmp_path):
        wav = _write_stereo(tmp_path / "m.wav", [(0.0, 1.0, "mic")])
        assert _correct_mic_bleed_segments(wav, [], sys_segments=[]) == []


class TestSplitSegmentByWordSpeaker:
    def test_single_speaker_returns_same_segment(self):
        seg = Segment(0.0, 2.0, "a b", speaker="YOU",
                      words=[{"word": "a", "start": 0.0, "end": 1.0},
                             {"word": "b", "start": 1.0, "end": 2.0}])
        out = _split_segment_by_word_speaker(seg, ["YOU", "YOU"])
        assert len(out) == 1
        assert out[0].speaker == "YOU"

    def test_two_runs_split(self):
        seg = Segment(0.0, 2.0, "a b", speaker="YOU",
                      words=[{"word": "a", "start": 0.0, "end": 1.0},
                             {"word": "b", "start": 1.0, "end": 2.0}])
        out = _split_segment_by_word_speaker(seg, ["YOU", "REMOTE"])
        assert len(out) == 2
        assert out[0].speaker == "YOU" and out[0].text == "a"
        assert out[1].speaker == "REMOTE" and out[1].text == "b"

    def test_no_words_returns_same(self):
        seg = Segment(0.0, 2.0, "x", speaker="YOU", words=None)
        out = _split_segment_by_word_speaker(seg, [])
        assert out == [seg]


class TestChannelCorrectConfig:
    def test_default_on(self):
        assert TranscriptionConfig(device="cpu").channel_correct is True

    def test_can_disable(self):
        assert TranscriptionConfig(device="cpu", channel_correct=False).channel_correct is False

    def test_default_margin(self):
        assert TranscriptionConfig(device="cpu").channel_correct_margin == 0.30


# ─── remote-cluster consolidation (dual-diarize over-segmentation fix) ───────


def _seg(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


class TestConsolidateRemoteClusters:
    def _cfg(self, **kw):
        base = dict(
            device="cpu",
            cluster_merge_similarity=0.80,
            cluster_min_speech_seconds=8.0,
        )
        base.update(kw)
        return TranscriptionConfig(**base)

    def test_voiceprint_merge_high_similarity(self):
        """Two clusters with near-identical embeddings merge into the larger."""
        import numpy as np
        segs = [
            _seg(0, 30, "SPEAKER_00"),   # dominant (30s embeddable)
            _seg(40, 50, "SPEAKER_01"),  # 10s — above absorb floor
        ]
        v = np.array([1.0, 0.0, 0.0])

        def embed_fn(_c):
            return v  # identical → cosine 1.0

        remap = _consolidate_remote_clusters(segs, self._cfg(), embed_fn=embed_fn)
        assert remap == {"SPEAKER_01": "SPEAKER_00"}

    def test_no_merge_when_distinct(self):
        """Genuinely-distinct embeddings (low cosine) are NOT merged."""
        import numpy as np
        segs = [
            _seg(0, 30, "SPEAKER_00"),
            _seg(40, 70, "SPEAKER_01"),  # 30s — well above absorb floor
        ]
        embs = {"SPEAKER_00": np.array([1.0, 0.0]), "SPEAKER_01": np.array([0.0, 1.0])}

        def embed_fn(c):
            return embs[c]

        remap = _consolidate_remote_clusters(segs, self._cfg(), embed_fn=embed_fn)
        assert remap == {}

    def test_small_cluster_absorbed_by_duration(self):
        """A thin cluster (< floor embeddable speech) is absorbed without embeddings."""
        segs = [
            _seg(0, 40, "SPEAKER_00"),   # 40s dominant
            _seg(50, 53, "SPEAKER_01"),  # 3s < 8s floor
        ]
        remap = _consolidate_remote_clusters(segs, self._cfg(), embed_fn=None)
        assert remap == {"SPEAKER_01": "SPEAKER_00"}

    def test_large_distinct_cluster_not_absorbed_without_embeddings(self):
        """Without embeddings, a sizable second cluster is kept (no over-merge)."""
        segs = [
            _seg(0, 40, "SPEAKER_00"),
            _seg(50, 80, "SPEAKER_01"),  # 30s > floor
        ]
        remap = _consolidate_remote_clusters(segs, self._cfg(), embed_fn=None)
        assert remap == {}

    def test_single_cluster_noop(self):
        segs = [_seg(0, 20, "SPEAKER_00")]
        assert _consolidate_remote_clusters(segs, self._cfg(), embed_fn=None) == {}


class TestMergeOrphanSystemSegments:
    def _cfg(self, **kw):
        return TranscriptionConfig(device="cpu", orphan_merge_max_seconds=1.0, **kw)

    def test_short_orphan_attached_to_nearest(self):
        segs = [
            _seg(0, 10, "SPEAKER_00"),
            _seg(10.2, 10.6, None),       # 0.4s orphan, nearest = SPEAKER_00
            _seg(50, 60, "SPEAKER_01"),
        ]
        _merge_orphan_system_segments(segs, self._cfg())
        assert segs[1]["speaker"] == "SPEAKER_00"

    def test_long_orphan_left_alone(self):
        segs = [
            _seg(0, 10, "SPEAKER_00"),
            _seg(20, 25, None),           # 5s > 1.0s → not merged
        ]
        _merge_orphan_system_segments(segs, self._cfg())
        assert segs[1]["speaker"] is None

    def test_orphan_picks_temporally_nearest(self):
        segs = [
            _seg(0, 10, "SPEAKER_00"),
            _seg(55, 55.5, None),         # closer to SPEAKER_01 at 50-60
            _seg(50, 60, "SPEAKER_01"),
        ]
        _merge_orphan_system_segments(segs, self._cfg())
        assert segs[1]["speaker"] == "SPEAKER_01"

    def test_no_assigned_clusters_noop(self):
        segs = [_seg(0, 0.5, None)]
        _merge_orphan_system_segments(segs, self._cfg())
        assert segs[0]["speaker"] is None


class TestIsolatedBackchannel:
    """B1 (issue #2): strip far-isolated sub-threshold backchannel fragments
    that pyannote mis-clustered into a later speaker."""

    def _cfg(self, **kw):
        return TranscriptionConfig(
            device="cpu",
            backchannel_max_seconds=1.5,
            cluster_isolation_gap_seconds=300.0,
            **kw,
        )

    def test_isolated_short_fragment_flagged(self):
        # Mirrors the Omar case: tiny early fragments, then real speech 25 min on.
        segs = [
            _seg(11.0, 11.5, "SPEAKER_02"),   # 0.5s, isolated
            _seg(53.5, 53.8, "SPEAKER_02"),   # 0.3s, isolated
            _seg(1588.0, 1600.0, "SPEAKER_02"),  # real mass (12s)
            _seg(1605.0, 1620.0, "SPEAKER_02"),
        ]
        flagged = _isolated_backchannel_outliers(segs, self._cfg())
        assert flagged == [0, 1]

    def test_short_fragment_near_own_speech_kept(self):
        # A quick "yeah" right next to the speaker's real speech is NOT stripped.
        segs = [
            _seg(0.0, 12.0, "SPEAKER_00"),    # real mass
            _seg(12.3, 12.7, "SPEAKER_00"),   # 0.4s, but adjacent → keep
        ]
        assert _isolated_backchannel_outliers(segs, self._cfg()) == []

    def test_long_isolated_segment_kept(self):
        # A substantive (>=1.5s) isolated segment carries embedding weight → keep.
        segs = [
            _seg(10.0, 13.0, "SPEAKER_02"),   # 3s isolated — NOT a backchannel
            _seg(1588.0, 1600.0, "SPEAKER_02"),
        ]
        assert _isolated_backchannel_outliers(segs, self._cfg()) == []

    def test_all_backchannel_cluster_kept(self):
        # A cluster with NO real (>=1.5s) segment is left alone (that's the
        # REMOTE-rescue's job, not this guard).
        segs = [
            _seg(10.0, 10.4, "SPEAKER_05"),
            _seg(900.0, 900.3, "SPEAKER_05"),
        ]
        assert _isolated_backchannel_outliers(segs, self._cfg()) == []

    def test_disabled_via_config_is_independent(self):
        # The helper itself doesn't check the flag (caller does); ensure it
        # still computes deterministically regardless.
        segs = [
            _seg(11.0, 11.4, "SPEAKER_02"),
            _seg(1588.0, 1600.0, "SPEAKER_02"),
        ]
        assert _isolated_backchannel_outliers(segs, self._cfg()) == [0]

    def test_config_defaults(self):
        c = TranscriptionConfig(device="cpu")
        assert c.strip_isolated_backchannel is True
        assert c.backchannel_max_seconds == 1.5
        assert c.cluster_isolation_gap_seconds == 300.0


class TestNearestSegment:
    def test_empty_returns_none(self):
        assert _nearest_segment([], 5.0) is None

    def test_picks_nearest_by_gap(self):
        a = {"start": 0, "end": 10, "speaker": "A"}
        b = {"start": 50, "end": 60, "speaker": "B"}
        assert _nearest_segment([a, b], 55.0)["speaker"] == "B"
        assert _nearest_segment([a, b], 5.0)["speaker"] == "A"

    def test_gap_zero_inside_span(self):
        seg = {"start": 10, "end": 20}
        assert _segment_gap(seg, 15.0) == 0.0
        assert _segment_gap(seg, 25.0) == 5.0
        assert _segment_gap(seg, 7.0) == 3.0


class TestConsolidationConfig:
    def test_default_on(self):
        assert TranscriptionConfig(device="cpu").consolidate_remote_clusters is True

    def test_can_disable(self):
        cfg = TranscriptionConfig(device="cpu", consolidate_remote_clusters=False)
        assert cfg.consolidate_remote_clusters is False

    def test_threshold_defaults(self):
        cfg = TranscriptionConfig(device="cpu")
        assert cfg.cluster_merge_similarity == 0.80
        assert cfg.cluster_min_speech_seconds == 8.0
        assert cfg.orphan_merge_max_seconds == 1.0


# ─── dominant-channel language selection (summary-language fix) ──────────────


class TestDominantChannelLanguage:
    def test_system_wins_when_more_speech(self):
        """Mic speaks a minority language; the system channel (more speech)
        dictates the summary language.  (The real pt/en regression.)"""
        mic = [{"start": 0, "end": 702}]   # pt, 702s
        sys = [{"start": 0, "end": 1062}]  # en, 1062s
        assert _dominant_channel_language(mic, sys, "pt", "en") == "en"

    def test_mic_wins_when_more_speech(self):
        mic = [{"start": 0, "end": 900}]
        sys = [{"start": 0, "end": 100}]
        assert _dominant_channel_language(mic, sys, "pt", "en") == "pt"

    def test_same_language_short_circuits(self):
        assert _dominant_channel_language([], [], "en", "en") == "en"

    def test_exact_tie_prefers_mic(self):
        mic = [{"start": 0, "end": 500}]
        sys = [{"start": 0, "end": 500}]
        assert _dominant_channel_language(mic, sys, "tr", "en") == "tr"

    def test_none_languages_default_english(self):
        assert _dominant_channel_language([], [], None, None) == "en"

    def test_segments_total_seconds(self):
        assert _segments_total_seconds(
            [{"start": 0, "end": 5}, {"start": 10, "end": 12.5}]
        ) == 7.5
        assert _segments_total_seconds([]) == 0.0


class TestDefaultLanguageBias:
    def _cfg(self, **kw):
        return TranscriptionConfig(
            device="cpu",
            default_language="en",
            default_language_override_confidence=0.70,
            **kw,
        )

    def test_low_confidence_minority_keeps_default(self):
        # The real DEVSYNC regression: es detected at 0.50 -> keep en.
        assert _apply_default_language_bias("es", 0.50, self._cfg()) == "en"

    def test_high_confidence_minority_overrides_default(self):
        assert _apply_default_language_bias("es", 0.85, self._cfg()) == "es"

    def test_detected_equals_default(self):
        assert _apply_default_language_bias("en", 0.40, self._cfg()) == "en"

    def test_no_default_returns_detected(self):
        cfg = TranscriptionConfig(device="cpu")  # default_language None
        assert _apply_default_language_bias("es", 0.50, cfg) == "es"

    def test_config_defaults(self):
        cfg = TranscriptionConfig(device="cpu")
        assert cfg.default_language is None
        assert cfg.language_detection_segments == 6
        assert cfg.default_language_override_confidence == 0.70


class TestResolveChannelLanguage:
    """The decode language forced into the ASR must honor the default bias.

    This is the fix for English audio being transcribed as Spanish: a
    low-confidence minority detection on a channel must be overridden by the
    team default BEFORE the ASR decode, not just in the metadata.
    """

    def _cfg(self, **kw):
        return TranscriptionConfig(
            device="cpu",
            default_language="en",
            default_language_override_confidence=0.70,
            **kw,
        )

    def test_low_confidence_minority_forces_default_decode(self):
        """es@0.50 on an en-default team -> decode the channel as en."""
        with patch(
            "millet.transcribe._detect_language_multiwindow",
            return_value=("es", 0.50),
        ):
            reported, conf, decode = _resolve_channel_language(
                object(), object(), self._cfg(), None, "system"
            )
        assert reported == "es"      # raw detection preserved for reference
        assert conf == 0.50
        assert decode == "en"        # but the ASR is forced to English

    def test_high_confidence_minority_decodes_in_minority(self):
        """A genuinely Spanish channel detected confidently is decoded as es."""
        with patch(
            "millet.transcribe._detect_language_multiwindow",
            return_value=("es", 0.90),
        ):
            _reported, _conf, decode = _resolve_channel_language(
                object(), object(), self._cfg(), None, "system"
            )
        assert decode == "es"

    def test_explicit_language_forces_everywhere(self):
        """--language en (not auto) forces the decode regardless of audio."""
        cfg = TranscriptionConfig(device="cpu", language="en")
        reported, conf, decode = _resolve_channel_language(
            object(), object(), cfg, "en", "mic"
        )
        assert reported == "en"
        assert decode == "en"

    def test_no_model_uses_default_decode(self):
        """Without a whisperx model to pre-detect, force the team default."""
        reported, conf, decode = _resolve_channel_language(
            None, object(), self._cfg(), None, "audio"
        )
        assert decode == "en"

    def test_detection_unavailable_without_default_auto_detects(self):
        """When detection returns None and no default is set, decode_lang is
        None so the ASR auto-detects instead of being forced to English."""
        cfg = TranscriptionConfig(device="cpu")  # no default_language
        with patch(
            "millet.transcribe._detect_language_multiwindow",
            return_value=(None, 0.0),
        ):
            _reported, _conf, decode = _resolve_channel_language(
                object(), object(), cfg, None, "system"
            )
        assert decode is None

    def test_detection_unavailable_with_default_uses_default(self):
        with patch(
            "millet.transcribe._detect_language_multiwindow",
            return_value=(None, 0.0),
        ):
            _reported, _conf, decode = _resolve_channel_language(
                object(), object(), self._cfg(), None, "system"
            )
        assert decode == "en"


class TestSegChannelRatio:
    def test_none_start_returns_none_no_crash(self):
        """A word without a start timestamp must not raise TypeError."""
        import numpy as np

        from millet.transcribe import _seg_channel_ratio

        mic = np.ones(1000, dtype=np.float32)
        sys_ = np.ones(1000, dtype=np.float32)
        assert _seg_channel_ratio(mic, sys_, None, 1.0, 100, 1000) is None

    def test_normal_window_returns_ratio(self):
        import numpy as np

        from millet.transcribe import _seg_channel_ratio

        mic = np.ones(1000, dtype=np.float32)
        sys_ = np.ones(1000, dtype=np.float32)
        r = _seg_channel_ratio(mic, sys_, 0.0, 5.0, 100, 1000)
        assert r == pytest.approx(0.5)
