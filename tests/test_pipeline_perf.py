"""Tests for single-decode stereo, shared alignment model, and OGG RMS mixdown."""

from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from millet import transcribe as tr_mod
from millet import voiceprint as vp_mod


def test_align_dual_channels_reuses_model_when_languages_match():
    load_calls: list[str] = []
    align_calls: list[int] = []

    def fake_load_align_model(*, language_code, device):
        load_calls.append(language_code)
        return object(), {"lang": language_code}

    def fake_align(segments, model, metadata, audio, device, return_char_alignments=False):
        align_calls.append(len(segments))
        return {"segments": segments}

    fake_wx = SimpleNamespace(
        load_align_model=fake_load_align_model,
        align=fake_align,
    )
    fake_torch = SimpleNamespace()
    cfg = tr_mod.TranscriptionConfig(device="cpu", compute_type="int8")

    # Same language -> 1 align model load, 2 align calls
    mic_res = {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}
    sys_res = {"segments": [{"start": 1.0, "end": 2.0, "text": "hello"}]}
    audio = np.zeros(16000, dtype=np.float32)

    orig_cached = tr_mod.check_alignment_model_cached
    tr_mod.check_alignment_model_cached = lambda _lang: True
    try:
        m_out, s_out = tr_mod._align_dual_channels(
            mic_res, audio, "en", sys_res, audio, "en", cfg, fake_torch, fake_wx
        )
        assert load_calls == ["en"]
        assert align_calls == [1, 1]
        assert m_out["segments"] == mic_res["segments"]
        assert s_out["segments"] == sys_res["segments"]

        # Different languages -> 2 separate loads
        load_calls.clear()
        tr_mod._align_dual_channels(
            mic_res, audio, "en", sys_res, audio, "de", cfg, fake_torch, fake_wx
        )
        assert load_calls == ["en", "de"]
    finally:
        tr_mod.check_alignment_model_cached = orig_cached


def test_mixdown_to_mono_balances_rms_for_non_wav_formats(tmp_path: Path, monkeypatch):
    fake_ogg = tmp_path / "meeting.ogg"
    fake_ogg.write_bytes(b"OggSfake")

    # Quiet mic (RMS ~200) + loud remote (RMS ~4000)
    left = np.full(16000, 200.0, dtype=np.float32)
    right = np.full(16000, 4000.0, dtype=np.float32)
    monkeypatch.setattr(tr_mod, "_load_stereo_int16", lambda _p: (left, right))

    out_wav = tr_mod._mixdown_to_mono(fake_ogg)
    try:
        with wave.open(str(out_wav), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        # Quiet channel scaled up to match 4000 -> average is ~4000 (not 2100)
        assert abs(float(np.mean(samples)) - 4000.0) < 5.0
    finally:
        out_wav.unlink(missing_ok=True)


def test_correct_mic_bleed_uses_in_memory_stereo_chans(tmp_path: Path, monkeypatch):
    # Ensure read_stereo_channels is never called when stereo_chans is provided
    from millet import audio as audio_mod

    monkeypatch.setattr(
        audio_mod,
        "read_stereo_channels",
        lambda _p: (_ for _ in ()).throw(AssertionError("must not re-decode stereo")),
    )

    sr = 16000
    mic = np.zeros(sr * 2, dtype=np.float32)
    sys_ch = np.full(sr * 2, 5000.0, dtype=np.float32)

    mic_segs = [tr_mod.Segment(start=0.0, end=1.5, text="remote bleed", speaker="YOU")]
    sys_segs = [tr_mod.Segment(start=0.0, end=1.5, text="remote bleed", speaker="SPEAKER_00")]

    out = tr_mod._correct_mic_bleed_segments(
        tmp_path / "unused.ogg",
        mic_segs,
        sys_segs,
        stereo_chans=(mic, sys_ch),
    )
    assert len(out) == 1
    assert out[0].speaker == "SPEAKER_00"


def test_voiceprint_extract_channel_audio_populates_both_channels_once(tmp_path: Path, monkeypatch):
    fake_audio = tmp_path / "call.ogg"
    fake_audio.write_bytes(b"OggS")

    decode_calls = 0

    def fake_stereo_decode(_p):
        nonlocal decode_calls
        decode_calls += 1
        return np.full(100, 0.1, dtype=np.float32), np.full(100, 0.2, dtype=np.float32)

    monkeypatch.setattr(vp_mod, "_decode_stereo_channels_16k", fake_stereo_decode)
    vp_mod._CHANNEL_CACHE_KEY = None
    vp_mod._CHANNEL_CACHE = {}

    mic_res = vp_mod._extract_channel_audio(fake_audio, "mic")
    sys_res = vp_mod._extract_channel_audio(fake_audio, "system")

    assert decode_calls == 1
    assert mic_res is not None and abs(float(mic_res[0][0]) - 0.1) < 1e-6
    assert sys_res is not None and abs(float(sys_res[0][0]) - 0.2) < 1e-6


def test_millet_sync_cli_fallback_meeting_type_single_pass(tmp_path: Path, monkeypatch):
    from click.testing import CliRunner

    from millet import sync as sync_mod
    from millet.cli.sync import sync as sync_cmd

    sdir = tmp_path / "01J0000000000000000000001A"
    sdir.mkdir()

    synced: list[str] = []
    monkeypatch.setattr(sync_mod, "is_sync_configured", lambda _team=None: True)
    monkeypatch.setattr(sync_mod, "detect_meeting_type", lambda _p, team=None: None)
    monkeypatch.setattr(
        sync_mod,
        "sync_session",
        lambda _p, match, progress_callback=None, team=None: synced.append(match.folder)
        or [sdir / "summary.md"],
    )

    runner = CliRunner()
    res = runner.invoke(
        sync_cmd, ["--fallback-meeting-type", "design-review-123", str(sdir)]
    )
    assert res.exit_code == 0, res.output
    assert synced == ["design-review-123"]

    # Reject path-traversal slugs
    res_bad = runner.invoke(
        sync_cmd, ["--fallback-meeting-type", "../escape", str(sdir)]
    )
    assert res_bad.exit_code == 1


def test_transcribe_over_budget_and_unknown_duration_skip_full_stereo_decode(
    tmp_path: Path, monkeypatch
):
    """Over-limit and unknown-duration recordings must never call _load_stereo_int16()."""
    import sys
    import types

    fake_audio = tmp_path / "long_call.ogg"
    fake_audio.write_bytes(b"OggS")

    # Stub heavy ML imports so this test runs in minimal CI (no torch/whisperx).
    fake_torch = types.ModuleType("torch")
    fake_wx = types.ModuleType("whisperx")
    fake_diarize = types.ModuleType("whisperx.diarize")
    fake_diarize.DiarizationPipeline = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "whisperx", fake_wx)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", fake_diarize)

    def _forbid_full_decode(_path: Path):
        raise AssertionError("_load_stereo_int16 must not be called over budget / unknown duration")

    stream_calls: list[Path] = []

    def _fake_stream_stats(path: Path, **_kw):
        stream_calls.append(path)
        return (3500.0, 3500.0, 32000, 0.05)

    seen_stereo_chans: list[object] = []

    def _fake_dual_diarize(audio_file, config, duration, *, stereo_chans=None):
        seen_stereo_chans.append(stereo_chans)
        return tr_mod.Transcript(
            segments=[],
            speakers=[],
            language="en",
            audio_file=str(audio_file),
        )

    monkeypatch.setattr(tr_mod, "_load_stereo_int16", _forbid_full_decode)
    monkeypatch.setattr(tr_mod, "_stream_stereo_stats", _fake_stream_stats)
    monkeypatch.setattr(tr_mod, "_transcribe_dual_diarize", _fake_dual_diarize)
    monkeypatch.setattr(tr_mod, "_is_stereo", lambda _p: True)

    for dur in (float(tr_mod.MAX_IN_MEMORY_STEREO_SECONDS + 120), 0.0, None):
        monkeypatch.setattr(tr_mod, "get_audio_duration", lambda _p, _d=dur: _d)
        cfg = tr_mod.TranscriptionConfig(
            mixdown="dual-diarize",
            use_dual_channel=True,
            single_source_fallback=True,
            auto_vocabulary=False,
        )
        tr_mod.transcribe(fake_audio, cfg)

    assert len(stream_calls) == 3
    assert seen_stereo_chans == [None, None, None]


class _FakePipePopen:
    """In-memory Popen stub streaming interleaved s16le chunks in O(1) RAM."""

    def __init__(self, chunks: list[bytes], returncode: int = 0) -> None:
        self._chunks = list(chunks)
        self._idx = 0
        self._buf = b""
        self._rc = returncode
        self.stdout = self

    def read(self, size: int = -1) -> bytes:
        while len(self._buf) < size and self._idx < len(self._chunks):
            self._buf += self._chunks[self._idx]
            self._idx += 1
        if size < 0 or len(self._buf) <= size:
            out, self._buf = self._buf, b""
            return out
        out, self._buf = self._buf[:size], self._buf[size:]
        return out

    def close(self) -> None:
        pass

    def kill(self) -> None:
        pass

    def wait(self) -> int:
        return self._rc


def _interleave_i16(left: np.ndarray, right: np.ndarray) -> bytes:
    out = np.empty(left.size * 2, dtype="<i2")
    out[0::2] = left.astype("<i2")
    out[1::2] = right.astype("<i2")
    return out.tobytes()


def test_stream_single_source_detects_late_system_audio_and_late_divergence(
    tmp_path: Path, monkeypatch
):
    """Streaming stats must see system audio and channel divergence after minute 3."""
    import subprocess

    fake_audio = tmp_path / "long_meeting.ogg"
    fake_audio.write_bytes(b"OggS")
    sr = 16000
    t_10s = np.linspace(0, 10.0, sr * 10, dtype=np.float32)
    mic_10s = (np.sin(2 * np.pi * 220 * t_10s) * 4000).astype(np.int16)
    sys_10s = (np.sin(2 * np.pi * 550 * t_10s) * 4000).astype(np.int16)
    silent_10s = np.zeros(sr * 10, dtype=np.int16)

    # Case 1: 200 s (20 x 10 s) of active mic + silent system, then 20 s of active
    # remote speech (well past any 180 s prefix window).
    silent_pair = _interleave_i16(mic_10s, silent_10s)
    active_pair = _interleave_i16(mic_10s, sys_10s)
    late_sys_chunks = [silent_pair] * 20 + [active_pair] * 2

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_a, **_kw: _FakePipePopen(late_sys_chunks),
    )
    assert tr_mod._is_single_source_stereo(fake_audio, tr_mod.TranscriptionConfig()) is False

    # Case 2: 200 s of identical L/R (dual-mono), followed by 40 s of independent L/R
    # so overall correlation drops well below channel_duplicate_corr (0.95).
    dup_pair = _interleave_i16(mic_10s, mic_10s)
    late_div_chunks = [dup_pair] * 20 + [active_pair] * 4

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_a, **_kw: _FakePipePopen(late_div_chunks),
    )
    assert tr_mod._is_single_source_stereo(fake_audio, tr_mod.TranscriptionConfig()) is False

    # Case 3: Entire 220 s is identical L/R -> truly single-source (True).
    all_dup_chunks = [dup_pair] * 22
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_a, **_kw: _FakePipePopen(all_dup_chunks),
    )
    assert tr_mod._is_single_source_stereo(fake_audio, tr_mod.TranscriptionConfig()) is True


def test_stream_stereo_stats_matches_in_memory_with_unaligned_pipe_reads(
    tmp_path: Path, monkeypatch
):
    """Online Chan merge matches full-array RMS/correlation even with unaligned reads."""
    import subprocess

    rng = np.random.default_rng(42)
    n = 48000  # 3 seconds
    left = rng.integers(-8000, 8000, size=n, dtype=np.int16)
    right = (0.6 * left.astype(np.float64) + rng.normal(0, 2000, size=n)).clip(
        -32768, 32767
    ).astype(np.int16)
    raw = _interleave_i16(left, right)

    # Deliberately awkward odd-sized read chunks (1001 bytes) to test carry handling.
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_a, **_kw: _FakePipePopen([raw]),
    )
    stats = tr_mod._stream_stereo_stats(tmp_path / "x.ogg", chunk_bytes=1001)
    assert stats is not None
    l_rms, r_rms, total_n, corr = stats

    lf = left.astype(np.float64)
    rf = right.astype(np.float64)
    exp_l = float(np.sqrt(np.mean(lf[np.abs(lf) > 50.0] ** 2)))
    exp_r = float(np.sqrt(np.mean(rf[np.abs(rf) > 50.0] ** 2)))
    exp_corr = float(np.corrcoef(lf, rf)[0, 1])

    assert total_n == n
    assert abs(l_rms - exp_l) < 1e-6
    assert abs(r_rms - exp_r) < 1e-6
    assert corr is not None and abs(corr - exp_corr) < 1e-6

