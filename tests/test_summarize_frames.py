"""Optional still-frame (vision) input to summarization — v0.20.0.

A narrated screen recording carries one PNG per transcript cue.  Sending
those to a vision-capable TEE model lets the summary report what is
visibly wrong, not just what the narrator said aloud.

Two invariants drive most of these tests:

1. **Frames never cause a failure.**  They are an enhancement; an
   unreadable file, a model that can't see, or a fallback to a text-only
   backend must all degrade to the previous text-only behaviour.
2. **The vision allowlist is not the vendor's `multimodal` flag.**
   Tinfoil advertised `multimodal: true` for deepseek-v4-1-flash while its
   vision endpoint answered 502 on every request (2026-09-12).  It was
   only allowlisted (as the new default, 0.21.6) after vision was
   re-verified live on 2026-10-06.  The sibling fallback (glm-5-3) cannot
   see at all, so a drained primary pool must degrade to text-only.
"""
from __future__ import annotations

import sys
import types

import pytest

import millet.summarize as sm
from millet.summarize import (
    MAX_FRAMES,
    SummaryConfig,
    _usable_frames,
    discover_cue_frames,
    model_supports_vision,
)


def _png(path, size: int = 64):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * size)
    return path


# ── vision allowlist ───────────────────────────────────────────────────────


class TestVisionAllowlist:
    def test_default_model_can_see(self):
        assert model_supports_vision(sm.DEFAULT_TINFOIL_MODEL)

    def test_sibling_fallback_is_excluded(self):
        """glm-5-3 is text-only (catalog multimodal: false)."""
        assert not model_supports_vision(sm.DEFAULT_TINFOIL_FALLBACK_MODEL)

    def test_tee_suffix_still_matches(self):
        # MeetingSummary.model carries a " (TEE)" suffix.
        assert model_supports_vision("deepseek-v4-1-flash (TEE)")
        assert model_supports_vision("glm-5-3-flash (TEE)")

    def test_glm_flash_prefix_does_not_admit_full_glm(self):
        # Prefix matching must not let "glm-5-3" ride on "glm-5-3-flash".
        assert model_supports_vision("glm-5-3-flash")
        assert not model_supports_vision("glm-5-3 (TEE)")

    @pytest.mark.parametrize("model", ["glm-5-3", "qwen3.8:27b", "", None])
    def test_non_vision_models(self, model):
        assert not model_supports_vision(model)


# ── frame normalization ────────────────────────────────────────────────────


class TestUsableFrames:
    def test_keeps_images_in_order(self, tmp_path):
        a = _png(tmp_path / "cue_00-00-01.png")
        b = _png(tmp_path / "cue_00-00-02.png")
        assert _usable_frames([a, b]) == [a, b]

    def test_drops_missing_files(self, tmp_path):
        good = _png(tmp_path / "cue_00-00-01.png")
        assert _usable_frames([good, tmp_path / "gone.png"]) == [good]

    def test_drops_non_image_suffixes(self, tmp_path):
        good = _png(tmp_path / "cue_00-00-01.png")
        notes = tmp_path / "notes.txt"
        notes.write_text("nope")
        assert _usable_frames([good, notes]) == [good]

    def test_deduplicates(self, tmp_path):
        a = _png(tmp_path / "cue_00-00-01.png")
        assert _usable_frames([a, a]) == [a]

    def test_drops_oversized_frames(self, tmp_path):
        big = _png(tmp_path / "cue_00-00-01.png", size=sm._MAX_FRAME_BYTES + 1)
        assert _usable_frames([big]) == []

    def test_caps_frame_count(self, tmp_path):
        frames = [_png(tmp_path / f"cue_{i:04d}.png") for i in range(MAX_FRAMES + 10)]
        assert len(_usable_frames(frames)) == MAX_FRAMES

    def test_over_cap_samples_evenly_not_head(self, tmp_path):
        """18 cue frames on a ~4-minute recording (session
        01M2P6FTRG4WAKKE5T7TV6HNFM) exceeded the endpoint's 10-image cap.
        The sample must span the whole timeline: first + last kept, not a
        head-truncation that summarizes only the opening minutes."""
        frames = [_png(tmp_path / f"cue_{i:04d}.png") for i in range(18)]
        picked = _usable_frames(frames)
        assert len(picked) == MAX_FRAMES
        assert picked[0] == frames[0]
        assert picked[-1] == frames[-1]
        assert len(set(picked)) == MAX_FRAMES  # no duplicates from rounding
        # Order preserved: picked indices are strictly increasing in `frames`.
        idx = [frames.index(p) for p in picked]
        assert idx == sorted(idx)

    def test_config_normalizes_frames(self, tmp_path):
        good = _png(tmp_path / "cue_00-00-01.png")
        cfg = SummaryConfig(backend="tinfoil", frames=[good, tmp_path / "missing.png"])
        assert cfg.frames == [good]

    def test_config_defaults_to_no_frames(self):
        assert SummaryConfig(backend="tinfoil").frames is None


# ── discovery ──────────────────────────────────────────────────────────────


class TestDiscoverCueFrames:
    def test_finds_frames_in_chronological_order(self, tmp_path):
        # Written out of order; cue_* names sort to narration order.
        _png(tmp_path / "attachments" / "cue_00-01-00.png")
        _png(tmp_path / "attachments" / "cue_00-00-02.png")
        names = [p.name for p in discover_cue_frames(tmp_path)]
        assert names == ["cue_00-00-02.png", "cue_00-01-00.png"]

    def test_ignores_other_attachments(self, tmp_path):
        _png(tmp_path / "attachments" / "cue_00-00-01.png")
        _png(tmp_path / "attachments" / "diagram.png")
        assert [p.name for p in discover_cue_frames(tmp_path)] == ["cue_00-00-01.png"]

    def test_no_attachments_dir_is_not_an_error(self, tmp_path):
        assert discover_cue_frames(tmp_path) == []


# ── message construction + degradation ─────────────────────────────────────


def _fake_tinfoil(monkeypatch):
    """Fake SDK capturing the create(**kwargs) of each attempt."""
    calls: list[dict] = []

    class _Msg:
        content = "## Overview\n\nSeen it."

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class FakeAI:
        def __init__(self, api_key=None):
            self.chat = _Chat()

    mod = types.ModuleType("tinfoil")
    mod.TinfoilAI = FakeAI
    monkeypatch.setitem(sys.modules, "tinfoil", mod)
    monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: "tk_fake")
    monkeypatch.setattr(sm, "verify_tinfoil_model", lambda *a, **k: None)
    return calls


def _user_content(call):
    return next(m["content"] for m in call["messages"] if m["role"] == "user")


class TestTinfoilMessageShape:
    def test_no_frames_sends_a_plain_string(self, monkeypatch):
        """The text-only path must be byte-identical to before."""
        calls = _fake_tinfoil(monkeypatch)
        cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash")
        result = sm._summarize_tinfoil("sys", "user text", cfg)
        assert _user_content(calls[0]) == "user text"
        assert result.frames_used == 0

    def test_frames_become_image_parts(self, monkeypatch, tmp_path):
        calls = _fake_tinfoil(monkeypatch)
        frames = [_png(tmp_path / f"cue_{i}.png") for i in range(3)]
        cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash", frames=frames)
        result = sm._summarize_tinfoil("sys", "user text", cfg)

        content = _user_content(calls[0])
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "user text"}
        images = [p for p in content if p["type"] == "image_url"]
        assert len(images) == 3
        assert images[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert result.frames_used == 3

    def test_non_vision_model_drops_frames(self, monkeypatch, tmp_path):
        """Degrade to text-only rather than erroring."""
        calls = _fake_tinfoil(monkeypatch)
        frames = [_png(tmp_path / "cue_0.png")]
        cfg = SummaryConfig(backend="tinfoil", model="glm-5-3", frames=frames)
        result = sm._summarize_tinfoil("sys", "user text", cfg)
        assert _user_content(calls[0]) == "user text"
        assert result.frames_used == 0

    def test_unreadable_frame_is_skipped_not_fatal(self, monkeypatch, tmp_path):
        calls = _fake_tinfoil(monkeypatch)
        good = _png(tmp_path / "cue_0.png")
        cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash", frames=[good])
        good.unlink()  # vanishes after config validation, before the send
        result = sm._summarize_tinfoil("sys", "user text", cfg)
        images = [p for p in _user_content(calls[0]) if p["type"] == "image_url"]
        assert images == []
        assert result.frames_used == 0

    def test_jpeg_gets_the_right_mime(self, monkeypatch, tmp_path):
        calls = _fake_tinfoil(monkeypatch)
        jpg = tmp_path / "cue_0.jpg"
        jpg.write_bytes(b"\xff\xd8\xff" + b"\x00" * 32)
        cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash", frames=[jpg])
        sm._summarize_tinfoil("sys", "user text", cfg)
        img = [p for p in _user_content(calls[0]) if p["type"] == "image_url"][0]
        assert img["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_over_cap_frames_send_at_most_ten_images(self, monkeypatch, tmp_path):
        """Regression (verified live 2026-09-17): the attested vision
        endpoints reject a request carrying more than 10 images — venice
        answered 400 "At most 10 image(s) may be provided in one request"
        and the job failed on every vision backend.  The request must
        carry at most MAX_FRAMES image parts no matter how many cue
        frames the session produced."""
        calls = _fake_tinfoil(monkeypatch)
        frames = [_png(tmp_path / f"cue_{i:04d}.png") for i in range(MAX_FRAMES + 8)]
        cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash", frames=frames)
        result = sm._summarize_tinfoil("sys", "user text", cfg)
        images = [p for p in _user_content(calls[0]) if p["type"] == "image_url"]
        assert len(images) == MAX_FRAMES
        assert result.frames_used == MAX_FRAMES


class TestSiblingFallbackDropsFrames:
    def test_frames_dropped_when_sibling_serves_the_request(self, monkeypatch, tmp_path):
        """The primary pool is drained; the sibling can't see images, so the
        request must still succeed -- as text."""
        seen: list[tuple[str, object]] = []

        class _Msg:
            content = "## Overview\n\nText only."

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            def create(self, **kwargs):
                model = kwargs["model"]
                user = next(m["content"] for m in kwargs["messages"] if m["role"] == "user")
                seen.append((model, user))
                if model == sm.DEFAULT_TINFOIL_MODEL:
                    raise RuntimeError(
                        "Error code: 503 - The engine is currently overloaded"
                    )
                return _Resp()

        class _Chat:
            completions = _Completions()

        class FakeAI:
            def __init__(self, api_key=None):
                self.chat = _Chat()

        mod = types.ModuleType("tinfoil")
        mod.TinfoilAI = FakeAI
        monkeypatch.setitem(sys.modules, "tinfoil", mod)
        monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: "tk_fake")
        monkeypatch.setattr(sm, "verify_tinfoil_model", lambda *a, **k: None)
        import time as _t
        monkeypatch.setattr(_t, "sleep", lambda *a, **k: None)

        frames = [_png(tmp_path / "cue_0.png")]
        cfg = SummaryConfig(backend="tinfoil", model=sm.DEFAULT_TINFOIL_MODEL, frames=frames)
        result = sm._summarize_tinfoil("sys", "user text", cfg)

        primary = [u for m, u in seen if m == sm.DEFAULT_TINFOIL_MODEL]
        sibling = [u for m, u in seen if m == sm.DEFAULT_TINFOIL_FALLBACK_MODEL]
        assert all(isinstance(u, list) for u in primary)   # primary saw images
        assert sibling and all(u == "user text" for u in sibling)  # sibling did not
        assert result.frames_used == 0
        assert result.fallback_used is True


# ── frames survive the backend-fallback config rebuild ─────────────────────


class TestFramesSurviveDispatch:
    def test_dispatch_carries_frames_into_rebuilt_config(self, monkeypatch, tmp_path):
        """_dispatch rebuilds SummaryConfig field-by-field for a fallback
        backend; a dropped field silently downgrades the run."""
        captured: list[SummaryConfig] = []

        def fake_tinfoil(system_prompt, user_prompt, config):
            captured.append(config)
            return sm.MeetingSummary(
                markdown="x" * 500, model=config.model,
                elapsed_seconds=1.0, backend="tinfoil",
            )

        monkeypatch.setattr(sm, "_summarize_tinfoil", fake_tinfoil)
        frames = [_png(tmp_path / "cue_0.png")]
        cfg = SummaryConfig(backend="ollama", frames=frames)
        sm._dispatch("tinfoil", "sys", "usr", cfg)
        assert captured[0].frames == frames


# ── provenance ─────────────────────────────────────────────────────────────


class TestFramesProvenance:
    def test_meta_records_frames_used(self, tmp_path):
        summary = sm.MeetingSummary(
            markdown="## Overview\n\nBody.", model="glm-5-3-flash (TEE)",
            elapsed_seconds=1.0, backend="tinfoil", frames_used=7,
        )
        summary.save(tmp_path, "sess")
        import json
        meta = json.loads((tmp_path / "sess.summary.meta.json").read_text())
        assert meta["frames_used"] == 7

    def test_meta_defaults_to_zero(self, tmp_path):
        summary = sm.MeetingSummary(
            markdown="## Overview\n\nBody.", model="m",
            elapsed_seconds=1.0, backend="tinfoil",
        )
        summary.save(tmp_path, "sess")
        import json
        meta = json.loads((tmp_path / "sess.summary.meta.json").read_text())
        assert meta["frames_used"] == 0


class TestAttestationRetry:
    """Tinfoil intermittently serves a malformed SEV attestation report and
    the SDK refuses it.  Observed 2026-09-12 on plain text: two failures
    then success.  Classified non-transient, it failed the job outright."""

    _ERR = (
        "SEV attestation verification failed: Failed to parse report: "
        "current_tcb not correctly formed: mbz range current_tcb[0x10:0x2f] "
        "not all zero: 0x6100000005020301"
    )

    def test_attestation_failure_is_transient(self):
        assert sm._is_transient_network_error(ValueError(self._ERR))

    def test_attestation_failure_is_not_a_pool_error(self):
        """A sibling model runs on the same enclave infrastructure, so
        switching models is the wrong response -- retry the same one."""
        assert not sm._is_model_pool_error(ValueError(self._ERR))

    def test_has_its_own_larger_budget(self):
        """3 attempts against a ~25% failure rate loses jobs to bad luck."""
        assert sm._TINFOIL_ATTEST_MAX_ATTEMPTS > sm._TINFOIL_MAX_ATTEMPTS

    def test_attestation_retries_do_not_consume_the_transient_budget(self, monkeypatch):
        """An attestation blip must not burn the attempts reserved for real
        network faults -- otherwise two blips leave nothing for a 503."""
        state = {"n": 0}

        class _Msg:
            content = "## Overview\n\nOK."

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            def create(self, **kwargs):
                state["n"] += 1
                if state["n"] <= sm._TINFOIL_MAX_ATTEMPTS + 1:
                    raise ValueError(TestAttestationRetry._ERR)
                return _Resp()

        class _Chat:
            completions = _Completions()

        class FakeAI:
            def __init__(self, api_key=None):
                self.chat = _Chat()

        mod = types.ModuleType("tinfoil")
        mod.TinfoilAI = FakeAI
        monkeypatch.setitem(sys.modules, "tinfoil", mod)
        monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: "tk_fake")
        monkeypatch.setattr(sm, "verify_tinfoil_model", lambda *a, **k: None)
        import time as _t
        monkeypatch.setattr(_t, "sleep", lambda *a, **k: None)

        cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash")
        result = sm._summarize_tinfoil("sys", "user", cfg)
        assert state["n"] == sm._TINFOIL_MAX_ATTEMPTS + 2
        assert "OK." in result.markdown

    def test_exhausted_attestation_budget_says_it_was_not_a_downgrade(self, monkeypatch):
        def _boom(**kwargs):
            raise ValueError(TestAttestationRetry._ERR)

        class _Completions:
            create = staticmethod(_boom)

        class _Chat:
            completions = _Completions()

        class FakeAI:
            def __init__(self, api_key=None):
                self.chat = _Chat()

        mod = types.ModuleType("tinfoil")
        mod.TinfoilAI = FakeAI
        monkeypatch.setitem(sys.modules, "tinfoil", mod)
        monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: "tk_fake")
        monkeypatch.setattr(sm, "verify_tinfoil_model", lambda *a, **k: None)
        import time as _t
        monkeypatch.setattr(_t, "sleep", lambda *a, **k: None)

        cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash")
        with pytest.raises(RuntimeError, match="never accepted unverified"):
            sm._summarize_tinfoil("sys", "user", cfg)

    def test_retries_then_succeeds(self, monkeypatch):
        state = {"n": 0}

        class _Msg:
            content = "## Overview\n\nVerified on retry."

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            def create(self, **kwargs):
                state["n"] += 1
                if state["n"] < 3:
                    raise ValueError(TestAttestationRetry._ERR)
                return _Resp()

        class _Chat:
            completions = _Completions()

        class FakeAI:
            def __init__(self, api_key=None):
                self.chat = _Chat()

        mod = types.ModuleType("tinfoil")
        mod.TinfoilAI = FakeAI
        monkeypatch.setitem(sys.modules, "tinfoil", mod)
        monkeypatch.setattr(sm, "_resolve_tinfoil_api_key", lambda: "tk_fake")
        monkeypatch.setattr(sm, "verify_tinfoil_model", lambda *a, **k: None)
        import time as _t
        monkeypatch.setattr(_t, "sleep", lambda *a, **k: None)

        cfg = SummaryConfig(backend="tinfoil", model="glm-5-3-flash")
        result = sm._summarize_tinfoil("sys", "user", cfg)
        assert state["n"] == 3
        assert "Verified on retry" in result.markdown
        # Same model throughout: not a sibling fallback.
        assert result.fallback_used is False


class TestSiblingFallbackProvenanceNotClobbered:
    def test_summarize_preserves_backend_internal_fallback(self, monkeypatch):
        """Regression: summarize() assigned fallback_used from the backend
        comparison, erasing the tinfoil sibling-model fallback (which keeps
        backend == config.backend) from the meta sidecar."""
        def fake_dispatch(backend, system_prompt, user_prompt, config, **kw):
            return sm.MeetingSummary(
                markdown="## Overview\n\n" + "x" * 400, model="sibling (TEE)",
                elapsed_seconds=1.0, backend="tinfoil", fallback_used=True,
            )

        monkeypatch.setattr(sm, "_dispatch", fake_dispatch)
        monkeypatch.setattr(sm, "is_backend_available", lambda cfg: True)
        result = sm.summarize("transcript", SummaryConfig(backend="tinfoil"))
        assert result.fallback_used is True
