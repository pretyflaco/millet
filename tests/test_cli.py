"""CLI-level tests for the meet command group.

These tests use ``click.testing.CliRunner`` and stub out the heavyweight
transcription pipeline so we can validate flag parsing, defaults, and
config plumbing without loading torch/whisperx.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from millet import cli as meet_cli


def _stub_transcribe_pipeline():
    """Patch the heavyweight pieces of `meet transcribe` so the CLI runs end-to-end
    without invoking whisperx, torch, or filesystem-dependent post-processing.

    Returns the patcher context manager.
    """
    from contextlib import ExitStack
    from types import SimpleNamespace

    stack = ExitStack()

    # Stub the transcription pipeline.
    fake_transcript = SimpleNamespace(
        segments=[],
        speakers=[],
        language="en",
        audio_file="x",
        duration=0.0,
        save=lambda out_dir, basename: {"text": out_dir / f"{basename}.txt"},
        to_text=lambda: "",
    )
    stack.enter_context(
        patch("millet.transcribe.transcribe", return_value=fake_transcript)
    )
    stack.enter_context(patch("millet.transcribe.ensure_gpu_available"))
    # Since the 0.10.0 cli/ package split, the helpers live in
    # millet.cli._helpers and are bound into the millet.cli.transcribe
    # MODULE namespace at import time — patch them where they are looked
    # up.  Two traps here (both bit this suite while it was excluded
    # from CI): the old target `millet.cli._generate_*` no longer exists,
    # and the dotted string "millet.cli.transcribe" resolves to the Click
    # *Command* (the package re-exports it, shadowing the submodule), so
    # we must patch the module object from sys.modules directly.
    import sys

    import millet.cli  # noqa: F401 — ensure the submodule is imported

    transcribe_mod = sys.modules["millet.cli.transcribe"]
    stack.enter_context(
        patch.object(transcribe_mod, "_generate_summary", return_value=None)
    )
    stack.enter_context(patch.object(transcribe_mod, "_generate_pdf"))
    return stack


class TestTranscribeDeviceDefault:
    def test_no_device_flag_resolves_to_cuda_on_linux(self, tmp_path, monkeypatch):
        monkeypatch.setattr("millet.transcribe._apple_silicon", lambda: False)
        # Stub validation (#7) so 'cuda' passes regardless of host GPU.
        monkeypatch.setattr(
            "millet.transcribe._torch_device_available", lambda d: True
        )

        # Capture the TranscriptionConfig actually constructed.
        captured: dict = {}

        from millet.transcribe import TranscriptionConfig as RealCfg

        def _spy_cfg(**kwargs):
            cfg = RealCfg(**kwargs)
            captured["cfg"] = cfg
            return cfg

        audio = tmp_path / "x.wav"
        audio.write_bytes(b"")  # empty file is fine — pipeline is stubbed

        with _stub_transcribe_pipeline(), patch(
            "millet.transcribe.TranscriptionConfig", side_effect=_spy_cfg
        ):
            runner = CliRunner()
            result = runner.invoke(
                meet_cli.transcribe,
                [str(audio), "--no-summarize"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert captured["cfg"].device == "cuda"
        assert captured["cfg"].torch_device == "cuda"

    def test_no_device_flag_resolves_to_cpu_mps_on_apple_silicon(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("millet.transcribe._apple_silicon", lambda: True)
        monkeypatch.setattr("millet.transcribe._mps_available", lambda: True)
        monkeypatch.setattr("millet.transcribe._mlx_available", lambda: False)
        # Also stub the device-availability validator (#7) so 'mps' passes
        # validation on a Linux CI runner.
        monkeypatch.setattr(
            "millet.transcribe._torch_device_available", lambda d: True
        )

        captured: dict = {}
        from millet.transcribe import TranscriptionConfig as RealCfg

        def _spy_cfg(**kwargs):
            cfg = RealCfg(**kwargs)
            captured["cfg"] = cfg
            return cfg

        audio = tmp_path / "x.wav"
        audio.write_bytes(b"")

        with _stub_transcribe_pipeline(), patch(
            "millet.transcribe.TranscriptionConfig", side_effect=_spy_cfg
        ):
            runner = CliRunner()
            result = runner.invoke(
                meet_cli.transcribe,
                [str(audio), "--no-summarize"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert captured["cfg"].device == "cpu"
        assert captured["cfg"].torch_device == "mps"

    def test_explicit_device_flag_overrides_default(self, tmp_path, monkeypatch):
        # On a Mac, an explicit --device cpu should still pass through unchanged.
        monkeypatch.setattr("millet.transcribe._apple_silicon", lambda: True)
        monkeypatch.setattr("millet.transcribe._mps_available", lambda: False)
        monkeypatch.setattr("millet.transcribe._mlx_available", lambda: False)

        captured: dict = {}
        from millet.transcribe import TranscriptionConfig as RealCfg

        def _spy_cfg(**kwargs):
            cfg = RealCfg(**kwargs)
            captured["cfg"] = cfg
            return cfg

        audio = tmp_path / "x.wav"
        audio.write_bytes(b"")

        with _stub_transcribe_pipeline(), patch(
            "millet.transcribe.TranscriptionConfig", side_effect=_spy_cfg
        ):
            runner = CliRunner()
            result = runner.invoke(
                meet_cli.transcribe,
                [str(audio), "--device", "cpu", "--no-summarize"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0, result.output
        assert captured["cfg"].device == "cpu"


class TestParakeetBackendOnGuiAndRun:
    """``gui`` and ``run`` accept the parakeet backend and its options and
    hand them on unchanged, like ``transcribe`` does."""

    def test_gui_passes_parakeet_options_to_launch(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        captured: dict = {}
        # Stub the module so the test does not need GTK bindings.
        monkeypatch.setitem(
            sys.modules,
            "millet.gui",
            SimpleNamespace(launch=lambda **kw: captured.update(kw)),
        )
        result = CliRunner().invoke(
            meet_cli.gui,
            [
                "--asr-backend", "parakeet",
                "--parakeet-model", "nemo-parakeet-tdt-0.6b-v3",
                "--parakeet-keep-alignment",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert captured["asr_backend"] == "parakeet"
        assert captured["parakeet_model"] == "nemo-parakeet-tdt-0.6b-v3"
        assert captured["parakeet_skip_alignment"] is False

    def test_gui_defaults_unchanged(self, monkeypatch):
        import sys
        from types import SimpleNamespace

        captured: dict = {}
        monkeypatch.setitem(
            sys.modules,
            "millet.gui",
            SimpleNamespace(launch=lambda **kw: captured.update(kw)),
        )
        result = CliRunner().invoke(meet_cli.gui, [], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert captured["asr_backend"] == "auto"
        assert captured["parakeet_model"] is None
        assert captured["parakeet_skip_alignment"] is True

    def test_run_builds_config_with_parakeet_options(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        monkeypatch.setattr("millet.transcribe._apple_silicon", lambda: False)
        monkeypatch.setattr(
            "millet.transcribe._torch_device_available", lambda d: True
        )
        monkeypatch.setattr("millet.capture.check_prerequisites", lambda: [])

        class _Stop(Exception):
            pass

        def _start():
            # The config is built before recording starts; abort here so
            # no real recording loop runs.
            raise _Stop

        session = SimpleNamespace(
            output_file=tmp_path / "x.wav",
            mic_source="mic",
            monitor_source="mon",
            start=_start,
        )
        monkeypatch.setattr("millet.capture.create_session", lambda **kw: session)

        captured: dict = {}
        from millet.transcribe import TranscriptionConfig as RealCfg

        def _spy_cfg(**kwargs):
            cfg = RealCfg(**kwargs)
            captured["cfg"] = cfg
            return cfg

        with patch("millet.transcribe.TranscriptionConfig", side_effect=_spy_cfg):
            result = CliRunner().invoke(
                meet_cli.run,
                [
                    "--asr-backend", "parakeet",
                    "--parakeet-model", "nemo-parakeet-tdt-0.6b-v3",
                    "--mixdown", "dual",
                    "--no-summarize",
                ],
            )
        assert isinstance(result.exception, _Stop), result.output
        cfg = captured["cfg"]
        assert cfg.asr_backend == "parakeet"
        assert cfg.parakeet_model == "nemo-parakeet-tdt-0.6b-v3"
        assert cfg.parakeet_skip_alignment is True
        assert cfg.mixdown == "dual"
        assert "parakeet (nemo-parakeet-tdt-0.6b-v3)" in result.output

    def test_run_rejects_unknown_backend(self):
        result = CliRunner().invoke(meet_cli.run, ["--asr-backend", "nope"])
        assert result.exit_code != 0
        assert "parakeet" in result.output

    def test_download_passes_parakeet_model(self, monkeypatch):
        calls: dict = {}
        monkeypatch.setattr(
            "millet.parakeet.ensure_parakeet_cached", lambda model=None: False
        )
        monkeypatch.setattr(
            "millet.parakeet.download_parakeet",
            lambda model=None, progress_callback=None: calls.update(model=model),
        )
        result = CliRunner().invoke(
            meet_cli.download,
            ["parakeet", "--parakeet-model", "nemo-parakeet-tdt-0.6b-v3"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert calls["model"] == "nemo-parakeet-tdt-0.6b-v3"
