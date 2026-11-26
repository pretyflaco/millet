"""GUI option-plumbing tests.

These tests exercise ``millet.gui.launch`` and verify that the ASR/Torch options
flow into the transcribe_kwargs dict that ultimately constructs a
``TranscriptionConfig``.  The GTK main loop is never started.

Skipped automatically when PyGObject is unavailable (e.g. on systems without
GTK, including some CI runners).
"""

from __future__ import annotations

import pytest

# Skip the whole module if GTK bindings aren't installed.
pytest.importorskip("gi")


def test_launch_threads_new_kwargs_into_transcribe_kwargs(monkeypatch):
    """``launch`` must pass asr_backend, torch_device, and mlx_model into
    ``MeetRecorderWindow``'s transcribe_kwargs dict so the next transcription
    picks them up via TranscriptionConfig."""
    import millet.gui as gui_mod

    captured: dict = {}

    class _StubWindow:
        # Match the attributes that launch() sets/hides after construction.
        _rec_btn_box = type("_X", (), {"hide": lambda self: None})()
        _alignment_box = type("_X", (), {"hide": lambda self: None})()
        _label_box = type("_X", (), {"hide": lambda self: None})()
        _sync_box = type("_X", (), {"hide": lambda self: None})()
        _progress_bar = type("_X", (), {"hide": lambda self: None})()
        _open_transcript_btn = type("_X", (), {"hide": lambda self: None})()
        _open_folder_btn = type("_X", (), {"hide": lambda self: None})()
        _bg_label = type("_X", (), {"hide": lambda self: None})()

        def __init__(self, capture_kwargs, transcribe_kwargs, **kwargs):
            captured["capture_kwargs"] = capture_kwargs
            captured["transcribe_kwargs"] = transcribe_kwargs
            captured["kwargs"] = kwargs

        def show_all(self):
            pass

    # Patch the Window class and Gtk.main so launch() doesn't enter a real loop.
    monkeypatch.setattr(gui_mod, "MeetRecorderWindow", _StubWindow)
    monkeypatch.setattr(gui_mod.Gtk, "main", lambda: None)

    gui_mod.launch(
        asr_backend="mlx",
        torch_device="mps",
        mlx_model="mlx-community/whisper-large-v3-turbo",
        device="cpu",
    )

    tk = captured["transcribe_kwargs"]
    assert tk["asr_backend"] == "mlx"
    assert tk["torch_device"] == "mps"
    assert tk["mlx_model"] == "mlx-community/whisper-large-v3-turbo"
    assert tk["device"] == "cpu"


def test_launch_threads_parakeet_kwargs(monkeypatch):
    """``launch`` passes the Parakeet model and alignment flag through, and the
    resulting kwargs build a valid TranscriptionConfig."""
    import millet.gui as gui_mod
    from millet.transcribe import TranscriptionConfig

    captured: dict = {}

    class _Hide:
        def hide(self):
            pass

    class _StubWindow:
        _rec_btn_box = _alignment_box = _label_box = _sync_box = _Hide()
        _progress_bar = _open_transcript_btn = _open_folder_btn = _Hide()
        _bg_label = _Hide()

        def __init__(self, capture_kwargs, transcribe_kwargs, **kwargs):
            captured["transcribe_kwargs"] = transcribe_kwargs

        def show_all(self):
            pass

    monkeypatch.setattr(gui_mod, "MeetRecorderWindow", _StubWindow)
    monkeypatch.setattr(gui_mod.Gtk, "main", lambda: None)

    gui_mod.launch(
        asr_backend="parakeet",
        parakeet_model="nemo-parakeet-tdt-0.6b-v3",
        device="cpu",
        torch_device="cpu",
    )

    tk = captured["transcribe_kwargs"]
    assert tk["asr_backend"] == "parakeet"
    assert tk["parakeet_model"] == "nemo-parakeet-tdt-0.6b-v3"
    assert tk["parakeet_skip_alignment"] is True
    cfg = TranscriptionConfig(**tk)
    assert cfg.asr_backend == "parakeet"
    assert cfg.parakeet_model == "nemo-parakeet-tdt-0.6b-v3"


def test_launch_defaults_pass_none_through(monkeypatch):
    """When the caller doesn't override device/torch_device/mlx_model,
    ``launch`` should pass None so TranscriptionConfig.__post_init__ resolves
    them via platform detection."""
    import millet.gui as gui_mod

    captured: dict = {}

    class _StubWindow:
        _rec_btn_box = type("_X", (), {"hide": lambda self: None})()
        _alignment_box = type("_X", (), {"hide": lambda self: None})()
        _label_box = type("_X", (), {"hide": lambda self: None})()
        _sync_box = type("_X", (), {"hide": lambda self: None})()
        _progress_bar = type("_X", (), {"hide": lambda self: None})()
        _open_transcript_btn = type("_X", (), {"hide": lambda self: None})()
        _open_folder_btn = type("_X", (), {"hide": lambda self: None})()
        _bg_label = type("_X", (), {"hide": lambda self: None})()

        def __init__(self, capture_kwargs, transcribe_kwargs, **kwargs):
            captured["transcribe_kwargs"] = transcribe_kwargs

        def show_all(self):
            pass

    monkeypatch.setattr(gui_mod, "MeetRecorderWindow", _StubWindow)
    monkeypatch.setattr(gui_mod.Gtk, "main", lambda: None)

    gui_mod.launch()

    tk = captured["transcribe_kwargs"]
    # asr_backend default is "auto" (string), so the dataclass picks the
    # platform-appropriate backend.
    assert tk["asr_backend"] == "auto"
    # device, torch_device, mlx_model default to None — the dataclass resolves.
    assert tk["device"] is None
    assert tk["torch_device"] is None
    assert tk["mlx_model"] is None
    # Language defaults to "auto" so Whisper auto-detects.  Users can override
    # via --language on the CLI or the GUI Advanced panel dropdown.
    assert tk["language"] == "auto"


def test_launch_passes_explicit_language(monkeypatch):
    """``launch(language='en')`` must thread into transcribe_kwargs so the
    GUI Advanced panel dropdown can override Whisper's auto-detect."""
    import millet.gui as gui_mod

    captured: dict = {}

    class _StubWindow:
        _rec_btn_box = type("_X", (), {"hide": lambda self: None})()
        _alignment_box = type("_X", (), {"hide": lambda self: None})()
        _label_box = type("_X", (), {"hide": lambda self: None})()
        _sync_box = type("_X", (), {"hide": lambda self: None})()
        _progress_bar = type("_X", (), {"hide": lambda self: None})()
        _open_transcript_btn = type("_X", (), {"hide": lambda self: None})()
        _open_folder_btn = type("_X", (), {"hide": lambda self: None})()
        _bg_label = type("_X", (), {"hide": lambda self: None})()

        def __init__(self, capture_kwargs, transcribe_kwargs, **kwargs):
            captured["transcribe_kwargs"] = transcribe_kwargs

        def show_all(self):
            pass

    monkeypatch.setattr(gui_mod, "MeetRecorderWindow", _StubWindow)
    monkeypatch.setattr(gui_mod.Gtk, "main", lambda: None)

    gui_mod.launch(language="en")

    assert captured["transcribe_kwargs"]["language"] == "en"
