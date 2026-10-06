"""GTK3 GUI widget for meet — a small always-on-top recording control.

Window layout (~300x180px):

    Idle / Done / Error:
    ┌──────────────────────────────┐
    │  Meet Recorder               │
    │     00:00:00    0 KB         │
    │     Ready                    │
    │        [ ● Record ]          │
    │   Open Transcript  Open Folder│
    │  Processing: meeting-...     │
    └──────────────────────────────┘

    Recording:
    ┌──────────────────────────────┐
    │  Meet Recorder               │
    │     00:07:23    14.2 MB      │
    │     Recording...             │
    │  [ ⏸ Pause ]  [ ■ Stop ]    │
    │  Processing: meeting-...     │
    └──────────────────────────────┘

    Paused:
    ┌──────────────────────────────┐
    │  Meet Recorder               │
    │     00:07:23    14.2 MB      │
    │     Paused                   │
    │  [ ▶ Resume ]  [ ■ Stop ]   │
    │  Processing: meeting-...     │
    └──────────────────────────────┘

Primary states (control recording):
    idle        → "Ready", green Record button
    recording   → "Recording...", Pause + Stop buttons, timer ticking
    paused      → "Paused", Resume + Stop buttons, timer frozen
    draining    → "Flushing buffer... Xs", buttons disabled
    done        → "Done — transcript saved", green Record button
    error       → error message, green Record button

Post-processing (transcription, summarization, labeling, sync) runs in a
background job queue.  After drain+stop the GUI returns to idle immediately
so the user can start a new recording.  Background progress is shown in a
small secondary status label at the bottom of the window.  Interactive
dialogs (alignment model prompt, speaker labeling, sync confirmation) are
deferred until the user is not actively recording.
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gdk, GLib, Gtk, Pango

from millet.capture import DRAIN_SECONDS
from millet.utils import fmt_elapsed, fmt_size

_log = logging.getLogger(__name__)


# ─── CSS ────────────────────────────────────────────────────────────────────

_CSS = b"""
.record-btn {
    background: #2ecc71;
    color: white;
    font-weight: bold;
    font-size: 14px;
    border-radius: 6px;
    padding: 8px 24px;
    border: none;
}
.record-btn:hover {
    background: #27ae60;
}
.stop-btn {
    background: #e74c3c;
    color: white;
    font-weight: bold;
    font-size: 14px;
    border-radius: 6px;
    padding: 8px 24px;
    border: none;
}
.stop-btn:hover {
    background: #c0392b;
}
.pause-btn {
    background: #f39c12;
    color: white;
    font-weight: bold;
    font-size: 14px;
    border-radius: 6px;
    padding: 8px 24px;
    border: none;
}
.pause-btn:hover {
    background: #e67e22;
}
.disabled-btn {
    background: #95a5a6;
    color: white;
    font-weight: bold;
    font-size: 14px;
    border-radius: 6px;
    padding: 8px 24px;
    border: none;
}
.timer-label {
    font-size: 28px;
    font-weight: bold;
    font-family: monospace;
}
.size-label {
    font-size: 14px;
    color: #7f8c8d;
    font-family: monospace;
}
.status-label {
    font-size: 13px;
    color: #7f8c8d;
}
.status-recording {
    font-size: 13px;
    color: #e74c3c;
    font-weight: bold;
}
.status-paused {
    font-size: 13px;
    color: #f39c12;
    font-weight: bold;
}
.status-draining {
    font-size: 13px;
    color: #f39c12;
    font-weight: bold;
}
.status-done {
    font-size: 13px;
    color: #2ecc71;
    font-weight: bold;
}
.status-error {
    font-size: 13px;
    color: #e74c3c;
    font-weight: bold;
}
.bg-status-label {
    font-size: 11px;
    color: #95a5a6;
}
.action-btn {
    background: transparent;
    color: #3498db;
    font-size: 12px;
    border: none;
    padding: 2px 8px;
    text-decoration: underline;
}
.action-btn:hover {
    color: #2980b9;
}
progressbar trough {
    min-height: 6px;
    background: #ecf0f1;
    border-radius: 3px;
}
progressbar progress {
    min-height: 6px;
    background: #e67e22;
    border-radius: 3px;
}
"""


# ─── State enum (primary states only) ──────────────────────────────────────


class _State:
    IDLE = "idle"
    RECORDING = "recording"
    PAUSED = "paused"
    DRAINING = "draining"
    DONE = "done"
    ERROR = "error"


# ─── Main Window ────────────────────────────────────────────────────────────


class MeetRecorderWindow(Gtk.Window):
    def __init__(
        self,
        capture_kwargs: dict,
        transcribe_kwargs: dict,
        summarize: bool = True,
        summary_preset: str | None = None,
        summary_backend: str | None = None,
        summary_model: str | None = None,
        ollama_singlepass: bool = False,
    ):
        super().__init__(title="Meet Recorder")

        self._capture_kwargs = capture_kwargs
        self._transcribe_kwargs = transcribe_kwargs
        self._summarize = summarize
        self._summary_preset = summary_preset
        self._summary_backend = summary_backend
        self._summary_model = summary_model
        self._ollama_singlepass = ollama_singlepass
        self._session = None
        self._state = _State.IDLE
        self._worker_thread = None
        self._drain_remaining = 0
        self._last_output: Path | None = None
        self._last_pdf: Path | None = None
        self._error_msg: str | None = None
        self._destroying = False

        # Background post-processing job queue
        self._job_queue: queue.Queue[Path] = queue.Queue()
        self._job_thread: threading.Thread | None = None
        # Serializes consumer start/stop decisions with enqueue so a job is
        # never stranded in the queue by a consumer exiting between put() and
        # _ensure_job_thread() (TOCTOU on is_alive()).
        self._job_lock = threading.Lock()

        # Threading synchronization for alignment model prompt
        self._alignment_event = threading.Event()
        self._alignment_choice: str | None = None  # "download" or "skip"
        self._alignment_lang: str | None = None

        # Threading synchronization for speaker labeling
        self._label_event = threading.Event()
        self._label_result: dict[str, str] | None = None  # label_map or None (skip)
        self._label_speakers: list = []  # SpeakerInfo list, set by worker
        self._label_entries: list = []  # Gtk.Entry widgets
        self._label_temp_clips: list[Path] = []  # temp WAV files for cleanup
        self._label_auto_matches: dict = {}  # speaker_id -> SpeakerMatch, set by worker
        self._label_channel_map: dict = {}  # speaker_id -> 'mic'|'system', set by worker
        self._label_audio_path: Path | None = None  # audio file for profile update

        # Threading synchronization for sync confirmation
        self._sync_event = threading.Event()
        self._sync_confirmed: bool = False  # True = Push, False = Skip

        # Window properties
        self.set_default_size(300, 150)
        self.set_keep_above(True)
        self.set_resizable(False)
        self.set_position(Gtk.WindowPosition.CENTER)

        # Load CSS
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(_CSS)
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

        # Layout
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        vbox.set_margin_top(12)
        vbox.set_margin_bottom(12)
        vbox.set_margin_start(20)
        vbox.set_margin_end(20)

        # ── Summarization quality preset (always visible, not "advanced") ──
        preset_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        preset_label = Gtk.Label(label="Summarization:")
        preset_label.set_xalign(1.0)
        preset_box.pack_start(preset_label, False, False, 0)

        self._preset_combo = Gtk.ComboBoxText()
        # 0.19.0: the preset axis is retired.  Every backend is private now,
        # so there is nothing to trade off and only the default is offered.
        # Legacy ids still resolve (see SUMMARY_PRESETS) for saved configs.
        _PRESET_OPTIONS = [
            ("confidential", "Confidential \u2014 DeepSeek V4.1 Flash (TEE)"),
        ]
        for pid, plabel in _PRESET_OPTIONS:
            self._preset_combo.append(pid, plabel)
        # Set active from CLI arg, env var, or default
        initial = self._summary_preset or "confidential"
        if not self._preset_combo.set_active_id(initial):
            self._preset_combo.set_active(0)
        def _on_preset_changed(combo):
            self._summary_preset = combo.get_active_id()

        self._preset_combo.connect("changed", _on_preset_changed)
        # Sync immediately — set_active_id() above fires before the handler
        # is connected, so self._summary_preset would stay None without this.
        self._summary_preset = self._preset_combo.get_active_id() or "confidential"
        preset_box.pack_start(self._preset_combo, True, True, 0)
        vbox.pack_start(preset_box, False, False, 0)

        # Advanced settings (collapsible) — keeps the widget compact by default
        # but lets users override ASR backend, torch device, and MLX model
        # without restarting the app.
        self._build_advanced_settings(vbox)

        # Timer
        self._timer_label = Gtk.Label(label="00:00:00")
        self._timer_label.get_style_context().add_class("timer-label")
        vbox.pack_start(self._timer_label, False, False, 0)

        # File size
        self._size_label = Gtk.Label(label="0 KB")
        self._size_label.get_style_context().add_class("size-label")
        vbox.pack_start(self._size_label, False, False, 0)

        # Status
        self._status_label = Gtk.Label(label="Ready")
        self._status_label.set_line_wrap(True)
        self._status_label.set_max_width_chars(40)
        self._status_label.get_style_context().add_class("status-label")
        vbox.pack_start(self._status_label, False, False, 4)

        # Buttons — two layouts:
        #   Idle/Done/Error: single centered "● Record" button
        #   Recording/Paused: side-by-side "⏸ Pause"/"▶ Resume" + "■ Stop"

        # Single record button (shown in idle/done/error states)
        self._record_btn = Gtk.Button(label="● Record")
        self._record_btn.get_style_context().add_class("record-btn")
        self._record_btn.connect("clicked", self._on_record_clicked)
        vbox.pack_start(self._record_btn, False, False, 4)

        # Two-button box (shown during recording/paused states)
        self._rec_btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._rec_btn_box.set_halign(Gtk.Align.CENTER)

        self._pause_btn = Gtk.Button(label="\u23f8 Pause")
        self._pause_btn.get_style_context().add_class("pause-btn")
        self._pause_btn.connect("clicked", self._on_pause_clicked)
        self._rec_btn_box.pack_start(self._pause_btn, False, False, 0)

        self._stop_btn = Gtk.Button(label="\u25a0 Stop")
        self._stop_btn.get_style_context().add_class("stop-btn")
        self._stop_btn.connect("clicked", self._on_stop_clicked)
        self._rec_btn_box.pack_start(self._stop_btn, False, False, 0)

        vbox.pack_start(self._rec_btn_box, False, False, 4)

        # Action buttons (shown after transcription completes)
        action_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        action_box.set_halign(Gtk.Align.CENTER)

        self._open_transcript_btn = Gtk.Button(label="Open Transcript")
        self._open_transcript_btn.get_style_context().add_class("action-btn")
        self._open_transcript_btn.connect("clicked", self._on_open_transcript)
        action_box.pack_start(self._open_transcript_btn, False, False, 0)

        self._open_folder_btn = Gtk.Button(label="Open Folder")
        self._open_folder_btn.get_style_context().add_class("action-btn")
        self._open_folder_btn.connect("clicked", self._on_open_folder)
        action_box.pack_start(self._open_folder_btn, False, False, 0)

        vbox.pack_start(action_box, False, False, 0)

        # Alignment model prompt (shown when model is missing)
        self._alignment_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._alignment_box.set_halign(Gtk.Align.CENTER)

        self._alignment_label = Gtk.Label()
        self._alignment_label.set_line_wrap(True)
        self._alignment_label.set_max_width_chars(35)
        self._alignment_label.get_style_context().add_class("status-label")
        self._alignment_box.pack_start(self._alignment_label, False, False, 0)

        align_btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        align_btn_box.set_halign(Gtk.Align.CENTER)

        self._download_btn = Gtk.Button(label="Download & Continue")
        self._download_btn.get_style_context().add_class("record-btn")
        self._download_btn.connect("clicked", self._on_alignment_download)
        align_btn_box.pack_start(self._download_btn, False, False, 0)

        self._skip_align_btn = Gtk.Button(label="Skip Alignment")
        self._skip_align_btn.get_style_context().add_class("action-btn")
        self._skip_align_btn.connect("clicked", self._on_alignment_skip)
        align_btn_box.pack_start(self._skip_align_btn, False, False, 0)

        self._alignment_box.pack_start(align_btn_box, False, False, 0)
        vbox.pack_start(self._alignment_box, False, False, 4)

        # Speaker labeling prompt (shown after transcription if 2+ speakers)
        self._label_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        label_header = Gtk.Label(label="Assign names to speakers:")
        label_header.set_line_wrap(True)
        label_header.set_max_width_chars(35)
        label_header.get_style_context().add_class("status-label")
        self._label_box.pack_start(label_header, False, False, 0)

        # Scrollable area for speaker rows (populated dynamically)
        self._label_scroll = Gtk.ScrolledWindow()
        self._label_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._label_scroll.set_min_content_height(60)
        self._label_scroll.set_max_content_height(200)
        self._label_rows_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._label_scroll.add(self._label_rows_box)
        self._label_box.pack_start(self._label_scroll, True, True, 0)

        label_btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        label_btn_box.set_halign(Gtk.Align.CENTER)

        self._label_apply_btn = Gtk.Button(label="Apply & Continue")
        self._label_apply_btn.get_style_context().add_class("record-btn")
        self._label_apply_btn.connect("clicked", self._on_label_apply)
        label_btn_box.pack_start(self._label_apply_btn, False, False, 0)

        self._label_skip_btn = Gtk.Button(label="Skip")
        self._label_skip_btn.get_style_context().add_class("action-btn")
        self._label_skip_btn.connect("clicked", self._on_label_skip)
        label_btn_box.pack_start(self._label_skip_btn, False, False, 0)

        self._label_box.pack_start(label_btn_box, False, False, 0)
        vbox.pack_start(self._label_box, False, False, 4)

        # Sync confirmation prompt (shown when a scheduled meeting is detected)
        self._sync_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._sync_box.set_halign(Gtk.Align.CENTER)

        self._sync_label = Gtk.Label()
        self._sync_label.set_line_wrap(True)
        self._sync_label.set_max_width_chars(35)
        self._sync_label.get_style_context().add_class("status-label")
        self._sync_box.pack_start(self._sync_label, False, False, 0)

        sync_btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        sync_btn_box.set_halign(Gtk.Align.CENTER)

        self._sync_push_btn = Gtk.Button(label="Sync to repo")
        self._sync_push_btn.get_style_context().add_class("record-btn")
        self._sync_push_btn.connect("clicked", self._on_sync_push)
        sync_btn_box.pack_start(self._sync_push_btn, False, False, 0)

        self._sync_skip_btn = Gtk.Button(label="Skip")
        self._sync_skip_btn.get_style_context().add_class("action-btn")
        self._sync_skip_btn.connect("clicked", self._on_sync_skip)
        sync_btn_box.pack_start(self._sync_skip_btn, False, False, 0)

        self._sync_box.pack_start(sync_btn_box, False, False, 0)
        vbox.pack_start(self._sync_box, False, False, 4)

        # Download progress bar (pulsing, shown during model downloads)
        self._progress_bar = Gtk.ProgressBar()
        self._progress_bar.set_pulse_step(0.05)
        vbox.pack_start(self._progress_bar, False, False, 2)

        # Background job status label (small muted text at the bottom)
        self._bg_label = Gtk.Label()
        self._bg_label.set_line_wrap(True)
        self._bg_label.set_max_width_chars(40)
        self._bg_label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        self._bg_label.get_style_context().add_class("bg-status-label")
        vbox.pack_start(self._bg_label, False, False, 0)

        self.add(vbox)
        self.connect("destroy", self._on_destroy)

        # Periodic UI update (every 500ms)
        self._poll_id = GLib.timeout_add(500, self._poll_status)

    # ── Advanced settings panel ─────────────────────────────────────────

    def _build_advanced_settings(self, parent_vbox: Gtk.Box) -> None:
        """Build a collapsible 'Advanced' panel for ASR backend / torch device /
        MLX model.  Selections are written back into ``self._transcribe_kwargs``
        so the next transcription picks them up."""
        expander = Gtk.Expander(label="Advanced")
        expander.set_expanded(False)

        grid = Gtk.Grid()
        grid.set_column_spacing(8)
        grid.set_row_spacing(4)
        grid.set_margin_top(4)
        grid.set_margin_bottom(4)

        # Helpers ----------------------------------------------------------
        def _label(text: str) -> Gtk.Label:
            lbl = Gtk.Label(label=text)
            lbl.set_halign(Gtk.Align.END)
            return lbl

        def _combo(values: list[str], current: str | None) -> Gtk.ComboBoxText:
            combo = Gtk.ComboBoxText()
            for v in values:
                combo.append_text(v)
            target = current if current in values else values[0]
            combo.set_active(values.index(target))
            return combo

        # ASR backend ------------------------------------------------------
        backend_values = ["auto", "whisperx", "mlx"]
        current_backend = self._transcribe_kwargs.get("asr_backend", "auto")
        self._asr_backend_combo = _combo(backend_values, current_backend)
        self._asr_backend_combo.connect(
            "changed",
            lambda c: self._transcribe_kwargs.__setitem__(
                "asr_backend", c.get_active_text() or "auto"
            ),
        )
        grid.attach(_label("ASR backend:"), 0, 0, 1, 1)
        grid.attach(self._asr_backend_combo, 1, 0, 1, 1)

        # Torch device -----------------------------------------------------
        # Use a sentinel "auto" entry that maps back to None so the dataclass
        # platform-detects on Apple Silicon (issue #8 behavior).
        torch_values = ["auto", "cuda", "cpu", "mps"]
        current_td = self._transcribe_kwargs.get("torch_device")
        current_td_label = current_td if current_td in torch_values else "auto"
        self._torch_device_combo = _combo(torch_values, current_td_label)

        def _on_torch_device_changed(combo):
            txt = combo.get_active_text()
            self._transcribe_kwargs["torch_device"] = None if txt == "auto" else txt

        self._torch_device_combo.connect("changed", _on_torch_device_changed)
        grid.attach(_label("Torch device:"), 0, 1, 1, 1)
        grid.attach(self._torch_device_combo, 1, 1, 1, 1)

        # MLX model --------------------------------------------------------
        self._mlx_model_entry = Gtk.Entry()
        self._mlx_model_entry.set_placeholder_text(
            "(default: mapped from --model)"
        )
        existing_mlx = self._transcribe_kwargs.get("mlx_model")
        if existing_mlx:
            self._mlx_model_entry.set_text(existing_mlx)
        self._mlx_model_entry.set_hexpand(True)

        def _on_mlx_model_changed(entry):
            text = entry.get_text().strip()
            self._transcribe_kwargs["mlx_model"] = text or None

        self._mlx_model_entry.connect("changed", _on_mlx_model_changed)
        grid.attach(_label("MLX model:"), 0, 2, 1, 1)
        grid.attach(self._mlx_model_entry, 1, 2, 1, 1)

        # Language ---------------------------------------------------------
        # Whisper auto-detect on sparse / quiet audio (long silences, short
        # opening utterances) is unreliable and can mis-classify English as
        # ja/zh, producing pages of CJK hallucinations.  Letting the user
        # set the meeting language up-front avoids that failure mode.
        # The list mirrors the alignment-models registry plus a few common
        # Whisper-supported languages.
        language_values = [
            "auto",
            "en",
            "de",
            "fr",
            "es",
            "tr",
            "fa",
            "it",
            "pt",
            "nl",
            "ja",
            "zh",
            "ko",
            "ar",
            "ru",
        ]
        current_language = self._transcribe_kwargs.get("language", "auto")
        self._language_combo = _combo(language_values, current_language)
        self._language_combo.connect(
            "changed",
            lambda c: self._transcribe_kwargs.__setitem__(
                "language", c.get_active_text() or "auto"
            ),
        )
        grid.attach(_label("Language:"), 0, 3, 1, 1)
        grid.attach(self._language_combo, 1, 3, 1, 1)

        expander.add(grid)
        parent_vbox.pack_start(expander, False, False, 0)

    # ── Button handlers ─────────────────────────────────────────────────

    def _on_record_clicked(self, _widget):
        """Handle click on the single Record button (idle/done/error states)."""
        if self._state in (_State.IDLE, _State.DONE, _State.ERROR):
            self._start_recording()

    def _on_pause_clicked(self, _widget):
        """Handle click on the Pause/Resume button."""
        if self._state == _State.RECORDING:
            self._pause_recording()
        elif self._state == _State.PAUSED:
            self._resume_recording()

    def _on_stop_clicked(self, _widget):
        """Handle click on the Stop button (recording or paused states)."""
        if self._state in (_State.RECORDING, _State.PAUSED):
            self._stop_recording()

    def _on_open_transcript(self, _widget):
        if self._last_pdf and self._last_pdf.exists():
            subprocess.Popen(["xdg-open", str(self._last_pdf)])
        elif self._last_output:
            txt_path = self._last_output.with_suffix(".txt")
            if txt_path.exists():
                subprocess.Popen(["xdg-open", str(txt_path)])

    def _on_open_folder(self, _widget):
        if self._last_output:
            folder = self._last_output.parent
            subprocess.Popen(["xdg-open", str(folder)])

    def _on_alignment_download(self, _widget):
        """User chose 'Download & Continue' for the missing alignment model."""
        self._alignment_choice = "download"
        self._alignment_box.hide()
        self.resize(300, 150)
        self.set_resizable(False)
        self._alignment_event.set()

    def _on_alignment_skip(self, _widget):
        """User chose 'Skip Alignment' for the missing alignment model."""
        self._alignment_choice = "skip"
        self._alignment_box.hide()
        self.resize(300, 150)
        self.set_resizable(False)
        self._alignment_event.set()

    def _on_label_apply(self, _widget):
        """User clicked 'Apply & Continue' on the speaker labeling dialog."""
        label_map = {}
        for sp, entry in zip(self._label_speakers, self._label_entries, strict=False):
            new_name = entry.get_text().strip()
            if new_name and new_name != sp.id:
                label_map[sp.id] = new_name
        self._label_result = label_map if label_map else None
        self._label_box.hide()
        self._cleanup_label_clips()
        self.resize(300, 150)
        self.set_resizable(False)
        self._label_event.set()

        # Update voice profiles in background with confirmed labels
        if self._label_result and self._label_audio_path:
            threading.Thread(
                target=self._update_voice_profiles,
                args=(self._label_result,),
                daemon=True,
            ).start()

    def _on_label_skip(self, _widget):
        """User clicked 'Skip' on the speaker labeling dialog."""
        self._label_result = None
        self._label_box.hide()
        self._cleanup_label_clips()
        self.resize(300, 150)
        self.set_resizable(False)
        self._label_event.set()

    def _on_sync_push(self, _widget):
        """User confirmed syncing to repo."""
        self._sync_confirmed = True
        self._sync_box.hide()
        self.resize(300, 150)
        self.set_resizable(False)
        self._sync_event.set()

    def _on_sync_skip(self, _widget):
        """User chose to skip the sync."""
        self._sync_confirmed = False
        self._sync_box.hide()
        self.resize(300, 150)
        self.set_resizable(False)
        self._sync_event.set()

    def _on_label_play(self, _widget, clip_path):
        """Play a speaker audio clip."""
        try:
            from millet.label import play_clip

            play_clip(clip_path)
            # Don't block the UI — fire and forget
        except Exception:
            pass

    def _cleanup_label_clips(self):
        """Remove temporary audio clips."""
        for clip in self._label_temp_clips:
            try:
                clip.unlink(missing_ok=True)
            except Exception:
                pass
        self._label_temp_clips.clear()

    def _update_voice_profiles(self, confirmed_label_map: dict):
        """Background task: update voice profiles with confirmed speaker labels."""
        if not self._label_audio_path or not confirmed_label_map:
            return
        try:
            # Use the in-memory pre-relabel segments (original speaker ids)
            # captured in _do_label_speakers_bg.  confirmed_label_map is keyed
            # by original id, so matching against the already-relabeled JSON on
            # disk would find nothing (and race with the job thread's save).
            from millet.voiceprint import update_profiles_from_confirmed_labels

            pre_relabel = getattr(self, "_label_pre_relabel_segments", None)
            if not pre_relabel:
                from millet.label import _find_session_files, _load_transcript

                files = _find_session_files(self._label_audio_path.parent)
                transcript_json = files.get("json")
                if not transcript_json or not transcript_json.exists():
                    return
                pre_relabel = _load_transcript(transcript_json).segments
            update_profiles_from_confirmed_labels(
                self._label_audio_path,
                pre_relabel,
                confirmed_label_map,
                self._label_channel_map,
            )
        except Exception as exc:
            _log.warning("Voice profile update failed: %s", exc)

    def _build_label_rows(self, speakers, wav_path, auto_matches=None):
        """Build the per-speaker label rows in the GTK label dialog.

        Called from GTK main thread via GLib.idle_add.

        Args:
            speakers: List of SpeakerInfo objects.
            wav_path: Path to audio for clip playback.
            auto_matches: Optional dict of speaker_id -> SpeakerMatch with
                          auto-recognized names and confidence scores.
        """
        from millet.label import extract_speaker_clip

        if auto_matches is None:
            auto_matches = {}

        # Clear previous rows
        for child in self._label_rows_box.get_children():
            self._label_rows_box.remove(child)
        self._label_entries.clear()
        self._label_temp_clips.clear()

        for sp in speakers:
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

            # Speaker ID label
            id_label = Gtk.Label(label=f"{sp.id}:")
            id_label.set_width_chars(10)
            id_label.set_xalign(1.0)
            row.pack_start(id_label, False, False, 0)

            # Text entry for new name — pre-fill if auto-recognized
            entry = Gtk.Entry()
            match = auto_matches.get(sp.id)
            if match:
                entry.set_text(match.name)
                pct = int(match.confidence * 100)
                entry.set_tooltip_text(
                    f"Auto-recognized: {match.name} ({pct}% confidence)"
                )
            else:
                entry.set_placeholder_text(sp.id)
            entry.set_width_chars(14)
            row.pack_start(entry, True, True, 0)
            self._label_entries.append(entry)

            # Play button (if audio available)
            if wav_path and wav_path.exists():
                try:
                    clip_path = extract_speaker_clip(wav_path, sp)
                    self._label_temp_clips.append(clip_path)
                    play_btn = Gtk.Button(label="Play")
                    play_btn.get_style_context().add_class("action-btn")
                    play_btn.connect("clicked", self._on_label_play, clip_path)
                    row.pack_start(play_btn, False, False, 0)
                except Exception:
                    pass

            self._label_rows_box.pack_start(row, False, False, 0)

        self._label_box.show_all()

    def _attempt_download(self, lang: str, config, session_name: str) -> bool:
        """Download alignment model with retry prompt on failure.

        Called from the background job thread.  Waits for the user to not
        be recording before showing interactive prompts.

        Returns True when ready to proceed (model downloaded or user
        chose to skip alignment).  Returns False if cancelled.
        """
        from millet.transcribe import download_alignment_model

        while True:
            GLib.idle_add(self._set_bg_status, f"{session_name}: downloading model...")
            GLib.idle_add(self._progress_bar.show)
            try:
                download_alignment_model(
                    lang,
                    progress_callback=lambda msg: GLib.idle_add(
                        self._set_bg_status, f"{session_name}: {msg}"
                    ),
                )
                GLib.idle_add(self._progress_bar.hide)
                return True  # download succeeded
            except Exception as dl_exc:
                GLib.idle_add(self._progress_bar.hide)

                if self._destroying:
                    return False

                # Wait until user is not recording before showing prompt
                self._wait_until_not_recording()
                if self._destroying:
                    return False

                # Show retry / skip prompt
                self._alignment_event.clear()
                self._alignment_choice = None
                err_text = str(dl_exc)
                if len(err_text) > 120:
                    err_text = err_text[:117] + "..."

                def _show_retry(msg=err_text):
                    self._alignment_label.set_text(f"Download failed:\n{msg}")
                    self._download_btn.set_label("Retry")
                    self._alignment_box.show_all()
                    self.set_resizable(True)
                    self.resize(300, 280)

                GLib.idle_add(_show_retry)
                self._alignment_event.wait()

                # Reset button label for future prompts
                GLib.idle_add(self._download_btn.set_label, "Download & Continue")

                if self._alignment_choice == "download":
                    continue  # retry the download
                elif self._alignment_choice == "skip":
                    config.skip_alignment = True
                    return True
                else:
                    return False  # unexpected — caller should abort

    # ── Recording lifecycle ─────────────────────────────────────────────

    def _start_recording(self):
        from millet.capture import check_prerequisites, create_session

        issues = check_prerequisites()
        if issues:
            self._set_error("Prerequisites failed: " + "; ".join(issues))
            return

        self._session = create_session(**self._capture_kwargs)
        self._session.start()
        self._error_msg = None
        self._set_state(_State.RECORDING)

    def _pause_recording(self):
        """Pause the current recording session."""
        if self._session:
            try:
                self._session.pause()
                self._set_state(_State.PAUSED)
            except Exception as exc:
                self._set_error(f"Pause failed: {exc}")

    def _resume_recording(self):
        """Resume the recording session after a pause."""
        if self._session:
            try:
                self._session.resume()
                self._set_state(_State.RECORDING)
            except Exception as exc:
                self._set_error(f"Resume failed: {exc}")

    def _stop_recording(self):
        """Start drain + stop, then enqueue post-processing as a background job."""
        was_paused = self._state == _State.PAUSED
        self._set_state(_State.DRAINING)
        self._drain_remaining = DRAIN_SECONDS
        self._worker_thread = threading.Thread(
            target=self._drain_and_enqueue,
            args=(was_paused,),
            daemon=True,
        )
        self._worker_thread.start()

    def _drain_and_enqueue(self, was_paused: bool = False):
        """Phase 1: Drain buffer, stop recording, enqueue for background processing.

        After the WAV is saved, the GUI returns to IDLE immediately so the
        user can start a new recording.
        """
        if was_paused:
            output = self._do_stop_only()
        else:
            output = self._do_drain()

        if output is None:
            # Error already shown via _set_error
            GLib.idle_add(self._set_state, _State.IDLE)
            return

        # Enqueue for background post-processing
        self._job_queue.put(output)
        self._ensure_job_thread()

        # Return to IDLE — user can immediately start a new recording
        GLib.idle_add(self._set_state, _State.IDLE)

    def _do_drain(self):
        """Drain the recording buffer, stop capture, and return the output path.

        Returns the output Path on success, or None if recording failed.
        """
        for remaining in range(DRAIN_SECONDS, 0, -1):
            self._drain_remaining = remaining
            time.sleep(1)
        self._drain_remaining = 0

        session = self._session
        output = session.stop()

        if not output.exists() or output.stat().st_size == 0:
            GLib.idle_add(self._set_error, "No audio was recorded")
            return None

        return output

    def _do_stop_only(self):
        """Stop a paused session (no drain needed) and return the output path.

        Returns the output Path on success, or None if recording failed.
        """
        session = self._session
        output = session.stop()

        if not output.exists() or output.stat().st_size == 0:
            GLib.idle_add(self._set_error, "No audio was recorded")
            return None

        return output

    # ── Background post-processing queue ────────────────────────────────

    def _ensure_job_thread(self):
        """Start the job consumer thread if it's not already running."""
        with self._job_lock:
            if self._job_thread is None or not self._job_thread.is_alive():
                self._job_thread = threading.Thread(
                    target=self._job_consumer,
                    name="meet-bg-jobs",
                    daemon=True,
                )
                self._job_thread.start()

    def _job_consumer(self):
        """Long-lived thread that processes post-recording jobs sequentially."""
        while not self._destroying:
            try:
                output = self._job_queue.get(timeout=1.0)
            except queue.Empty:
                # No more jobs — exit the thread (will be restarted if needed).
                # Take the lock and re-check emptiness so we never exit while a
                # job was enqueued concurrently: _ensure_job_thread holds the
                # same lock and sees is_alive() True only until we clear it here.
                with self._job_lock:
                    if self._job_queue.empty():
                        # Mark ourselves gone while holding the lock so a
                        # concurrent _ensure_job_thread reliably starts a
                        # fresh consumer instead of trusting a stale
                        # is_alive() on this exiting thread.
                        self._job_thread = None
                        return
                    continue

            session_name = output.parent.name
            try:
                self._process_recording(output, session_name)
            except Exception as exc:
                _log.exception("Background processing failed for %s", session_name)
                GLib.idle_add(self._set_bg_status, f"Error: {session_name}: {exc}")
            finally:
                self._job_queue.task_done()

    def _wait_until_not_recording(self):
        """Block the calling thread until the user is not actively recording.

        Used by background jobs to defer interactive dialogs (alignment,
        labeling, sync) until the user is free.
        """
        while self._state in (_State.RECORDING, _State.PAUSED, _State.DRAINING):
            if self._destroying:
                return
            time.sleep(0.5)

    def _process_recording(self, output: Path, session_name: str):
        """Background pipeline: transcribe → label → summarize → PDF → sync.

        Runs on the job consumer thread. Updates the secondary bg status
        label. Defers interactive dialogs until user is not recording.
        """
        GLib.idle_add(self._set_bg_status, f"Transcribing: {session_name}...")

        _config, transcript = self._do_transcribe_bg(output, session_name)
        if transcript is None:
            return

        transcript = self._do_label_speakers_bg(output, transcript, session_name)

        # Save transcript (or re-save with updated labels)
        transcript.save(output.parent, basename=output.stem)

        pdf_path = self._do_post_process_bg(output, transcript, session_name)
        self._do_sync_bg(output, session_name)

        # Job complete — update results for "Open" buttons
        def _on_job_done():
            self._last_output = output
            self._last_pdf = pdf_path
            # If user is idle, transition to DONE with Open buttons
            if self._state in (_State.IDLE, _State.DONE, _State.ERROR):
                self._set_state(_State.DONE)
                self._set_bg_status(None)
            else:
                # User is recording — show completion in bg label
                self._set_bg_status(f"{session_name} ready")

        GLib.idle_add(_on_job_done)

    def _do_transcribe_bg(self, output: Path, session_name: str):
        """Background-aware transcription with deferred interactive prompts.

        Returns (config, transcript) on success, or (None, None) on failure.
        """
        from millet.transcribe import (
            _LANG_NAMES,
            _MODEL_SIZES,
            ALIGNMENT_MODELS,
            AlignmentModelMissing,
            TranscriptionConfig,
            check_alignment_model_cached,
            ensure_gpu_available,
        )
        from millet.transcribe import (
            transcribe as do_transcribe,
        )

        config = TranscriptionConfig(**self._transcribe_kwargs)
        if not config.hf_token:
            GLib.idle_add(
                self._set_bg_status, f"{session_name}: HF_TOKEN not set — skipping"
            )
            return None, None

        # ── Prepare GPU (unload Ollama) ──
        GLib.idle_add(self._set_bg_status, f"{session_name}: preparing GPU...")
        ensure_gpu_available(
            progress_callback=lambda msg: GLib.idle_add(
                self._set_bg_status, f"{session_name}: {msg}"
            )
        )

        # ── Pre-flight: check alignment model cache (when language is known) ──
        preflight_lang = config.language if config.language != "auto" else None
        if preflight_lang and preflight_lang in ALIGNMENT_MODELS:
            if not config.skip_alignment and not check_alignment_model_cached(
                preflight_lang
            ):
                lang_name = _LANG_NAMES.get(preflight_lang, preflight_lang)
                size = _MODEL_SIZES.get(preflight_lang, "unknown size")
                self._alignment_lang = lang_name

                # Wait until user is not recording before showing prompt
                if self._state in (_State.RECORDING, _State.PAUSED, _State.DRAINING):
                    GLib.idle_add(
                        self._set_bg_status,
                        f"{session_name}: waiting (alignment model needed)...",
                    )
                self._wait_until_not_recording()
                if self._destroying:
                    return None, None

                self._alignment_event.clear()
                self._alignment_choice = None

                def _show_preflight_prompt():
                    self._alignment_label.set_text(
                        f"Alignment model for {lang_name} not downloaded ({size})."
                    )
                    self._alignment_box.show_all()
                    self.set_resizable(True)
                    self.resize(300, 280)

                GLib.idle_add(_show_preflight_prompt)
                self._alignment_event.wait()

                if self._alignment_choice == "download":
                    if not self._attempt_download(preflight_lang, config, session_name):
                        GLib.idle_add(
                            self._set_bg_status,
                            f"{session_name}: alignment download cancelled",
                        )
                        return None, None
                elif self._alignment_choice == "skip":
                    config.skip_alignment = True
                else:
                    GLib.idle_add(
                        self._set_bg_status,
                        f"{session_name}: alignment prompt cancelled",
                    )
                    return None, None

        # ── Transcribe (with alignment model handling) ──
        GLib.idle_add(self._set_bg_status, f"Transcribing: {session_name}...")
        transcript = None

        try:
            transcript = do_transcribe(output, config)
        except AlignmentModelMissing as exc:
            lang_name = _LANG_NAMES.get(exc.lang, exc.lang)
            size = _MODEL_SIZES.get(exc.lang, "unknown size")
            self._alignment_lang = lang_name

            # Wait until user is not recording before showing prompt
            if self._state in (_State.RECORDING, _State.PAUSED, _State.DRAINING):
                GLib.idle_add(
                    self._set_bg_status,
                    f"{session_name}: waiting (alignment model needed)...",
                )
            self._wait_until_not_recording()
            if self._destroying:
                return None, None

            self._alignment_event.clear()
            self._alignment_choice = None

            def _show_prompt():
                self._alignment_label.set_text(
                    f"Alignment model for {lang_name} not downloaded ({size})."
                )
                self._alignment_box.show_all()
                self.set_resizable(True)
                self.resize(300, 280)

            GLib.idle_add(_show_prompt)
            self._alignment_event.wait()

            if self._alignment_choice == "download":
                if not self._attempt_download(exc.lang, config, session_name):
                    GLib.idle_add(
                        self._set_bg_status,
                        f"{session_name}: alignment download cancelled",
                    )
                    return None, None

                GLib.idle_add(self._set_bg_status, f"Transcribing: {session_name}...")
                try:
                    transcript = do_transcribe(output, config)
                except Exception as retry_exc:
                    GLib.idle_add(
                        self._set_bg_status,
                        f"{session_name}: transcription failed: {retry_exc}",
                    )
                    return None, None

            elif self._alignment_choice == "skip":
                GLib.idle_add(self._set_bg_status, f"Transcribing: {session_name}...")
                config.skip_alignment = True
                try:
                    transcript = do_transcribe(output, config)
                except Exception as skip_exc:
                    GLib.idle_add(
                        self._set_bg_status,
                        f"{session_name}: transcription failed: {skip_exc}",
                    )
                    return None, None
            else:
                GLib.idle_add(
                    self._set_bg_status, f"{session_name}: alignment prompt cancelled"
                )
                return None, None

        except Exception as exc:
            GLib.idle_add(
                self._set_bg_status, f"{session_name}: transcription failed: {exc}"
            )
            return None, None

        return config, transcript

    def _do_label_speakers_bg(self, output, transcript, session_name: str):
        """Background-aware speaker labeling with deferred interactive dialog.

        Returns the (possibly relabeled) transcript.
        """
        if len(transcript.speakers) < 2:
            return transcript

        from millet.label import (
            find_session_files,
            relabel_transcript_in_memory,
        )
        from millet.label import (
            get_speakers as _get_speakers,
        )

        try:
            transcript.save(output.parent, basename=output.stem)

            spk_infos = _get_speakers(output.parent)
            if spk_infos:
                session_files = find_session_files(output.parent)
                wav_path = session_files.get("wav")

                # Build channel map: speaker_id -> 'mic' | 'system'
                from millet.audio import (
                    compute_speaker_channel_energy,
                    read_stereo_channels,
                )

                channel_map: dict[str, str] = {}
                if wav_path and wav_path.exists():
                    stereo = read_stereo_channels(wav_path)
                    if stereo is not None:
                        mic_ratio = compute_speaker_channel_energy(
                            stereo.mic,
                            stereo.system,
                            transcript.segments,
                            stereo.sample_rate,
                        )
                        for spk_id, ratio in mic_ratio.items():
                            channel_map[spk_id] = "mic" if ratio > 0.5 else "system"

                # Run voice identification against profile database
                auto_matches: dict = {}
                if wav_path and wav_path.exists():
                    try:
                        from millet.voiceprint import identify_speakers

                        auto_matches = identify_speakers(
                            wav_path,
                            transcript.segments,
                            transcript.speakers,
                            channel_map,
                        )
                    except Exception as exc:
                        _log.warning("Voice identification failed: %s", exc)

                self._label_speakers = spk_infos
                self._label_auto_matches = auto_matches
                self._label_channel_map = channel_map
                self._label_audio_path = wav_path
                # Snapshot pre-relabel segments (original speaker ids) so the
                # background profile update matches by id.  Reloading the JSON
                # in _update_voice_profiles would race with the relabeled save
                # below and match nothing.
                self._label_pre_relabel_segments = list(transcript.segments)

                GLib.idle_add(
                    self._set_bg_status, f"{session_name}: labeling speakers..."
                )

                self._label_event.clear()
                self._label_result = None

                def _show_label_dialog(
                    _spk_infos=spk_infos,
                    _wav_path=wav_path,
                    _auto_matches=auto_matches,
                ):
                    self._build_label_rows(_spk_infos, _wav_path, _auto_matches)
                    self._label_box.show_all()
                    self.set_resizable(True)
                    self.resize(340, 350)

                GLib.idle_add(_show_label_dialog)
                self._label_event.wait()

                if self._label_result:
                    transcript = relabel_transcript_in_memory(
                        transcript,
                        self._label_result,
                    )
                    import json as _json

                    session_json = session_files.get("session")
                    if session_json and session_json.exists():
                        try:
                            meta = _json.loads(session_json.read_text(encoding="utf-8"))
                            meta["speaker_labels"] = self._label_result
                            session_json.write_text(
                                _json.dumps(meta, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                        except Exception:
                            pass
        except Exception:
            pass  # labeling is optional; don't fail the pipeline

        return transcript

    def _do_post_process_bg(self, output, transcript, session_name: str):
        """Run summarization and PDF generation in background.

        Returns the PDF path if generated, or None.
        """
        from millet.transcribe import post_process

        if self._summarize:
            GLib.idle_add(self._set_bg_status, f"Summarizing: {session_name}...")

        result = post_process(
            transcript,
            output.parent,
            output.stem,
            summarize=self._summarize,
            summary_preset=self._summary_preset,
            summary_backend=self._summary_backend,
            summary_model=self._summary_model,
            ollama_singlepass=self._ollama_singlepass,
            progress_callback=lambda msg: GLib.idle_add(
                self._set_bg_status, f"{session_name}: {msg}"
            ),
        )
        return result.get("pdf")

    def _do_sync_bg(self, output: Path, session_name: str) -> None:
        """Check for scheduled meeting, prompt for sync (deferred), then sync."""
        try:
            from millet.sync import check_sync_candidate, is_sync_configured, sync_session

            if not is_sync_configured():
                return

            candidate = check_sync_candidate(output.parent)
            if candidate is None:
                return

            # Show confirmation prompt
            from millet.sync import _date_from_session

            date_str = _date_from_session(output.parent)
            self._sync_event.clear()
            self._sync_confirmed = False

            def _show_sync_prompt(_c=candidate, _d=date_str):
                self._sync_label.set_text(f"Sync to repo?\n{_c.match.name} · {_d}")
                self._sync_box.show_all()
                self.set_resizable(True)
                self.resize(300, 240)

            GLib.idle_add(_show_sync_prompt)
            self._sync_event.wait()

            if not self._sync_confirmed:
                return

            # User confirmed — push
            GLib.idle_add(self._set_bg_status, f"{session_name}: syncing...")
            sync_session(
                output.parent,
                candidate.match,
                progress_callback=lambda msg: GLib.idle_add(
                    self._set_bg_status, f"{session_name}: {msg}"
                ),
            )
        except Exception as exc:
            _log.warning("Sync failed: %s", exc)

    # ── State management ────────────────────────────────────────────────

    def _set_state(self, state):
        """Set the primary UI state (recording controls + main status)."""
        self._state = state

        # Remove all status classes
        sctx = self._status_label.get_style_context()
        for cls in (
            "status-label",
            "status-recording",
            "status-paused",
            "status-draining",
            "status-done",
            "status-error",
        ):
            sctx.remove_class(cls)

        # Remove all pause button classes
        pctx = self._pause_btn.get_style_context()
        for cls in ("pause-btn", "record-btn"):
            pctx.remove_class(cls)

        # Hide action buttons by default
        self._open_transcript_btn.hide()
        self._open_folder_btn.hide()

        # ── Button visibility: record button vs pause+stop button box ──
        if state in (_State.RECORDING, _State.PAUSED):
            self._record_btn.hide()
            self._rec_btn_box.show_all()
        else:
            self._rec_btn_box.hide()
            self._record_btn.show()

        if state == _State.IDLE:
            self._record_btn.set_sensitive(True)
            self._status_label.set_text("Ready")
            sctx.add_class("status-label")
            self._timer_label.set_text("00:00:00")
            self._size_label.set_text("0 KB")

        elif state == _State.RECORDING:
            self._pause_btn.set_label("\u23f8 Pause")
            pctx.add_class("pause-btn")
            self._pause_btn.set_sensitive(True)
            self._stop_btn.set_sensitive(True)
            self._status_label.set_text("Recording...")
            sctx.add_class("status-recording")

        elif state == _State.PAUSED:
            self._pause_btn.set_label("\u25b6 Resume")
            pctx.add_class("record-btn")
            self._pause_btn.set_sensitive(True)
            self._stop_btn.set_sensitive(True)
            self._status_label.set_text("Paused")
            sctx.add_class("status-paused")

        elif state == _State.DRAINING:
            self._record_btn.set_sensitive(False)
            self._status_label.set_text(f"Flushing buffer... {DRAIN_SECONDS}s")
            sctx.add_class("status-draining")

        elif state == _State.DONE:
            self._record_btn.set_sensitive(True)
            if self._last_output:
                # Prefer showing PDF if it exists, otherwise .txt
                txt_path = self._last_output.with_suffix(".txt")
                if self._last_pdf and self._last_pdf.exists():
                    self._status_label.set_text(f"Done — {self._last_pdf.name}")
                    self._open_transcript_btn.set_label("Open PDF")
                    self._open_transcript_btn.show()
                elif txt_path.exists():
                    self._status_label.set_text(f"Done — {txt_path.name}")
                    self._open_transcript_btn.set_label("Open Transcript")
                    self._open_transcript_btn.show()
                else:
                    self._status_label.set_text("Done — transcript saved")
                self._open_folder_btn.show()
            else:
                self._status_label.set_text("Done")
            sctx.add_class("status-done")

        elif state == _State.ERROR:
            self._record_btn.set_sensitive(True)
            self._status_label.set_text(self._error_msg or "Error")
            sctx.add_class("status-error")

    def _set_error(self, msg: str):
        self._error_msg = msg
        self._set_state(_State.ERROR)

    def _set_bg_status(self, text: str | None):
        """Update the secondary background-job status label.

        Called from GTK main thread (via GLib.idle_add).
        Pass None to clear/hide the label.
        """
        if text:
            self._bg_label.set_text(text)
            self._bg_label.show()
        else:
            self._bg_label.set_text("")
            self._bg_label.hide()

    # ── Periodic UI update ──────────────────────────────────────────────

    def _poll_status(self) -> bool:
        """Called every 500ms by GLib timer. Returns True to keep running."""
        if self._state == _State.RECORDING:
            if self._session:
                status = self._session.status()
                self._timer_label.set_text(fmt_elapsed(status.elapsed_seconds))
                self._size_label.set_text(fmt_size(status.file_size_bytes))

                if status.failed:
                    reason = status.fail_reason or "unknown error"
                    self._set_error(f"Recording failed: {reason}")

        elif self._state == _State.DRAINING:
            if self._session:
                status = self._session.status()
                self._timer_label.set_text(fmt_elapsed(status.elapsed_seconds))
                self._size_label.set_text(fmt_size(status.file_size_bytes))
            remaining = self._drain_remaining
            self._status_label.set_text(f"Flushing buffer... {remaining}s")

        elif self._state == _State.PAUSED:
            # Timer and size are frozen — no updates needed
            pass

        # Check if a background job completed while user was recording
        # and we should now show the DONE state
        if self._state == _State.IDLE and self._last_output:
            # Only transition to DONE if no background jobs are pending
            job_idle = self._job_queue.empty() and (
                self._job_thread is None or not self._job_thread.is_alive()
            )
            if job_idle:
                self._set_state(_State.DONE)
                self._set_bg_status(None)

        return True  # keep polling

    # ── Cleanup ─────────────────────────────────────────────────────────

    def _on_destroy(self, _widget):
        self._destroying = True

        if self._poll_id:
            GLib.source_remove(self._poll_id)
            self._poll_id = None

        # If still recording or paused, try to stop gracefully
        if self._session and self._state in (
            _State.RECORDING,
            _State.PAUSED,
            _State.DRAINING,
        ):
            try:
                self._session.stop()
            except Exception:
                pass

        # Unblock any background threads waiting for user input
        self._alignment_event.set()
        self._label_event.set()
        self._sync_event.set()

        Gtk.main_quit()


# ─── Public entry point ─────────────────────────────────────────────────────


def launch(
    *,
    output_dir: str | None = None,
    model: str = "large-v3-turbo",
    device: str | None = None,
    torch_device: str | None = None,
    asr_backend: str = "auto",
    mlx_model: str | None = None,
    compute_type: str = "float16",
    batch_size: int = 16,
    language: str = "auto",
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    virtual_sink: bool = False,
    mic: str | None = None,
    monitor: str | None = None,
    summarize: bool = True,
    summary_preset: str | None = None,
    summary_backend: str | None = None,
    summary_model: str | None = None,
    ollama_singlepass: bool = False,
) -> None:
    """Launch the Meet Recorder GTK3 window.

    Accepts the same options as ``meet run`` so the CLI can pass them through.
    """
    capture_kwargs = {
        "output_dir": output_dir,
        "mic": mic,
        "monitor": monitor,
        "virtual_sink": virtual_sink,
    }

    transcribe_kwargs = {
        "model": model,
        "device": device,
        "torch_device": torch_device,
        "asr_backend": asr_backend,
        "mlx_model": mlx_model,
        "compute_type": compute_type,
        "batch_size": batch_size,
        "language": language,
        "hf_token": hf_token,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
    }

    win = MeetRecorderWindow(
        capture_kwargs,
        transcribe_kwargs,
        summarize=summarize,
        summary_preset=summary_preset,
        summary_backend=summary_backend,
        summary_model=summary_model,
        ollama_singlepass=ollama_singlepass,
    )
    win.show_all()
    # Hide widgets that should only appear on demand
    win._rec_btn_box.hide()
    win._alignment_box.hide()
    win._label_box.hide()
    win._sync_box.hide()
    win._progress_bar.hide()
    win._open_transcript_btn.hide()
    win._open_folder_btn.hide()
    win._bg_label.hide()
    Gtk.main()
