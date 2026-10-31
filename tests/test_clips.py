"""video-clipping template: prompt wiring + clips.json snapping (millet 0.23.0)."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from millet import clips
from millet.clips import build_clips, parse_timestamp, relabel_clips_file, write_clips
from millet.frontmatter import split_body_and_data
from millet.label import _find_session_files, apply_labels
from millet.summarize import (
    MeetingSummary,
    _build_system_prompt,
    _load_user_prompt_template,
)
from millet.transcribe import Segment, Speaker, Transcript


def _segs(n: int = 40, length: float = 6.0, gap: float = 1.0, speaker="SPEAKER_00"):
    """``n`` segments of ``length`` s separated by ``gap`` s of silence."""
    out = []
    t = 3.4
    for i in range(n):
        out.append(Segment(
            start=t, end=t + length, text=f"line {i}", speaker=speaker,
            words=[{"word": f"w{i}a", "start": t, "end": t + 1.0},
                   {"word": f"w{i}b", "start": t + 1.5, "end": t + length}],
        ))
        t += length + gap
    return out


def _txn(segments, duration=None):
    speakers = [Speaker(id=s, label=s) for s in sorted({x.speaker for x in segments})]
    return Transcript(segments=segments, speakers=speakers, language="en",
                      audio_file="/srv/x/01ABC.ogg",
                      duration=duration or segments[-1].end + 2.0)


def _ts(seconds: float) -> str:
    from millet.utils import fmt_time
    return fmt_time(seconds)


def _clip(segs, first, last, kind="short", score=8, title="t", **kw):
    """A clip copied from the transcript text the way the prompt asks."""
    return {"kind": kind, "start": _ts(segs[first].start), "end": _ts(segs[last].end),
            "title": title, "hook": "h", "reason": "r", "score": score,
            "tags": ["Pricing"], **kw}


# ── prompt wiring ───────────────────────────────────────────────────────────


class TestPrompt:
    def test_system_prompt_loads_and_formats(self):
        p = _build_system_prompt("en", template="video-clipping")
        assert "## Short Clips" in p and "## Long Clips" in p
        assert '"clips"' in p
        # No unformatted placeholders / doubled braces left behind.
        assert "{none_stated}" not in p and "{lang_instruction}" not in p
        assert "{{" not in p

    def test_language_instruction_appended(self):
        assert "German" in _build_system_prompt("de", template="video-clipping")

    def test_user_prompt(self):
        tpl = _load_user_prompt_template("video-clipping")
        assert "{transcript}" in tpl and "clip" in tpl.lower()

    def test_json_block_round_trips_through_split(self):
        raw = "## Overview\nx\n\n```json\n" + json.dumps({
            "participants": [], "topics": [], "action_items": [], "decisions": [],
            "clips": [{"kind": "short", "start": "00:00:03", "end": "00:00:30"}],
        }) + "\n```"
        _, data, err = split_body_and_data(raw)
        assert err is None and data["clips"][0]["kind"] == "short"


# ── timestamps ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value,expected", [
    ("00:01:02", 62.0), ("01:02", 62.0), ("[00:01:02]", 62.0), ("1:00:00", 3600.0),
    ("00:00:01.5", 1.5), (62, 62.0), ("62.5", 62.5),
])
def test_parse_timestamp(value, expected):
    assert parse_timestamp(value) == expected


@pytest.mark.parametrize("value", [None, "", "abc", "00:61:00", -3, True, "00:00:75"])
def test_parse_timestamp_rejects(value):
    assert parse_timestamp(value) is None


# ── snapping ────────────────────────────────────────────────────────────────


class TestBuildClips:
    def test_exact_copy_snaps_to_segment_boundaries_with_padding(self):
        segs = _segs()
        doc = build_clips({"clips": [_clip(segs, 2, 5)]}, segs)
        assert doc["error"] is None and doc["dropped"] == []
        (c,) = doc["clips"]
        assert c["id"] == "short-01"
        assert c["segments"] == [2, 5]
        assert c["start_s"] == pytest.approx(segs[2].start - clips.PAD_BEFORE_S)
        assert c["end_s"] == pytest.approx(segs[5].end + clips.PAD_AFTER_S)
        assert c["duration_s"] == pytest.approx(c["end_s"] - c["start_s"])
        assert c["text"].splitlines()[0] == "SPEAKER_00: line 2"
        assert c["speakers"] == ["SPEAKER_00"]
        assert c["tags"] == ["pricing"]
        assert c["ffmpeg_args"][:2] == ["-ss", f"{c['start_s']:.3f}"]
        assert "{input}" in c["ffmpeg_args"] and "{output}" in c["ffmpeg_args"]
        # Word timings are rebased to the clip's first frame.
        assert c["words"][0]["start"] == pytest.approx(clips.PAD_BEFORE_S)

    def test_padding_never_enters_neighbour_speech(self):
        # Back-to-back segments: no silence to pad into.
        segs = _segs(gap=0.0)
        (c,) = build_clips({"clips": [_clip(segs, 2, 5)]}, segs)["clips"]
        assert c["start_s"] == pytest.approx(segs[2].start)
        assert c["end_s"] == pytest.approx(segs[5].end)

    def test_first_segment_and_end_of_recording_clamp(self):
        segs = _segs(n=6)
        doc = build_clips({"clips": [_clip(segs, 0, 5, kind="short")]}, segs,
                          duration=segs[5].end + 0.1)
        (c,) = doc["clips"]
        assert c["start_s"] >= 0
        assert c["end_s"] == pytest.approx(segs[5].end + 0.1)

    def test_mid_line_timestamp_snaps_to_containing_segment(self):
        segs = _segs()
        clip = _clip(segs, 2, 5)
        clip["start"] = _ts(segs[2].start + 3)  # mid-line
        (c,) = build_clips({"clips": [clip]}, segs)["clips"]
        assert c["segments"][0] == 2

    def test_end_copied_from_line_start_still_resolves(self):
        segs = _segs()
        clip = _clip(segs, 2, 5)
        clip["end"] = _ts(segs[5].start)  # model copied the wrong timestamp
        (c,) = build_clips({"clips": [clip]}, segs)["clips"]
        assert c["segments"][1] in (4, 5)

    def test_hallucinated_timestamp_dropped_with_reason(self):
        segs = _segs()
        clip = _clip(segs, 2, 5)
        clip["start"] = "05:00:00"
        doc = build_clips({"clips": [clip]}, segs)
        assert doc["clips"] == []
        assert "matches no transcript line" in doc["dropped"][0]["reason"]
        assert doc["error"]

    def test_end_before_start_dropped(self):
        segs = _segs()
        clip = _clip(segs, 2, 5)
        clip["start"], clip["end"] = clip["end"], clip["start"]
        doc = build_clips({"clips": [clip]}, segs)
        assert "before start" in doc["dropped"][0]["reason"]

    def test_kind_windows_enforced(self):
        segs = _segs()  # 7 s per line
        doc = build_clips({"clips": [
            _clip(segs, 1, 1, kind="short", title="tiny"),      # ~6.75 s
            _clip(segs, 2, 20, kind="short", title="huge"),     # ~133 s
            _clip(segs, 2, 20, kind="long", title="fine long"),
        ]}, segs)
        assert [c["title"] for c in doc["clips"]] == ["fine long"]
        reasons = {d["title"]: d["reason"] for d in doc["dropped"]}
        assert "short window" in reasons["tiny"] and "short window" in reasons["huge"]

    def test_unknown_kind_and_bad_entries_dropped(self):
        segs = _segs()
        doc = build_clips({"clips": [_clip(segs, 2, 5, kind="medium"), "junk",
                                     {"kind": "short", "start": "x", "end": "y"}]}, segs)
        assert doc["clips"] == []
        assert len(doc["dropped"]) == 3

    def test_same_kind_overlap_keeps_higher_score(self):
        segs = _segs()
        doc = build_clips({"clips": [
            _clip(segs, 2, 5, score=6, title="weaker"),
            _clip(segs, 4, 7, score=9, title="stronger"),
        ]}, segs)
        assert [c["title"] for c in doc["clips"]] == ["stronger"]
        assert "overlaps" in doc["dropped"][0]["reason"]

    def test_short_inside_long_is_allowed(self):
        segs = _segs()
        doc = build_clips({"clips": [
            _clip(segs, 2, 20, kind="long", title="L"),
            _clip(segs, 4, 7, kind="short", title="S"),
        ]}, segs)
        assert sorted(c["id"] for c in doc["clips"]) == ["long-01", "short-01"]

    def test_clips_sorted_by_time_and_numbered_per_kind(self):
        segs = _segs()
        doc = build_clips({"clips": [
            _clip(segs, 20, 23, score=9), _clip(segs, 2, 5, score=5),
        ]}, segs)
        assert [c["id"] for c in doc["clips"]] == ["short-01", "short-02"]
        assert doc["clips"][0]["segments"] == [2, 5]

    def test_max_clips_per_kind(self):
        segs = _segs(n=120)
        raw = [_clip(segs, i, i + 3) for i in range(0, 100, 5)]
        doc = build_clips({"clips": raw}, segs)
        assert len(doc["clips"]) == clips.MAX_CLIPS["short"]
        assert any("more than" in d["reason"] for d in doc["dropped"])

    def test_score_clamped(self):
        segs = _segs()
        (c,) = build_clips({"clips": [_clip(segs, 2, 5, score=42)]}, segs)["clips"]
        assert c["score"] == 10

    def test_no_data_is_an_explicit_error(self):
        doc = build_clips(None, _segs(), data_error="fenced JSON parse error: x")
        assert doc["clips"] == [] and "parse error" in doc["error"]

    def test_data_without_clips_list(self):
        doc = build_clips({"topics": []}, _segs())
        assert doc["clips"] == [] and "no 'clips' list" in doc["error"]

    def test_empty_clips_list_is_not_an_error(self):
        doc = build_clips({"clips": []}, _segs())
        assert doc["clips"] == [] and doc["error"] is None

    def test_no_top_level_duration_key(self):
        # vezir's duration probe reads a top-level "duration" from any *.json.
        assert "duration" not in build_clips({"clips": []}, _segs())


# ── writing + relabel ───────────────────────────────────────────────────────


def _summary(data, data_error=None):
    return MeetingSummary(markdown="## Overview\nx", model="m", elapsed_seconds=0.1,
                          backend="tinfoil", template="video-clipping",
                          data=data, data_error=data_error)


def test_write_clips(tmp_path):
    segs = _segs()
    t = _txn(segs)
    path = write_clips(_summary({"clips": [_clip(segs, 2, 5)]}), t, tmp_path, "01ABC")
    assert path.name == "01ABC.clips.json"
    doc = json.loads(path.read_text())
    assert doc["source"]["audio"] == "01ABC.ogg"  # name only, never the path
    assert doc["source"]["transcript"] == "01ABC.json"
    assert doc["summary"] == {"backend": "tinfoil", "model": "m"}
    assert len(doc["clips"]) == 1
    lang = write_clips(_summary({"clips": []}), t, tmp_path, "01ABC", lang_suffix="de")
    assert lang.name == "01ABC.clips.de.json"
    assert not list(tmp_path.glob(".*.tmp"))


def test_relabel_clips_file(tmp_path):
    segs = _segs()
    path = write_clips(_summary({"clips": [_clip(segs, 2, 5)]}), _txn(segs), tmp_path, "s")
    before = json.loads(path.read_text())["clips"][0]
    assert relabel_clips_file(path, {"SPEAKER_00": "Kemal"})
    after = json.loads(path.read_text())["clips"][0]
    assert after["speakers"] == ["Kemal"]
    assert after["text"].startswith("Kemal: line 2")
    assert after["start_s"] == before["start_s"]
    assert not relabel_clips_file(path, {"NOBODY": "x"})


# ── label.py integration ────────────────────────────────────────────────────


class TestApplyLabels:
    def _session(self, tmp_path):
        sdir = tmp_path / "01ABC"
        sdir.mkdir()
        segs = _segs()
        _txn(segs).save(sdir, basename="01ABC")
        return sdir, segs

    def test_regenerate_writes_clips(self, tmp_path):
        sdir, segs = self._session(tmp_path)
        fake = _summary({"clips": [_clip(segs, 2, 5)]})
        with patch("millet.summarize.is_backend_available", return_value=True), \
             patch("millet.summarize.summarize", return_value=fake), \
             patch("millet.transcribe.ensure_gpu_available", lambda *a, **k: None):
            files = apply_labels(sdir, {"SPEAKER_00": "Kemal"}, regenerate_summary=True,
                                 summary_template="video-clipping")
        assert files["clips"] == sdir / "01ABC.clips.json"
        doc = json.loads(files["clips"].read_text())
        assert doc["clips"][0]["speakers"] == ["Kemal"]
        assert (sdir / "01ABC.video-clipping.md").exists()

    def test_other_templates_write_no_clips(self, tmp_path):
        sdir, segs = self._session(tmp_path)
        fake = _summary({"clips": [_clip(segs, 2, 5)]})
        with patch("millet.summarize.is_backend_available", return_value=True), \
             patch("millet.summarize.summarize", return_value=fake), \
             patch("millet.transcribe.ensure_gpu_available", lambda *a, **k: None):
            apply_labels(sdir, {}, regenerate_summary=True, summary_template="iteration-plan")
        assert not (sdir / "01ABC.clips.json").exists()

    def test_relabel_without_regenerate_renames_inside_clips(self, tmp_path):
        sdir, segs = self._session(tmp_path)
        write_clips(_summary({"clips": [_clip(segs, 2, 5)]}), _txn(segs), sdir, "01ABC")
        apply_labels(sdir, {"SPEAKER_00": "Kemal"}, regenerate_summary=False)
        doc = json.loads((sdir / "01ABC.clips.json").read_text())
        assert doc["clips"][0]["speakers"] == ["Kemal"]

    def test_clips_json_never_mistaken_for_transcript(self, tmp_path):
        # "<base>.clips.json" sorts before "<base>.json".  Use a session dir
        # whose name differs from the basename so the by-name rescue can't help.
        sdir = tmp_path / "meeting-dir"
        sdir.mkdir()
        segs = _segs()
        _txn(segs).save(sdir, basename="a")
        write_clips(_summary({"clips": []}), _txn(segs), sdir, "a")
        (sdir / "a.video-clipping.meta.json").write_text("{}")
        assert _find_session_files(sdir)["json"].name == "a.json"


# ── sync naming ─────────────────────────────────────────────────────────────


def test_sync_pushes_clips_under_own_name(tmp_path):
    from millet.sync import _collect_files

    sdir = tmp_path / "s"
    sdir.mkdir()
    for name in ("m.json", "m.clips.json", "m.clips.de.json", "m.video-clipping.md",
                 "m.video-clipping.meta.json"):
        (sdir / name).write_text("{}")
    dests = {src.name: dest for src, dest in _collect_files(sdir)}
    assert dests["m.json"] == "transcript.json"
    assert dests["m.clips.json"] == "clips.json"
    assert dests["m.clips.de.json"] == "clips.de.json"
    assert dests["m.video-clipping.md"] == "video-clipping.md"
    assert "m.video-clipping.meta.json" not in dests
