"""Tests for millet.frontmatter — schema v1 build / parse / validate / render."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from millet.frontmatter import (
    SCHEMA_VERSION,
    FrontmatterContext,
    FrontmatterValidationError,
    build_frontmatter,
    context_from_transcript,
    empty_frontmatter,
    parse_frontmatter_block,
    render_frontmatter_block,
    split_body_and_data,
    validate_frontmatter,
    write_frontmatter_sidecar,
)

# ─── split_body_and_data ───────────────────────────────────────────────────


class TestSplitBodyAndData:
    def test_fenced_json_at_end(self):
        raw = (
            "## Overview\nA short meeting.\n\n"
            "## Action Items\n* Send doc — Alice\n\n"
            '```json\n{"action_items": [{"task": "Send doc", "assignee": "Alice", "status": "open"}]}\n```\n'
        )
        body, data, err = split_body_and_data(raw)
        assert err is None
        assert data is not None
        assert data["action_items"][0]["task"] == "Send doc"
        # body should NOT include the JSON fence
        assert "```" not in body
        assert "## Overview" in body

    def test_bare_fenced_block_at_end(self):
        raw = (
            "## Overview\nFoo.\n\n"
            '```\n{"topics": ["A", "B"], "decisions": []}\n```\n'
        )
        body, data, err = split_body_and_data(raw)
        assert err is None
        assert data == {"topics": ["A", "B"], "decisions": []}
        assert "```" not in body

    def test_no_json_block_returns_error(self):
        raw = "## Overview\nMeeting happened.\n## Decisions\n* X over Y."
        body, data, err = split_body_and_data(raw)
        assert data is None
        assert err == "no JSON block found"
        assert body == raw  # unchanged

    def test_malformed_fenced_json(self):
        raw = "Body.\n\n```json\n{not valid json}\n```\n"
        body, data, err = split_body_and_data(raw)
        assert data is None
        assert err and "parse error" in err
        # body should still have the JSON stripped — we recovered the body
        assert body == "Body."

    def test_trailing_object_without_fence(self):
        raw = (
            "## Overview\nHi.\n\n"
            '{"action_items": [], "decisions": [], "topics": ["t1"]}'
        )
        body, data, err = split_body_and_data(raw)
        assert err is None
        assert data is not None
        assert data["topics"] == ["t1"]

    def test_trailing_object_without_known_keys_is_left_in_body(self):
        # Don't eat unrelated JSON-looking content from the body.
        raw = "Some output\n\n{\"foo\": 1, \"bar\": 2}"
        body, data, err = split_body_and_data(raw)
        assert data is None
        assert err == "no JSON block found"
        assert body == raw

    def test_empty_input(self):
        body, data, err = split_body_and_data("")
        assert data is None
        assert err == "empty completion"

    def test_trailing_object_recovered_despite_earlier_brace_line(self):
        # Regression: a greedy match anchored at the FIRST line-leading '{'
        # would span the body's example object + the real trailing block,
        # fail json.loads, and discard valid trailing data.
        raw = (
            "## Overview\n"
            '{"example": "not the data block"}\n'
            "Some more body text.\n\n"
            '{"action_items": [], "decisions": [], "topics": ["real"]}'
        )
        body, data, err = split_body_and_data(raw)
        assert err is None
        assert data is not None
        assert data["topics"] == ["real"]
        assert "Some more body text." in body


# ─── build_frontmatter ─────────────────────────────────────────────────────


class TestBuildFrontmatter:
    def test_minimal_context(self):
        ctx = FrontmatterContext(title="Sync", duration_seconds=60)
        fm = build_frontmatter(None, ctx)
        validate_frontmatter(fm)
        assert fm["schema_version"] == SCHEMA_VERSION
        assert fm["title"] == "Sync"
        assert fm["duration"] == "PT1M"
        assert fm["action_items"] == []
        assert fm["decisions"] == []
        assert fm["topics"] == []
        assert fm["participants"] == []
        assert fm["type"] == "meeting"
        assert fm["source"] == {"session_id": None, "audio_sha256": None}

    def test_crosstalk_is_never_a_participant(self):
        ctx = FrontmatterContext(
            transcript_speakers=["Nora", "CROSSTALK"],
            speaker_channels={"CROSSTALK": "system"},
        )
        fm = build_frontmatter({"participants": ["crosstalk", "Pia"]}, ctx)
        validate_frontmatter(fm)
        assert [p["name"] for p in fm["participants"]] == ["Nora", "Pia"]

    def test_with_extracted_data(self):
        ctx = FrontmatterContext(
            title="Pricing",
            language="en",
            duration_seconds=42 * 60 + 17,
            transcript_speakers=["YOU", "REMOTE_1"],
            speaker_channels={"YOU": "mic", "REMOTE_1": "system"},
        )
        extracted = {
            "topics": ["pricing", "onboarding"],
            "action_items": [
                {"task": "Send doc", "assignee": "Alice", "due": "Friday"},
                {"task": "Review competitor grid"},  # no assignee/due
            ],
            "decisions": [
                {"text": "Run experiment at $99/mo", "topic": "pricing"},
            ],
            "participants": [{"name": "Alice"}],
        }
        fm = build_frontmatter(extracted, ctx)
        validate_frontmatter(fm)
        assert fm["duration"] == "PT42M17S"
        # transcript participants come first, with channels;
        # alice from extraction is appended, no channel.
        names = [p["name"] for p in fm["participants"]]
        assert "YOU" in names
        assert "REMOTE_1" in names
        assert "Alice" in names
        ai = fm["action_items"]
        assert len(ai) == 2
        assert ai[0]["assignee"] == "Alice"
        assert ai[0]["due"] == "Friday"
        assert ai[0]["status"] == "open"
        assert ai[1]["assignee"] is None
        assert ai[1]["due"] is None

    def test_invalid_status_normalizes_to_open(self):
        ctx = FrontmatterContext()
        fm = build_frontmatter(
            {"action_items": [{"task": "x", "status": "weird"}]}, ctx,
        )
        assert fm["action_items"][0]["status"] == "open"

    def test_extraction_error_is_recorded(self):
        ctx = FrontmatterContext()
        fm = build_frontmatter(None, ctx, extraction_error="bad json")
        assert fm["extraction_error"] == "bad json"
        validate_frontmatter(fm)

    def test_empty_frontmatter_is_valid(self):
        ctx = FrontmatterContext()
        fm = empty_frontmatter(ctx, error="oops")
        validate_frontmatter(fm)
        assert fm["extraction_error"] == "oops"


# ─── validate_frontmatter ──────────────────────────────────────────────────


class TestValidate:
    def _good(self) -> dict:
        return build_frontmatter(None, FrontmatterContext())

    def test_good_passes(self):
        validate_frontmatter(self._good())

    def test_wrong_schema_version_rejected(self):
        fm = self._good()
        fm["schema_version"] = 2
        with pytest.raises(FrontmatterValidationError):
            validate_frontmatter(fm)

    def test_unknown_type_rejected(self):
        fm = self._good()
        fm["type"] = "not-a-type"
        with pytest.raises(FrontmatterValidationError):
            validate_frontmatter(fm)

    def test_missing_required_list_rejected(self):
        fm = self._good()
        fm["action_items"] = None
        with pytest.raises(FrontmatterValidationError):
            validate_frontmatter(fm)

    def test_action_item_without_task_rejected(self):
        fm = self._good()
        fm["action_items"] = [{"task": "", "status": "open"}]
        with pytest.raises(FrontmatterValidationError):
            validate_frontmatter(fm)


# ─── YAML render + parse round-trip ────────────────────────────────────────


class TestYamlRoundtrip:
    def test_roundtrip_minimal(self):
        ctx = FrontmatterContext(title="Hi", date="2026-04-25T10:00:00+00:00")
        fm = build_frontmatter(None, ctx)
        block = render_frontmatter_block(fm)
        assert block.startswith("---\n")
        assert block.endswith("---\n")
        parsed, body = parse_frontmatter_block(block + "## Body\n")
        assert parsed is not None
        assert parsed["title"] == "Hi"
        assert parsed["schema_version"] == SCHEMA_VERSION
        assert parsed["action_items"] == []
        assert body == "## Body\n"

    def test_roundtrip_with_lists_of_dicts(self):
        ctx = FrontmatterContext(
            title="Pricing",
            transcript_speakers=["YOU", "REMOTE_1"],
            speaker_channels={"YOU": "mic", "REMOTE_1": "system"},
        )
        extracted = {
            "topics": ["pricing"],
            "action_items": [
                {"task": "Send doc: with colon", "assignee": "Alice"},
            ],
            "decisions": [{"text": "Use $99/mo", "topic": "pricing"}],
        }
        fm = build_frontmatter(extracted, ctx)
        block = render_frontmatter_block(fm)
        parsed, _ = parse_frontmatter_block(block)
        assert parsed is not None
        assert parsed["topics"] == ["pricing"]
        assert parsed["action_items"][0]["task"] == "Send doc: with colon"
        assert parsed["action_items"][0]["assignee"] == "Alice"
        assert parsed["decisions"][0]["text"] == "Use $99/mo"
        assert parsed["decisions"][0]["topic"] == "pricing"
        # participants survived with channel
        names = {p["name"]: p["channel"] for p in parsed["participants"]}
        assert names["YOU"] == "mic"
        assert names["REMOTE_1"] == "system"

    def test_no_frontmatter_returns_none(self):
        parsed, body = parse_frontmatter_block("## Just a body\nnothing here")
        assert parsed is None
        assert body == "## Just a body\nnothing here"

    def test_quoted_strings_with_special_chars(self):
        ctx = FrontmatterContext(title='He said "hi" — and left')
        fm = build_frontmatter(None, ctx)
        block = render_frontmatter_block(fm)
        parsed, _ = parse_frontmatter_block(block)
        assert parsed is not None
        assert parsed["title"] == 'He said "hi" — and left'


# ─── Injection resistance (security audit 2026-10-30, M-6) ─────────────────

_INJECTIONS = [
    "Alice\n---\n# injected heading",
    "Bob\n- injected: item",
    "Carol\nevil_key: evil_value",
    "Dave\r\n---\r\n",
    "tab\there",
    "nul\x00byte",
    "bell\x07 and esc\x1b[31m",
    "line\u2028separator",
    "para\u2029separator",
    "nel\x85next",
    "trailing backslash\\",
    "backslash-n literal \\n stays text",
    "plain ünïcödé — ok",
]


class TestFrontmatterInjection:
    @pytest.mark.parametrize("name", _INJECTIONS)
    def test_participant_name_cannot_break_the_block(self, name):
        ctx = FrontmatterContext(
            title="T",
            transcript_speakers=[name],
            speaker_channels={name: "system"},
        )
        fm = build_frontmatter(None, ctx)
        block = render_frontmatter_block(fm)
        # The block is exactly one YAML document: opening and closing fence,
        # and no raw line of the body equals a fence or starts a new top-level
        # key/list item that we did not emit.
        lines = block.splitlines()
        assert lines[0] == "---" and lines[-1] == "---"
        assert lines[1:-1].count("---") == 0
        assert not any(line.startswith(("# injected", "evil_key:")) for line in lines)
        assert not any("\x00" in line or "\x07" in line or "\x1b" in line for line in lines)
        parsed, _ = parse_frontmatter_block(block)
        assert parsed is not None
        names = [p["name"] for p in parsed["participants"]]
        assert names == [name]  # exact logical round-trip

    @pytest.mark.parametrize("value", _INJECTIONS)
    def test_llm_fields_round_trip_exactly(self, value):
        ctx = FrontmatterContext(title=value)
        extracted = {
            "topics": [value],
            "action_items": [{"task": value, "assignee": value}],
            "decisions": [{"text": value, "topic": value}],
        }
        fm = build_frontmatter(extracted, ctx)
        block = render_frontmatter_block(fm)
        parsed, _ = parse_frontmatter_block(block)
        assert parsed is not None
        assert parsed["title"] == value
        # Extracted fields are whitespace-stripped by the normalizers (that is
        # pre-existing, not escaping); the logical content must survive exactly.
        assert parsed["topics"] == [value.strip()]
        assert parsed["action_items"][0]["task"] == value.strip()
        assert parsed["decisions"][0]["text"] == value.strip()

    @pytest.mark.parametrize("value", _INJECTIONS)
    def test_fallback_parser_round_trips_too(self, value, monkeypatch):
        """Without PyYAML the bespoke parser must decode the same escapes."""
        import builtins

        real_import = builtins.__import__

        def no_yaml(name, *a, **k):
            if name == "yaml":
                raise ImportError("yaml disabled for test")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_yaml)
        ctx = FrontmatterContext(title=value)
        fm = build_frontmatter(None, ctx)
        block = render_frontmatter_block(fm)
        parsed, _ = parse_frontmatter_block(block)
        assert parsed is not None
        assert parsed["title"] == value

    def test_participant_name_cannot_inject_key_via_yaml_safe_load(self):
        name = "x\nschema_version: 999"
        ctx = FrontmatterContext(title="T", transcript_speakers=[name])
        fm = build_frontmatter(None, ctx)
        parsed, _ = parse_frontmatter_block(render_frontmatter_block(fm))
        assert parsed["schema_version"] == SCHEMA_VERSION


# ─── context_from_transcript ───────────────────────────────────────────────


class TestContextFromTranscript:
    def test_no_session_dir(self, tmp_path):
        transcript = SimpleNamespace(
            speakers=[
                SimpleNamespace(id="YOU", label="YOU"),
                SimpleNamespace(id="REMOTE_1", label="Alice"),
            ],
            language="en",
            duration=120.0,
        )
        ctx = context_from_transcript(transcript, tmp_path)
        assert ctx.language == "en"
        assert ctx.duration_seconds == 120.0
        # YOU stays as YOU; REMOTE_1 was renamed to Alice (but still mapped to system)
        assert "YOU" in ctx.transcript_speakers
        assert "Alice" in ctx.transcript_speakers
        assert ctx.speaker_channels.get("YOU") == "mic"
        assert ctx.speaker_channels.get("Alice") == "system"

    def test_reads_session_json(self, tmp_path):
        transcript = SimpleNamespace(
            speakers=[SimpleNamespace(id="YOU", label="YOU")],
            language="de",
            duration=60.0,
        )
        (tmp_path / "meeting.session.json").write_text(
            json.dumps(
                {
                    "started_at": "2026-04-25T14:00:00+00:00",
                    "title": "Standup",
                }
            ),
            encoding="utf-8",
        )
        ctx = context_from_transcript(transcript, tmp_path)
        assert ctx.title == "Standup"
        assert ctx.date == "2026-04-25T14:00:00+00:00"


# ─── sidecar writing ───────────────────────────────────────────────────────


def test_write_frontmatter_sidecar(tmp_path):
    ctx = FrontmatterContext(title="X")
    fm = build_frontmatter(None, ctx)
    path = write_frontmatter_sidecar(tmp_path, "meeting-1", fm)
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["title"] == "X"
