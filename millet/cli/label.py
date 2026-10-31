"""millet label command."""
from __future__ import annotations

import json
from pathlib import Path

import click


def weak_match_reason(match) -> str | None:
    """Return a human reason a voiceprint match should NOT auto-apply, else None.

    A match at/above MATCH_THRESHOLD is still suppressed when it looks like a
    false positive:

      * **Thin cluster** — less than ``MATCH_MIN_SPEECH_SECONDS`` of embeddable
        speech (noisy embedding).
      * **Ambiguous** — neither a strong absolute score
        (``>= MATCH_AUTOAPPLY_CONFIDENCE``) nor a clear margin over the
        runner-up profile (``>= MATCH_AUTOAPPLY_MARGIN``).

    ``evidence_seconds`` (default 0.0) and ``margin`` (default 1.0) are
    "trustworthy" sentinels, so matches constructed positionally by older
    callers/tests are never gated.
    """
    from millet.voiceprint import (
        MATCH_AUTOAPPLY_CONFIDENCE,
        MATCH_AUTOAPPLY_MARGIN,
        MATCH_MIN_SPEECH_SECONDS,
    )

    ev = getattr(match, "evidence_seconds", 0.0)
    margin = getattr(match, "margin", 1.0)
    if ev != 0.0 and ev < MATCH_MIN_SPEECH_SECONDS:
        return f"only {ev:.1f}s of speech < {MATCH_MIN_SPEECH_SECONDS:.0f}s"
    if (match.confidence < MATCH_AUTOAPPLY_CONFIDENCE
            and margin < MATCH_AUTOAPPLY_MARGIN):
        return (
            f"ambiguous ({match.confidence:.0%} < "
            f"{MATCH_AUTOAPPLY_CONFIDENCE:.0%} and margin {margin:.2f} < "
            f"{MATCH_AUTOAPPLY_MARGIN:.2f})"
        )
    return None


def _write_autoid_sidecar(json_path: Path, auto_matches: dict) -> None:
    """Write voiceprint auto-ID suggestions next to the transcript JSON.

    Produces ``<session>.autoid.json``:
        {
          "version": 1,
          "suggestions": {
            "SPEAKER_00": {"name": "Nancy", "confidence": 0.76},
            ...
          }
        }

    Keyed by the *original* speaker id (SPEAKER_NN / REMOTE_N / YOU) so a
    labeling UI can pre-fill names regardless of whether the match was
    applied to the transcript.  Best-effort; callers swallow errors.
    """
    sidecar = json_path.with_name(json_path.stem + ".autoid.json")
    payload = {
        "version": 1,
        "suggestions": {
            spk_id: {
                "name": match.name,
                "confidence": round(float(match.confidence), 4),
            }
            for spk_id, match in auto_matches.items()
        },
    }
    sidecar.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _apply_json_mode(
    *,
    session_path: Path,
    apply_json: str,
    no_summary: bool,
    summary_preset: str | None,
    summary_backend: str | None,
    summary_model: str | None,
    summary_language: str | None,
    summary_template: str | None,
    ollama_singlepass: bool,
    update_profiles: bool,
    team_profiles_path: Path | None,
) -> None:
    """Non-interactive label application from a JSON map.

    This is the subprocess boundary for embedders (vezir): apply a
    speaker->name map, regenerate outputs, and optionally update the
    voiceprint DB from the applied labels — all without prompting,
    audio playback, or any dependence on the caller's terminal.

    Exit codes: 0 on success, 1 on failure (message on stderr).  A
    profile-update failure is non-fatal (warning only), matching the
    long-standing embedder behavior.
    """
    from millet.label import (
        _detect_speaker_channels,
        _find_session_files,
        _load_transcript,
        apply_labels,
    )

    # Load the label map: {"labels": {...}} envelope or a plain map.
    try:
        if apply_json == "-":
            import sys as _sys

            raw = json.loads(_sys.stdin.read() or "{}")
        else:
            raw = json.loads(Path(apply_json).read_text(encoding="utf-8") or "{}")
    except Exception as exc:
        click.echo(f"Error: could not parse label JSON: {exc}", err=True)
        raise SystemExit(1) from None

    if isinstance(raw, dict) and isinstance(raw.get("labels"), dict):
        raw = raw["labels"]
    if not isinstance(raw, dict):
        click.echo(
            "Error: label JSON must be an object map or {'labels': {...}}",
            err=True,
        )
        raise SystemExit(1)

    label_map: dict[str, str] = {}
    for spk_id, name in raw.items():
        if isinstance(name, str) and name.strip():
            label_map[str(spk_id)] = name.strip()

    regenerate_summary = not no_summary
    if not label_map and not regenerate_summary and not summary_language:
        click.echo("No labels to apply and no summary regeneration requested.")
        return

    # Snapshot pre-relabel segments (original speaker ids) before apply_labels
    # rewrites the transcript, so the id-keyed profile update below matches.
    pre_relabel_segments = None
    pre_relabel_speakers = None
    if update_profiles and label_map:
        try:
            _pre_files = _find_session_files(session_path)
            _pre_json = _pre_files.get("json")
            if _pre_json and _pre_json.exists():
                _pre_t = _load_transcript(_pre_json)
                pre_relabel_segments = list(_pre_t.segments)
                pre_relabel_speakers = _pre_t.speakers
        except Exception:
            pass

    click.echo(f"Applying {len(label_map)} label(s) (non-interactive)...")
    try:
        result_files = apply_labels(
            session_path,
            label_map=label_map,
            regenerate_summary=regenerate_summary,
            summary_preset=summary_preset,
            summary_backend=summary_backend,
            summary_model=summary_model,
            summary_language=summary_language,
            summary_template=summary_template,
            ollama_singlepass=ollama_singlepass,
            progress_callback=lambda msg: click.echo(f"  {msg}"),
        )
    except Exception as exc:
        click.echo(f"Error: apply failed: {exc}", err=True)
        raise SystemExit(1) from None

    click.echo("Updated files:")
    for fmt, path in result_files.items():
        click.echo(f"  {fmt}: {path}")

    if update_profiles and label_map:
        click.echo("Updating voice profiles from confirmed labels...")
        try:
            from millet.voiceprint import update_profiles_from_confirmed_labels

            files = _find_session_files(session_path)
            wav_path = files.get("wav")
            json_path = files.get("json")
            if wav_path and json_path and wav_path.exists():
                if pre_relabel_segments is not None:
                    prof_segments = pre_relabel_segments
                    prof_speakers = pre_relabel_speakers
                else:
                    reloaded = _load_transcript(json_path)
                    prof_segments = reloaded.segments
                    prof_speakers = reloaded.speakers
                channel_map = _detect_speaker_channels(
                    wav_path, prof_segments, prof_speakers,
                )
                update_profiles_from_confirmed_labels(
                    wav_path,
                    prof_segments,
                    label_map,
                    channel_map,
                    profiles_path=team_profiles_path,
                )
                click.echo("  Voice profiles updated.")
            else:
                click.echo(
                    "  Skipping profile update (audio or transcript missing).",
                    err=True,
                )
        except Exception as exc:
            # Non-fatal by contract: labels are already applied.
            click.echo(f"  Warning: profile update failed: {exc}", err=True)


@click.command()
@click.argument("session_dir", type=click.Path(exists=True))
@click.option(
    "--no-audio",
    is_flag=True,
    default=False,
    help="Skip audio playback (just show text samples)",
)
@click.option(
    "--no-summary",
    is_flag=True,
    default=False,
    help="Skip summary regeneration (use find-and-replace on existing summary)",
)
@click.option(
    "--auto",
    is_flag=True,
    default=False,
    help="Auto-label using voice profiles. Confident matches are applied "
    "without prompting; unrecognized speakers are prompted interactively.",
)
@click.option(
    "--summary-preset",
    type=click.Choice(["high-quality", "confidential", "alternative"], case_sensitive=False),
    default=None,
    help="DEPRECATED (removed in 0.21.0): all presets now resolve to the default private backend. Kept so existing scripts keep working.",
)
@click.option(
    "--summary-backend",
    type=click.Choice(
        ["tinfoil", "ollama"], case_sensitive=False
    ),
    default=None,
    help="Summary backend (default: tinfoil, or MILLET_SUMMARY_BACKEND env var)",
)
@click.option(
    "--summary-model",
    type=str,
    default=None,
    help="Model for summary (default: per-backend, or MILLET_SUMMARY_MODEL env var)",
)
@click.option(
    "--ollama-singlepass",
    is_flag=True,
    default=False,
    help="Use the legacy single-pass Ollama flow instead of the default two-pass (extract+format) flow. The two-pass flow is more accurate on local 20B-class models but adds one extra LLM call. Also configurable via MILLET_OLLAMA_SINGLEPASS=1.",
)
@click.option(
    "--team",
    type=str,
    default=None,
    help="Use this team's voiceprint DB (~/.config/meet/<team>/"
         "speaker_profiles.json) for auto-label matching and profile "
         "updates, instead of the global DB.",
)
@click.option(
    "--apply-json",
    "apply_json",
    type=click.Path(exists=True, dir_okay=False, allow_dash=True),
    default=None,
    help="Non-interactive mode: apply the speaker->name map from a JSON "
         "file ({'labels': {...}} or a plain object; '-' reads stdin) and "
         "regenerate outputs. No prompting, no audio. An empty map with "
         "summary regeneration enabled just re-runs the summary+PDF step. "
         "Intended for embedders (vezir) that previously imported millet "
         "in-process.",
)
@click.option(
    "--summary-language",
    type=str,
    default=None,
    help="Generate the (re)generated summary in this language, saved as an "
         "additional <base>.summary.<lang>.md next to the primary summary.",
)
@click.option(
    "--summary-template",
    type=str,
    default=None,
    help="Summary prompt template name ('iteration-plan' or 'video-clipping') for the "
         "regenerated summary. Writes <base>.<template>.md instead of "
         "<base>.summary.md. Also configurable via MILLET_SUMMARY_TEMPLATE.",
)
@click.option(
    "--update-profiles",
    is_flag=True,
    default=False,
    help="After applying labels from --apply-json, update the voiceprint "
         "DB from the applied (human-confirmed) labels. Non-fatal on "
         "failure. Uses --team's DB when given.",
)
def label(session_dir, no_audio, no_summary, auto, summary_preset, summary_backend, summary_model, ollama_singlepass, team, apply_json, summary_language, summary_template, update_profiles):
    """Assign real names to speakers in a transcribed session.

    \b
    SESSION_DIR is the path to a meet recording session directory.

    For each speaker detected in the transcript, plays a short audio clip
    (from the appropriate channel) and prompts you to enter a name.
    Press Enter to keep the current label unchanged.

    With --auto, speaker voice profiles are used to automatically identify
    known speakers. Confident matches are applied without prompting.
    Unrecognized speakers are still prompted interactively.

    After labeling, all outputs (txt, srt, json, summary, pdf) are
    regenerated with the new speaker names.

    \b
    Examples:
        meet label ~/meet-recordings/meeting-20260313-214133
        meet label ~/meet-recordings/meeting-20260313-214133 --no-audio
        meet label ~/meet-recordings/meeting-20260313-214133 --auto
        meet label ~/meet-recordings/meeting-20260313-214133 --auto --no-summary
    """
    from millet import paths
    from millet.label import (
        _detect_speaker_channels,
        _find_session_files,
        _load_transcript,
        apply_labels,
        extract_speaker_clip,
        get_speakers,
        play_clip,
    )

    # Resolve the team's voiceprint DB once; None => global default.
    team_profiles_path = paths.profiles_path(team) if team else None

    session_path = Path(session_dir)

    # ── Non-interactive apply mode (embedders: vezir) ──
    if apply_json is not None:
        _apply_json_mode(
            session_path=session_path,
            apply_json=apply_json,
            no_summary=no_summary,
            summary_preset=summary_preset,
            summary_backend=summary_backend,
            summary_model=summary_model,
            summary_language=summary_language,
            summary_template=summary_template,
            ollama_singlepass=ollama_singlepass,
            update_profiles=update_profiles,
            team_profiles_path=team_profiles_path,
        )
        return
    files = _find_session_files(session_path)

    if "json" not in files:
        click.echo(f"Error: no transcript JSON found in {session_path}", err=True)
        click.echo("  Run 'meet transcribe' on this session first.", err=True)
        raise SystemExit(1)

    # Get speaker info
    try:
        speakers = get_speakers(session_path)
    except FileNotFoundError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from None

    if not speakers:
        click.echo("No speakers found in this session.")
        return

    if len(speakers) == 1:
        click.echo(f"Only one speaker found: {speakers[0].id}")
        click.echo("You can still assign a name if you like.")
        click.echo()

    click.echo(f"Session: {session_path.name}")
    click.echo(f"Speakers found: {len(speakers)}")
    click.echo()

    # ── Voice profile auto-identification ──
    auto_matches: dict = {}
    transcript = None
    wav_path = files.get("wav")
    channel_map: dict[str, str] = {}

    if auto and wav_path and wav_path.exists():
        click.echo("Running voice identification against speaker profiles...")
        transcript = _load_transcript(files["json"])
        channel_map = _detect_speaker_channels(
            wav_path,
            transcript.segments,
            transcript.speakers,
        )

        try:
            from millet.voiceprint import identify_speakers, load_profiles

            profiles = load_profiles(profiles_path=team_profiles_path)
            if not profiles:
                click.echo("  No speaker profiles found. Run 'millet enroll' first.")
                click.echo("  Falling back to interactive labeling.")
                click.echo()
            else:
                click.echo(f"  {len(profiles)} voice profiles loaded.")
                auto_matches = identify_speakers(
                    wav_path,
                    transcript.segments,
                    transcript.speakers,
                    channel_map,
                    profiles_path=team_profiles_path,
                )
                if auto_matches:
                    click.echo(
                        f"  Identified {len(auto_matches)}/{len(speakers)} speakers:"
                    )
                    for spk_id, match in sorted(auto_matches.items()):
                        click.echo(
                            f"    {spk_id} -> {click.style(match.name, fg='green')}"
                            f"  (confidence: {match.confidence:.2f})"
                        )
                else:
                    click.echo("  No confident matches found.")
                click.echo()
        except Exception as exc:
            click.echo(f"  Voice identification failed: {exc}", err=True)
            click.echo("  Falling back to interactive labeling.")
            click.echo()

    # Show summary table
    click.echo(
        f"  {'#':<4} {'Label':<14} {'Channel':<10} {'Segments':<10} {'Auto-ID':<20} {'Sample Text'}"
    )
    click.echo(f"  {'─' * 4} {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 20} {'─' * 40}")
    for i, sp in enumerate(speakers, 1):
        auto_name = ""
        if sp.id in auto_matches:
            m = auto_matches[sp.id]
            auto_name = f"{m.name} ({m.confidence:.0%})"
        click.echo(
            f"  {i:<4} {sp.id:<14} {sp.channel:<10} {sp.segment_count:<10} {auto_name:<20} {sp.sample_text[:40]}"
        )
    click.echo()

    # ── Build label map ──
    can_play = not no_audio and wav_path and wav_path.exists()

    if not can_play and not no_audio:
        click.echo("  (No WAV file found — skipping audio playback)")
        click.echo()

    label_map: dict[str, str] = {}
    temp_clips: list[Path] = []

    # Gate weak matches: a barely-confident score is untrustworthy when the
    # cluster is thin (little real speech) OR the match is ambiguous (small
    # margin over the runner-up profile).  pyannote backchannel splits produce
    # exactly this signature — e.g. a 0.69 match with a 0.13 margin on a noisy
    # cluster, which mis-names a phantom speaker.  Such matches are NOT
    # auto-applied; the speaker stays raw and routes to needs_labeling for a
    # human.  Strong, well-separated matches (e.g. 0.93 / 0.40 margin) apply.
    weak_matches = {
        spk_id for spk_id, m in auto_matches.items()
        if weak_match_reason(m) is not None
    }
    if weak_matches:
        for spk_id in sorted(weak_matches):
            m = auto_matches[spk_id]
            click.echo(
                f"  Skipping weak match {spk_id} -> {m.name} "
                f"({weak_match_reason(m)}) — leaving unlabeled."
            )
        click.echo()

    # A ghost REMOTE bucket (filler crosstalk from several people) wins over
    # any voiceprint match: there is no single voice in it to match.  Worse,
    # its stretched-filler segments are mostly room tone and the scribe's own
    # voice echoing back through the call, so a profile that once learned
    # them (vezir 2026-09: "Pattern", ~0.86 to the scribe's echo) matches
    # every such bucket at ~0.9 and keeps it out of CROSSTALK.  Replay data:
    # 1 in 120 ghost-shaped buckets was ever one real person.
    ghost_ids: set[str] = set()
    if auto and transcript is not None:
        from millet.label import crosstalk_label_map

        ghost_ids = set(crosstalk_label_map(transcript, set()))
        for spk_id in sorted(ghost_ids & set(auto_matches)):
            click.echo(
                f"  Ignoring voiceprint match {spk_id} -> "
                f"{auto_matches[spk_id].name} (filler crosstalk, not one voice)."
            )

    # Separate speakers into auto-matched (applicable) and unrecognized.
    # Weak matches count as unrecognized so they stay raw.
    applied_matches = {
        spk_id: m for spk_id, m in auto_matches.items()
        if spk_id not in weak_matches and spk_id not in ghost_ids
    }
    unrecognized = [sp for sp in speakers if sp.id not in applied_matches]

    # Apply auto-matched labels directly
    if auto and applied_matches:
        click.echo("Auto-applying confident voice matches:")
        for sp in speakers:
            if sp.id in applied_matches:
                match = applied_matches[sp.id]
                label_map[sp.id] = match.name
                click.echo(
                    f"  {sp.id} -> {click.style(match.name, fg='green')}  ({match.confidence:.0%})"
                )
        click.echo()

    # Raw ids whose label was *derived* here (CROSSTALK, tiny-noise folds)
    # rather than voiceprint-matched or typed by a human.  They must never
    # feed the voiceprint DB: a fold is a guess and CROSSTALK is mixed audio.
    synthetic_ids: set[str] = set()

    # NOTE: both passes below take the *raw* ids already resolved — the keys
    # of label_map.  The transcript still carries raw ids at this point, so
    # passing the names (label_map.values()) matches nothing: that bug made
    # the 0.12.12 REMOTE rescue a silent no-op in production, and let the
    # tiny-noise fold overwrite a short-but-confident voiceprint match.

    # ── Ghost REMOTE bucket → CROSSTALK ──
    # The dual-diarize path leaves a literal REMOTE bucket of sub-second
    # fillers ("Bye.", "Yeah.") from several people that no voiceprint can
    # match.  Rather than guess an owner, label it CROSSTALK: honest, readable,
    # and no longer an unresolved speaker, so it can't force needs_labeling.
    # Needs no named target; a substantial REMOTE stays raw for a human.
    if auto and transcript is not None:
        from millet.label import crosstalk_label_map

        crosstalk = crosstalk_label_map(transcript, set(label_map))
        if crosstalk:
            click.echo("Labeling unattributable filler crosstalk:")
            for raw_id, name in sorted(crosstalk.items()):
                label_map[raw_id] = name
                synthetic_ids.add(raw_id)
                click.echo(
                    f"  {raw_id} -> {click.style(name, fg='yellow')}  "
                    f"(cannot be assigned to a speaker)"
                )
            click.echo()
            unrecognized = [sp for sp in speakers if sp.id not in label_map]

    # ── Fold spurious TINY noise clusters into the dominant speaker ──
    # A backchannel one-liner or distorted blip becomes its own raw cluster
    # that voiceprint can't match.  Fold it into whoever speaks most (even
    # another raw id), so a single noise blip can't force an otherwise-clean
    # session into needs_labeling.  Runs in auto mode even when no confident
    # voiceprint match was applied.
    if auto and transcript is not None:
        from millet.label import absorb_tiny_speakers

        tiny_absorb = absorb_tiny_speakers(transcript, set(label_map))
        if tiny_absorb:
            click.echo("Folding tiny noise speaker(s) into the dominant speaker:")
            for raw_id, target in sorted(tiny_absorb.items()):
                label_map[raw_id] = label_map.get(target, target)
                synthetic_ids.add(raw_id)
                click.echo(
                    f"  {raw_id} -> {click.style(label_map[raw_id], fg='green')}  "
                    f"(tiny-noise-absorbed)"
                )
            click.echo()
            unrecognized = [sp for sp in speakers if sp.id not in label_map]

    # Interactive labeling for unrecognized speakers (or all speakers if not --auto)
    speakers_to_prompt = unrecognized if auto else speakers

    # Don't prompt when there's no interactive terminal (e.g. the vezir worker
    # runs `label --auto --no-audio` with no stdin).  Previously click.prompt
    # hit EOF and raised Abort, which discarded the confident auto-matches that
    # had already been collected into label_map.  In a non-interactive context
    # we apply the auto-matches and leave unmatched speakers as their raw
    # SPEAKER_N / REMOTE_N ids (the documented "unknowns remain as REMOTE_N"
    # behavior), so the session can still route to needs_labeling for a human.
    import sys

    interactive = sys.stdin.isatty()
    if speakers_to_prompt and not interactive:
        if auto and unrecognized:
            click.echo(
                f"{len(unrecognized)} unrecognized speaker(s) left unlabeled "
                "(non-interactive); applying auto-matches only."
            )
            click.echo()
        speakers_to_prompt = []

    if speakers_to_prompt:
        if auto and unrecognized:
            click.echo(
                f"{len(unrecognized)} unrecognized speaker(s) — prompting interactively:"
            )
            click.echo()

        try:
            for i, sp in enumerate(speakers_to_prompt, 1):
                click.echo(
                    f"Speaker {i}/{len(speakers_to_prompt)}: {click.style(sp.id, bold=True)}"
                )
                click.echo(f"  Channel: {sp.channel}  |  Segments: {sp.segment_count}")
                click.echo(f'  Sample:  "{sp.sample_text}"')

                # Play audio clip
                if can_play:
                    try:
                        clip_path = extract_speaker_clip(wav_path, sp)
                        temp_clips.append(clip_path)
                        click.echo("  Playing audio clip... ", nl=False)
                        proc = play_clip(clip_path)
                        proc.wait()
                        click.echo("done")
                    except Exception as exc:
                        click.echo(f"  (Audio playback failed: {exc})")

                # Prompt for name
                new_name = click.prompt(
                    f"  Enter name for {sp.id} (Enter to keep)",
                    default="",
                    show_default=False,
                ).strip()

                if new_name and new_name != sp.id:
                    label_map[sp.id] = new_name
                    click.echo(f"  {sp.id} -> {click.style(new_name, fg='green')}")
                else:
                    click.echo(f"  Keeping: {sp.id}")
                click.echo()

        finally:
            # Clean up temp clips
            for clip in temp_clips:
                try:
                    clip.unlink(missing_ok=True)
                except Exception:
                    pass

    if not label_map:
        click.echo("No labels changed. Nothing to do.")
        return

    click.echo("Applying labels:")
    for old, new in sorted(label_map.items()):
        click.echo(f"  {old} -> {new}")
    click.echo()

    # Snapshot the pre-relabel segments (original speaker ids) BEFORE
    # apply_labels rewrites the transcript JSON with the confirmed names.
    # The profile update below matches segments by original id; reloading
    # the transcript afterwards would yield relabeled segments and silently
    # match nothing.
    pre_relabel_segments = list(transcript.segments) if transcript is not None else None

    # Apply labels and regenerate outputs
    regenerate_summary = not no_summary

    result_files = apply_labels(
        session_path,
        label_map,
        regenerate_summary=regenerate_summary,
        summary_preset=summary_preset,
        summary_backend=summary_backend,
        summary_model=summary_model,
        summary_template=summary_template,
        ollama_singlepass=ollama_singlepass,
        progress_callback=lambda msg: click.echo(f"  {msg}"),
    )

    click.echo()
    click.echo("Updated files:")
    for fmt, path in result_files.items():
        click.echo(f"  {fmt}: {path}")

    # ── Persist auto-id suggestions sidecar ──
    # Record each auto-matched speaker's name + confidence, keyed by the
    # speaker id AS IT APPEARS IN THE FINAL TRANSCRIPT (the applied name when
    # a match was applied, else the original id).  vezir's labeling screen
    # reads this to show match confidence and pre-fill recognized names.
    if auto and applied_matches:
        try:
            final_suggestions = {
                label_map.get(spk_id, spk_id): match
                for spk_id, match in applied_matches.items()
            }
            _write_autoid_sidecar(files["json"], final_suggestions)
        except Exception as exc:
            click.echo(f"  (Could not write auto-id sidecar: {exc})", err=True)

    # ── Update voice profiles with confirmed labels ──
    # Only update profiles for speakers that were manually confirmed by the
    # user (typed interactively), NOT for auto-matched speakers.  Auto-matches
    # can be wrong (especially when the mic channel is very quiet), and
    # updating profiles from incorrect matches causes profile drift.
    if auto and label_map:
        # manual_labels: speakers the user typed a name for during this session
        # (never derived labels: CROSSTALK / tiny-noise folds are not a person
        # confirmed by a human).
        manual_labels = {
            sp_id: name
            for sp_id, name in label_map.items()
            if sp_id not in auto_matches and sp_id not in synthetic_ids
        }
        profile_labels = manual_labels
        if not profile_labels:
            click.echo()
            click.echo(
                "  Skipping profile update (no manually confirmed labels; "
                "use 'meet enroll' to update profiles from verified labels)."
            )
        else:
            click.echo()
            click.echo("Updating voice profiles with manually confirmed labels...")
            try:
                from millet.voiceprint import update_profiles_from_confirmed_labels

                # Use the pre-relabel segments (original speaker ids) so the
                # id-keyed profile_labels actually match segments.  Fall back
                # to loading the (already-relabeled) transcript only when no
                # snapshot exists — in that case profile_labels keys won't
                # match, but we avoid crashing.
                if pre_relabel_segments is not None:
                    prof_segments = pre_relabel_segments
                    prof_speakers = transcript.speakers
                else:
                    reloaded = _load_transcript(files["json"])
                    prof_segments = reloaded.segments
                    prof_speakers = reloaded.speakers
                # Rebuild channel_map if not already done
                if not channel_map:
                    channel_map = _detect_speaker_channels(
                        wav_path,
                        prof_segments,
                        prof_speakers,
                    )
                update_profiles_from_confirmed_labels(
                    wav_path,
                    prof_segments,
                    profile_labels,
                    channel_map,
                    profiles_path=team_profiles_path,
                )
                click.echo("  Voice profiles updated.")
            except Exception as exc:
                click.echo(f"  Profile update failed: {exc}", err=True)
