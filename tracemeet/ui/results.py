import difflib
import html
import mimetypes
import json
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from tracemeet.export.bundle import ExportBundle, build_exports
from tracemeet.schemas import MeetingRecord, Transcript
from tracemeet.ui.run_labels import run_label, custom_title, save_title

RECORD_FILES = ("candidate_meeting_record.json", "citation_checked_meeting_record.json")
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm"}
UNSPECIFIED = "Unspecified"
MAX_EMBED_BYTES = 200 * 1024 * 1024
MIME_TYPES = {
    ".json": "application/json",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".txt": "text/plain",
}


@dataclass
class RunData:
    raw: Optional[Transcript]
    refined: Optional[Transcript]
    record: Optional[dict]
    bundle: Optional[ExportBundle]
    guard_report: Optional[dict]
    media: Optional[Path]
    notes: list[str] = field(default_factory=list)


# ---------- small helpers ----------

def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _esc(text: Any) -> str:
    value = str(text).replace("\\", "\\\\")
    for char in ("$", "[", "]", "*", "_", "~", "`", "<", ">", "!"):
        value = value.replace(char, "\\" + char)
    return value


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_transcript(path: Path) -> Optional[Transcript]:
    try:
        return Transcript.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def list_runs(runs_dir: Path) -> list[Path]:
    if not runs_dir.is_dir():
        return []
    return sorted(
        (p for p in runs_dir.iterdir() if p.is_dir() and ((p / "raw_transcript.json").is_file() or (p / "meta.json").is_file())),
        key=lambda p: p.name,
        reverse=True,
    )


def find_record_file(run_dir: Path) -> Optional[Path]:
    for name in RECORD_FILES:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    return None


def find_media(run_dir: Path) -> Optional[Path]:
    for candidate in sorted(run_dir.glob("input.*")):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES | {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
            return candidate
    return None


def diff_markdown(raw: str, refined: str) -> str:
    a, b = raw.split(), refined.split()
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            parts.append(_esc(" ".join(a[i1:i2])))
            continue
        if i2 > i1:
            parts.append("~~" + _esc(" ".join(a[i1:i2])) + "~~")
        if j2 > j1:
            parts.append("**" + _esc(" ".join(b[j1:j2])) + "**")
    return " ".join(parts)


def changed_segments(raw: Transcript, refined: Transcript) -> list[tuple[str, float, str, str]]:
    refined_by_id = refined.by_id()
    rows = []
    for seg in raw.segments:
        other = refined_by_id.get(seg.id)
        if other is not None and other.text != seg.text:
            rows.append((seg.id, seg.start, seg.text, other.text))
    return rows


def evidence_list(item: dict) -> list[dict]:
    evidence = item.get("evidence") or []
    if isinstance(evidence, dict):
        evidence = [evidence]
    return [e for e in evidence if isinstance(e, dict)]


def segment_start(transcript: Optional[Transcript], seg_id: str) -> Optional[float]:
    if transcript is None:
        return None
    seg = transcript.by_id().get(seg_id)
    return seg.start if seg is not None else None


def guard_rows(report: dict) -> list[dict]:
    rows = []
    for item in report.get("corrections") or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Segment": item.get("segment_id", ""),
                "Original": item.get("original", ""),
                "Replacement": item.get("replacement", ""),
                "Status": item.get("status", ""),
                "Flags": ", ".join(str(f) for f in item.get("flags") or []),
            }
        )
    return rows


# ---------- loading ----------

def _matches(path: Path, expected) -> bool:
    if not path.is_file() or not isinstance(expected, str):
        return False
    content = path.read_bytes()
    return expected.lower() in {
        hashlib.sha256(content).hexdigest(),
        hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest(),
    }


def load_run(run_dir: Path) -> RunData:
    run_dir = Path(run_dir)
    notes = []
    raw = load_transcript(run_dir / "raw_transcript.json")
    refined = load_transcript(run_dir / "refined_transcript.json")
    report = read_json(run_dir / "guard_report.json")
    report = report if isinstance(report, dict) else None
    record_dict = bundle = None
    state = read_json(run_dir / "run_state.json") or {}
    if state.get("version") == "pipeline-disk-v1" and "minutes" not in state.get("stages", {}):
        notes.append("Meeting documentation has not completed for this workflow state. Older output files are not displayed as current results.")
    else:
        path = run_dir / "candidate_meeting_record.json"
        meta = read_json(run_dir / "minutes_meta.json") or {}
        if not path.exists():
            notes.append("No candidate meeting record is available yet.")
        elif not _matches(path, meta.get("record_sha256")) or not _matches(
                run_dir / "refined_transcript.json", meta.get("source_sha256")):
            notes.append("Meeting record provenance does not match its metadata and refined transcript. Results and exports are withheld; resume processing to regenerate the affected stage.")
        elif raw is None or refined is None:
            notes.append("Both transcripts are required to display checked results and build exports.")
        elif report is None or not _matches(run_dir / "raw_transcript.json", report.get("raw_sha256")):
            notes.append("Raw transcript does not match the guard report. Results and exports are withheld.")
        elif not _matches(run_dir / "candidate_refined_transcript.json", report.get("candidate_sha256")):
            notes.append("Refinement candidate does not match the guard report. Results and exports are withheld.")
        elif report.get("refined_sha256") and not _matches(run_dir / "refined_transcript.json", report["refined_sha256"]):
            notes.append("Refined transcript does not match the guard report. Results and exports are withheld.")
        else:
            try:
                candidate = MeetingRecord.model_validate_json(path.read_text(encoding="utf-8"))
                bundle = build_exports(candidate, raw_transcript=raw, refined_transcript=refined)
                record_dict = bundle.record.model_dump(mode="json")
                if not report.get("refined_sha256"):
                    notes.append("Legacy guard report: raw/candidate and minutes-source hashes match, but the guard report has no independent refined-output hash.")
            except (ValueError, OSError) as exc:
                notes.append(f"Results failed validation ({type(exc).__name__}); exports are unavailable.")
    if raw is None:
        notes.append("The raw transcript could not be loaded.")
    if refined is None:
        notes.append("The refined transcript could not be loaded.")
    media = find_media(run_dir)
    if media is not None:
        metadata = read_json(run_dir / "meta.json") or {}
        expected = metadata.get("media_sha256")
        digest = hashlib.sha256()
        try:
            with media.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if not isinstance(expected, str) or digest.hexdigest() != expected.lower():
                notes.append("Source playback is disabled because the recording's identity cannot be verified against run metadata.")
                media = None
        except OSError:
            notes.append("The source recording could not be read.")
            media = None
    return RunData(raw, refined, record_dict, bundle, report, media, notes)


# ---------- rendering ----------

def _render_player(media: Optional[Path]) -> None:
    st.markdown("**Source recording**")
    if media is None:
        st.caption("The recording was not saved with this run.")
        return
    if media.stat().st_size > 50 * 1024 * 1024:
        if not st.checkbox("Load this large recording into the browser for playback",
                           key=f"large_media_{media.parent.name}"):
            return
    play = st.session_state.get("play_from")
    start = max(0, math.floor(play[1])) if play else 0
    if play:
        st.caption(f"Playing from {fmt_time(play[1])} ({play[0]}); segment timing, not word timing.")
        st.text(st.session_state.get("play_quote", ""))
    if media.suffix.lower() in VIDEO_SUFFIXES:
        st.video(str(media), format=mimetypes.guess_type(str(media))[0] or "video/mp4", start_time=start, end_time=max(start+1, math.ceil(play[2])) if play and len(play)>2 else None)
    else:
        st.audio(str(media), format=mimetypes.guess_type(str(media))[0] or "audio/wav", start_time=start, end_time=max(start+1, math.ceil(play[2])) if play and len(play)>2 else None)


def _select_playback(seg_id, start, end, quote=""):
    st.session_state["play_quote"] = quote
    st.session_state["play_from"] = (seg_id, start, end)


def _render_evidence(item: dict, refined: Optional[Transcript], prefix: str) -> None:
    for n, ev in enumerate(evidence_list(item)):
        seg_id = str(ev.get("segment_id", ""))
        start = segment_start(refined, seg_id)
        button_col, text_col = st.columns([1, 7])
        with button_col:
            if start is not None:
                segment = refined.by_id()[seg_id]
                st.button(f"▶ {fmt_time(start)}", key=f"{prefix}_{n}",
                          on_click=_select_playback, args=(seg_id, start, segment.end, ev.get("quote", "")))
        with text_col:
            st.caption(f"{_esc(seg_id)}: “{_esc(ev.get('quote', ''))}”")


def _render_record(record: dict, refined: Optional[Transcript]) -> None:
    st.caption(
        "Citation matching checks that quoted text exists in the cited segment. "
        "It does not prove the claim follows from it, and the overall summary is uncited."
    )

    st.subheader("Summary")
    st.write(_esc(record.get("summary") or "No summary."))

    st.subheader("Decisions")
    decisions = record.get("decisions") or []
    if not decisions:
        st.caption("No decisions were recorded.")
    for i, item in sorted(enumerate(decisions), key=lambda pair: ("decided", "proposed", "rejected", "unresolved").index(pair[1]["status"])):
        if i == next(j for j, value in enumerate(decisions) if value["status"] == item["status"]):
            st.markdown(f"**{item['status'].capitalize()}**")
        with st.container(border=True):
            st.markdown(f"**{_esc(item.get('text', ''))}**  ·  `{item.get('status', 'unknown')}`")
            _render_evidence(item, refined, f"dec{i}")

    st.subheader("Action items")
    tasks = record.get("action_items") or []
    if not tasks:
        st.caption("No action items were recorded.")
    for i, item in sorted(enumerate(tasks), key=lambda pair: pair[1]["status"] != "confirmed"):
        if i == next(j for j, value in enumerate(tasks) if value["status"] == item["status"]):
            st.markdown("**Confirmed tasks**" if item["status"] == "confirmed" else "**Tentative actions — not confirmed assignments**")
        with st.container(border=True):
            st.markdown(f"**{_esc(item.get('description', ''))}**  ·  `{item.get('status', 'unknown')}`")
            st.caption(
                f"Owner: {_esc(item.get('owner') or UNSPECIFIED)}  ·  "
                f"Deadline: {_esc(item.get('deadline') or UNSPECIFIED)}"
            )
            _render_evidence(item, refined, f"task{i}")
            with st.expander("Owner and deadline evidence"):
                for field in ("owner_evidence", "deadline_evidence"):
                    if item.get(field):
                        st.caption(field.replace("_", " ").capitalize())
                        _render_evidence({"evidence": item[field]}, refined, f"task{i}_{field}")

    st.subheader("Open questions")
    questions = record.get("open_questions") or []
    if not questions:
        st.caption("No open questions were recorded.")
    for i, item in enumerate(questions):
        with st.container(border=True):
            st.markdown(_esc(item.get("text", "")))
            _render_evidence(item, refined, f"q{i}")

    st.subheader("Minutes")
    topics = record.get("minutes") or []
    if not topics:
        st.caption("No minutes were recorded.")
    for i, item in enumerate(topics):
        with st.expander(_esc(item.get("title", f"Topic {i + 1}"))):
            st.write(_esc(item.get("summary", "")))
            _render_evidence(item, refined, f"min{i}")


def _render_transcripts(raw: Optional[Transcript], refined: Optional[Transcript]) -> None:
    if raw is None:
        st.warning("The raw transcript could not be loaded.")
        return
    refined_by_id = refined.by_id() if refined else {}
    st.dataframe(
        pd.DataFrame(
            {
                "ID": [s.id for s in raw.segments],
                "Start": [fmt_time(s.start) for s in raw.segments],
                "End": [fmt_time(s.end) for s in raw.segments],
                "Raw": [s.text for s in raw.segments],
                "Refined": [
                    refined_by_id[s.id].text if s.id in refined_by_id else "—" for s in raw.segments
                ],
            }
        ),
        width="stretch",
        hide_index=True,
    )


def _render_corrections(run: RunData) -> None:
    if run.raw is None or run.refined is None:
        st.warning("Both transcripts are needed to show corrections.")
        return

    rows = changed_segments(run.raw, run.refined)
    report = run.guard_report or {}
    left, middle, right = st.columns(3)
    left.metric("Segments changed", len(rows))
    middle.metric("Applied (guard report)", report.get("changed_segments_applied", "n/a"))
    right.metric("Held for review", report.get("segments_needing_review", "n/a"))

    st.subheader("Applied changes")
    st.caption("Struck-through words were removed and **bold** words were added.")
    for seg_id, start, raw_text, refined_text in rows:
        with st.container(border=True):
            st.caption(f"{seg_id} · {fmt_time(start)}")
            st.markdown(diff_markdown(raw_text, refined_text))

    st.subheader("Guard decisions")
    st.caption(
        "Edits marked needs_review were not applied: the raw wording was kept for the whole segment."
    )
    table = guard_rows(report)
    if table:
        st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    else:
        st.caption("No guard report was found for this run.")


def _render_downloads(bundle: Optional[ExportBundle]) -> None:
    if bundle is None:
        st.info("Downloads are unavailable until the run has a valid meeting record.")
        return
    st.download_button(
        "Download everything (ZIP)",
        data=bundle.zip_bytes,
        file_name="tracemeet_exports.zip",
        mime="application/zip",
        type="primary",
        key="dl_zip",
    )
    for name, content in bundle.files.items():
        st.download_button(
            f"Download {name}",
            data=content,
            file_name=name,
            mime=MIME_TYPES.get(Path(name).suffix, "application/octet-stream"),
            key=f"dl_{name}",
        )
    st.caption("Every file is generated from the same citation-checked record shown here.")


def render_results(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    st.subheader(_esc(run_label(run_dir)))
    st.caption(f"Run ID: {run_dir.name}")
    with st.expander("Meeting display name"):
        st.caption("Automatic labels use the first minutes topic when available. Dates use server local time.")
        with st.form(f"title_form_{run_dir.name}"):
            title = st.text_input("Custom meeting title (optional)", value=custom_title(run_dir), max_chars=120,
                                  help="Leave blank to use the automatic title. This does not change the transcript or meeting record.")
            submitted = st.form_submit_button("Save display name")
        if submitted:
            try:
                save_title(run_dir, title)
            except OSError:
                st.error("Could not save the display name. Check folder permissions and try again.")
            else:
                st.rerun()
    identity = str(Path(run_dir).resolve())
    if st.session_state.get("play_run") != identity:
        st.session_state["play_run"] = identity
        st.session_state.pop("play_from", None)
        st.session_state.pop("play_quote", None)
    run = load_run(run_dir)
    provenance = read_json(Path(run_dir) / "minutes_meta.json") or {}
    if run.record is not None:
        st.caption(f"Saved documentation: {provenance.get('provider', 'unknown provider')} / {provenance.get('model', 'unknown model')}")

    with st.sidebar:
        _render_player(run.media)

    for note in run.notes:
        st.warning(note)

    tab_record, tab_transcripts, tab_corrections, tab_downloads = st.tabs(
        ["Meeting record", "Transcripts", "Corrections", "Downloads"]
    )
    with tab_record:
        if run.record is not None:
            _render_record(run.record, run.refined)
        if run.bundle is not None:
            with st.expander("Citation audit and excluded candidate items"):
                st.json(json.loads(run.bundle.files["evidence_report.json"]))
                st.caption("Original candidate, before citation filtering:")
                st.json(read_json(Path(run_dir) / "candidate_meeting_record.json"))
        audit = read_json(Path(run_dir) / "finalization_report.json")
        if isinstance(audit, dict):
            with st.expander("Finalization audit — model review reasons, not semantic proof"):
                st.json(audit)
    with tab_transcripts:
        _render_transcripts(run.raw, run.refined)
    with tab_corrections:
        _render_corrections(run)
        _render_proposal_inspector(Path(run_dir), run)
    with tab_downloads:
        _render_downloads(run.bundle)
    _render_run_details(Path(run_dir), run)

def highlighted_text(text, edits, side):
    if side == "raw":
        start_key, end_key, tag = "raw_start", "raw_end", "del"
    else:
        start_key, end_key, tag = "candidate_start", "candidate_end", "ins"

    pieces = []
    cursor = 0

    for edit in edits:
        start, end = edit[start_key], edit[end_key]
        if not isinstance(start, int) or not isinstance(end, int) or not cursor <= start <= end <= len(text):
            raise ValueError("Invalid correction offsets")
        pieces.append(html.escape(text[cursor:start]))
        if end > start:
            pieces.append(f"<{tag}>{html.escape(text[start:end])}</{tag}>")
        cursor = end

    pieces.append(html.escape(text[cursor:]))
    return '<div style="white-space:pre-wrap">' + "".join(pieces) + "</div>"


def partial_downloads(run):
    """Completed transcript artifacts remain available if documentation fails."""
    files = {}
    if run.raw is not None:
        files["raw_transcript.json"] = run.raw.model_dump_json(indent=2)
    if run.refined is not None:
        files["refined_transcript.json"] = run.refined.model_dump_json(indent=2)
    return files


def _render_proposal_inspector(run_dir, run):
    candidate = load_transcript(run_dir / "candidate_refined_transcript.json")
    report = run.guard_report or {}
    edits = report.get("edits") or []
    if not edits or run.raw is None or candidate is None:
        return
    if not _matches(run_dir / "raw_transcript.json", report.get("raw_sha256")) or not _matches(
            run_dir / "candidate_refined_transcript.json", report.get("candidate_sha256")):
        st.warning("Correction inspector withheld: transcript hashes do not match the guard report.")
        return
    ids = list(dict.fromkeys(e["segment_id"] for e in edits))
    selected = st.selectbox("Inspect a proposed correction", ids, key=f"diff_segment_{run_dir.name}")
    chosen = [e for e in edits if e["segment_id"] == selected]
    try:
        raw_html = highlighted_text(run.raw.by_id()[selected].text, chosen, "raw")
        proposed_html = highlighted_text(candidate.by_id()[selected].text, chosen, "candidate")
    except (KeyError, TypeError, ValueError):
        st.warning("Saved edit offsets could not be validated for this segment.")
        return
    left, right = st.columns(2)
    with left:
        st.caption("Raw wording")
        st.markdown(raw_html, unsafe_allow_html=True)
    with right:
        st.caption("Model proposal — may have been withheld")
        st.markdown(proposed_html, unsafe_allow_html=True)
    if any(not edit.get("applied", False) for edit in chosen):
        st.warning("This segment was retained in its raw form. The proposal was not passed to the minutes model.")
    if run.refined and selected in run.refined.by_id():
        st.caption("Wording passed to meeting documentation")
        st.text(run.refined.by_id()[selected].text)
    st.dataframe(pd.DataFrame([{key: e.get(key) for key in
        ("original", "replacement", "status", "applied", "flags", "reason")} for e in chosen]),
        hide_index=True, width="stretch")


def _render_run_details(run_dir, run):
    state = read_json(run_dir / "run_state.json") or {}
    meta = read_json(run_dir / "meta.json") or {}
    with st.expander("Run details and transcript downloads"):
        st.json({"status": state.get("status", "CLI run"), "stage": state.get("stage"),
                 "stage_seconds": state.get("stage_seconds", {}), "attempts": state.get("attempts", []),
                 "config": state.get("config", meta.get("config", {})),
                 "terms": state.get("terms", meta.get("terms", "")),
                 "protected_names": state.get("protected_names", meta.get("protected_names", []))})
        st.caption("Saved transcript artifacts remain available even when documentation is unfinished. See integrity warnings above.")
        for name, content in partial_downloads(run).items():
            st.download_button(f"Download {name}", content, file_name=name,
                               mime="application/json", key=f"partial_{name}")
