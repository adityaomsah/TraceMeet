import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from tracemeet.export.bundle import ExportBundle, build_exports
from tracemeet.schemas import MeetingRecord, Transcript

RECORD_FILES = ("citation_checked_meeting_record.json", "candidate_meeting_record.json")
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
    return str(text).replace("$", "\\$")


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
        (p for p in runs_dir.iterdir() if p.is_dir() and (p / "raw_transcript.json").is_file()),
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
        if candidate.is_file():
            return candidate
    return None


def diff_markdown(raw: str, refined: str) -> str:
    a, b = raw.split(), refined.split()
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            parts.append(" ".join(a[i1:i2]))
            continue
        if i2 > i1:
            parts.append("~~" + " ".join(a[i1:i2]) + "~~")
        if j2 > j1:
            parts.append("**" + " ".join(b[j1:j2]) + "**")
    return _esc(" ".join(parts))


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

def load_run(run_dir: Path) -> RunData:
    notes: list[str] = []
    raw = load_transcript(run_dir / "raw_transcript.json")
    refined = load_transcript(run_dir / "refined_transcript.json")
    if raw is None:
        notes.append("The raw transcript could not be loaded.")
    if refined is None:
        notes.append("The refined transcript could not be loaded.")

    report = read_json(run_dir / "guard_report.json")
    if not isinstance(report, dict):
        report = None

    record_dict: Optional[dict] = None
    bundle: Optional[ExportBundle] = None
    record_path = find_record_file(run_dir)
    if record_path is None:
        notes.append("This run has no meeting record yet.")
    else:
        data = read_json(record_path)
        if not isinstance(data, dict):
            notes.append("The meeting record could not be read.")
        else:
            record_dict = data
            if raw is not None and refined is not None:
                try:
                    bundle = build_exports(
                        MeetingRecord.model_validate(data),
                        raw_transcript=raw,
                        refined_transcript=refined,
                    )
                    record_dict = bundle.record.model_dump(mode="json")
                except ValueError as exc:
                    notes.append(
                        "Exports are unavailable and the record below is unverified: "
                        f"it did not pass validation ({type(exc).__name__})."
                    )

    return RunData(raw, refined, record_dict, bundle, report, find_media(run_dir), notes)


# ---------- rendering ----------

def _render_player(media: Optional[Path]) -> None:
    st.markdown("**Source recording**")
    if media is None:
        st.caption("The recording was not saved with this run.")
        return
    if media.stat().st_size > MAX_EMBED_BYTES:
        st.caption("The saved file is too large to play in the browser.")
        return
    play = st.session_state.get("play_from")
    start = int(play[1]) if play else 0
    if play:
        st.caption(f"Playing from {fmt_time(play[1])} ({play[0]})")
    if media.suffix.lower() in VIDEO_SUFFIXES:
        st.video(str(media), start_time=start)
    else:
        st.audio(str(media), start_time=start)


def _render_evidence(item: dict, refined: Optional[Transcript], prefix: str) -> None:
    for n, ev in enumerate(evidence_list(item)):
        seg_id = str(ev.get("segment_id", ""))
        start = segment_start(refined, seg_id)
        button_col, text_col = st.columns([1, 7])
        with button_col:
            if start is not None and st.button(f"▶ {fmt_time(start)}", key=f"{prefix}_{n}"):
                st.session_state["play_from"] = (seg_id, start)
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
    for i, item in enumerate(decisions):
        with st.container(border=True):
            st.markdown(f"**{_esc(item.get('text', ''))}**  ·  `{item.get('status', 'unknown')}`")
            _render_evidence(item, refined, f"dec{i}")

    st.subheader("Action items")
    tasks = record.get("action_items") or []
    if not tasks:
        st.caption("No action items were recorded.")
    for i, item in enumerate(tasks):
        with st.container(border=True):
            st.markdown(f"**{_esc(item.get('description', ''))}**  ·  `{item.get('status', 'unknown')}`")
            st.caption(
                f"Owner: {_esc(item.get('owner') or UNSPECIFIED)}  ·  "
                f"Deadline: {_esc(item.get('deadline') or UNSPECIFIED)}"
            )
            _render_evidence(item, refined, f"task{i}")

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
    run = load_run(run_dir)

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
    with tab_transcripts:
        _render_transcripts(run.raw, run.refined)
    with tab_corrections:
        _render_corrections(run)
    with tab_downloads:
        _render_downloads(run.bundle)