import html
import json
import logging
import math
import mimetypes
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from tracemeet.audio.validate import AudioValidationError
from tracemeet.config import ConfigError, get_api_key, load_config
from tracemeet.llm.base import LLMError
from tracemeet.llm.gemini import GeminiProvider
from tracemeet.pipeline import (
    continue_run,
    create_run,
    settings_signature,
)
from tracemeet.schemas import UNSPECIFIED
from tracemeet.stages.minutes import MinutesError
from tracemeet.stages.refine import RefinementError
from tracemeet.stt.local_whisper import LocalWhisper, TranscriptionError

AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg"]
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm"]
RUNS_DIR = Path(__file__).resolve().parent / "runs"
MAX_PREVIEW_BYTES = 50 * 1024 * 1024

st.set_page_config(page_title="TraceMeet", page_icon="🎙️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_transcriber(model_name, device, compute_type):
    return LocalWhisper(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
    )


def clear_recording_results():
    for key in ("tm_run", "tm_error", "tm_source"):
        st.session_state.pop(key, None)


def fmt_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return (
        f"{hours}:{minutes:02d}:{seconds:02d}"
        if hours
        else f"{minutes:02d}:{seconds:02d}"
    )


def new_run_dir():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    directory = RUNS_DIR / run_id
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def save_upload(uploaded, directory):
    path = directory / f"input{Path(uploaded.name).suffix.lower()}"
    uploaded.seek(0)
    try:
        with path.open("wb") as destination:
            shutil.copyfileobj(uploaded, destination, length=1024 * 1024)
    finally:
        uploaded.seek(0)
    return path


def choose_source(segment_id, quote):
    st.session_state["tm_source"] = (segment_id, quote)


def show_citations(evidence_list, prefix):
    for index, evidence in enumerate(evidence_list):
        st.text(evidence.quote)
        st.button(
            f"Listen · {evidence.segment_id}",
            key=f"{prefix}_{index}",
            on_click=choose_source,
            args=(evidence.segment_id, evidence.quote),
        )


def transcript_table(transcript):
    return pd.DataFrame(
        [
            {
                "ID": segment.id,
                "Start": fmt_time(segment.start),
                "End": fmt_time(segment.end),
                "Text": segment.text,
            }
            for segment in transcript.segments
        ]
    )


def highlighted_text(text, edits, side):
    if side == "raw":
        start_key, end_key, tag = "raw_start", "raw_end", "del"
    else:
        start_key, end_key, tag = "candidate_start", "candidate_end", "ins"

    pieces = []
    cursor = 0

    for edit in edits:
        start, end = edit[start_key], edit[end_key]
        pieces.append(html.escape(text[cursor:start]))
        if end > start:
            pieces.append(f"<{tag}>{html.escape(text[start:end])}</{tag}>")
        cursor = end

    pieces.append(html.escape(text[cursor:]))
    return '<div style="white-space:pre-wrap">' + "".join(pieces) + "</div>"


def execute(run, key):
    provider = None
    st.session_state.pop("tm_error", None)

    with st.status("Processing recording…", expanded=True) as status:
        progress = st.progress(0.0, text="Preparing…")

        def report(message):
            status.update(label=message)
            st.write(message)

        try:
            provider = GeminiProvider(key)
            stt_cfg = run.cfg["stt"]

            continue_run(
                run,
                provider=provider,
                engine_factory=lambda: get_transcriber(
                    stt_cfg["model"],
                    stt_cfg["device"],
                    stt_cfg["compute_type"],
                ),
                on_status=report,
                on_progress=lambda value: progress.progress(
                    max(0.0, min(float(value), 1.0)),
                    text=f"Transcription audio position: {value:.0%}",
                ),
            )
            status.update(label="Processing complete", state="complete")

        except (
            AudioValidationError,
            TranscriptionError,
            ConfigError,
            LLMError,
            RefinementError,
            MinutesError,
            OSError,
            ValueError,
        ) as exc:
            st.session_state["tm_error"] = str(exc)
            status.update(label=f"Stopped: {run.stage}", state="error")

        except Exception:
            logging.exception("Unexpected TraceMeet pipeline failure")
            st.session_state["tm_error"] = (
                "An unexpected error occurred. Completed stages were retained. "
                "Check the terminal for details."
            )
            status.update(label=f"Stopped: {run.stage}", state="error")

        finally:
            progress.empty()
            if provider is not None:
                try:
                    provider.close()
                except Exception:
                    logging.exception("Could not close Gemini client")


st.title("TraceMeet")
st.caption("Meeting records you can inspect and trace to the recording.")

uploaded = st.file_uploader(
    "Upload an English meeting recording",
    type=AUDIO_TYPES + VIDEO_TYPES,
    key="meeting_upload",
    on_change=clear_recording_results,
)

if uploaded is None:
    st.info("Upload an English meeting recording to begin.")
    st.stop()

if uploaded.size == 0:
    st.error("This file is empty. Please upload a valid recording.")
    st.stop()

suffix = Path(uploaded.name).suffix.lower()
is_video = suffix.lstrip(".") in VIDEO_TYPES
st.text(f"{uploaded.name} · {uploaded.size / (1024 * 1024):.2f} MiB")

with st.expander("Preview uploaded recording", expanded=False):
    if uploaded.size <= MAX_PREVIEW_BYTES:
        uploaded.seek(0)
        if is_video:
            st.video(uploaded)
        else:
            st.audio(uploaded, format=uploaded.type or "audio/wav")
        uploaded.seek(0)
    else:
        st.info(
            "Automatic preview is skipped above 50 MiB. "
            "Source playback can be loaded after processing."
        )

terms = st.text_input(
    "Names and technical terms for transcription/refinement (optional)",
    placeholder="Aditya Om Sah, IIT Guwahati, Kubernetes",
    help="Contextual hints only; these do not prove that a term was spoken.",
).strip()

names_text = st.text_input(
    "Protected participant names (optional; separate with commas)",
    placeholder="Aditya Om Sah, Alice",
    help="Changes involving these names are flagged for review.",
).strip()
names = [name.strip() for name in names_text.split(",") if name.strip()]

try:
    loaded = load_config()
    # Store only the settings used by this workflow, never environment secrets.
    cfg = {
        "stt": {
            field: loaded["stt"][field]
            for field in ("model", "device", "compute_type")
        },
        "llm": {
            field: loaded["llm"][field]
            for field in ("refine_model", "minutes_model", "temperature")
        },
    }
    current_signature = settings_signature(cfg, terms, names)
except (ConfigError, OSError, KeyError, ValueError) as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

run = st.session_state.get("tm_run")
settings_changed = run is not None and run.signature != current_signature

if settings_changed:
    st.warning(
        "These results belong to earlier settings, prompts, or code. "
        "Start a new run to apply the changes."
    )

start_col, retry_col = st.columns(2)

with start_col:
    start_clicked = st.button(
        "Process recording" if run is None else "Start new run",
        type="primary",
    )

with retry_col:
    can_retry = (
        run is not None
        and run.status != "complete"
        and not settings_changed
    )
    retry_label = (
        "Retry meeting documentation"
        if run is not None
        and run.guarded is not None
        and run.candidate_record is None
        else "Retry unfinished stages"
    )
    retry_clicked = st.button(retry_label, disabled=not can_retry)

if start_clicked:
    try:
        key = get_api_key()
        clear_recording_results()

        directory = new_run_dir()
        media_path = save_upload(uploaded, directory)
        run = create_run(
            directory, media_path, uploaded.name, cfg, terms, names
        )
        st.session_state["tm_run"] = run
        execute(run, key)

    except (ConfigError, LLMError, OSError, ValueError) as exc:
        st.session_state["tm_error"] = str(exc)

    st.rerun()

elif retry_clicked:
    try:
        execute(run, get_api_key())
    except ConfigError as exc:
        st.session_state["tm_error"] = str(exc)

    st.rerun()

if st.session_state.get("tm_error"):
    st.error(st.session_state["tm_error"])

run = st.session_state.get("tm_run")
if run is None:
    st.stop()

st.caption(
    f"Run {run.directory.name} · {run.status} · "
    f"STT: {run.cfg['stt']['model']} · "
    f"Refinement: {run.cfg['llm']['refine_model']} · "
    f"Minutes: {run.cfg['llm']['minutes_model']}"
)

# Citation callbacks update selection before this script reruns.
selection = st.session_state.get("tm_source")
if selection and run.guarded is not None:
    segment_id, quote = selection
    segment = run.guarded.transcript.by_id().get(segment_id)

    if segment is not None:
        st.subheader("Source playback")
        st.caption(
            f"{segment.id} · {segment.start:.2f}–{segment.end:.2f}s. "
            "Playback covers the source segment, not word-level timing."
        )
        st.text(quote)

        load_large = True
        if run.media_path.stat().st_size > MAX_PREVIEW_BYTES:
            load_large = st.checkbox(
                "Load this large recording into the browser for playback",
                key=f"large_media_{run.directory.name}",
            )

        if load_large:
            mime = mimetypes.guess_type(str(run.media_path))[0]
            start = max(0, math.floor(segment.start))
            end = max(start + 1, math.ceil(segment.end))

            if run.media_path.suffix.lstrip(".") in VIDEO_TYPES:
                st.video(
                    str(run.media_path),
                    format=mime or "video/mp4",
                    start_time=start,
                    end_time=end,
                )
            else:
                st.audio(
                    str(run.media_path),
                    format=mime or "audio/wav",
                    start_time=start,
                    end_time=end,
                )

            st.caption(
                "Press Play if playback does not start. "
                "If the browser cannot play this format, use a WAV or "
                "browser-compatible MP4 recording."
            )

if run.raw is not None:
    st.divider()
    raw_tab, refined_tab, edits_tab = st.tabs(
        ["Raw transcript", "Guarded refined transcript", "Corrections"]
    )

    with raw_tab:
        st.dataframe(
            transcript_table(run.raw), width="stretch", hide_index=True
        )

    with refined_tab:
        if run.guarded is None:
            st.info("Refinement and guard checks have not completed.")
        else:
            st.dataframe(
                transcript_table(run.guarded.transcript),
                width="stretch",
                hide_index=True,
            )

    with edits_tab:
        if run.guarded is None:
            st.info("The correction report is not ready.")
        elif not run.guarded.edits:
            st.info("No transcript changes were proposed.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Segment": edit["segment_id"],
                            "Original": edit["original"],
                            "Proposed": edit["replacement"],
                            "Status": edit["status"],
                            "Applied": edit["applied"],
                            "Flags": ", ".join(edit["flags"]),
                        }
                        for edit in run.guarded.edits
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

            changed_ids = list(
                dict.fromkeys(edit["segment_id"] for edit in run.guarded.edits)
            )
            selected_id = st.selectbox(
                "Inspect a proposed correction",
                changed_ids,
                key=f"diff_segment_{run.directory.name}",
            )
            segment_edits = [
                edit for edit in run.guarded.edits
                if edit["segment_id"] == selected_id
            ]

            left, right = st.columns(2)
            with left:
                st.caption("Raw wording")
                st.markdown(
                    highlighted_text(
                        run.raw.by_id()[selected_id].text,
                        segment_edits,
                        "raw",
                    ),
                    unsafe_allow_html=True,
                )
            with right:
                st.caption("Model proposal — may have been withheld")
                st.markdown(
                    highlighted_text(
                        run.candidate.by_id()[selected_id].text,
                        segment_edits,
                        "candidate",
                    ),
                    unsafe_allow_html=True,
                )

            if any(not edit["applied"] for edit in segment_edits):
                st.warning(
                    "This segment was retained in its raw form. "
                    "The proposal was not passed to the minutes model."
                )

if run.checked is not None:
    record = run.checked.record
    prefix = run.directory.name

    st.divider()
    st.subheader("Meeting documentation")
    st.info(
        "Citations were checked against the refined transcript. "
        "This does not verify claim meaning or speaker attribution."
    )

    st.markdown("**Overview — uncited**")
    st.write(record.summary)

    st.markdown("**Topic minutes**")
    for index, topic in enumerate(record.minutes):
        with st.expander(topic.title, expanded=True):
            st.write(topic.summary)
            show_citations(topic.evidence, f"{prefix}_topic_{index}")
    if not record.minutes:
        st.info("No topic minutes remain after citation checks.")

    st.markdown("**Decisions and discussion outcomes**")
    for status_name in ("decided", "proposed", "rejected", "unresolved"):
        items = [
            (index, item)
            for index, item in enumerate(record.decisions)
            if item.status.value == status_name
        ]
        if items:
            st.markdown(f"**{status_name.capitalize()}**")
            for index, item in items:
                st.write(item.text)
                with st.expander("Sources"):
                    show_citations(item.evidence, f"{prefix}_decision_{index}")

    if not record.decisions:
        st.info("No decisions were extracted and retained.")

    st.markdown("**Action items**")
    for status_name, heading in (
        ("confirmed", "Confirmed tasks"),
        ("tentative", "Tentative actions — not confirmed assignments"),
    ):
        items = [
            (index, item)
            for index, item in enumerate(record.action_items)
            if item.status.value == status_name
        ]
        if items:
            st.markdown(f"**{heading}**")
            for index, item in items:
                st.write(item.description)
                st.caption(
                    f"Owner: {item.owner or UNSPECIFIED} · "
                    f"Deadline: {item.deadline or UNSPECIFIED}"
                )
                with st.expander("Task, owner, and deadline sources"):
                    for field, evidence_list in (
                        ("Task", item.evidence),
                        ("Owner", item.owner_evidence),
                        ("Deadline", item.deadline_evidence),
                    ):
                        if evidence_list:
                            st.caption(field)
                            show_citations(
                                evidence_list,
                                f"{prefix}_task_{index}_{field}",
                            )

    if not record.action_items:
        st.info("No action items were extracted and retained.")

    st.markdown("**Open questions**")
    for index, question in enumerate(record.open_questions):
        st.write(question.text)
        with st.expander("Sources"):
            show_citations(question.evidence, f"{prefix}_question_{index}")
    if not record.open_questions:
        st.caption("No unresolved questions were extracted.")

    with st.expander("Citation audit and excluded candidate items"):
        st.json(run.checked.report)
        st.caption("Original candidate, before citation filtering:")
        st.json(run.candidate_record.model_dump(mode="json"))

    st.download_button(
        "Download citation-checked record · JSON",
        record.model_dump_json(indent=2),
        file_name="citation_checked_meeting_record.json",
        mime="application/json",
    )

elif run.candidate_record is not None:
    st.warning(
        "Meeting documentation was generated, but citation checks did not "
        "complete. Retry unfinished stages."
    )
elif run.guarded is not None:
    st.info(
        "Transcription and guarded refinement are available. "
        "Meeting documentation has not completed."
    )

with st.expander("Run details and transcript downloads"):
    st.json(
        {
            "status": run.status,
            "stage_seconds": run.seconds,
            "attempts": run.attempts,
        }
    )
    if run.raw is not None:
        st.download_button(
            "Download raw transcript · JSON",
            run.raw.model_dump_json(indent=2),
            file_name="raw_transcript.json",
            mime="application/json",
        )
    if run.guarded is not None:
        st.download_button(
            "Download refined transcript · JSON",
            run.guarded.transcript.model_dump_json(indent=2),
            file_name="refined_transcript.json",
            mime="application/json",
        )