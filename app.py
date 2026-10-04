import json
import shutil
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd
import streamlit as st

from tracemeet.audio.validate import AudioValidationError, validate_media
from tracemeet.stt.local_whisper import LocalWhisper, TranscriptionError

AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg"]
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm"]
RUNS_DIR = Path(__file__).parent / "runs"
WHISPER_MODEL = "small.en"
MAX_PREVIEW_BYTES = 50 * 1024 * 1024
STATE_KEYS = (
    "raw_transcript", "transcription_seconds", "transcription_terms", "run_dir", "error",
)

st.set_page_config(page_title="TraceMeet", page_icon="🎙️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_transcriber(model_name: str) -> LocalWhisper:
    return LocalWhisper(model_name=model_name, device="cpu", compute_type="int8")


def clear_recording_results() -> None:
    for key in STATE_KEYS:
        st.session_state.pop(key, None)


def new_run_dir() -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_upload(uploaded, run_dir: Path) -> Path:
    path = run_dir / f"input{Path(uploaded.name).suffix.lower()}"
    uploaded.seek(0)
    try:
        with path.open("wb") as destination:
            shutil.copyfileobj(uploaded, destination, length=1024 * 1024)
    finally:
        uploaded.seek(0)
    return path


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


st.title("TraceMeet")
st.caption("Meeting records you can verify")

uploaded = st.file_uploader(
    "Upload a meeting recording (audio or video)",
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
st.write(f"**{uploaded.name}**  ·  {uploaded.size / (1024 * 1024):.2f} MiB")

if uploaded.size <= MAX_PREVIEW_BYTES:
    uploaded.seek(0)
    if is_video:
        preview_col, _ = st.columns([1, 2])
        with preview_col:
            st.video(uploaded)
    else:
        st.audio(uploaded, format=uploaded.type or "audio/wav")
else:
    st.info("Preview skipped for files larger than 50 MiB.")

terms = st.text_input(
    "Participants and key terms (optional)",
    placeholder="e.g. Aditya Om Sah, IIT Guwahati, Kubernetes",
    help="Names and terms expected in the meeting. They give the speech model context; "
         "they do not prove a name was spoken.",
).strip()

if st.button("Transcribe recording", type="primary"):
    clear_recording_results()
    progress_bar = st.progress(0.0, text="Preparing...")
    started = perf_counter()
    run_dir = None
    succeeded = False
    try:
        run_dir = new_run_dir()
        path = save_upload(uploaded, run_dir)

        progress_bar.progress(0.0, text="Checking the file...")
        validate_media(path)

        with st.spinner("Loading the speech model and transcribing..."):
            engine = get_transcriber(WHISPER_MODEL)
            transcript = engine.transcribe(
                path,
                on_progress=lambda p: progress_bar.progress(
                    p, text=f"Audio position: {p:.0%}"
                ),
                terms=terms or None,
            )

        (run_dir / "raw_transcript.json").write_text(
            transcript.model_dump_json(indent=2), encoding="utf-8"
        )
        (run_dir / "meta.json").write_text(
            json.dumps(
                {
                    "original_filename": uploaded.name,
                    "terms": terms,
                    "stt_model": transcript.stt_model,
                    "created": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        st.session_state["raw_transcript"] = transcript
        st.session_state["transcription_seconds"] = perf_counter() - started
        st.session_state["transcription_terms"] = terms
        st.session_state["run_dir"] = str(run_dir)
        succeeded = True
    except (AudioValidationError, TranscriptionError) as exc:
        st.session_state["error"] = str(exc)
    except OSError:
        st.session_state["error"] = (
            "Could not read or write the run files. Check disk space and folder permissions."
        )
    finally:
        progress_bar.empty()
        if run_dir is not None and not succeeded:
            shutil.rmtree(run_dir, ignore_errors=True)

if st.session_state.get("error"):
    st.error(st.session_state["error"])

transcript = st.session_state.get("raw_transcript")
if transcript is not None:
    st.divider()
    st.subheader("Raw transcript")
    st.caption(
        f"{len(transcript.segments)} segments  ·  "
        f"{st.session_state['transcription_seconds']:.1f} s total (includes model loading)  ·  "
        f"{transcript.stt_model}"
    )
    if terms != st.session_state.get("transcription_terms", ""):
        st.info(
            "The terms have changed since this transcript was generated. "
            "Click Transcribe recording to apply them."
        )
    st.dataframe(
        pd.DataFrame(
            {
                "ID": [s.id for s in transcript.segments],
                "Start": [fmt_time(s.start) for s in transcript.segments],
                "End": [fmt_time(s.end) for s in transcript.segments],
                "Text": [s.text for s in transcript.segments],
            }
        ),
        width="stretch",
        hide_index=True,
    )