import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from tracemeet.audio.validate import AudioValidationError, validate_media
from tracemeet.stt.local_whisper import LocalWhisper, TranscriptionError

AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg"]
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm"]
RUNS_DIR = Path(__file__).parent / "runs"
WHISPER_MODEL = "small.en"

st.set_page_config(page_title="TraceMeet", page_icon="🎙️", layout="wide")


@st.cache_resource(show_spinner="Loading speech model (the first run downloads it)...")
def get_stt(model_name: str) -> LocalWhisper:
    return LocalWhisper(model_name)


def save_upload(uploaded) -> Path:
    """Write the upload to disk once so Whisper can read it by path."""
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"input{Path(uploaded.name).suffix.lower()}"
    uploaded.seek(0)
    with open(path, "wb") as f:
        shutil.copyfileobj(uploaded, f)
    return path


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def reset_state(file_id: str) -> None:
    st.session_state.file_id = file_id
    st.session_state.media_path = None
    st.session_state.transcript = None
    st.session_state.error = None


st.title("TraceMeet")
st.caption("Meeting records you can verify")

uploaded = st.file_uploader(
    "Upload a meeting recording (audio or video)",
    type=AUDIO_TYPES + VIDEO_TYPES,
)

if uploaded is None:
    st.info("Upload an English meeting recording to begin.")
    st.stop()

if uploaded.size == 0:
    st.error("This file is empty. Please upload a valid recording.")
    st.stop()

file_id = getattr(uploaded, "file_id", f"{uploaded.name}-{uploaded.size}")
if st.session_state.get("file_id") != file_id:
    reset_state(file_id)  # a new upload clears the previous results

st.write(f"**{uploaded.name}**  ·  {uploaded.size / 1024 / 1024:.2f} MB")
if (uploaded.type or "").startswith("video/"):
    preview_col, _ = st.columns([1, 2])
    with preview_col:
        st.video(uploaded)
else:
    st.audio(uploaded, format=uploaded.type)

if st.button("Transcribe", type="primary"):
    st.session_state.transcript = None
    st.session_state.error = None
    with st.status("Processing...", expanded=True) as status:
        try:
            if st.session_state.media_path is None:
                st.session_state.media_path = save_upload(uploaded)
            path = st.session_state.media_path

            status.update(label="Checking the file...")
            info = validate_media(path)
            if info.duration_s is not None:
                st.write(f"File OK, duration {fmt_time(info.duration_s)}.")

            status.update(label="Transcribing...")
            stt = get_stt(WHISPER_MODEL)
            bar = st.progress(0.0, text="Starting...")
            st.session_state.transcript = stt.transcribe(
                path,
                on_progress=lambda p: bar.progress(p, text=f"{p:.0%} of the audio processed"),
            )
            status.update(label="Transcription complete", state="complete")
        except (AudioValidationError, TranscriptionError) as exc:
            st.session_state.error = str(exc)
            status.update(label="Failed", state="error")

if st.session_state.error:
    st.error(st.session_state.error)

transcript = st.session_state.transcript
if transcript is not None:
    st.subheader("Raw transcript")
    st.caption(
        f"{len(transcript.segments)} segments  ·  model {transcript.stt_model}"
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
        use_container_width=True,
        hide_index=True,
    )