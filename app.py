import streamlit as st

AUDIO_TYPES = ["wav", "mp3", "m4a", "flac", "ogg"]
VIDEO_TYPES = ["mp4", "mov", "mkv", "webm"]

st.set_page_config(page_title="TraceMeet", page_icon="🎙️", layout="wide")
st.title("TraceMeet")
st.caption("Turn meeting recordings into clear, traceable records.")

uploaded = st.file_uploader(
    "Upload a meeting recording (audio or video)",
    type=AUDIO_TYPES + VIDEO_TYPES,
)

if uploaded is None:
    st.info("Upload an English meeting recording to begin.")
else:
    data = uploaded.getvalue()
    if len(data) == 0:
        st.error("This file is empty. Please upload a valid recording.")
    else:
        st.write(f"**{uploaded.name}**  ·  {len(data) / 1024 / 1024:.2f} MB")
        is_video = bool(uploaded.type) and uploaded.type.startswith("video/")
        if is_video:
            preview_col, _ = st.columns([1, 1])   # video takes the left third
            with preview_col:
                st.video(data)
        else:
            st.audio(data, format=uploaded.type)