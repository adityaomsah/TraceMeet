from pathlib import Path

import streamlit as st

from tracemeet.ui.results import list_runs, render_results

st.set_page_config(page_title="TraceMeet · Saved runs", page_icon="🎙️", layout="wide")
st.title("Saved runs")
st.caption("Open an earlier run from the runs folder. This page makes no API calls.")

runs = list_runs(Path(__file__).resolve().parent.parent / "runs")
if not runs:
    st.info("No saved runs found yet.")
    st.stop()

choice = st.selectbox("Run", runs, format_func=lambda p: p.name)
render_results(choice)