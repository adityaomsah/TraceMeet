from pathlib import Path
import streamlit as st
from tracemeet.config import load_config
from tracemeet.pipeline import recover_run
from tracemeet.ui.processing import execute
from tracemeet.ui.results import list_runs, render_results, read_json

st.set_page_config(page_title='TraceMeet · Saved runs',page_icon='🎙️',layout='wide')
st.title('Saved runs')
st.caption('Viewing and downloading make no API calls. Resume may use API quota for unfinished or invalidated stages.')
runs=list_runs(Path(__file__).resolve().parent.parent/'runs')
if not runs:
    st.info('No saved transcripts found yet.');st.stop()
choice=st.selectbox('Run',runs,format_func=lambda path:path.name)
if st.session_state.get('saved_choice') != str(choice):
    st.session_state['saved_choice'] = str(choice)
    st.session_state.pop('tm_error', None)
state=read_json(choice/'run_state.json') or {}
st.caption(f"Saved status: {state.get('status','CLI run')} · {state.get('stage','')}")
if state.get('error'):
    st.warning('The previous processing attempt stopped. Validated saved stages can be reused.')
apply_settings=st.checkbox('Apply current configuration to this run',value=True,
                          help='Only stages whose inputs, configuration or code changed are regenerated. Original glossary and protected names are retained.')
if st.button('Resume processing',type='primary'):
    try:
        # Always load .env; use current or saved configuration as selected.
        cfg=load_config()
        run=recover_run(choice,cfg=cfg if apply_settings else None)
        for note in run.notes:st.info(note)
        execute(run)
    except Exception as exc:
        st.session_state['tm_error']=str(exc)
    st.rerun()
if st.session_state.get('tm_error'):st.error(st.session_state['tm_error'])
render_results(choice)