import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4
import streamlit as st
from tracemeet.config import load_config
from tracemeet.pipeline import create_run, recover_run, settings_signature
from tracemeet.ui.processing import execute
from tracemeet.ui.results import render_results, read_json

AUDIO_TYPES=['wav','mp3','m4a','flac','ogg']
VIDEO_TYPES=['mp4','mov','mkv','webm']
RUNS_DIR=Path(__file__).resolve().parent/'runs'
st.set_page_config(page_title='TraceMeet',page_icon='🎙️',layout='wide')


def clear_upload():
    for key in ('tm_run_dir','tm_error','play_from','play_run'):
        st.session_state.pop(key,None)


st.title('TraceMeet')
st.caption('Meeting records you can inspect and trace to the recording.')
st.page_link('pages/1_Saved_runs.py',label='Open or resume a saved run',icon='📂')
try:
    cfg=load_config()
except Exception as exc:
    st.error(f'Configuration could not be loaded: {exc}');st.stop()

uploaded=st.file_uploader('Upload an English meeting recording',type=AUDIO_TYPES+VIDEO_TYPES,
                          key='meeting_upload',on_change=clear_upload)
if uploaded is None:
    st.info('Upload a recording, or open a saved run.');st.stop()
if uploaded.size==0:
    st.error('This file is empty. Please upload a valid recording.');st.stop()
st.text(f'{uploaded.name} · {uploaded.size/(1024*1024):.2f} MiB')
if uploaded.size<=50*1024*1024:
    with st.expander('Preview recording'):
        uploaded.seek(0)
        if Path(uploaded.name).suffix.lower().lstrip('.') in VIDEO_TYPES:st.video(uploaded)
        else:st.audio(uploaded,format=uploaded.type or 'audio/wav')
        uploaded.seek(0)
else:st.caption('Preview skipped above 50 MiB. Source playback is available in results when supported.')
terms=st.text_input('Names and technical terms (optional)',help='Contextual hints, not proof that a name or term was spoken.').strip()
names=[x.strip() for x in st.text_input('Protected participant names (comma separated)').split(',') if x.strip()]
st.caption(f"Speech: {cfg['stt']['model']} · Refinement: {cfg['llm']['refine_model']} · Minutes: {cfg['llm']['minutes_model']}")
st.caption('Processing uses cloud API quota. Long meetings take several paced requests; completed work is saved.')
run_dir=st.session_state.get('tm_run_dir')
if run_dir:
    saved = read_json(Path(run_dir) / 'run_state.json') or {}
    if saved.get('settings_signature') != settings_signature(cfg, terms, names):
        st.warning('The displayed artifacts belong to earlier settings. Resume / apply current settings regenerates affected stages; Start new run preserves this run separately.')
left,right=st.columns(2)
start=left.button('Process recording' if not run_dir else 'Start new run',type='primary')
resume=right.button('Resume / apply current settings',disabled=not run_dir)
if start or resume:
    try:
        if start:
            directory=RUNS_DIR/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8])
            directory.mkdir(parents=True,exist_ok=False)
            path=directory/f'input{Path(uploaded.name).suffix.lower()}'
            uploaded.seek(0)
            try:
                with path.open('wb') as destination:shutil.copyfileobj(uploaded,destination,length=1024*1024)
            finally:uploaded.seek(0)
            run=create_run(directory,path,uploaded.name,cfg,terms,names)
            st.session_state['tm_run_dir']=str(directory)
        else:
            run=recover_run(Path(run_dir),cfg=cfg)
            run.terms=terms;run.names=names;run.signature=settings_signature(cfg,terms,names)
        execute(run)
    except (OSError,ValueError) as exc:
        st.session_state['tm_error']=str(exc)
    st.rerun()
if st.session_state.get('tm_error'):st.error(st.session_state['tm_error'])
run_dir=st.session_state.get('tm_run_dir')
if run_dir:
    st.caption(f'Run: {Path(run_dir).name}')
    render_results(Path(run_dir))
