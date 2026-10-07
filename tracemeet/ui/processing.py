"""Shared execution UI; provider clients are scoped to one processing attempt."""
import logging
import streamlit as st
from tracemeet.config import get_api_key, get_groq_api_key
from tracemeet.pipeline import continue_run


@st.cache_resource(show_spinner=False)
def get_transcriber(model_name, device, compute_type):
    from tracemeet.stt.local_whisper import LocalWhisper
    return LocalWhisper(model_name=model_name, device=device, compute_type=compute_type)


def execute(run):
    clients = {}
    st.session_state.pop('tm_error',None)
    with st.status('Processing recording…',expanded=True) as status:
        progress=st.progress(0.0,text='Preparing…')
        def report(message):
            status.update(label=message);st.write(message)
        def refinement():
            if 'gemini' not in clients:
                from tracemeet.llm.gemini import GeminiProvider
                clients['gemini']=GeminiProvider(get_api_key())
            return clients['gemini']
        def minutes(grouping=False):
            key='groq_group' if grouping else 'groq'
            if key not in clients:
                from tracemeet.llm.groq_provider import GroqProvider
                cfg=run.cfg['llm'].get('groq',{})
                clients[key]=GroqProvider(get_groq_api_key(),request_budget=cfg.get('request_budget',7400),
                    max_completion_tokens=cfg.get('group_completion_tokens',2048) if grouping else cfg.get('max_completion_tokens',3072),
                    timeout_s=cfg.get('timeout_s',120))
            return clients[key]
        try:
            cfg=run.cfg['stt']
            continue_run(run,refine_provider_factory=refinement,minutes_provider_factory=minutes,
                engine_factory=lambda:get_transcriber(cfg['model'],cfg['device'],cfg['compute_type']),
                on_status=report,on_progress=lambda value:progress.progress(
                    max(0.0,min(float(value),1.0)),text=f'Transcription audio position: {value:.0%}'))
            status.update(label='Processing complete',state='complete')
        except Exception as exc:
            # No traceback or raw server payload is shown in the UI. Saved
            # stages and provider diagnostics remain available in run artifacts.
            message=str(exc)
            import os
            for name in ('GEMINI_API_KEY','GROQ_API_KEY'):
                key=os.getenv(name,'')
                if key:message=message.replace(key,'[REDACTED]')
            st.session_state['tm_error']=f'{run.stage}: {message}'
            status.update(label=f'Stopped: {run.stage}',state='error')
        finally:
            progress.empty()
            for client in clients.values():
                try:client.close()
                except Exception:logging.warning('A model client could not be closed.')
