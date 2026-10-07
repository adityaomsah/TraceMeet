"""Orchestration tests use fake model calls; no cloud quota or downloads."""
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from tracemeet import pipeline as p
from tracemeet.schemas import Segment,Transcript,MeetingRecord


@pytest.fixture
def setup_run(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir()
    monkeypatch.setattr(p,'ROOT',root)
    for paths in p.STAGE_FILES.values():
        for name in paths:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    directory=root/'runs'/'test';directory.mkdir(parents=True)
    media=directory/'input.wav';media.write_bytes(b'fake audio; decoding is mocked')
    cfg={'stt':{'model':'small.en','device':'cpu','compute_type':'int8'},
         'llm':{'refine_model':'refiner','minutes_model':'openai/gpt-oss-120b',
                'minutes_provider':'groq','temperature':1.0,'groq':{}}}
    run=p.create_run(directory,media,'meeting.wav',cfg,'',[])
    calls={'stt':0,'refine':0,'minutes':0}
    def transcript():
        return Transcript(segments=[Segment(id='s1',start=0,end=2,text='We discussed the report.')],
                          stt_model='faster-whisper/small.en')
    class Engine:
        def transcribe(self,*args,**kwargs):calls['stt']+=1;return transcript()
    def refine(raw,provider,**kwargs):calls['refine']+=1;return raw.model_copy(deep=True)
    def minutes(*args,**kwargs):
        calls['minutes']+=1
        return MeetingRecord(summary='Report discussion.',minutes=[],decisions=[],action_items=[],open_questions=[]),{'path':'short'}
    monkeypatch.setattr(p,'validate_media',lambda path:None)
    monkeypatch.setattr(p,'refine_transcript',refine)
    monkeypatch.setattr(p,'generate_documentation',minutes)
    monkeypatch.setattr(p,'guard_refinement',lambda raw,candidate,**kw:SimpleNamespace(transcript=candidate,corrections=[],edits=[]))
    monkeypatch.setattr(p,'verify_evidence',lambda record,transcript:SimpleNamespace(record=record,report={}))
    def execute(target):
        p.continue_run(target,engine_factory=Engine,
                       refine_provider_factory=lambda:SimpleNamespace(name='gemini'),
                       minutes_provider_factory=lambda grouping:SimpleNamespace(name='groq'),
                       on_status=lambda message:None,on_progress=lambda value:None)
    return run,calls,execute


def test_disk_resume_reuses_all_model_stages(setup_run):
    run,calls,execute=setup_run;execute(run)
    execute(p.recover_run(run.directory,run.cfg))
    assert calls=={'stt':1,'refine':1,'minutes':1}


def test_minutes_failure_does_not_repeat_transcription(setup_run,monkeypatch):
    run,calls,execute=setup_run
    good=p.generate_documentation
    monkeypatch.setattr(p,'generate_documentation',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('temporary failure')))
    with pytest.raises(RuntimeError):execute(run)
    assert not (run.directory/'.processing.lock').exists()
    monkeypatch.setattr(p,'generate_documentation',good)
    execute(p.recover_run(run.directory,run.cfg))
    assert calls=={'stt':1,'refine':1,'minutes':1}


def test_minutes_model_change_invalidates_only_documentation(setup_run):
    run,calls,execute=setup_run;execute(run)
    cfg=deepcopy(run.cfg);cfg['llm']['minutes_model']='openai/gpt-oss-20b'
    execute(p.recover_run(run.directory,cfg))
    assert calls=={'stt':1,'refine':1,'minutes':2}


def test_minutes_code_change_preserves_stt_and_refinement(setup_run):
    run,calls,execute=setup_run;execute(run)
    (p.ROOT/'tracemeet/stages/minutes_workflow.py').write_text('changed implementation')
    execute(p.recover_run(run.directory,run.cfg))
    assert calls=={'stt':1,'refine':1,'minutes':2}


def test_media_change_is_rejected(setup_run):
    run,_,execute=setup_run;execute(run)
    run.media_path.write_bytes(b'different audio')
    with pytest.raises(ValueError,match='recording'):
        p.recover_run(run.directory,run.cfg)


def test_corrupt_candidate_regenerates_minutes(setup_run):
    run,calls,execute=setup_run;execute(run)
    (run.directory/'candidate_meeting_record.json').write_text('{}')
    execute(p.recover_run(run.directory,run.cfg))
    assert calls=={'stt':1,'refine':1,'minutes':2}


def test_existing_run_lock_prevents_model_calls(setup_run):
    run,calls,execute=setup_run
    (run.directory/'.processing.lock').write_text('another process')
    with pytest.raises(ValueError,match='already processing'):execute(run)
    assert calls=={'stt':0,'refine':0,'minutes':0}


def test_invalidated_minutes_not_marked_complete_on_failure(setup_run,monkeypatch):
    run,_,execute=setup_run;execute(run)
    cfg=deepcopy(run.cfg);cfg['llm']['minutes_model']='openai/gpt-oss-20b'
    monkeypatch.setattr(p,'generate_documentation',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('failure')))
    with pytest.raises(RuntimeError):execute(p.recover_run(run.directory,cfg))
    state=json.loads((run.directory/'run_state.json').read_text())
    assert 'minutes' not in state['stages'] and 'evidence' not in state['stages']
    assert (run.directory/'candidate_meeting_record.json').exists()  # previous output preserved


def test_legacy_adoption_is_recorded_and_checks_hash_links(setup_run):
    run,calls,execute=setup_run;execute(run)
    (run.directory/'run_state.json').unlink()
    recovered=p.recover_run(run.directory,run.cfg)
    execute(recovered)
    assert calls['stt']==1 and calls['refine']==1
    assert recovered.stages['transcription']['provenance']=='legacy-metadata-import'


def test_legacy_wrong_raw_hash_is_not_adopted(setup_run):
    run,calls,execute=setup_run;execute(run)
    meta=p.read_json(run.directory/'refinement_meta.json');meta['source_sha256']='bad'
    p.save_json(run.directory/'refinement_meta.json',meta)
    (run.directory/'run_state.json').unlink()
    execute(p.recover_run(run.directory,run.cfg))
    assert calls['stt']==2
