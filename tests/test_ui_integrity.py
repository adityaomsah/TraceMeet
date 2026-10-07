import json
from pathlib import Path
from tracemeet.ui import results as ui
from tracemeet.schemas import Transcript,Segment,MeetingRecord


def fixture_run(tmp_path):
    transcript=Transcript(segments=[Segment(id='s1',start=0,end=1,text='Report discussion.')])
    record=MeetingRecord(summary='Report discussion.',minutes=[],decisions=[],action_items=[],open_questions=[])
    for name in ['raw_transcript.json','candidate_refined_transcript.json','refined_transcript.json']:
        (tmp_path/name).write_text(transcript.model_dump_json(indent=2),encoding='utf-8')
    (tmp_path/'candidate_meeting_record.json').write_text(record.model_dump_json(indent=2),encoding='utf-8')
    import hashlib
    digest=lambda name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
    (tmp_path/'minutes_meta.json').write_text(json.dumps({'source_sha256':digest('refined_transcript.json'),'record_sha256':digest('candidate_meeting_record.json')}))
    (tmp_path/'guard_report.json').write_text(json.dumps({'raw_sha256':digest('raw_transcript.json'),
        'candidate_sha256':digest('candidate_refined_transcript.json'),'refined_sha256':digest('refined_transcript.json')}))
    return tmp_path


def test_valid_artifacts_enable_downloads(tmp_path):
    run=ui.load_run(fixture_run(tmp_path))
    assert run.bundle is not None and run.record['summary']=='Report discussion.'


def test_modified_record_blocks_exports(tmp_path):
    fixture_run(tmp_path)
    (tmp_path/'candidate_meeting_record.json').write_text('{}')
    run=ui.load_run(tmp_path)
    assert run.record is None and run.bundle is None


def test_stale_checked_file_is_not_preferred(tmp_path):
    fixture_run(tmp_path)
    (tmp_path/'citation_checked_meeting_record.json').write_text('{"summary":"STALE"}')
    assert ui.load_run(tmp_path).record['summary']=='Report discussion.'


def test_incomplete_new_workflow_does_not_show_old_record(tmp_path):
    fixture_run(tmp_path)
    (tmp_path/'run_state.json').write_text(json.dumps({'version':'pipeline-disk-v1','stages':{}}))
    assert ui.load_run(tmp_path).bundle is None


def test_playback_callback_sets_start_and_end(monkeypatch):
    state={};monkeypatch.setattr(ui.st,'session_state',state)
    ui._select_playback('s1',3.5,7.2)
    assert state['play_from']==('s1',3.5,7.2)
