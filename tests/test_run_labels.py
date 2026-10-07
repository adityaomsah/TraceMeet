import hashlib
import json
from tracemeet.ui.run_labels import run_label, save_title


def test_old_run_has_date_and_filename(tmp_path):
    run=tmp_path/'20261008_010000_abc';run.mkdir()
    (run/'meta.json').write_text(json.dumps({'original_filename':'demo.wav'}))
    assert run_label(run)=='08 Oct 2026, 01:00 · Meeting · demo.wav'


def test_custom_title_does_not_change_pipeline_files(tmp_path):
    (tmp_path/'run_state.json').write_text('{"status":"failed"}')
    before=(tmp_path/'run_state.json').read_bytes()
    save_title(tmp_path,'  Sprint   planning ')
    assert 'Sprint planning' in run_label(tmp_path)
    assert (tmp_path/'run_state.json').read_bytes()==before
    save_title(tmp_path,'')
    assert 'Needs retry' in run_label(tmp_path)


def test_topic_requires_matching_sources(tmp_path):
    record=tmp_path/'candidate_meeting_record.json'
    record.write_text(json.dumps({'minutes':[{'title':'Release planning'}]}))
    source=tmp_path/'refined_transcript.json';source.write_text('{}')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path/'minutes_meta.json').write_text(json.dumps({'record_sha256':digest(record),'source_sha256':digest(source)}))
    assert 'Release planning' in run_label(tmp_path)
    source.write_text('{"changed":true}')
    assert 'Release planning' not in run_label(tmp_path)


def test_bad_metadata_falls_back(tmp_path):
    (tmp_path/'meta.json').write_text('not json')
    assert run_label(tmp_path)=='Unknown date · Meeting · Unknown input'
