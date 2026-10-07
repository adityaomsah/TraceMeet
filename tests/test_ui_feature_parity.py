from pathlib import Path
import pytest
from tracemeet.schemas import Transcript, Segment
from tracemeet.ui import results as ui


def test_exact_offsets_preserve_spaces_and_escape_html():
    edit={'raw_start':2,'raw_end':6,'candidate_start':2,'candidate_end':6}
    rendered=ui.highlighted_text('  <Jay>  ',[edit],'raw')
    assert '  <del>&lt;Jay</del>&gt;  ' in rendered
    assert '<Jay>' not in rendered


def test_bad_offsets_fail_instead_of_highlighting_wrong_text():
    with pytest.raises(ValueError):
        ui.highlighted_text('abc',[{'raw_start':-1,'raw_end':2}], 'raw')


def test_raw_download_available_without_minutes_or_refinement():
    raw=Transcript(segments=[Segment(id='s1',start=0,end=1,text='Hello')])
    run=ui.RunData(raw,None,None,None,None,None)
    downloads=ui.partial_downloads(run)
    assert set(downloads)=={'raw_transcript.json'}
    assert Transcript.model_validate_json(downloads['raw_transcript.json'])==raw


def test_refined_download_available_after_minutes_failure():
    raw=Transcript(segments=[Segment(id='s1',start=0,end=1,text='Hello')])
    run=ui.RunData(raw,raw,None,None,None,None)
    assert set(ui.partial_downloads(run))=={'raw_transcript.json','refined_transcript.json'}


def test_saved_runs_include_failures_before_transcription(tmp_path):
    run=tmp_path/'failed';run.mkdir();(run/'meta.json').write_text('{}')
    assert ui.list_runs(tmp_path)==[run]


def test_playback_retains_selected_quote(monkeypatch):
    state={};monkeypatch.setattr(ui.st,'session_state',state)
    ui._select_playback('s1',1,3,'Do not deploy.')
    assert state['play_quote']=='Do not deploy.'
