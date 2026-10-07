from types import SimpleNamespace
import json
from pathlib import Path
from tracemeet.schemas import Transcript,Segment,MeetingRecord
from tracemeet.stages import minutes_workflow as w


def source():
    return Transcript(segments=[Segment(id='s1',start=0,end=1,text='Discuss the report.')])


def empty_record():
    return MeetingRecord(summary='Report discussion.',minutes=[],decisions=[],action_items=[],open_questions=[])


def test_short_path_routes_by_budget(monkeypatch):
    monkeypatch.setattr(w,'short_request',lambda *a:('payload',{'fits':True}))
    calls=[]
    monkeypatch.setattr(w,'generate_minutes',lambda *a,**kw:calls.append(kw) or empty_record())
    result,meta=w.generate_documentation(source(),Path('.'),provider=object(),grouping_provider=object(),model='test',on_status=lambda _:None)
    assert meta['path']=='short' and calls[0]['temperature']==0.0


def test_over_budget_routes_without_truncating(monkeypatch):
    transcript=source();seen=[]
    monkeypatch.setattr(w,'short_request',lambda *a:('payload',{'fits':False}))
    monkeypatch.setattr(w,'generate_long_documentation',lambda t,*a,**kw:seen.append(t) or (empty_record(),{'path':'long'}))
    _,meta=w.generate_documentation(transcript,Path('.'),provider=object(),grouping_provider=object(),model='test')
    assert meta['path']=='long' and seen[0] is transcript


def test_shared_pacing_clock_across_clients(monkeypatch):
    now=[0.0];waits=[]
    monkeypatch.setattr(w.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(w.time,'sleep',lambda seconds:(waits.append(seconds),now.__setitem__(0,now[0]+seconds)))
    client=SimpleNamespace(generate_structured=lambda **kw:'ok')
    clock={};one=w.PacedProvider(client,clock,lambda _:None);two=w.PacedProvider(client,clock,lambda _:None)
    assert one.generate_structured()=='ok'
    assert two.generate_structured()=='ok'
    assert sum(waits)==61.0


def test_long_workflow_calls_stages_and_writes_compatible_artifacts(tmp_path,monkeypatch):
    from tracemeet.stages import minutes_map as m,minutes_group as g,minutes_reconcile as r,minutes_finalize as f
    transcript=source();(tmp_path/'refined_transcript.json').write_text(transcript.model_dump_json(indent=2))
    prompt=tmp_path/'map.txt';prompt.write_text('fixture')
    monkeypatch.setattr(m,'DEFAULT_MAP_PROMPT',prompt)
    ob=m.Observation(kind='topic',statement='Report discussion.',evidence=[{'segment_id':'s1','quote':'Discuss the report.'}])
    monkeypatch.setattr(m,'extract_meeting_notes',lambda *a,**kw:[m.ChunkNotes(input_segment_ids=['s1'],observations=[ob])])
    monkeypatch.setattr(g,'generate_groups',lambda *a,**kw:'plan')
    groups=[{'group_id':'group_001','label':'Report','observation_ids':['chunk_001_note_001']}]
    monkeypatch.setattr(g,'materialize_groups',lambda *a:groups)
    monkeypatch.setattr(r,'build_group_jobs',lambda *a:[SimpleNamespace(group_id='group_001',prompt='fixture')])
    record={'minutes':[],'decisions':[],'action_items':[],'open_questions':[]}
    results=[{**groups[0],'source_segment_ids':['s1'],'record':record}]
    monkeypatch.setattr(r,'reconcile_groups',lambda *a,**kw:results)
    monkeypatch.setattr(f,'finalize_record',lambda *a,**kw:empty_record())
    provider=SimpleNamespace(name='groq',estimate_request=lambda **kw:{'fits':True})
    result,meta=w.generate_long_documentation(transcript,tmp_path,provider=provider,
        grouping_provider=provider,model='openai/gpt-oss-120b',on_status=lambda _:None)
    assert result.summary=='Report discussion.' and meta['groups']==1
    notes=json.loads((tmp_path/'provisional_meeting_notes.json').read_text())
    assert notes['input_segment_ids']==['s1']
    assert notes['observations'][0]['observation_id']=='chunk_001_note_001'
    assert (tmp_path/'meeting_groups.json').exists()
    assert (tmp_path/'finalization_report.json').exists()
