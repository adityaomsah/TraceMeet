import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tracemeet.schemas import Evidence, MinutesTopic, Segment, Transcript
from tracemeet.stages.minutes_finalize import (
    CondensedTopic, FinalizationError, FinalizationRunner, OmittedTopic,
    OverviewDraft, ReviewDisposition, ReviewedTasks, TASK_SYSTEM,
    candidate_catalog, finalize_record, plan_review_jobs, prepare_review,
    restore_overview, review_ids, validate_dispositions,
)
from tracemeet.stages.minutes_reconcile import GroupRecord, TaskDraft


def fixture_data():
    transcript = Transcript(segments=[
        Segment(id='s1', start=0, end=2, text='Jay, this concerns your refinement work.'),
        Segment(id='s2', start=2, end=4, text='You will report back tomorrow.'),
        Segment(id='s3', start=4, end=6, text='We have not decided to deploy.'),
    ])
    record = GroupRecord.model_validate(dict(
        minutes=[dict(title='Report', summary='Report discussion.',
                      evidence=[dict(segment_id='s2', quote=transcript.segments[1].text)])],
        decisions=[], open_questions=[], action_items=[dict(
            description='Report back', status='confirmed', owner=None, deadline=None,
            evidence=[dict(segment_id='s2', quote=transcript.segments[1].text)],
            owner_evidence=[], deadline_evidence=[],
        )],
    ))
    return transcript, record


def task():
    return TaskDraft(description='Report back', status='confirmed', owner=None, deadline=None,
                     evidence=['s2'], owner_evidence=[], deadline_evidence=[])


def review(**overrides):
    data = dict(candidate_ids=['task_001'], action='retain', output_index=0,
                reason='The source explicitly assigns this work.', evidence=['s2'])
    return ReviewDisposition(**(data | overrides))


def check(reviews, outputs=None, catalog=None):
    if catalog is None:
        _, record = fixture_data()
        catalog = candidate_catalog('tasks', [record])
    if outputs is None:
        outputs = [task()]
    validate_dispositions(reviews, catalog, outputs, ['action_items'] * len(outputs), ['s1','s2','s3'])


def test_review_includes_context_for_owner_attribution():
    transcript, record = fixture_data()
    job = prepare_review('tasks', [record], transcript)
    payload = json.loads(job.prompt)
    assert payload['source_segments'][0] == ['s1', transcript.segments[0].text]
    assert review_ids(job) == ['task_001']


def test_answer_evidence_in_other_outcome_role_is_supplied():
    transcript, record = fixture_data()
    record.decisions = []
    from tracemeet.schemas import OpenQuestion
    record.open_questions = [OpenQuestion(text='Deploy?', evidence=[Evidence(segment_id='s3',quote=transcript.segments[2].text)])]
    assert 's3' in prepare_review('tasks', [record], transcript).allowed_ids


def test_retained_candidate_passes():
    check([review()])


@pytest.mark.parametrize('entries', [[], [review(), review()], [review(candidate_ids=['invented'])]])
def test_missing_duplicate_or_unknown_candidate_fails(entries):
    with pytest.raises(FinalizationError, match='coverage'):
        check(entries)


def test_removal_has_audited_reason_and_no_output():
    check([review(action='remove', output_index=None, reason='The source cancels it.', evidence=['s2','s3'])], outputs=[])


def test_remove_cannot_hide_an_output():
    with pytest.raises(FinalizationError, match='must not'):
        check([review(action='remove')])


def test_no_orphan_output():
    with pytest.raises(FinalizationError, match='Every output'):
        check([review()], [task(), task()])


@pytest.mark.parametrize('index', [-1, 1, None])
def test_invalid_output_index(index):
    with pytest.raises(FinalizationError, match='index'):
        check([review(output_index=index)])


def test_retain_cannot_change_deadline():
    value=task().model_copy(update={'deadline':'tomorrow','deadline_evidence':['s2']})
    with pytest.raises(FinalizationError, match='use revise'):
        check([review()], [value])
    check([review(action='revise')], [value])


def test_merge_requires_multiple_existing_candidates():
    with pytest.raises(FinalizationError, match='at least two'):
        check([review(action='merge')])
    _, record=fixture_data()
    record.action_items.append(record.action_items[0].model_copy(deep=True))
    catalog=candidate_catalog('tasks',[record])
    check([review(action='merge',candidate_ids=list(catalog))], catalog=catalog)


def test_merge_unknown_reference_is_rejected():
    with pytest.raises(FinalizationError, match='coverage'):
        check([review(action='merge',candidate_ids=['task_001','task_999'])])


def test_unknown_review_evidence_rejected():
    with pytest.raises(FinalizationError, match='source IDs'):
        check([review(evidence=['outside'])])


def topics_fixture():
    ev=Evidence(segment_id='s1',quote='Discuss the report.')
    topics={key:MinutesTopic(title='Report',summary='Report discussion.',evidence=[ev]) for key in ['t1','t2']}
    return topics, {'t1':'g1','t2':'g1'}


def overview(ids=('t1','t2'), omitted=()):
    return OverviewDraft(summary='Reports were discussed.',minutes=[CondensedTopic(
        title='Reports',summary='Reports were discussed.',source_topic_ids=list(ids))], omitted_topics=list(omitted))


def test_overview_restores_source_evidence_and_deduplicates():
    topics,groups=topics_fixture()
    result=restore_overview(overview(),topics,groups)
    assert len(result[0].evidence)==1


def test_overview_rejects_invented_topic_reference():
    topics,groups=topics_fixture()
    with pytest.raises(FinalizationError,match='Unknown'):
        restore_overview(overview(('t1','t999')),topics,groups)


def test_overview_cannot_merge_unrelated_groups():
    topics,groups=topics_fixture();groups['t2']='g2'
    with pytest.raises(FinalizationError,match='Cross-group'):
        restore_overview(overview(),topics,groups)


def test_topic_cannot_silently_disappear():
    topics,groups=topics_fixture()
    with pytest.raises(FinalizationError,match='coverage'):
        restore_overview(overview(('t1',)),topics,groups)
    result=restore_overview(overview(('t1',),[OmittedTopic(topic_id='t2',reason='Duplicate report topic.')]),topics,groups)
    assert len(result)==1


def test_topic_cannot_be_both_retained_and_omitted():
    topics,groups=topics_fixture()
    with pytest.raises(FinalizationError,match='coverage'):
        restore_overview(overview(omitted=[OmittedTopic(topic_id='t2',reason='Duplicate.')]),topics,groups)


def test_overview_schema_retains_actual_title_property():
    definition=OverviewDraft.model_json_schema()['$defs']['CondensedTopic']
    assert 'title' in definition['properties'] and 'title' in definition['required']


class FakeProvider:
    name='fake'
    def __init__(self):
        self.calls=0
    def estimate_request(self,**kwargs):
        return dict(fits=True,estimated_input_tokens=100,reserved_completion_tokens=100,
                    estimated_total_tokens=200,request_budget=7400)
    def generate_structured(self,**kwargs):
        self.calls+=1
        return ReviewedTasks(action_items=[task()],reviews=[review()])


def test_checkpoint_reuse_and_input_change_invalidation(tmp_path,monkeypatch):
    import tracemeet.stages.minutes_finalize as mod
    monkeypatch.setattr(mod,'wait_for_request_gap',lambda *args:None)
    provider=FakeProvider()
    def run(source_hash):
        runner=FinalizationRunner(provider,'fake',tmp_path,{'source':source_hash})
        result=runner.call('tasks','rules','prompt',ReviewedTasks,
                           lambda response: check(response.reviews,response.action_items),lambda _:None)
        assert 'tasks' in runner.audit
        return result
    run('first');run('first')
    assert provider.calls==1
    run('changed');assert provider.calls==2


def test_invalid_cached_disposition_is_revalidated(tmp_path,monkeypatch):
    import tracemeet.stages.minutes_finalize as mod
    monkeypatch.setattr(mod,'wait_for_request_gap',lambda *args:None)
    provider=FakeProvider()
    runner=FinalizationRunner(provider,'fake',tmp_path,{})
    validate=lambda response:check(response.reviews,response.action_items)
    runner.call('tasks','rules','prompt',ReviewedTasks,validate,lambda _:None)
    path=next(tmp_path.glob('*.json'))
    data=json.loads(path.read_text());data['response']['reviews']=[];path.write_text(json.dumps(data))
    runner.call('tasks','rules','prompt',ReviewedTasks,validate,lambda _:None)
    assert provider.calls==2


def test_planner_keeps_groups_complete_and_accounts_for_every_candidate():
    transcript,record=fixture_data()
    class BatchProvider(FakeProvider):
        def estimate_request(self,**kwargs):
            payload=json.loads(kwargs['prompt'])
            count=len(payload['candidates'])
            return dict(fits=count<=2,estimated_total_tokens=5000 if count<=2 else 8000,request_budget=7400)
    record.action_items.append(record.action_items[0].model_copy(deep=True))
    jobs=plan_review_jobs('tasks',[record,deepcopy(record)],transcript,BatchProvider())
    assert [review_ids(job) for job in jobs]==[['task_001','task_002'],['task_003','task_004']]


def test_oversize_group_fails_without_request():
    transcript,record=fixture_data()
    class TooSmall(FakeProvider):
        def estimate_request(self,**kwargs):return dict(estimated_total_tokens=8000,request_budget=7400)
    provider=TooSmall()
    with pytest.raises(FinalizationError,match='complete discussion group'):
        plan_review_jobs('tasks',[record],transcript,provider)
    assert provider.calls==0


def test_finalize_with_fake_provider_preserves_assignment_and_topic(tmp_path,monkeypatch):
    import tracemeet.stages.minutes_finalize as mod
    monkeypatch.setattr(mod,'wait_for_request_gap',lambda *args:None)
    transcript,record=fixture_data()
    class EndToEnd(FakeProvider):
        def generate_structured(self,**kwargs):
            self.calls+=1
            if kwargs['schema'] is ReviewedTasks:
                return ReviewedTasks(action_items=[task()],reviews=[review()])
            return overview(('topic_001',))
    provider=EndToEnd();runner=FinalizationRunner(provider,'fake',tmp_path,{})
    result=finalize_record([record],transcript,runner,lambda _:None)
    assert len(result.action_items)==1 and result.action_items[0].description=='Report back'
    assert len(result.minutes)==1 and result.minutes[0].evidence[0].segment_id=='s2'
    assert provider.calls==2


def test_review_cannot_attach_only_unrelated_source():
    with pytest.raises(FinalizationError, match='anchor each candidate'):
        check([review(evidence=['s3'])])
