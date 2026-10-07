"""Choose single-call or checkpointed long-meeting documentation by budget."""
import json
import time
from pathlib import Path

from tracemeet.schemas import MeetingRecord
from tracemeet.stages.minutes import DEFAULT_PROMPT, generate_minutes

WORKFLOW_VERSION = 'minutes-workflow-v1'


class PacedProvider:
    """Share one clock across stages, including the smaller grouping client.

    Stage-local pacing may already have waited; only the remaining gap is added.
    No extra retry loop is introduced here.
    """
    def __init__(self, provider, clock, on_status, gap=61.0):
        self._provider, self._clock = provider, clock
        self._status, self._gap = on_status, gap

    def __getattr__(self, name):
        return getattr(self._provider, name)

    def generate_structured(self, **kwargs):
        previous = self._clock.get('finished')
        if previous is not None:
            remaining = previous + self._gap - time.monotonic()
            if remaining > 0:
                self._status(f'API pacing: waiting {remaining:.0f}s before the next request.')
                while remaining > 0:
                    time.sleep(min(1.0, remaining))
                    remaining = previous + self._gap - time.monotonic()
        try:
            return self._provider.generate_structured(**kwargs)
        finally:
            self._clock['finished'] = time.monotonic()


def short_request(transcript, provider):
    system = DEFAULT_PROMPT.read_text(encoding='utf-8')
    payload = json.dumps({'segments': [{'id': s.id, 'text': s.text}
                                     for s in transcript.segments]}, ensure_ascii=False)
    return payload, provider.estimate_request(system=system, prompt=payload, schema=MeetingRecord)


def generate_documentation(transcript, run_dir, *, provider, grouping_provider,
                           model, on_status=print):
    # Keep the input budget and the single-stage character bound consistent.
    payload, budget = short_request(transcript, provider)
    from tracemeet.stages.minutes import DEFAULT_MAX_CHARS
    if budget['fits'] and len(payload) <= DEFAULT_MAX_CHARS:
        on_status('Meeting documentation: single-request path.')
        record = generate_minutes(transcript, provider, model=model,
                                  temperature=0.0, on_status=on_status)
        return record, {'path': 'short', 'budget': budget, 'version': WORKFLOW_VERSION}

    on_status('Meeting documentation: checkpointed long-meeting path.')
    return generate_long_documentation(transcript, Path(run_dir), provider=provider,
                                        grouping_provider=grouping_provider,
                                        model=model, on_status=on_status)


def generate_long_documentation(transcript, run_dir, *, provider, grouping_provider,
                                model, on_status=print):
    # Import long stages lazily so the short path does not initialize them.
    from hashlib import sha256
    from tracemeet.stages.minutes_map import (
        DEFAULT_MAP_PROMPT, MAP_VERSION, extract_meeting_notes, save_json_atomic,
    )
    from tracemeet.stages.minutes_group import GROUP_VERSION, generate_groups, materialize_groups
    from tracemeet.stages.minutes_reconcile import (
        RECONCILE_VERSION, RECONCILE_SYSTEM, GroupDraft, build_group_jobs, reconcile_groups, GroupRecord,
    )
    from tracemeet.stages.minutes_finalize import FINALIZE_VERSION, FinalizationRunner, finalize_record

    source_text = (run_dir / 'refined_transcript.json').read_text(encoding='utf-8')
    saved = type(transcript).model_validate_json(source_text)
    if saved.model_dump(mode='json') != transcript.model_dump(mode='json'):
        raise ValueError('Saved refined transcript differs from the workflow input.')
    source_hash = sha256(source_text.encode('utf-8')).hexdigest()
    clock = {}
    main = PacedProvider(provider, clock, on_status)
    grouping = PacedProvider(grouping_provider, clock, on_status)

    on_status('Long meeting: extracting provisional observations.')
    chunks = extract_meeting_notes(transcript, main, model=model,
                                   checkpoint_dir=run_dir / 'minutes_map_chunks', on_status=on_status)
    observations = [
        {'observation_id': f'chunk_{ci:03d}_note_{ni:03d}', **ob.model_dump(mode='json')}
        for ci, chunk in enumerate(chunks, 1)
        for ni, ob in enumerate(chunk.observations, 1)
    ]
    # Match the established CLI document layout to preserve downstream hashes.
    notes_path = run_dir / 'provisional_meeting_notes.json'
    save_json_atomic(notes_path, {
        'status': 'provisional_not_final_minutes', 'implementation_version': MAP_VERSION,
        'provider': main.name, 'model': model, 'source_text_sha256': source_hash,
        'planning_prompt_file_sha256': sha256(DEFAULT_MAP_PROMPT.read_bytes()).hexdigest(),
        'input_segment_ids': [sid for chunk in chunks for sid in chunk.input_segment_ids],
        'observations': observations, 'semantic_support_checked': False,
        'generation_provenance': 'See minutes_map_chunks success checkpoints for per-chunk generation, repair and migration history.',
    })
    notes_hash = sha256(notes_path.read_bytes()).hexdigest()
    if not observations:
        return MeetingRecord(summary='No substantive meeting observations were extracted.',
                             minutes=[], decisions=[], action_items=[], open_questions=[]), {
            'path': 'long', 'version': WORKFLOW_VERSION, 'observations': 0,
        }

    on_status('Long meeting: grouping observations across the meeting.')
    plan = generate_groups(observations, grouping, model=model,
                           checkpoint_dir=run_dir / 'minutes_group_chunks',
                           source_sha256=source_hash, notes_sha256=notes_hash, on_status=on_status)
    groups = materialize_groups(plan, observations)
    groups_path = run_dir / 'meeting_groups.json'
    save_json_atomic(groups_path, {
        'status': 'grouped_not_reconciled', 'implementation_version': GROUP_VERSION,
        'provider': main.name, 'model': model, 'source_text_sha256': source_hash,
        'provisional_notes_file_sha256': notes_hash, 'groups': groups,
        'semantic_support_checked': False,
    })
    hashes = {'source_text_sha256': source_hash, 'provisional_notes_file_sha256': notes_hash,
              'groups_file_sha256': sha256(groups_path.read_bytes()).hexdigest()}
    on_status('Long meeting: reconciling groups against source segments.')
    jobs = build_group_jobs(transcript, observations, groups)
    for job in jobs:
        budget = main.estimate_request(system=RECONCILE_SYSTEM, prompt=job.prompt, schema=GroupDraft)
        if not budget['fits']:
            raise ValueError(f"Discussion group {job.group_id} exceeds the request budget. "
                             "Completed extraction/grouping checkpoints are retained; no source was truncated.")
    results = reconcile_groups(jobs, transcript, main, model=model,
                                checkpoint_dir=run_dir / 'minutes_reconcile_chunks',
                                input_hashes=hashes, on_status=on_status)
    reconciled_path = run_dir / 'reconciled_group_records.json'
    save_json_atomic(reconciled_path, {
        'status': 'candidate_group_records', 'implementation_version': RECONCILE_VERSION,
        'provider': main.name, 'model': model, **hashes, 'groups': results,
        'semantic_support_checked': False,
        'limitations': [
            'Source IDs and quote retrieval were checked.',
            'Claim interpretation remains model-generated.',
            'Neighbouring context does not guarantee speaker identity.',
            'Incorrect grouping can miss cross-group relationships.',
            'Final consolidation and overall summary are pending.',
        ],
    })
    records_hash = sha256(reconciled_path.read_bytes()).hexdigest()
    records = [GroupRecord.model_validate(value['record']) for value in results]
    on_status('Long meeting: reviewing candidates and preparing the overview.')
    runner = FinalizationRunner(main, model, run_dir / 'minutes_finalize_chunks', {
        'source_text_sha256': source_hash, 'reconciled_records_file_sha256': records_hash,
    })
    record = finalize_record(records, transcript, runner, on_status)
    from tracemeet.stages.minutes_finalize import candidate_catalog, plan_review_jobs, review_ids
    save_json_atomic(run_dir / 'finalization_report.json', {
        'version': FINALIZE_VERSION, **hashes, 'reconciled_records_file_sha256': records_hash,
        'semantic_support_checked': False,
        'candidate_catalogs': {kind: candidate_catalog(kind, records) for kind in ('tasks','outcomes')},
        'review_batches': {kind: [{'candidate_ids': review_ids(job), 'allowed_segment_ids': list(job.allowed_ids)}
                                  for job in plan_review_jobs(kind, records, transcript, main)]
                           for kind in ('tasks','outcomes')},
        'validated_responses': runner.audit,
        'limitations': ['Candidate accounting and citation presence do not verify semantics or extraction completeness.'],
    })
    return record, {'path': 'long', 'version': WORKFLOW_VERSION,
                    'observations': len(observations), 'groups': len(groups),
                    'finalization_version': FINALIZE_VERSION}