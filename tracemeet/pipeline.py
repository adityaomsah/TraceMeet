"""Disk-backed stages with input/config/code fingerprints and separate LLM providers."""
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from tracemeet.audio.validate import validate_media
from tracemeet.guards.diff import edit_metrics, transcript_edits
from tracemeet.guards.evidence import verify_evidence
from tracemeet.guards.sensitive import guard_refinement
from tracemeet.schemas import MeetingRecord, Transcript
from tracemeet.stages.refine import refine_transcript
from tracemeet.stages.minutes_workflow import generate_documentation

ROOT = Path(__file__).resolve().parent.parent
STATE_VERSION = 'pipeline-disk-v1'
STAGE_FILES = {
    'transcription': ['tracemeet/stt/local_whisper.py','tracemeet/audio/validate.py','tracemeet/schemas.py'],
    'refinement': ['tracemeet/stages/refine.py','tracemeet/llm/gemini.py','tracemeet/llm/router.py',
                   'tracemeet/llm/base.py','tracemeet/schemas.py','prompts/refine_v1.txt'],
    'guards': ['tracemeet/guards/sensitive.py','tracemeet/guards/diff.py','tracemeet/schemas.py'],
    'minutes': ['tracemeet/stages/minutes_workflow.py','tracemeet/stages/minutes.py',
                'tracemeet/stages/minutes_chunks.py','tracemeet/stages/minutes_map.py',
                'tracemeet/stages/minutes_group.py','tracemeet/stages/minutes_reconcile.py',
                'tracemeet/stages/minutes_finalize.py','tracemeet/llm/groq_provider.py',
                'tracemeet/llm/base.py','tracemeet/llm/router.py','tracemeet/schemas.py',
                'prompts/minutes_v1.txt','prompts/minutes_map_v2.txt'],
    'evidence': ['tracemeet/guards/evidence.py','tracemeet/schemas.py'],
}


def utc_now(): return datetime.now(timezone.utc).isoformat()
def text_hash(text): return hashlib.sha256(text.encode('utf-8')).hexdigest()
def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()
def save_text(path, text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    try:
        temporary.write_bytes(text.encode('utf-8'));temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
def save_json(path, data): save_text(path,json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False))
def read_json(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,dict):raise ValueError(f'{Path(path).name} must contain an object.')
    return data

def settings_signature(cfg, terms, names):
    # UI comparison only. Stage-specific fingerprints control reuse.
    return text_hash(json.dumps({'config':cfg,'terms':terms,'names':names},sort_keys=True,ensure_ascii=False))


def stage_fingerprint(stage, inputs):
    return text_hash(json.dumps({'version':STATE_VERSION,'stage':stage,'inputs':inputs,
        'dependencies':{name:file_hash(ROOT/name) for name in STAGE_FILES[stage]}},
        sort_keys=True,ensure_ascii=False))


@dataclass
class Run:
    directory: Path
    media_path: Path
    original_filename: str
    cfg: dict
    terms: str
    names: list[str]
    signature: str
    media_sha256: str
    raw: object = None
    candidate: object = None
    guarded: object = None
    candidate_record: object = None
    checked: object = None
    status: str = 'ready'
    stage: str = 'Preparation'
    error: str | None = None
    seconds: dict = field(default_factory=dict)
    attempts: list = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def save_run_state(run):
    save_json(run.directory/'run_state.json',{
        'version':STATE_VERSION,'status':run.status,'stage':run.stage,'error':run.error,
        'settings_signature':run.signature,'media_sha256':run.media_sha256,
        'config':run.cfg,'terms':run.terms,'protected_names':run.names,
        'stage_seconds':run.seconds,'attempts':run.attempts,'stages':run.stages,'notes':run.notes,
        'completed':{name:name in run.stages for name in STAGE_FILES},
    })


def create_run(directory, media_path, original_filename, cfg, terms, names):
    directory,media_path=Path(directory),Path(media_path)
    cfg=json.loads(json.dumps(cfg))
    run=Run(directory,media_path,original_filename,cfg,terms,list(names),
            settings_signature(cfg,terms,names),file_hash(media_path))
    save_json(directory/'meta.json',{'original_filename':original_filename,'terms':terms,
        'protected_names':list(names),'config':cfg,'created':utc_now(),'media_sha256':run.media_sha256})
    save_run_state(run)
    return run


def recover_run(directory, cfg=None):
    directory=Path(directory).resolve();meta=read_json(directory/'meta.json')
    media=[p for p in directory.glob('input.*') if p.is_file() and p.suffix.lower() in
           {'.wav','.mp3','.m4a','.flac','.ogg','.mp4','.mov','.mkv','.webm'}]
    if len(media)!=1:raise ValueError('Run needs exactly one saved input recording.')
    if file_hash(media[0])!=meta.get('media_sha256'):
        raise ValueError('Saved recording does not match its metadata.')
    state=read_json(directory/'run_state.json') if (directory/'run_state.json').exists() else {}
    current=state.get('config',meta['config']) if cfg is None else cfg
    terms=state.get('terms',meta.get('terms',''));names=state.get('protected_names',meta.get('protected_names',[]))
    run=Run(directory,media[0],meta.get('original_filename',media[0].name),current,terms,names,
            settings_signature(current,terms,names),meta['media_sha256'])
    if state.get('version')==STATE_VERSION:
        run.stages=state.get('stages',{});run.seconds=state.get('stage_seconds',{})
        run.attempts=state.get('attempts',[]);run.notes=state.get('notes',[])
    else:
        run.notes.append('Legacy run: matching transcription/refinement metadata may be adopted; original code-version provenance is unavailable.')
    run.status=state.get('status','ready');run.stage=state.get('stage','Preparation')
    run.error=state.get('error')
    return run


def _valid_stage(run, stage, fingerprint):
    entry=run.stages.get(stage,{})
    if entry.get('fingerprint')!=fingerprint:return False
    artifacts=entry.get('artifacts')
    if not isinstance(artifacts,dict) or not artifacts:return False
    for name,digest in artifacts.items():
        if Path(name).name!=name:return False
        path=run.directory/name
        if not path.is_file() or file_hash(path)!=digest:return False
    return True


def _complete(run, stage, fingerprint, files, provenance='generated'):
    run.stages[stage]={'fingerprint':fingerprint,'artifacts':{name:file_hash(run.directory/name) for name in files},
                       'provenance':provenance,'completed':utc_now()}
    save_run_state(run)


def _hash_matches(path, expected):
    return isinstance(expected,str) and expected.lower() in {
        file_hash(path),text_hash(Path(path).read_text(encoding='utf-8'))}


def _adopt_legacy(run, stage, fingerprint):
    # This path is explicit in the saved notes. It does not assert knowledge of
    # the historical source-code version. Verify settings and artifact links.
    if not any(note.startswith('Legacy run:') for note in run.notes):return False
    if stage in run.stages:return False
    try:
        meta=read_json(run.directory/'meta.json')
        refinement=read_json(run.directory/'refinement_meta.json')
        raw_path=run.directory/'raw_transcript.json'
        if meta['config']['stt']!=run.cfg['stt'] or meta.get('terms','')!=run.terms:return False
        if not _hash_matches(raw_path,refinement.get('source_sha256')):return False
        raw=Transcript.model_validate_json(raw_path.read_text(encoding='utf-8'))
        if raw.stt_model!=f"faster-whisper/{run.cfg['stt']['model']}":return False
        if stage=='transcription':
            _complete(run,stage,fingerprint,['raw_transcript.json'],'legacy-metadata-import')
            return True
        if stage=='refinement':
            cfg=run.cfg['llm']
            if (refinement.get('model')!=cfg['refine_model'] or refinement.get('provider')!='gemini'
                or refinement.get('temperature')!=cfg['temperature'] or refinement.get('glossary','')!=run.terms
                or refinement.get('prompt_sha256')!=file_hash(ROOT/'prompts/refine_v1.txt')):return False
            path=run.directory/'candidate_refined_transcript.json'
            guard=read_json(run.directory/'guard_report.json')
            if not _hash_matches(path,guard.get('candidate_sha256')):return False
            if not _hash_matches(raw_path,guard.get('raw_sha256')):return False
            candidate=Transcript.model_validate_json(path.read_text(encoding='utf-8'))
            if [(s.id,s.start,s.end) for s in candidate.segments]!=[(s.id,s.start,s.end) for s in raw.segments]:return False
            _complete(run,stage,fingerprint,['candidate_refined_transcript.json','refinement_meta.json'],'legacy-metadata-import')
            return True
    except (OSError,ValueError,KeyError,TypeError):return False
    return False


def continue_run(run, *, engine_factory, on_status, on_progress, provider=None,
                 refine_provider_factory=None, minutes_provider_factory=None):
    if file_hash(run.media_path)!=run.media_sha256:raise ValueError('The saved recording changed.')
    if run.cfg['llm'].get('minutes_provider','gemini')!='groq':
        raise ValueError('This workflow requires minutes_provider: groq. Update config/default.yaml.')
    run.raw=run.candidate=run.guarded=run.candidate_record=run.checked=None
    run.status='running';run.error=None
    attempt={'id':uuid4().hex,'started':utc_now(),'status':'running'};run.attempts.append(attempt)
    # Disk lock prevents two sessions writing the same run. A crashed server
    # can leave the lock; the UI explains recovery instead of stealing it.
    lock=run.directory/'.processing.lock'
    try:
        fd=lock.open('x',encoding='utf-8');fd.write(utc_now());fd.close()
    except FileExistsError as exc:
        raise ValueError('This run is already processing. If the server crashed, stop all processing before removing its .processing.lock file.') from exc
    def begin(label):
        run.stage=label;save_run_state(run);on_status(label);return perf_counter()
    def done(stage,fp,files,started):
        run.seconds[stage]=perf_counter()-started;_complete(run,stage,fp,files)
    def invalidate(stage):
        names=list(STAGE_FILES)
        for name in names[names.index(stage):]:
            run.stages.pop(name,None)
        save_run_state(run)
    def cached(stage,fp):
        okay=_valid_stage(run,stage,fp) or _adopt_legacy(run,stage,fp)
        if okay:on_status(f'{stage}: reused validated saved output.')
        return okay
    try:
        fp=stage_fingerprint('transcription',{'media':run.media_sha256,'stt':run.cfg['stt'],'terms':run.terms})
        if not cached('transcription',fp):
            invalidate('transcription')
            started=begin('Transcription');validate_media(run.media_path)
            raw=engine_factory().transcribe(run.media_path,terms=run.terms or None,on_progress=on_progress)
            save_text(run.directory/'raw_transcript.json',raw.model_dump_json(indent=2))
            done('transcription',fp,['raw_transcript.json'],started)
        run.raw=Transcript.model_validate_json((run.directory/'raw_transcript.json').read_text(encoding='utf-8'))
        cfg=run.cfg['llm']
        fp=stage_fingerprint('refinement',{'raw':file_hash(run.directory/'raw_transcript.json'),
            'model':cfg['refine_model'],'provider':'gemini','temperature':cfg['temperature'],'glossary':run.terms})
        if not cached('refinement',fp):
            invalidate('refinement')
            started=begin('Transcript refinement')
            llm=refine_provider_factory() if refine_provider_factory else provider
            if llm is None:raise ValueError('Refinement provider is unavailable.')
            candidate=refine_transcript(run.raw,llm,model=cfg['refine_model'],temperature=cfg['temperature'],
                glossary=run.terms,checkpoint_dir=run.directory/'refinement_chunks',on_status=on_status)
            edits=transcript_edits(run.raw,candidate)
            save_text(run.directory/'candidate_refined_transcript.json',candidate.model_dump_json(indent=2))
            save_json(run.directory/'candidate_edits.json',{'status':'candidate_not_guarded','edits':edits})
            save_json(run.directory/'refinement_meta.json',{'status':'candidate_not_guarded','provider':llm.name,
                'model':cfg['refine_model'],'temperature':cfg['temperature'],'glossary':run.terms,
                'source_sha256':file_hash(run.directory/'raw_transcript.json'),
                'prompt_sha256':file_hash(ROOT/'prompts/refine_v1.txt'),'metrics':edit_metrics(edits),'created':utc_now()})
            done('refinement',fp,['candidate_refined_transcript.json','candidate_edits.json','refinement_meta.json'],started)
        run.candidate=Transcript.model_validate_json((run.directory/'candidate_refined_transcript.json').read_text(encoding='utf-8'))
        fp=stage_fingerprint('guards',{'raw':file_hash(run.directory/'raw_transcript.json'),
            'candidate':file_hash(run.directory/'candidate_refined_transcript.json'),'names':run.names})
        # Recompute inexpensive guards to rebuild runtime objects and validate
        # segment alignment, even on a cached run; only rewrite on invalidation.
        started=begin('Sensitive-change checks')
        run.guarded=guard_refinement(run.raw,run.candidate,protected_names=run.names)
        if not _valid_stage(run,'guards',fp):
            invalidate('guards')
            save_text(run.directory/'refined_transcript.json',run.guarded.transcript.model_dump_json(indent=2))
            corrections=run.guarded.corrections
            save_json(run.directory/'guard_report.json',{'protected_names':run.names,
                'raw_sha256':file_hash(run.directory/'raw_transcript.json'),
                'candidate_sha256':file_hash(run.directory/'candidate_refined_transcript.json'),
                'refined_sha256':file_hash(run.directory/'refined_transcript.json'),
                'changed_segments_applied':len({c.segment_id for c in corrections if c.status.value=='accepted'}),
                'segments_needing_review':len({c.segment_id for c in corrections if c.status.value=='needs_review'}),
                'corrections':[c.model_dump(mode='json') for c in corrections],'edits':run.guarded.edits})
            done('guards',fp,['refined_transcript.json','guard_report.json'],started)
        # All downstream consumers use precisely the persisted transcript.
        run.guarded.transcript=Transcript.model_validate_json((run.directory/'refined_transcript.json').read_text(encoding='utf-8'))
        fp=stage_fingerprint('minutes',{'refined':file_hash(run.directory/'refined_transcript.json'),
            'model':cfg['minutes_model'],'provider':'groq','temperature':0.0,'groq':cfg.get('groq',{})})
        if not cached('minutes',fp):
            invalidate('minutes')
            started=begin('Meeting documentation')
            if minutes_provider_factory is None:raise ValueError('Minutes provider is unavailable.')
            main=minutes_provider_factory(False);grouping=minutes_provider_factory(True)
            record,details=generate_documentation(run.guarded.transcript,run.directory,
                provider=main,grouping_provider=grouping,model=cfg['minutes_model'],on_status=on_status)
            save_text(run.directory/'candidate_meeting_record.json',record.model_dump_json(indent=2))
            save_json(run.directory/'minutes_meta.json',{'status':'candidate_evidence_unverified',
                'provider':'groq','model':cfg['minutes_model'],'temperature':0.0,
                'source_sha256':file_hash(run.directory/'refined_transcript.json'),
                'record_sha256':file_hash(run.directory/'candidate_meeting_record.json'),
                'workflow':details,'created':utc_now()})
            done('minutes',fp,['candidate_meeting_record.json','minutes_meta.json'],started)
        run.candidate_record=MeetingRecord.model_validate_json((run.directory/'candidate_meeting_record.json').read_text(encoding='utf-8'))
        started=begin('Citation checks')
        fp=stage_fingerprint('evidence',{'refined':file_hash(run.directory/'refined_transcript.json'),
                                     'candidate':file_hash(run.directory/'candidate_meeting_record.json')})
        run.checked=verify_evidence(run.candidate_record,run.guarded.transcript)
        if not _valid_stage(run,'evidence',fp):
            save_text(run.directory/'citation_checked_meeting_record.json',run.checked.record.model_dump_json(indent=2))
            run.checked.report.update(source_sha256=file_hash(run.directory/'refined_transcript.json'),
                candidate_sha256=file_hash(run.directory/'candidate_meeting_record.json'),
                checked_record_sha256=file_hash(run.directory/'citation_checked_meeting_record.json'))
            save_json(run.directory/'evidence_report.json',run.checked.report)
            done('evidence',fp,['citation_checked_meeting_record.json','evidence_report.json'],started)
        run.status='complete';run.stage='Complete';attempt['status']='complete'
    except Exception as exc:
        run.status='failed';run.error=str(exc);attempt.update(status='failed',failed_stage=run.stage)
        raise
    except BaseException:
        run.status='interrupted';run.error='Processing interrupted; saved checkpoints retained.'
        attempt['status']='interrupted';raise
    finally:
        attempt['finished']=utc_now()
        try:
            save_run_state(run)
        finally:
            lock.unlink(missing_ok=True)
