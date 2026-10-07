"""Display names only: never rename run folders or change pipeline inputs."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def _object(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _clean(value, limit=120):
    return ' '.join(value.split())[:limit] if isinstance(value, str) else ''


def _matches(path, digest):
    if not isinstance(digest, str):
        return False
    try:
        return digest.lower() in {
            hashlib.sha256(path.read_bytes()).hexdigest(),
            hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest(),
        }
    except (OSError, ValueError):
        return False


def custom_title(directory):
    return _clean(_object(Path(directory) / 'display_meta.json').get('title'))


def save_title(directory, title):
    """An empty title restores the automatic label. No inference state is touched."""
    directory = Path(directory)
    target = directory / 'display_meta.json'
    metadata = _object(target)
    metadata['title'] = _clean(title)
    temporary = directory / f'display_meta.{uuid4().hex}.tmp'
    try:
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def run_label(directory):
    directory = Path(directory)
    meta = _object(directory / 'meta.json')
    state = _object(directory / 'run_state.json')
    date = 'Unknown date'
    try:
        created = datetime.fromisoformat(meta.get('created', '').replace('Z', '+00:00'))
        if created.tzinfo is not None:
            created = created.astimezone()  # local time of the Streamlit server
        date = created.strftime('%d %b %Y, %H:%M')
    except (ValueError, TypeError, AttributeError):
        try:
            date = datetime.strptime(directory.name[:15], '%Y%m%d_%H%M%S').strftime('%d %b %Y, %H:%M')
        except ValueError:
            pass
    title = custom_title(directory)
    if not title:
        provenance = _object(directory / 'minutes_meta.json')
        current = state.get('version') != 'pipeline-disk-v1' or 'minutes' in state.get('stages', {})
        if current and _matches(directory / 'candidate_meeting_record.json', provenance.get('record_sha256')) and _matches(directory / 'refined_transcript.json', provenance.get('source_sha256')):
            topics = _object(directory / 'candidate_meeting_record.json').get('minutes', [])
            if isinstance(topics, list):
                title = next((_clean(t.get('title')) for t in topics if isinstance(t, dict) and _clean(t.get('title'))), '')
    if not title:
        title = {'running': 'Processing', 'failed': 'Needs retry', 'interrupted': 'Interrupted',
                 'complete': 'Meeting'}.get(state.get('status'), 'Meeting')
    filename = _clean(meta.get('original_filename'), 180) or 'Unknown input'
    return f'{date} · {title} · {filename}'
