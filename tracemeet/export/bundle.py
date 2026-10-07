import io
import json
import zipfile
from dataclasses import dataclass
from hashlib import sha256

from tracemeet.export.csv_export import to_task_csv
from tracemeet.export.markdown import to_markdown
from tracemeet.guards.evidence import verify_evidence
from tracemeet.schemas import MeetingRecord, Transcript

EXPORT_VERSION = "exports-v1"


@dataclass
class ExportBundle:
    record: MeetingRecord
    files: dict[str, bytes]
    zip_bytes: bytes


def json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def build_exports(
    candidate: MeetingRecord,
    *,
    raw_transcript: Transcript,
    refined_transcript: Transcript,
) -> ExportBundle:
    # Revalidate snapshots rather than mutating the caller's objects.
    candidate = MeetingRecord.model_validate(
        candidate.model_dump(mode="json")
    )
    raw = Transcript.model_validate(
        raw_transcript.model_dump(mode="json")
    )
    refined = Transcript.model_validate(
        refined_transcript.model_dump(mode="json")
    )

    raw_identity = [
        (segment.id, segment.start, segment.end)
        for segment in raw.segments
    ]
    refined_identity = [
        (segment.id, segment.start, segment.end)
        for segment in refined.segments
    ]
    if raw_identity != refined_identity:
        raise ValueError(
            "Raw and refined transcript IDs and timestamps must match."
        )

    checked = verify_evidence(candidate, refined)
    record = MeetingRecord.model_validate(
        checked.record.model_dump(mode="json")
    )

    files = {
        "meeting_record.json": json_bytes(record.model_dump(mode="json")),
        "meeting_record.md": to_markdown(record, refined).encode("utf-8"),
        # UTF-8 BOM helps Windows spreadsheet applications identify encoding.
        "tasks.csv": to_task_csv(record).encode("utf-8-sig"),
        "raw_transcript.json": json_bytes(raw.model_dump(mode="json")),
        "refined_transcript.json": json_bytes(
            refined.model_dump(mode="json")
        ),
        "evidence_report.json": json_bytes(checked.report),
        "EXPORT_NOTES.txt": (
            "TraceMeet exports\n\n"
            "Markdown, task CSV and meeting JSON use the same "
            "citation-checked record.\n"
            "Citation matching does not prove semantic support, "
            "speaker identity or correct decision/task classification.\n"
            "The overall summary is uncited.\n"
            "Timestamps cover source segments, not individual words.\n"
            "Missing owners/deadlines are null in JSON and "
            "Unspecified in readable exports.\n"
            "CSV formula-like cells are prefixed with an apostrophe; "
            "meeting_record.json preserves original values.\n"
            "The recording itself is not included.\n"
        ).encode("utf-8"),
    }

    manifest = {
        "export_version": EXPORT_VERSION,
        "record_status": "citation_checked_semantics_unverified",
        "counts": {
            "minutes": len(record.minutes),
            "decisions": len(record.decisions),
            "action_items": len(record.action_items),
            "open_questions": len(record.open_questions),
        },
        "sha256": {
            name: sha256(content).hexdigest()
            for name, content in files.items()
        },
    }
    files["manifest.json"] = json_bytes(manifest)

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, content in files.items():
            archive.writestr(name, content)

    return ExportBundle(
        record=record,
        files=files,
        zip_bytes=buffer.getvalue(),
    )