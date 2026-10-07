import csv
import io
import json
import zipfile
from hashlib import sha256

import pytest

from tracemeet.export.bundle import build_exports
from tracemeet.export.csv_export import CSV_COLUMNS, spreadsheet_safe
from tracemeet.schemas import MeetingRecord, Segment, Transcript


@pytest.fixture
def sample():
    transcript = Transcript(
        segments=[
            Segment(
                id="seg_0001",
                start=0,
                end=5,
                text="Alice will send the report by Monday.",
            )
        ],
        stt_model="test",
    )
    evidence = {
        "segment_id": "seg_0001",
        "quote": "Alice will send the report by Monday.",
    }
    record = MeetingRecord.model_validate({
        "summary": "Alice will send the report.",
        "minutes": [],
        "decisions": [],
        "action_items": [
            {
                "description": 'Send the report, including "results"\nand notes',
                "status": "confirmed",
                "owner": "Alice",
                "deadline": "Monday",
                "evidence": [evidence],
                "owner_evidence": [evidence],
                "deadline_evidence": [evidence],
            }
        ],
        "open_questions": [],
    })
    return record, transcript


def export(record, transcript):
    return build_exports(
        record,
        raw_transcript=transcript,
        refined_transcript=transcript,
    )


def csv_rows(bundle):
    return list(csv.DictReader(
        io.StringIO(
            bundle.files["tasks.csv"].decode("utf-8-sig"),
            newline="",
        )
    ))


def test_formats_preserve_task_fields_and_csv_quoting(sample):
    record, transcript = sample
    bundle = export(record, transcript)

    saved = MeetingRecord.model_validate_json(
        bundle.files["meeting_record.json"]
    )
    rows = csv_rows(bundle)
    task = saved.action_items[0]

    assert len(rows) == len(saved.action_items) == 1
    assert rows[0]["description"] == task.description
    assert rows[0]["status"] == task.status.value
    assert rows[0]["owner"] == task.owner
    assert rows[0]["deadline"] == task.deadline

    markdown = bundle.files["meeting_record.md"].decode("utf-8")
    assert "task_001" in markdown
    assert "confirmed" in markdown
    assert "Alice" in markdown
    assert "Monday" in markdown


def test_invalid_primary_citation_removes_task_from_every_format(sample):
    record, transcript = sample
    data = record.model_dump(mode="json")
    data["action_items"][0]["evidence"][0]["quote"] = "Not in the source"
    candidate = MeetingRecord.model_validate(data)

    bundle = export(candidate, transcript)
    assert bundle.record.action_items == []
    assert csv_rows(bundle) == []
    assert "task_001" not in bundle.files["meeting_record.md"].decode()

    # Exporting does not mutate the input candidate.
    assert len(candidate.action_items) == 1


def test_missing_assignment_stays_null_in_json(sample):
    record, transcript = sample
    data = record.model_dump(mode="json")
    task = data["action_items"][0]
    task.update(
        owner=None,
        deadline=None,
        owner_evidence=[],
        deadline_evidence=[],
    )

    bundle = export(MeetingRecord.model_validate(data), transcript)
    saved = json.loads(bundle.files["meeting_record.json"])
    row = csv_rows(bundle)[0]

    assert saved["action_items"][0]["owner"] is None
    assert saved["action_items"][0]["deadline"] is None
    assert row["owner"] == "Unspecified"
    assert row["deadline"] == "Unspecified"


def test_zip_contains_exact_files_and_manifest_hashes(sample):
    record, transcript = sample
    bundle = export(record, transcript)

    with zipfile.ZipFile(io.BytesIO(bundle.zip_bytes)) as archive:
        assert set(archive.namelist()) == set(bundle.files)
        for name, content in bundle.files.items():
            assert archive.read(name) == content

    manifest = json.loads(bundle.files["manifest.json"])
    for name, digest in manifest["sha256"].items():
        assert sha256(bundle.files[name]).hexdigest() == digest


def test_empty_task_csv_still_has_headers(sample):
    record, transcript = sample
    data = record.model_dump(mode="json")
    data["action_items"] = []

    bundle = export(MeetingRecord.model_validate(data), transcript)
    reader = csv.DictReader(io.StringIO(
        bundle.files["tasks.csv"].decode("utf-8-sig")
    ))

    assert reader.fieldnames == CSV_COLUMNS
    assert list(reader) == []


@pytest.mark.parametrize("value", ["=1+1", " +SUM(A1:A2)", "@example"])
def test_formula_like_csv_cells_are_escaped(value):
    assert spreadsheet_safe(value) == "'" + value