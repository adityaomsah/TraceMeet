import json

import pytest
from pydantic import ValidationError

from tracemeet.schemas import Segment, Transcript
from tracemeet.stages.minutes_reconcile import (
    GroupDraft,
    ReconciliationError,
    build_group_jobs,
    materialize_record,
)


@pytest.fixture
def case():
    transcript = Transcript(
        segments=[
            Segment(
                id="seg_0001", start=0, end=2,
                text="Alice is presenting.",
            ),
            Segment(
                id="seg_0002", start=2, end=4,
                text="The report must be sent by Monday.",
            ),
            Segment(
                id="seg_0003", start=4, end=6,
                text="No owner has been assigned.",
            ),
        ],
        stt_model="test",
    )
    observations = [
        {
            "observation_id": "note_1",
            "kind": "task",
            "statement": "Send the report by Monday.",
            "evidence": [
                {
                    "segment_id": "seg_0002",
                    "quote": "The report must be sent by Monday.",
                }
            ],
        }
    ]
    groups = [
        {
            "group_id": "group_001",
            "label": "Report",
            "observation_ids": ["note_1"],
        }
    ]
    job = build_group_jobs(transcript, observations, groups)[0]
    return transcript, observations, groups, job


def task_draft(**overrides):
    task = {
        "description": "Send the report",
        "status": "confirmed",
        "owner": None,
        "deadline": "Monday",
        "evidence": ["seg_0002"],
        "owner_evidence": [],
        "deadline_evidence": ["seg_0002"],
    }
    task.update(overrides)
    return GroupDraft(
        minutes=[],
        decisions=[],
        action_items=[task],
        open_questions=[],
    )


def test_context_is_included_in_source_order(case):
    transcript, _, _, job = case
    payload = json.loads(job.prompt)

    assert job.core_ids == ("seg_0002",)
    assert payload["source_columns"] == ["id", "text"]
    assert payload["source_segments"] == [
        [segment.id, segment.text]
        for segment in transcript.segments
    ]


def test_quotes_come_from_source_and_unknown_owner_stays_null(case):
    transcript, _, _, job = case
    record = materialize_record(task_draft(), job, transcript)
    task = record.action_items[0]

    assert task.owner is None
    assert task.owner_evidence == []
    assert task.deadline == "Monday"
    assert task.evidence[0].quote == transcript.segments[1].text


def test_context_only_primary_claim_is_rejected(case):
    transcript, _, _, job = case

    with pytest.raises(ReconciliationError, match="core"):
        materialize_record(
            task_draft(evidence=["seg_0001"]),
            job,
            transcript,
        )


def test_owner_without_owner_evidence_is_rejected(case):
    transcript, _, _, job = case

    with pytest.raises(ValidationError):
        materialize_record(
            task_draft(owner="Alice"),
            job,
            transcript,
        )


def test_missing_group_membership_is_rejected(case):
    transcript, observations, _, _ = case
    groups = [
        {
            "group_id": "group_001",
            "label": "Report",
            "observation_ids": [],
        }
    ]

    with pytest.raises(ReconciliationError):
        build_group_jobs(transcript, observations, groups)


def test_unknown_evidence_id_is_rejected(case):
    transcript, _, _, job = case

    with pytest.raises(ReconciliationError, match="outside"):
        materialize_record(
            task_draft(evidence=["seg_9999"]),
            job,
            transcript,
        )


def test_schema_compaction_preserves_real_title_field():
    schema = GroupDraft.model_json_schema()
    topic = schema["$defs"]["TopicDraft"]

    assert "title" in topic["properties"]
    assert "title" in topic["required"]
    assert topic["properties"]["title"]["type"] == "string"
    assert topic["properties"]["title"]["minLength"] == 1

    # Descriptive metadata is removed, but the actual field survives.
    assert "title" not in topic
    assert "title" not in topic["properties"]["title"]