import json

import pytest

from tracemeet.schemas import Evidence, MinutesTopic, Segment, Transcript
from tracemeet.stages.minutes_finalize import (
    CondensedTopic,
    FinalizationError,
    OverviewDraft,
    prepare_review,
    restore_overview,
)
from tracemeet.stages.minutes_reconcile import GroupRecord


def test_review_includes_context_for_owner_attribution():
    transcript = Transcript(
        segments=[
            Segment(
                id="seg_0001", start=0, end=2,
                text="Jay, this concerns your refinement work.",
            ),
            Segment(
                id="seg_0002", start=2, end=4,
                text="You will report back tomorrow.",
            ),
        ]
    )
    record = GroupRecord.model_validate({
        "minutes": [],
        "decisions": [],
        "action_items": [{
            "description": "Report back",
            "status": "confirmed",
            "owner": None,
            "deadline": None,
            "evidence": [{
                "segment_id": "seg_0002",
                "quote": "You will report back tomorrow.",
            }],
            "owner_evidence": [],
            "deadline_evidence": [],
        }],
        "open_questions": [],
    })

    job = prepare_review("tasks", [record], transcript)
    source = json.loads(job.prompt)["source_segments"]

    assert source == [
        ["seg_0001", "Jay, this concerns your refinement work."],
        ["seg_0002", "You will report back tomorrow."],
    ]


def test_overview_restores_source_evidence_and_deduplicates():
    evidence = Evidence(segment_id="seg_0001", quote="Discuss the report.")
    topics = {
        "topic_001": MinutesTopic(
            title="Report", summary="Report discussion.", evidence=[evidence]
        ),
        "topic_002": MinutesTopic(
            title="More report", summary="More discussion.", evidence=[evidence]
        ),
    }
    draft = OverviewDraft(
        summary="The report was discussed.",
        minutes=[CondensedTopic(
            title="Report",
            summary="The report was discussed.",
            source_topic_ids=["topic_001", "topic_002"],
        )],
    )

    result = restore_overview(draft, topics)
    assert len(result[0].evidence) == 1
    assert result[0].evidence[0].quote == evidence.quote


def test_overview_rejects_invented_topic_reference():
    draft = OverviewDraft(
        summary="Example",
        minutes=[CondensedTopic(
            title="Example",
            summary="Example",
            source_topic_ids=["topic_999"],
        )],
    )

    with pytest.raises(FinalizationError, match="Unknown"):
        restore_overview(draft, {})


def test_overview_schema_retains_actual_title_property():
    schema = OverviewDraft.model_json_schema()
    topic = schema["$defs"]["CondensedTopic"]

    assert "title" in topic["properties"]
    assert "title" in topic["required"]