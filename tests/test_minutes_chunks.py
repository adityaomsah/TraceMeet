import json

import pytest

from tracemeet.schemas import Segment, Transcript
from tracemeet.stages.minutes_chunks import plan_minutes_chunks


class FakeEstimator:
    def __init__(self, limit):
        self.limit = limit

    def estimate_request(self, *, system, prompt, schema):
        payload = json.loads(prompt)
        size = sum(
            len(segment["text"])
            for group in payload.values()
            for segment in group
        )
        return {
            "estimated_total_tokens": size,
            "fits": size <= self.limit,
        }


def make_transcript(count=7):
    return Transcript(
        segments=[
            Segment(
                id=f"seg_{index + 1:04d}",
                start=index,
                end=index + 1,
                text=f"Original sentence {index}.",
            )
            for index in range(count)
        ]
    )


def test_targets_cover_source_once_and_context_overlaps():
    raw = make_transcript()

    chunks = plan_minutes_chunks(
        raw,
        FakeEstimator(10000),
        system="test",
        max_target_segments=3,
        context_size=2,
    )

    assert [len(chunk.target_ids) for chunk in chunks] == [3, 3, 1]
    assert [
        segment_id
        for chunk in chunks
        for segment_id in chunk.target_ids
    ] == [segment.id for segment in raw.segments]

    assert chunks[1].context_before_ids == ("seg_0002", "seg_0003")
    assert chunks[0].context_after_ids == ("seg_0004", "seg_0005")

    targets = [
        item
        for chunk in chunks
        for item in json.loads(chunk.prompt)["target_segments"]
    ]
    assert targets == [
        {"id": segment.id, "text": segment.text}
        for segment in raw.segments
    ]


def test_budget_shrinks_chunk_without_losing_segments():
    raw = make_transcript(5)
    two_segments = sum(len(segment.text) for segment in raw.segments[:2])

    chunks = plan_minutes_chunks(
        raw,
        FakeEstimator(two_segments),
        system="test",
        max_target_segments=5,
        context_size=0,
    )

    assert [len(chunk.target_ids) for chunk in chunks] == [2, 2, 1]
    assert all(chunk.budget["fits"] for chunk in chunks)


def test_oversized_single_segment_is_not_truncated():
    with pytest.raises(ValueError, match="No text was truncated"):
        plan_minutes_chunks(
            make_transcript(1),
            FakeEstimator(1),
            system="test",
        )


def test_empty_transcript_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        plan_minutes_chunks(
            Transcript(segments=[]),
            FakeEstimator(100),
            system="test",
        )