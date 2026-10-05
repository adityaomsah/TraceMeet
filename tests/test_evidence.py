import pytest

from tracemeet.guards.evidence import check_citation, verify_evidence
from tracemeet.schemas import Evidence, MeetingRecord, Segment, Transcript


@pytest.fixture
def transcript():
    return Transcript(
        segments=[
            Segment(
                id="seg_0001",
                start=1.5,
                end=5.0,
                text="We should not deploy on Friday.",
            ),
            Segment(
                id="seg_0002",
                start=5.0,
                end=10.0,
                text="Alice will send the report by Monday.",
            ),
            Segment(
                id="seg_0003",
                start=10.0,
                end=12.0,
                text="Annette will attend.",
            ),
        ]
    )


def evidence(segment_id, quote):
    return {"segment_id": segment_id, "quote": quote}


def record(**overrides):
    data = {
        "summary": "The group discussed deployment and a report.",
        "minutes": [],
        "decisions": [],
        "action_items": [],
        "open_questions": [],
    }
    data.update(overrides)
    return MeetingRecord.model_validate(data)


def task(**overrides):
    data = {
        "description": "Send the report.",
        "status": "confirmed",
        "owner": "Alice",
        "deadline": "Monday",
        "evidence": [
            evidence("seg_0002", "Alice will send the report by Monday.")
        ],
        "owner_evidence": [evidence("seg_0002", "Alice")],
        "deadline_evidence": [evidence("seg_0002", "Monday")],
    }
    data.update(overrides)
    return data


def test_exact_quote_has_correct_offsets_and_playback_times(transcript):
    quote = "send the report"
    result = check_citation(
        Evidence(segment_id="seg_0002", quote=quote), transcript
    )

    assert result["status"] == "exact_match"
    source = transcript.by_id()["seg_0002"].text
    assert source[result["quote_start"]:result["quote_end"]] == quote
    assert result["start_s"] == 5.0
    assert result["end_s"] == 10.0


@pytest.mark.parametrize(
    "segment_id,quote,status",
    [
        ("missing", "Alice", "unknown_segment"),
        (
            "seg_0001",
            "We should deploy on Friday.",
            "quote_mismatch",
        ),
        ("seg_0003", "Ann", "quote_mismatch"),
        ("seg_0002", "...", "empty_or_punctuation_only"),
    ],
)
def test_invalid_citations(transcript, segment_id, quote, status):
    result = check_citation(
        Evidence(segment_id=segment_id, quote=quote), transcript
    )
    assert result["status"] == status
    assert result["start_s"] is None


def test_omitting_trailing_punctuation_can_still_match(transcript):
    result = check_citation(
        Evidence(
            segment_id="seg_0001",
            quote="We should not deploy on Friday",
        ),
        transcript,
    )
    assert result["status"] == "exact_match"


def test_invalid_decision_is_dropped_without_mutating_candidate(transcript):
    candidate = record(
        decisions=[
            {
                "text": "Deploy on Friday.",
                "status": "decided",
                "evidence": [
                    evidence("seg_0001", "We should deploy on Friday.")
                ],
            }
        ]
    )
    before = candidate.model_dump()

    result = verify_evidence(candidate, transcript)

    assert result.record.decisions == []
    assert result.report["counts"]["dropped_items"] == 1
    assert candidate.model_dump() == before


def test_one_invalid_required_quote_drops_whole_item(transcript):
    candidate = record(
        minutes=[
            {
                "title": "Deployment",
                "summary": "Deployment was discussed.",
                "evidence": [
                    evidence("seg_0001", "We should not deploy on Friday."),
                    evidence("missing", "Everyone agreed."),
                ],
            }
        ]
    )

    result = verify_evidence(candidate, transcript)
    assert result.record.minutes == []


@pytest.mark.parametrize(
    "field,bad_quote",
    [("owner", "Bob"), ("deadline", "Tuesday")],
)
def test_invalid_field_citation_clears_only_that_field(
    transcript, field, bad_quote
):
    candidate = record(
        action_items=[
            task(
                **{
                    field: bad_quote,
                    f"{field}_evidence": [
                        evidence("seg_0002", bad_quote)
                    ],
                }
            )
        ]
    )

    result = verify_evidence(candidate, transcript)
    retained = result.record.action_items[0]

    assert getattr(retained, field) is None
    assert getattr(retained, f"{field}_evidence") == []
    assert retained.description == "Send the report."
    assert result.report["counts"]["cleared_fields"] == 1

    other_field = "deadline" if field == "owner" else "owner"
    assert getattr(retained, other_field) == (
        "Monday" if other_field == "deadline" else "Alice"
    )


def test_invalid_task_evidence_drops_task(transcript):
    candidate = record(
        action_items=[
            task(evidence=[evidence("missing", "Send the report.")])
        ]
    )

    result = verify_evidence(candidate, transcript)
    assert result.record.action_items == []


def test_valid_task_is_preserved_and_summary_remains_flagged(transcript):
    candidate = record(action_items=[task()])
    result = verify_evidence(candidate, transcript)

    assert result.record.action_items == candidate.action_items
    assert result.report["counts"]["invalid_citations"] == 0
    assert result.report["summary_status"] == "uncited"
    assert result.report["semantic_support_checked"] is False


def test_exact_citation_does_not_claim_semantic_verification(transcript):
    # The quote exists, but the decision contradicts it.
    # This deliberately documents the verifier's scope.
    candidate = record(
        decisions=[
            {
                "text": "Deploy on Friday.",
                "status": "decided",
                "evidence": [
                    evidence("seg_0001", "We should not deploy on Friday.")
                ],
            }
        ]
    )

    result = verify_evidence(candidate, transcript)

    assert result.report["counts"]["invalid_citations"] == 0
    assert result.report["semantic_support_checked"] is False


def test_quote_search_skips_partial_word_and_finds_later_match():
    source = Transcript(
        segments=[
            Segment(
                id="seg_0001",
                start=0,
                end=3,
                text="Annette introduced Ann.",
            )
        ]
    )

    result = check_citation(
        Evidence(segment_id="seg_0001", quote="Ann"),
        source,
    )

    assert result["status"] == "exact_match"
    assert result["quote_start"] == 19
    assert result["quote_end"] == 22