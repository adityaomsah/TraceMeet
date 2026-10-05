import pytest

from tracemeet.guards.diff import edit_metrics, segment_edits
from tracemeet.guards.sensitive import guard_refinement, sensitive_flags
from tracemeet.schemas import EditStatus, Segment, Transcript


@pytest.mark.parametrize(
    "original,candidate",
    [
        ("I cannot deploy it.", "I can not deploy it."),
        ("I can't deploy it.", "I cannot deploy it."),
        ("I can’t deploy it.", "I can not deploy it."),
        ("Cannot deploy it.", "Can not deploy it."),
        ("Can't deploy it.", "Cannot deploy it."),
        ("We won't deploy.", "We will not deploy."),
        ("We shouldn't deploy.", "We should not deploy."),
    ],
)
def test_equivalent_negation_spellings_do_not_trigger_flags(
    original, candidate
):
    assert sensitive_flags(original, candidate) == []


def test_removing_one_of_two_negations_is_still_detected():
    original = "Do not deploy. Do not delete the backup."
    candidate = "Do deploy. Do not delete the backup."

    assert "negation_changed" in sensitive_flags(original, candidate)


def test_cannot_to_can_still_triggers_negation_guard():
    flags = sensitive_flags("I cannot deploy.", "I can deploy.")
    assert "negation_changed" in flags


@pytest.mark.parametrize(
    "original,candidate,expected_flag",
    [
        ("Value is 1.5.", "Value is 15.", "number_changed"),
        ("Value is -5.", "Value is 5.", "number_changed"),
        ("Limit is 50%.", "Limit is 50.", "currency_or_percent_changed"),
    ],
)
def test_meaningful_punctuation_is_not_stripped(
    original, candidate, expected_flag
):
    assert expected_flag in sensitive_flags(original, candidate)


def test_whitespace_edit_is_labelled_and_offsets_remain_exact():
    original = "Use  the API."
    candidate = "Use the API."
    edits = segment_edits("seg_0001", original, candidate)

    assert len(edits) == 1
    assert edits[0]["whitespace_only"] is True
    assert edit_metrics(edits)["whitespace_only_spans"] == 1

    for edit in edits:
        assert original[
            edit["raw_start"]:edit["raw_end"]
        ] == edit["original"]
        assert candidate[
            edit["candidate_start"]:edit["candidate_end"]
        ] == edit["replacement"]


def test_punctuation_edit_is_not_labelled_whitespace_only():
    edits = segment_edits("seg_0001", "Use API, please.", "Use API please.")

    assert edits
    assert all(not edit["whitespace_only"] for edit in edits)


def test_insertion_metrics_count_new_tokens():
    edits = segment_edits("seg_0001", "deploy", "deploy now")
    metrics = edit_metrics(edits)

    assert metrics["inserted_tokens"] == 1
    assert metrics["deleted_tokens"] == 0


def test_whitespace_label_does_not_bypass_segment_guard():
    def transcript(text):
        return Transcript(
            segments=[
                Segment(id="seg_0001", start=0, end=1, text=text)
            ]
        )

    raw = transcript("Do  not deploy.")
    candidate = transcript("Do deploy.")
    result = guard_refinement(raw, candidate)

    assert result.transcript.full_text() == raw.full_text()
    assert all(
        correction.status == EditStatus.NEEDS_REVIEW
        for correction in result.corrections
    )