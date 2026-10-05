import pytest

from tracemeet.guards.sensitive import guard_refinement, sensitive_flags
from tracemeet.schemas import EditStatus, Segment, Transcript


def transcript(text, *, end=2.0):
    return Transcript(
        segments=[
            Segment(id="seg_0001", start=0.0, end=end, text=text)
        ],
        stt_model="test",
    )


@pytest.mark.parametrize(
    "raw,candidate,flag",
    [
        ("Budget is 15,000.", "Budget is 50,000.", "number_changed"),
        ("We need five servers.", "We need six servers.", "number_changed"),
        ("Use 5 then 10.", "Use 10 then 5.", "number_changed"),
        ("Do not deploy.", "Do deploy.", "negation_changed"),
        ("We can't deploy.", "We can deploy.", "negation_changed"),
        ("We should deploy.", "We will deploy.", "commitment_changed"),
        ("Delivery is Friday.", "Delivery is Monday.", "date_or_time_changed"),
        ("Wait 5 hours.", "Wait 5 days.", "unit_changed"),
        ("Budget is $500.", "Budget is ₹500.", "currency_or_percent_changed"),
        ("Alice will send it.", "Bob will send it.", "capitalized_token_changed"),
    ],
)
def test_sensitive_examples_are_flagged(raw, candidate, flag):
    assert flag in sensitive_flags(raw, candidate)


def test_protected_name_matching_handles_lowercase():
    flags = sensitive_flags(
        "aditya will send it.",
        "rahul will send it.",
        protected_names=["Aditya"],
    )
    assert "protected_name_changed" in flags


@pytest.mark.parametrize(
    "raw,candidate",
    [
        ("recent hiding seasons.", "recent hiring seasons."),
        ("data science and code engineering.", "data science and core engineering."),
    ],
)
def test_known_terminology_corrections_pass(raw, candidate):
    result = guard_refinement(transcript(raw), transcript(candidate))

    assert result.transcript.segments[0].text == candidate
    assert all(
        correction.status == EditStatus.ACCEPTED
        for correction in result.corrections
    )
    assert all(edit["applied"] for edit in result.edits)


def test_risky_segment_is_retained_whole_without_mutating_inputs():
    raw = transcript("Use code engineering with a budget of 15,000.")
    candidate = transcript("Use core engineering with a budget of 50,000.")
    raw_before = raw.model_dump()
    candidate_before = candidate.model_dump()

    result = guard_refinement(raw, candidate)

    assert result.transcript.full_text() == raw.full_text()
    assert raw.model_dump() == raw_before
    assert candidate.model_dump() == candidate_before
    assert all(
        correction.status == EditStatus.NEEDS_REVIEW
        for correction in result.corrections
    )
    assert all(not edit["applied"] for edit in result.edits)


def test_unchanged_segment_has_no_corrections():
    raw = transcript("Do not deploy before Friday.")
    result = guard_refinement(raw, raw)

    assert result.corrections == []
    assert result.edits == []
    assert result.transcript.full_text() == raw.full_text()


def test_timestamp_changes_are_rejected():
    with pytest.raises(ValueError, match="Timestamps changed"):
        guard_refinement(
            transcript("Hello.", end=2.0),
            transcript("Hello.", end=3.0),
        )