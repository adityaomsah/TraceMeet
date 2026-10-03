import pytest
from pydantic import ValidationError

from tracemeet.schemas import (
    ActionItem, Correction, EditStatus, Evidence, MeetingRecord,
    Segment, TaskStatus, Transcript,
)

# ---------- Segment ----------

def test_valid_segment_passes():
    s = Segment(id="seg_001", start=0.0, end=4.2, text="Let's discuss the upload feature.")
    assert s.end == 4.2


def test_end_before_start_fails():
    with pytest.raises(ValidationError):
        Segment(id="seg_001", start=5.0, end=2.0, text="Hello")


def test_negative_start_fails():
    with pytest.raises(ValidationError):
        Segment(id="seg_001", start=-1.0, end=2.0, text="Hello")


def test_empty_text_fails():
    with pytest.raises(ValidationError):
        Segment(id="seg_001", start=0.0, end=2.0, text="   ")


def test_text_is_stripped():
    s = Segment(id="seg_001", start=0.0, end=2.0, text="  Hello  ")
    assert s.text == "Hello"


def test_segment_is_immutable():
    s = Segment(id="seg_001", start=0, end=1, text="Hi")
    with pytest.raises(ValidationError):
        s.text = "Changed"


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Segment(id="seg_001", start=0, end=1, text="Hi", speaker="A")


def test_nan_timestamp_rejected():
    with pytest.raises(ValidationError):
        Segment(id="seg_001", start=float("nan"), end=1, text="Hi")


# ---------- Transcript ----------

def test_transcript_lookup_by_id():
    t = Transcript(segments=[Segment(id="seg_001", start=0, end=1, text="Hi")])
    assert t.by_id()["seg_001"].text == "Hi"
    assert t.full_text() == "Hi"


def test_duplicate_ids_rejected():
    seg = Segment(id="seg_001", start=0, end=1, text="Hi")
    with pytest.raises(ValidationError):
        Transcript(segments=[seg, seg])


# ---------- ActionItem and MeetingRecord ----------

EV = Evidence(segment_id="seg_001", quote="Priya will send the report")


def make_task(**kwargs):
    base = dict(description="Send report", status=TaskStatus.CONFIRMED, evidence=[EV])
    base.update(kwargs)
    return ActionItem(**base)


def test_blank_owner_becomes_none():
    assert make_task(owner="Unspecified").owner is None


def test_owner_without_evidence_is_rejected():
    with pytest.raises(ValidationError):
        make_task(owner="Priya")


def test_owner_with_evidence_is_accepted():
    assert make_task(owner="Priya", owner_evidence=[EV]).owner == "Priya"


def test_evidence_without_owner_is_rejected():
    with pytest.raises(ValidationError):
        make_task(owner_evidence=[EV])


def test_deadline_without_evidence_is_rejected():
    with pytest.raises(ValidationError):
        make_task(deadline="Friday")


def test_task_status_is_required():
    with pytest.raises(ValidationError):
        ActionItem(description="Send report", evidence=[EV])


def test_record_requires_all_lists():
    with pytest.raises(ValidationError):
        MeetingRecord(summary="A short meeting.")


def test_record_accepts_empty_lists():
    r = MeetingRecord(
        summary="A short meeting.",
        minutes=[], decisions=[], action_items=[], open_questions=[],
    )
    assert r.decisions == []

# ---------- Correction ----------

def test_correction_status_is_required():
    with pytest.raises(ValidationError, match="status"):
        Correction(segment_id="seg_001", original="cube control", replacement="kubectl")


def test_correction_with_status_is_accepted():
    c = Correction(
        segment_id="seg_001", original="cube control", replacement="kubectl",
        status=EditStatus.ACCEPTED,
    )
    assert c.flags == []