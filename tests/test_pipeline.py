import pytest

import tracemeet.pipeline as pipeline
from tracemeet.llm.base import LLMTemporaryError
from tracemeet.schemas import MeetingRecord, Segment, Transcript


def test_minutes_failure_retains_completed_stages_and_retry_reuses_them(
    tmp_path, monkeypatch
):
    media = tmp_path / "input.wav"
    media.write_bytes(b"fake audio for an isolated orchestration test")

    transcript = Transcript(
        segments=[
            Segment(id="seg_0001", start=0, end=1, text="Hello.")
        ],
        stt_model="fake",
    )
    record = MeetingRecord(
        summary="A greeting.",
        minutes=[],
        decisions=[],
        action_items=[],
        open_questions=[],
    )

    calls = {"stt": 0, "refine": 0, "minutes": 0}

    class FakeEngine:
        def transcribe(self, *args, **kwargs):
            calls["stt"] += 1
            return transcript

    class FakeProvider:
        name = "fake"

    def fake_refine(*args, **kwargs):
        calls["refine"] += 1
        return transcript

    def fake_minutes(*args, **kwargs):
        calls["minutes"] += 1
        if calls["minutes"] == 1:
            raise LLMTemporaryError("simulated unavailable service")
        return record

    monkeypatch.setattr(pipeline, "validate_media", lambda path: None)
    monkeypatch.setattr(pipeline, "refine_transcript", fake_refine)
    monkeypatch.setattr(pipeline, "generate_minutes", fake_minutes)

    cfg = {
        "stt": {"model": "fake", "device": "cpu", "compute_type": "int8"},
        "llm": {
            "refine_model": "fake-refiner",
            "minutes_model": "fake-minutes",
            "temperature": 1.0,
        },
    }
    run = pipeline.create_run(tmp_path, media, "input.wav", cfg, "", [])

    kwargs = {
        "provider": FakeProvider(),
        "engine_factory": FakeEngine,
        "on_status": lambda message: None,
        "on_progress": lambda value: None,
    }

    with pytest.raises(LLMTemporaryError):
        pipeline.continue_run(run, **kwargs)

    assert run.status == "failed"
    assert run.guarded is not None
    assert run.candidate_record is None
    assert run.checked is None
    assert (tmp_path / "refined_transcript.json").exists()
    assert not (tmp_path / "citation_checked_meeting_record.json").exists()

    pipeline.continue_run(run, **kwargs)

    assert calls == {"stt": 1, "refine": 1, "minutes": 2}
    assert run.status == "complete"
    assert run.checked is not None
    assert [attempt["status"] for attempt in run.attempts] == [
        "failed", "complete"
    ]

    # A completed run should not call any model again.
    pipeline.continue_run(run, **kwargs)
    assert calls == {"stt": 1, "refine": 1, "minutes": 2}