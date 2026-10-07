import pytest

import tracemeet.pipeline as pipeline
from tracemeet.llm.base import LLMTemporaryError
from tracemeet.schemas import MeetingRecord, Segment, Transcript


def test_minutes_failure_retains_completed_stages_and_retry_reuses_them(
    tmp_path, monkeypatch
):
    # Exercise real fingerprint logic against an isolated dependency tree.
    root = tmp_path / "source"
    monkeypatch.setattr(pipeline, "ROOT", root)
    for paths in pipeline.STAGE_FILES.values():
        for name in paths:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test dependency", encoding="utf-8")

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
        def __init__(self, name):
            self.name = name

    refiner = FakeProvider("gemini")
    documenter = FakeProvider("groq")

    def fake_refine(*args, **kwargs):
        assert args[1] is refiner
        calls["refine"] += 1
        return transcript

    def fake_minutes(*args, **kwargs):
        assert kwargs["provider"] is documenter
        calls["minutes"] += 1
        if calls["minutes"] == 1:
            raise LLMTemporaryError("simulated unavailable service")
        return record, {"path": "short"}

    monkeypatch.setattr(pipeline, "validate_media", lambda path: None)
    monkeypatch.setattr(pipeline, "refine_transcript", fake_refine)
    monkeypatch.setattr(pipeline, "generate_documentation", fake_minutes)

    cfg = {
        "stt": {"model": "fake", "device": "cpu", "compute_type": "int8"},
        "llm": {
            "refine_provider": "gemini",
            "minutes_provider": "groq",
            "refine_model": "fake-refiner",
            "minutes_model": "fake-minutes",
            "temperature": 1.0,
        },
    }
    run = pipeline.create_run(tmp_path, media, "input.wav", cfg, "", [])

    kwargs = {
        "refine_provider_factory": lambda: refiner,
        "minutes_provider_factory": lambda grouping: documenter,
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

    # Reconstruct from disk as after a browser/server restart.
    run = pipeline.recover_run(tmp_path)
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