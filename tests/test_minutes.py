import json

import pytest

from tracemeet.schemas import MeetingRecord, Segment, Transcript
from tracemeet.stages.minutes import MinutesError, generate_minutes


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.calls = []

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["schema"].model_validate(
            {
                "summary": "The speaker introduced the project.",
                "minutes": [
                    {
                        "title": "Project introduction",
                        "summary": "The speaker introduced TraceMeet.",
                        "evidence": [
                            {
                                "segment_id": "seg_0001",
                                "quote": "This project is called TraceMeet.",
                            }
                        ],
                    }
                ],
                "decisions": [],
                "action_items": [],
                "open_questions": [],
            }
        )


@pytest.fixture
def transcript():
    return Transcript(
        segments=[
            Segment(
                id="seg_0001",
                start=0,
                end=3,
                text="This project is called TraceMeet.",
            )
        ],
        stt_model="test",
    )


@pytest.fixture
def prompt_path(tmp_path):
    path = tmp_path / "minutes.txt"
    path.write_text("Produce a structured meeting record.", encoding="utf-8")
    return path


def test_stage_uses_selected_model_and_preserves_input(transcript, prompt_path):
    provider = FakeProvider()
    before = transcript.model_dump()

    result = generate_minutes(
        transcript,
        provider,
        model="separate-minutes-model",
        prompt_path=prompt_path,
    )

    assert isinstance(result, MeetingRecord)
    assert transcript.model_dump() == before
    assert len(provider.calls) == 1
    assert provider.calls[0]["model"] == "separate-minutes-model"
    assert provider.calls[0]["schema"] is MeetingRecord

    payload = json.loads(provider.calls[0]["prompt"])
    assert payload["segments"] == [
        {
            "id": "seg_0001",
            "text": "This project is called TraceMeet.",
        }
    ]

    # Empty extraction lists are valid.
    assert result.decisions == []
    assert result.action_items == []


def test_empty_transcript_fails_before_api_call(prompt_path):
    provider = FakeProvider()

    with pytest.raises(MinutesError, match="empty transcript"):
        generate_minutes(
            Transcript(segments=[]),
            provider,
            model="fake",
            prompt_path=prompt_path,
        )

    assert provider.calls == []


def test_oversized_input_is_rejected_without_truncation(
    transcript, prompt_path
):
    provider = FakeProvider()
    before = transcript.model_dump()

    with pytest.raises(MinutesError, match="not been truncated"):
        generate_minutes(
            transcript,
            provider,
            model="fake",
            prompt_path=prompt_path,
            max_chars=10,
        )

    assert provider.calls == []
    assert transcript.model_dump() == before


@pytest.mark.parametrize("empty_file", [False, True])
def test_missing_or_empty_prompt_fails_before_api_call(
    transcript, tmp_path, empty_file
):
    provider = FakeProvider()
    path = tmp_path / "prompt.txt"

    if empty_file:
        path.write_text("   ", encoding="utf-8")

    with pytest.raises(MinutesError):
        generate_minutes(
            transcript,
            provider,
            model="fake",
            prompt_path=path,
        )

    assert provider.calls == []