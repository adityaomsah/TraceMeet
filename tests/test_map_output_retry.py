import json

import pytest

from tracemeet.llm.base import LLMError
from tracemeet.llm.groq_provider import (
    GeneratedOutputError,
    is_generated_schema_failure,
)
from tracemeet.schemas import Segment, Transcript
from tracemeet.stages import minutes_map as mm


class FakeProvider:
    name = "groq"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.last_call = {}

    def estimate_request(self, **kwargs):
        return {
            "fits": True,
            "estimated_total_tokens": 100,
        }

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        self.last_call = {"test_call": len(self.calls)}
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def valid_output():
    return mm.ExtractedNotes(
        observations=[
            mm.ExtractedObservation(
                kind="task",
                statement="Alice will send the report.",
                evidence_segment_ids=["seg_0001"],
            )
        ]
    )


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    transcript = Transcript(
        segments=[
            Segment(
                id="seg_0001",
                start=0,
                end=5,
                text="Alice will send the report.",
            )
        ],
        stt_model="test",
    )
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Extract observations.", encoding="utf-8")

    waits = []
    monkeypatch.setattr(
        mm,
        "wait_for_request_gap",
        lambda last, gap, status: waits.append(last),
    )

    def run(provider):
        return mm.extract_meeting_notes(
            transcript,
            provider,
            model="openai/gpt-oss-120b",
            checkpoint_dir=tmp_path / "chunks",
            prompt_path=prompt,
            legacy_prompt_path=tmp_path / "absent.txt",
        )

    return run, waits, tmp_path / "chunks"


def test_only_specific_generated_schema_400_is_recognized():
    message = {
        "message": "Generated JSON does not match the expected schema."
    }
    assert is_generated_schema_failure(400, message)
    assert not is_generated_schema_failure(401, message)
    assert not is_generated_schema_failure(
        400, {"message": "Invalid response_format schema."}
    )


def test_schema_retry_succeeds_then_reuses_checkpoint(setup_run):
    run, waits, folder = setup_run
    provider = FakeProvider([
        GeneratedOutputError("Bad enum"),
        valid_output(),
    ])

    result = run(provider)

    assert len(provider.calls) == 2
    assert result[0].input_segment_ids == ["seg_0001"]
    assert "OUTPUT VALIDATION RETRY" in provider.calls[1]["system"]
    assert waits[0] is None
    assert waits[1] is not None

    failed = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in folder.glob("*.attempt.json")
    ]
    assert any(
        item["status"] == "validation_failed"
        and item["failure_type"] == "schema"
        for item in failed
    )

    run(provider)
    assert len(provider.calls) == 2


def test_two_schema_failures_stop_without_success_checkpoint(setup_run):
    run, _, folder = setup_run
    provider = FakeProvider([
        GeneratedOutputError("Bad enum"),
        GeneratedOutputError("Bad enum again"),
    ])

    with pytest.raises(mm.MapError, match="2 attempts"):
        run(provider)

    assert len(provider.calls) == 2
    assert not [
        path for path in folder.glob("*.json")
        if not path.name.endswith(".attempt.json")
    ]


def test_schema_and_evidence_share_attempt_limit(setup_run):
    run, _, _ = setup_run
    invalid_evidence = mm.ExtractedNotes(
        observations=[
            mm.ExtractedObservation(
                kind="task",
                statement="Send the report.",
                evidence_segment_ids=["seg_9999"],
            )
        ]
    )
    provider = FakeProvider([
        GeneratedOutputError("Bad enum"),
        invalid_evidence,
    ])

    with pytest.raises(mm.MapError, match="2 attempts"):
        run(provider)

    assert len(provider.calls) == 2


def test_ordinary_provider_error_is_not_retried(setup_run):
    run, _, _ = setup_run
    provider = FakeProvider([LLMError("Invalid request configuration")])

    with pytest.raises(LLMError, match="Invalid request"):
        run(provider)

    assert len(provider.calls) == 1