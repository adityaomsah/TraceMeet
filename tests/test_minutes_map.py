import pytest

from tracemeet.schemas import Segment, Transcript
from tracemeet.stages.minutes_chunks import MinutesChunk
from tracemeet.stages.minutes_map import (
    LEGACY_VERSION,
    ExtractedNotes,
    ExtractedObservation,
    LegacySchema,
    MapError,
    attach_source_quotes,
    extract_meeting_notes,
    fingerprint,
    legacy_schema,
    save_json_atomic,
    settings_for,
    try_legacy_checkpoint,
)


def source():
    return Transcript(
        segments=[
            Segment(
                id="seg_0001", start=0, end=3,
                text="Do not deploy on Friday.",
            ),
            Segment(
                id="seg_0002", start=3, end=5,
                text="Alice will send the report.",
            ),
        ]
    )


def chunk():
    return MinutesChunk(
        index=1,
        target_ids=("seg_0001",),
        context_before_ids=(),
        context_after_ids=(),
        prompt="{}",
        budget={"fits": True, "estimated_total_tokens": 100},
    )


def extracted(ids=None):
    return ExtractedNotes(
        observations=[
            ExtractedObservation(
                kind="agreement",
                statement="Do not deploy on Friday.",
                evidence_segment_ids=(
                    ["seg_0001"] if ids is None else ids
                ),
            )
        ]
    )


class FakeProvider:
    name = "fake"
    last_call = {}

    def __init__(self):
        self.calls = 0

    def estimate_request(self, **kwargs):
        return {"fits": True, "estimated_total_tokens": 100}

    def generate_structured(self, **kwargs):
        self.calls += 1
        assert kwargs["schema"] is ExtractedNotes
        return extracted()


def test_python_records_input_and_retrieves_exact_quotes():
    result = attach_source_quotes(extracted(), chunk(), source())
    assert result.input_segment_ids == ["seg_0001"]
    assert result.observations[0].evidence[0].quote == (
        "Do not deploy on Friday."
    )
    assert "covered_segment_ids" not in (
        ExtractedNotes.model_json_schema()["properties"]
    )


@pytest.mark.parametrize(
    "ids, message",
    [
        (["seg_9999"], "Unknown evidence"),
        (["seg_0001", "seg_0002"], "outside this window"),
        (["seg_0001", "seg_0001"], "Duplicated"),
        (["seg_0002"], "no target-segment"),
    ],
)
def test_invalid_evidence_is_still_rejected(ids, message):
    with pytest.raises(MapError, match=message):
        attach_source_quotes(extracted(ids), chunk(), source())


def test_successful_checkpoint_reuse_and_prompt_invalidation(tmp_path):
    prompt = tmp_path / "map.txt"
    prompt.write_text("Extract observations.", encoding="utf-8")
    provider = FakeProvider()
    kwargs = {
        "model": "fake-model",
        "checkpoint_dir": tmp_path / "checkpoints",
        "prompt_path": prompt,
        "legacy_prompt_path": tmp_path / "absent.txt",
    }

    first = extract_meeting_notes(source(), provider, **kwargs)
    second = extract_meeting_notes(source(), provider, **kwargs)

    assert provider.calls == 1
    assert first == second
    assert first[0].input_segment_ids == ["seg_0001", "seg_0002"]

    prompt.write_text("Changed instructions.", encoding="utf-8")
    extract_meeting_notes(source(), provider, **kwargs)
    assert provider.calls == 2


def test_only_validated_legacy_checkpoint_can_migrate(tmp_path):
    provider = FakeProvider()
    old_prompt = tmp_path / "old.txt"
    old_prompt.write_text("Old instructions.", encoding="utf-8")
    target = chunk()
    budget = provider.estimate_request(schema=LegacySchema)

    key = fingerprint(
        settings_for(
            provider, "fake-model", "Old instructions.", target,
            LEGACY_VERSION, legacy_schema(), budget,
        )
    )
    saved = {
        "status": "source_ids_validated",
        "version": LEGACY_VERSION,
        "fingerprint": key,
        "extracted": {
            "covered_segment_ids": ["seg_0001"],
            **extracted().model_dump(mode="json"),
        },
        "repair": {"policy": "recorded-previous-repair"},
    }

    kwargs = {
        "folder": tmp_path,
        "chunk": target,
        "transcript": source(),
        "provider": provider,
        "model": "fake-model",
        "legacy_prompt_path": old_prompt,
    }

    # An attempt file must never count as a successful checkpoint.
    save_json_atomic(tmp_path / f"{key}.attempt.json", saved)
    assert try_legacy_checkpoint(**kwargs) is None

    success = tmp_path / f"{key}.json"
    save_json_atomic(success, saved)
    result = try_legacy_checkpoint(**kwargs)
    assert result is not None
    assert result[1]["source_repair"] == saved["repair"]

    saved["extracted"]["covered_segment_ids"] = []
    save_json_atomic(success, saved)
    assert try_legacy_checkpoint(**kwargs) is None