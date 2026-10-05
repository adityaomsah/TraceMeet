import json
from types import SimpleNamespace

import pytest

from tracemeet.guards.diff import segment_edits
from tracemeet.llm import router
from tracemeet.llm.base import LLMError, LLMQuotaError, LLMTemporaryError
from tracemeet.schemas import Segment, Transcript
from tracemeet.stages.refine import (
    RefinementError,
    build_chunks,
    refine_transcript,
)


def make_transcript(count=7):
    return Transcript(
        segments=[
            Segment(
                id=f"seg_{i:04d}",
                start=float(i),
                end=float(i + 1),
                text=f"Original text {i}.",
            )
            for i in range(count)
        ],
        stt_model="test",
    )


class EchoProvider:
    name = "fake"

    def __init__(self, bad_calls=0):
        self.calls = 0
        self.bad_calls = bad_calls

    def generate_structured(self, **kwargs):
        self.calls += 1
        rows = json.loads(kwargs["prompt"])["target_segments"]
        rows = [dict(row) for row in rows]

        if self.calls <= self.bad_calls:
            rows[0]["id"] = "wrong_id"

        return kwargs["schema"].model_validate({"segments": rows})


def test_chunks_cover_every_target_once_and_context_is_read_only():
    raw = make_transcript()
    chunks = build_chunks(raw, "", chunk_size=3, context_size=2)

    target_ids = [
        row["id"]
        for chunk in chunks
        for row in chunk["target_segments"]
    ]
    assert target_ids == [segment.id for segment in raw.segments]

    for chunk in chunks:
        targets = {row["id"] for row in chunk["target_segments"]}
        context = {
            row["id"]
            for row in chunk["context_before"] + chunk["context_after"]
        }
        assert targets.isdisjoint(context)

    assert [row["id"] for row in chunks[1]["context_before"]] == [
        "seg_0001", "seg_0002"
    ]


def test_character_budget_splits_requests_and_rejects_oversize():
    raw = make_transcript()
    chunks = build_chunks(
        raw, "", chunk_size=7, context_size=0, max_chars=200
    )
    assert len(chunks) > 1
    assert all(
        len(json.dumps(chunk, ensure_ascii=False)) <= 200
        for chunk in chunks
    )

    with pytest.raises(RefinementError, match="cannot fit"):
        build_chunks(raw, "x" * 500, max_chars=200)


def test_alignment_repair_and_matching_checkpoint_reuse(tmp_path):
    raw = make_transcript(2)
    provider = EchoProvider(bad_calls=1)
    before = raw.model_dump()

    kwargs = {
        "model": "fake-model",
        "checkpoint_dir": tmp_path / "chunks",
    }

    result = refine_transcript(raw, provider, **kwargs)
    assert provider.calls == 2
    assert raw.model_dump() == before
    assert result.full_text() == raw.full_text()
    assert [(s.start, s.end) for s in result.segments] == [
        (s.start, s.end) for s in raw.segments
    ]

    refine_transcript(raw, provider, **kwargs)
    assert provider.calls == 2

    refine_transcript(raw, provider, glossary="New glossary", **kwargs)
    assert provider.calls == 3


def test_persistent_alignment_failure_stops():
    provider = EchoProvider(bad_calls=100)
    with pytest.raises(RefinementError, match="after one repair"):
        refine_transcript(make_transcript(2), provider, model="fake")
    assert provider.calls == 2


@pytest.mark.parametrize(
    "old,new",
    [
        ("recent hiding seasons.", "recent hiring seasons."),
        ("deploy deploy now", "deploy now"),
        ("Do not deploy.", "Do deploy."),
        ("budget 15,000", "budget 50,000"),
        ("hello", "hello world"),
        ("hello  world", "hello world"),
        ("", "hello"),
        ("hello", ""),
    ],
)
def test_diff_offsets_reconstruct_candidate(old, new):
    edits = segment_edits("seg_0001", old, new)

    reconstructed = old
    for edit in reversed(edits):
        assert old[edit["raw_start"]:edit["raw_end"]] == edit["original"]
        assert new[
            edit["candidate_start"]:edit["candidate_end"]
        ] == edit["replacement"]
        reconstructed = (
            reconstructed[:edit["raw_start"]]
            + edit["replacement"]
            + reconstructed[edit["raw_end"]:]
        )

    assert reconstructed == new


def call_router(provider):
    return router.generate_with_retry(
        provider,
        model="fake",
        system="",
        prompt="",
        schema=object,
    )


def test_transient_errors_retry_with_increasing_delays(monkeypatch):
    delays = []
    monkeypatch.setattr(router.time, "sleep", delays.append)
    monkeypatch.setattr(router.random, "uniform", lambda a, b: 0.0)
    events = iter([
        LLMTemporaryError("503"),
        LLMTemporaryError("503"),
        "success",
    ])

    def generate(**kwargs):
        event = next(events)
        if isinstance(event, Exception):
            raise event
        return event

    assert call_router(SimpleNamespace(generate_structured=generate)) == "success"
    assert delays == [2, 4]


@pytest.mark.parametrize(
    "error,expected_calls",
    [(LLMError("400"), 1), (LLMQuotaError("429"), 3)],
)
def test_retry_limits(monkeypatch, error, expected_calls):
    monkeypatch.setattr(router.time, "sleep", lambda seconds: None)
    calls = []

    def generate(**kwargs):
        calls.append(1)
        raise error

    with pytest.raises(type(error)):
        call_router(SimpleNamespace(generate_structured=generate))

    assert len(calls) == expected_calls


def test_long_server_retry_hint_stops_without_sleep(monkeypatch):
    cause = Exception("server")
    cause.response_json = {
        "error": {
            "details": [{
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "120s",
            }]
        }
    }
    error = LLMQuotaError("429")
    error.__cause__ = cause
    delays = []
    monkeypatch.setattr(router.time, "sleep", delays.append)

    def generate(**kwargs):
        raise error

    with pytest.raises(LLMQuotaError):
        call_router(SimpleNamespace(generate_structured=generate))

    assert delays == []