from types import SimpleNamespace

from pydantic import create_model

from tracemeet.guards.sensitive import guard_refinement
from tracemeet.schemas import Segment, Transcript


def test_guard_reconstructs_segments_from_another_class():
    OtherSegment = create_model(
        "Segment",
        id=(str, ...),
        start=(float, ...),
        end=(float, ...),
        text=(str, ...),
        avg_logprob=(float, -0.3),
        no_speech_prob=(float, 0.01),
    )

    old_segment = OtherSegment(
        id="seg_0001",
        start=0.0,
        end=2.0,
        text="We should not deploy.",
    )
    assert not isinstance(old_segment, Segment)

    # Simulate the serialization boundary of a retained transcript.
    retained_raw = SimpleNamespace(
        model_dump=lambda **kwargs: {
            "segments": [old_segment.model_dump()],
            "stt_model": "test",
            "language": "en",
            "duration_s": 2.0,
        }
    )

    candidate = Transcript(
        segments=[
            Segment(
                id="seg_0001",
                start=0.0,
                end=2.0,
                text="We should deploy.",
            )
        ],
        stt_model="test",
        duration_s=2.0,
    )

    result = guard_refinement(retained_raw, candidate)

    restored = result.transcript.segments[0]
    assert isinstance(restored, Segment)
    assert restored.text == "We should not deploy."
    assert restored.avg_logprob == -0.3
    assert result.corrections[0].status.value == "needs_review"
    assert "negation_changed" in result.corrections[0].flags