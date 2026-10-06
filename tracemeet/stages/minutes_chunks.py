import json
from dataclasses import dataclass
from typing import Protocol

from tracemeet.schemas import MeetingRecord, Segment, Transcript


class BudgetEstimator(Protocol):
    def estimate_request(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[MeetingRecord],
    ) -> dict:
        ...


@dataclass(frozen=True)
class MinutesChunk:
    index: int
    target_ids: tuple[str, ...]
    context_before_ids: tuple[str, ...]
    context_after_ids: tuple[str, ...]
    prompt: str
    budget: dict


def _segment_payload(segment: Segment) -> dict:
    return {"id": segment.id, "text": segment.text}


def chunk_payload(
    segments: list[Segment],
    start: int,
    end: int,
    context_size: int,
) -> dict:
    return {
        "context_before": [
            _segment_payload(segment)
            for segment in segments[max(0, start - context_size):start]
        ],
        "target_segments": [
            _segment_payload(segment)
            for segment in segments[start:end]
        ],
        "context_after": [
            _segment_payload(segment)
            for segment in segments[end:end + context_size]
        ],
    }


def plan_minutes_chunks(
    transcript: Transcript,
    estimator: BudgetEstimator,
    *,
    system: str,
    max_target_segments: int = 80,
    context_size: int = 2,
) -> list[MinutesChunk]:
    """Partition every segment into one target chunk without truncation."""
    if not transcript.segments:
        raise ValueError("Cannot plan minutes for an empty transcript.")

    if max_target_segments < 1:
        raise ValueError("max_target_segments must be positive.")

    if context_size < 0:
        raise ValueError("context_size cannot be negative.")

    segments = transcript.segments
    chunks: list[MinutesChunk] = []
    start = 0

    while start < len(segments):
        end = min(start + max_target_segments, len(segments))

        # Shrink targets until prompt + schema + output reserve fits.
        # The original segment text is never shortened.
        while end > start:
            payload = chunk_payload(
                segments, start, end, context_size
            )
            prompt = json.dumps(payload, ensure_ascii=False)
            budget = estimator.estimate_request(
                system=system,
                prompt=prompt,
                schema=MeetingRecord,
            )

            if budget["fits"]:
                break

            end -= 1

        if end == start:
            raise ValueError(
                f"Segment {segments[start].id}, its surrounding context, "
                "and the output reserve cannot fit in one request. "
                "No text was truncated. Inspect this segment before "
                "changing the planning policy."
            )

        chunks.append(
            MinutesChunk(
                index=len(chunks) + 1,
                target_ids=tuple(
                    segment.id for segment in segments[start:end]
                ),
                context_before_ids=tuple(
                    item["id"] for item in payload["context_before"]
                ),
                context_after_ids=tuple(
                    item["id"] for item in payload["context_after"]
                ),
                prompt=prompt,
                budget=budget,
            )
        )
        start = end

    actual_ids = [
        segment_id
        for chunk in chunks
        for segment_id in chunk.target_ids
    ]
    expected_ids = [segment.id for segment in segments]

    if actual_ids != expected_ids:
        raise ValueError(
            "Internal planning error: target coverage or order changed."
        )

    return chunks