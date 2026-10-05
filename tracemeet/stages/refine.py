import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from tracemeet.llm.base import LLMProvider
from tracemeet.schemas import Segment, Transcript

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = ROOT / "prompts" / "refine_v1.txt"

# Temporary development limit, not a model context-window specification.
# We will replace this with chunking for longer recordings.
MAX_INPUT_CHARS = 20_000


class RefinementError(Exception):
    """The refinement stage could not produce a usable candidate."""


class RevisedSegment(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class RefinementResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segments: list[RevisedSegment] = Field(min_length=1)


def refine_transcript(
    raw: Transcript,
    provider: LLMProvider,
    *,
    model: str,
    glossary: str = "",
    temperature: float = 1.0,
    prompt_path: str | Path = DEFAULT_PROMPT,
) -> Transcript:
    """Return a candidate transcript; semantic guards run separately."""
    if not raw.segments:
        raise RefinementError("Cannot refine an empty transcript.")

    payload = json.dumps(
        {
            "glossary": glossary.strip(),
            "segments": [
                {"id": segment.id, "text": segment.text}
                for segment in raw.segments
            ],
        },
        ensure_ascii=False,
    )

    if len(payload) > MAX_INPUT_CHARS:
        raise RefinementError(
            "This transcript exceeds the development refinement limit "
            f"of {MAX_INPUT_CHARS:,} characters. "
            "Use a shorter recording until chunking is implemented."
        )

    try:
        system_prompt = Path(prompt_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RefinementError(
            f"Could not read the refinement prompt: {prompt_path}"
        ) from exc

    if not system_prompt.strip():
        raise RefinementError("The refinement prompt is empty.")

    response = provider.generate_structured(
        model=model,
        system=system_prompt,
        prompt=payload,
        schema=RefinementResponse,
        temperature=temperature,
    )

    expected_ids = [segment.id for segment in raw.segments]
    returned_ids = [segment.id for segment in response.segments]

    if len(returned_ids) != len(set(returned_ids)):
        raise RefinementError(
            "The refiner returned duplicate segment IDs."
        )

    if returned_ids != expected_ids:
        raise RefinementError(
            "The refiner omitted, added, changed, or reordered segment IDs."
        )

    # Build new objects. Never modify the raw transcript.
    # ASR diagnostic scores describe the original recognition output,
    # so we do not attach them to the revised text.
    revised_segments = [
        Segment(
            id=original.id,
            start=original.start,
            end=original.end,
            text=revised.text,
        )
        for original, revised in zip(
            raw.segments, response.segments, strict=True
        )
    ]

    return Transcript(
        segments=revised_segments,
        stt_model=raw.stt_model,
        language=raw.language,
        duration_s=raw.duration_s,
    )