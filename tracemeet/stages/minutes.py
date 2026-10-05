import json
from pathlib import Path
from typing import Callable

from tracemeet.llm.base import LLMProvider
from tracemeet.llm.router import generate_with_retry
from tracemeet.schemas import MeetingRecord, Transcript

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = ROOT / "prompts" / "minutes_v1.txt"

MINUTES_VERSION = "minutes-v1"
DEFAULT_MAX_CHARS = 60_000


class MinutesError(Exception):
    """Meeting documentation could not be generated."""


def generate_minutes(
    transcript: Transcript,
    provider: LLMProvider,
    *,
    model: str,
    temperature: float = 1.0,
    prompt_path: str | Path = DEFAULT_PROMPT,
    max_chars: int = DEFAULT_MAX_CHARS,
    on_status: Callable[[str], None] | None = None,
) -> MeetingRecord:
    """Return a schema-validated candidate; citation checks run separately."""
    if not transcript.segments:
        raise MinutesError("Cannot document an empty transcript.")

    if max_chars < 1:
        raise ValueError("max_chars must be positive.")

    try:
        system = Path(prompt_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise MinutesError("Could not read the minutes prompt.") from exc

    if not system.strip():
        raise MinutesError("The minutes prompt is empty.")

    payload = json.dumps(
        {
            "segments": [
                {"id": segment.id, "text": segment.text}
                for segment in transcript.segments
            ]
        },
        ensure_ascii=False,
    )

    if len(payload) > max_chars:
        raise MinutesError(
            f"The minutes input exceeds the current {max_chars:,}-character "
            "payload budget. Long-document processing is not implemented "
            "for this stage yet. The transcript has not been truncated."
        )

    if on_status:
        on_status(
            f"Generating meeting documentation from "
            f"{len(transcript.segments)} segments."
        )

    return generate_with_retry(
    provider,
    model=model,
    system=system,
    prompt=payload,
    schema=MeetingRecord,
    temperature=temperature,
    max_attempts=5,
    on_status=on_status,
)