import json
from hashlib import sha256
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tracemeet.llm.base import LLMProvider
from tracemeet.llm.router import generate_with_retry
from tracemeet.schemas import Segment, Transcript

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = ROOT / "prompts" / "refine_v1.txt"

REFINEMENT_VERSION = "chunked-v1"
DEFAULT_CHUNK_SIZE = 25
DEFAULT_CONTEXT_SIZE = 2
DEFAULT_MAX_CHARS = 20_000


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


def _rows(segments: list[Segment]) -> list[dict[str, str]]:
    return [{"id": segment.id, "text": segment.text} for segment in segments]


def build_chunks(
    raw: Transcript,
    glossary: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[dict]:
    """Partition targets; overlap is read-only context."""
    if chunk_size < 1 or context_size < 0 or max_chars < 1:
        raise ValueError("Invalid chunk settings.")

    chunks = []
    start = 0
    segments = raw.segments

    while start < len(segments):
        end = min(start + chunk_size, len(segments))

        while end > start:
            payload = {
                "glossary": glossary.strip(),
                "context_before": _rows(
                    segments[max(0, start - context_size):start]
                ),
                "target_segments": _rows(segments[start:end]),
                "context_after": _rows(segments[end:end + context_size]),
            }

            encoded = json.dumps(payload, ensure_ascii=False)
            if len(encoded) <= max_chars:
                break

            end -= 1
        else:
            raise RefinementError(
                f"Segment {segments[start].id}, its context, and glossary "
                f"cannot fit the {max_chars:,}-character request budget. "
                "Reduce the glossary/context or increase the configured "
                "budget after checking model limits."
            )

        chunks.append(payload)
        start = end

    return chunks


def _check_alignment(
    response: RefinementResponse,
    expected_ids: list[str],
) -> None:
    returned_ids = [segment.id for segment in response.segments]

    if len(returned_ids) != len(expected_ids):
        raise RefinementError(
            f"Segment count mismatch: expected {len(expected_ids)}, "
            f"received {len(returned_ids)}."
        )

    if len(returned_ids) != len(set(returned_ids)):
        raise RefinementError("Duplicate segment IDs returned.")

    if returned_ids != expected_ids:
        raise RefinementError(
            "Segment IDs were changed or reordered."
        )


def refine_transcript(
    raw: Transcript,
    provider: LLMProvider,
    *,
    model: str,
    glossary: str = "",
    temperature: float = 1.0,
    prompt_path: str | Path = DEFAULT_PROMPT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_size: int = DEFAULT_CONTEXT_SIZE,
    max_chars: int = DEFAULT_MAX_CHARS,
    checkpoint_dir: Path | None = None,
    on_status: Callable[[str], None] | None = None,
) -> Transcript:
    """Return an aligned candidate; semantic guards run separately."""
    if not raw.segments:
        raise RefinementError("Cannot refine an empty transcript.")

    try:
        system = Path(prompt_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise RefinementError("Could not read the refinement prompt.") from exc

    if not system.strip():
        raise RefinementError("The refinement prompt is empty.")

    chunks = build_chunks(
        raw,
        glossary,
        chunk_size=chunk_size,
        context_size=context_size,
        max_chars=max_chars,
    )

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    revised = []

    for index, payload in enumerate(chunks, start=1):
        expected_ids = [row["id"] for row in payload["target_segments"]]
        prompt = json.dumps(payload, ensure_ascii=False)

        identity = {
            "version": REFINEMENT_VERSION,
            "provider": provider.name,
            "model": model,
            "temperature": temperature,
            "system": system,
            "payload": payload,
        }
        fingerprint = sha256(
            json.dumps(
                identity, sort_keys=True, ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()

        checkpoint = (
            checkpoint_dir / f"{fingerprint}.json"
            if checkpoint_dir is not None
            else None
        )
        response = None

        if checkpoint is not None and checkpoint.exists():
            try:
                cached = RefinementResponse.model_validate_json(
                    checkpoint.read_text(encoding="utf-8")
                )
                _check_alignment(cached, expected_ids)
                response = cached
            except (ValidationError, RefinementError):
                if on_status:
                    on_status(f"Chunk {index}: invalid checkpoint; regenerating.")

        if response is not None:
            if on_status:
                on_status(f"Chunk {index}/{len(chunks)}: reused checkpoint.")
        else:
            if on_status:
                on_status(
                    f"Chunk {index}/{len(chunks)}: "
                    f"{len(expected_ids)} target segments."
                )

            repair_instruction = ""

            # One original generation and at most one alignment repair.
            for alignment_attempt in range(2):
                candidate = generate_with_retry(
                    provider,
                    model=model,
                    system=system + repair_instruction,
                    prompt=prompt,
                    schema=RefinementResponse,
                    temperature=temperature,
                    on_status=on_status,
                )

                try:
                    _check_alignment(candidate, expected_ids)
                except RefinementError as exc:
                    if alignment_attempt == 1:
                        raise RefinementError(
                            f"Chunk {index}/{len(chunks)} failed alignment "
                            f"after one repair attempt: {exc}"
                        ) from exc

                    repair_instruction = (
                        "\nYour previous response failed alignment validation. "
                        f"Problem: {exc} "
                        "Return exactly the target segments, in their input "
                        "order. Return no context segments."
                    )
                    if on_status:
                        on_status(f"Chunk {index}: requesting alignment repair.")
                else:
                    response = candidate
                    break

            if response is None:
                raise RefinementError("No aligned response was produced.")

            if checkpoint is not None:
                temporary = checkpoint.with_suffix(".tmp")
                temporary.write_text(
                    response.model_dump_json(indent=2), encoding="utf-8"
                )
                temporary.replace(checkpoint)

        revised.extend(response.segments)

    # Validate the full merge as well as every individual chunk.
    merged = RefinementResponse(segments=revised)
    _check_alignment(merged, [segment.id for segment in raw.segments])

    return Transcript(
        segments=[
            Segment(
                id=original.id,
                start=original.start,
                end=original.end,
                text=candidate.text,
            )
            for original, candidate in zip(
                raw.segments, merged.segments, strict=True
            )
        ],
        stt_model=raw.stt_model,
        language=raw.language,
        duration_s=raw.duration_s,
    )