import json
import time
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from tracemeet.guards.evidence import _exact_span
from tracemeet.llm.groq_provider import GeneratedOutputError
from tracemeet.schemas import Evidence, Transcript
from tracemeet.stages.minutes_chunks import plan_minutes_chunks

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_PROMPT = ROOT / "prompts" / "minutes_map_v2.txt"
LEGACY_MAP_PROMPT = ROOT / "prompts" / "minutes_map_v1.txt"

MAP_VERSION = "minutes-map-v3-python-coverage"
LEGACY_VERSION = "minutes-map-v2-source-quotes"
RETRY_POLICY = "schema-and-evidence-two-total-attempts-v1"

ObservationKind = Literal[
    "topic", "proposal", "agreement", "task", "rejection",
    "amendment", "cancellation", "open_question",
]


class ExtractedObservation(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True
    )
    kind: ObservationKind
    statement: str = Field(min_length=1)
    evidence_segment_ids: list[str] = Field(min_length=1)


class ExtractedNotes(BaseModel):
    """The model returns observations, not input bookkeeping."""

    model_config = ConfigDict(extra="forbid")
    observations: list[ExtractedObservation]


class Observation(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True
    )
    kind: ObservationKind
    statement: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)


class ChunkNotes(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_segment_ids: list[str]
    observations: list[Observation]


class MapError(Exception):
    """Extraction or evidence-reference validation failed."""


def legacy_schema() -> dict:
    """Reconstruct the previous response schema for fingerprint checking."""
    schema = deepcopy(ExtractedNotes.model_json_schema())
    schema.pop("description", None)
    schema["properties"] = {
        "covered_segment_ids": {
            "items": {"type": "string"},
            "title": "Covered Segment Ids",
            "type": "array",
        },
        **schema["properties"],
    }
    schema["required"] = ["covered_segment_ids", "observations"]
    return schema


class LegacySchema:
    @classmethod
    def model_json_schema(cls):
        return legacy_schema()


class MapEstimator:
    def __init__(self, provider):
        self.provider = provider

    def estimate_request(self, *, system, prompt, schema):
        return self.provider.estimate_request(
            system=system, prompt=prompt, schema=ExtractedNotes
        )


def validate_chunk_notes(notes, chunk, transcript):
    if notes.input_segment_ids != list(chunk.target_ids):
        raise MapError("Internal error: recorded input IDs changed.")

    sources = transcript.by_id()
    allowed = set(
        chunk.context_before_ids
        + chunk.target_ids
        + chunk.context_after_ids
    )
    targets = set(chunk.target_ids)

    for observation in notes.observations:
        ids = [item.segment_id for item in observation.evidence]

        if len(ids) != len(set(ids)):
            raise MapError("Duplicated evidence IDs.")

        if not targets.intersection(ids):
            raise MapError("Observation has no target-segment evidence.")

        for evidence in observation.evidence:
            if evidence.segment_id not in allowed:
                raise MapError(
                    f"Citation outside this window: {evidence.segment_id}."
                )

            segment = sources.get(evidence.segment_id)
            if (
                segment is None
                or not any(char.isalnum() for char in evidence.quote)
                or _exact_span(segment.text, evidence.quote) is None
            ):
                raise MapError(
                    f"Invalid source quote: {evidence.segment_id}."
                )


def attach_source_quotes(extracted, chunk, transcript) -> ChunkNotes:
    sources = transcript.by_id()
    observations = []

    for item in extracted.observations:
        evidence = []
        for segment_id in item.evidence_segment_ids:
            segment = sources.get(segment_id)
            if segment is None:
                raise MapError(f"Unknown evidence segment: {segment_id}.")
            evidence.append(
                Evidence(segment_id=segment_id, quote=segment.text)
            )

        observations.append(
            Observation(
                kind=item.kind,
                statement=item.statement,
                evidence=evidence,
            )
        )

    notes = ChunkNotes(
        input_segment_ids=list(chunk.target_ids),
        observations=observations,
    )
    validate_chunk_notes(notes, chunk, transcript)
    return notes


def save_json_atomic(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
    temporary.replace(path)


def fingerprint(settings: dict) -> str:
    return sha256(
        json.dumps(
            settings, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def settings_for(provider, model, system, chunk, version, schema, budget):
    return {
        "version": version,
        "provider": provider.name,
        "model": model,
        "temperature": 0.0,
        "reasoning_effort": "low",
        "system": system,
        "prompt": chunk.prompt,
        "schema": schema,
        "budget": budget,
    }


def try_legacy_checkpoint(
    *,
    folder,
    chunk,
    transcript,
    provider,
    model,
    legacy_prompt_path,
):
    """Reuse only successful old checkpoints with matching source/settings."""
    if not legacy_prompt_path.exists():
        return None

    old_system = legacy_prompt_path.read_text(encoding="utf-8")
    old_budget = provider.estimate_request(
        system=old_system, prompt=chunk.prompt, schema=LegacySchema
    )
    old_settings = settings_for(
        provider, model, old_system, chunk,
        LEGACY_VERSION, legacy_schema(), old_budget,
    )
    old_fingerprint = fingerprint(old_settings)
    old_path = folder / f"{old_fingerprint}.json"

    if not old_path.exists():
        return None

    try:
        old_bytes = old_path.read_bytes()
        saved = json.loads(old_bytes)

        if (
            saved.get("fingerprint") != old_fingerprint
            or saved.get("version") != LEGACY_VERSION
            or saved.get("status") != "source_ids_validated"
        ):
            return None

        old_extracted = saved["extracted"]
        if old_extracted["covered_segment_ids"] != list(chunk.target_ids):
            return None

        extracted = ExtractedNotes.model_validate(
            {"observations": old_extracted["observations"]}
        )
        attach_source_quotes(extracted, chunk, transcript)
    except (ValueError, KeyError, TypeError, MapError):
        return None

    migration = {
        "policy": "reuse-validated-v2-observations",
        "source_checkpoint": old_path.name,
        "source_checkpoint_sha256": sha256(old_bytes).hexdigest(),
        "source_fingerprint": old_fingerprint,
        "source_version": LEGACY_VERSION,
        "source_repair": saved.get("repair"),
        "note": (
            "Reused existing observations after input and evidence checks. "
            "No new generation under the v3 prompt was performed."
        ),
    }
    return extracted, migration, saved.get("provider_call", {})


def wait_for_request_gap(last_finished, min_gap_s, status):
    if last_finished is None:
        return

    remaining = min_gap_s - (time.monotonic() - last_finished)
    if remaining <= 0:
        return

    status(f"Pacing: waiting {remaining:.0f}s.")
    deadline = time.monotonic() + remaining
    while time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def retry_feedback(failure_type: str) -> str:
    # Fixed instructions: do not inject model/server error text into prompts.
    if failure_type == "schema":
        return (
            "\n\nOUTPUT VALIDATION RETRY:\n"
            "Regenerate observations from this window using the exact schema. "
            "Every kind must be exactly one of: topic, proposal, agreement, "
            "task, rejection, amendment, cancellation, open_question. "
            "Do not invent other labels or add fields. "
            "Use only supplied evidence segment IDs. "
            "Preserve the observations supported by the transcript."
        )

    return (
        "\n\nEVIDENCE VALIDATION RETRY:\n"
        "Regenerate observations from this window. "
        "Use only supplied segment IDs, avoid duplicate citations, "
        "and cite at least one target segment for every observation. "
        "Include all segments needed to support each statement."
    )


def extract_meeting_notes(
    transcript: Transcript,
    provider,
    *,
    model: str,
    checkpoint_dir: Path,
    prompt_path: Path = DEFAULT_MAP_PROMPT,
    legacy_prompt_path: Path = LEGACY_MAP_PROMPT,
    on_status: Callable[[str], None] | None = None,
    min_gap_s: float = 61.0,
) -> list[ChunkNotes]:
    if not 61 <= min_gap_s <= 3600:
        raise ValueError("min_gap_s must be between 61 and 3600.")

    system = prompt_path.read_text(encoding="utf-8")
    if not system.strip():
        raise MapError("Extraction prompt is empty.")

    chunks = plan_minutes_chunks(
        transcript,
        MapEstimator(provider),
        system=system,
        max_target_segments=80,
        context_size=2,
    )

    def status(message):
        if on_status is not None:
            on_status(message)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    last_call_finished = None

    for chunk in chunks:
        settings = settings_for(
            provider, model, system, chunk,
            MAP_VERSION, ExtractedNotes.model_json_schema(), chunk.budget,
        )
        key = fingerprint(settings)
        path = checkpoint_dir / f"{key}.json"

        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                if (
                    saved.get("fingerprint") != key
                    or saved.get("input_segment_ids") != list(chunk.target_ids)
                    or saved.get("status") != "source_ids_validated"
                    or saved.get("version") != MAP_VERSION
                ):
                    raise ValueError("Checkpoint identity mismatch.")

                extracted = ExtractedNotes.model_validate(saved["extracted"])
                notes = attach_source_quotes(extracted, chunk, transcript)
            except (ValueError, KeyError, TypeError, MapError):
                status(f"Chunk {chunk.index}: invalid checkpoint.")
            else:
                results.append(notes)
                status(
                    f"Chunk {chunk.index}/{len(chunks)}: "
                    "reused validated checkpoint."
                )
                continue

        migrated = try_legacy_checkpoint(
            folder=checkpoint_dir,
            chunk=chunk,
            transcript=transcript,
            provider=provider,
            model=model,
            legacy_prompt_path=legacy_prompt_path,
        )
        if migrated is not None:
            extracted, migration, old_call = migrated
            notes = attach_source_quotes(extracted, chunk, transcript)
            save_json_atomic(
                path,
                {
                    "status": "source_ids_validated",
                    "version": MAP_VERSION,
                    "fingerprint": key,
                    "input_segment_ids": list(chunk.target_ids),
                    "extracted": extracted.model_dump(mode="json"),
                    "migration": migration,
                    "provider_call": old_call,
                },
            )
            results.append(notes)
            status(
                f"Chunk {chunk.index}/{len(chunks)}: migrated validated "
                "legacy checkpoint; no API call."
            )
            continue

        feedback = ""
        for attempt in range(1, 3):
            request_system = system + feedback
            budget = provider.estimate_request(
                system=request_system,
                prompt=chunk.prompt,
                schema=ExtractedNotes,
            )
            if not budget["fits"]:
                raise MapError(
                    f"Chunk {chunk.index}: request including validation "
                    "feedback exceeds budget. No API call made."
                )

            wait_for_request_gap(last_call_finished, min_gap_s, status)
            status(
                f"Chunk {chunk.index}/{len(chunks)}, attempt {attempt}/2: "
                f"{len(chunk.target_ids)} targets."
            )

            audit = {
                "fingerprint": key,
                "attempt": attempt,
                "retry_policy": RETRY_POLICY,
                "request_system_sha256": sha256(
                    request_system.encode("utf-8")
                ).hexdigest(),
                "validation_feedback": feedback,
            }
            attempt_path = checkpoint_dir / (
                f"{key}.{time.time_ns()}.attempt.json"
            )
            failure_type = "schema"

            try:
                try:
                    extracted = provider.generate_structured(
                        model=model,
                        system=request_system,
                        prompt=chunk.prompt,
                        schema=ExtractedNotes,
                        temperature=0.0,
                    )
                finally:
                    last_call_finished = time.monotonic()
                    audit["provider_call"] = dict(
                        getattr(provider, "last_call", {})
                    )

                audit["extracted"] = extracted.model_dump(mode="json")
                failure_type = "evidence"
                notes = attach_source_quotes(extracted, chunk, transcript)

            except (GeneratedOutputError, MapError, ValidationError) as exc:
                save_json_atomic(
                    attempt_path,
                    {
                        **audit,
                        "status": "validation_failed",
                        "failure_type": failure_type,
                        "error": str(exc),
                    },
                )

                if attempt == 2:
                    raise MapError(
                        f"Chunk {chunk.index}: validation failed after "
                        f"2 attempts ({failure_type}). "
                        "Earlier checkpoints remain saved."
                    ) from exc

                status(
                    f"Chunk {chunk.index}: {failure_type} validation failed; "
                    "one paced retry will follow."
                )
                feedback = retry_feedback(failure_type)
                continue

            save_json_atomic(
                attempt_path,
                {**audit, "status": "source_ids_validated"},
            )
            save_json_atomic(
                path,
                {
                    **audit,
                    "status": "source_ids_validated",
                    "version": MAP_VERSION,
                    "input_segment_ids": list(chunk.target_ids),
                },
            )
            results.append(notes)
            status(
                f"Chunk {chunk.index}: saved "
                f"{len(notes.observations)} observations."
            )
            break

    actual = [
        segment_id
        for result in results
        for segment_id in result.input_segment_ids
    ]
    expected = [segment.id for segment in transcript.segments]
    if actual != expected:
        raise MapError("Internal error: aggregate input coverage changed.")

    return results