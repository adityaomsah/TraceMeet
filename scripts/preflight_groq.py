import json
import os
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from tracemeet.guards.evidence import verify_evidence
from tracemeet.schemas import MeetingRecord, Segment, Transcript

ROOT = Path(__file__).resolve().parents[1]
MODEL = "openai/gpt-oss-120b"
URL = "https://api.groq.com/openai/v1/chat/completions"


def strict_schema(schema: dict) -> dict:
    """Adapt our schema for strict output without changing Pydantic models."""
    result = deepcopy(schema)

    def visit(node):
        if not isinstance(node, dict):
            return

        # Defaults remain a local Pydantic concern.
        node.pop("default", None)

        properties = node.get("properties")
        if isinstance(properties, dict):
            node["additionalProperties"] = False
            node["required"] = list(properties)
            for child in properties.values():
                visit(child)

        for definitions_key in ("$defs", "definitions"):
            definitions = node.get(definitions_key)
            if isinstance(definitions, dict):
                for child in definitions.values():
                    visit(child)

        items = node.get("items")
        if isinstance(items, dict):
            visit(items)

        for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
            children = node.get(keyword)
            if isinstance(children, list):
                for child in children:
                    visit(child)

    visit(result)
    return result


def probe_transcript() -> Transcript:
    """Synthetic test input, never substituted for a user's recording."""
    return Transcript(
        stt_model="synthetic-provider-probe",
        language="en",
        duration_s=18,
        segments=[
            Segment(
                id="seg_0001",
                start=0,
                end=6,
                text=(
                    "Alice: I will send the security report by Monday."
                ),
            ),
            Segment(
                id="seg_0002",
                start=6,
                end=12,
                text="Bob: I propose deploying on Friday.",
            ),
            Segment(
                id="seg_0003",
                start=12,
                end=18,
                text=(
                    "Team lead: That proposal is rejected. "
                    "Our decision is not to deploy on Friday."
                ),
            ),
        ],
    )


def main() -> int:
    load_dotenv(ROOT / ".env")
    key = os.getenv("GROQ_API_KEY", "").strip()

    if not key:
        print("ERROR: Add GROQ_API_KEY to the project .env file.")
        return 1

    try:
        system = (ROOT / "prompts" / "minutes_v1.txt").read_text(
            encoding="utf-8"
        )
    except OSError:
        print("ERROR: Could not read prompts/minutes_v1.txt.")
        return 1

    transcript = probe_transcript()
    prompt = json.dumps(
        {
            "segments": [
                {"id": segment.id, "text": segment.text}
                for segment in transcript.segments
            ]
        },
        ensure_ascii=False,
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "reasoning_effort": "low",
        "max_completion_tokens": 2048,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "meeting_record",
                "strict": True,
                "schema": strict_schema(
                    MeetingRecord.model_json_schema()
                ),
            },
        },
    }

    print(f"Model: {MODEL}")
    print("Input: three synthetic meeting segments.")
    print("One API request; no automatic retries. Consumes API quota.")
    started = perf_counter()

    try:
        with httpx.Client(
            timeout=httpx.Timeout(120.0, connect=15.0)
        ) as client:
            response = client.post(
                URL,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            )
    except httpx.TimeoutException:
        print(
            f"ERROR: Request timed out after "
            f"{perf_counter() - started:.1f}s."
        )
        return 2
    except httpx.TransportError as exc:
        print(f"ERROR: Network failure ({type(exc).__name__}).")
        return 2

    elapsed = perf_counter() - started
    print(f"HTTP {response.status_code} in {elapsed:.1f}s")

    if response.is_error:
        # Display the API message, never request headers or the key.
        try:
            body = response.json()
            error = body.get("error", {})
            detail = (
                error.get("message", "No error message provided.")
                if isinstance(error, dict)
                else "Unexpected error response."
            )
        except (ValueError, AttributeError):
            detail = "The server did not return a JSON error."

        print(f"ERROR: {str(detail).replace(key, '[REDACTED]')[:1200]}")
        retry_after = response.headers.get("retry-after")
        if retry_after:
            print(f"Server Retry-After: {retry_after}")
        return 2

    try:
        body = response.json()
        choice = body["choices"][0]
        finish_reason = choice.get("finish_reason")

        if finish_reason != "stop":
            print(
                f"ERROR: Incomplete generation: {finish_reason}. "
                "No record accepted."
            )
            return 2

        message = choice["message"]
        if message.get("refusal"):
            print("ERROR: The model refused the request.")
            return 2

        record = MeetingRecord.model_validate_json(
            message.get("content") or ""
        )
        checked = verify_evidence(record, transcript)

    except ValidationError as exc:
        print(
            f"ERROR: Output failed local validation "
            f"({exc.error_count()} errors)."
        )
        return 2
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"ERROR: Unexpected response ({type(exc).__name__}).")
        return 2

    out_dir = ROOT / "runs" / "groq_probe"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "candidate_meeting_record.json").write_text(
            record.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (out_dir / "evidence_report.json").write_text(
            json.dumps(checked.report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        print("ERROR: Could not save the probe output.")
        return 2

    print("\nGenerated record:")
    print(record.model_dump_json(indent=2))
    print("\nCitation report:")
    print(json.dumps(checked.report, indent=2, ensure_ascii=False))
    print("\nToken usage:")
    print(json.dumps(body.get("usage", {}), indent=2))
    print(f"\nSaved probe output in: {out_dir}")
    print("Schema validation passed. Inspect extraction accuracy below.")
    print("- Task: send the security report; Alice; Monday.")
    print("- Friday deployment proposal: rejected, not approved.")
    print("- Final decision preserves 'not to deploy on Friday'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())