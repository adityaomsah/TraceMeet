import argparse
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from scripts.minutes_dev import load_guarded_transcript, text_hash
from tracemeet.config import ConfigError, load_config
from tracemeet.llm.base import LLMError
from tracemeet.llm.groq_provider import GroqProvider
from tracemeet.schemas import MeetingRecord
from tracemeet.stages.minutes import (
    DEFAULT_PROMPT,
    MINUTES_VERSION,
    MinutesError,
    generate_minutes,
)


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    # Explicit LF makes the saved bytes predictable on Windows too.
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        file.write(text)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate minutes with Groq from a guarded transcript."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    args = parser.parse_args()

    provider = None

    try:
        # Retains config validation and loads the project .env.
        cfg = load_config()
        if args.model == cfg["llm"]["refine_model"]:
            raise ValueError("Use distinct refinement and minutes models.")

        transcript, source_text = load_guarded_transcript(args.run_dir)

        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            raise ConfigError("GROQ_API_KEY is missing from the environment.")

        provider = GroqProvider(
            key,
            request_budget=7400,
            max_completion_tokens=3072,
        )

        system = DEFAULT_PROMPT.read_text(encoding="utf-8")
        payload = json.dumps(
            {
                "segments": [
                    {"id": segment.id, "text": segment.text}
                    for segment in transcript.segments
                ]
            },
            ensure_ascii=False,
        )
        budget = provider.estimate_request(
            system=system,
            prompt=payload,
            schema=MeetingRecord,
        )

        print(f"Provider: groq | model: {args.model}")
        print(f"Guarded segments: {len(transcript.segments)}")
        print(json.dumps(budget, indent=2))

        if args.check_only:
            print("Budget check complete. No Groq API call was made.")
            return 0

        if not budget["fits"]:
            raise MinutesError(
                "This meeting exceeds the configured single-request "
                "budget. No API call was made. Keep the saved transcript "
                "for the long-meeting processing step."
            )

        print("Generating minutes: one API request; consumes quota.")
        started = perf_counter()

        record = generate_minutes(
            transcript,
            provider,
            model=args.model,
            temperature=0.0,
            on_status=print,
        )

        elapsed = perf_counter() - started
        record_json = record.model_dump_json(indent=2)
        metadata = {
            "status": "candidate_evidence_unverified",
            "implementation_version": MINUTES_VERSION,
            "provider": provider.name,
            "model": args.model,
            "temperature": 0.0,
            "reasoning_effort": "low",
            "source_file": "refined_transcript.json",
            "source_sha256": text_hash(source_text),
            "prompt_file": DEFAULT_PROMPT.name,
            "prompt_sha256": sha256(
                DEFAULT_PROMPT.read_bytes()
            ).hexdigest(),
            "record_sha256": text_hash(record_json),
            "elapsed_seconds": elapsed,
            "provider_call": provider.last_call,
            "created": datetime.now(timezone.utc).isoformat(),
        }

        output = args.run_dir / "candidate_meeting_record.json"
        metadata_path = args.run_dir / "minutes_meta.json"

        # Preserve any previous successful generation before replacement.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        for path in (output, metadata_path):
            if path.exists():
                backup = path.with_name(f"{path.stem}.{stamp}.bak.json")
                backup.write_bytes(path.read_bytes())

        write_text_atomic(output, record_json)
        write_text_atomic(
            metadata_path,
            json.dumps(metadata, indent=2, ensure_ascii=False),
        )

        print("\n" + record_json)
        print(f"\nCompleted in {elapsed:.1f}s.")
        print(f"Saved: {output}")
        print("Candidate only. Run evidence_dev next.")
        return 0

    except (ConfigError, LLMError, MinutesError, ValueError, OSError) as exc:
        print(f"\nERROR: {exc}")
        return 2

    finally:
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())