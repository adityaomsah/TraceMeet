import argparse
import os
from hashlib import sha256
from pathlib import Path

from scripts.minutes_dev import load_guarded_transcript
from tracemeet.config import ConfigError, load_config
from tracemeet.llm.base import LLMError
from tracemeet.llm.groq_provider import GroqProvider
from tracemeet.stages.minutes_map import (
    DEFAULT_MAP_PROMPT,
    MAP_VERSION,
    MapError,
    extract_meeting_notes,
    save_json_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract checkpointed provisional meeting observations."
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    provider = None

    try:
        load_config()
        transcript, source_text = load_guarded_transcript(args.run_dir)
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            raise ConfigError("GROQ_API_KEY is missing.")

        provider = GroqProvider(
            key, request_budget=7400, max_completion_tokens=3072
        )
        model = "openai/gpt-oss-120b"

        print("Uncached chunks consume API quota.")
        print("Input coverage is recorded by Python.")
        print("Do not run another Groq script alongside this one.")

        chunks = extract_meeting_notes(
            transcript,
            provider,
            model=model,
            checkpoint_dir=args.run_dir / "minutes_map_chunks",
            on_status=print,
        )

        observations = [
            {
                "observation_id": (
                    f"chunk_{chunk_index:03d}_note_{note_index:03d}"
                ),
                **observation.model_dump(mode="json"),
            }
            for chunk_index, chunk in enumerate(chunks, start=1)
            for note_index, observation in enumerate(
                chunk.observations, start=1
            )
        ]

        output = args.run_dir / "provisional_meeting_notes.json"
        save_json_atomic(
            output,
            {
                "status": "provisional_not_final_minutes",
                "implementation_version": MAP_VERSION,
                "provider": provider.name,
                "model": model,
                "source_text_sha256": sha256(
                    source_text.encode("utf-8")
                ).hexdigest(),
                "planning_prompt_file_sha256": sha256(
                    DEFAULT_MAP_PROMPT.read_bytes()
                ).hexdigest(),
                "input_segment_ids": [
                    segment_id
                    for chunk in chunks
                    for segment_id in chunk.input_segment_ids
                ],
                "observations": observations,
                "semantic_support_checked": False,
                "generation_provenance": (
                    "See minutes_map_chunks success checkpoints for "
                    "per-chunk generation, repair and migration history."
                ),
            },
        )

        print(f"\nSaved {len(observations)} provisional observations.")
        print(f"Saved: {output}")
        print("All input segments accounted for by Python.")
        print("Semantic completeness and global reconciliation are pending.")
        return 0

    except (ConfigError, LLMError, MapError, ValueError, OSError) as exc:
        print(f"\nERROR: {exc}")
        return 2
    finally:
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())