import argparse
import json
import os
from hashlib import sha256
from pathlib import Path

from scripts.minutes_dev import load_guarded_transcript
from tracemeet.config import ConfigError, load_config
from tracemeet.llm.base import LLMError
from tracemeet.llm.groq_provider import GroqProvider
from tracemeet.stages.minutes import DEFAULT_PROMPT
from tracemeet.stages.minutes_chunks import plan_minutes_chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan budgeted minutes chunks without API calls."
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    provider = None

    try:
        load_config()
        transcript, _ = load_guarded_transcript(args.run_dir)
        system = DEFAULT_PROMPT.read_text(encoding="utf-8")

        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            raise ConfigError("GROQ_API_KEY is missing.")

        # Construction initializes the local tokenizer and HTTP client.
        # Only estimate_request() is used below.
        provider = GroqProvider(
            key,
            request_budget=7400,
            max_completion_tokens=3072,
        )

        chunks = plan_minutes_chunks(
            transcript,
            provider,
            system=system,
            max_target_segments=80,
            context_size=2,
        )

        print(f"Segments: {len(transcript.segments)}")
        print(f"Planned chunks: {len(chunks)}")
        print("No inference API calls are made by this script.\n")

        rows = []
        for chunk in chunks:
            count = chunk.budget["estimated_total_tokens"]
            print(
                f"Chunk {chunk.index}: "
                f"{chunk.target_ids[0]} -> {chunk.target_ids[-1]} | "
                f"{len(chunk.target_ids)} targets | "
                f"{count:,} estimated tokens including output reserve"
            )
            rows.append(
                {
                    "index": chunk.index,
                    "target_ids": list(chunk.target_ids),
                    "context_before_ids": list(chunk.context_before_ids),
                    "context_after_ids": list(chunk.context_after_ids),
                    "budget": chunk.budget,
                }
            )

        report = {
            "status": "planning_only_no_generation",
            "source_file_sha256": sha256(
                (args.run_dir / "refined_transcript.json").read_bytes()
            ).hexdigest(),
            "prompt_file_sha256": sha256(
                DEFAULT_PROMPT.read_bytes()
            ).hexdigest(),
            "target_segments": len(transcript.segments),
            "context_size": 2,
            "request_budget": provider.request_budget,
            "reserved_completion_tokens": provider.max_completion_tokens,
            "chunks": rows,
            "note": (
                "Recalculate this plan with the final extraction prompt "
                "and schema before generation. This is not a checkpoint "
                "or a completed meeting record."
            ),
        }

        path = args.run_dir / "minutes_plan.json"
        with path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        print(f"\nSaved: {path}")
        print("Every segment is targeted exactly once, in source order.")
        return 0

    except (ConfigError, LLMError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 2
    finally:
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())