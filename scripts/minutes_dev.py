import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from tracemeet.config import ConfigError, get_api_key, load_config
from tracemeet.guards.sensitive import guard_refinement
from tracemeet.llm.base import LLMError
from tracemeet.llm.gemini import GeminiProvider
from tracemeet.schemas import Transcript
from tracemeet.stages.minutes import (
    DEFAULT_PROMPT,
    MINUTES_VERSION,
    MinutesError,
    generate_minutes,
)


def text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def load_guarded_transcript(run_dir: Path) -> tuple[Transcript, str]:
    raw_text = (run_dir / "raw_transcript.json").read_text(encoding="utf-8")
    candidate_text = (
        run_dir / "candidate_refined_transcript.json"
    ).read_text(encoding="utf-8")
    refined_text = (
        run_dir / "refined_transcript.json"
    ).read_text(encoding="utf-8")

    report = json.loads(
        (run_dir / "guard_report.json").read_text(encoding="utf-8")
    )
    if not isinstance(report, dict):
        raise ValueError("guard_report.json must contain an object.")

    if report.get("raw_sha256") != text_hash(raw_text):
        raise ValueError("Raw transcript changed. Run guard_dev again.")

    if report.get("candidate_sha256") != text_hash(candidate_text):
        raise ValueError("Candidate transcript changed. Run guard_dev again.")

    names = report.get("protected_names")
    if not isinstance(names, list) or not all(
        isinstance(name, str) for name in names
    ):
        raise ValueError("Guard report has invalid protected_names.")

    raw = Transcript.model_validate_json(raw_text)
    candidate = Transcript.model_validate_json(candidate_text)
    refined = Transcript.model_validate_json(refined_text)

    expected = guard_refinement(
        raw, candidate, protected_names=names
    ).transcript

    if refined.model_dump() != expected.model_dump():
        raise ValueError(
            "The saved refined transcript does not match the current "
            "guard output. Run guard_dev again."
        )

    return refined, refined_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate candidate meeting documentation."
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    provider = None

    try:
        cfg = load_config()
        transcript, source_text = load_guarded_transcript(args.run_dir)

        model = cfg["llm"]["minutes_model"]
        temperature = cfg["llm"]["temperature"]

        print(f"Minutes model: {model}")
        print("Input: guarded refined transcript")
        print("This request consumes API quota.")

        provider = GeminiProvider(get_api_key())
        started = perf_counter()

        record = generate_minutes(
            transcript,
            provider,
            model=model,
            temperature=temperature,
            on_status=print,
        )

        elapsed = perf_counter() - started
        record_json = record.model_dump_json(indent=2)

        metadata = {
            "status": "candidate_evidence_unverified",
            "implementation_version": MINUTES_VERSION,
            "provider": provider.name,
            "model": model,
            "temperature": temperature,
            "source_file": "refined_transcript.json",
            "source_sha256": text_hash(source_text),
            "prompt_file": DEFAULT_PROMPT.name,
            "prompt_sha256": sha256(DEFAULT_PROMPT.read_bytes()).hexdigest(),
            "record_sha256": text_hash(record_json),
            "elapsed_seconds": elapsed,
            "created": datetime.now(timezone.utc).isoformat(),
        }

        output_path = args.run_dir / "candidate_meeting_record.json"
        output_path.write_text(record_json, encoding="utf-8")

        (args.run_dir / "minutes_meta.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n" + record_json)
        print(f"\nCompleted in {elapsed:.1f}s.")
        print(f"Saved: {output_path}")
        print("Schema validation passed. Evidence has not been verified.")
        return 0

    except (ConfigError, LLMError, MinutesError, ValueError, OSError) as exc:
        print(f"\nERROR: {exc}")
        return 2

    finally:
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())