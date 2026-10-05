import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from tracemeet.config import ConfigError, get_api_key, load_config
from tracemeet.llm.base import LLMError
from tracemeet.llm.gemini import GeminiProvider
from tracemeet.schemas import Transcript
from tracemeet.stages.refine import (
    DEFAULT_PROMPT,
    RefinementError,
    refine_transcript,
)


def read_saved_terms(run_dir: Path) -> str:
    meta_path = run_dir / "meta.json"

    if not meta_path.exists():
        return ""

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("meta.json must contain a JSON object.")

    terms = metadata.get("terms", "")
    if not isinstance(terms, str):
        raise ValueError("The terms field in meta.json must be a string.")

    return terms.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate an unguarded refinement candidate."
    )
    parser.add_argument(
        "transcript",
        type=Path,
        help="Path to a saved raw_transcript.json",
    )
    parser.add_argument(
        "--terms",
        default=None,
        help="Override the glossary saved in the run's meta.json.",
    )
    args = parser.parse_args()

    provider = None

    try:
        cfg = load_config()
        source_text = args.transcript.read_text(encoding="utf-8")
        raw = Transcript.model_validate_json(source_text)
        run_dir = args.transcript.parent

        glossary = (
            args.terms.strip()
            if args.terms is not None
            else read_saved_terms(run_dir)
        )

        model = cfg["llm"]["refine_model"]
        temperature = cfg["llm"]["temperature"]

        print(f"Refinement model: {model}")
        print(f"Segments: {len(raw.segments)}")
        print(f"Glossary: {glossary or '(none)'}")
        print("Calling the model; this consumes API quota.")

        provider = GeminiProvider(get_api_key())
        started = perf_counter()

        candidate = refine_transcript(
            raw,
            provider,
            model=model,
            glossary=glossary,
            temperature=temperature,
        )

        elapsed = perf_counter() - started
        changed_count = 0

        for original, revised in zip(
            raw.segments, candidate.segments, strict=True
        ):
            if original.text != revised.text:
                changed_count += 1
                print(f"\n[{original.id}]")
                print(f"  RAW:       {original.text}")
                print(f"  CANDIDATE: {revised.text}")

        output_path = run_dir / "candidate_refined_transcript.json"
        output_path.write_text(
            candidate.model_dump_json(indent=2),
            encoding="utf-8",
        )

        metadata = {
            "status": "candidate_not_guarded",
            "provider": provider.name,
            "model": model,
            "temperature": temperature,
            "glossary": glossary,
            "prompt_file": DEFAULT_PROMPT.name,
            "prompt_sha256": sha256(
                DEFAULT_PROMPT.read_bytes()
            ).hexdigest(),
            "source_sha256": sha256(
                source_text.encode("utf-8")
            ).hexdigest(),
            "elapsed_seconds": elapsed,
            "changed_segments": changed_count,
            "created": datetime.now(timezone.utc).isoformat(),
        }

        (run_dir / "refinement_meta.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(
            f"\nChanged {changed_count}/{len(raw.segments)} segments "
            f"in {elapsed:.1f}s."
        )
        print(f"Saved: {output_path}")
        print("Candidate only: sensitive-change guards have not run.")
        return 0

    except (ConfigError, LLMError, RefinementError, ValueError, OSError) as exc:
        print(f"\nERROR: {exc}")
        return 2
    finally:
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())