import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from tracemeet.config import ConfigError, get_api_key, load_config
from tracemeet.guards.diff import edit_metrics, transcript_edits
from tracemeet.llm.base import LLMError
from tracemeet.llm.gemini import GeminiProvider
from tracemeet.schemas import Transcript
from tracemeet.stages.refine import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_MAX_CHARS,
    DEFAULT_PROMPT,
    REFINEMENT_VERSION,
    RefinementError,
    refine_transcript,
)


def read_saved_terms(run_dir: Path) -> str:
    path = run_dir / "meta.json"
    if not path.exists():
        return ""

    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("meta.json must contain a JSON object.")

    terms = metadata.get("terms", "")
    if not isinstance(terms, str):
        raise ValueError("The terms field in meta.json must be a string.")

    return terms.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refine a saved transcript with chunking and checkpoints."
    )
    parser.add_argument("transcript", type=Path)
    parser.add_argument("--terms", default=None)
    parser.add_argument("--chunk-size", type=int, default=25)
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
        print("Uncached chunks consume API quota.")

        provider = GeminiProvider(get_api_key())
        started = perf_counter()

        candidate = refine_transcript(
            raw,
            provider,
            model=model,
            glossary=glossary,
            temperature=temperature,
            chunk_size=args.chunk_size,
            checkpoint_dir=run_dir / "refinement_chunks",
            on_status=print,
        )

        elapsed = perf_counter() - started
        edits = transcript_edits(raw, candidate)
        metrics = edit_metrics(edits)

        for original, revised in zip(
            raw.segments, candidate.segments, strict=True
        ):
            if original.text != revised.text:
                print(f"\n[{original.id}]")
                print(f"  RAW:       {original.text}")
                print(f"  CANDIDATE: {revised.text}")

        metadata = {
            "status": "candidate_not_guarded",
            "implementation_version": REFINEMENT_VERSION,
            "provider": provider.name,
            "model": model,
            "temperature": temperature,
            "glossary": glossary,
            "chunk_size": args.chunk_size,
            "context_size": DEFAULT_CONTEXT_SIZE,
            "max_payload_characters": DEFAULT_MAX_CHARS,
            "prompt_sha256": sha256(DEFAULT_PROMPT.read_bytes()).hexdigest(),
            "source_sha256": sha256(source_text.encode("utf-8")).hexdigest(),
            "elapsed_seconds": elapsed,
            "metrics": metrics,
            "created": datetime.now(timezone.utc).isoformat(),
        }

        (run_dir / "candidate_refined_transcript.json").write_text(
            candidate.model_dump_json(indent=2), encoding="utf-8"
        )
        (run_dir / "candidate_edits.json").write_text(
            json.dumps(
                {"status": "candidate_not_guarded", "edits": edits},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (run_dir / "refinement_meta.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print(f"\nCompleted in {elapsed:.1f}s.")
        print(json.dumps(metrics, indent=2))
        print(f"Saved candidate, edits, and metadata in: {run_dir}")
        print("Sensitive-change guards have not run.")
        return 0

    except (ConfigError, LLMError, RefinementError, ValueError, OSError) as exc:
        print(f"\nERROR: {exc}")
        print("Any completed chunk checkpoints are retained.")
        return 2

    finally:
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())