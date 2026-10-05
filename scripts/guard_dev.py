import argparse
import json
from hashlib import sha256
from pathlib import Path

from tracemeet.guards.sensitive import guard_refinement
from tracemeet.schemas import EditStatus, Transcript

GUARD_VERSION = "sensitive-v1"


def hash_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def read_json_source(path: Path) -> tuple[bytes, str]:
    """Read once; reproduce read_text() newline handling explicitly."""
    data = path.read_bytes()
    text = data.decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return data, text


def verify_source_hash(
    expected: str,
    raw_bytes: bytes,
    raw_text: str,
) -> str:
    """Accept the two hash conventions used by existing TraceMeet writers."""
    if not isinstance(expected, str) or not expected:
        raise ValueError("Refinement metadata is missing source_sha256.")

    expected = expected.lower()

    if expected == hash_bytes(raw_bytes):
        return "file_bytes"

    if expected == hash_bytes(raw_text.encode("utf-8")):
        return "normalized_text"

    raise ValueError(
        "Refinement metadata matches neither the raw file bytes nor "
        "its newline-normalized text. The source may have changed. "
        "Keep the run files and investigate before rerunning refinement."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check proposed edits and save a guarded transcript."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Folder containing raw and candidate transcript JSON files.",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=[],
        help='Known names, each quoted separately: --names "Aditya Om Sah"',
    )
    args = parser.parse_args()

    try:
        raw_bytes, raw_text = read_json_source(
            args.run_dir / "raw_transcript.json"
        )
        candidate_bytes, candidate_text = read_json_source(
            args.run_dir / "candidate_refined_transcript.json"
        )

        metadata = json.loads(
            (args.run_dir / "refinement_meta.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(metadata, dict):
            raise ValueError("refinement_meta.json must contain an object.")

        source_hash_mode = verify_source_hash(
            metadata.get("source_sha256"),
            raw_bytes,
            raw_text,
        )
        print(f"Source integrity check passed: {source_hash_mode}.")

        raw = Transcript.model_validate_json(raw_text)
        candidate = Transcript.model_validate_json(candidate_text)

        result = guard_refinement(
            raw,
            candidate,
            protected_names=args.names,
        )

        accepted_ids = {
            correction.segment_id
            for correction in result.corrections
            if correction.status == EditStatus.ACCEPTED
        }
        review_ids = {
            correction.segment_id
            for correction in result.corrections
            if correction.status == EditStatus.NEEDS_REVIEW
        }

        report = {
            "guard_version": GUARD_VERSION,
            "protected_names": args.names,
            "source_hash_mode": source_hash_mode,

            # Keep the existing CLI convention for downstream readers.
            "raw_sha256": hash_bytes(raw_text.encode("utf-8")),
            "candidate_sha256": hash_bytes(candidate_text.encode("utf-8")),
            "text_hash_mode": "utf8_normalized_newlines",

            # Explicit byte hashes for auditing the actual files.
            "raw_file_sha256": hash_bytes(raw_bytes),
            "candidate_file_sha256": hash_bytes(candidate_bytes),

            "changed_segments_applied": len(accepted_ids),
            "segments_needing_review": len(review_ids),
            "corrections": [
                correction.model_dump(mode="json")
                for correction in result.corrections
            ],
            "edits": result.edits,
        }

        (args.run_dir / "refined_transcript.json").write_text(
            result.transcript.model_dump_json(indent=2),
            encoding="utf-8",
        )
        (args.run_dir / "guard_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        for correction in result.corrections:
            print(
                f"[{correction.segment_id}] {correction.status.value}: "
                f"{correction.original!r} -> {correction.replacement!r}"
            )
            if correction.flags:
                print(f"  Flags: {', '.join(correction.flags)}")

        print(f"\nChanged segments applied: {len(accepted_ids)}")
        print(f"Segments needing review: {len(review_ids)}")
        print("Saved refined_transcript.json and guard_report.json")
        print("Flagged segments retain their raw wording.")
        return 0

    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())