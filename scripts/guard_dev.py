import argparse
import json
from hashlib import sha256
from pathlib import Path

from tracemeet.guards.sensitive import guard_refinement
from tracemeet.schemas import EditStatus, Transcript

GUARD_VERSION = "sensitive-v1"


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
        raw_text = (args.run_dir / "raw_transcript.json").read_text(
            encoding="utf-8"
        )
        candidate_text = (
            args.run_dir / "candidate_refined_transcript.json"
        ).read_text(encoding="utf-8")

        # Check that the candidate metadata refers to this raw transcript.
        metadata = json.loads(
            (args.run_dir / "refinement_meta.json").read_text(encoding="utf-8")
        )
        if not isinstance(metadata, dict):
            raise ValueError("refinement_meta.json must contain an object.")

        raw_hash = sha256(raw_text.encode("utf-8")).hexdigest()
        if metadata.get("source_sha256") != raw_hash:
            raise ValueError(
                "Refinement metadata does not match this raw transcript. "
                "Run refine_dev again before running the guards."
            )

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
            "raw_sha256": raw_hash,
            "candidate_sha256": sha256(
                candidate_text.encode("utf-8")
            ).hexdigest(),
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