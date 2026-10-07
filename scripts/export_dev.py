import argparse
import json
from hashlib import sha256
from pathlib import Path

from scripts.minutes_dev import load_guarded_transcript
from tracemeet.export.bundle import build_exports
from tracemeet.schemas import MeetingRecord, Transcript


def text_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export one existing meeting record without API calls."
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    try:
        refined, source_text = load_guarded_transcript(args.run_dir)

        candidate_text = (
            args.run_dir / "candidate_meeting_record.json"
        ).read_text(encoding="utf-8")

        metadata = json.loads(
            (args.run_dir / "minutes_meta.json").read_text(
                encoding="utf-8"
            )
        )
        if not isinstance(metadata, dict):
            raise ValueError("minutes_meta.json must contain an object.")

        if metadata.get("source_sha256") != text_hash(source_text):
            raise ValueError(
                "Meeting record metadata does not match the transcript."
            )
        if metadata.get("record_sha256") != text_hash(candidate_text):
            raise ValueError(
                "Meeting record contents do not match their metadata."
            )

        candidate = MeetingRecord.model_validate_json(candidate_text)
        raw = Transcript.model_validate_json(
            (args.run_dir / "raw_transcript.json").read_text(
                encoding="utf-8"
            )
        )

        bundle = build_exports(
            candidate,
            raw_transcript=raw,
            refined_transcript=refined,
        )

        output_dir = args.run_dir / "exports"
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, content in {
            **bundle.files,
            "tracemeet_bundle.zip": bundle.zip_bytes,
        }.items():
            destination = output_dir / name
            temporary = destination.with_name(destination.name + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(destination)

        report = json.loads(bundle.files["evidence_report.json"])
        print("Citation checks:")
        print(json.dumps(report.get("counts", {}), indent=2))
        print(f"\nSaved exports: {output_dir}")
        print(f"Tasks exported: {len(bundle.record.action_items)}")
        print("No API calls were made.")
        return 0

    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())