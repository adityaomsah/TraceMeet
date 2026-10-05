import argparse
import json
from hashlib import sha256
from pathlib import Path

from tracemeet.guards.evidence import verify_evidence
from tracemeet.schemas import MeetingRecord, Transcript


def text_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check meeting-record citations against the refined transcript."
    )
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    try:
        transcript_text = (
            args.run_dir / "refined_transcript.json"
        ).read_text(encoding="utf-8")

        record_text = (
            args.run_dir / "candidate_meeting_record.json"
        ).read_text(encoding="utf-8")

        metadata = json.loads(
            (args.run_dir / "minutes_meta.json").read_text(encoding="utf-8")
        )
        if not isinstance(metadata, dict):
            raise ValueError("minutes_meta.json must contain an object.")

        source_hash = text_hash(transcript_text)
        record_hash = text_hash(record_text)

        if metadata.get("source_sha256") != source_hash:
            raise ValueError(
                "The refined transcript no longer matches the minutes input. "
                "Regenerate minutes before checking evidence."
            )

        if metadata.get("record_sha256") != record_hash:
            raise ValueError(
                "The candidate record no longer matches its metadata. "
                "Regenerate minutes or restore the original record."
            )

        transcript = Transcript.model_validate_json(transcript_text)
        candidate = MeetingRecord.model_validate_json(record_text)

        result = verify_evidence(candidate, transcript)
        checked_json = result.record.model_dump_json(indent=2)

        result.report.update(
            source_sha256=source_hash,
            candidate_sha256=record_hash,
            checked_record_sha256=text_hash(checked_json),
        )

        checked_path = (
            args.run_dir / "citation_checked_meeting_record.json"
        )
        report_path = args.run_dir / "evidence_report.json"

        checked_path.write_text(checked_json, encoding="utf-8")
        report_path.write_text(
            json.dumps(result.report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("Citation checks:")
        print(json.dumps(result.report["counts"], indent=2))

        for check in result.report["citation_checks"]:
            if check["status"] != "exact_match":
                print(
                    f"\nINVALID: {check['path']} "
                    f"({check['segment_id']}, {check['status']})"
                )
                print(f"  Quote: {check['quote']!r}")

        print("\nRecord changes and flags:")
        for issue in result.report["issues"]:
            print(
                f"- {issue['path']}: {issue['action']} "
                f"— {issue['reason']}"
            )

        print(f"\nSaved: {checked_path}")
        print(f"Saved: {report_path}")
        print("Citation presence checked; semantic support is not verified.")
        return 0

    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())