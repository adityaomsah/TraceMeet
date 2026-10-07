import argparse
import json
import os
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

from scripts.group_minutes_dev import load_observations
from scripts.minutes_dev import load_guarded_transcript
from tracemeet.config import ConfigError, load_config
from tracemeet.guards.evidence import _exact_span, verify_evidence
from tracemeet.llm.base import LLMError
from tracemeet.llm.groq_provider import GroqProvider
from tracemeet.stages.minutes_finalize import (
    FINALIZE_VERSION,
    OUTCOME_SYSTEM,
    TASK_SYSTEM,
    FinalizationError,
    FinalizationRunner,
    ReviewedOutcomes,
    ReviewedTasks,
    finalize_record,
    prepare_review,
)
from tracemeet.stages.minutes_map import save_json_atomic
from tracemeet.stages.minutes_reconcile import (
    GroupRecord,
    ReconciliationError,
)


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def write_text_atomic(path: Path, text: str):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(text.encode("utf-8"))
    temporary.replace(path)


def load_records(run_dir, transcript, source_hash, notes_hash):
    groups_bytes = (run_dir / "meeting_groups.json").read_bytes()
    groups_document = json.loads(groups_bytes)

    data_bytes = (run_dir / "reconciled_group_records.json").read_bytes()
    document = json.loads(data_bytes)

    if not isinstance(document, dict) or not isinstance(groups_document, dict):
        raise ValueError("Group files must contain JSON objects.")

    if (
        document.get("source_text_sha256") != source_hash
        or document.get("provisional_notes_file_sha256") != notes_hash
        or document.get("groups_file_sha256") != digest(groups_bytes)
    ):
        raise ValueError("Reconciled records do not match current inputs.")

    expected = groups_document["groups"]
    actual = document["groups"]
    if [g["group_id"] for g in actual] != [
        g["group_id"] for g in expected
    ]:
        raise ValueError("Reconciled group list is incomplete or reordered.")

    sources = transcript.by_id()
    records = []

    for group, planned in zip(actual, expected, strict=True):
        if group["observation_ids"] != planned["observation_ids"]:
            raise ValueError("Group observation membership changed.")

        record = GroupRecord.model_validate(group["record"])
        for field in (
            "minutes", "decisions", "action_items", "open_questions"
        ):
            for item in getattr(record, field):
                for evidence_field in (
                    "evidence", "owner_evidence", "deadline_evidence"
                ):
                    for evidence in getattr(item, evidence_field, []):
                        segment = sources.get(evidence.segment_id)
                        if (
                            segment is None
                            or _exact_span(segment.text, evidence.quote) is None
                        ):
                            raise ValueError(
                                "A reconciled citation does not match source."
                            )
        records.append(record)

    return records, digest(data_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Review and finalize saved meeting group records."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    provider = None
    try:
        load_config()
        transcript, source_text = load_guarded_transcript(args.run_dir)
        _, source_hash, notes_hash = load_observations(
            args.run_dir, transcript, source_text
        )
        records, records_hash = load_records(
            args.run_dir, transcript, source_hash, notes_hash
        )

        provider = GroqProvider(
            os.getenv("GROQ_API_KEY", ""),
            request_budget=7400,
            max_completion_tokens=3072,
        )
        model = "openai/gpt-oss-120b"

        budgets = []
        for kind, system, schema in (
            ("tasks", TASK_SYSTEM, ReviewedTasks),
            ("outcomes", OUTCOME_SYSTEM, ReviewedOutcomes),
        ):
            job = prepare_review(kind, records, transcript)
            budget = provider.estimate_request(
                system=system, prompt=job.prompt, schema=schema
            )
            budgets.append(budget)
            print(f"\n{kind}:")
            print(json.dumps(budget, indent=2))

        if args.check_only:
            print("\nNo inference API calls made.")
            print(
                "Overview budget is checked after reviewed outcomes exist."
            )
            return 0

        if not all(budget["fits"] for budget in budgets):
            raise FinalizationError(
                "A review exceeds budget. No inference calls made."
            )

        runner = FinalizationRunner(
            provider,
            model,
            args.run_dir / "minutes_finalize_chunks",
            {
                "source_text_sha256": source_hash,
                "reconciled_records_file_sha256": records_hash,
            },
        )
        candidate = finalize_record(records, transcript, runner)
        checked = verify_evidence(candidate, transcript)

        candidate_text = candidate.model_dump_json(indent=2)
        write_text_atomic(
            args.run_dir / "candidate_meeting_record.json",
            candidate_text,
        )

        save_json_atomic(
            args.run_dir / "minutes_meta.json",
            {
                "implementation_version": FINALIZE_VERSION,
                "provider": provider.name,
                "model": model,
                "temperature": 0.0,
                "source_sha256": source_hash,
                "record_sha256": digest(candidate_text.encode("utf-8")),
                "reconciled_records_file_sha256": records_hash,
                "created": datetime.now(timezone.utc).isoformat(),
                "generation_provenance": "minutes_finalize_chunks/",
            },
        )
        write_text_atomic(
            args.run_dir / "citation_checked_meeting_record.json",
            checked.record.model_dump_json(indent=2),
        )
        save_json_atomic(
            args.run_dir / "evidence_report.json",
            checked.report,
        )

        print("\nFinal candidate counts:")
        for field in (
            "minutes", "decisions", "action_items", "open_questions"
        ):
            print(f"  {field}: {len(getattr(checked.record, field))}")

        print("\nCitation checks:")
        print(json.dumps(checked.report["counts"], indent=2))
        print(f"\nSaved finalization outputs in: {args.run_dir}")
        print("Semantic accuracy still requires inspection.")
        return 0

    except (
        ConfigError, LLMError, FinalizationError, ReconciliationError,
        OSError, ValueError, KeyError, TypeError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 2
    finally:
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())