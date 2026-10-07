import argparse
import json
import os
import shutil
from uuid import uuid4
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
    plan_review_jobs,
    review_ids,
    candidate_catalog,
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

        plans = {}
        for kind, system, schema in (
            ("tasks", TASK_SYSTEM, ReviewedTasks),
            ("outcomes", OUTCOME_SYSTEM, ReviewedOutcomes),
        ):
            plans[kind] = plan_review_jobs(kind, records, transcript, provider)
            print(f"\n{kind}: {len(candidate_catalog(kind, records))} candidates; "
                  f"{len(plans[kind])} planned batches")
            for index, job in enumerate(plans[kind], 1):
                budget = provider.estimate_request(system=system, prompt=job.prompt, schema=schema)
                print(f"  Batch {index}: {len(review_ids(job))} candidates; "
                      f"{budget['estimated_total_tokens']:,} estimated tokens including output reserve")

        if args.check_only:
            print("\nNo inference API calls made.")
            print("Overview batching is budget-checked after reviewed outcomes exist.")
            return 0

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

        # Preserve previous outputs before replacing them, including the last
        # accuracy baseline. No earlier transcription/reconciliation is changed.
        names = (
            "candidate_meeting_record.json", "minutes_meta.json",
            "citation_checked_meeting_record.json", "evidence_report.json",
            "finalization_report.json",
        )
        existing = [args.run_dir / name for name in names if (args.run_dir / name).is_file()]
        if existing:
            archive = args.run_dir / "finalization_history" / (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
            )
            archive.mkdir(parents=True, exist_ok=False)
            for source in existing:
                shutil.copy2(source, archive / source.name)
            print(f"Previous outputs preserved: {archive}")

        finalization_report = {
            "version": FINALIZE_VERSION,
            "source_sha256": source_hash,
            "reconciled_records_file_sha256": records_hash,
            "semantic_support_checked": False,
            "candidate_catalogs": {
                kind: candidate_catalog(kind, records) for kind in plans
            },
            "review_batches": {
                kind: [{"candidate_ids": review_ids(job),
                        "allowed_segment_ids": list(job.allowed_ids)} for job in jobs]
                for kind, jobs in plans.items()
            },
            "validated_responses": runner.audit,
            "limitations": [
                "Coverage means every supplied candidate has a disposition, not that extraction found every real task.",
                "Removal reasons, statuses and assignments require semantic inspection.",
                "Review uses source excerpts; omitted context can contain answers or cancellations.",
                "Whole groups remain together, but semantic reconciliation across review batches is not guaranteed.",
                "Overview preserves group boundaries and topic accounting; it summarizes candidate topics, not full source speech.",
                "Summary is uncited. More than one overview batch produces concatenated batch summaries.",
            ],
        }

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

        save_json_atomic(args.run_dir / "finalization_report.json", finalization_report)

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

    except KeyboardInterrupt:
        print("Stopped; validated checkpoints remain saved.")
        return 130
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