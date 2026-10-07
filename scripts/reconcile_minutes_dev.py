import argparse
import json
import os
from hashlib import sha256
from pathlib import Path

from scripts.group_minutes_dev import load_observations
from scripts.minutes_dev import load_guarded_transcript
from tracemeet.config import ConfigError, load_config
from tracemeet.llm.base import LLMError
from tracemeet.llm.groq_provider import GroqProvider
from tracemeet.stages.minutes_map import save_json_atomic
from tracemeet.stages.minutes_reconcile import (
    RECONCILE_SYSTEM,
    RECONCILE_VERSION,
    GroupDraft,
    ReconciliationError,
    build_group_jobs,
    reconcile_groups,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile discussion groups against source transcripts."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    provider = None
    try:
        load_config()
        transcript, source_text = load_guarded_transcript(args.run_dir)
        observations, source_hash, notes_hash = load_observations(
            args.run_dir, transcript, source_text
        )

        groups_bytes = (args.run_dir / "meeting_groups.json").read_bytes()
        group_document = json.loads(groups_bytes)
        if not isinstance(group_document, dict):
            raise ValueError("meeting_groups.json must contain an object.")

        if group_document.get("source_text_sha256") != source_hash:
            raise ValueError("Grouping belongs to a different transcript.")

        if (
            group_document.get("provisional_notes_file_sha256")
            != notes_hash
        ):
            raise ValueError(
                "Provisional notes changed. Run group_minutes_dev again."
            )

        jobs = build_group_jobs(
            transcript,
            observations,
            group_document["groups"],
        )

        provider = GroqProvider(
            os.getenv("GROQ_API_KEY", ""),
            request_budget=7400,
            max_completion_tokens=3072,
        )
        model = "openai/gpt-oss-120b"

        plan = []
        for job in jobs:
            budget = provider.estimate_request(
                system=RECONCILE_SYSTEM,
                prompt=job.prompt,
                schema=GroupDraft,
            )
            plan.append(
                {
                    "group_id": job.group_id,
                    "label": job.label,
                    "observations": len(job.observation_ids),
                    "source_segments": len(job.allowed_ids),
                    "budget": budget,
                }
            )
            print(
                f"{job.group_id}: "
                f"{budget['estimated_total_tokens']:,} estimated tokens "
                f"| fits={budget['fits']} | {job.label}"
            )

        save_json_atomic(
            args.run_dir / "reconciliation_plan.json",
            {"version": RECONCILE_VERSION, "groups": plan},
        )

        all_fit = all(item["budget"]["fits"] for item in plan)
        if args.check_only:
            print("\nNo inference API calls made.")
            print(f"All groups fit: {all_fit}")
            return 0

        if not all_fit:
            raise ReconciliationError(
                "At least one group exceeds the request budget. "
                "No inference calls made. Share the planning output."
            )

        print("\nUncached groups consume quota; run no other Groq script.")
        hashes = {
            "source_text_sha256": source_hash,
            "provisional_notes_file_sha256": notes_hash,
            "groups_file_sha256": sha256(groups_bytes).hexdigest(),
        }

        results = reconcile_groups(
            jobs,
            transcript,
            provider,
            model=model,
            checkpoint_dir=args.run_dir / "minutes_reconcile_chunks",
            input_hashes=hashes,
        )

        output_path = args.run_dir / "reconciled_group_records.json"
        save_json_atomic(
            output_path,
            {
                "status": "candidate_group_records",
                "implementation_version": RECONCILE_VERSION,
                "provider": provider.name,
                "model": model,
                **hashes,
                "groups": results,
                "semantic_support_checked": False,
                "limitations": [
                    "Source IDs and quote retrieval were checked.",
                    "Claim interpretation remains model-generated.",
                    "Neighbouring context does not guarantee speaker identity.",
                    "Incorrect grouping can miss cross-group relationships.",
                    "Final consolidation and overall summary are pending.",
                ],
            },
        )

        print(f"\nSaved: {output_path}")
        print("All groups processed; final consolidation is next.")
        return 0

    except (
        ConfigError, LLMError, ReconciliationError,
        OSError, ValueError, KeyError, TypeError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 2
    finally:
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())