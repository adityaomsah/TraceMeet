import argparse
import json
import os
from hashlib import sha256
from pathlib import Path

from scripts.minutes_dev import load_guarded_transcript
from tracemeet.config import ConfigError, load_config
from tracemeet.guards.evidence import _exact_span
from tracemeet.llm.base import LLMError
from tracemeet.llm.groq_provider import GroqProvider
from tracemeet.stages.minutes_group import (
    GROUP_SYSTEM,
    GROUP_VERSION,
    GroupingError,
    assignment_schema,
    generate_groups,
    grouping_payload,
    materialize_groups,
)
from tracemeet.stages.minutes_map import Observation, save_json_atomic


def load_observations(run_dir, transcript, source_text):
    path = run_dir / "provisional_meeting_notes.json"
    notes_bytes = path.read_bytes()
    document = json.loads(notes_bytes)

    if not isinstance(document, dict):
        raise ValueError("Provisional notes must contain a JSON object.")

    source_hash = sha256(source_text.encode("utf-8")).hexdigest()
    if document.get("source_text_sha256") != source_hash:
        raise ValueError(
            "Provisional notes do not match the current guarded transcript. "
            "Run map_minutes_dev again."
        )

    expected_ids = [segment.id for segment in transcript.segments]
    if document.get("input_segment_ids") != expected_ids:
        raise ValueError("Provisional notes have inconsistent input coverage.")

    items = document.get("observations")
    if not isinstance(items, list) or not items:
        raise ValueError("No provisional observations were found.")

    sources = transcript.by_id()
    seen = set()
    observations = []

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each observation must be an object.")

        observation_id = item.get("observation_id")
        if (
            not isinstance(observation_id, str)
            or not observation_id.strip()
            or observation_id in seen
        ):
            raise ValueError("Observation IDs must be nonempty and unique.")
        seen.add(observation_id)

        observation = Observation.model_validate(
            {
                key: value
                for key, value in item.items()
                if key != "observation_id"
            }
        )

        evidence_ids = []
        for evidence in observation.evidence:
            segment = sources.get(evidence.segment_id)
            if (
                segment is None
                or _exact_span(segment.text, evidence.quote) is None
            ):
                raise ValueError(
                    f"Invalid source evidence in {observation_id}."
                )
            evidence_ids.append(evidence.segment_id)

        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(f"Duplicate evidence in {observation_id}.")

        observations.append(
            {
                "observation_id": observation_id,
                **observation.model_dump(mode="json"),
            }
        )

    return observations, source_hash, sha256(notes_bytes).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Group observations across the entire meeting."
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

        model = "openai/gpt-oss-120b"
        provider = GroqProvider(
            os.getenv("GROQ_API_KEY", ""),
            request_budget=7400,
            max_completion_tokens=2048,
        )
        schema = assignment_schema(len(observations))
        budget = provider.estimate_request(
            system=GROUP_SYSTEM,
            prompt=grouping_payload(observations),
            schema=schema,
        )

        print(f"Observations: {len(observations)}")
        print("One required assignment field per observation.")
        print(json.dumps(budget, indent=2))

        if args.check_only:
            print("Budget check only. No inference API call made.")
            return 0

        plan = generate_groups(
            observations,
            provider,
            model=model,
            checkpoint_dir=args.run_dir / "minutes_group_chunks",
            source_sha256=source_hash,
            notes_sha256=notes_hash,
        )
        groups = materialize_groups(plan, observations)

        output = {
            "status": "grouped_not_reconciled",
            "implementation_version": GROUP_VERSION,
            "provider": provider.name,
            "model": model,
            "source_text_sha256": source_hash,
            "provisional_notes_file_sha256": notes_hash,
            "groups": groups,
            "semantic_support_checked": False,
        }
        output_path = args.run_dir / "meeting_groups.json"
        save_json_atomic(output_path, output)

        for group in groups:
            print(
                f"{group['group_id']}: {group['label']} "
                f"({len(group['observation_ids'])} observations)"
            )

        print(f"\nSaved: {output_path}")
        print("Every observation assigned exactly once.")
        print("Source-grounded reconciliation is still pending.")
        return 0

    except (
        ConfigError, LLMError, GroupingError, OSError, ValueError,
    ) as exc:
        print(f"ERROR: {exc}")
        return 2
    finally:
        if provider is not None:
            provider.close()


if __name__ == "__main__":
    raise SystemExit(main())