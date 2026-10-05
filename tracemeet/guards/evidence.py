from dataclasses import dataclass

from tracemeet.schemas import Evidence, MeetingRecord, Transcript

EVIDENCE_VERSION = "evidence-v1"


@dataclass
class EvidenceResult:
    record: MeetingRecord
    report: dict


def _is_word_character(character: str) -> bool:
    return character.isalnum() or character == "_"


def _exact_span(text: str, quote: str) -> tuple[int, int] | None:
    """Find an exact substring without cutting through a word."""
    position = text.find(quote)

    while position != -1:
        end = position + len(quote)

        cuts_left_word = (
            position > 0
            and _is_word_character(quote[0])
            and _is_word_character(text[position - 1])
        )
        cuts_right_word = (
            end < len(text)
            and _is_word_character(quote[-1])
            and _is_word_character(text[end])
        )

        if not cuts_left_word and not cuts_right_word:
            return position, end

        position = text.find(quote, position + 1)

    return None


def check_citation(
    evidence: Evidence,
    transcript: Transcript,
) -> dict:
    """Check textual presence only, not semantic support."""
    segment = transcript.by_id().get(evidence.segment_id)

    result = {
        "segment_id": evidence.segment_id,
        "quote": evidence.quote,
        "status": None,
        "quote_start": None,
        "quote_end": None,
        "start_s": None,
        "end_s": None,
    }

    if segment is None:
        result["status"] = "unknown_segment"
        return result

    # Punctuation or whitespace alone is not useful supporting evidence.
    if not any(character.isalnum() for character in evidence.quote):
        result["status"] = "empty_or_punctuation_only"
        return result

    span = _exact_span(segment.text, evidence.quote)

    if span is None:
        result["status"] = "quote_mismatch"
        return result

    result.update(
        status="exact_match",
        quote_start=span[0],
        quote_end=span[1],
        start_s=segment.start,
        end_s=segment.end,
    )
    return result


def verify_evidence(
    candidate: MeetingRecord,
    transcript: Transcript,
) -> EvidenceResult:
    """
    Filter invalid citations without mutating the candidate.

    An exact match establishes source-text presence.
    It does not establish entailment, correct speaker attribution,
    decision status, or task ownership.
    """
    checks = []
    issues = [
        {
            "path": "summary",
            "action": "flag",
            "reason": (
                "The current summary field has no citations. "
                "It is retained as an unverified overview."
            ),
        }
    ]
    dropped_items = 0
    cleared_fields = 0

    def check_group(evidence_list: list[Evidence], path: str) -> bool:
        if not evidence_list:
            return False

        valid = True
        for index, evidence in enumerate(evidence_list):
            check = check_citation(evidence, transcript)
            check["path"] = f"{path}[{index}]"
            checks.append(check)
            if check["status"] != "exact_match":
                valid = False

        return valid

    # Serialization gives us independent data to modify.
    output = candidate.model_dump(mode="json")

    for field in ("minutes", "decisions", "open_questions"):
        retained = []

        for index, item in enumerate(getattr(candidate, field)):
            path = f"{field}[{index}]"

            if check_group(item.evidence, f"{path}.evidence"):
                retained.append(item.model_dump(mode="json"))
            else:
                dropped_items += 1
                issues.append(
                    {
                        "path": path,
                        "action": "drop_item",
                        "reason": (
                            "At least one required citation is missing "
                            "or does not match its source segment."
                        ),
                    }
                )

        output[field] = retained

    retained_tasks = []

    for index, task in enumerate(candidate.action_items):
        path = f"action_items[{index}]"

        task_valid = check_group(task.evidence, f"{path}.evidence")

        owner_valid = (
            check_group(task.owner_evidence, f"{path}.owner_evidence")
            if task.owner is not None
            else True
        )
        deadline_valid = (
            check_group(task.deadline_evidence, f"{path}.deadline_evidence")
            if task.deadline is not None
            else True
        )

        if not task_valid:
            dropped_items += 1
            issues.append(
                {
                    "path": path,
                    "action": "drop_item",
                    "reason": "The task's required evidence failed citation checks.",
                }
            )
            continue

        task_data = task.model_dump(mode="json")

        for field, valid in (
            ("owner", owner_valid),
            ("deadline", deadline_valid),
        ):
            if not valid:
                task_data[field] = None
                task_data[f"{field}_evidence"] = []
                cleared_fields += 1

                issues.append(
                    {
                        "path": f"{path}.{field}",
                        "action": "clear_field",
                        "reason": (
                            f"The {field} citation failed. The task is retained "
                            f"with {field} unspecified."
                        ),
                    }
                )

        retained_tasks.append(task_data)

    output["action_items"] = retained_tasks

    # Revalidate after filtering and clearing related fields.
    checked_record = MeetingRecord.model_validate(output)

    report = {
        "verifier_version": EVIDENCE_VERSION,
        "status": "citation_checked_semantics_unverified",
        "match_policy": "exact_substring_with_word_boundaries",
        "semantic_support_checked": False,
        "summary_status": "uncited",
        "counts": {
            "exact_matches": sum(
                check["status"] == "exact_match" for check in checks
            ),
            "invalid_citations": sum(
                check["status"] != "exact_match" for check in checks
            ),
            "dropped_items": dropped_items,
            "cleared_fields": cleared_fields,
        },
        # Paths refer to the original candidate, before filtering.
        "citation_checks": checks,
        "issues": issues,
        "limitations": [
            "Exact quotes do not prove that an associated claim follows from them.",
            "Speaker attribution and decision/task statuses are not verified.",
            "The overall summary has no citations in the current schema.",
            "Playback timestamps cover the source segment, not the exact quote.",
        ],
    }

    return EvidenceResult(record=checked_record, report=report)