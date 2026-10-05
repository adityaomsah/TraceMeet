import re
from dataclasses import dataclass
from typing import Iterable

from tracemeet.guards.diff import transcript_edits
from tracemeet.schemas import (
    Correction,
    EditStatus,
    Segment,
    Transcript,
)

_WORD = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")

_NUMBER = re.compile(
    r"(?<!\w)[+-]?\d+(?:[.,:/-]\d+)*"
    r"|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|million|billion|lakh|crore)\b",
    re.IGNORECASE,
)

NEGATIONS = {
    "no", "not", "never", "without", "neither", "nor",
}

COMMITMENTS = {
    "will", "shall", "must", "should", "would", "could", "can",
    "may", "might", "agree", "agreed", "approve", "approved",
    "reject", "rejected", "propose", "proposed", "suggest",
    "suggested", "commit", "committed", "promise", "promised",
    "confirm", "confirmed", "decide", "decided", "tentative",
    "perhaps", "probably", "possibly",
}

DATES = {
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "today", "tomorrow", "yesterday", "tonight",
    "next", "previous", "before", "after",
}

UNITS = {
    "percent", "percentage", "dollar", "dollars", "rupee", "rupees",
    "usd", "inr", "eur", "gbp",
    "second", "seconds", "minute", "minutes", "hour", "hours",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "kg", "gram", "grams", "kilogram", "kilograms",
    "meter", "meters", "metre", "metres", "km", "cm", "mm",
    "mb", "gb", "tb", "mbps", "gbps",
}

CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "shan't": "shall not",
}


@dataclass
class GuardResult:
    transcript: Transcript
    corrections: list[Correction]
    edits: list[dict]


def _normalise(text: str) -> str:
    """Normalize known equivalents for comparison, not output."""
    text = text.casefold().replace("’", "'")

    for original, expanded in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(original)}\b", expanded, text)

    text = re.sub(r"\bcannot\b", "can not", text)

    # don't -> do not; isn't -> is not; shouldn't -> should not
    return re.sub(r"\b([a-z]+)n't\b", r"\1 not", text)


def _words(text: str) -> list[str]:
    return _WORD.findall(_normalise(text))


def _selected(words: list[str], vocabulary: set[str]) -> list[str]:
    # Preserve order and repeated markers.
    return [word for word in words if word in vocabulary]


def _name_count(text: str, name: str) -> int:
    parts = name.casefold().split()
    if not parts:
        return 0

    pattern = r"(?<!\w)" + r"\s+".join(
        re.escape(part) for part in parts
    ) + r"(?!\w)"
    return len(re.findall(pattern, text.casefold()))


def sensitive_flags(
    original: str,
    candidate: str,
    protected_names: Iterable[str] = (),
) -> list[str]:
    old_words = _words(original)
    new_words = _words(candidate)
    flags = []

    old_numbers = [m.group().casefold() for m in _NUMBER.finditer(original)]
    new_numbers = [m.group().casefold() for m in _NUMBER.finditer(candidate)]

    if old_numbers != new_numbers:
        flags.append("number_changed")

    checks = (
        ("negation_changed", NEGATIONS),
        ("commitment_changed", COMMITMENTS),
        ("date_or_time_changed", DATES),
        ("unit_changed", UNITS),
    )

    for flag, vocabulary in checks:
        if _selected(old_words, vocabulary) != _selected(new_words, vocabulary):
            flags.append(flag)

    if re.findall(r"[$₹€£%]", original) != re.findall(r"[$₹€£%]", candidate):
        flags.append("currency_or_percent_changed")

    if any(
        _name_count(original, name) != _name_count(candidate, name)
        for name in protected_names
        if name.strip()
    ):
        flags.append("protected_name_changed")

    # This remains a cautious capitalization heuristic, not NER.
    # Equivalent normalized wording should not trigger it solely
    # because a contraction was expanded at the start of a sentence.
    if old_words != new_words:
        old_capitalized = [
            word.casefold()
            for word in _WORD.findall(original)
            if word[0].isupper()
        ]
        new_capitalized = [
            word.casefold()
            for word in _WORD.findall(candidate)
            if word[0].isupper()
        ]

        if old_capitalized != new_capitalized:
            flags.append("capitalized_token_changed")

    return flags


def guard_refinement(
    raw: Transcript,
    candidate: Transcript,
    *,
    protected_names: Iterable[str] = (),
) -> GuardResult:
    # Reconstruct from plain data so objects retained across a code reload
    # do not have to share the current Pydantic class identity.
    # This performs normal validation; it does not bypass any checks.
    raw = Transcript.model_validate(raw.model_dump(mode="python"))
    candidate = Transcript.model_validate(candidate.model_dump(mode="python"))

    raw_ids = [segment.id for segment in raw.segments]
    candidate_ids = [segment.id for segment in candidate.segments]

    if raw_ids != candidate_ids:
        raise ValueError("Raw and candidate segment IDs must match exactly.")

    names = tuple(protected_names)
    edits = transcript_edits(raw, candidate)
    edits_by_segment: dict[str, list[dict]] = {}

    for edit in edits:
        edits_by_segment.setdefault(edit["segment_id"], []).append(edit)

    output_segments: list[dict] = []
    corrections: list[Correction] = []

    for original, revised in zip(
        raw.segments, candidate.segments, strict=True
    ):
        if (original.start, original.end) != (revised.start, revised.end):
            raise ValueError(
                f"Timestamps changed for segment {original.id}."
            )

        segment_changes = edits_by_segment.get(original.id, [])

        if not segment_changes:
            # Preserve raw wording and its ASR diagnostic fields.
            output_segments.append(original.model_dump(mode="python"))
            continue

        # Check the entire segment, including edits marked whitespace-only.
        flags = sensitive_flags(original.text, revised.text, names)
        status = (
            EditStatus.NEEDS_REVIEW if flags else EditStatus.ACCEPTED
        )
        reason = (
            "Sensitive change detected; the entire raw segment was retained."
            if flags
            else "No configured sensitive-change heuristic was triggered."
        )

        if flags:
            output_segments.append(original.model_dump(mode="python"))
        else:
            # ASR confidence belongs to the raw wording, so do not attach
            # it to text rewritten by the refiner.
            output_segments.append(
                {
                    "id": original.id,
                    "start": original.start,
                    "end": original.end,
                    "text": revised.text,
                    "avg_logprob": None,
                    "no_speech_prob": None,
                }
            )

        for edit in segment_changes:
            corrections.append(
                Correction(
                    segment_id=original.id,
                    original=edit["original"],
                    replacement=edit["replacement"],
                    status=status,
                    flags=list(flags),
                    reason=reason,
                )
            )

            edit.update(
                status=status.value,
                flags=list(flags),
                reason=reason,
                applied=status == EditStatus.ACCEPTED,
            )

    guarded = Transcript.model_validate(
        {
            "segments": output_segments,
            "stt_model": raw.stt_model,
            "language": raw.language,
            "duration_s": raw.duration_s,
        }
    )

    return GuardResult(
        transcript=guarded,
        corrections=corrections,
        edits=edits,
    )