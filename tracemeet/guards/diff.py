import re
from difflib import SequenceMatcher

from tracemeet.schemas import Transcript

_TOKEN_PATTERN = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)


def _tokens(text: str):
    return list(_TOKEN_PATTERN.finditer(text))


def _bounds(matches, start: int, end: int, text_length: int):
    left = matches[start].start() if start < len(matches) else text_length
    right = matches[end - 1].end() if end > start else left
    return left, right


def segment_edits(
    segment_id: str,
    original: str,
    replacement: str,
) -> list[dict]:
    old_tokens = _tokens(original)
    new_tokens = _tokens(replacement)

    matcher = SequenceMatcher(
        a=[token.group() for token in old_tokens],
        b=[token.group() for token in new_tokens],
        autojunk=False,
    )

    edits = []

    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        if operation == "equal":
            continue

        old_start, old_end = _bounds(old_tokens, i1, i2, len(original))
        new_start, new_end = _bounds(new_tokens, j1, j2, len(replacement))

        old_text = original[old_start:old_end]
        new_text = replacement[new_start:new_end]

        edits.append(
            {
                "segment_id": segment_id,
                "operation": operation,
                "raw_start": old_start,
                "raw_end": old_end,
                "candidate_start": new_start,
                "candidate_end": new_end,
                "original": old_text,
                "replacement": new_text,
                # An insertion/deletion may have an empty side.
                # Every character on both sides must be whitespace.
                "whitespace_only": all(
                    character.isspace()
                    for character in old_text + new_text
                ),
            }
        )

    return edits


def transcript_edits(raw: Transcript, candidate: Transcript) -> list[dict]:
    if [s.id for s in raw.segments] != [s.id for s in candidate.segments]:
        raise ValueError("Cannot diff transcripts with different segment IDs.")

    edits = []

    for original, revised in zip(
        raw.segments, candidate.segments, strict=True
    ):
        edits.extend(
            segment_edits(original.id, original.text, revised.text)
        )

    return edits


def edit_metrics(edits: list[dict]) -> dict[str, int]:
    def count_tokens(text: str) -> int:
        # Non-whitespace tokens include punctuation.
        return sum(
            not token.group().isspace()
            for token in _tokens(text)
        )

    return {
        "changed_segments": len({edit["segment_id"] for edit in edits}),
        "edit_spans": len(edits),
        "whitespace_only_spans": sum(
            edit["whitespace_only"] for edit in edits
        ),
        "inserted_tokens": sum(
            count_tokens(edit["replacement"])
            for edit in edits if edit["operation"] == "insert"
        ),
        "deleted_tokens": sum(
            count_tokens(edit["original"])
            for edit in edits if edit["operation"] == "delete"
        ),
        "replaced_source_tokens": sum(
            count_tokens(edit["original"])
            for edit in edits if edit["operation"] == "replace"
        ),
        "replacement_tokens": sum(
            count_tokens(edit["replacement"])
            for edit in edits if edit["operation"] == "replace"
        ),
        "removed_characters": sum(len(edit["original"]) for edit in edits),
        "inserted_characters": sum(len(edit["replacement"]) for edit in edits),
    }