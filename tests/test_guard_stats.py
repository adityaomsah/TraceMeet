from eval.guard_stats import analyse, is_cosmetic


def c(seg, original, replacement, status="needs_review", flags=("capitalized_token_changed",)):
    return {"segment_id": seg, "original": original, "replacement": replacement,
            "status": status, "flags": list(flags)}


def test_case_and_punctuation_only_changes_are_cosmetic():
    assert is_cosmetic("secure", "Secure")
    assert is_cosmetic("govern", "Govern,")
    assert not is_cosmetic("sick", "SIC")


def test_segments_held_only_for_cosmetic_reasons_are_listed():
    report = {"corrections": [
        c("seg_0001", "secure", "Secure"),
        c("seg_0001", "govern", "Govern,"),
        c("seg_0002", "250", "205", flags=("number_changed",)),
        c("seg_0002", "secure", "Secure"),
    ]}
    result = analyse(report)
    assert result["held_segments"] == 2
    assert result["held_only_for_cosmetic_reasons"] == ["seg_0001"]
    assert result["cosmetic_held_items"] == 3