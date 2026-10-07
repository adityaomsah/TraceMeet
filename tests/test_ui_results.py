from tracemeet.schemas import Segment, Transcript
from tracemeet.ui.results import (
    changed_segments, diff_markdown, evidence_list, fmt_time, list_runs, segment_start,
)


def make(*texts):
    return Transcript(
        segments=[
            Segment(id=f"seg_{i:04d}", start=i * 5.0, end=i * 5.0 + 4, text=t)
            for i, t in enumerate(texts, start=1)
        ]
    )


def test_fmt_time():
    assert fmt_time(65) == "01:05"
    assert fmt_time(3661) == "1:01:01"


def test_diff_marks_removed_and_added_words():
    assert diff_markdown("recent hiding seasons", "recent hiring seasons") == (
        "recent ~~hiding~~ **hiring** seasons"
    )


def test_diff_escapes_dollar_signs():
    assert "\\$" in diff_markdown("costs $5", "costs $6")


def test_changed_segments_lists_only_differences():
    raw, refined = make("same", "hiding seasons"), make("same", "hiring seasons")
    rows = changed_segments(raw, refined)
    assert [r[0] for r in rows] == ["seg_0002"]


def test_evidence_list_handles_dict_missing_and_junk():
    assert evidence_list({"evidence": {"segment_id": "a"}}) == [{"segment_id": "a"}]
    assert evidence_list({}) == []
    assert evidence_list({"evidence": ["x", {"segment_id": "b"}]}) == [{"segment_id": "b"}]


def test_segment_start_lookup():
    t = make("a", "b")
    assert segment_start(t, "seg_0002") == 10.0
    assert segment_start(t, "missing") is None
    assert segment_start(None, "seg_0001") is None


def test_list_runs_only_returns_runs_with_a_raw_transcript(tmp_path):
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_a" / "raw_transcript.json").write_text("{}")
    (tmp_path / "run_b").mkdir()
    assert [p.name for p in list_runs(tmp_path)] == ["run_a"]