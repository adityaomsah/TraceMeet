from tracemeet.schemas import Decision, DecisionStatus, Evidence, MeetingRecord, Segment, Transcript
from tracemeet.ui.results import (
    changed_segments, diff_markdown, evidence_list, fmt_time,
    guard_rows, list_runs, load_run, segment_start,
)


def make(*texts):
    return Transcript(
        segments=[
            Segment(id=f"seg_{i:04d}", start=i * 5.0, end=i * 5.0 + 4, text=t)
            for i, t in enumerate(texts, start=1)
        ]
    )


def write_run(tmp_path, record_json=None):
    run = tmp_path / "run"
    run.mkdir()
    transcript = make("Priya will send the report by Friday")
    (run / "raw_transcript.json").write_text(transcript.model_dump_json(), encoding="utf-8")
    (run / "refined_transcript.json").write_text(transcript.model_dump_json(), encoding="utf-8")
    if record_json is None:
        record = MeetingRecord(
            summary="A short meeting.",
            minutes=[],
            decisions=[
                Decision(
                    text="Priya sends the report",
                    status=DecisionStatus.DECIDED,
                    evidence=[Evidence(segment_id="seg_0001", quote="Priya will send the report")],
                )
            ],
            action_items=[],
            open_questions=[],
        )
        record_json = record.model_dump_json()
    (run / "citation_checked_meeting_record.json").write_text(record_json, encoding="utf-8")
    return run


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
    rows = changed_segments(make("same", "hiding seasons"), make("same", "hiring seasons"))
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


def test_guard_rows_flattens_corrections():
    report = {"corrections": [
        {"segment_id": "seg_0001", "original": "sick", "replacement": "SIC",
         "status": "needs_review", "flags": ["capitalized_token_changed"], "reason": "x"},
        "junk",
    ]}
    rows = guard_rows(report)
    assert len(rows) == 1
    assert rows[0]["Flags"] == "capitalized_token_changed"
    assert guard_rows({}) == []


def test_load_run_builds_exports_from_the_checked_record(tmp_path):
    data = load_run(write_run(tmp_path))
    assert data.bundle is not None
    assert "meeting_record.md" in data.bundle.files
    assert data.record["decisions"][0]["status"] == "decided"
    assert data.notes == []


def test_load_run_without_a_record_says_so(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "raw_transcript.json").write_text(make("hello there").model_dump_json(), encoding="utf-8")
    data = load_run(run)
    assert data.bundle is None
    assert any("no meeting record" in note for note in data.notes)


def test_invalid_record_is_shown_but_exports_are_unavailable(tmp_path):
    data = load_run(write_run(tmp_path, record_json='{"summary": 1}'))
    assert data.bundle is None
    assert data.record == {"summary": 1}
    assert any("Exports are unavailable" in note for note in data.notes)