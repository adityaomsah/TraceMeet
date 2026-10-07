import html
import re

from tracemeet.schemas import MeetingRecord, Transcript, UNSPECIFIED


def escape_markdown(value: str) -> str:
    """Render meeting text as text rather than Markdown/HTML instructions."""
    value = html.escape(value, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|~-])", r"\\\1", value)


def format_time(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def to_markdown(record: MeetingRecord, transcript: Transcript) -> str:
    segments = transcript.by_id()
    lines = [
        "# TraceMeet meeting record",
        "",
        "> Citation matching checks source text, not whether every claim "
        "is semantically supported. The overall summary is uncited.",
        "",
        "## Summary",
        "",
        escape_markdown(record.summary),
        "",
        "## Minutes",
        "",
    ]

    def add_evidence(items, label="Evidence"):
        if not items:
            return

        lines.extend([f"**{label}**", ""])
        for item in items:
            segment = segments[item.segment_id]
            location = (
                f"{escape_markdown(item.segment_id)} "
                f"({format_time(segment.start)}–{format_time(segment.end)})"
            )
            lines.extend([f"Source: {location}", ""])
            for line in escape_markdown(item.quote).splitlines():
                lines.append(f"> {line}")
            lines.append("")

    if not record.minutes:
        lines.extend(["None recorded.", ""])

    for topic in record.minutes:
        lines.extend([
            f"### {escape_markdown(topic.title)}",
            "",
            escape_markdown(topic.summary),
            "",
        ])
        add_evidence(topic.evidence)

    lines.extend(["## Decisions", ""])
    if not record.decisions:
        lines.extend(["None recorded.", ""])

    for index, decision in enumerate(record.decisions, start=1):
        lines.extend([
            f"### Decision {index} — {decision.status.value}",
            "",
            escape_markdown(decision.text),
            "",
        ])
        add_evidence(decision.evidence)

    lines.extend(["## Action items", ""])
    if not record.action_items:
        lines.extend(["None recorded.", ""])

    for index, task in enumerate(record.action_items, start=1):
        lines.extend([
            f"### task_{index:03d} — {task.status.value}",
            "",
            escape_markdown(task.description),
            "",
            f"- Owner: {escape_markdown(task.owner or UNSPECIFIED)}",
            f"- Deadline: {escape_markdown(task.deadline or UNSPECIFIED)}",
            "",
        ])
        add_evidence(task.evidence, "Task evidence")
        add_evidence(task.owner_evidence, "Owner evidence")
        add_evidence(task.deadline_evidence, "Deadline evidence")

    lines.extend(["## Open questions", ""])
    if not record.open_questions:
        lines.extend(["None recorded.", ""])

    for index, question in enumerate(record.open_questions, start=1):
        lines.extend([
            f"### Question {index}",
            "",
            escape_markdown(question.text),
            "",
        ])
        add_evidence(question.evidence)

    return "\n".join(lines).rstrip() + "\n"