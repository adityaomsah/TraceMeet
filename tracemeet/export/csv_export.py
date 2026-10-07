import csv
import io
import json

from tracemeet.schemas import MeetingRecord, UNSPECIFIED

CSV_COLUMNS = [
    "task_id",
    "description",
    "status",
    "owner",
    "deadline",
    "task_evidence",
    "owner_evidence",
    "deadline_evidence",
]


def spreadsheet_safe(value: str) -> str:
    """Prevent meeting text from being interpreted as a spreadsheet formula."""
    if value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    if value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def evidence_json(items) -> str:
    return json.dumps(
        [item.model_dump(mode="json") for item in items],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def to_task_csv(record: MeetingRecord) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=CSV_COLUMNS,
        lineterminator="\r\n",
    )
    writer.writeheader()

    for index, task in enumerate(record.action_items, start=1):
        row = {
            "task_id": f"task_{index:03d}",
            "description": task.description,
            "status": task.status.value,
            "owner": task.owner or UNSPECIFIED,
            "deadline": task.deadline or UNSPECIFIED,
            "task_evidence": evidence_json(task.evidence),
            "owner_evidence": evidence_json(task.owner_evidence),
            "deadline_evidence": evidence_json(task.deadline_evidence),
        }
        writer.writerow({
            key: spreadsheet_safe(value)
            for key, value in row.items()
        })

    return buffer.getvalue()