import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


def _core(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text).strip().lower()


def is_cosmetic(original: str, replacement: str) -> bool:
    """True when the edit changes only capitalisation or punctuation."""
    return _core(original) == _core(replacement)


def analyse(report: dict) -> dict:
    corrections = [c for c in report.get("corrections", []) if isinstance(c, dict)]
    held = [c for c in corrections if c.get("status") == "needs_review"]

    by_segment: dict[str, list[dict]] = {}
    for item in held:
        by_segment.setdefault(item.get("segment_id", "?"), []).append(item)

    cosmetic_only = sorted(
        seg_id
        for seg_id, items in by_segment.items()
        if all(is_cosmetic(i.get("original", ""), i.get("replacement", "")) for i in items)
    )
    return {
        "total_corrections": len(corrections),
        "by_status": dict(Counter(c.get("status", "?") for c in corrections)),
        "held_by_flag": dict(Counter(f for c in held for f in c.get("flags", []))),
        "held_items": len(held),
        "cosmetic_held_items": sum(
            is_cosmetic(i.get("original", ""), i.get("replacement", "")) for i in held
        ),
        "held_segments": len(by_segment),
        "held_only_for_cosmetic_reasons": cosmetic_only,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise a guard_report.json.")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(json.dumps(analyse(report), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())