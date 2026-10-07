import traceback
from pathlib import Path

from pydantic import ValidationError


def summarize_error(exc: BaseException, limit: int = 400) -> str:
    """One short line for the UI and run_state.json. Never includes input values."""
    if isinstance(exc, ValidationError):
        count = exc.error_count()
        text = f"{count} validation error{'s' if count != 1 else ''} for {exc.title}"
        first = exc.errors()[0] if count else None
        if first:
            where = ".".join(str(part) for part in first.get("loc", ()))
            text += f" (first at {where}: {first.get('msg', '')})"
    else:
        text = f"{type(exc).__name__}: {exc}"
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def write_error_log(path: Path, exc: BaseException) -> None:
    """Full traceback for debugging. It can contain transcript text, so keep it out of Git."""
    path.write_text("".join(traceback.format_exception(exc)), encoding="utf-8")