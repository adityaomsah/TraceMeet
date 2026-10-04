from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import av
import math

MIN_DURATION_S = 1.0


class AudioValidationError(Exception):
    """Raised with a user-readable message when a file cannot be processed."""


@dataclass(frozen=True)
class MediaInfo:
    duration_s: Optional[float]   # None if the container does not report it
    has_video: bool
    sample_rate: Optional[int]


def _duration_seconds(container, stream) -> Optional[float]:
    """Prefer the audio track's own duration; a video can outlast its audio."""
    candidates = []
    if stream.duration is not None and stream.time_base is not None:
        candidates.append(float(stream.duration * stream.time_base))
    if container.duration is not None:
        candidates.append(container.duration / av.time_base)
    for duration in candidates:
        if math.isfinite(duration) and duration >= 0:
            return duration
    return None


def validate_media(path: str | Path) -> MediaInfo:
    """Structural checks only. Does not decide whether anyone is speaking."""
    path = Path(path)

    if not path.is_file():
        raise AudioValidationError(f"File not found: {path.name}")
    if path.stat().st_size == 0:
        raise AudioValidationError("The file is empty.")

    try:
        container = av.open(str(path))
    except Exception as exc:
        raise AudioValidationError(
            "This file could not be opened as audio or video. "
            "It may be corrupt or in an unsupported format."
        ) from exc

    with container:
        if len(container.streams.audio) == 0:
            raise AudioValidationError("This file has no audio track.")

        stream = container.streams.audio[0]
        duration = _duration_seconds(container, stream)
        if duration is not None and duration < MIN_DURATION_S:
                        raise AudioValidationError(
                f"The recording is too short ({duration:.1f}s). "
                f"Minimum supported duration is {MIN_DURATION_S:.0f} second."
            )

        return MediaInfo(
            duration_s=duration,
            has_video=len(container.streams.video) > 0,
            sample_rate=getattr(stream, "rate", None),
        )