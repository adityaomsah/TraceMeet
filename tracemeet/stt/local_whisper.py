import math
from pathlib import Path
from typing import Callable, Iterable, Optional

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from faster_whisper import WhisperModel

from tracemeet.schemas import Segment, Transcript


class TranscriptionError(Exception):
    """Raised with a user-readable message when transcription cannot proceed."""


def _normalize_probability(value: Optional[float]) -> Optional[float]:
    """Tolerate tiny float overshoot only; reject genuinely invalid values."""
    if value is None:
        return None
    if not math.isfinite(value) or not -1e-6 <= value <= 1 + 1e-6:
        raise ValueError(f"Invalid no_speech_prob returned by the model: {value}")
    return min(max(value, 0.0), 1.0)

def convert_segments(
    raw_segments: Iterable,
    duration: Optional[float] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> list[Segment]:
    """Convert faster-whisper segments into our Segment format (the boundary)."""
    segments: list[Segment] = []
    for raw in raw_segments:
        text = (raw.text or "").strip()
        if not text:
            continue  # blank output is not an error

        segment = Segment(
            id=f"seg_{len(segments) + 1:04d}",
            start=raw.start,
            end=raw.end,
            text=text,
            avg_logprob=raw.avg_logprob,
            no_speech_prob=_normalize_probability(raw.no_speech_prob),
        )
        segments.append(segment)

        if (
            on_progress is not None
            and duration is not None
            and math.isfinite(duration)
            and duration > 0
        ):
            on_progress(min(segment.end / duration, 1.0))
    return segments


class LocalWhisper:
    def __init__(
        self,
        model_name: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        model=None,  # injectable for tests
    ):
        self.model_name = model_name
        if model is not None:
            self.model = model
            return
        try:
            self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as exc:
            raise TranscriptionError(
                "Could not load the transcription model. "
                "Check the first-run download (needs internet) and your device configuration."
            ) from exc

    def transcribe(
        self,
        audio_path: str | Path,
        on_progress: Optional[Callable[[float], None]] = None,
        terms: Optional[str] = None,
    ) -> Transcript:
        path = Path(audio_path)

        if not path.is_file():
            raise TranscriptionError(f"Audio file not found: {path}")
        if path.stat().st_size == 0:
            raise TranscriptionError("The audio file is empty.")

        try:
            raw_segments, info = self.model.transcribe(
                str(path),
                language="en",
                task="transcribe",
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                condition_on_previous_text=False,
                initial_prompt=f"Names and terms: {terms}." if terms else None,
                hotwords=terms or None,
            )
            # raw_segments is lazy: decoding actually happens as we iterate.
            segments = convert_segments(raw_segments, info.duration, on_progress)

            if not segments:
                raise TranscriptionError(
                    "No speech was transcribed. Check that the recording contains audible speech."
                )

            transcript = Transcript(
                segments=segments,
                stt_model=f"faster-whisper/{self.model_name}",
                language=info.language,
                duration_s=info.duration,
            )
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(
                f"Transcription failed ({type(exc).__name__}): {exc}"
            ) from exc

        if on_progress is not None:
            on_progress(1.0)  # trailing silence can leave the display below 100%
        return transcript