import argparse
import sys
import wave
from pathlib import Path

import av
import numpy as np

RATE = 16_000


def extract(src: Path, dst: Path, start: float, duration: float, rate: int = RATE) -> float:
    """Write [start, start+duration] of src as mono 16-bit WAV. Returns seconds written."""
    if start < 0 or duration <= 0:
        raise ValueError("start must be >= 0 and duration must be positive.")

    if not src.is_file():
        raise ValueError(f"File not found: {src}")

    resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
    first, last = int(start * rate), int((start + duration) * rate)
    pieces: list[np.ndarray] = []
    total = 0

    try:
        with av.open(str(src)) as container:
            if not container.streams.audio:
                raise ValueError("The file has no audio track.")
            for frame in container.decode(audio=0):
                frames = resampler.resample(frame)
                if frames is None:
                    continue
                if not isinstance(frames, list):
                    frames = [frames]
                for out in frames:
                    samples = out.to_ndarray().reshape(-1)
                    pieces.append(samples)
                    total += samples.size
                if total >= last:
                    break
    except av.error.FFmpegError as exc:
        raise ValueError(f"Could not read the file ({type(exc).__name__}).") from exc

    audio = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int16)
    clip = audio[first:last]
    if clip.size == 0:
        raise ValueError("The selected window contains no audio.")

    dst.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dst), "wb") as out_file:
        out_file.setnchannels(1)
        out_file.setsampwidth(2)
        out_file.setframerate(rate)
        out_file.writeframes(clip.astype("<i2", copy=False).tobytes())
    return clip.size / rate


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut a short, shareable WAV clip.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, default=0.0, help="Start in seconds")
    parser.add_argument("--duration", type=float, default=90.0, help="Length in seconds")
    args = parser.parse_args()
    try:
        seconds = extract(args.source, args.output, args.start, args.duration)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output} ({seconds:.1f}s, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())