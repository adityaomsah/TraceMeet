import sys
import time
from pathlib import Path

from tracemeet.stt.local_whisper import LocalWhisper, TranscriptionError


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.transcribe_dev <audio_or_video_file> ["names, terms, ..."]')
        sys.exit(1)

    path = Path(sys.argv[1])
    terms = sys.argv[2] if len(sys.argv) > 2 else None
    if terms:
        print(f"Using terms hint: {terms}")

    t0 = time.perf_counter()
    try:
        stt = LocalWhisper()
        transcript = stt.transcribe(
            path,
            on_progress=lambda p: print(f"\r  audio position: {p:4.0%}", end="", flush=True),
            terms=terms,
        )
    except TranscriptionError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(2)
    elapsed = time.perf_counter() - t0

    print()
    for s in transcript.segments:
        lp = f"{s.avg_logprob:.2f}" if s.avg_logprob is not None else "n/a"
        print(f"[{s.start:7.2f} - {s.end:7.2f}] {s.id} (lp {lp}): {s.text}")

    print(
        f"\n{len(transcript.segments)} segments | audio {transcript.duration_s or 0:.1f}s | "
        f"total time {elapsed:.1f}s (includes model loading; first run also downloads it)"
    )

    out_file = Path("runs") / "dev" / ("raw_transcript_terms.json" if terms else "raw_transcript.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()