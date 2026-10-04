import wave

import pytest

from tracemeet.audio.validate import AudioValidationError, validate_media


def make_wav(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def test_silent_wav_is_structurally_valid(tmp_path):
    f = tmp_path / "silence.wav"
    make_wav(f, 2.0)
    info = validate_media(f)
    assert info.has_video is False
    assert 1.9 < info.duration_s < 2.1


def test_empty_file_is_rejected(tmp_path):
    f = tmp_path / "empty.wav"
    f.write_bytes(b"")
    with pytest.raises(AudioValidationError, match="empty"):
        validate_media(f)


def test_garbage_bytes_are_rejected(tmp_path):
    f = tmp_path / "broken.mp3"
    f.write_bytes(b"this is definitely not audio" * 100)
    with pytest.raises(AudioValidationError, match="could not be opened"):
        validate_media(f)


def test_too_short_recording_is_rejected(tmp_path):
    f = tmp_path / "short.wav"
    make_wav(f, 0.2)
    with pytest.raises(AudioValidationError, match="too short"):
        validate_media(f)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(AudioValidationError, match="not found"):
        validate_media(tmp_path / "nope.wav")