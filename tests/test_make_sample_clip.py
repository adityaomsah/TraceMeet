import wave

import pytest

from scripts.make_sample_clip import extract


def make_wav(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * int(rate * seconds))


def test_extract_writes_the_requested_window(tmp_path):
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    make_wav(src, 5.0)
    seconds = extract(src, dst, start=1.0, duration=2.0)
    assert 1.9 < seconds < 2.1
    with wave.open(str(dst)) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000


def test_window_past_the_end_is_rejected(tmp_path):
    src = tmp_path / "in.wav"
    make_wav(src, 3.0)
    with pytest.raises(ValueError, match="no audio"):
        extract(src, tmp_path / "out.wav", start=10.0, duration=2.0)


def test_invalid_window_is_rejected(tmp_path):
    src = tmp_path / "in.wav"
    make_wav(src, 3.0)
    with pytest.raises(ValueError, match="positive"):
        extract(src, tmp_path / "out.wav", start=0.0, duration=0.0)


def test_missing_source_names_the_file(tmp_path):
    with pytest.raises(ValueError, match="File not found"):
        extract(tmp_path / "missing.wav", tmp_path / "out.wav", start=0.0, duration=5.0)