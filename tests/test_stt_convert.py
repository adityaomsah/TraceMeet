from types import SimpleNamespace

import pytest

from tracemeet.stt.local_whisper import LocalWhisper, TranscriptionError, convert_segments


def raw(start, end, text, lp=-0.3, nsp=0.01):
    return SimpleNamespace(start=start, end=end, text=text, avg_logprob=lp, no_speech_prob=nsp)


class FakeInfo:
    duration = 10.0
    language = "en"


class FakeModel:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, *args, **kwargs):
        return iter(self._segments), FakeInfo()


class ExplodingModel:
    def transcribe(self, *args, **kwargs):
        def gen():
            yield raw(0, 1, "Hello")
            raise RuntimeError("decoder crashed")
        return gen(), FakeInfo()


@pytest.fixture
def audio_file(tmp_path):
    f = tmp_path / "meeting.wav"
    f.write_bytes(b"not real audio, never decoded by the fake model")
    return f


def test_blank_segments_are_skipped_and_ids_stay_contiguous():
    segs = convert_segments([raw(0, 1, "Hello"), raw(1, 2, "   "), raw(2, 3, "World")])
    assert [s.id for s in segs] == ["seg_0001", "seg_0002"]
    assert [s.text for s in segs] == ["Hello", "World"]


def test_tiny_probability_overshoot_is_tolerated():
    segs = convert_segments([raw(0, 1, "Hi", nsp=1.0000001)])
    assert segs[0].no_speech_prob == 1.0


def test_reversed_timestamps_are_rejected():
    with pytest.raises(ValueError, match="end must be greater"):
        convert_segments([raw(5, 4, "Odd timing")])


def test_grossly_invalid_probability_is_rejected():
    with pytest.raises(ValueError, match="no_speech_prob"):
        convert_segments([raw(0, 1, "Hi", nsp=1.5)])


def test_progress_is_not_reported_for_an_invalid_segment():
    seen = []
    with pytest.raises(ValueError, match="end must be greater"):
        convert_segments([raw(5, 4, "Odd timing")], duration=10.0, on_progress=seen.append)
    assert seen == []


def test_progress_uses_validated_end_times():
    seen = []
    convert_segments([raw(0, 2, "A"), raw(2, 5, "B")], duration=10.0, on_progress=seen.append)
    assert seen == [0.2, 0.5]


def test_failure_during_iteration_becomes_transcription_error(audio_file):
    stt = LocalWhisper(model=ExplodingModel())
    with pytest.raises(TranscriptionError, match="RuntimeError"):
        stt.transcribe(audio_file)


def test_progress_ends_at_one(audio_file):
    seen = []
    stt = LocalWhisper(model=FakeModel([raw(0, 4, "Hello there")]))
    stt.transcribe(audio_file, on_progress=seen.append)
    assert seen[-1] == 1.0


def test_directory_is_not_accepted_as_audio(tmp_path):
    stt = LocalWhisper(model=FakeModel([]))
    with pytest.raises(TranscriptionError, match="not found"):
        stt.transcribe(tmp_path)

def test_terms_are_forwarded_to_the_model(audio_file):
    captured = {}

    class SpyModel:
        def transcribe(self, *args, **kwargs):
            captured.update(kwargs)
            return iter([raw(0, 4, "Hello there")]), FakeInfo()

    LocalWhisper(model=SpyModel()).transcribe(audio_file, terms="Aditya Om Sah")
    assert "Aditya Om Sah" in captured["initial_prompt"]
    assert captured["hotwords"] == "Aditya Om Sah"