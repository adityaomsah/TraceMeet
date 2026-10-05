import pytest

from scripts.guard_dev import (
    hash_bytes,
    read_json_source,
    verify_source_hash,
)


def test_accepts_windows_file_byte_hash(tmp_path):
    path = tmp_path / "raw.json"
    original = b'{\r\n  "segments": []\r\n}\r\n'
    path.write_bytes(original)

    data, text = read_json_source(path)

    assert "\r" not in text
    assert hash_bytes(data) != hash_bytes(text.encode("utf-8"))
    assert verify_source_hash(
        hash_bytes(original), data, text
    ) == "file_bytes"


def test_accepts_existing_cli_text_hash(tmp_path):
    path = tmp_path / "raw.json"
    path.write_bytes(b'{\r\n  "segments": []\r\n}\r\n')
    data, text = read_json_source(path)

    expected = hash_bytes(text.encode("utf-8"))

    assert verify_source_hash(
        expected, data, text
    ) == "normalized_text"


def test_rejects_changed_content(tmp_path):
    path = tmp_path / "raw.json"
    path.write_bytes(b'{"text": "deploy"}\r\n')
    data, text = read_json_source(path)

    previous_hash = hash_bytes(b'{"text": "do not deploy"}\r\n')

    with pytest.raises(ValueError, match="matches neither"):
        verify_source_hash(previous_hash, data, text)