import pytest
import yaml

from tracemeet.config import ConfigError, get_api_key, load_config


@pytest.fixture(autouse=True)
def isolate_dotenv(monkeypatch):
    monkeypatch.setattr(
        "tracemeet.config.load_dotenv",
        lambda *args, **kwargs: False,
    )


@pytest.fixture
def valid_config():
    return {
        "stt": {
            "model": "small.en",
            "device": "cpu",
            "compute_type": "int8",
        },
        "llm": {
            "refine_model": "gemini-2.5-flash-lite",
            "minutes_model": "gemini-2.5-flash",
            "temperature": 0.0,
        },
    }


def write_config(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_missing_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        get_api_key()


def test_key_is_read_from_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "  fake-test-key  ")

    assert get_api_key() == "fake-test-key"


def test_valid_configuration_loads(tmp_path, valid_config):
    path = write_config(tmp_path, valid_config)
    cfg = load_config(path)

    assert cfg["stt"]["model"] == "small.en"
    assert cfg["llm"]["temperature"] == 0.0


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="Could not read"):
        load_config(tmp_path / "missing.yaml")


def test_malformed_yaml(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("llm: [", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


def test_null_section_is_rejected(tmp_path, valid_config):
    valid_config["llm"] = None
    path = write_config(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_missing_model_is_rejected(tmp_path, valid_config):
    del valid_config["llm"]["refine_model"]
    path = write_config(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="refine_model"):
        load_config(path)


def test_identical_models_are_rejected(tmp_path, valid_config):
    valid_config["llm"]["minutes_model"] = (
        "models/" + valid_config["llm"]["refine_model"]
    )
    path = write_config(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="distinct"):
        load_config(path)


@pytest.mark.parametrize("temperature", [-1, 3, True, "cold", float("nan")])
def test_invalid_temperature(tmp_path, valid_config, temperature):
    valid_config["llm"]["temperature"] = temperature
    path = write_config(tmp_path, valid_config)

    with pytest.raises(ConfigError, match="temperature"):
        load_config(path)