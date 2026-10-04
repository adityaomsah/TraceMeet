import math
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


class ConfigError(Exception):
    """Configuration is missing or invalid."""


def _required_string(section: dict, field: str, label: str) -> str:
    value = section.get(field)

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label}.{field} must be a non-empty string.")

    return value.strip()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    load_dotenv(ROOT / ".env", override=False)
    path = Path(path)

    try:
        with path.open(encoding="utf-8") as file:
            cfg = yaml.safe_load(file)
    except OSError as exc:
        raise ConfigError(f"Could not read config file: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in config file: {path}") from exc

    if not isinstance(cfg, dict):
        raise ConfigError("Configuration must be a mapping.")

    for section in ("stt", "llm"):
        if not isinstance(cfg.get(section), dict):
            raise ConfigError(f"'{section}' must be a mapping.")

    for field in ("model", "device", "compute_type"):
        cfg["stt"][field] = _required_string(cfg["stt"], field, "stt")

    if cfg["stt"]["device"] not in {"cpu", "cuda", "auto"}:
        raise ConfigError("stt.device must be cpu, cuda or auto.")

    for field in ("refine_model", "minutes_model"):
        model = _required_string(cfg["llm"], field, "llm")
        model = model.removeprefix("models/")

        if not model:
            raise ConfigError(f"llm.{field} contains an empty model ID.")

        cfg["llm"][field] = model

    if cfg["llm"]["refine_model"] == cfg["llm"]["minutes_model"]:
        raise ConfigError("Use distinct refinement and minutes models.")

    temperature = cfg["llm"].get("temperature", 0.0)

    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or not 0 <= temperature <= 2
    ):
        raise ConfigError("llm.temperature must be a number between 0 and 2.")

    cfg["llm"]["temperature"] = float(temperature)
    return cfg


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "").strip()

    if not key:
        raise ConfigError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )

    return key