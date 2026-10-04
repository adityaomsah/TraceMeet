import sys

from pydantic import BaseModel, Field

from tracemeet.config import ConfigError, get_api_key, load_config
from tracemeet.llm.base import (
    LLMError,
    LLMQuotaError,
    LLMTemporaryError,
)
from tracemeet.llm.gemini import GeminiProvider


class ProbeResponse(BaseModel):
    project: str = Field(min_length=1)
    number: int
    owner: str | None


def main() -> int:
    try:
        cfg = load_config()
        provider = GeminiProvider(get_api_key())
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}")
        return 1
    except LLMError as exc:
        print(f"CLIENT ERROR: {exc}")
        return 1

    print("Config loaded and API key found.")
    print("Live checks consume a small amount of API quota.")

    ok = True

    try:
        print("\nFlash-family models visible to your key:")

        try:
            for model_info in provider.client.models.list():
                name = model_info.name or ""

                if "flash" in name.lower():
                    print("  ", name.removeprefix("models/"))

        except Exception as exc:
            print(
                f"  Model listing failed ({type(exc).__name__}). "
                "Continuing with configured-model checks."
            )

        print("\nTesting both configured models:")

        for role in ("refine_model", "minutes_model"):
            model = cfg["llm"][role]

            try:
                result = provider.generate_structured(
                    model=model,
                    system=(
                        "Extract only the information explicitly supplied. "
                        "Use null when the owner is not stated. "
                        "Do not invent an owner."
                    ),
                    prompt=(
                        "The project is TraceMeet. "
                        "The number is 7. "
                        "No owner was stated."
                    ),
                    schema=ProbeResponse,
                    temperature=cfg["llm"]["temperature"],
                )

                if (
                    result.project != "TraceMeet"
                    or result.number != 7
                    or result.owner is not None
                ):
                    raise LLMError(
                        "The response matched the schema, "
                        "but the extraction check failed."
                    )

                print(
                    f"  {role} = {model}: "
                    f"PASS -> {result.model_dump()}"
                )

            except LLMQuotaError as exc:
                ok = False
                print(f"  {role} = {model}: QUOTA -> {exc}")

            except LLMTemporaryError as exc:
                ok = False
                print(f"  {role} = {model}: TEMPORARY ERROR -> {exc}")

            except LLMError as exc:
                ok = False
                print(f"  {role} = {model}: FAILED -> {exc}")

    finally:
        provider.close()

    if ok:
        print("\nPASS: both models completed the structured extraction probe.")
        print("This confirms this small request worked, not meeting accuracy.")
    else:
        print("\nPreflight incomplete. Resolve the reported errors.")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())