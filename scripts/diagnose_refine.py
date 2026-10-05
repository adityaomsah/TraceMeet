import argparse
import json
from importlib.metadata import version
from pathlib import Path

from google import genai
from google.genai import errors, types

from tracemeet.config import get_api_key, load_config
from tracemeet.schemas import Transcript
from tracemeet.stages.refine import DEFAULT_PROMPT, RefinementResponse


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect Gemini's detailed refinement request error."
    )
    parser.add_argument("transcript", type=Path)
    args = parser.parse_args()

    cfg = load_config()
    key = get_api_key()

    raw = Transcript.model_validate_json(
        args.transcript.read_text(encoding="utf-8")
    )

    meta_path = args.transcript.parent / "meta.json"
    glossary = ""
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        glossary = metadata.get("terms", "")

    payload = json.dumps(
        {
            "glossary": glossary,
            "segments": [
                {"id": segment.id, "text": segment.text}
                for segment in raw.segments
            ],
        },
        ensure_ascii=False,
    )

    print(f"google-genai version: {version('google-genai')}")
    print(f"Model: {cfg['llm']['refine_model']}")
    print(f"Temperature: {cfg['llm']['temperature']}")
    print(f"Segments: {len(raw.segments)}")
    print("Sending one diagnostic request; this may consume API quota.")

    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(timeout=60_000),
    )

    try:
        response = client.models.generate_content(
            model=cfg["llm"]["refine_model"],
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=DEFAULT_PROMPT.read_text(
                    encoding="utf-8"
                ),
                temperature=cfg["llm"]["temperature"],
                response_mime_type="application/json",
                response_schema=RefinementResponse,
            ),
        )

        print("Request succeeded.")
        candidates = response.candidates or []
        if candidates:
            print(f"Finish reason: {candidates[0].finish_reason}")
        print("No run files were changed.")
        return 0

    except errors.APIError as exc:
        print(f"\nHTTP code: {exc.code}")
        message = getattr(exc, "message", None) or str(exc)
        print(f"Server message: {str(message).replace(key, '[REDACTED]')}")
        return 2

    except Exception as exc:
        print(f"\nException type: {type(exc).__name__}")
        print(f"Details: {str(exc).replace(key, '[REDACTED]')}")
        return 2

    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())