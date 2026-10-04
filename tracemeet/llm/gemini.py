from typing import Type

import httpx
from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from tracemeet.llm.base import (
    LLMError,
    LLMProvider,
    LLMQuotaError,
    LLMTemporaryError,
    T,
)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, api_key: str):
        if not api_key.strip():
            raise LLMError("A non-empty Gemini API key is required.")

        try:
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=60_000),
            )
        except Exception as exc:
            raise LLMError(
                f"Could not initialize Gemini ({type(exc).__name__})."
            ) from exc

    def generate_structured(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.0,
    ) -> T:
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )

        except errors.APIError as exc:
            code = getattr(exc, "code", None)

            if code == 429:
                raise LLMQuotaError(
                    "Gemini rate limit or quota reached. "
                    "Check your quota and any suggested retry delay."
                ) from exc

            if code in {500, 502, 503, 504}:
                raise LLMTemporaryError(
                    f"Gemini is temporarily unavailable (HTTP {code})."
                ) from exc

            raise LLMError(
                f"Gemini rejected the request (HTTP {code}). "
                "Check credentials, model access and request settings."
            ) from exc

        except httpx.TransportError as exc:
            raise LLMTemporaryError(
                "Could not reach Gemini or the request timed out."
            ) from exc

        except Exception as exc:
            raise LLMError(
                f"Gemini client failed ({type(exc).__name__})."
            ) from exc

        if (
            not response.candidates
            or response.candidates[0].finish_reason
            != types.FinishReason.STOP
        ):
            raise LLMError(
                "Gemini did not complete the response normally. "
                "The output may have been blocked or truncated."
            )

        parsed = response.parsed

        if isinstance(parsed, schema):
            return parsed

        text = response.text

        if not text:
            raise LLMError("Gemini returned no usable text.")

        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(
                "Gemini output did not match the expected schema."
            ) from exc

    def close(self) -> None:
        self.client.close()