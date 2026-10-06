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

    def __init__(self, api_key: str, timeout_s: float = 180.0):
        key = api_key.strip()
        if not key:
            raise LLMError("The Gemini API key is empty.")

        self._api_key = key

        try:
            self.client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=int(timeout_s * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
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
        temperature: float = 1.0,
    ) -> T:
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_json_schema=schema.model_json_schema(),
                ),
            )
        except errors.APIError as exc:
            code = getattr(exc, "code", None)

            if code == 429:
                raise LLMQuotaError(
                    "Gemini rate limit or quota reached. "
                    "If retries fail, check the project's quota."
                ) from exc

            if code in {408, 499, 500, 502, 503, 504}:
                raise LLMTemporaryError(
                    f"Gemini is temporarily unavailable (HTTP {code})."
                ) from exc

            detail = str(
                getattr(exc, "message", None) or "No details provided."
            ).replace(self._api_key, "[REDACTED]")

            raise LLMError(
                f"Gemini rejected the request (HTTP {code}): {detail}"
            ) from exc

        except httpx.TransportError as exc:
            raise LLMTemporaryError(
                "Could not reach Gemini. Check your connection."
            ) from exc

        except Exception as exc:
            raise LLMError(
                f"Gemini request failed ({type(exc).__name__})."
            ) from exc

        candidates = response.candidates or []
        if not candidates:
            raise LLMError(
                "Gemini returned no response candidate. "
                "The request may have been blocked."
            )

        finish_reason = candidates[0].finish_reason
        if finish_reason != types.FinishReason.STOP:
            raise LLMError(
                "Gemini did not finish a complete response "
                f"(finish reason: {finish_reason})."
            )

        text = response.text
        if not text or not text.strip():
            raise LLMError("Gemini returned an empty response.")

        try:
            return schema.model_validate_json(text)
        except ValidationError as exc:
            raise LLMError(
                "Gemini returned JSON that did not match the "
                f"expected schema ({exc.error_count()} validation errors)."
            ) from exc

    def close(self) -> None:
        self.client.close()