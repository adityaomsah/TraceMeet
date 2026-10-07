import json
import math
from copy import deepcopy
from time import perf_counter
from typing import Type

import httpx
import tiktoken
from pydantic import ValidationError

from tracemeet.llm.base import LLMError, LLMProvider, T


class RequestBudgetError(LLMError):
    """The estimated request exceeds our configured token allowance."""


class GeneratedOutputError(LLMError):
    """Generated content failed schema validation; a bounded retry may help."""


def is_generated_schema_failure(status: int, error: dict) -> bool:
    """Recognize specific generated-output failures, not every HTTP 400."""
    if status != 400:
        return False

    message = str(error.get("message", "")).strip().casefold()

    return (
        message.startswith(
            "generated json does not match the expected schema"
        )
        or message.startswith("failed to generate json.")
    )


def strict_schema(schema: dict) -> dict:
    """Adapt a copy for strict structured output."""
    result = deepcopy(schema)

    def visit(node):
        if not isinstance(node, dict):
            return

        node.pop("default", None)

        properties = node.get("properties")
        if isinstance(properties, dict):
            node["additionalProperties"] = False
            node["required"] = list(properties)
            for child in properties.values():
                visit(child)

        for keyword in ("$defs", "definitions"):
            definitions = node.get(keyword)
            if isinstance(definitions, dict):
                for child in definitions.values():
                    visit(child)

        if isinstance(node.get("items"), dict):
            visit(node["items"])

        for keyword in ("anyOf", "oneOf", "allOf", "prefixItems"):
            children = node.get(keyword, [])
            if isinstance(children, list):
                for child in children:
                    visit(child)

    visit(result)
    return result


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_s: float = 120,
        request_budget: int = 7400,
        max_completion_tokens: int = 3072,
    ):
        key = api_key.strip()
        if not key:
            raise LLMError("GROQ_API_KEY is empty.")

        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite.")

        if not 0 < max_completion_tokens < request_budget:
            raise ValueError(
                "Output allowance must be positive and smaller "
                "than the request budget."
            )

        self._key = key
        self.request_budget = request_budget
        self.max_completion_tokens = max_completion_tokens
        self.last_call: dict = {}

        try:
            self._encoding = tiktoken.get_encoding("o200k_harmony")
        except Exception as exc:
            raise LLMError(
                "Could not initialize the tokenizer. Its first use "
                "may require an internet download."
            ) from exc

        self.client = httpx.Client(
            base_url="https://api.groq.com/openai/v1/",
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(timeout_s, connect=15.0),
        )

    def estimate_request(
        self,
        *,
        system: str,
        prompt: str,
        schema: Type[T],
    ) -> dict:
        schema_text = json.dumps(
            strict_schema(schema.model_json_schema()),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        text_tokens = sum(
            len(self._encoding.encode_ordinary(value))
            for value in (system, prompt, schema_text)
        )
        estimated_input = math.ceil(text_tokens * 1.10) + 256
        total_reserved = estimated_input + self.max_completion_tokens

        return {
            "tokenizer": "o200k_harmony",
            "estimated_input_tokens": estimated_input,
            "reserved_completion_tokens": self.max_completion_tokens,
            "estimated_total_tokens": total_reserved,
            "request_budget": self.request_budget,
            "fits": total_reserved <= self.request_budget,
        }

    def generate_structured(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.0,
    ) -> T:
        if model not in {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}:
            raise LLMError(
                "This provider currently supports the tested "
                "GPT-OSS request format only."
            )

        budget = self.estimate_request(
            system=system, prompt=prompt, schema=schema
        )
        self.last_call = {"model": model, "budget": budget}

        if not budget["fits"]:
            raise RequestBudgetError(
                f"Estimated request needs "
                f"{budget['estimated_total_tokens']:,} tokens including "
                f"output reserve; configured budget is "
                f"{self.request_budget:,}. No API request was sent. "
                "Long-meeting processing is required; do not truncate "
                "the transcript."
            )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "reasoning_effort": "low",
            "max_completion_tokens": self.max_completion_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "tracemeet_record",
                    "strict": True,
                    "schema": strict_schema(schema.model_json_schema()),
                },
            },
        }

        started = perf_counter()
        try:
            response = self.client.post("chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            elapsed = perf_counter() - started
            self.last_call.update(
                elapsed_seconds=elapsed, failure="timeout"
            )
            raise LLMError(
                f"Groq timed out after {elapsed:.1f}s. "
                "Earlier successful checkpoints remain saved."
            ) from exc
        except httpx.TransportError as exc:
            self.last_call.update(
                elapsed_seconds=perf_counter() - started,
                failure=type(exc).__name__,
            )
            raise LLMError(
                f"Groq network failure ({type(exc).__name__})."
            ) from exc

        elapsed = perf_counter() - started
        self.last_call.update(
            elapsed_seconds=elapsed,
            http_status=response.status_code,
            request_id=response.headers.get("x-request-id"),
        )

        if response.is_error:
            error = {}
            try:
                body = response.json()
                if isinstance(body, dict):
                    value = body.get("error")
                    if isinstance(value, dict):
                        error = value
            except ValueError:
                pass

            detail = str(
                error.get("message", "No readable error details.")
            ).replace(self._key, "[REDACTED]")[:1000]

            self.last_call["error_message"] = detail

            if is_generated_schema_failure(response.status_code, error):
                self.last_call["failure"] = "generated_schema_failure"
                raise GeneratedOutputError(
                    f"Groq generated output failed schema validation: "
                    f"{detail}"
                )

            wait = response.headers.get("retry-after")
            wait_message = f" Retry-After: {wait}." if wait else ""

            raise LLMError(
                f"Groq HTTP {response.status_code} after {elapsed:.1f}s: "
                f"{detail}{wait_message} No automatic retry."
            )

        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
            finish_reason = choice.get("finish_reason")

            self.last_call.update(
                usage=body.get("usage", {}),
                finish_reason=finish_reason,
            )

            if message.get("refusal"):
                raise LLMError("Groq returned a refusal.")

            if finish_reason != "stop":
                raise LLMError(
                    f"Groq generation was incomplete ({finish_reason}). "
                    "Partial output was not accepted."
                )

            content = message.get("content")
            if not isinstance(content, str):
                raise LLMError("Groq returned no text content.")

            return schema.model_validate_json(content)

        except ValidationError as exc:
            self.last_call["failure"] = "local_schema_failure"
            raise GeneratedOutputError(
                "Groq output failed local schema validation "
                f"({exc.error_count()} errors)."
            ) from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(
                f"Unexpected Groq response ({type(exc).__name__})."
            ) from exc

    def close(self) -> None:
        self.client.close()