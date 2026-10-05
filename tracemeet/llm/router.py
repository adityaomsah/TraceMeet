import math
import random
import time
from typing import Callable, Type

from tracemeet.llm.base import (
    LLMProvider,
    LLMQuotaError,
    LLMTemporaryError,
    T,
)


def _retry_hints(exc: Exception) -> tuple[float | None, bool]:
    """Read structured Google error details when the provider retains them."""
    cause = exc.__cause__
    body = getattr(cause, "response_json", None)

    if not isinstance(body, dict):
        return None, False

    error = body.get("error", body)
    if not isinstance(error, dict):
        return None, False

    details = error.get("details", [])
    if not isinstance(details, list):
        return None, False

    delay = None
    daily_quota = False

    for detail in details:
        if not isinstance(detail, dict):
            continue

        if str(detail.get("@type", "")).endswith("RetryInfo"):
            value = detail.get("retryDelay")
            if isinstance(value, str) and value.endswith("s"):
                try:
                    seconds = float(value[:-1])
                    if math.isfinite(seconds) and seconds >= 0:
                        delay = max(delay or 0.0, seconds)
                except ValueError:
                    pass

        violations = detail.get("violations", [])
        if isinstance(violations, list):
            for violation in violations:
                if not isinstance(violation, dict):
                    continue
                quota_id = str(violation.get("quotaId", "")).lower()
                if "perday" in quota_id or "per_day" in quota_id:
                    daily_quota = True

    return delay, daily_quota


def generate_with_retry(
    provider: LLMProvider,
    *,
    model: str,
    system: str,
    prompt: str,
    schema: Type[T],
    temperature: float = 1.0,
    max_attempts: int = 3,
    on_status: Callable[[str], None] | None = None,
) -> T:
    """Make at most max_attempts calls; retry only temporary failures."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return provider.generate_structured(
                model=model,
                system=system,
                prompt=prompt,
                schema=schema,
                temperature=temperature,
            )
        except (LLMTemporaryError, LLMQuotaError) as exc:
            server_delay, daily_quota = _retry_hints(exc)

            if daily_quota:
                if on_status:
                    on_status("Daily quota reached; automatic retries stopped.")
                raise

            if attempt == max_attempts:
                raise

            delay = min(2 ** attempt, 30) + random.uniform(0.0, 0.5)
            if server_delay is not None:
                delay = max(delay, server_delay)

            if delay > 60:
                if on_status:
                    on_status(
                        f"Server requests a {delay:.1f}s wait. "
                        "Automatic retries stopped; try again later."
                    )
                raise

            if on_status:
                on_status(
                    f"Temporary API failure; retry {attempt + 1}/"
                    f"{max_attempts} in {delay:.1f}s."
                )

            time.sleep(delay)

    raise AssertionError("Retry loop ended unexpectedly")