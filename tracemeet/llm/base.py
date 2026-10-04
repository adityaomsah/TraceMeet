from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base error for an unsuccessful model call."""


class LLMTemporaryError(LLMError):
    """A transient failure that may succeed after a bounded retry."""


class LLMQuotaError(LLMTemporaryError):
    """Rate limit or quota exhausted; waiting may or may not help."""


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate_structured(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.0,
    ) -> T:
        """Return an instance of schema or raise LLMError."""
        raise NotImplementedError

    def close(self) -> None:
        """Release resources if the provider owns any."""