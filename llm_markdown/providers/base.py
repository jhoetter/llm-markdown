from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Iterator, Union

from llm_markdown.providers.failure_info import ProviderFailure


class ProviderError(RuntimeError):
    """Normalized provider error surfaced by all built-in providers."""

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        original_error: Exception | None = None,
        retryable: bool = False,
        failure: Union[ProviderFailure, None] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error
        self.retryable = retryable
        self.failure = failure


@dataclass(slots=True)
class RetryConfig:
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5


class LLMProvider(ABC):
    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ):
        self.retry_config = RetryConfig(
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        self._last_usage: dict[str, Any] | None = None
        self._last_response_metadata: dict[str, Any] | None = None
        self._last_retry_attempts: int = 0

    @staticmethod
    def _error_message(exc: Exception) -> str:
        return str(exc) or exc.__class__.__name__

    @staticmethod
    def _status_code(exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exc, "response", None)
        if response is not None:
            code = getattr(response, "status_code", None)
            if isinstance(code, int):
                return code
        return None

    def _is_retryable_error(self, exc: Exception) -> bool:
        code = self._status_code(exc)
        if code in {408, 409, 429}:
            return True
        if isinstance(code, int) and 500 <= code <= 599:
            return True
        msg = self._error_message(exc).lower()
        retry_tokens = (
            "rate limit",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "connection aborted",
            "server error",
            "overloaded",
        )
        return any(token in msg for token in retry_tokens)

    def _normalize_error(self, exc: Exception, *, retryable: bool) -> ProviderError:
        from llm_markdown.providers.failure_info import infer_provider_failure

        provider_name = type(self).__name__
        message = f"{provider_name} request failed: {self._error_message(exc)}"
        failure = infer_provider_failure(exc)
        return ProviderError(
            provider=provider_name,
            message=message,
            original_error=exc,
            retryable=retryable,
            failure=failure,
        )

    def _call_with_retries(self, func: Callable[[], Any]) -> Any:
        attempts = self.retry_config.max_retries + 1
        for attempt in range(attempts):
            try:
                self._last_retry_attempts = attempt
                return func()
            except Exception as exc:
                retryable = self._is_retryable_error(exc)
                if attempt >= attempts - 1 or not retryable:
                    self._last_retry_attempts = attempt
                    raise self._normalize_error(exc, retryable=retryable) from exc
                delay = self.retry_config.retry_backoff_seconds * (2**attempt)
                time.sleep(delay)

    async def _acall_with_retries(self, func: Callable[[], Any]) -> Any:
        attempts = self.retry_config.max_retries + 1
        for attempt in range(attempts):
            try:
                self._last_retry_attempts = attempt
                return await func()
            except Exception as exc:
                retryable = self._is_retryable_error(exc)
                if attempt >= attempts - 1 or not retryable:
                    self._last_retry_attempts = attempt
                    raise self._normalize_error(exc, retryable=retryable) from exc
                delay = self.retry_config.retry_backoff_seconds * (2**attempt)
                await asyncio.sleep(delay)

    def last_response_metadata(self) -> dict[str, Any] | None:
        """Return metadata from the most recent provider response."""
        return self._last_response_metadata

    """Base class for LLM providers.

    Subclasses must implement `complete` and `complete_async`.
    Override `complete_structured` / `complete_structured_async` to enable
    native structured JSON output (e.g. OpenAI's response_format).
    """

    @abstractmethod
    def complete(
        self, messages: list[dict], **kwargs
    ) -> Union[str, Iterator[str]]:
        """Send messages to the LLM and return the response.

        Args:
            messages: List of message dicts (role + content).
            **kwargs: Provider-specific options (e.g. stream=True).

        Returns:
            A response string, or an iterator of chunks when streaming.
        """

    @abstractmethod
    async def complete_async(
        self, messages: list[dict], **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Async version of complete."""

    def complete_structured(self, messages: list[dict], schema: dict, **kwargs) -> dict:
        """Query with structured output. Returns parsed JSON dict.

        Providers that support native structured output (e.g. OpenAI's
        response_format with json_schema) should override this method.

        Args:
            messages: Chat messages.
            schema: JSON Schema for the expected output structure.

        Returns:
            Parsed JSON response as dict.

        Raises:
            NotImplementedError: If the provider does not support structured output.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support structured output. "
            "Override complete_structured() to enable it."
        )

    async def complete_structured_async(
        self, messages: list[dict], schema: dict, **kwargs
    ) -> dict:
        """Async version of complete_structured."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support structured output. "
            "Override complete_structured_async() to enable it."
        )

    def generate_image(self, prompt: str, **kwargs) -> dict:
        """Generate an image from text prompt.

        Returns a normalized dict payload containing image data/url and metadata.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support image generation. "
            "Override generate_image() to enable it."
        )

    async def generate_image_async(self, prompt: str, **kwargs) -> dict:
        """Async version of generate_image."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support image generation. "
            "Override generate_image_async() to enable it."
        )
