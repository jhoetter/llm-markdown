from abc import ABC, abstractmethod
from typing import Iterator, Union, AsyncIterator


class LLMProvider(ABC):
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

    def complete_structured(self, messages: list[dict], schema: dict) -> dict:
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
        self, messages: list[dict], schema: dict
    ) -> dict:
        """Async version of complete_structured."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support structured output. "
            "Override complete_structured_async() to enable it."
        )
