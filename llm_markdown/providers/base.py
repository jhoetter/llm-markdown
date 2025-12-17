from abc import ABC, abstractmethod
from typing import Iterator, Union, Awaitable, AsyncIterator


class LLMProvider(ABC):
    @abstractmethod
    def query(
        self, messages: list[dict], stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Send a list of messages to the LLM and return the response.
        Messages should follow the format:
        [
            {"role": "system", "content": "System message here"},
            {"role": "user", "content": "User message here"}
        ]

        Args:
            messages: List of message dictionaries
            stream: If True, return an iterator of response chunks

        Returns:
            Either a complete response string or an iterator of response chunks
        """
        pass

    @abstractmethod
    async def query_async(
        self, messages: list[dict], stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """
        Async version of query method.
        Send a list of messages to the LLM and return the response asynchronously.

        Args:
            messages: List of message dictionaries
            stream: If True, return an async iterator of response chunks

        Returns:
            Either a complete response string or an async iterator of response chunks
        """
        pass

    # Optional structured output support - providers can override these methods
    def supports_structured_output(self) -> bool:
        """
        Return True if this provider supports native structured JSON output.

        Providers that support structured output (e.g., OpenAI with response_format)
        should override this method to return True.
        """
        return False

    def query_structured(self, messages: list[dict], schema: dict) -> dict:
        """
        Query with structured output. Returns parsed JSON dict.

        This method uses provider-native structured output capabilities
        (e.g., OpenAI's response_format with json_schema) to guarantee
        valid JSON output matching the provided schema.

        Args:
            messages: Chat messages
            schema: JSON Schema for the expected output structure

        Returns:
            Parsed JSON response as dict with 'reasoning' and 'answer' keys

        Raises:
            NotImplementedError: If provider does not support structured output
        """
        raise NotImplementedError("This provider does not support structured output")

    async def query_structured_async(self, messages: list[dict], schema: dict) -> dict:
        """
        Async version of query_structured.

        Args:
            messages: Chat messages
            schema: JSON Schema for the expected output structure

        Returns:
            Parsed JSON response as dict with 'reasoning' and 'answer' keys

        Raises:
            NotImplementedError: If provider does not support structured output
        """
        raise NotImplementedError("This provider does not support structured output")
