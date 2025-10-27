import openai
from .base import LLMProvider
from typing import Union, Iterator, AsyncIterator


class OpenAILegacyProvider(LLMProvider):
    def __init__(
        self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 4096
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(api_key=self.api_key)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key)

    def query(
        self, messages: list[dict], stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Send a chat-style conversation to OpenAI and return the response content.
        Handles both text-only and multimodal messages.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=stream,
        )

        if stream:

            def response_generator():
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            return response_generator()

        return response.choices[0].message.content

    async def query_async(
        self, messages: list[dict], stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """
        Async version that sends a chat-style conversation to OpenAI and return the response content.
        """
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=stream,
        )

        if stream:

            async def async_response_generator():
                async for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            return async_response_generator()

        return response.choices[0].message.content


class OpenAIProvider(LLMProvider):
    """
    Modern OpenAI Provider for GPT-5, o-series, and other models that use max_completion_tokens.
    """

    def __init__(
        self, api_key: str, model: str = "gpt-5", max_completion_tokens: int = 4096
    ):
        self.api_key = api_key
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.client = openai.OpenAI(api_key=self.api_key)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key)

    def query(
        self, messages: list[dict], stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Send a chat-style conversation to OpenAI and return the response content.
        Handles both text-only and multimodal messages.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
            stream=stream,
        )

        if stream:

            def response_generator():
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            return response_generator()

        return response.choices[0].message.content

    async def query_async(
        self, messages: list[dict], stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """
        Async version that sends a chat-style conversation to OpenAI and return the response content.
        """
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
            stream=stream,
        )

        if stream:

            async def async_response_generator():
                async for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content

            return async_response_generator()

        return response.choices[0].message.content
