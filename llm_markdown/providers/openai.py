import openai
from .base import LLMProvider
from typing import Union, Iterator, AsyncIterator


class OpenAILegacyProvider(LLMProvider):
    """
    Legacy OpenAI Provider for older model families that use the 'max_tokens' parameter.

    This provider is designed for OpenAI models that were released before the parameter
    naming convention change. It uses the traditional 'max_tokens' parameter to control
    the maximum number of tokens in the generated response.

    Supported Models:
        - GPT-3.5 Turbo (gpt-3.5-turbo)
        - GPT-4 (gpt-4)
        - GPT-4 Turbo (gpt-4-turbo)
        - GPT-4o (gpt-4o)
        - GPT-4o Mini (gpt-4o-mini)
        - And other models that use the legacy parameter naming

    Why Legacy Provider Exists:
        OpenAI changed their API parameter naming convention for newer models.
        Older models use 'max_tokens' while newer models (GPT-5, o-series) use
        'max_completion_tokens'. This provider ensures backward compatibility
        and provides a clean separation between model families.

    Args:
        api_key (str): Your OpenAI API key
        model (str): The model name (default: "gpt-4o-mini")
        max_tokens (int): Maximum tokens in the response (default: 4096)

    Example:
        >>> provider = OpenAILegacyProvider(
        ...     api_key="your-api-key",
        ...     model="gpt-4o-mini",
        ...     max_tokens=2048
        ... )
        >>> response = provider.query([{"role": "user", "content": "Hello!"}])

    Note:
        Use OpenAIProvider (not this legacy version) for GPT-5 and o-series models.
    """

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
            # For streaming, store usage from the last chunk if available
            def response_generator():
                last_chunk = None
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                    last_chunk = chunk
                # Store usage from the last chunk if available
                if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
                    self._last_usage = {
                        'prompt_tokens': last_chunk.usage.prompt_tokens if hasattr(last_chunk.usage, 'prompt_tokens') else None,
                        'completion_tokens': last_chunk.usage.completion_tokens if hasattr(last_chunk.usage, 'completion_tokens') else None,
                        'total_tokens': last_chunk.usage.total_tokens if hasattr(last_chunk.usage, 'total_tokens') else None,
                    }
                else:
                    self._last_usage = None

            return response_generator()

        # Store usage information for non-streaming responses
        if hasattr(response, 'usage') and response.usage:
            self._last_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            }
        else:
            self._last_usage = None

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
            # For streaming, store usage from the last chunk if available
            async def async_response_generator():
                last_chunk = None
                async for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                    last_chunk = chunk
                # Store usage from the last chunk if available
                if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
                    self._last_usage = {
                        'prompt_tokens': last_chunk.usage.prompt_tokens if hasattr(last_chunk.usage, 'prompt_tokens') else None,
                        'completion_tokens': last_chunk.usage.completion_tokens if hasattr(last_chunk.usage, 'completion_tokens') else None,
                        'total_tokens': last_chunk.usage.total_tokens if hasattr(last_chunk.usage, 'total_tokens') else None,
                    }
                else:
                    self._last_usage = None

            return async_response_generator()

        # Store usage information for non-streaming responses
        if hasattr(response, 'usage') and response.usage:
            self._last_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            }
        else:
            self._last_usage = None

        return response.choices[0].message.content


class OpenAIProvider(LLMProvider):
    """
    Modern OpenAI Provider for newer model families that use the 'max_completion_tokens' parameter.

    This provider is designed for OpenAI's latest generation models that use the updated
    parameter naming convention. It uses 'max_completion_tokens' instead of 'max_tokens'
    to provide more precise control over token generation, including both visible output
    and reasoning tokens.

    Supported Models:
        - GPT-5 (gpt-5)
        - o1 series (o1, o1-mini, o1-preview)
        - o3 series (o3, o3-mini)
        - o4 series (o4, o4-mini)
        - Future models that adopt the new parameter convention

    Why This Provider Exists:
        OpenAI introduced 'max_completion_tokens' to provide clearer control over token
        limits in newer models, especially those with advanced reasoning capabilities.
        This parameter distinguishes between input tokens and completion tokens more
        explicitly than the legacy 'max_tokens' parameter.

    Key Differences from Legacy Provider:
        - Uses 'max_completion_tokens' instead of 'max_tokens'
        - Optimized for models with enhanced reasoning capabilities
        - Better token accounting for complex reasoning workflows
        - Future-proof for upcoming OpenAI model releases

    Args:
        api_key (str): Your OpenAI API key
        model (str): The model name (default: "gpt-5")
        max_completion_tokens (int): Maximum tokens in the completion (default: 4096)

    Example:
        >>> provider = OpenAIProvider(
        ...     api_key="your-api-key",
        ...     model="gpt-5",
        ...     max_completion_tokens=8192
        ... )
        >>> response = provider.query([{"role": "user", "content": "Explain quantum computing"}])

    Note:
        Use OpenAILegacyProvider for older models (GPT-4o, GPT-4, GPT-3.5, etc.).
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
            # For streaming, store usage from the last chunk if available
            def response_generator():
                last_chunk = None
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                    last_chunk = chunk
                # Store usage from the last chunk if available
                if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
                    self._last_usage = {
                        'prompt_tokens': last_chunk.usage.prompt_tokens if hasattr(last_chunk.usage, 'prompt_tokens') else None,
                        'completion_tokens': last_chunk.usage.completion_tokens if hasattr(last_chunk.usage, 'completion_tokens') else None,
                        'total_tokens': last_chunk.usage.total_tokens if hasattr(last_chunk.usage, 'total_tokens') else None,
                    }
                else:
                    self._last_usage = None

            return response_generator()

        # Store usage information for non-streaming responses
        if hasattr(response, 'usage') and response.usage:
            self._last_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            }
        else:
            self._last_usage = None

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
            # For streaming, store usage from the last chunk if available
            async def async_response_generator():
                last_chunk = None
                async for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                    last_chunk = chunk
                # Store usage from the last chunk if available
                if last_chunk and hasattr(last_chunk, 'usage') and last_chunk.usage:
                    self._last_usage = {
                        'prompt_tokens': last_chunk.usage.prompt_tokens if hasattr(last_chunk.usage, 'prompt_tokens') else None,
                        'completion_tokens': last_chunk.usage.completion_tokens if hasattr(last_chunk.usage, 'completion_tokens') else None,
                        'total_tokens': last_chunk.usage.total_tokens if hasattr(last_chunk.usage, 'total_tokens') else None,
                    }
                else:
                    self._last_usage = None

            return async_response_generator()

        # Store usage information for non-streaming responses
        if hasattr(response, 'usage') and response.usage:
            self._last_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            }
        else:
            self._last_usage = None

        return response.choices[0].message.content
