import json
import warnings
import openai
from .base import LLMProvider
from typing import Union, Iterator, AsyncIterator

_MODERN_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _uses_modern_tokens(model: str) -> bool:
    return any(model.startswith(p) for p in _MODERN_MODEL_PREFIXES)


def _extract_usage(response) -> dict | None:
    if hasattr(response, "usage") and response.usage:
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return None


def _extract_chunk_usage(chunk) -> dict | None:
    if chunk and hasattr(chunk, "usage") and chunk.usage:
        return {
            "prompt_tokens": getattr(chunk.usage, "prompt_tokens", None),
            "completion_tokens": getattr(chunk.usage, "completion_tokens", None),
            "total_tokens": getattr(chunk.usage, "total_tokens", None),
        }
    return None


class OpenAIProvider(LLMProvider):
    """Unified OpenAI provider that works with both legacy and modern models.

    Automatically selects the correct token-limit parameter based on the model:
    - Modern models (GPT-5, o1/o3/o4 series) use ``max_completion_tokens``
    - Legacy models (GPT-4o, GPT-4, GPT-3.5, etc.) use ``max_tokens``

    Args:
        api_key: Your OpenAI API key.
        model: The model name (default: "gpt-4o-mini").
        max_tokens: Maximum tokens in the response (default: 4096).
    """

    def __init__(
        self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 4096
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._token_param = (
            "max_completion_tokens" if _uses_modern_tokens(model) else "max_tokens"
        )
        self.client = openai.OpenAI(api_key=self.api_key)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key)

    def _token_kwargs(self) -> dict:
        return {self._token_param: self.max_tokens}

    # -- core completions ------------------------------------------------

    def complete(
        self, messages: list[dict], **kwargs
    ) -> Union[str, Iterator[str]]:
        stream = kwargs.get("stream", False)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            **self._token_kwargs(),
        )

        if stream:
            def _gen():
                last_chunk = None
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                    last_chunk = chunk
                self._last_usage = _extract_chunk_usage(last_chunk)
            return _gen()

        self._last_usage = _extract_usage(response)
        return response.choices[0].message.content

    async def complete_async(
        self, messages: list[dict], **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        stream = kwargs.get("stream", False)
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=stream,
            **self._token_kwargs(),
        )

        if stream:
            async def _gen():
                last_chunk = None
                async for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                    last_chunk = chunk
                self._last_usage = _extract_chunk_usage(last_chunk)
            return _gen()

        self._last_usage = _extract_usage(response)
        return response.choices[0].message.content

    # -- structured output -----------------------------------------------

    def complete_structured(self, messages: list[dict], schema: dict) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": schema,
                },
            },
            **self._token_kwargs(),
        )
        self._last_usage = _extract_usage(response)
        return json.loads(response.choices[0].message.content)

    async def complete_structured_async(
        self, messages: list[dict], schema: dict
    ) -> dict:
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": schema,
                },
            },
            **self._token_kwargs(),
        )
        self._last_usage = _extract_usage(response)
        return json.loads(response.choices[0].message.content)


class OpenAILegacyProvider(OpenAIProvider):
    """Deprecated -- use OpenAIProvider instead.

    This alias exists for backward compatibility. OpenAIProvider now
    auto-detects the correct token parameter for all model families.
    """

    def __init__(
        self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 4096
    ):
        warnings.warn(
            "OpenAILegacyProvider is deprecated. Use OpenAIProvider instead -- "
            "it auto-detects the correct token parameter for all models.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(api_key=api_key, model=model, max_tokens=max_tokens)
