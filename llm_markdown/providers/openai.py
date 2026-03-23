import json
import time
import warnings
import openai
from .base import LLMProvider
from typing import Union, Iterator, AsyncIterator

from llm_markdown.agent_stream import (
    AgentContentDelta,
    AgentMessageFinish,
    AgentReasoningDelta,
    AgentStreamEvent,
    AgentToolCallDelta,
)

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


def _delta_reasoning_text(delta) -> str | None:
    for key in ("reasoning_content", "reasoning"):
        frag = getattr(delta, key, None)
        if frag is None:
            continue
        return frag if isinstance(frag, str) else str(frag)
    return None


def _normalize_image_response(provider_name: str, model: str, response) -> dict:
    images = []
    for item in getattr(response, "data", []) or []:
        images.append(
            {
                "url": getattr(item, "url", None),
                "b64_json": getattr(item, "b64_json", None),
                "revised_prompt": getattr(item, "revised_prompt", None),
            }
        )
    return {
        "provider": provider_name,
        "model": model,
        "response_id": getattr(response, "id", None),
        "images": images,
    }


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
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 4096,
        base_url: str | None = None,
        default_headers: dict | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ):
        super().__init__(
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self._token_param = (
            "max_completion_tokens" if _uses_modern_tokens(model) else "max_tokens"
        )
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=self.default_headers or None,
        )
        self.async_client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=self.default_headers or None,
        )

    def _token_kwargs(self, options: dict | None = None) -> dict:
        options = options or {}
        max_tokens = options.pop("max_tokens", self.max_tokens)
        token_kwargs = {self._token_param: max_tokens}
        if self._token_param == "max_completion_tokens" and "max_completion_tokens" in options:
            token_kwargs["max_completion_tokens"] = options.pop("max_completion_tokens")
        if self._token_param == "max_tokens" and "max_completion_tokens" in options:
            token_kwargs["max_tokens"] = options.pop("max_completion_tokens")
        return token_kwargs

    def _capture_metadata(self, response, *, started_at: float | None = None):
        self._last_response_metadata = {
            "provider": type(self).__name__,
            "model": self.model,
            "response_id": getattr(response, "id", None),
            "request_id": getattr(response, "id", None),
            "latency_ms": (
                int((time.perf_counter() - started_at) * 1000)
                if started_at is not None
                else None
            ),
            "retry_attempts": self._last_retry_attempts,
            "token_usage": self._last_usage,
        }

    def _request_options(self, kwargs: dict) -> dict:
        options = dict(kwargs)
        options.pop("stream", None)
        token_kwargs = self._token_kwargs(options)
        options["timeout"] = self.retry_config.timeout_seconds
        return {**token_kwargs, **options}

    # -- core completions ------------------------------------------------

    def complete(
        self, messages: list[dict], **kwargs
    ) -> Union[str, Iterator[str]]:
        stream = kwargs.get("stream", False)
        started = time.perf_counter()
        request_kwargs = self._request_options(kwargs)
        response = self._call_with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **request_kwargs,
            )
        )

        if stream:
            def _gen():
                last_chunk = None
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                    last_chunk = chunk
                self._last_usage = _extract_chunk_usage(last_chunk)
                self._last_response_metadata = {
                    "provider": type(self).__name__,
                    "model": self.model,
                    "response_id": getattr(last_chunk, "id", None),
                    "request_id": getattr(last_chunk, "id", None),
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "retry_attempts": self._last_retry_attempts,
                    "token_usage": self._last_usage,
                }
            return _gen()

        self._last_usage = _extract_usage(response)
        self._capture_metadata(response, started_at=started)
        return response.choices[0].message.content

    def stream_chat_completion_events(
        self, messages: list[dict], **kwargs
    ) -> Iterator[AgentStreamEvent]:
        """Stream one chat completion as normalized agent events.

        Yields :class:`~llm_markdown.agent_stream.AgentContentDelta`,
        :class:`~llm_markdown.agent_stream.AgentReasoningDelta`,
        :class:`~llm_markdown.agent_stream.AgentToolCallDelta`, and a final
        :class:`~llm_markdown.agent_stream.AgentMessageFinish`.

        Accepts the same extra kwargs as ``chat.completions.create`` (e.g.
        ``tools``, ``tool_choice``, ``temperature``). ``stream`` is forced
        True. Optional ``model`` overrides the provider's default model.

        Uses ``stream_options={"include_usage": True}`` when supported.
        """
        started = time.perf_counter()
        kw = dict(kwargs)
        kw.pop("stream", None)
        model = kw.pop("model", self.model)
        request_kwargs = self._request_options(kw)

        def _create_stream():
            try:
                return self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    stream_options={"include_usage": True},
                    **request_kwargs,
                )
            except TypeError:
                return self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    **request_kwargs,
                )

        def _gen() -> Iterator[AgentStreamEvent]:
            stream = self._call_with_retries(_create_stream)
            last_chunk: object | None = None
            last_choice_chunk: object | None = None
            last_usage: dict | None = None
            for chunk in stream:
                last_chunk = chunk
                u = _extract_chunk_usage(chunk)
                if u:
                    last_usage = u
                if not chunk.choices:
                    continue
                last_choice_chunk = chunk
                ch0 = chunk.choices[0]
                delta = ch0.delta
                c = getattr(delta, "content", None) or None
                if c:
                    yield AgentContentDelta(text=c)
                rtext = _delta_reasoning_text(delta)
                if rtext:
                    yield AgentReasoningDelta(text=rtext)
                tcd = getattr(delta, "tool_calls", None)
                if tcd:
                    for tc in tcd:
                        idx = int(tc.index)
                        tid = getattr(tc, "id", None) or None
                        fn = getattr(tc, "function", None)
                        nm = None
                        arg_frag = None
                        if fn is not None:
                            nm = getattr(fn, "name", None) or None
                            if not nm:
                                nm = None
                            a = getattr(fn, "arguments", None) or None
                            if a:
                                arg_frag = a if isinstance(a, str) else str(a)
                        yield AgentToolCallDelta(
                            index=idx,
                            tool_call_id=tid,
                            name=nm,
                            arguments=arg_frag,
                        )

            finish_reason: str | None = None
            if last_choice_chunk and last_choice_chunk.choices:
                fr = getattr(last_choice_chunk.choices[0], "finish_reason", None)
                if fr:
                    finish_reason = fr

            self._last_usage = last_usage
            self._last_response_metadata = {
                "provider": type(self).__name__,
                "model": model,
                "response_id": getattr(last_chunk, "id", None) if last_chunk else None,
                "request_id": getattr(last_chunk, "id", None) if last_chunk else None,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "retry_attempts": self._last_retry_attempts,
                "token_usage": self._last_usage,
            }
            yield AgentMessageFinish(finish_reason=finish_reason, usage=last_usage)

        return _gen()

    async def complete_async(
        self, messages: list[dict], **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        stream = kwargs.get("stream", False)
        started = time.perf_counter()
        request_kwargs = self._request_options(kwargs)
        response = await self._acall_with_retries(
            lambda: self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **request_kwargs,
            )
        )

        if stream:
            async def _gen():
                last_chunk = None
                async for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                    last_chunk = chunk
                self._last_usage = _extract_chunk_usage(last_chunk)
                self._last_response_metadata = {
                    "provider": type(self).__name__,
                    "model": self.model,
                    "response_id": getattr(last_chunk, "id", None),
                    "request_id": getattr(last_chunk, "id", None),
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "retry_attempts": self._last_retry_attempts,
                    "token_usage": self._last_usage,
                }
            return _gen()

        self._last_usage = _extract_usage(response)
        self._capture_metadata(response, started_at=started)
        return response.choices[0].message.content

    # -- structured output -----------------------------------------------

    def complete_structured(self, messages: list[dict], schema: dict, **kwargs) -> dict:
        started = time.perf_counter()
        request_kwargs = self._request_options(kwargs)
        response = self._call_with_retries(
            lambda: self.client.chat.completions.create(
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
                **request_kwargs,
            )
        )
        self._last_usage = _extract_usage(response)
        self._capture_metadata(response, started_at=started)
        return json.loads(response.choices[0].message.content)

    async def complete_structured_async(
        self, messages: list[dict], schema: dict, **kwargs
    ) -> dict:
        started = time.perf_counter()
        request_kwargs = self._request_options(kwargs)
        response = await self._acall_with_retries(
            lambda: self.async_client.chat.completions.create(
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
                **request_kwargs,
            )
        )
        self._last_usage = _extract_usage(response)
        self._capture_metadata(response, started_at=started)
        return json.loads(response.choices[0].message.content)

    # -- image generation ------------------------------------------------

    def generate_image(self, prompt: str, **kwargs) -> dict:
        started = time.perf_counter()
        image_model = kwargs.pop("model", self.model)
        request_kwargs = dict(kwargs)
        request_kwargs.setdefault("size", "1024x1024")
        request_kwargs.setdefault("quality", "standard")
        request_kwargs.setdefault("n", 1)
        request_kwargs["timeout"] = self.retry_config.timeout_seconds

        response = self._call_with_retries(
            lambda: self.client.images.generate(
                model=image_model,
                prompt=prompt,
                **request_kwargs,
            )
        )
        normalized = _normalize_image_response(type(self).__name__, image_model, response)
        self._last_usage = None
        self._last_response_metadata = {
            "provider": type(self).__name__,
            "model": image_model,
            "response_id": normalized.get("response_id"),
            "request_id": normalized.get("response_id"),
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "retry_attempts": self._last_retry_attempts,
            "token_usage": None,
            "image_usage": {"count": len(normalized.get("images", []))},
            "image_generation": True,
        }
        return normalized

    async def generate_image_async(self, prompt: str, **kwargs) -> dict:
        started = time.perf_counter()
        image_model = kwargs.pop("model", self.model)
        request_kwargs = dict(kwargs)
        request_kwargs.setdefault("size", "1024x1024")
        request_kwargs.setdefault("quality", "standard")
        request_kwargs.setdefault("n", 1)
        request_kwargs["timeout"] = self.retry_config.timeout_seconds

        response = await self._acall_with_retries(
            lambda: self.async_client.images.generate(
                model=image_model,
                prompt=prompt,
                **request_kwargs,
            )
        )
        normalized = _normalize_image_response(type(self).__name__, image_model, response)
        self._last_usage = None
        self._last_response_metadata = {
            "provider": type(self).__name__,
            "model": image_model,
            "response_id": normalized.get("response_id"),
            "request_id": normalized.get("response_id"),
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "retry_attempts": self._last_retry_attempts,
            "token_usage": None,
            "image_usage": {"count": len(normalized.get("images", []))},
            "image_generation": True,
        }
        return normalized


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
