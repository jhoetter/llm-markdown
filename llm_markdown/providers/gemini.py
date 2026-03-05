import asyncio
import base64
import importlib
import json
from typing import AsyncIterator, Iterator, Union

from .base import LLMProvider


def _parse_data_uri(data_uri: str) -> tuple[str, str]:
    if not data_uri.startswith("data:") or ";base64," not in data_uri:
        raise ValueError("Gemini image inputs must be provided as data URIs")
    prefix, data = data_uri.split(";base64,", 1)
    media_type = prefix.replace("data:", "", 1)
    return media_type, data


class GeminiProvider(LLMProvider):
    """Google Gemini provider.

    Uses the Google GenAI SDK and supports sync/async, streaming,
    image input, and native structured JSON output.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 4096,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ):
        super().__init__(
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
        try:
            self._genai = importlib.import_module("google.genai")
            self._types = importlib.import_module("google.genai.types")
        except ImportError as exc:
            raise ImportError(
                "GeminiProvider requires the 'google-genai' package. "
                "Install with: pip install llm-markdown[gemini]"
            ) from exc

        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.client = self._genai.Client(api_key=api_key)

    @staticmethod
    def _convert_content(content):
        if isinstance(content, str):
            return [{"text": content}]

        if not isinstance(content, list):
            return [{"text": str(content)}]

        parts = []
        for part in content:
            if part.get("type") == "text":
                parts.append({"text": part.get("text", "")})
            elif part.get("type") == "image_url":
                image_url = part.get("image_url", {}).get("url", "")
                media_type, b64 = _parse_data_uri(image_url)
                base64.b64decode(b64)
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": media_type,
                            "data": b64,
                        }
                    }
                )
        return parts or [{"text": ""}]

    @classmethod
    def _to_gemini_contents(cls, messages: list[dict]) -> list[dict]:
        contents = []
        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                role = "user"
            if role not in ("user", "model"):
                role = "model" if role == "assistant" else "user"
            contents.append(
                {
                    "role": role,
                    "parts": cls._convert_content(message.get("content", "")),
                }
            )
        return contents

    def _set_usage(self, response):
        usage = getattr(response, "usage_metadata", None)
        if not usage:
            self._last_usage = None
            return
        prompt_tokens = (
            getattr(usage, "prompt_token_count", None)
            or getattr(usage, "input_token_count", None)
            or 0
        )
        completion_tokens = (
            getattr(usage, "candidates_token_count", None)
            or getattr(usage, "output_token_count", None)
            or 0
        )
        total_tokens = getattr(usage, "total_token_count", None) or (
            prompt_tokens + completion_tokens
        )
        self._last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _config(self, options: dict | None = None, *, response_mime_type=None, response_schema=None):
        options = dict(options or {})
        kwargs = {"max_output_tokens": options.pop("max_tokens", self.max_tokens)}
        if response_mime_type:
            kwargs["response_mime_type"] = response_mime_type
        if response_schema is not None:
            kwargs["response_schema"] = response_schema
        kwargs.update(options)
        return self._types.GenerateContentConfig(**kwargs)

    def _request_options(self, kwargs: dict) -> dict:
        options = dict(kwargs)
        options.pop("stream", None)
        return options

    def complete(
        self, messages: list[dict], **kwargs
    ) -> Union[str, Iterator[str]]:
        stream = kwargs.get("stream", False)
        contents = self._to_gemini_contents(messages)
        options = self._request_options(kwargs)
        config = self._config(options)

        if stream:
            response = self._call_with_retries(
                lambda: self.client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            )

            def _gen():
                chunks = []
                for chunk in response:
                    text = getattr(chunk, "text", None) or ""
                    if text:
                        chunks.append(text)
                        yield text
                # Streaming responses may not expose usage consistently.
                self._last_usage = None
                self._last_response_metadata = {
                    "provider": type(self).__name__,
                    "model": self.model,
                    "response_id": None,
                    "usage": self._last_usage,
                }
                _ = chunks

            return _gen()

        response = self._call_with_retries(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        )
        self._set_usage(response)
        self._last_response_metadata = {
            "provider": type(self).__name__,
            "model": self.model,
            "response_id": getattr(response, "response_id", None),
            "usage": self._last_usage,
        }
        return response.text or ""

    async def complete_async(
        self, messages: list[dict], **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        stream = kwargs.get("stream", False)

        if stream:
            sync_iter = await asyncio.to_thread(
                lambda: self.complete(messages, stream=True)
            )

            async def _gen():
                for chunk in sync_iter:
                    yield chunk
                    await asyncio.sleep(0)

            return _gen()

        return await asyncio.to_thread(self.complete, messages, stream=False)

    def complete_structured(self, messages: list[dict], schema: dict, **kwargs) -> dict:
        contents = self._to_gemini_contents(messages)
        options = self._request_options(kwargs)
        config = self._config(
            options,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = self._call_with_retries(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
        )
        self._set_usage(response)
        self._last_response_metadata = {
            "provider": type(self).__name__,
            "model": self.model,
            "response_id": getattr(response, "response_id", None),
            "usage": self._last_usage,
        }
        if not response.text:
            raise ValueError("Gemini structured output returned empty response text.")
        return json.loads(response.text)

    async def complete_structured_async(
        self, messages: list[dict], schema: dict, **kwargs
    ) -> dict:
        return await asyncio.to_thread(self.complete_structured, messages, schema, **kwargs)
