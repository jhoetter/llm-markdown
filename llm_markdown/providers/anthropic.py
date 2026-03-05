import asyncio
import base64
import importlib
from typing import AsyncIterator, Iterator, Union

from .base import LLMProvider


def _parse_data_uri(data_uri: str) -> tuple[str, str]:
    if not data_uri.startswith("data:") or ";base64," not in data_uri:
        raise ValueError("Anthropic image inputs must be provided as data URIs")
    prefix, data = data_uri.split(";base64,", 1)
    media_type = prefix.replace("data:", "", 1)
    return media_type, data


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider.

    Supports sync/async completion, streaming, image input, and structured output
    via a required tool-call schema.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-latest",
        max_tokens: int = 4096,
    ):
        try:
            anthropic = importlib.import_module("anthropic")
        except ImportError as exc:
            raise ImportError(
                "AnthropicProvider requires the 'anthropic' package. "
                "Install with: pip install llm-markdown[anthropic]"
            ) from exc

        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._last_usage = None
        self.client = anthropic.Anthropic(api_key=api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=api_key)

    @staticmethod
    def _convert_content(content):
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        if not isinstance(content, list):
            return [{"type": "text", "text": str(content)}]

        parts = []
        for part in content:
            if part.get("type") == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                image_url = part.get("image_url", {}).get("url", "")
                media_type, b64 = _parse_data_uri(image_url)
                # Validate base64 early for clearer errors.
                base64.b64decode(b64)
                parts.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    }
                )
        return parts or [{"type": "text", "text": ""}]

    @classmethod
    def _to_anthropic_messages(cls, messages: list[dict]):
        system_parts = []
        converted = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "system":
                system_parts.append(content if isinstance(content, str) else str(content))
                continue
            if role not in ("user", "assistant"):
                role = "user"
            converted.append(
                {
                    "role": role,
                    "content": cls._convert_content(content),
                }
            )
        return "\n\n".join(system_parts).strip(), converted

    def _set_usage(self, usage):
        if not usage:
            self._last_usage = None
            return
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is None and output_tokens is None:
            self._last_usage = None
            return
        self._last_usage = {
            "prompt_tokens": input_tokens or 0,
            "completion_tokens": output_tokens or 0,
            "total_tokens": (input_tokens or 0) + (output_tokens or 0),
        }

    def complete(
        self, messages: list[dict], **kwargs
    ) -> Union[str, Iterator[str]]:
        stream = kwargs.get("stream", False)
        system, anthropic_messages = self._to_anthropic_messages(messages)

        if stream:
            stream_resp = self.client.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system or None,
                messages=anthropic_messages,
            )

            def _gen():
                with stream_resp as s:
                    for text in s.text_stream:
                        yield text
                    final_message = s.get_final_message()
                    self._set_usage(getattr(final_message, "usage", None))

            return _gen()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or None,
            messages=anthropic_messages,
        )
        self._set_usage(getattr(response, "usage", None))
        text_parts = [
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ]
        return "".join(text_parts)

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

        system, anthropic_messages = self._to_anthropic_messages(messages)
        response = await self.async_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or None,
            messages=anthropic_messages,
        )
        self._set_usage(getattr(response, "usage", None))
        text_parts = [
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ]
        return "".join(text_parts)

    def complete_structured(self, messages: list[dict], schema: dict) -> dict:
        system, anthropic_messages = self._to_anthropic_messages(messages)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system or None,
            messages=anthropic_messages,
            tools=[
                {
                    "name": "structured_response",
                    "description": "Return a structured response matching the schema.",
                    "input_schema": schema,
                }
            ],
            tool_choice={"type": "tool", "name": "structured_response"},
        )
        self._set_usage(getattr(response, "usage", None))
        for block in response.content:
            if getattr(block, "type", "") == "tool_use":
                return getattr(block, "input", {})
        raise ValueError("Anthropic structured output did not return a tool_use payload.")

    async def complete_structured_async(self, messages: list[dict], schema: dict) -> dict:
        # AsyncAnthropic tool-use endpoint behavior matches sync path.
        return await asyncio.to_thread(self.complete_structured, messages, schema)
