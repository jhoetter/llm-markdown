import asyncio
import base64
import importlib
import time
from typing import AsyncIterator, Iterator, Union

from llm_markdown.agent_stream import (
    AgentContentDelta,
    AgentMessageFinish,
    AgentReasoningDelta,
    AgentStreamEvent,
    AgentToolCallDelta,
    openai_chat_tools_to_anthropic,
)

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
            anthropic = importlib.import_module("anthropic")
        except ImportError as exc:
            raise ImportError(
                "AnthropicProvider requires the 'anthropic' package. "
                "Install with: pip install llm-markdown[anthropic]"
            ) from exc

        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
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

    def _request_options(self, kwargs: dict) -> dict:
        options = dict(kwargs)
        options.pop("stream", None)
        options["max_tokens"] = options.pop("max_tokens", self.max_tokens)
        options["timeout"] = self.retry_config.timeout_seconds
        return options

    def complete(
        self, messages: list[dict], **kwargs
    ) -> Union[str, Iterator[str]]:
        stream = kwargs.get("stream", False)
        system, anthropic_messages = self._to_anthropic_messages(messages)
        request_kwargs = self._request_options(kwargs)

        if stream:
            stream_resp = self._call_with_retries(
                lambda: self.client.messages.stream(
                    model=self.model,
                    system=system or None,
                    messages=anthropic_messages,
                    **request_kwargs,
                )
            )

            def _gen():
                with stream_resp as s:
                    for text in s.text_stream:
                        yield text
                    final_message = s.get_final_message()
                    self._set_usage(getattr(final_message, "usage", None))
                    self._last_response_metadata = {
                        "provider": type(self).__name__,
                        "model": self.model,
                        "response_id": getattr(final_message, "id", None),
                        "usage": self._last_usage,
                    }

            return _gen()

        response = self._call_with_retries(
            lambda: self.client.messages.create(
                model=self.model,
                system=system or None,
                messages=anthropic_messages,
                **request_kwargs,
            )
        )
        self._set_usage(getattr(response, "usage", None))
        self._last_response_metadata = {
            "provider": type(self).__name__,
            "model": self.model,
            "response_id": getattr(response, "id", None),
            "usage": self._last_usage,
        }
        text_parts = [
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ]
        return "".join(text_parts)

    def stream_messages_events(self, messages: list[dict], **kwargs) -> Iterator[AgentStreamEvent]:
        """Stream one Messages API turn as :class:`~llm_markdown.agent_stream.AgentStreamEvent`.

        ``messages`` use the same shape as :meth:`complete` (OpenAI-style roles are
        converted). ``tools`` may be OpenAI chat tool specs (``type: function``)
        or Anthropic-native tool dicts. ``tool_choice`` ``\"auto\"`` (default) is
        mapped to Anthropic's auto tool choice. Extended thinking deltas are
        emitted as :class:`~llm_markdown.agent_stream.AgentReasoningDelta`.
        """
        system, anthropic_messages = self._to_anthropic_messages(messages)
        stream_kw = dict(kwargs)
        stream_kw.pop("stream", None)
        tools = stream_kw.pop("tools", None)
        tool_choice = stream_kw.pop("tool_choice", None)
        model = stream_kw.pop("model", self.model)
        request_base = self._request_options(stream_kw)

        anthropic_tools = None
        if tools:
            if isinstance(tools[0], dict) and tools[0].get("type") == "function":
                anthropic_tools = openai_chat_tools_to_anthropic(tools)
            else:
                anthropic_tools = list(tools)

        if tool_choice in (None, "auto"):
            tc_param: dict | None = {"type": "auto"}
        elif isinstance(tool_choice, dict):
            tc_param = tool_choice
        else:
            tc_param = {"type": "auto"}

        def _create_stream():
            call_kw = {**request_base, **stream_kw}
            call_kw["model"] = model
            call_kw["system"] = system or None
            call_kw["messages"] = anthropic_messages
            if anthropic_tools:
                call_kw["tools"] = anthropic_tools
                call_kw["tool_choice"] = tc_param
            return self.client.messages.stream(**call_kw)

        def _gen() -> Iterator[AgentStreamEvent]:
            started = time.perf_counter()
            active_tool_idx: int | None = None
            saw_message_stop = False
            with self._call_with_retries(_create_stream) as stream:
                for chunk in stream:
                    t = getattr(chunk, "type", None)
                    if t == "text":
                        yield AgentContentDelta(text=chunk.text)
                    elif t == "thinking":
                        yield AgentReasoningDelta(text=chunk.thinking)
                    elif t == "content_block_start":
                        block = getattr(chunk, "content_block", None)
                        idx = int(getattr(chunk, "index", 0))
                        btype = getattr(block, "type", None) if block is not None else None
                        if btype == "tool_use":
                            active_tool_idx = idx
                            bid = getattr(block, "id", None)
                            bname = getattr(block, "name", None)
                            yield AgentToolCallDelta(
                                index=idx,
                                tool_call_id=str(bid) if bid is not None else None,
                                name=str(bname) if bname is not None else None,
                                arguments=None,
                            )
                    elif t == "input_json":
                        pj = getattr(chunk, "partial_json", None) or ""
                        idx = active_tool_idx if active_tool_idx is not None else 0
                        if pj:
                            yield AgentToolCallDelta(
                                index=idx,
                                tool_call_id=None,
                                name=None,
                                arguments=str(pj),
                            )
                    elif t == "content_block_stop":
                        active_tool_idx = None
                    elif t == "message_stop":
                        msg = chunk.message
                        usage_obj = getattr(msg, "usage", None)
                        usage_dict = None
                        if usage_obj is not None:
                            inp = getattr(usage_obj, "input_tokens", None)
                            out_tok = getattr(usage_obj, "output_tokens", None)
                            usage_dict = {
                                "prompt_tokens": inp,
                                "completion_tokens": out_tok,
                                "total_tokens": (
                                    (inp or 0) + (out_tok or 0)
                                    if inp is not None and out_tok is not None
                                    else None
                                ),
                            }
                        sr = getattr(msg, "stop_reason", None)
                        finish = "tool_calls" if sr == "tool_use" else (sr or "stop")
                        yield AgentMessageFinish(
                            finish_reason=finish,
                            usage=usage_dict,
                        )
                        saw_message_stop = True

                final = stream.get_final_message()
                self._set_usage(getattr(final, "usage", None))
                self._last_response_metadata = {
                    "provider": type(self).__name__,
                    "model": model,
                    "response_id": getattr(final, "id", None),
                    "usage": self._last_usage,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "retry_attempts": self._last_retry_attempts,
                }

            if not saw_message_stop:
                yield AgentMessageFinish(
                    finish_reason="stop",
                    usage=self._last_usage,
                )

        return _gen()

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
        request_kwargs = self._request_options(kwargs)
        response = await self._acall_with_retries(
            lambda: self.async_client.messages.create(
                model=self.model,
                system=system or None,
                messages=anthropic_messages,
                **request_kwargs,
            )
        )
        self._set_usage(getattr(response, "usage", None))
        self._last_response_metadata = {
            "provider": type(self).__name__,
            "model": self.model,
            "response_id": getattr(response, "id", None),
            "usage": self._last_usage,
        }
        text_parts = [
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ]
        return "".join(text_parts)

    def complete_structured(self, messages: list[dict], schema: dict, **kwargs) -> dict:
        system, anthropic_messages = self._to_anthropic_messages(messages)
        request_kwargs = self._request_options(kwargs)
        response = self._call_with_retries(
            lambda: self.client.messages.create(
                model=self.model,
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
                **request_kwargs,
            )
        )
        self._set_usage(getattr(response, "usage", None))
        self._last_response_metadata = {
            "provider": type(self).__name__,
            "model": self.model,
            "response_id": getattr(response, "id", None),
            "usage": self._last_usage,
        }
        for block in response.content:
            if getattr(block, "type", "") == "tool_use":
                return getattr(block, "input", {})
        raise ValueError("Anthropic structured output did not return a tool_use payload.")

    async def complete_structured_async(
        self, messages: list[dict], schema: dict, **kwargs
    ) -> dict:
        # AsyncAnthropic tool-use endpoint behavior matches sync path.
        return await asyncio.to_thread(self.complete_structured, messages, schema, **kwargs)
