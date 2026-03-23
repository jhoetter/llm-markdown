__version__ = "0.3.6"

import inspect
import json
import base64
import binascii
import mimetypes
import os
import re
import socket
import ipaddress
import requests
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import (
    get_type_hints,
    get_origin,
    get_args,
    List,
    Dict,
    Any,
    Generic,
    TypeVar,
)
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_IMAGE_URL_ALLOWLIST_ENV = "LLM_MARKDOWN_IMAGE_URL_ALLOWLIST"
_IMAGE_BLOCK_PRIVATE_ENV = "LLM_MARKDOWN_IMAGE_BLOCK_PRIVATE_NETWORKS"
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Image type for multimodal prompts
# ---------------------------------------------------------------------------

def _is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def _decode_base64_image(source: str) -> bytes:
    try:
        decoded = base64.b64decode(source, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(
            "Image source must be a URL, local file path, data URI, or base64-encoded string"
        ) from exc
    if not decoded:
        raise ValueError("Image source cannot be empty")
    if len(decoded) > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image payload exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)}MB limit"
        )
    return decoded


def _validate_image_content(content_type: str, payload: bytes):
    if not content_type.startswith("image/"):
        raise ValueError(f"Unsupported image media type: {content_type}")
    if len(payload) > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image payload exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)}MB limit"
        )


def _is_host_allowlisted(hostname: str) -> bool:
    allowlist = os.environ.get(_IMAGE_URL_ALLOWLIST_ENV, "").strip()
    if not allowlist:
        return True
    allowed = [item.strip().lower() for item in allowlist.split(",") if item.strip()]
    host = hostname.lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed)


def _host_resolves_to_private_network(hostname: str) -> bool:
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        # If host cannot be resolved, requests will fail later.
        return False
    for info in infos:
        raw_ip = info[4][0]
        try:
            ip = ipaddress.ip_address(raw_ip)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
        ):
            return True
    return False


def _validate_remote_image_url(source: str):
    parsed = urlparse(source)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Image URL must include a hostname")
    if not _is_host_allowlisted(hostname):
        raise ValueError(
            f"Image URL host '{hostname}' is not in {_IMAGE_URL_ALLOWLIST_ENV}"
        )
    block_private = os.environ.get(_IMAGE_BLOCK_PRIVATE_ENV, "true").lower() != "false"
    if block_private and _host_resolves_to_private_network(hostname):
        raise ValueError(
            f"Image URL host '{hostname}' resolves to a private network address"
        )


def _to_data_uri(source: str) -> str:
    """Convert a URL, file path, or raw base64 string into a data URI."""
    if _is_url(source):
        _validate_remote_image_url(source)
        response = requests.get(source, timeout=30, allow_redirects=False)
        response.raise_for_status()
        length = response.headers.get("content-length")
        if length and int(length) > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"Image payload exceeds {_MAX_IMAGE_BYTES // (1024 * 1024)}MB limit"
            )
        content_type = response.headers.get("content-type", "image/jpeg")
        content_type = content_type.split(";", 1)[0]
        _validate_image_content(content_type, response.content)
        b64 = base64.b64encode(response.content).decode("utf-8")
        return f"data:{content_type};base64,{b64}"

    if os.path.isfile(source):
        with open(source, "rb") as image_file:
            image_bytes = image_file.read()
        guessed, _ = mimetypes.guess_type(source)
        content_type = guessed or "image/jpeg"
        _validate_image_content(content_type, image_bytes)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{content_type};base64,{b64}"

    if source.startswith("data:"):
        if ";base64," not in source:
            raise ValueError("Data URI image source must include base64 payload")
        prefix, encoded = source.split(";base64,", 1)
        content_type = prefix.replace("data:", "", 1) or "image/jpeg"
        decoded = _decode_base64_image(encoded)
        _validate_image_content(content_type, decoded)
        return source

    # Assume raw base64
    _decode_base64_image(source)
    return f"data:image/jpeg;base64,{source}"


class Image:
    """Represents an image input for multimodal prompts.

    Accepts a URL, base64 string, or data URI as ``source``.
    """

    def __init__(self, source: str):
        self.source = source

    def to_content_part(self) -> dict:
        """Convert to an OpenAI-format image content part."""
        data_uri = _to_data_uri(self.source)
        return {"type": "image_url", "image_url": {"url": data_uri}}

    def __repr__(self) -> str:
        preview = self.source[:60] + "..." if len(self.source) > 60 else self.source
        return f"Image({preview!r})"


@dataclass(frozen=True)
class PromptResult(Generic[T]):
    """Output wrapper for prompt calls when metadata is requested."""

    output: T
    metadata: dict[str, Any] | None


class Session:
    """Maintains chat history for multi-turn prompt calls."""

    def __init__(
        self,
        provider,
        *,
        system_prompt: str = "You are a helpful assistant.",
        max_messages: int | None = None,
        max_tokens: int | None = None,
        generation_options: dict | None = None,
        langfuse_metadata: dict | None = None,
        return_metadata: bool = False,
    ):
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.generation_options = generation_options or {}
        self.langfuse_metadata = langfuse_metadata or {}
        self.return_metadata = return_metadata
        self.history: list[dict] = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def reset(self):
        self.history = []
        if self.system_prompt:
            self.history.append({"role": "system", "content": self.system_prompt})

    def prompt(
        self,
        *,
        stream: bool = False,
        stream_mode: str = "text",
        generation_options: dict | None = None,
        langfuse_metadata: dict | None = None,
        return_metadata: bool | None = None,
    ):
        merged_options = dict(self.generation_options)
        merged_options.update(generation_options or {})
        metadata = dict(self.langfuse_metadata)
        metadata.update(langfuse_metadata or {})
        return _PromptDecorator(
            provider=self.provider,
            stream=stream,
            stream_mode=stream_mode,
            langfuse_metadata=metadata,
            generation_options=merged_options,
            return_metadata=self.return_metadata if return_metadata is None else return_metadata,
            session=self,
        )

    def _prepare_messages(self, user_content):
        messages = list(self.history)
        messages.append({"role": "user", "content": user_content})
        return messages

    @staticmethod
    def _content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, dict):
                    parts.append(str(item))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content)

    def _trim_history(self):
        if self.max_messages is not None:
            system = [m for m in self.history if m.get("role") == "system"]
            non_system = [m for m in self.history if m.get("role") != "system"]
            if len(non_system) > self.max_messages:
                non_system = non_system[-self.max_messages :]
            self.history = system[:1] + non_system if system else non_system

        if self.max_tokens is not None and self.max_tokens > 0:
            def _size(msg):
                return len(self._content_to_text(msg.get("content", "")))

            while len(self.history) > 1:
                total = sum(_size(m) for m in self.history)
                if total <= self.max_tokens:
                    break
                # Preserve leading system prompt when available.
                remove_idx = 1 if self.history[0].get("role") == "system" else 0
                self.history.pop(remove_idx)

    def _append_turn(self, user_content, assistant_output):
        self.history.append({"role": "user", "content": user_content})
        self.history.append(
            {
                "role": "assistant",
                "content": _response_to_history_text(assistant_output),
            }
        )
        self._trim_history()


def get_last_response_metadata(provider) -> dict[str, Any] | None:
    """Return metadata from the most recent provider call."""
    if hasattr(provider, "last_response_metadata"):
        return provider.last_response_metadata()
    return getattr(provider, "_last_response_metadata", None)


def _response_to_history_text(value) -> str:
    if isinstance(value, PromptResult):
        return _response_to_history_text(value.output)
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _make_schema_strict(schema: dict) -> dict:
    """Make a JSON schema compatible with OpenAI's strict mode."""
    if not isinstance(schema, dict):
        return schema

    result = schema.copy()

    if result.get("type") == "object":
        result["additionalProperties"] = False
        if "properties" in result:
            result["properties"] = {
                k: _make_schema_strict(v) for k, v in result["properties"].items()
            }
            result["required"] = list(result["properties"].keys())

    if result.get("type") == "array" and "items" in result:
        result["items"] = _make_schema_strict(result["items"])

    if "$defs" in result:
        result["$defs"] = {
            k: _make_schema_strict(v) for k, v in result["$defs"].items()
        }

    for key in ("allOf", "anyOf", "oneOf"):
        if key in result:
            result[key] = [_make_schema_strict(s) for s in result[key]]

    return result


def _needs_structured_output(return_type) -> bool:
    """Return True if the return type requires structured (JSON) output."""
    if return_type is None:
        return False

    origin = get_origin(return_type)
    if origin in (list, List, dict, Dict):
        return True

    try:
        if issubclass(return_type, BaseModel):
            return True
    except TypeError:
        pass

    return False


def _build_schema(return_type) -> dict:
    """Build a JSON schema for the return type (no reasoning wrapper)."""
    origin = get_origin(return_type)

    if origin in (list, List):
        args = get_args(return_type)
        if args:
            inner = args[0]
            try:
                if issubclass(inner, BaseModel):
                    item_schema = _make_schema_strict(inner.model_json_schema())
                    result = {"type": "array", "items": item_schema}
                    if "$defs" in item_schema:
                        result["$defs"] = item_schema["items"].pop("$defs", {}) or item_schema.pop("$defs", {})
                    return result
            except TypeError:
                pass
            type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
            return {"type": "array", "items": {"type": type_map.get(inner, "string")}}
        return {"type": "array", "items": {"type": "string"}}

    if origin in (dict, Dict):
        return {"type": "object", "additionalProperties": False}

    try:
        if issubclass(return_type, BaseModel):
            schema = _make_schema_strict(return_type.model_json_schema())
            return schema
    except TypeError:
        pass

    return {"type": "string"}


def _system_instructions(return_type) -> str:
    """Generate system instructions for structured output mode."""
    try:
        if get_origin(return_type) is None and issubclass(return_type, BaseModel):
            schema = return_type.model_json_schema()
            return (
                "You are a helpful assistant. Respond with valid JSON matching "
                f"this schema: {json.dumps(schema)}"
            )
    except TypeError:
        pass

    return "You are a helpful assistant. Respond with valid JSON."


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def _cast_type(value: str, target_type) -> Any:
    """Cast a string value to the target type."""
    if not value:
        return value

    origin = get_origin(target_type)
    if origin is not None:
        if origin in (list, List):
            try:
                if value.startswith("[") and value.endswith("]"):
                    return json.loads(value)
                return [value]
            except json.JSONDecodeError:
                return [
                    item.strip()
                    for item in value.strip("[]").split(",")
                    if item.strip()
                ]
        return value

    try:
        if target_type == bool:
            return value.lower() in ("true", "t", "yes", "y", "1")
        return target_type(value)
    except (ValueError, TypeError):
        return value


def _parse_response(response, return_type) -> Any:
    """Parse a raw LLM response into the expected return type.

    Handles both structured output (dict from complete_structured) and
    plain text (str from complete) responses.
    """
    if return_type is None:
        if isinstance(response, dict):
            return json.dumps(response)
        return response

    origin = get_origin(return_type)

    # Already the right shape from structured output
    if isinstance(response, dict):
        try:
            if issubclass(return_type, BaseModel):
                return return_type.model_validate(response)
        except TypeError:
            pass
        if origin in (list, List):
            return response if isinstance(response, list) else response
        return response

    if isinstance(response, list):
        if origin in (list, List):
            return response
        try:
            if issubclass(return_type, BaseModel) and len(response) == 1:
                return return_type.model_validate(response[0])
        except TypeError:
            pass
        return response

    # String response -- needs parsing
    if not isinstance(response, str):
        return response

    if origin in (list, List):
        return _cast_type(response, return_type)

    try:
        if issubclass(return_type, BaseModel):
            text = response.strip()
            if not text.startswith("{"):
                json_match = re.search(r"\{.*\}", text, re.DOTALL)
                if json_match:
                    text = json_match.group(0)
                else:
                    raise ValueError("Could not find JSON object in response")
            data = json.loads(text)
            return return_type.model_validate(data)
    except TypeError:
        pass

    return _cast_type(response, return_type)


# ---------------------------------------------------------------------------
# Core decorator
# ---------------------------------------------------------------------------

class _PromptDecorator:
    """Internal decorator class -- use the ``prompt()`` factory instead."""

    def __init__(
        self,
        provider,
        stream: bool = False,
        stream_mode: str = "text",
        langfuse_metadata: dict | None = None,
        generation_options: dict | None = None,
        return_metadata: bool = False,
        session: Session | None = None,
    ):
        self.provider = provider
        self.stream = stream
        self.stream_mode = stream_mode
        self.langfuse_metadata = langfuse_metadata or {}
        self.generation_options = generation_options or {}
        self.return_metadata = return_metadata
        self.session = session
        if self.stream_mode not in {"text", "json_events"}:
            raise ValueError("stream_mode must be 'text' or 'json_events'")

    # -- prompt & message building ---------------------------------------

    @staticmethod
    def _extract_prompt(func, arguments: dict) -> str:
        """Get the prompt from the function's docstring and interpolate arguments."""
        raw = inspect.getdoc(func)
        if not raw:
            raise ValueError(
                f"Function {func.__name__} must have a docstring to use as a prompt."
            )
        return raw.format(**arguments)

    @staticmethod
    def _build_user_message(prompt_text: str, image_args: list[Image]):
        """Build the user message, handling multimodal content when Images are present."""
        if not image_args:
            return prompt_text

        content: list[dict] = [{"type": "text", "text": prompt_text}]
        for img in image_args:
            content.append(img.to_content_part())
        return content

    @staticmethod
    def _collect_images(arguments: dict, type_hints: dict) -> list[Image]:
        """Find all Image-typed arguments."""
        images = []
        for name, hint in type_hints.items():
            if name == "return":
                continue
            if hint is Image:
                val = arguments.get(name)
                if isinstance(val, Image):
                    images.append(val)
            elif get_origin(hint) in (list, List):
                args = get_args(hint)
                if args and args[0] is Image:
                    val = arguments.get(name, [])
                    images.extend(v for v in val if isinstance(v, Image))
        return images

    @staticmethod
    def _split_runtime_options(kwargs: dict) -> tuple[dict, dict]:
        runtime_kwargs = dict(kwargs)
        runtime_options = runtime_kwargs.pop("_llm_options", None) or {}
        if not isinstance(runtime_options, dict):
            raise TypeError("_llm_options must be a dict when provided")
        return runtime_kwargs, runtime_options

    def _merged_generation_options(self, runtime_options: dict) -> dict:
        merged = dict(self.generation_options)
        merged.update(runtime_options)
        return merged

    def _finalize_output(self, result):
        if not self.return_metadata:
            return result
        metadata = get_last_response_metadata(self.provider)
        return PromptResult(output=result, metadata=metadata)

    @staticmethod
    def _try_parse_partial_json(text: str):
        candidate = text.strip()
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def _wrap_stream_with_session(self, stream, user_msg):
        def _gen():
            chunks = []
            for chunk in stream:
                chunks.append(chunk)
                yield chunk
            if self.session:
                self.session._append_turn(user_msg, "".join(chunks))

        return _gen()

    async def _wrap_stream_with_session_async(self, stream, user_msg):
        async def _gen():
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
                yield chunk
            if self.session:
                self.session._append_turn(user_msg, "".join(chunks))

        return _gen()

    def _stream_json_events(self, stream, return_type, user_msg):
        def _gen():
            text = ""
            last_partial = object()
            try:
                for chunk in stream:
                    text += chunk
                    yield {"type": "delta_text", "delta": chunk}
                    partial = self._try_parse_partial_json(text)
                    if partial is not None and partial != last_partial:
                        last_partial = partial
                        yield {"type": "partial_json", "data": partial}

                parsed = _parse_response(text, return_type)
                if self.session:
                    self.session._append_turn(user_msg, parsed)
                done_output = self._finalize_output(parsed)
                yield {"type": "done", "output": done_output}
            except Exception as exc:
                yield {"type": "error", "error": str(exc)}

        return _gen()

    async def _stream_json_events_async(self, stream, return_type, user_msg):
        async def _gen():
            text = ""
            last_partial = object()
            try:
                async for chunk in stream:
                    text += chunk
                    yield {"type": "delta_text", "delta": chunk}
                    partial = self._try_parse_partial_json(text)
                    if partial is not None and partial != last_partial:
                        last_partial = partial
                        yield {"type": "partial_json", "data": partial}

                parsed = _parse_response(text, return_type)
                if self.session:
                    self.session._append_turn(user_msg, parsed)
                done_output = self._finalize_output(parsed)
                yield {"type": "done", "output": done_output}
            except Exception as exc:
                yield {"type": "error", "error": str(exc)}

        return _gen()

    # -- execution paths -------------------------------------------------

    def _execute_structured(self, messages: list[dict], return_type, generation_options: dict):
        """Structured output path via complete_structured."""
        schema = _build_schema(return_type)
        sys_msg = _system_instructions(return_type)
        structured_messages = [
            {"role": "system", "content": sys_msg},
            messages[-1],
        ]
        result = self.provider.complete_structured(
            structured_messages, schema, **generation_options
        )
        logger.debug(f"Structured output: {result}")
        return _parse_response(result, return_type)

    async def _execute_structured_async(
        self, messages: list[dict], return_type, generation_options: dict
    ):
        """Async structured output path via complete_structured_async."""
        schema = _build_schema(return_type)
        sys_msg = _system_instructions(return_type)
        structured_messages = [
            {"role": "system", "content": sys_msg},
            messages[-1],
        ]
        result = await self.provider.complete_structured_async(
            structured_messages, schema, **generation_options
        )
        logger.debug(f"Structured output: {result}")
        return _parse_response(result, return_type)

    def _execute_json_fallback(self, messages: list[dict], return_type, generation_options: dict):
        """Fallback: ask for JSON via system prompt, parse the text response."""
        sys_msg = _system_instructions(return_type)
        fallback_messages = [
            {"role": "system", "content": sys_msg},
            messages[-1],
        ]
        raw = self.provider.complete(
            fallback_messages,
            stream=False,
            **generation_options,
        )
        raw = raw.strip()
        logger.debug(f"JSON fallback response:\n{raw}")
        try:
            parsed = json.loads(raw)
            return _parse_response(parsed, return_type)
        except json.JSONDecodeError:
            return _parse_response(raw, return_type)

    async def _execute_json_fallback_async(
        self, messages: list[dict], return_type, generation_options: dict
    ):
        """Async fallback: ask for JSON via system prompt, parse the text response."""
        sys_msg = _system_instructions(return_type)
        fallback_messages = [
            {"role": "system", "content": sys_msg},
            messages[-1],
        ]
        raw = await self.provider.complete_async(
            fallback_messages,
            stream=False,
            **generation_options,
        )
        raw = raw.strip()
        logger.debug(f"JSON fallback response:\n{raw}")
        try:
            parsed = json.loads(raw)
            return _parse_response(parsed, return_type)
        except json.JSONDecodeError:
            return _parse_response(raw, return_type)

    # -- decorator -------------------------------------------------------

    def __call__(self, func):
        is_async = inspect.iscoroutinefunction(func)
        hints = get_type_hints(func)

        if is_async:
            async def async_wrapper(*args, **kwargs):
                return_type = hints.get("return")
                sig = inspect.signature(func)
                runtime_kwargs, runtime_options = self._split_runtime_options(kwargs)
                generation_options = self._merged_generation_options(runtime_options)
                bound = sig.bind(*args, **runtime_kwargs)
                bound.apply_defaults()

                prompt_text = self._extract_prompt(func, bound.arguments)
                images = self._collect_images(bound.arguments, hints)
                user_msg = self._build_user_message(prompt_text, images)

                if self.session:
                    messages = self.session._prepare_messages(user_msg)
                else:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_msg},
                    ]

                if hasattr(self.provider, "set_request_metadata"):
                    self.provider.set_request_metadata(self.langfuse_metadata)

                if self.stream:
                    raw_stream = await self.provider.complete_async(
                        messages,
                        stream=True,
                        **generation_options,
                    )
                    if self.stream_mode == "json_events":
                        return await self._stream_json_events_async(
                            raw_stream, return_type, user_msg
                        )
                    if self.session:
                        return await self._wrap_stream_with_session_async(
                            raw_stream, user_msg
                        )
                    return raw_stream

                if _needs_structured_output(return_type):
                    try:
                        structured = await self._execute_structured_async(
                            messages,
                            return_type,
                            generation_options,
                        )
                        if self.session:
                            self.session._append_turn(user_msg, structured)
                        return self._finalize_output(structured)
                    except NotImplementedError:
                        logger.debug(
                            f"{type(self.provider).__name__} does not support "
                            "structured output, falling back to JSON prompting"
                        )
                        fallback = await self._execute_json_fallback_async(
                            messages,
                            return_type,
                            generation_options,
                        )
                        if self.session:
                            self.session._append_turn(user_msg, fallback)
                        return self._finalize_output(fallback)

                raw = await self.provider.complete_async(
                    messages,
                    stream=False,
                    **generation_options,
                )
                raw = raw.strip()
                logger.debug(f"Raw LLM response:\n{raw}")
                parsed = _parse_response(raw, return_type)
                if self.session:
                    self.session._append_turn(user_msg, parsed)
                return self._finalize_output(parsed)

            async_wrapper.__name__ = func.__name__
            async_wrapper.__doc__ = func.__doc__
            return async_wrapper
        else:
            def wrapper(*args, **kwargs):
                return_type = hints.get("return")
                sig = inspect.signature(func)
                runtime_kwargs, runtime_options = self._split_runtime_options(kwargs)
                generation_options = self._merged_generation_options(runtime_options)
                bound = sig.bind(*args, **runtime_kwargs)
                bound.apply_defaults()

                prompt_text = self._extract_prompt(func, bound.arguments)
                images = self._collect_images(bound.arguments, hints)
                user_msg = self._build_user_message(prompt_text, images)

                if self.session:
                    messages = self.session._prepare_messages(user_msg)
                else:
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_msg},
                    ]

                if hasattr(self.provider, "set_request_metadata"):
                    self.provider.set_request_metadata(self.langfuse_metadata)

                if self.stream:
                    raw_stream = self.provider.complete(
                        messages,
                        stream=True,
                        **generation_options,
                    )
                    if self.stream_mode == "json_events":
                        return self._stream_json_events(raw_stream, return_type, user_msg)
                    if self.session:
                        return self._wrap_stream_with_session(raw_stream, user_msg)
                    return raw_stream

                if _needs_structured_output(return_type):
                    try:
                        structured = self._execute_structured(
                            messages,
                            return_type,
                            generation_options,
                        )
                        if self.session:
                            self.session._append_turn(user_msg, structured)
                        return self._finalize_output(structured)
                    except NotImplementedError:
                        logger.debug(
                            f"{type(self.provider).__name__} does not support "
                            "structured output, falling back to JSON prompting"
                        )
                        fallback = self._execute_json_fallback(
                            messages,
                            return_type,
                            generation_options,
                        )
                        if self.session:
                            self.session._append_turn(user_msg, fallback)
                        return self._finalize_output(fallback)

                raw = self.provider.complete(
                    messages,
                    stream=False,
                    **generation_options,
                )
                raw = raw.strip()
                logger.debug(f"Raw LLM response:\n{raw}")
                parsed = _parse_response(raw, return_type)
                if self.session:
                    self.session._append_turn(user_msg, parsed)
                return self._finalize_output(parsed)

            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper


def prompt(
    provider,
    *,
    stream: bool = False,
    stream_mode: str = "text",
    langfuse_metadata: dict | None = None,
    generation_options: dict | None = None,
    return_metadata: bool = False,
):
    """Decorator factory that turns a function's docstring into an LLM prompt.

    The execution path is chosen automatically based on the return type:

    - ``str``, ``int``, ``float``, ``bool`` (or no annotation) use plain
      text completion.
    - Pydantic ``BaseModel`` subclasses and generic collection types
      (``List[...]``, ``Dict[...]``) use the provider's structured output
      when available, with an automatic JSON-prompting fallback.
    - ``stream=True`` always uses plain streaming regardless of return type.

    Args:
        provider: An LLMProvider instance.
        stream: If True, return a streaming iterator instead of a complete response.
        stream_mode: Streaming behavior. ``text`` yields text chunks;
            ``json_events`` yields event dicts.
        langfuse_metadata: Optional metadata dict passed to LangfuseWrapper.
        generation_options: Default provider options passed to each call
            (for example ``temperature``, ``top_p``, ``max_tokens``).
        return_metadata: If True, return ``PromptResult`` with output + metadata.

    Example::

        @prompt(provider=my_provider)
        def summarize(text: str) -> str:
            \"\"\"Summarize this text in 2 sentences: {text}\"\"\"

        # Per-call overrides:
        summarize("text", _llm_options={"temperature": 0.2})
    """
    return _PromptDecorator(
        provider=provider,
        stream=stream,
        stream_mode=stream_mode,
        langfuse_metadata=langfuse_metadata,
        generation_options=generation_options,
        return_metadata=return_metadata,
    )


def generate_image(
    provider,
    prompt: str,
    *,
    model: str | None = None,
    generation_options: dict | None = None,
    return_metadata: bool = False,
):
    """Generate an image using a provider's image-generation endpoint."""
    options = dict(generation_options or {})
    if model is not None:
        options["model"] = model
    result = provider.generate_image(prompt, **options)
    if not return_metadata:
        return result
    return PromptResult(output=result, metadata=get_last_response_metadata(provider))


async def generate_image_async(
    provider,
    prompt: str,
    *,
    model: str | None = None,
    generation_options: dict | None = None,
    return_metadata: bool = False,
):
    """Async image-generation helper."""
    options = dict(generation_options or {})
    if model is not None:
        options["model"] = model
    result = await provider.generate_image_async(prompt, **options)
    if not return_metadata:
        return result
    return PromptResult(output=result, metadata=get_last_response_metadata(provider))
