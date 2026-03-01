__version__ = "0.3.1"

import inspect
import json
import base64
import re
import requests
from urllib.parse import urlparse
from typing import get_type_hints, get_origin, get_args, List, Dict, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image type for multimodal prompts
# ---------------------------------------------------------------------------

def _is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def _to_data_uri(source: str) -> str:
    """Convert a URL, file path, or raw base64 string into a data URI."""
    if _is_url(source):
        response = requests.get(source)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "image/jpeg")
        b64 = base64.b64encode(response.content).decode("utf-8")
        return f"data:{content_type};base64,{b64}"

    if source.startswith("data:"):
        return source

    # Assume raw base64
    try:
        base64.b64decode(source)
        return f"data:image/jpeg;base64,{source}"
    except Exception:
        raise ValueError(
            "Image source must be a URL, data URI, or base64-encoded string"
        )


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
        langfuse_metadata: dict | None = None,
    ):
        self.provider = provider
        self.stream = stream
        self.langfuse_metadata = langfuse_metadata or {}

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

    # -- execution paths -------------------------------------------------

    def _execute_structured(self, messages: list[dict], return_type):
        """Structured output path via complete_structured."""
        schema = _build_schema(return_type)
        sys_msg = _system_instructions(return_type)
        structured_messages = [
            {"role": "system", "content": sys_msg},
            messages[-1],
        ]
        result = self.provider.complete_structured(structured_messages, schema)
        logger.debug(f"Structured output: {result}")
        return _parse_response(result, return_type)

    async def _execute_structured_async(self, messages: list[dict], return_type):
        """Async structured output path via complete_structured_async."""
        schema = _build_schema(return_type)
        sys_msg = _system_instructions(return_type)
        structured_messages = [
            {"role": "system", "content": sys_msg},
            messages[-1],
        ]
        result = await self.provider.complete_structured_async(
            structured_messages, schema
        )
        logger.debug(f"Structured output: {result}")
        return _parse_response(result, return_type)

    def _execute_json_fallback(self, messages: list[dict], return_type):
        """Fallback: ask for JSON via system prompt, parse the text response."""
        sys_msg = _system_instructions(return_type)
        fallback_messages = [
            {"role": "system", "content": sys_msg},
            messages[-1],
        ]
        raw = self.provider.complete(fallback_messages, stream=False)
        raw = raw.strip()
        logger.debug(f"JSON fallback response:\n{raw}")
        try:
            parsed = json.loads(raw)
            return _parse_response(parsed, return_type)
        except json.JSONDecodeError:
            return _parse_response(raw, return_type)

    async def _execute_json_fallback_async(self, messages: list[dict], return_type):
        """Async fallback: ask for JSON via system prompt, parse the text response."""
        sys_msg = _system_instructions(return_type)
        fallback_messages = [
            {"role": "system", "content": sys_msg},
            messages[-1],
        ]
        raw = await self.provider.complete_async(fallback_messages, stream=False)
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
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                prompt_text = self._extract_prompt(func, bound.arguments)
                images = self._collect_images(bound.arguments, hints)
                user_msg = self._build_user_message(prompt_text, images)

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_msg},
                ]

                if hasattr(self.provider, "set_request_metadata"):
                    self.provider.set_request_metadata(self.langfuse_metadata)

                if self.stream:
                    return await self.provider.complete_async(
                        messages, stream=True
                    )

                if _needs_structured_output(return_type):
                    try:
                        return await self._execute_structured_async(
                            messages, return_type
                        )
                    except NotImplementedError:
                        logger.debug(
                            f"{type(self.provider).__name__} does not support "
                            "structured output, falling back to JSON prompting"
                        )
                        return await self._execute_json_fallback_async(
                            messages, return_type
                        )

                raw = await self.provider.complete_async(messages, stream=False)
                raw = raw.strip()
                logger.debug(f"Raw LLM response:\n{raw}")
                return _parse_response(raw, return_type)

            async_wrapper.__name__ = func.__name__
            async_wrapper.__doc__ = func.__doc__
            return async_wrapper
        else:
            def wrapper(*args, **kwargs):
                return_type = hints.get("return")
                sig = inspect.signature(func)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                prompt_text = self._extract_prompt(func, bound.arguments)
                images = self._collect_images(bound.arguments, hints)
                user_msg = self._build_user_message(prompt_text, images)

                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_msg},
                ]

                if hasattr(self.provider, "set_request_metadata"):
                    self.provider.set_request_metadata(self.langfuse_metadata)

                if self.stream:
                    return self.provider.complete(messages, stream=True)

                if _needs_structured_output(return_type):
                    try:
                        return self._execute_structured(messages, return_type)
                    except NotImplementedError:
                        logger.debug(
                            f"{type(self.provider).__name__} does not support "
                            "structured output, falling back to JSON prompting"
                        )
                        return self._execute_json_fallback(messages, return_type)

                raw = self.provider.complete(messages, stream=False)
                raw = raw.strip()
                logger.debug(f"Raw LLM response:\n{raw}")
                return _parse_response(raw, return_type)

            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper


def prompt(
    provider,
    *,
    stream: bool = False,
    langfuse_metadata: dict | None = None,
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
        langfuse_metadata: Optional metadata dict passed to LangfuseWrapper.

    Example::

        @prompt(provider=my_provider)
        def summarize(text: str) -> str:
            \"\"\"Summarize this text in 2 sentences: {text}\"\"\"
    """
    return _PromptDecorator(
        provider=provider,
        stream=stream,
        langfuse_metadata=langfuse_metadata,
    )
