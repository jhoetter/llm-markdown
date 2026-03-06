import json
from .base import LLMProvider
from langfuse import get_client
import logging
from typing import Union, Iterator, AsyncIterator

logger = logging.getLogger(__name__)


class LangfuseWrapper(LLMProvider):
    """A wrapper provider that logs LLM interactions to Langfuse."""

    def __init__(
        self,
        provider: LLMProvider,
        secret_key: str,
        public_key: str,
        host: str = "https://cloud.langfuse.com",
    ):
        super().__init__()
        self.provider = provider
        import os

        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_HOST"] = host
        self.langfuse = get_client()
        self._request_metadata = {}

    def set_request_metadata(self, metadata: dict):
        """Set metadata for the next request (e.g. categories for Langfuse filtering)."""
        self._request_metadata = metadata or {}

    def _log_generation(self, name: str, messages: list, output, usage: dict | None):
        try:
            gen_kwargs = {"name": name, "input": messages, "output": output}

            model = getattr(self.provider, "model", None)
            if model:
                gen_kwargs["model"] = model

            if self._request_metadata:
                gen_kwargs["metadata"] = self._request_metadata

            if usage and usage.get("total_tokens"):
                gen_kwargs["usage_details"] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

            gen = self.langfuse.start_observation(as_type="generation", **gen_kwargs)
            gen.end()
        except Exception as e:
            logger.warning(f"Failed to log to Langfuse: {e}")
        finally:
            self._request_metadata = {}

    # -- core completions ------------------------------------------------

    def complete(
        self, messages: list[dict], **kwargs
    ) -> Union[str, Iterator[str]]:
        stream = kwargs.get("stream", False)
        response = self.provider.complete(messages, **kwargs)

        if not stream:
            usage = getattr(self.provider, "_last_usage", None)
            self._log_generation("llm_complete", messages, response, usage)
            self._last_response_metadata = getattr(
                self.provider, "_last_response_metadata", None
            )
            return response

        def _wrapped():
            chunks = []
            for chunk in response:
                chunks.append(chunk)
                yield chunk
            full = "".join(chunks)
            usage = getattr(self.provider, "_last_usage", None)
            self._log_generation("llm_complete", messages, full, usage)
            self._last_response_metadata = getattr(
                self.provider, "_last_response_metadata", None
            )

        return _wrapped()

    async def complete_async(
        self, messages: list[dict], **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        stream = kwargs.get("stream", False)
        response = await self.provider.complete_async(messages, **kwargs)

        if not stream:
            usage = getattr(self.provider, "_last_usage", None)
            self._log_generation("llm_complete_async", messages, response, usage)
            self._last_response_metadata = getattr(
                self.provider, "_last_response_metadata", None
            )
            return response

        async def _wrapped():
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
                yield chunk
            full = "".join(chunks)
            usage = getattr(self.provider, "_last_usage", None)
            self._log_generation("llm_complete_async", messages, full, usage)
            self._last_response_metadata = getattr(
                self.provider, "_last_response_metadata", None
            )

        return _wrapped()

    # -- structured output -----------------------------------------------

    def complete_structured(self, messages: list[dict], schema: dict, **kwargs) -> dict:
        result = self.provider.complete_structured(messages, schema, **kwargs)
        usage = getattr(self.provider, "_last_usage", None)
        self._log_generation(
            "llm_complete_structured", messages, json.dumps(result), usage
        )
        self._last_response_metadata = getattr(
            self.provider, "_last_response_metadata", None
        )
        return result

    async def complete_structured_async(
        self, messages: list[dict], schema: dict, **kwargs
    ) -> dict:
        result = await self.provider.complete_structured_async(messages, schema, **kwargs)
        usage = getattr(self.provider, "_last_usage", None)
        self._log_generation(
            "llm_complete_structured_async", messages, json.dumps(result), usage
        )
        self._last_response_metadata = getattr(
            self.provider, "_last_response_metadata", None
        )
        return result

    # -- image generation ------------------------------------------------

    def generate_image(self, prompt: str, **kwargs) -> dict:
        result = self.provider.generate_image(prompt, **kwargs)
        self._log_generation("llm_generate_image", [{"role": "user", "content": prompt}], json.dumps(result), None)
        self._last_response_metadata = getattr(
            self.provider, "_last_response_metadata", None
        )
        return result

    async def generate_image_async(self, prompt: str, **kwargs) -> dict:
        result = await self.provider.generate_image_async(prompt, **kwargs)
        self._log_generation("llm_generate_image_async", [{"role": "user", "content": prompt}], json.dumps(result), None)
        self._last_response_metadata = getattr(
            self.provider, "_last_response_metadata", None
        )
        return result

    def __del__(self):
        try:
            if hasattr(self, "langfuse") and self.langfuse:
                self.langfuse.flush()
        except Exception:
            pass
