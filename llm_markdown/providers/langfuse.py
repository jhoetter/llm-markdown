import json
from .base import LLMProvider
from langfuse import get_client
from langfuse.media import LangfuseMedia
import logging
import re
import base64
from typing import Union, Iterator, AsyncIterator

logger = logging.getLogger(__name__)


class LangfuseWrapper(LLMProvider):
    """
    A wrapper provider that logs LLM interactions to Langfuse.
    """

    def __init__(
        self,
        provider: LLMProvider,
        secret_key: str,
        public_key: str,
        host: str = "https://cloud.langfuse.com",
    ):
        self.provider = provider
        # Initialize Langfuse client with new SDK pattern
        import os

        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_HOST"] = host
        self.langfuse = get_client()
        self._request_metadata = {}  # Store metadata for the current request

    def _process_base64_image(self, data_uri: str) -> LangfuseMedia:
        """
        Convert a base64 data URI to a LangfuseMedia object.
        """
        # Extract mime type and base64 data
        mime_type = data_uri.split(";")[0].split(":")[1]
        base64_data = data_uri.split(",")[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_data)

        # Create LangfuseMedia object
        return LangfuseMedia(
            obj=data_uri,  # Keep original for provider
            content_bytes=image_bytes,
            content_type=mime_type,
        )

    def _sanitize_message(self, message: dict) -> dict:
        """
        Process a message, handling images with LangfuseMedia.
        """
        if not isinstance(message, dict):
            return message

        sanitized = message.copy()

        # Handle string content with !image[] syntax
        if isinstance(message.get("content"), str):
            pattern = r"!image\[(data:image/[^;]+;base64,[^\]]+)\]"
            matches = re.finditer(pattern, message["content"])
            content = message["content"]

            # Replace each image with LangfuseMedia
            for match in matches:
                data_uri = match.group(1)
                media = self._process_base64_image(data_uri)
                content = content.replace(f"!image[{data_uri}]", f"!image[{media}]")

            sanitized["content"] = content

        # Handle array content (OpenAI format)
        elif isinstance(message.get("content"), list):
            sanitized_content = []
            for item in message["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        media = self._process_base64_image(image_url)
                        sanitized_content.append(
                            {"type": "image_url", "image_url": {"url": media}}
                        )
                    else:
                        sanitized_content.append(item)
                else:
                    sanitized_content.append(item)
            sanitized["content"] = sanitized_content

        return sanitized

    def set_request_metadata(self, metadata: dict):
        """
        Set metadata for the next request. This is called by the llm decorator
        to pass metadata like categories for filtering in Langfuse.

        Args:
            metadata: Dictionary with metadata to include in Langfuse generation
        """
        self._request_metadata = metadata or {}

    def query(
        self, messages: list[dict], stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Send messages to the underlying provider and log to Langfuse.
        """
        # Extract model info if available
        model = getattr(self.provider, "model", "unknown")

        # Process messages with proper media handling
        sanitized_messages = [self._sanitize_message(msg) for msg in messages]

        # Call provider
        response = self.provider.query(messages, stream=stream)

        if not stream:
            # Get usage information from provider if available
            usage = getattr(self.provider, "_last_usage", None)

            # Create generation with usage info for cost tracking
            try:
                gen_kwargs = {
                    "name": "llm_query",
                    "input": sanitized_messages,
                    "output": response,
                }

                # Add model if available
                if model != "unknown":
                    gen_kwargs["model"] = model

                # Add metadata if provided (for filtering/categorization)
                if self._request_metadata:
                    gen_kwargs["metadata"] = self._request_metadata

                # Add usage info if available for cost tracking
                if usage and usage.get("total_tokens"):
                    gen_kwargs["usage_details"] = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                gen = self.langfuse.start_observation(
                    as_type="generation", **gen_kwargs
                )
                gen.end()

                # Clear metadata after use
                self._request_metadata = {}
            except Exception as e:
                logger.warning(f"Failed to log to Langfuse: {e}")
                self._request_metadata = {}

            return response

        # For streaming, we need to collect all chunks to log the complete response
        def wrapped_stream():
            chunks = []
            for chunk in response:
                chunks.append(chunk)
                yield chunk

            # After streaming completes, collect full response and log with usage info
            full_response = "".join(chunks)
            usage = getattr(self.provider, "_last_usage", None)

            try:
                gen_kwargs = {
                    "name": "llm_query",
                    "input": sanitized_messages,
                    "output": full_response,
                }

                # Add model if available
                if model != "unknown":
                    gen_kwargs["model"] = model

                # Add metadata if provided (for filtering/categorization)
                if self._request_metadata:
                    gen_kwargs["metadata"] = self._request_metadata

                # Add usage info if available for cost tracking
                if usage and usage.get("total_tokens"):
                    gen_kwargs["usage_details"] = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                gen = self.langfuse.start_observation(
                    as_type="generation", **gen_kwargs
                )
                gen.end()

                # Clear metadata after use
                self._request_metadata = {}
            except Exception as e:
                logger.warning(f"Failed to log to Langfuse: {e}")
                self._request_metadata = {}

        return wrapped_stream()

    async def query_async(
        self, messages: list[dict], stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """
        Async version to send messages to the underlying provider and log to Langfuse.
        """
        # Extract model info if available
        model = getattr(self.provider, "model", "unknown")

        # Process messages with proper media handling
        sanitized_messages = [self._sanitize_message(msg) for msg in messages]

        # Call provider asynchronously
        response = await self.provider.query_async(messages, stream=stream)

        if not stream:
            # Get usage information from provider if available
            usage = getattr(self.provider, "_last_usage", None)

            # Create generation with usage info for cost tracking
            try:
                gen_kwargs = {
                    "name": "llm_query_async",
                    "input": sanitized_messages,
                    "output": response,
                }

                # Add model if available
                if model != "unknown":
                    gen_kwargs["model"] = model

                # Add metadata if provided (for filtering/categorization)
                if self._request_metadata:
                    gen_kwargs["metadata"] = self._request_metadata

                # Add usage info if available for cost tracking
                if usage and usage.get("total_tokens"):
                    gen_kwargs["usage_details"] = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                gen = self.langfuse.start_observation(
                    as_type="generation", **gen_kwargs
                )
                gen.end()

                # Clear metadata after use
                self._request_metadata = {}
            except Exception as e:
                logger.warning(f"Failed to log to Langfuse: {e}")
                self._request_metadata = {}

            return response

        # For streaming, we need to collect all chunks to log the complete response
        async def wrapped_stream():
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
                yield chunk

            # After streaming completes, collect full response and log with usage info
            full_response = "".join(chunks)
            usage = getattr(self.provider, "_last_usage", None)

            try:
                gen_kwargs = {
                    "name": "llm_query_async",
                    "input": sanitized_messages,
                    "output": full_response,
                }

                # Add model if available
                if model != "unknown":
                    gen_kwargs["model"] = model

                # Add metadata if provided (for filtering/categorization)
                if self._request_metadata:
                    gen_kwargs["metadata"] = self._request_metadata

                # Add usage info if available for cost tracking
                if usage and usage.get("total_tokens"):
                    gen_kwargs["usage_details"] = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                gen = self.langfuse.start_observation(
                    as_type="generation", **gen_kwargs
                )
                gen.end()

                # Clear metadata after use
                self._request_metadata = {}
            except Exception as e:
                logger.warning(f"Failed to log to Langfuse: {e}")
                self._request_metadata = {}

        return wrapped_stream()

    def supports_structured_output(self) -> bool:
        """Delegate to inner provider to check structured output support."""
        return self.provider.supports_structured_output()

    def query_structured(self, messages: list[dict], schema: dict) -> dict:
        """
        Delegate structured query to inner provider and log to Langfuse.

        Args:
            messages: Chat messages
            schema: JSON Schema for the expected output structure

        Returns:
            Parsed JSON response as dict
        """
        # Extract model info if available
        model = getattr(self.provider, "model", "unknown")

        # Process messages with proper media handling
        sanitized_messages = [self._sanitize_message(msg) for msg in messages]

        # Call provider's structured query
        result = self.provider.query_structured(messages, schema)

        # Get usage information from provider if available
        usage = getattr(self.provider, "_last_usage", None)

        # Log to Langfuse
        try:
            gen_kwargs = {
                "name": "llm_query_structured",
                "input": sanitized_messages,
                "output": json.dumps(result),
            }

            # Add model if available
            if model != "unknown":
                gen_kwargs["model"] = model

            # Add metadata if provided (for filtering/categorization)
            if self._request_metadata:
                gen_kwargs["metadata"] = self._request_metadata

            # Add usage info if available for cost tracking
            if usage and usage.get("total_tokens"):
                gen_kwargs["usage_details"] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

            gen = self.langfuse.start_observation(as_type="generation", **gen_kwargs)
            gen.end()

            # Clear metadata after use
            self._request_metadata = {}
        except Exception as e:
            logger.warning(f"Failed to log to Langfuse: {e}")
            self._request_metadata = {}

        return result

    async def query_structured_async(self, messages: list[dict], schema: dict) -> dict:
        """
        Async version: Delegate structured query to inner provider and log to Langfuse.

        Args:
            messages: Chat messages
            schema: JSON Schema for the expected output structure

        Returns:
            Parsed JSON response as dict
        """
        # Extract model info if available
        model = getattr(self.provider, "model", "unknown")

        # Process messages with proper media handling
        sanitized_messages = [self._sanitize_message(msg) for msg in messages]

        # Call provider's structured query asynchronously
        result = await self.provider.query_structured_async(messages, schema)

        # Get usage information from provider if available
        usage = getattr(self.provider, "_last_usage", None)

        # Log to Langfuse
        try:
            gen_kwargs = {
                "name": "llm_query_structured_async",
                "input": sanitized_messages,
                "output": json.dumps(result),
            }

            # Add model if available
            if model != "unknown":
                gen_kwargs["model"] = model

            # Add metadata if provided (for filtering/categorization)
            if self._request_metadata:
                gen_kwargs["metadata"] = self._request_metadata

            # Add usage info if available for cost tracking
            if usage and usage.get("total_tokens"):
                gen_kwargs["usage_details"] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

            gen = self.langfuse.start_observation(as_type="generation", **gen_kwargs)
            gen.end()

            # Clear metadata after use
            self._request_metadata = {}
        except Exception as e:
            logger.warning(f"Failed to log to Langfuse: {e}")
            self._request_metadata = {}

        return result

    def __del__(self):
        try:
            # Flush any remaining observations
            if hasattr(self, "langfuse") and self.langfuse:
                self.langfuse.flush()
        except Exception:
            # Silently ignore errors during shutdown
            pass
