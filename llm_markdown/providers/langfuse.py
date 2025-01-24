from .base import LLMProvider
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from langfuse.media import LangfuseMedia
import logging
import re
import base64
from typing import Union, Iterator

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
        host: str = "https://cloud.langfuse.com"
    ):
        self.provider = provider
        self.langfuse = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host
        )

    def _process_base64_image(self, data_uri: str) -> LangfuseMedia:
        """
        Convert a base64 data URI to a LangfuseMedia object.
        """
        # Extract mime type and base64 data
        mime_type = data_uri.split(';')[0].split(':')[1]
        base64_data = data_uri.split(',')[1]
        
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_data)
        
        # Create LangfuseMedia object
        return LangfuseMedia(
            obj=data_uri,  # Keep original for provider
            content_bytes=image_bytes,
            content_type=mime_type
        )

    def _sanitize_message(self, message: dict) -> dict:
        """
        Process a message, handling images with LangfuseMedia.
        """
        if not isinstance(message, dict):
            return message

        sanitized = message.copy()
        
        # Handle string content with !image[] syntax
        if isinstance(message.get('content'), str):
            pattern = r'!image\[(data:image/[^;]+;base64,[^\]]+)\]'
            matches = re.finditer(pattern, message['content'])
            content = message['content']
            
            # Replace each image with LangfuseMedia
            for match in matches:
                data_uri = match.group(1)
                media = self._process_base64_image(data_uri)
                content = content.replace(f"!image[{data_uri}]", f"!image[{media}]")
            
            sanitized['content'] = content
        
        # Handle array content (OpenAI format)
        elif isinstance(message.get('content'), list):
            sanitized_content = []
            for item in message['content']:
                if isinstance(item, dict) and item.get('type') == 'image_url':
                    image_url = item.get('image_url', {}).get('url', '')
                    if image_url.startswith('data:'):
                        media = self._process_base64_image(image_url)
                        sanitized_content.append({
                            'type': 'image_url',
                            'image_url': {'url': media}
                        })
                    else:
                        sanitized_content.append(item)
                else:
                    sanitized_content.append(item)
            sanitized['content'] = sanitized_content

        return sanitized

    @observe(as_type="generation")
    def query(self, messages: list[dict], stream: bool = False) -> Union[str, Iterator[str]]:
        """
        Send messages to the underlying provider and log to Langfuse.
        """
        # Extract model info if available
        model = getattr(self.provider, 'model', 'unknown')
        
        # Process messages with proper media handling
        sanitized_messages = [self._sanitize_message(msg) for msg in messages]
        
        # Update observation
        langfuse_context.update_current_observation(
            name="llm_query",
            input=sanitized_messages,
            model=model,
            metadata={
                "provider": self.provider.__class__.__name__,
                "original_message_count": len(messages),
                "streaming": stream
            }
        )

        # Call provider
        response = self.provider.query(messages, stream=stream)

        if not stream:
            # Update with complete response
            langfuse_context.update_current_observation(
                output=response
            )
            return response
        
        # For streaming, wrap the generator to log the complete response at the end
        def wrapped_stream():
            chunks = []
            try:
                for chunk in response:
                    chunks.append(chunk)
                    yield chunk
            finally:
                # Log complete response when stream ends
                complete_response = ''.join(chunks)
                langfuse_context.update_current_observation(
                    output=complete_response
                )
        
        return wrapped_stream()

    def __del__(self):
        try:
            if hasattr(self, '_trace') and self._trace:
                self._trace.end()
        except Exception:
            # Silently ignore errors during shutdown
            pass
