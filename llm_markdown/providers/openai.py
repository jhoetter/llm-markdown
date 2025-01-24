import openai
from .base import LLMProvider
from typing import Union, Iterator


class OpenAIProvider(LLMProvider):
    def __init__(
        self, api_key: str, model: str = "gpt-4o-mini", max_tokens: int = 4096
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.client = openai.OpenAI(api_key=self.api_key)

    def query(self, messages: list[dict], stream: bool = False) -> Union[str, Iterator[str]]:
        """
        Send a chat-style conversation to OpenAI and return the response content.
        Handles both text-only and multimodal messages.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            stream=stream
        )

        if stream:
            def response_generator():
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            return response_generator()
        
        return response.choices[0].message.content
