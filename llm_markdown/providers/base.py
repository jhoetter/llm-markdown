from abc import ABC, abstractmethod
from typing import Iterator, Union

class LLMProvider(ABC):
    @abstractmethod
    def query(self, messages: list[dict], stream: bool = False) -> Union[str, Iterator[str]]:
        """
        Send a list of messages to the LLM and return the response.
        Messages should follow the format:
        [
            {"role": "system", "content": "System message here"},
            {"role": "user", "content": "User message here"}
        ]

        Args:
            messages: List of message dictionaries
            stream: If True, return an iterator of response chunks

        Returns:
            Either a complete response string or an iterator of response chunks
        """
        pass
