from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def query(self, messages: list[dict]) -> str:
        """
        Send a list of messages to the LLM and return the response.
        Messages should follow the format:
        [
            {"role": "system", "content": "System message here"},
            {"role": "user", "content": "User message here"}
        ]
        """
        pass
