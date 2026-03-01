import os
import pytest
from llm_markdown.providers.base import LLMProvider


class MockProvider(LLMProvider):
    """Test provider that records calls and returns canned responses."""

    def __init__(self, response="mock response", structured_response=None):
        self.calls: list[tuple] = []
        self._response = response
        self._structured = structured_response or response

    def complete(self, messages, **kwargs):
        self.calls.append(("complete", messages, kwargs))
        if kwargs.get("stream"):
            return iter(list(self._response))
        return self._response

    async def complete_async(self, messages, **kwargs):
        self.calls.append(("complete_async", messages, kwargs))
        if kwargs.get("stream"):
            async def _gen():
                for c in self._response:
                    yield c
            return _gen()
        return self._response

    def complete_structured(self, messages, schema):
        self.calls.append(("complete_structured", messages, schema))
        return self._structured

    async def complete_structured_async(self, messages, schema):
        self.calls.append(("complete_structured_async", messages, schema))
        return self._structured


class BareProvider(LLMProvider):
    """Provider that does NOT implement structured output.

    Falls back to complete() with JSON in the system prompt when the
    decorator needs structured output.
    """

    def __init__(self, response="bare response"):
        self.calls: list[tuple] = []
        self._response = response

    def complete(self, messages, **kwargs):
        self.calls.append(("complete", messages, kwargs))
        return self._response

    async def complete_async(self, messages, **kwargs):
        self.calls.append(("complete_async", messages, kwargs))
        return self._response


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def bare_provider():
    return BareProvider()


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: tests that hit a real OpenAI API"
    )


def has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))
