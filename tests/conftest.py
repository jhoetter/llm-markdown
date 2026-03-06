import os
import pytest
from llm_markdown.providers.base import LLMProvider

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # Keep tests working when python-dotenv is not installed.
    pass


class MockProvider(LLMProvider):
    """Test provider that records calls and returns canned responses."""

    def __init__(self, response="mock response", structured_response=None):
        self.calls: list[tuple] = []
        self._response = response
        self._structured = structured_response or response
        self._last_response_metadata = None

    def complete(self, messages, **kwargs):
        self.calls.append(("complete", messages, kwargs))
        self._last_response_metadata = {
            "provider": "MockProvider",
            "model": "mock-model",
            "response_id": "mock-response-id",
            "usage": None,
        }
        if kwargs.get("stream"):
            return iter(list(self._response))
        return self._response

    def generate_image(self, prompt, **kwargs):
        self.calls.append(("generate_image", prompt, kwargs))
        self._last_response_metadata = {
            "provider": "MockProvider",
            "model": kwargs.get("model", "mock-image-model"),
            "response_id": "mock-image-id",
            "usage": None,
            "image_generation": True,
        }
        return {
            "provider": "MockProvider",
            "model": kwargs.get("model", "mock-image-model"),
            "response_id": "mock-image-id",
            "images": [{"url": "https://example.com/mock.png", "b64_json": None}],
        }

    async def generate_image_async(self, prompt, **kwargs):
        self.calls.append(("generate_image_async", prompt, kwargs))
        return self.generate_image(prompt, **kwargs)

    async def complete_async(self, messages, **kwargs):
        self.calls.append(("complete_async", messages, kwargs))
        self._last_response_metadata = {
            "provider": "MockProvider",
            "model": "mock-model",
            "response_id": "mock-response-id",
            "usage": None,
        }
        if kwargs.get("stream"):
            async def _gen():
                for c in self._response:
                    yield c
            return _gen()
        return self._response

    def complete_structured(self, messages, schema, **kwargs):
        self.calls.append(("complete_structured", messages, schema, kwargs))
        self._last_response_metadata = {
            "provider": "MockProvider",
            "model": "mock-model",
            "response_id": "mock-structured-id",
            "usage": None,
        }
        return self._structured

    async def complete_structured_async(self, messages, schema, **kwargs):
        self.calls.append(("complete_structured_async", messages, schema, kwargs))
        self._last_response_metadata = {
            "provider": "MockProvider",
            "model": "mock-model",
            "response_id": "mock-structured-id",
            "usage": None,
        }
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
        "markers", "integration: tests that hit real provider APIs"
    )


def has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))
