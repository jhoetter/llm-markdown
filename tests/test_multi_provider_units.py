import json
from types import SimpleNamespace

import pytest

from llm_markdown.providers.anthropic import AnthropicProvider
from llm_markdown.providers.gemini import GeminiProvider
from llm_markdown.providers.openai import OpenAIProvider
from llm_markdown.providers.openrouter import OpenRouterProvider
from llm_markdown.providers.router import RouterProvider
from tests.conftest import MockProvider


def test_openrouter_provider_uses_openrouter_base_url(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.setdefault("calls", []).append(kwargs)

    monkeypatch.setattr("llm_markdown.providers.openai.openai.OpenAI", FakeClient)
    monkeypatch.setattr("llm_markdown.providers.openai.openai.AsyncOpenAI", FakeClient)

    provider = OpenRouterProvider(
        api_key="test-key",
        model="openai/gpt-4o-mini",
        app_name="llm-markdown-test",
        app_url="https://example.com",
    )

    assert isinstance(provider, OpenAIProvider)
    assert captured["calls"][0]["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["calls"][0]["default_headers"]["HTTP-Referer"] == "https://example.com"
    assert captured["calls"][0]["default_headers"]["X-Title"] == "llm-markdown-test"


def test_anthropic_provider_structured_tool_use(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get("api_key")

    class FakeMessages:
        def create(self, **kwargs):
            assert kwargs["tool_choice"]["name"] == "structured_response"
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="tool_use", input={"answer": "ok"}),
                ],
                usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get("api_key")
            self.messages = FakeMessages()

    def fake_import(name):
        if name == "anthropic":
            return SimpleNamespace(Anthropic=FakeClient, AsyncAnthropic=FakeAsyncClient)
        raise ImportError(name)

    monkeypatch.setattr("llm_markdown.providers.anthropic.importlib.import_module", fake_import)

    provider = AnthropicProvider(api_key="test-key")
    result = provider.complete_structured(
        messages=[{"role": "user", "content": "Return JSON"}],
        schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    assert result == {"answer": "ok"}
    assert provider._last_usage["total_tokens"] == 15
    assert provider.last_response_metadata()["provider"] == "AnthropicProvider"


@pytest.mark.asyncio
async def test_gemini_provider_stream_and_structured(monkeypatch):
    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeModels:
        def generate_content(self, **kwargs):
            config = kwargs["config"]
            if config.kwargs.get("response_mime_type") == "application/json":
                return SimpleNamespace(
                    text=json.dumps({"sentiment": "positive"}),
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=3,
                        candidates_token_count=2,
                        total_token_count=5,
                    ),
                )
            return SimpleNamespace(
                text="hello",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=2,
                    candidates_token_count=1,
                    total_token_count=3,
                ),
            )

        def generate_content_stream(self, **kwargs):
            _ = kwargs
            return [SimpleNamespace(text="a"), SimpleNamespace(text="b")]

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = FakeModels()

    def fake_import(name):
        if name == "google.genai":
            return SimpleNamespace(Client=FakeClient)
        if name == "google.genai.types":
            return SimpleNamespace(GenerateContentConfig=FakeConfig)
        raise ImportError(name)

    monkeypatch.setattr("llm_markdown.providers.gemini.importlib.import_module", fake_import)

    provider = GeminiProvider(api_key="test-key")
    text = provider.complete([{"role": "user", "content": "Hi"}], stream=False)
    assert text == "hello"

    structured = provider.complete_structured(
        [{"role": "user", "content": "JSON please"}],
        schema={"type": "object", "properties": {"sentiment": {"type": "string"}}},
    )
    assert structured["sentiment"] == "positive"
    assert provider.last_response_metadata()["provider"] == "GeminiProvider"

    stream = await provider.complete_async([{"role": "user", "content": "stream"}], stream=True)
    chunks = [chunk async for chunk in stream]
    assert "".join(chunks) == "ab"
    assert provider._last_usage is None


def test_router_provider_composes_providers():
    router = RouterProvider(routes=[MockProvider(response="ok")])
    result = router.complete([{"role": "user", "content": "hi"}])
    assert result == "ok"
