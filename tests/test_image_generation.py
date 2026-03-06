from types import SimpleNamespace

import pytest

from llm_markdown import PromptResult, generate_image, generate_image_async
from llm_markdown.providers.openai import OpenAIProvider
from llm_markdown.providers.router import RouterProvider
from tests.conftest import MockProvider


def _fake_image_response():
    return SimpleNamespace(
        id="img-resp-1",
        data=[
            SimpleNamespace(
                url="https://example.com/image.png",
                b64_json=None,
                revised_prompt="updated prompt",
            )
        ],
    )


def test_openai_provider_generate_image(monkeypatch):
    provider = OpenAIProvider(api_key="test-key", model="gpt-image-1")
    monkeypatch.setattr(provider.client.images, "generate", lambda **kwargs: _fake_image_response())

    result = provider.generate_image("A red fox")
    assert result["provider"] == "OpenAIProvider"
    assert result["model"] == "gpt-image-1"
    assert result["images"][0]["url"] == "https://example.com/image.png"
    assert provider.last_response_metadata()["image_generation"] is True


@pytest.mark.asyncio
async def test_openai_provider_generate_image_async(monkeypatch):
    provider = OpenAIProvider(api_key="test-key", model="gpt-image-1")

    async def _fake_generate(**kwargs):
        _ = kwargs
        return _fake_image_response()

    monkeypatch.setattr(provider.async_client.images, "generate", _fake_generate)
    result = await provider.generate_image_async("A blue bird")
    assert result["response_id"] == "img-resp-1"


def test_high_level_generate_image_with_metadata():
    provider = MockProvider()
    result = generate_image(
        provider,
        "A mountain at sunset",
        model="openai/gpt-image-1",
        return_metadata=True,
    )
    assert isinstance(result, PromptResult)
    assert result.output["images"][0]["url"] == "https://example.com/mock.png"
    assert result.metadata["image_generation"] is True


@pytest.mark.asyncio
async def test_high_level_generate_image_async():
    provider = MockProvider()
    result = await generate_image_async(provider, "A city skyline")
    assert result["provider"] == "MockProvider"


def test_router_provider_generate_image_fallback():
    first = MockProvider()
    second = MockProvider()

    def _broken(prompt, **kwargs):
        _ = (prompt, kwargs)
        raise RuntimeError("temporary error")

    first.generate_image = _broken
    router = RouterProvider(routes=[first, second])
    result = router.generate_image("A tree")
    assert result["provider"] == "MockProvider"
    assert router.last_response_metadata()["selected_provider"] == "MockProvider"
