"""Integration tests for non-OpenAI providers.

Run examples:
  ANTHROPIC_API_KEY=... pytest tests/test_integration_multi_providers.py -m integration
  GOOGLE_API_KEY=... pytest tests/test_integration_multi_providers.py -m integration
  OPENROUTER_API_KEY=... pytest tests/test_integration_multi_providers.py -m integration
"""

import os

import pytest
from pydantic import BaseModel

from llm_markdown import prompt
from llm_markdown.providers import AnthropicProvider, GeminiProvider, OpenRouterProvider

pytestmark = [pytest.mark.integration]


class BasicInfo(BaseModel):
    item: str
    category: str


ANTHROPIC_TEST_MODELS = [
    "claude-3-5-haiku-latest",
    "claude-3-5-sonnet-latest",
]

GEMINI_TEST_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

OPENROUTER_TEST_MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3.5-haiku",
]


def _skip_if_model_unavailable(provider_name: str, model: str, exc: Exception):
    message = str(exc).lower()
    if any(
        token in message
        for token in (
            "model",
            "not found",
            "does not exist",
            "unsupported",
            "access",
            "permission",
            "not available",
        )
    ):
        pytest.skip(f"{provider_name} model not available for this key: {model}")
    raise exc


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
@pytest.mark.parametrize("model", ANTHROPIC_TEST_MODELS)
def test_anthropic_basic_and_structured(model):
    pytest.importorskip("anthropic")
    provider = AnthropicProvider(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=model,
        max_tokens=256,
    )

    @prompt(provider=provider)
    def short_fact(topic: str) -> str:
        """Say one short fact about {topic}."""

    @prompt(provider=provider)
    def classify(item: str) -> BasicInfo:
        """Return a category and normalized item for {item}."""

    try:
        fact = short_fact("Berlin")
        result = classify("carrot")
        chunks = list(short_poem("sunrise"))
    except Exception as exc:
        _skip_if_model_unavailable("Anthropic", model, exc)

    assert isinstance(fact, str)
    assert isinstance(result, BasicInfo)
    assert result.item
    assert result.category

    @prompt(provider=provider, stream=True)
    def short_poem(topic: str) -> str:
        """Write a very short poem about {topic}."""

    assert len("".join(chunks)) > 10


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)
@pytest.mark.parametrize("model", GEMINI_TEST_MODELS)
def test_gemini_basic_and_structured(model):
    pytest.importorskip("google.genai")
    provider = GeminiProvider(
        api_key=os.environ["GOOGLE_API_KEY"],
        model=model,
        max_tokens=256,
    )

    @prompt(provider=provider)
    def short_fact(topic: str) -> str:
        """Say one short fact about {topic}."""

    @prompt(provider=provider)
    def classify(item: str) -> BasicInfo:
        """Return a category and normalized item for {item}."""

    try:
        fact = short_fact("Rome")
        result = classify("tomato")
        chunks = list(short_poem("rain"))
    except Exception as exc:
        _skip_if_model_unavailable("Gemini", model, exc)

    assert isinstance(fact, str)
    assert isinstance(result, BasicInfo)
    assert result.item
    assert result.category

    @prompt(provider=provider, stream=True)
    def short_poem(topic: str) -> str:
        """Write a very short poem about {topic}."""

    assert len("".join(chunks)) > 10


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
@pytest.mark.parametrize("model", OPENROUTER_TEST_MODELS)
def test_openrouter_basic_and_structured(model):
    pytest.importorskip("openai")
    provider = OpenRouterProvider(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=model,
        max_tokens=256,
    )

    @prompt(provider=provider)
    def short_fact(topic: str) -> str:
        """Say one short fact about {topic}."""

    @prompt(provider=provider)
    def classify(item: str) -> BasicInfo:
        """Return a category and normalized item for {item}."""

    try:
        fact = short_fact("Tokyo")
        result = classify("apple")
        chunks = list(short_poem("moon"))
    except Exception as exc:
        _skip_if_model_unavailable("OpenRouter", model, exc)

    assert isinstance(fact, str)
    assert isinstance(result, BasicInfo)
    assert result.item
    assert result.category

    @prompt(provider=provider, stream=True)
    def short_poem(topic: str) -> str:
        """Write a very short poem about {topic}."""

    assert len("".join(chunks)) > 10
