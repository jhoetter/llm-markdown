"""Integration tests that hit the real OpenAI API.

These tests are skipped unless:
  1. The OPENAI_API_KEY environment variable is set, AND
  2. The test is run with the --run-integration flag or -m integration marker.

Run them with:
    OPENAI_API_KEY=sk-... pytest tests/test_integration.py -m integration
"""

import os

import pytest
from pydantic import BaseModel

from llm_markdown import prompt, Image
from llm_markdown.providers.openai import OpenAIProvider

_skip = not os.environ.get("OPENAI_API_KEY")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(_skip, reason="OPENAI_API_KEY not set"),
]


@pytest.fixture(scope="module")
def provider():
    return OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model="gpt-4o-mini",
        max_tokens=256,
    )


# ---- simple string return --------------------------------------------------

def test_simple_string_return(provider):
    @prompt(provider=provider)
    def capital(country: str) -> str:
        """What is the capital of {country}? Reply with just the city name."""

    result = capital("France")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "paris" in result.lower()


# ---- structured output with Pydantic ---------------------------------------

class CityInfo(BaseModel):
    city: str
    country: str
    population_estimate: str


def test_pydantic_structured_output(provider):
    @prompt(provider=provider)
    def city_info(city_name: str) -> CityInfo:
        """Provide information about {city_name}."""

    result = city_info("Tokyo")
    assert isinstance(result, CityInfo)
    assert result.city.lower() == "tokyo"
    assert len(result.country) > 0


# ---- streaming -------------------------------------------------------------

def test_streaming_response(provider):
    @prompt(provider=provider, stream=True)
    def haiku(topic: str) -> str:
        """Write a haiku about {topic}."""

    chunks = list(haiku("rain"))
    full = "".join(chunks)
    assert len(full) > 10


# ---- multimodal (image) ----------------------------------------------------

def test_image_description(provider):
    @prompt(provider=provider)
    def describe(image: Image) -> str:
        """Describe this image briefly."""

    result = describe(Image("https://picsum.photos/id/237/200/200"))
    assert isinstance(result, str)
    assert len(result) > 10
