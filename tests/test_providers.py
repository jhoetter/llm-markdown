import warnings
from unittest.mock import MagicMock

import pytest

from llm_markdown.providers.openai import (
    _uses_modern_tokens,
    _extract_usage,
    _extract_chunk_usage,
    OpenAIProvider,
    OpenAILegacyProvider,
)


# ---- _uses_modern_tokens ---------------------------------------------------

@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-5", True),
        ("gpt-5-turbo", True),
        ("o1", True),
        ("o1-mini", True),
        ("o1-preview", True),
        ("o3", True),
        ("o3-mini", True),
        ("o4", True),
        ("o4-mini", True),
        ("gpt-4o-mini", False),
        ("gpt-4o", False),
        ("gpt-4", False),
        ("gpt-4-turbo", False),
        ("gpt-3.5-turbo", False),
    ],
)
def test_uses_modern_tokens(model, expected):
    assert _uses_modern_tokens(model) is expected


# ---- OpenAIProvider token param auto-detection -----------------------------

@pytest.mark.parametrize(
    "model,expected_param",
    [
        ("gpt-4o-mini", "max_tokens"),
        ("gpt-4o", "max_tokens"),
        ("gpt-3.5-turbo", "max_tokens"),
        ("gpt-5", "max_completion_tokens"),
        ("o1-mini", "max_completion_tokens"),
        ("o3", "max_completion_tokens"),
        ("o4-mini", "max_completion_tokens"),
    ],
)
def test_openai_provider_token_param(model, expected_param):
    provider = OpenAIProvider(api_key="test-key", model=model)
    assert provider._token_param == expected_param
    assert provider._token_kwargs() == {expected_param: 4096}


def test_openai_provider_custom_max_tokens():
    provider = OpenAIProvider(api_key="test-key", model="gpt-4o-mini", max_tokens=8192)
    assert provider._token_kwargs() == {"max_tokens": 8192}


# ---- OpenAILegacyProvider deprecation warning ------------------------------

def test_legacy_provider_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        OpenAILegacyProvider(api_key="test-key")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()


def test_legacy_provider_is_subclass_of_openai_provider():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        provider = OpenAILegacyProvider(api_key="test-key")
    assert isinstance(provider, OpenAIProvider)


# ---- _extract_usage --------------------------------------------------------

def test_extract_usage_with_usage():
    response = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30

    result = _extract_usage(response)
    assert result == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }


def test_extract_usage_without_usage():
    response = MagicMock(spec=[])
    result = _extract_usage(response)
    assert result is None


def test_extract_usage_with_none_usage():
    response = MagicMock()
    response.usage = None
    result = _extract_usage(response)
    assert result is None


# ---- _extract_chunk_usage --------------------------------------------------

def test_extract_chunk_usage_with_usage():
    chunk = MagicMock()
    chunk.usage.prompt_tokens = 5
    chunk.usage.completion_tokens = 10
    chunk.usage.total_tokens = 15

    result = _extract_chunk_usage(chunk)
    assert result == {
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "total_tokens": 15,
    }


def test_extract_chunk_usage_none_chunk():
    assert _extract_chunk_usage(None) is None


def test_extract_chunk_usage_no_usage_attr():
    chunk = MagicMock(spec=[])
    assert _extract_chunk_usage(chunk) is None


def test_extract_chunk_usage_none_usage():
    chunk = MagicMock()
    chunk.usage = None
    assert _extract_chunk_usage(chunk) is None
