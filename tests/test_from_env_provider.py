"""Unit tests for llm_markdown.providers.from_env (no live API)."""

import pytest

from llm_markdown.providers.anthropic import AnthropicProvider
from llm_markdown.providers.from_env import (
    build_llm_provider_for_model,
    infer_llm_markdown_backend_for_model,
    resolve_llm_markdown_backend,
)
from llm_markdown.providers.openai import OpenAIProvider


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("", "openai"),
        ("gpt-4o-mini", "openai"),
        ("o3-mini", "openai"),
        ("claude-3-5-sonnet-latest", "anthropic"),
        ("claude-sonnet-4-6", "anthropic"),
        ("Anthropic.claude-foo", "anthropic"),
        ("anthropic/claude-3-opus", "anthropic"),
    ],
)
def test_infer_backend_from_model(model: str, expected: str) -> None:
    assert infer_llm_markdown_backend_for_model(model) == expected


def test_resolve_backend_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MARKDOWN_PROVIDER", raising=False)
    assert resolve_llm_markdown_backend("gpt-4o") == "openai"
    monkeypatch.setenv("LLM_MARKDOWN_PROVIDER", "anthropic")
    assert resolve_llm_markdown_backend("gpt-4o") == "anthropic"
    monkeypatch.setenv("LLM_MARKDOWN_PROVIDER", "claude")
    assert resolve_llm_markdown_backend("gpt-4o") == "anthropic"
    monkeypatch.setenv("LLM_MARKDOWN_PROVIDER", "openai")
    assert resolve_llm_markdown_backend("claude-3-opus") == "openai"


def test_resolve_backend_invalid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MARKDOWN_PROVIDER", "azure")
    with pytest.raises(ValueError, match="Invalid LLM_MARKDOWN_PROVIDER"):
        resolve_llm_markdown_backend("gpt-4o")


def test_resolve_backend_override_arg() -> None:
    assert resolve_llm_markdown_backend("claude-x", provider_override="openai") == "openai"
    assert resolve_llm_markdown_backend("gpt-4o", provider_override="anthropic") == "anthropic"
    with pytest.raises(ValueError, match="Invalid provider override"):
        resolve_llm_markdown_backend("gpt-4o", provider_override="azure")


def test_build_openai_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MARKDOWN_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = build_llm_provider_for_model("gpt-4o-mini")
    assert isinstance(p, OpenAIProvider)
    assert p.model == "gpt-4o-mini"


def test_build_anthropic_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_MARKDOWN_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    p = build_llm_provider_for_model("claude-sonnet-4-6")
    assert isinstance(p, AnthropicProvider)
    assert p.model == "claude-sonnet-4-6"


def test_build_openai_explicit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    p = build_llm_provider_for_model("gpt-4o", openai_api_key="inline-key")
    assert isinstance(p, OpenAIProvider)


def test_build_missing_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MARKDOWN_PROVIDER", "openai")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_llm_provider_for_model("gpt-4o-mini")


def test_build_missing_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_llm_provider_for_model("claude-3-haiku-20240307")


def test_default_model_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    p = build_llm_provider_for_model("")
    assert isinstance(p, OpenAIProvider)
    assert p.model == "gpt-4o-mini"
