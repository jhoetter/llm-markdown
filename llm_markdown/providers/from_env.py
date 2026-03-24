"""Build OpenAI or Anthropic providers from model id and optional env overrides."""

from __future__ import annotations

import os
from typing import Literal

from llm_markdown.providers.base import LLMProvider

BackendName = Literal["openai", "anthropic"]


def _normalize_backend_token(raw: str) -> BackendName | None:
    s = (raw or "").strip().lower()
    if s in ("openai",):
        return "openai"
    if s in ("anthropic", "claude"):
        return "anthropic"
    return None


def infer_llm_markdown_backend_for_model(model: str) -> BackendName:
    """Choose API backend from a model id (primary signal for provider selection)."""
    m = (model or "").strip().lower()
    if not m:
        return "openai"
    if "claude" in m or m.startswith("anthropic.") or m.startswith("anthropic/"):
        return "anthropic"
    return "openai"


def resolve_llm_markdown_backend(
    model: str,
    *,
    provider_override: str | None = None,
) -> BackendName:
    """Resolve backend: explicit override, then ``LLM_MARKDOWN_PROVIDER`` env, then model inference."""
    if provider_override is not None:
        o = provider_override.strip()
        if not o:
            return infer_llm_markdown_backend_for_model(model)
        norm = _normalize_backend_token(o)
        if norm is None:
            msg = (
                f"Invalid provider override {provider_override!r} "
                "(use openai, anthropic, or claude)"
            )
            raise ValueError(msg)
        return norm

    env = os.environ.get("LLM_MARKDOWN_PROVIDER", "").strip()
    if env:
        norm = _normalize_backend_token(env)
        if norm is None:
            msg = (
                f"Invalid LLM_MARKDOWN_PROVIDER env {env!r} "
                "(use openai, anthropic, or claude)"
            )
            raise ValueError(msg)
        return norm

    return infer_llm_markdown_backend_for_model(model)


def build_llm_provider_for_model(
    model: str,
    *,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    provider: str | None = None,
) -> LLMProvider:
    """Instantiate ``OpenAIProvider`` or ``AnthropicProvider`` for ``model``.

    Keys: non-empty ``openai_api_key`` / ``anthropic_api_key`` args, else
    ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` from the environment.

    Backend: ``provider`` argument if set, else ``LLM_MARKDOWN_PROVIDER`` env,
    else inferred from the model id (e.g. Claude ids → Anthropic).
    """
    m = (model or "").strip() or "gpt-4o-mini"
    backend = resolve_llm_markdown_backend(m, provider_override=provider)

    if backend == "anthropic":
        key = (anthropic_api_key or "").strip() or (
            os.environ.get("ANTHROPIC_API_KEY", "").strip()
        )
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for Anthropic-backed models. "
                "Set LLM_MARKDOWN_PROVIDER=openai only if you intend to call an OpenAI API model."
            )
        from llm_markdown.providers.anthropic import AnthropicProvider

        return AnthropicProvider(api_key=key, model=m)

    key = (openai_api_key or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "OPENAI_API_KEY is required for OpenAI-backed models. "
            "Set LLM_MARKDOWN_PROVIDER=anthropic if your model uses the Anthropic API."
        )
    from llm_markdown.providers.openai import OpenAIProvider

    return OpenAIProvider(api_key=key, model=m)
