"""Unit tests for infer_provider_failure (no live API calls)."""

from __future__ import annotations

import pytest

from llm_markdown.providers.failure_info import (
    ProviderFailureCategory,
    infer_provider_failure,
)


def test_infer_generic_429_with_retry_after_header() -> None:
    class Resp:
        status_code = 429
        headers = {"retry-after": "15"}

    class Exc(Exception):
        status_code = 429
        response = Resp()

    f = infer_provider_failure(Exc())
    assert f is not None
    assert f.category == ProviderFailureCategory.RATE_LIMIT
    assert f.http_status == 429
    assert f.retry_after_seconds == 15.0
    assert "15" in f.public_message or "seconds" in f.public_message
    assert f.public_message
    assert f.technical_detail


def test_infer_rate_limit_from_message_heuristic() -> None:
    class Exc(Exception):
        def __str__(self) -> str:
            return "upstream said: rate limit exceeded for tenant"

    f = infer_provider_failure(Exc())
    assert f is not None
    assert f.category == ProviderFailureCategory.RATE_LIMIT


def test_infer_overloaded_from_message() -> None:
    class Exc(Exception):
        def __str__(self) -> str:
            return "model is overloaded, try later"

    f = infer_provider_failure(Exc())
    assert f is not None
    assert f.category == ProviderFailureCategory.OVERLOADED


def test_infer_server_from_status() -> None:
    class Exc(Exception):
        status_code = 503

    f = infer_provider_failure(Exc())
    assert f is not None
    assert f.category == ProviderFailureCategory.SERVER
    assert f.http_status == 503


def test_unknown_exception_returns_none() -> None:
    class Exc(Exception):
        def __str__(self) -> str:
            return "something else"

    assert infer_provider_failure(Exc()) is None


def test_infer_anthropic_api_status_rate_limit() -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")

    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(429, request=req, headers={"retry-after": "3"})
    body = {"error": {"type": "rate_limit_error", "message": "Too many requests"}}
    exc = anthropic.APIStatusError("rate limited", response=resp, body=body)

    f = infer_provider_failure(exc)
    assert f is not None
    assert f.category == ProviderFailureCategory.RATE_LIMIT
    assert f.http_status == 429
    assert f.retry_after_seconds == 3.0
    assert "rate limit" in f.public_message.lower()
    assert "HTTP 429" in f.technical_detail or "rate_limit" in f.technical_detail


def test_infer_httpx_429() -> None:
    httpx = pytest.importorskip("httpx")

    req = httpx.Request("GET", "https://example.com/v1/messages")
    resp = httpx.Response(429, request=req, headers={"retry-after": "7"})
    exc = httpx.HTTPStatusError("throttled", request=req, response=resp)

    f = infer_provider_failure(exc)
    assert f is not None
    assert f.category == ProviderFailureCategory.RATE_LIMIT
    assert f.http_status == 429
    assert f.retry_after_seconds == 7.0


def test_infer_openai_rate_limit_error() -> None:
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, request=req, headers={})
    body = {"error": {"message": "Rate limit reached"}}
    exc = openai.RateLimitError("rate limited", response=resp, body=body)

    f = infer_provider_failure(exc)
    assert f is not None
    assert f.category == ProviderFailureCategory.RATE_LIMIT
    assert f.http_status == 429
