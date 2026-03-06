import pytest

from llm_markdown.providers import RouterProvider
from llm_markdown.providers.base import LLMProvider, ProviderError
from tests.conftest import MockProvider, BareProvider


class FailingProvider(LLMProvider):
    def __init__(self, retryable: bool):
        super().__init__()
        self.retryable = retryable

    def complete(self, messages, **kwargs):
        _ = (messages, kwargs)
        raise ProviderError(
            provider="FailingProvider",
            message="boom",
            retryable=self.retryable,
        )

    async def complete_async(self, messages, **kwargs):
        return self.complete(messages, **kwargs)


def test_router_fallback_on_retryable_error():
    router = RouterProvider(
        routes=[FailingProvider(retryable=True), MockProvider(response="ok")],
    )
    result = router.complete([{"role": "user", "content": "hi"}])
    assert result == "ok"
    metadata = router.last_response_metadata()
    assert metadata["selected_provider"] == "MockProvider"
    assert len(metadata["attempts"]) == 2


def test_router_stops_on_non_retryable_error():
    router = RouterProvider(
        routes=[FailingProvider(retryable=False), MockProvider(response="ok")],
    )
    with pytest.raises(ProviderError):
        router.complete([{"role": "user", "content": "hi"}])


def test_router_structured_uses_capable_provider():
    structured_provider = MockProvider(structured_response={"value": "x"})
    router = RouterProvider(routes=[BareProvider(response="[]"), structured_provider])
    result = router.complete_structured(
        [{"role": "user", "content": "hi"}],
        schema={"type": "object", "properties": {"value": {"type": "string"}}},
    )
    assert result["value"] == "x"


def test_router_sticky_routing_key():
    p1 = MockProvider(response="one")
    p2 = MockProvider(response="two")
    router = RouterProvider(
        routes=[p1, p2],
        sticky_key_fn=lambda messages, kwargs: kwargs.get("request_id"),
    )

    first = router.complete([{"role": "user", "content": "hi"}], request_id="abc")
    second = router.complete([{"role": "user", "content": "again"}], request_id="abc")
    assert first == "one"
    assert second == "one"


def test_router_circuit_breaker_skips_open_provider():
    failing = FailingProvider(retryable=True)
    backup = MockProvider(response="ok")
    router = RouterProvider(
        routes=[failing, backup],
        failure_threshold=1,
        circuit_cooldown_seconds=60,
    )
    first = router.complete([{"role": "user", "content": "one"}])
    assert first == "ok"
    # Failing provider circuit should now be open, so only backup is attempted.
    second = router.complete([{"role": "user", "content": "two"}])
    assert second == "ok"
    attempts = router.last_response_metadata()["attempts"]
    assert len(attempts) == 1


def test_router_respects_cost_budget():
    expensive = MockProvider(response="expensive")
    cheap = MockProvider(response="cheap")

    router = RouterProvider(
        routes=[expensive, cheap],
        on_cost=lambda provider: 10.0 if provider is expensive else 1.0,
        max_cost=2.0,
    )
    result = router.complete([{"role": "user", "content": "hi"}])
    assert result == "cheap"


def test_router_metadata_contract_fields():
    router = RouterProvider(routes=[MockProvider(response="ok")])
    router.complete([{"role": "user", "content": "hi"}])
    metadata = router.last_response_metadata()
    assert metadata["request_id"].startswith("router-")
    assert "latency_ms" in metadata
    assert "retry_attempts" in metadata
    assert "fallback_attempts" in metadata
