import pytest

from llm_markdown.providers.base import LLMProvider, ProviderError


class RetryProvider(LLMProvider):
    def __init__(self):
        super().__init__(max_retries=2, retry_backoff_seconds=0)
        self.calls = 0

    def complete(self, messages, **kwargs):
        _ = (messages, kwargs)
        self.calls += 1

        class RetryableError(Exception):
            status_code = 429

        if self.calls < 3:
            raise RetryableError("rate limit")
        return "ok"

    async def complete_async(self, messages, **kwargs):
        return self.complete(messages, **kwargs)


def test_retry_helper_retries_on_retryable_errors():
    provider = RetryProvider()
    result = provider._call_with_retries(lambda: provider.complete([{"role": "user"}]))
    assert result == "ok"
    assert provider.calls == 3


def test_retry_helper_raises_provider_error_for_non_retryable():
    provider = RetryProvider()

    class BadError(Exception):
        pass

    with pytest.raises(ProviderError):
        provider._call_with_retries(lambda: (_ for _ in ()).throw(BadError("boom")))
