import time
import uuid
from typing import Any, AsyncIterator, Callable, Iterator, Union

from .base import LLMProvider, ProviderError


class RouterProvider(LLMProvider):
    """Provider router with ordered fallback and optional policy hooks."""

    def __init__(
        self,
        routes: list[LLMProvider],
        fallback: list[LLMProvider] | None = None,
        *,
        on_error: Callable[[Exception, LLMProvider, int], None] | None = None,
        on_latency: Callable[[LLMProvider], float] | None = None,
        on_cost: Callable[[LLMProvider], float] | None = None,
        sticky_key_fn: Callable[[list[dict], dict], str | None] | None = None,
        max_latency_ms: float | None = None,
        max_cost: float | None = None,
        failure_threshold: int = 3,
        circuit_cooldown_seconds: float = 30.0,
    ):
        super().__init__()
        if not routes and not fallback:
            raise ValueError("RouterProvider requires at least one provider route")
        self.routes = list(routes)
        self.fallback = list(fallback or [])
        self.on_error = on_error
        self.on_latency = on_latency
        self.on_cost = on_cost
        self.sticky_key_fn = sticky_key_fn
        self.max_latency_ms = max_latency_ms
        self.max_cost = max_cost
        self.failure_threshold = failure_threshold
        self.circuit_cooldown_seconds = circuit_cooldown_seconds
        self._sticky_map: dict[str, LLMProvider] = {}
        self._failure_counts: dict[int, int] = {}
        self._circuit_open_until: dict[int, float] = {}

    @staticmethod
    def _provider_id(provider: LLMProvider) -> int:
        return id(provider)

    def _is_circuit_open(self, provider: LLMProvider) -> bool:
        until = self._circuit_open_until.get(self._provider_id(provider), 0.0)
        return until > time.time()

    def _record_success(self, provider: LLMProvider):
        pid = self._provider_id(provider)
        self._failure_counts[pid] = 0
        self._circuit_open_until[pid] = 0.0

    def _record_failure(self, provider: LLMProvider):
        pid = self._provider_id(provider)
        count = self._failure_counts.get(pid, 0) + 1
        self._failure_counts[pid] = count
        if count >= self.failure_threshold:
            self._circuit_open_until[pid] = (
                time.time() + self.circuit_cooldown_seconds
            )

    def _all_routes(self) -> list[LLMProvider]:
        merged = self.routes + self.fallback
        seen = set()
        unique = []
        for provider in merged:
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            unique.append(provider)
        return unique

    def _ordered_routes(self, messages: list[dict], kwargs: dict) -> list[LLMProvider]:
        ordered = [provider for provider in self._all_routes() if not self._is_circuit_open(provider)]
        if not ordered:
            return []
        if self.on_cost:
            ordered.sort(key=self.on_cost)
        if self.on_latency:
            ordered.sort(key=self.on_latency)

        if self.max_cost is not None and self.on_cost:
            ordered = [provider for provider in ordered if self.on_cost(provider) <= self.max_cost]
        if self.max_latency_ms is not None and self.on_latency:
            ordered = [
                provider
                for provider in ordered
                if self.on_latency(provider) <= self.max_latency_ms
            ]
        if not ordered:
            return []

        if self.sticky_key_fn:
            key = self.sticky_key_fn(messages, kwargs)
            if key and key in self._sticky_map and self._sticky_map[key] in ordered:
                sticky_provider = self._sticky_map[key]
                ordered = [sticky_provider] + [
                    provider for provider in ordered if provider is not sticky_provider
                ]
        return ordered

    def _normalize_exc(self, exc: Exception, provider: LLMProvider) -> ProviderError:
        if isinstance(exc, ProviderError):
            return exc
        return ProviderError(
            provider=type(provider).__name__,
            message=f"{type(provider).__name__} request failed: {exc}",
            original_error=exc,
            retryable=True,
        )

    def _capture_metadata(
        self,
        provider: LLMProvider,
        attempts: list[dict[str, Any]],
        *,
        started_at: float | None = None,
    ):
        child = provider.last_response_metadata() if hasattr(provider, "last_response_metadata") else getattr(provider, "_last_response_metadata", None)
        usage = getattr(provider, "_last_usage", None)
        self._last_usage = usage
        total_latency_ms = (
            int((time.perf_counter() - started_at) * 1000)
            if started_at is not None
            else None
        )
        request_id = (
            child.get("request_id")
            if isinstance(child, dict) and child.get("request_id")
            else f"router-{uuid.uuid4()}"
        )
        self._last_response_metadata = {
            "provider": "RouterProvider",
            "request_id": request_id,
            "selected_provider": type(provider).__name__,
            "selected_model": getattr(provider, "model", None),
            "attempts": attempts,
            "fallback_attempts": max(0, len(attempts) - 1),
            "retry_attempts": sum(
                1
                for attempt in attempts
                if not attempt.get("ok")
            ),
            "latency_ms": total_latency_ms,
            "child_metadata": child,
            "token_usage": usage,
        }

    def complete(
        self, messages: list[dict], **kwargs
    ) -> Union[str, Iterator[str]]:
        started_total = time.perf_counter()
        ordered = self._ordered_routes(messages, kwargs)
        sticky_key = self.sticky_key_fn(messages, kwargs) if self.sticky_key_fn else None
        attempts = []
        last_error: ProviderError | None = None

        for idx, provider in enumerate(ordered):
            started = time.perf_counter()
            try:
                result = provider.complete(messages, **kwargs)
                attempts.append(
                    {
                        "provider": type(provider).__name__,
                        "ok": True,
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                if sticky_key:
                    self._sticky_map[sticky_key] = provider
                self._record_success(provider)
                if kwargs.get("stream"):
                    def _stream_wrapper():
                        for chunk in result:
                            yield chunk
                        self._capture_metadata(provider, attempts, started_at=started_total)

                    return _stream_wrapper()
                self._capture_metadata(provider, attempts, started_at=started_total)
                return result
            except Exception as exc:
                normalized = self._normalize_exc(exc, provider)
                self._record_failure(provider)
                attempts.append(
                    {
                        "provider": type(provider).__name__,
                        "ok": False,
                        "retryable": normalized.retryable,
                        "error": str(normalized),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                if self.on_error:
                    self.on_error(normalized, provider, idx)
                last_error = normalized
                if not normalized.retryable:
                    break

        if last_error:
            raise last_error
        raise ProviderError("RouterProvider", "No provider routes available", retryable=False)

    async def complete_async(
        self, messages: list[dict], **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        started_total = time.perf_counter()
        ordered = self._ordered_routes(messages, kwargs)
        sticky_key = self.sticky_key_fn(messages, kwargs) if self.sticky_key_fn else None
        attempts = []
        last_error: ProviderError | None = None

        for idx, provider in enumerate(ordered):
            started = time.perf_counter()
            try:
                result = await provider.complete_async(messages, **kwargs)
                attempts.append(
                    {
                        "provider": type(provider).__name__,
                        "ok": True,
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                if sticky_key:
                    self._sticky_map[sticky_key] = provider
                self._record_success(provider)
                if kwargs.get("stream"):
                    async def _stream_wrapper():
                        async for chunk in result:
                            yield chunk
                        self._capture_metadata(provider, attempts, started_at=started_total)

                    return _stream_wrapper()
                self._capture_metadata(provider, attempts, started_at=started_total)
                return result
            except Exception as exc:
                normalized = self._normalize_exc(exc, provider)
                self._record_failure(provider)
                attempts.append(
                    {
                        "provider": type(provider).__name__,
                        "ok": False,
                        "retryable": normalized.retryable,
                        "error": str(normalized),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                if self.on_error:
                    self.on_error(normalized, provider, idx)
                last_error = normalized
                if not normalized.retryable:
                    break

        if last_error:
            raise last_error
        raise ProviderError("RouterProvider", "No provider routes available", retryable=False)

    def _supports_structured(self, provider: LLMProvider) -> bool:
        return type(provider).complete_structured is not LLMProvider.complete_structured

    def complete_structured(self, messages: list[dict], schema: dict, **kwargs) -> dict:
        started_total = time.perf_counter()
        ordered = [provider for provider in self._ordered_routes(messages, kwargs) if self._supports_structured(provider)]
        if not ordered:
            raise NotImplementedError("No routed provider supports structured output")

        attempts = []
        last_error: ProviderError | None = None
        for idx, provider in enumerate(ordered):
            started = time.perf_counter()
            try:
                result = provider.complete_structured(messages, schema, **kwargs)
                attempts.append(
                    {
                        "provider": type(provider).__name__,
                        "ok": True,
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                self._record_success(provider)
                self._capture_metadata(provider, attempts, started_at=started_total)
                return result
            except Exception as exc:
                normalized = self._normalize_exc(exc, provider)
                self._record_failure(provider)
                attempts.append(
                    {
                        "provider": type(provider).__name__,
                        "ok": False,
                        "retryable": normalized.retryable,
                        "error": str(normalized),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                if self.on_error:
                    self.on_error(normalized, provider, idx)
                last_error = normalized
                if not normalized.retryable:
                    break

        if last_error:
            raise last_error
        raise ProviderError("RouterProvider", "No provider routes available", retryable=False)

    async def complete_structured_async(
        self, messages: list[dict], schema: dict, **kwargs
    ) -> dict:
        started_total = time.perf_counter()
        ordered = [provider for provider in self._ordered_routes(messages, kwargs) if self._supports_structured(provider)]
        if not ordered:
            raise NotImplementedError("No routed provider supports structured output")

        attempts = []
        last_error: ProviderError | None = None
        for idx, provider in enumerate(ordered):
            started = time.perf_counter()
            try:
                result = await provider.complete_structured_async(messages, schema, **kwargs)
                attempts.append(
                    {
                        "provider": type(provider).__name__,
                        "ok": True,
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                self._record_success(provider)
                self._capture_metadata(provider, attempts, started_at=started_total)
                return result
            except Exception as exc:
                normalized = self._normalize_exc(exc, provider)
                self._record_failure(provider)
                attempts.append(
                    {
                        "provider": type(provider).__name__,
                        "ok": False,
                        "retryable": normalized.retryable,
                        "error": str(normalized),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                if self.on_error:
                    self.on_error(normalized, provider, idx)
                last_error = normalized
                if not normalized.retryable:
                    break

        if last_error:
            raise last_error
        raise ProviderError("RouterProvider", "No provider routes available", retryable=False)

    def _supports_image_generation(self, provider: LLMProvider) -> bool:
        return type(provider).generate_image is not LLMProvider.generate_image

    def generate_image(self, prompt: str, **kwargs) -> dict:
        started_total = time.perf_counter()
        ordered = [provider for provider in self._ordered_routes([], kwargs) if self._supports_image_generation(provider)]
        if not ordered:
            raise NotImplementedError("No routed provider supports image generation")

        attempts = []
        last_error: ProviderError | None = None
        for idx, provider in enumerate(ordered):
            started = time.perf_counter()
            try:
                result = provider.generate_image(prompt, **kwargs)
                attempts.append(
                    {
                        "provider": type(provider).__name__,
                        "ok": True,
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                self._record_success(provider)
                self._capture_metadata(provider, attempts, started_at=started_total)
                if self._last_response_metadata is not None:
                    self._last_response_metadata["image_usage"] = {
                        "count": len(result.get("images", [])),
                    }
                return result
            except Exception as exc:
                normalized = self._normalize_exc(exc, provider)
                self._record_failure(provider)
                attempts.append(
                    {
                        "provider": type(provider).__name__,
                        "ok": False,
                        "retryable": normalized.retryable,
                        "error": str(normalized),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                if self.on_error:
                    self.on_error(normalized, provider, idx)
                last_error = normalized
                if not normalized.retryable:
                    break

        if last_error:
            raise last_error
        raise ProviderError("RouterProvider", "No provider routes available", retryable=False)

    async def generate_image_async(self, prompt: str, **kwargs) -> dict:
        started_total = time.perf_counter()
        ordered = [provider for provider in self._ordered_routes([], kwargs) if self._supports_image_generation(provider)]
        if not ordered:
            raise NotImplementedError("No routed provider supports image generation")

        attempts = []
        last_error: ProviderError | None = None
        for idx, provider in enumerate(ordered):
            started = time.perf_counter()
            try:
                result = await provider.generate_image_async(prompt, **kwargs)
                attempts.append(
                    {
                        "provider": type(provider).__name__,
                        "ok": True,
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                self._record_success(provider)
                self._capture_metadata(provider, attempts, started_at=started_total)
                if self._last_response_metadata is not None:
                    self._last_response_metadata["image_usage"] = {
                        "count": len(result.get("images", [])),
                    }
                return result
            except Exception as exc:
                normalized = self._normalize_exc(exc, provider)
                self._record_failure(provider)
                attempts.append(
                    {
                        "provider": type(provider).__name__,
                        "ok": False,
                        "retryable": normalized.retryable,
                        "error": str(normalized),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    }
                )
                if self.on_error:
                    self.on_error(normalized, provider, idx)
                last_error = normalized
                if not normalized.retryable:
                    break

        if last_error:
            raise last_error
        raise ProviderError("RouterProvider", "No provider routes available", retryable=False)
