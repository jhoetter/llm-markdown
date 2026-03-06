# Providers

Built-in providers:

- `OpenAIProvider`
- `AnthropicProvider`
- `GeminiProvider`
- `OpenRouterProvider`
- `RouterProvider`

All providers support sync + async + streaming via the same `@prompt` API.

## Common options

All built-in providers now accept:

- `timeout_seconds`
- `max_retries`
- `retry_backoff_seconds`

Generation controls (for example `temperature`, `top_p`, `max_tokens`) can be passed through:

- default: `@prompt(..., generation_options={...})`
- per call: `func(..., _llm_options={...})`

## Errors

Built-in providers normalize request failures to `ProviderError` for consistent handling.

## RouterProvider

Use `RouterProvider` to provide ordered fallback and routing metadata across providers:

```python
import os

from llm_markdown.providers import RouterProvider, OpenAIProvider, AnthropicProvider

router = RouterProvider(
    routes=[
        OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini"),
        AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"], model="claude-3-5-sonnet-latest"),
    ],
    max_latency_ms=2500,
    max_cost=0.5,
    failure_threshold=3,
    circuit_cooldown_seconds=30,
)
```

Router resilience controls:

- `failure_threshold`: open circuit for provider after N consecutive failures.
- `circuit_cooldown_seconds`: cooldown before provider can be retried.
- `max_latency_ms`: optional route filter (requires `on_latency` hook).
- `max_cost`: optional route filter (requires `on_cost` hook).

## Telemetry contract

Provider metadata includes a normalized baseline for operations:

- `request_id`
- `provider`
- `model`
- `latency_ms`
- `retry_attempts`
- `fallback_attempts` (router-level)
- `token_usage` and `image_usage` (when applicable)

## OpenRouter model access

Configuration control points for OpenRouter:

- **Credential source**: `OPENROUTER_API_KEY` in your `.env`.
- **Runtime model selection**: pass `model="provider/model-id"` to `OpenRouterProvider(...)`.
- **Integration test model list**: `OPENROUTER_TEST_MODELS` in `tests/test_integration_multi_providers.py`.

To find accessible model IDs for your account, check the OpenRouter models catalog: [OpenRouter Models](https://openrouter.ai/models).

Example Inception Labs model ID via OpenRouter:

- `inception/mercury-2`
