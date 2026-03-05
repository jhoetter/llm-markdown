# Providers

Built-in providers:

- `OpenAIProvider`
- `AnthropicProvider`
- `GeminiProvider`
- `OpenRouterProvider`

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
