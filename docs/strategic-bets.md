# Strategic Bets

This document scopes larger roadmap items that are intentionally not fully implemented yet.

## 1) Conversation/session API

### Goal

Support multi-turn state while preserving the function-as-prompt ergonomics.

### Proposed shape

- `Session(provider, system_prompt=...)`
- `session.prompt(fn)(...)` for typed prompts with shared history
- optional history window controls (`max_messages`, `max_tokens`)

### Risks

- Message-growth cost
- Return-type consistency across turns
- Provider differences in role handling

## 2) Provider routing/fallback

### Goal

Allow failover and cost-aware routing across providers/models.

### Proposed shape

- `RouterProvider(routes=[...], fallback=[...])`
- policy hooks: `on_error`, `on_latency`, `on_cost`
- optional sticky routing per session or request key

### Risks

- Normalizing errors/metadata consistently
- Capability mismatch (images, structured output)

## 3) Structured streaming

### Goal

Provide incremental typed data while streaming.

### Proposed shape

- `stream_mode="text" | "json_events"`
- event objects (`partial_json`, `delta_text`, `done`, `error`)
- optional final model validation at end of stream

### Risks

- Partial JSON correctness guarantees
- Increased complexity for provider adapters

## Suggested order

1. Conversation/session API
2. Router/fallback provider
3. Structured streaming events
