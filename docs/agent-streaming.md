# Agent streaming (tools + reasoning)

Hof-style agents consume a single stream of normalized events from **llm-markdown** (`AgentContentDelta`, `AgentReasoningDelta`, `AgentToolCallDelta`, `AgentMessageFinish`, and **`AgentSegmentStart`** for agentic turns). Use [`stream_agent_turn()`](../llm_markdown/agent_turn.py) so **reasoning** is configured in one place (native vs off vs fallback), similar to structured output (native tool schema vs JSON-in-text fallback).

## Agentic segment contract (`tools` non-empty)

When `stream_agent_turn(..., tools=[...])` is used with a non-empty tool list, the library emits **`AgentSegmentStart`** markers:

1. **`AgentSegmentStart(segment="reasoning")`** is always **first** — consumers can open the “thinking” lane without inferring from the first delta type.
2. **`AgentReasoningDelta`** events belong to that reasoning segment until the first assistant-visible **content** or **tool** delta.
3. **`AgentSegmentStart(segment="content")`** is emitted immediately **before** the first `AgentContentDelta` or `AgentToolCallDelta`.

Turns **without** tools do **not** emit `AgentSegmentStart` (non-agentic chat completion).

**FALLBACK** is hybrid (see policy below): tool-selection rounds use two provider calls (Phase A forces text-only reasoning, then Phase B with tools); answer rounds use one call with `<think>` tag parsing. Segment markers wrap the combined iterator on agentic turns.

## Capability matrix (by provider)

Prefer **native** API reasoning whenever the model and endpoint support it. Use **FALLBACK** only when you want a provider-agnostic planning phase (see below) or native reasoning is unavailable.

| Provider | Native mechanism on the wire | Typical request knobs | Tools + reasoning | llm-markdown notes |
|----------|------------------------------|----------------------|-------------------|-------------------|
| **OpenAI** | Some models stream `reasoning_content` or `reasoning` on chat completion **deltas** (model-dependent). | Extra kwargs to `chat.completions.create` (e.g. `reasoning_effort` where supported). See [OpenAI chat streaming](https://platform.openai.com/docs/api-reference/chat-streaming) and model docs. | Model-dependent; many chat models support `tool_calls` on the delta. | [`OpenAIProvider.stream_chat_completion_events`](../llm_markdown/providers/openai.py) maps deltas. **NATIVE** = pass-through + optional `ReasoningConfig.native(openai_extras=...)`. |
| **Anthropic** | Messages API emits `thinking` stream events (“extended thinking”). | `thinking: { "type": "enabled", "budget_tokens": <n> }` on `messages.stream`. | Extended thinking and tool use can be combined per [Anthropic Messages](https://docs.anthropic.com/en/api/messages) docs. | [`AnthropicProvider.stream_messages_events`](../llm_markdown/providers/anthropic.py). **NATIVE** = `ReasoningConfig.native(anthropic_thinking=...)`. OpenAI-style `tool` / `tool_calls` message history is converted for the Messages API. |
| **Gemini** | Thinking / thought summaries (model family–dependent). | e.g. `includeThoughts`, thinking budgets — [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking). | Function calling is separate from normalized agent streaming here. | [`GeminiProvider`](../llm_markdown/providers/gemini.py) has no unified `AgentStreamEvent` tool loop yet. `stream_agent_turn(..., backend="gemini")` raises **`ValueError`** until streaming is normalized. |
| **OpenRouter** | Depends on the routed upstream model; OpenAI-compatible stream. | Same as upstream OpenAI parameters where applicable. | Same as upstream. | [`OpenRouterProvider`](../llm_markdown/providers/openrouter.py) subclasses OpenAI. **`backend="openrouter"`** uses the same branch as OpenAI in `stream_agent_turn`. **NATIVE** = same as OpenAI. |

**Policy**

- **NATIVE:** one provider call with tools; emit only what the API provides (`AgentReasoningDelta`, content, tools). No synthetic reasoning text.
- **OFF:** do not merge `openai_extras` / Anthropic `thinking`; strip `AgentReasoningDelta` from the stream.
- **FALLBACK:** hybrid in [`agent_fallback.py`](../llm_markdown/agent_fallback.py). **Tool-selection** (non-empty `tools`, no `tool` messages in history yet): **Phase A** — one completion **without** tools and a think-tag instruction; all streamed assistant text is emitted as **`AgentReasoningDelta`** (internal notes are also injected for Phase B). **Phase B** — one completion **with** tools; stream is a pass-through (no tag parsing). **Answer rounds** (no tools, or tool results already in history): **one** completion with think-tag parsing — inside `<think>` tags → **`AgentReasoningDelta`**, outside → **`AgentContentDelta`**. Provider-native **`AgentReasoningDelta`** is forwarded where emitted. Not supported for **`gemini`** (use `openai`, `openrouter`, or `anthropic`).

## When `AgentReasoningDelta` appears

**Regular models** such as **`gpt-4o`** / **`gpt-4o-mini`** often do **not** populate separate reasoning fields on streamed deltas. You will still see tools and assistant text; `reasoning_deltas` may stay `0`. That is expected for those models.

## `ReasoningConfig` modes

- **`native`** (default): Forward provider-native reasoning when the API returns it. Optional `openai_extras` (merged into the OpenAI/OpenRouter request) or `anthropic_thinking` (passed as `thinking=…` on Anthropic) tune behavior where supported.
- **`off`**: Drop Anthropic `thinking` from the request; do not merge `openai_extras` for OpenAI-compatible backends. Filter out **`AgentReasoningDelta`** from the stream.
- **`fallback`**: Hybrid two-phase + think-tags (see **FALLBACK** above). Cannot be combined with `openai_extras` or `anthropic_thinking` (use **native** for those).

## Entry point

- **`stream_agent_turn(provider, backend, messages, model=…, tools=…, reasoning=ReasoningConfig.native(), …)`** — `backend` is **`"openai"`**, **`"openrouter"`**, **`"anthropic"`**, or **`"gemini"`** (unsupported until implemented). Dispatches to `OpenAIProvider.stream_chat_completion_events` or `AnthropicProvider.stream_messages_events` as appropriate.

Optional **`planning_max_tokens`** (with **`reasoning.mode=fallback`**) caps phase A length.

Direct provider methods remain available for advanced use; new code should prefer `stream_agent_turn` for consistent reasoning policy.

## Local configuration (`.env`)

See [`.env.example`](../.env.example). Typical variables:

- **`LLM_MARKDOWN_AGENT_BACKEND`** — `openai` \| `anthropic` \| `openrouter` (not `gemini` for `stream_agent_turn` yet).
- **`LLM_MARKDOWN_AGENT_MODEL`** — optional override (per backend defaults exist in tests/examples).
- **`LLM_MARKDOWN_AGENT_REASONING_MODE`** — `native` \| `off` \| `fallback`. The calculator example
  script defaults to **`fallback`** when this is unset (internal planning + tools); set **`native`**
  when your model streams real reasoning on the wire.

Provider API keys: **`OPENAI_API_KEY`**, **`ANTHROPIC_API_KEY`**, **`OPENROUTER_API_KEY`**, etc.

## Runnable example and integration tests

From the repo root (with a populated `.env`):

```bash
cd /path/to/llm-markdown
# Example defaults to fallback when LLM_MARKDOWN_AGENT_REASONING_MODE is unset.
# Optional: LLM_MARKDOWN_AGENT_BACKEND=anthropic LLM_MARKDOWN_AGENT_REASONING_MODE=native
python examples/agent_calculator_with_reasoning.py
```

Integration tests (require keys; skipped when missing):

```bash
pytest tests/test_agent_reasoning_tools_live.py -m integration -v
```

Unit tests (mocked, no network):

```bash
pytest tests/test_agent_turn.py -q
```

## hof-engine

[Hof](https://github.com/jhoetter/hof-engine) can set **`agent_reasoning_mode=fallback`** (or **`AGENT_REASONING_MODE=fallback`**) so the agent loop uses llm-markdown **FALLBACK** for OpenAI-backed runs.
