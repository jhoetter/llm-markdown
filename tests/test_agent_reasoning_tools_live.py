"""Live provider tests for ``stream_agent_turn`` (tools + reasoning)."""

from __future__ import annotations

import os
from typing import cast

import pytest

from llm_markdown.agent_stream import AgentReasoningDelta, AgentSegmentStart, AgentToolCallDelta
from llm_markdown.agent_turn import stream_agent_turn
from llm_markdown.providers import AnthropicProvider, OpenAIProvider, OpenRouterProvider
from llm_markdown.reasoning import BackendName, ReasoningConfig, ReasoningMode

_CALC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Return a + b.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Return a * b.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


def _backend() -> str:
    return (os.environ.get("LLM_MARKDOWN_AGENT_BACKEND") or "openai").strip().lower()


def _reasoning_mode() -> ReasoningMode:
    raw = (os.environ.get("LLM_MARKDOWN_AGENT_REASONING_MODE") or "native").strip().lower()
    if raw == "off":
        return ReasoningMode.OFF
    if raw == "fallback":
        return ReasoningMode.FALLBACK
    return ReasoningMode.NATIVE


def _model_for(backend: str) -> str:
    env_model = (os.environ.get("LLM_MARKDOWN_AGENT_MODEL") or "").strip()
    if env_model:
        return env_model
    if backend == "openai":
        return "gpt-4o-mini"
    if backend == "openrouter":
        return "openai/gpt-4o-mini"
    if backend == "anthropic":
        return "claude-3-5-haiku-latest"
    pytest.fail(f"unknown backend {backend!r}")


def _provider_for(backend: str, model: str):
    if backend == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            pytest.skip("OPENAI_API_KEY not set")
        return OpenAIProvider(api_key=key, model=model)
    if backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            pytest.skip("OPENROUTER_API_KEY not set")
        return OpenRouterProvider(api_key=key, model=model)
    if backend == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            pytest.skip("ANTHROPIC_API_KEY not set")
        return AnthropicProvider(api_key=key, model=model)
    pytest.skip(f"LLM_MARKDOWN_AGENT_BACKEND={backend!r} not supported for live agent tests")


def _reasoning_config(mode: ReasoningMode) -> ReasoningConfig:
    if mode is ReasoningMode.NATIVE and _backend() == "anthropic":
        return ReasoningConfig.native(
            anthropic_thinking={"type": "enabled", "budget_tokens": 2048},
        )
    if mode is ReasoningMode.NATIVE:
        return ReasoningConfig.native()
    if mode is ReasoningMode.OFF:
        return ReasoningConfig.off()
    return ReasoningConfig(mode=ReasoningMode.FALLBACK)


@pytest.mark.integration
def test_live_stream_agent_turn_requests_add_tool():
    backend = _backend()
    if backend == "gemini":
        pytest.skip("gemini is not supported for stream_agent_turn")
    mode = _reasoning_mode()
    model = _model_for(backend)
    provider = _provider_for(backend, model)
    rc = _reasoning_config(mode)

    messages = [
        {
            "role": "user",
            "content": (
                "You must call the add tool exactly once with a=19 and b=23 to answer. "
                "Do not answer with plain arithmetic in the final message without calling add."
            ),
        }
    ]
    events = list(
        stream_agent_turn(
            provider,
            cast(BackendName, backend),
            messages,
            model=model,
            tools=_CALC_TOOLS,
            tool_choice="auto",
            reasoning=rc,
        )
    )
    tool_names = [e.name for e in events if isinstance(e, AgentToolCallDelta) and e.name]
    assert "add" in tool_names


@pytest.mark.integration
def test_live_fallback_think_tags_reasoning_and_tool():
    """FALLBACK uses <think> tags; reasoning streams as AgentReasoningDelta, then tool call."""
    if _reasoning_mode() is not ReasoningMode.FALLBACK:
        pytest.skip("set LLM_MARKDOWN_AGENT_REASONING_MODE=fallback")

    backend = _backend()
    if backend == "gemini":
        pytest.skip("gemini is not supported for stream_agent_turn")
    model = _model_for(backend)
    provider = _provider_for(backend, model)

    messages = [
        {
            "role": "user",
            "content": (
                "Call the add tool once with a=3 and b=5. Use the tool; do not skip it."
            ),
        }
    ]
    events = list(
        stream_agent_turn(
            provider,
            cast(BackendName, backend),
            messages,
            model=model,
            tools=_CALC_TOOLS,
            tool_choice="auto",
            reasoning=ReasoningConfig(mode=ReasoningMode.FALLBACK),
        )
    )
    idx_t = next((i for i, e in enumerate(events) if isinstance(e, AgentToolCallDelta)), None)
    assert idx_t is not None, "expected a tool call"
    tool_names = [e.name for e in events if isinstance(e, AgentToolCallDelta) and e.name]
    assert "add" in tool_names
    idx_content_seg = next(
        i for i, e in enumerate(events) if isinstance(e, AgentSegmentStart) and e.segment == "content"
    )
    assert idx_content_seg < idx_t
    idx_r = next((i for i, e in enumerate(events) if isinstance(e, AgentReasoningDelta)), None)
    assert idx_r is not None, "expected Phase A reasoning deltas"
    assert idx_r < idx_t
