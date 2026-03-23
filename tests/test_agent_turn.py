"""Tests for stream_agent_turn and ReasoningConfig validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_markdown.agent_stream import (
    AgentContentDelta,
    AgentMessageFinish,
    AgentReasoningDelta,
    AgentToolCallDelta,
)
from llm_markdown.agent_turn import stream_agent_turn
from llm_markdown.providers.anthropic import AnthropicProvider
from llm_markdown.providers.openai import OpenAIProvider
from llm_markdown.reasoning import ReasoningConfig, ReasoningMode


def test_reasoning_off_conflicts_with_anthropic_thinking():
    with pytest.raises(ValueError, match="OFF cannot be combined"):
        ReasoningConfig(
            mode=ReasoningMode.OFF,
            anthropic_thinking={"type": "enabled", "budget_tokens": 100},
        ).validate_for_backend("anthropic")


def test_openai_backend_rejects_anthropic_thinking():
    with pytest.raises(ValueError, match="anthropic_thinking"):
        ReasoningConfig.native(
            anthropic_thinking={"type": "enabled", "budget_tokens": 100},
        ).validate_for_backend("openai")


def test_anthropic_backend_rejects_openai_extras():
    with pytest.raises(ValueError, match="openai_extras"):
        ReasoningConfig.native(openai_extras={"reasoning_effort": "low"}).validate_for_backend(
            "anthropic"
        )


def test_fallback_openai_two_phase_reasoning_before_tool():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            },
        }
    ]
    phase_a = iter(
        [
            AgentContentDelta(text="plan: use add"),
            AgentMessageFinish(finish_reason="stop", usage=None),
        ]
    )
    phase_b = iter(
        [
            AgentToolCallDelta(index=0, tool_call_id="t1", name="add", arguments="{}"),
            AgentMessageFinish(finish_reason="tool_calls", usage=None),
        ]
    )
    provider = MagicMock()
    provider.stream_chat_completion_events.side_effect = [phase_a, phase_b]

    events = list(
        stream_agent_turn(
            provider,
            "openai",
            [{"role": "user", "content": "x"}],
            model="gpt-4o-mini",
            tools=tools,
            reasoning=ReasoningConfig(mode=ReasoningMode.FALLBACK),
        )
    )
    idx_reasoning = next(i for i, e in enumerate(events) if isinstance(e, AgentReasoningDelta))
    idx_tool = next(i for i, e in enumerate(events) if isinstance(e, AgentToolCallDelta))
    assert idx_reasoning < idx_tool
    assert provider.stream_chat_completion_events.call_count == 2


def test_fallback_anthropic_two_phase_reasoning_before_tool():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            },
        }
    ]
    phase_a = iter(
        [
            AgentContentDelta(text="thinking step"),
            AgentMessageFinish(finish_reason="stop", usage=None),
        ]
    )
    phase_b = iter(
        [
            AgentToolCallDelta(index=0, tool_call_id="t1", name="add", arguments="{}"),
            AgentMessageFinish(finish_reason="tool_calls", usage=None),
        ]
    )
    provider = MagicMock()
    provider.stream_messages_events.side_effect = [phase_a, phase_b]

    events = list(
        stream_agent_turn(
            provider,
            "anthropic",
            [{"role": "user", "content": "x"}],
            model="claude-3-5-haiku-latest",
            tools=tools,
            reasoning=ReasoningConfig(mode=ReasoningMode.FALLBACK),
        )
    )
    idx_reasoning = next(i for i, e in enumerate(events) if isinstance(e, AgentReasoningDelta))
    idx_tool = next(i for i, e in enumerate(events) if isinstance(e, AgentToolCallDelta))
    assert idx_reasoning < idx_tool
    assert provider.stream_messages_events.call_count == 2


def test_gemini_backend_raises():
    with pytest.raises(ValueError, match="gemini"):
        next(
            stream_agent_turn(
                MagicMock(),
                "gemini",
                [{"role": "user", "content": "x"}],
                model="gemini-2.0-flash",
            )
        )


def test_fallback_rejects_openai_extras():
    with pytest.raises(ValueError, match="FALLBACK"):
        ReasoningConfig(
            mode=ReasoningMode.FALLBACK,
            openai_extras={"reasoning_effort": "low"},
        ).validate_for_backend("openai")


def test_openrouter_backend_rejects_anthropic_thinking():
    with pytest.raises(ValueError, match="anthropic_thinking"):
        ReasoningConfig.native(
            anthropic_thinking={"type": "enabled", "budget_tokens": 100},
        ).validate_for_backend("openrouter")


def _choice_delta(content=None, reasoning_content=None):
    delta = SimpleNamespace(content=content, tool_calls=None)
    if reasoning_content is not None:
        delta.reasoning_content = reasoning_content
    return SimpleNamespace(delta=delta, finish_reason=None)


def test_openai_off_filters_reasoning_deltas():
    chunks = [
        SimpleNamespace(
            choices=[_choice_delta(None, reasoning_content="secret")],
            id="1",
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="hi", tool_calls=None),
                    finish_reason=None,
                )
            ],
            id="2",
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content=None, tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            id="3",
        ),
    ]
    provider = OpenAIProvider(api_key="k", model="gpt-4o-mini")
    mock_create = MagicMock(return_value=iter(chunks))
    with patch.object(provider.client.chat.completions, "create", mock_create):
        events = list(
            stream_agent_turn(
                provider,
                "openai",
                [{"role": "user", "content": "z"}],
                model="gpt-4o-mini",
                reasoning=ReasoningConfig.off(),
            )
        )
    assert not any(isinstance(e, AgentReasoningDelta) for e in events)
    assert any(getattr(e, "text", None) == "hi" for e in events)
    assert isinstance(events[-1], AgentMessageFinish)


def test_openai_native_merges_openai_extras():
    provider = OpenAIProvider(api_key="k", model="gpt-4o-mini")
    captured: dict = {}

    def _fake_create(**kw):
        captured.update(kw)
        return iter([])

    with patch.object(provider.client.chat.completions, "create", _fake_create):
        list(
            stream_agent_turn(
                provider,
                "openai",
                [{"role": "user", "content": "z"}],
                model="gpt-4o-mini",
                reasoning=ReasoningConfig.native(openai_extras={"reasoning_effort": "low"}),
            )
        )
    assert captured.get("reasoning_effort") == "low"


def test_anthropic_native_passes_thinking_kw():
    events = [
        SimpleNamespace(
            type="message_stop",
            message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            ),
        ),
    ]

    class CM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            return iter(events)

        def get_final_message(self):
            return SimpleNamespace(
                id="m1",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            )

    provider = AnthropicProvider(api_key="k", model="claude-3-5-haiku-latest")
    mock_stream = MagicMock(return_value=CM())
    thinking = {"type": "enabled", "budget_tokens": 2048}
    with patch.object(provider.client.messages, "stream", mock_stream):
        list(
            stream_agent_turn(
                provider,
                "anthropic",
                [{"role": "user", "content": "hello"}],
                model="claude-3-5-haiku-latest",
                reasoning=ReasoningConfig.native(anthropic_thinking=thinking),
            )
        )
    call_kw = mock_stream.call_args.kwargs
    assert call_kw.get("thinking") == thinking


def test_anthropic_off_drops_thinking_kw():
    events = [
        SimpleNamespace(
            type="message_stop",
            message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            ),
        ),
    ]

    class CM:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            return iter(events)

        def get_final_message(self):
            return SimpleNamespace(
                id="m1",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
            )

    provider = AnthropicProvider(api_key="k", model="claude-3-5-haiku-latest")
    mock_stream = MagicMock(return_value=CM())
    with patch.object(provider.client.messages, "stream", mock_stream):
        list(
            stream_agent_turn(
                provider,
                "anthropic",
                [{"role": "user", "content": "hello"}],
                model="claude-3-5-haiku-latest",
                thinking={"type": "enabled", "budget_tokens": 99},
                reasoning=ReasoningConfig.off(),
            )
        )
    call_kw = mock_stream.call_args.kwargs
    assert "thinking" not in call_kw
