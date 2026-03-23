"""Tests for stream_agent_turn and ReasoningConfig validation."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_markdown.agent_stream import (
    AgentContentDelta,
    AgentMessageFinish,
    AgentReasoningDelta,
    AgentSegmentStart,
    AgentToolCallDelta,
)
from llm_markdown.agent_fallback import stream_agent_turn_fallback
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
    assert isinstance(events[0], AgentSegmentStart)
    assert events[0].segment == "reasoning"
    idx_reasoning = next(i for i, e in enumerate(events) if isinstance(e, AgentReasoningDelta))
    idx_tool = next(i for i, e in enumerate(events) if isinstance(e, AgentToolCallDelta))
    idx_content_seg = next(
        i for i, e in enumerate(events) if isinstance(e, AgentSegmentStart) and e.segment == "content"
    )
    assert idx_reasoning < idx_content_seg < idx_tool
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
    assert isinstance(events[0], AgentSegmentStart)
    assert events[0].segment == "reasoning"
    idx_reasoning = next(i for i, e in enumerate(events) if isinstance(e, AgentReasoningDelta))
    idx_tool = next(i for i, e in enumerate(events) if isinstance(e, AgentToolCallDelta))
    idx_content_seg = next(
        i for i, e in enumerate(events) if isinstance(e, AgentSegmentStart) and e.segment == "content"
    )
    assert idx_reasoning < idx_content_seg < idx_tool
    assert provider.stream_messages_events.call_count == 2


def test_fallback_phase_b_forwards_reasoning_before_tool():
    """Phase B passes through provider-native reasoning deltas before tool calls."""
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
            AgentContentDelta(text="plan"),
            AgentMessageFinish(finish_reason="stop", usage=None),
        ]
    )
    phase_b = iter(
        [
            AgentReasoningDelta(text="wire think"),
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
    rb = next(i for i, e in enumerate(events) if isinstance(e, AgentReasoningDelta) and e.text == "wire think")
    tb = next(i for i, e in enumerate(events) if isinstance(e, AgentToolCallDelta))
    assert rb < tb


def test_agentic_opens_reasoning_segment_before_content_when_no_reasoning_delta():
    tools = [
        {
            "type": "function",
            "function": {"name": "x", "description": "x", "parameters": {"type": "object", "properties": {}}},
        }
    ]
    inner = iter(
        [
            AgentContentDelta(text="hello"),
            AgentMessageFinish(finish_reason="stop", usage=None),
        ]
    )
    provider = MagicMock()
    provider.stream_chat_completion_events.return_value = inner

    events = list(
        stream_agent_turn(
            provider,
            "openai",
            [{"role": "user", "content": "z"}],
            model="gpt-4o-mini",
            tools=tools,
            reasoning=ReasoningConfig.native(),
        )
    )
    assert isinstance(events[0], AgentSegmentStart) and events[0].segment == "reasoning"
    assert isinstance(events[1], AgentSegmentStart) and events[1].segment == "content"
    assert isinstance(events[2], AgentContentDelta)


def test_non_agentic_turn_has_no_segment_markers():
    inner = iter([AgentContentDelta(text="a"), AgentMessageFinish(finish_reason="stop", usage=None)])
    provider = MagicMock()
    provider.stream_chat_completion_events.return_value = inner

    events = list(
        stream_agent_turn(
            provider,
            "openai",
            [{"role": "user", "content": "z"}],
            model="gpt-4o-mini",
            tools=None,
            reasoning=ReasoningConfig.native(),
        )
    )
    assert not any(isinstance(e, AgentSegmentStart) for e in events)


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


def test_fallback_phase_b_bridge_and_plan_wrapping():
    """Phase B injects a bridge into the system message and wraps the plan as internal notes."""
    tools = [
        {"type": "function", "function": {"name": "greet", "parameters": {"type": "object", "properties": {}}}},
    ]
    phase_a = iter(
        [
            AgentContentDelta(text="internal plan"),
            AgentMessageFinish(finish_reason="stop", usage=None),
        ]
    )
    phase_b = iter([AgentMessageFinish(finish_reason="stop", usage=None)])
    provider = MagicMock()
    provider.stream_chat_completion_events.side_effect = [phase_a, phase_b]
    list(
        stream_agent_turn_fallback(
            provider,
            "openai",
            [{"role": "system", "content": "App policy."}, {"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            tools=tools,
        )
    )
    assert provider.stream_chat_completion_events.call_count == 2
    msgs_b = provider.stream_chat_completion_events.call_args_list[1][0][0]
    sys0 = next(m for m in msgs_b if m.get("role") == "system")
    assert "internal notes" in sys0["content"].lower()
    assert "language" in sys0["content"].lower()
    plan_msg = msgs_b[-1]
    assert plan_msg["role"] == "assistant"
    assert plan_msg["content"].startswith("[Internal notes")
    assert "internal plan" in plan_msg["content"]
    assert plan_msg["content"].endswith("[End internal notes]")


def test_fallback_phase_a_asks_for_analysis_not_plan():
    """Phase A system appendix should elicit structured analysis, not answer-shaped prose."""
    from llm_markdown.agent_fallback import _FALLBACK_PLANNING_APPENDIX

    lower = _FALLBACK_PLANNING_APPENDIX.lower()
    assert "intent" in lower
    assert "language" in lower
    assert "tools" in lower or "tool" in lower
    assert "never shown" in lower or "not shown" in lower


def test_fallback_skips_planning_when_no_tools():
    """No tools → single direct call, no Phase A planning overhead."""
    direct = iter(
        [
            AgentContentDelta(text="Hello!"),
            AgentMessageFinish(finish_reason="stop", usage=None),
        ]
    )
    provider = MagicMock()
    provider.stream_chat_completion_events.return_value = direct
    events = list(
        stream_agent_turn_fallback(
            provider,
            "openai",
            [{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
            tools=None,
        )
    )
    assert provider.stream_chat_completion_events.call_count == 1
    assert any(isinstance(e, AgentContentDelta) and e.text == "Hello!" for e in events)
    assert not any(isinstance(e, AgentReasoningDelta) for e in events)


def test_fallback_skips_planning_when_tool_results_present():
    """Post-tool round (tool results in history) → single direct call, no Phase A."""
    direct = iter(
        [
            AgentContentDelta(text="You have 10 expenses."),
            AgentMessageFinish(finish_reason="stop", usage=None),
        ]
    )
    provider = MagicMock()
    provider.stream_chat_completion_events.return_value = direct
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "how many expenses?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_stats", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"total": 10}'},
    ]
    tools = [{"type": "function", "function": {"name": "get_stats", "parameters": {}}}]
    events = list(
        stream_agent_turn_fallback(
            provider, "openai", msgs, model="gpt-4o-mini", tools=tools,
        )
    )
    assert provider.stream_chat_completion_events.call_count == 1
    assert any(isinstance(e, AgentContentDelta) for e in events)
    assert not any(isinstance(e, AgentReasoningDelta) for e in events)
