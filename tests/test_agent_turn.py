"""Tests for stream_agent_turn, ReasoningConfig validation, and ThinkTagParser."""

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
from llm_markdown.agent_fallback import ThinkTagParser, stream_agent_turn_fallback
from llm_markdown.agent_turn import stream_agent_turn
from llm_markdown.providers.anthropic import AnthropicProvider
from llm_markdown.providers.openai import OpenAIProvider
from llm_markdown.reasoning import ReasoningConfig, ReasoningMode


# ---------------------------------------------------------------------------
# ReasoningConfig validation
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ThinkTagParser unit tests
# ---------------------------------------------------------------------------

class TestThinkTagParser:
    def test_no_tags_all_content(self):
        p = ThinkTagParser()
        assert p.feed("hello world") == [("content", "hello world")]
        assert p.flush() == []

    def test_basic_think_then_content(self):
        p = ThinkTagParser()
        out = p.feed("<think>reasoning here</think>visible reply")
        assert out == [("reasoning", "reasoning here"), ("content", "visible reply")]
        assert p.flush() == []

    def test_only_think_tags(self):
        p = ThinkTagParser()
        out = p.feed("<think>internal only</think>")
        assert out == [("reasoning", "internal only")]
        assert p.flush() == []

    def test_streaming_split_across_chunks(self):
        p = ThinkTagParser()
        results = []
        for chunk in ["<thi", "nk>reas", "oning</thi", "nk>answer"]:
            results.extend(p.feed(chunk))
        results.extend(p.flush())
        reasoning = "".join(t for r, t in results if r == "reasoning")
        content = "".join(t for r, t in results if r == "content")
        assert reasoning == "reasoning"
        assert content == "answer"

    def test_partial_open_tag_buffered(self):
        p = ThinkTagParser()
        assert p.feed("hi <thi") == [("content", "hi ")]
        assert p.feed("nk>inside</think>done") == [("reasoning", "inside"), ("content", "done")]
        assert p.flush() == []

    def test_partial_close_tag_buffered(self):
        p = ThinkTagParser()
        results = []
        results.extend(p.feed("<think>reason"))
        results.extend(p.feed("ing</thi"))
        results.extend(p.feed("nk>after"))
        results.extend(p.flush())
        reasoning = "".join(t for r, t in results if r == "reasoning")
        content = "".join(t for r, t in results if r == "content")
        assert reasoning == "reasoning"
        assert content == "after"

    def test_flush_inside_think(self):
        p = ThinkTagParser()
        results = p.feed("<think>unfinished")
        results.extend(p.flush())
        reasoning = "".join(t for r, t in results if r == "reasoning")
        assert reasoning == "unfinished"

    def test_multiple_think_sections(self):
        p = ThinkTagParser()
        out = p.feed("<think>a</think>b<think>c</think>d")
        assert out == [
            ("reasoning", "a"),
            ("content", "b"),
            ("reasoning", "c"),
            ("content", "d"),
        ]

    def test_empty_think_tags(self):
        p = ThinkTagParser()
        out = p.feed("<think></think>answer")
        assert out == [("content", "answer")]


# ---------------------------------------------------------------------------
# Fallback: single-completion <think> tag tests
# ---------------------------------------------------------------------------

_SIMPLE_TOOLS = [
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


def test_fallback_openai_hybrid_tool_round_two_calls():
    """Tool round: Phase A streams all text as reasoning; Phase B pass-through (content + tool)."""
    phase_a = iter([
        AgentContentDelta(text="The user wants addition; I will call add."),
        AgentMessageFinish(finish_reason="stop", usage=None),
    ])
    phase_b = iter([
        AgentContentDelta(text="Calling the tool now."),
        AgentToolCallDelta(index=0, tool_call_id="t1", name="add", arguments='{"a":1,"b":2}'),
        AgentMessageFinish(finish_reason="tool_calls", usage=None),
    ])
    provider = MagicMock()
    provider.stream_chat_completion_events.side_effect = [phase_a, phase_b]

    events = list(
        stream_agent_turn(
            provider, "openai",
            [{"role": "user", "content": "add 1+2"}],
            model="gpt-4o-mini",
            tools=_SIMPLE_TOOLS,
            reasoning=ReasoningConfig(mode=ReasoningMode.FALLBACK),
        )
    )
    reasoning = [e for e in events if isinstance(e, AgentReasoningDelta)]
    content = [e for e in events if isinstance(e, AgentContentDelta)]
    tools = [e for e in events if isinstance(e, AgentToolCallDelta)]
    assert any("addition" in e.text for e in reasoning)
    assert any("Calling the tool" in e.text for e in content)
    assert len(tools) == 1
    assert provider.stream_chat_completion_events.call_count == 2


def test_fallback_anthropic_hybrid_tool_round_two_calls():
    """Anthropic: two calls for tool round; Phase A all reasoning, Phase B split path."""
    phase_a = iter([
        AgentContentDelta(text="User said hi; no tool needed yet in plan."),
        AgentMessageFinish(finish_reason="end_turn", usage=None),
    ])
    phase_b = iter([
        AgentContentDelta(text="Hello!"),
        AgentMessageFinish(finish_reason="end_turn", usage=None),
    ])
    provider = MagicMock()
    provider.stream_messages_events.side_effect = [phase_a, phase_b]

    events = list(
        stream_agent_turn(
            provider, "anthropic",
            [{"role": "user", "content": "hi"}],
            model="claude-3-5-haiku-latest",
            tools=_SIMPLE_TOOLS,
            reasoning=ReasoningConfig(mode=ReasoningMode.FALLBACK),
        )
    )
    reasoning = [e for e in events if isinstance(e, AgentReasoningDelta)]
    content = [e for e in events if isinstance(e, AgentContentDelta)]
    assert any("User said hi" in e.text for e in reasoning)
    assert any("Hello!" in e.text for e in content)
    assert provider.stream_messages_events.call_count == 2


def test_fallback_no_think_tags_all_content():
    """If model doesn't produce <think> tags, everything is AgentContentDelta."""
    completion = iter([
        AgentContentDelta(text="Just a plain answer"),
        AgentMessageFinish(finish_reason="stop", usage=None),
    ])
    provider = MagicMock()
    provider.stream_chat_completion_events.return_value = completion

    events = list(
        stream_agent_turn_fallback(
            provider, "openai",
            [{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )
    )
    assert any(isinstance(e, AgentContentDelta) and "plain answer" in e.text for e in events)
    assert not any(isinstance(e, AgentReasoningDelta) for e in events)


def test_fallback_streaming_think_tag_across_chunks():
    """<think> tag split across multiple AgentContentDelta chunks."""
    completion = iter([
        AgentContentDelta(text="<thi"),
        AgentContentDelta(text="nk>internal"),
        AgentContentDelta(text="</think>"),
        AgentContentDelta(text="visible"),
        AgentMessageFinish(finish_reason="stop", usage=None),
    ])
    provider = MagicMock()
    provider.stream_chat_completion_events.return_value = completion

    events = list(
        stream_agent_turn_fallback(
            provider, "openai",
            [{"role": "user", "content": "test"}],
            model="gpt-4o-mini",
        )
    )
    reasoning_text = "".join(e.text for e in events if isinstance(e, AgentReasoningDelta))
    content_text = "".join(e.text for e in events if isinstance(e, AgentContentDelta))
    assert reasoning_text == "internal"
    assert content_text == "visible"


def test_fallback_native_reasoning_forwarded():
    """Provider-native AgentReasoningDelta (e.g. extended thinking) is forwarded unchanged."""
    completion = iter([
        AgentReasoningDelta(text="native thinking"),
        AgentContentDelta(text="<think>tag thinking</think>reply"),
        AgentMessageFinish(finish_reason="stop", usage=None),
    ])
    provider = MagicMock()
    provider.stream_chat_completion_events.return_value = completion

    events = list(
        stream_agent_turn_fallback(
            provider, "openai",
            [{"role": "user", "content": "x"}],
            model="gpt-4o-mini",
        )
    )
    reasoning = [e for e in events if isinstance(e, AgentReasoningDelta)]
    assert any(e.text == "native thinking" for e in reasoning)
    assert any(e.text == "tag thinking" for e in reasoning)
    content = [e for e in events if isinstance(e, AgentContentDelta)]
    assert any("reply" in e.text for e in content)


def test_fallback_think_instruction_injected_into_system():
    """The think-tag instruction is appended to the system message."""
    completion = iter([
        AgentContentDelta(text="ok"),
        AgentMessageFinish(finish_reason="stop", usage=None),
    ])
    provider = MagicMock()
    provider.stream_chat_completion_events.return_value = completion

    list(
        stream_agent_turn_fallback(
            provider, "openai",
            [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )
    )
    called_msgs = provider.stream_chat_completion_events.call_args[0][0]
    sys_content = next(m["content"] for m in called_msgs if m["role"] == "system")
    assert "Be helpful." in sys_content
    assert "<think>" in sys_content
    assert "reasoning" in sys_content.lower()


def test_fallback_think_instruction_added_when_no_system():
    """When no system message exists, a new one with the instruction is prepended."""
    completion = iter([
        AgentContentDelta(text="ok"),
        AgentMessageFinish(finish_reason="stop", usage=None),
    ])
    provider = MagicMock()
    provider.stream_chat_completion_events.return_value = completion

    list(
        stream_agent_turn_fallback(
            provider, "openai",
            [{"role": "user", "content": "hi"}],
            model="gpt-4o-mini",
        )
    )
    called_msgs = provider.stream_chat_completion_events.call_args[0][0]
    assert called_msgs[0]["role"] == "system"
    assert "<think>" in called_msgs[0]["content"]


def test_fallback_post_tool_history_single_call_think_split():
    """History with tool results → one completion; think tags split reasoning vs content."""
    completion = iter([
        AgentContentDelta(text="<think>Total is 10 from stats.</think>You have 10 expenses."),
        AgentMessageFinish(finish_reason="stop", usage=None),
    ])
    provider = MagicMock()
    provider.stream_chat_completion_events.return_value = completion
    msgs = [
        {"role": "user", "content": "how many?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_stats", "arguments": "{}"}},
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
    reasoning = "".join(e.text for e in events if isinstance(e, AgentReasoningDelta))
    content = "".join(e.text for e in events if isinstance(e, AgentContentDelta))
    assert "Total is 10" in reasoning
    assert "10 expenses" in content


def test_fallback_phase_b_bridge_and_wrapped_plan():
    """Tool round: Phase B system gets bridge; last message is wrapped internal notes."""
    tools = [
        {"type": "function", "function": {"name": "greet", "parameters": {"type": "object", "properties": {}}}},
    ]
    phase_a = iter([
        AgentContentDelta(text="internal plan text"),
        AgentMessageFinish(finish_reason="stop", usage=None),
    ])
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
    assert "internal plan text" in plan_msg["content"]
    assert plan_msg["content"].endswith("[End internal notes]")


# ---------------------------------------------------------------------------
# Agentic segment markers
# ---------------------------------------------------------------------------

def test_fallback_agentic_segments_with_think_tags():
    """With tools: Phase A reasoning, then content segment before tool in Phase B."""
    phase_a = iter([
        AgentContentDelta(text="I need the add tool."),
        AgentMessageFinish(finish_reason="stop", usage=None),
    ])
    phase_b = iter([
        AgentToolCallDelta(index=0, tool_call_id="t1", name="add", arguments="{}"),
        AgentMessageFinish(finish_reason="tool_calls", usage=None),
    ])
    provider = MagicMock()
    provider.stream_chat_completion_events.side_effect = [phase_a, phase_b]

    events = list(
        stream_agent_turn(
            provider, "openai",
            [{"role": "user", "content": "x"}],
            model="gpt-4o-mini",
            tools=_SIMPLE_TOOLS,
            reasoning=ReasoningConfig(mode=ReasoningMode.FALLBACK),
        )
    )
    assert isinstance(events[0], AgentSegmentStart)
    assert events[0].segment == "reasoning"
    idx_tool = next(i for i, e in enumerate(events) if isinstance(e, AgentToolCallDelta))
    idx_content_seg = next(
        i for i, e in enumerate(events) if isinstance(e, AgentSegmentStart) and e.segment == "content"
    )
    assert idx_content_seg < idx_tool


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


# ---------------------------------------------------------------------------
# Native mode (non-fallback)
# ---------------------------------------------------------------------------

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
