"""Unit tests for OpenAIProvider.stream_chat_completion_events (mocked HTTP)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_markdown.agent_stream import (
    AgentMessageFinish,
    AgentReasoningDelta,
    AgentToolCallDelta,
    openai_chat_tools_to_anthropic,
)
from llm_markdown.providers.openai import OpenAIProvider, _delta_content_text, _delta_reasoning_text


def test_delta_content_text_list_parts():
    d = SimpleNamespace(
        content=[
            {"type": "text", "text": "hello"},
            {"type": "text", "text": " world"},
        ],
    )
    assert _delta_content_text(d) == "hello world"


def test_delta_reasoning_text():
    d = SimpleNamespace(reasoning_content="a", reasoning=None)
    assert _delta_reasoning_text(d) == "a"
    d2 = SimpleNamespace(reasoning="b")
    assert _delta_reasoning_text(d2) == "b"
    assert _delta_reasoning_text(SimpleNamespace()) is None


def test_openai_chat_tools_to_anthropic():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "fn_x",
                "description": "d",
                "parameters": {"type": "object", "properties": {"a": {"type": "string"}}},
            },
        }
    ]
    out = openai_chat_tools_to_anthropic(tools)
    assert len(out) == 1
    assert out[0]["name"] == "fn_x"
    assert out[0]["description"] == "d"
    assert out[0]["input_schema"]["type"] == "object"


def _choice_delta(content=None, tool_calls=None, reasoning_content=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning_content is not None:
        delta.reasoning_content = reasoning_content
    return SimpleNamespace(delta=delta, finish_reason=None)


def _chunk(choice, usage=None, cid="c1"):
    ch = SimpleNamespace(choices=[choice] if choice is not None else [])
    if usage is not None:
        ch.usage = usage
    ch.id = cid
    return ch


def test_stream_chat_completion_events_text_and_finish():
    chunks = [
        _chunk(_choice_delta("Hello")),
        _chunk(_choice_delta(" world")),
        _chunk(
            SimpleNamespace(
                delta=SimpleNamespace(content=None),
                finish_reason="stop",
            ),
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=2,
                total_tokens=3,
            ),
        ),
    ]

    provider = OpenAIProvider(api_key="k", model="gpt-4o-mini")

    mock_stream = iter(chunks)
    mock_create = MagicMock(return_value=mock_stream)

    with patch.object(provider.client.chat.completions, "create", mock_create):
        events = list(provider.stream_chat_completion_events([{"role": "user", "content": "hi"}]))

    assert [e.kind for e in events] == ["content_delta", "content_delta", "message_finish"]
    assert events[0].text == "Hello"
    assert events[1].text == " world"
    fin: AgentMessageFinish = events[-1]
    assert fin.finish_reason == "stop"
    assert fin.usage == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }


def test_stream_chat_completion_events_reasoning_and_tools():
    tc0 = SimpleNamespace(index=0, id="t1", function=SimpleNamespace(name="foo", arguments='{"a":'))
    tc1 = SimpleNamespace(index=0, id=None, function=SimpleNamespace(name="", arguments='1}'))
    chunks = [
        _chunk(_choice_delta(None, reasoning_content="think")),
        _chunk(_choice_delta(None, tool_calls=[tc0])),
        _chunk(_choice_delta(None, tool_calls=[tc1])),
        _chunk(
            SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=None),
                finish_reason="tool_calls",
            ),
        ),
    ]

    provider = OpenAIProvider(api_key="k", model="gpt-4o-mini")
    mock_create = MagicMock(return_value=iter(chunks))

    with patch.object(provider.client.chat.completions, "create", mock_create):
        events = list(
            provider.stream_chat_completion_events(
                [{"role": "user", "content": "x"}],
                tools=[{"type": "function", "function": {"name": "foo", "parameters": {}}}],
            )
        )

    kinds = [e.kind for e in events]
    assert kinds[:-1].count("tool_call_delta") == 2
    assert kinds[-1] == "message_finish"
    assert any(isinstance(e, AgentReasoningDelta) and e.text == "think" for e in events)
    t0 = next(e for e in events if isinstance(e, AgentToolCallDelta) and e.arguments == '{"a":')
    assert t0.tool_call_id == "t1"
    assert t0.name == "foo"


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-4o-mini", "max_tokens"),
        ("gpt-5", "max_completion_tokens"),
    ],
)
def test_stream_passes_token_param(model, expected):
    provider = OpenAIProvider(api_key="k", model=model)
    captured: dict = {}

    def _fake_create(**kw):
        captured.update(kw)
        return iter([])

    with patch.object(provider.client.chat.completions, "create", _fake_create):
        list(provider.stream_chat_completion_events([{"role": "user", "content": "z"}]))

    assert expected in captured


def test_stream_finish_reason_preserved_when_usage_chunk_has_no_choices():
    """Some streams emit a final usage-only chunk with choices=[]; keep prior finish_reason."""
    tc0 = SimpleNamespace(index=0, id="c1", function=SimpleNamespace(name="foo", arguments="{}"))
    chunks = [
        _chunk(_choice_delta(None, tool_calls=[tc0])),
        _chunk(
            SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=None),
                finish_reason="tool_calls",
            ),
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
            ),
        ),
        _chunk(None, usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12)),
    ]

    provider = OpenAIProvider(api_key="k", model="gpt-4o-mini")
    mock_create = MagicMock(return_value=iter(chunks))

    with patch.object(provider.client.chat.completions, "create", mock_create):
        events = list(
            provider.stream_chat_completion_events(
                [{"role": "user", "content": "x"}],
                tools=[{"type": "function", "function": {"name": "foo", "parameters": {}}}],
            )
        )

    fin: AgentMessageFinish = events[-1]
    assert fin.finish_reason == "tool_calls"
    assert fin.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }
