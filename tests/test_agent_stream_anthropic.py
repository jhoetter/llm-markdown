"""Unit tests for AnthropicProvider.stream_messages_events (mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from llm_markdown.agent_stream import (
    AgentContentDelta,
    AgentMessageFinish,
    AgentToolCallDelta,
)
from llm_markdown.providers.anthropic import AnthropicProvider


def test_stream_messages_events_text_and_stop():
    events = [
        SimpleNamespace(type="text", text="Hi"),
        SimpleNamespace(
            type="message_stop",
            message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=3, output_tokens=4),
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
                id="msg-1",
                usage=SimpleNamespace(input_tokens=3, output_tokens=4),
                stop_reason="end_turn",
            )

        def close(self):
            pass

    provider = AnthropicProvider(api_key="k", model="claude-3-5-haiku-latest")
    with patch.object(provider.client.messages, "stream", return_value=CM()):
        out = list(
            provider.stream_messages_events([{"role": "user", "content": "hello"}])
        )

    assert isinstance(out[0], AgentContentDelta)
    assert out[0].text == "Hi"
    assert isinstance(out[1], AgentMessageFinish)
    assert out[1].finish_reason == "end_turn"
    assert out[1].usage["prompt_tokens"] == 3


def test_stream_messages_events_tool_use_finish():
    events = [
        SimpleNamespace(
            type="content_block_start",
            index=0,
            content_block=SimpleNamespace(type="tool_use", id="tu_1", name="my_tool"),
        ),
        SimpleNamespace(type="input_json", partial_json='{"x":1}'),
        SimpleNamespace(
            type="message_stop",
            message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
                stop_reason="tool_use",
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
                id="m2",
                usage=SimpleNamespace(input_tokens=1, output_tokens=2),
                stop_reason="tool_use",
            )

        def close(self):
            pass

    provider = AnthropicProvider(api_key="k", model="claude-3-5-haiku-latest")
    with patch.object(provider.client.messages, "stream", return_value=CM()):
        out = list(
            provider.stream_messages_events(
                [{"role": "user", "content": "go"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "my_tool",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
        )

    assert out[-1].finish_reason == "tool_calls"
    deltas = [x for x in out if isinstance(x, AgentToolCallDelta)]
    assert any(x.name == "my_tool" for x in deltas)
    assert any(x.arguments == '{"x":1}' for x in deltas)


def test_to_anthropic_messages_openai_tool_thread():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [
                {
                    "id": "tu_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tu_1", "content": '{"result": 3}'},
    ]
    system, out = AnthropicProvider._to_anthropic_messages(messages)
    assert system == ""
    assert len(out) == 3
    assert out[0]["role"] == "user"
    assert out[1]["role"] == "assistant"
    blocks = out[1]["content"]
    assert blocks[0] == {"type": "text", "text": "calling tool"}
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["id"] == "tu_1"
    assert blocks[1]["name"] == "add"
    assert blocks[1]["input"] == {"a": 1, "b": 2}
    assert out[2]["role"] == "user"
    assert out[2]["content"][0]["type"] == "tool_result"
    assert out[2]["content"][0]["tool_use_id"] == "tu_1"
