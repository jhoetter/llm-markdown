"""Single-completion fallback with ``<think>`` tags for :class:`~llm_markdown.reasoning.ReasoningMode.FALLBACK`.

Instead of two API calls (plan then answer), injects a thinking-tag instruction
into the system prompt so the model writes ``<think>reasoning</think>answer`` in
one stream.  A streaming parser splits the content into ``AgentReasoningDelta``
(inside tags) and ``AgentContentDelta`` (outside tags).  Tool calls flow through
unchanged because they are a separate field on the provider delta.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

from llm_markdown.agent_stream import (
    AgentContentDelta,
    AgentMessageFinish,
    AgentReasoningDelta,
    AgentStreamEvent,
    AgentToolCallDelta,
)

_OPEN_TAG = "<think>"
_CLOSE_TAG = "</think>"

_THINK_TAG_INSTRUCTION = (
    "Before responding, think step-by-step inside <think>...</think> tags. "
    "Your thinking should be internal reasoning: what the user wants, which tools (if any) "
    "to call, what language they are writing in, and how you will approach it. "
    "After the closing </think> tag, write your actual response to the user. "
    "Always reply in the same language as the user's message."
)


class ThinkTagParser:
    """Streaming state machine that splits text on ``<think>``/``</think>`` boundaries.

    Feed chunks of text as they arrive from the provider.  The parser yields
    ``(role, text)`` pairs where *role* is ``"reasoning"`` (inside tags) or
    ``"content"`` (outside tags).  Handles partial tags split across chunks.
    """

    __slots__ = ("_inside", "_buf")

    def __init__(self) -> None:
        self._inside = False
        self._buf = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        self._buf += text
        out: list[tuple[str, str]] = []
        while self._buf:
            if self._inside:
                idx = self._buf.find(_CLOSE_TAG)
                if idx >= 0:
                    if idx > 0:
                        out.append(("reasoning", self._buf[:idx]))
                    self._buf = self._buf[idx + len(_CLOSE_TAG) :]
                    self._inside = False
                else:
                    safe = self._flush_safe(_CLOSE_TAG)
                    if safe:
                        out.append(("reasoning", safe))
                    break
            else:
                idx = self._buf.find(_OPEN_TAG)
                if idx >= 0:
                    if idx > 0:
                        out.append(("content", self._buf[:idx]))
                    self._buf = self._buf[idx + len(_OPEN_TAG) :]
                    self._inside = True
                else:
                    safe = self._flush_safe(_OPEN_TAG)
                    if safe:
                        out.append(("content", safe))
                    break
        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self._buf:
            return []
        role = "reasoning" if self._inside else "content"
        result = [(role, self._buf)]
        self._buf = ""
        return result

    def _flush_safe(self, tag: str) -> str:
        """Flush buffer up to the point where a partial tag might start at the end."""
        for i in range(1, min(len(tag), len(self._buf)) + 1):
            if tag.startswith(self._buf[-i:]):
                safe = self._buf[: -i]
                self._buf = self._buf[-i:]
                return safe
        safe = self._buf
        self._buf = ""
        return safe


def _inject_system_suffix(
    messages: list[dict[str, Any]],
    suffix: str,
) -> list[dict[str, Any]]:
    """Append *suffix* to the first system message (or prepend a new one)."""
    out = deepcopy(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                out[i] = {**m, "content": c.rstrip() + "\n\n" + suffix}
                return out
            break
    return [{"role": "system", "content": suffix}, *out]


def _stream_completion(
    provider: Any,
    backend: str,
    msgs: list[dict[str, Any]],
    *,
    model: str,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    max_tokens: int | None,
    kwargs: dict[str, Any],
) -> Iterator[AgentStreamEvent]:
    call_kw = dict(kwargs)
    call_kw["model"] = model
    if max_tokens is not None:
        call_kw["max_tokens"] = max_tokens
    if tools is not None:
        call_kw["tools"] = tools
    if tool_choice is not None:
        call_kw["tool_choice"] = tool_choice

    if backend in ("openai", "openrouter"):
        call_kw.pop("thinking", None)
        yield from provider.stream_chat_completion_events(msgs, **call_kw)
    elif backend == "anthropic":
        call_kw.pop("thinking", None)
        yield from provider.stream_messages_events(msgs, **call_kw)
    else:
        msg = f"unknown backend: {backend!r}"
        raise ValueError(msg)


def stream_agent_turn_fallback(
    provider: Any,
    backend: str,
    messages: list[dict[str, Any]],
    *,
    model: str,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = "auto",
    max_tokens: int | None = None,
    planning_max_tokens: int | None = None,
    **kwargs: Any,
) -> Iterator[AgentStreamEvent]:
    """Single completion with ``<think>`` tag parsing for reasoning/content separation.

    The system prompt is augmented with a thinking-tag instruction.  Content
    inside ``<think>...</think>`` tags is emitted as ``AgentReasoningDelta``;
    content outside is emitted as ``AgentContentDelta``.  Tool calls flow
    through unchanged.  Provider-native ``AgentReasoningDelta`` (e.g. from
    a model with real extended thinking) is also forwarded.

    ``backend`` is ``openai``, ``openrouter`` (OpenAI-compatible), or ``anthropic``.
    """
    if backend not in ("openai", "openrouter", "anthropic"):
        msg = f"FALLBACK not implemented for backend={backend!r}"
        raise NotImplementedError(msg)

    msgs = _inject_system_suffix(messages, _THINK_TAG_INSTRUCTION)
    parser = ThinkTagParser()

    for ev in _stream_completion(
        provider,
        backend,
        msgs,
        model=model,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        kwargs=kwargs,
    ):
        if isinstance(ev, AgentContentDelta):
            for role, text in parser.feed(ev.text):
                if role == "reasoning":
                    yield AgentReasoningDelta(text=text)
                else:
                    yield AgentContentDelta(text=text)
        elif isinstance(ev, AgentReasoningDelta):
            yield ev
        elif isinstance(ev, AgentToolCallDelta):
            yield ev
        elif isinstance(ev, AgentMessageFinish):
            for role, text in parser.flush():
                if role == "reasoning":
                    yield AgentReasoningDelta(text=text)
                else:
                    yield AgentContentDelta(text=text)
            yield ev
        else:
            yield ev


__all__ = ["stream_agent_turn_fallback", "ThinkTagParser"]
