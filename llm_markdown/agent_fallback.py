"""Hybrid FALLBACK for :class:`~llm_markdown.reasoning.ReasoningMode.FALLBACK`.

**Tool-selection rounds** (non-empty ``tools`` and no ``tool`` messages in history):
two provider calls. Phase A has no tools, so the model must emit text; that
stream is parsed with :class:`ThinkTagParser` but **every** text chunk is
emitted as :class:`~llm_markdown.agent_stream.AgentReasoningDelta` (all internal).
The raw completion is injected into Phase B as hidden notes, then Phase B runs
with tools and is a **pass-through** stream (no tag parsing).

**Answer rounds** (no tools, or tool results already in history): a single
completion with a thinking-tag instruction; the parser splits streamed text so
that segments inside the think delimiters become ``AgentReasoningDelta`` and
segments outside become ``AgentContentDelta``.
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

_FALLBACK_PHASE_B_BRIDGE = (
    "The next assistant-role message is **hidden internal notes** from a reasoning step — "
    "the user never saw it and it is not part of the conversation.  Ignore its tone and phrasing.  "
    "Respond to the user's last message directly: call tools if needed, then write one clear reply.  "
    "Match the language of the **user's message** (not the notes)."
)

_PLAN_WRAPPER_PREFIX = "[Internal notes — not shown to user]\n"
_PLAN_WRAPPER_SUFFIX = "\n[End internal notes]"


class ThinkTagParser:
    """Streaming state machine that splits text on ``</think>`` boundaries.

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


def _inject_phase_b_bridge(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = deepcopy(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                out[i] = {
                    **m,
                    "content": c.rstrip() + "\n\n" + _FALLBACK_PHASE_B_BRIDGE,
                }
                return out
            break
    return [{"role": "system", "content": _FALLBACK_PHASE_B_BRIDGE}, *out]


def _has_tool_results(messages: list[dict[str, Any]]) -> bool:
    return any(m.get("role") == "tool" for m in messages)


def _needs_planning(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> bool:
    """True when a separate no-tools Phase A should run before the tool-capable Phase B."""
    if not tools:
        return False
    if _has_tool_results(messages):
        return False
    return True


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


def _yield_think_tag_split_stream(
    events: Iterator[AgentStreamEvent],
) -> Iterator[AgentStreamEvent]:
    """Map content through :class:`ThinkTagParser`; reasoning vs content by tag position."""
    parser = ThinkTagParser()
    for ev in events:
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


def _stream_phase_a_all_reasoning(
    provider: Any,
    backend: str,
    messages: list[dict[str, Any]],
    *,
    model: str,
    plan_cap: int,
    kwargs: dict[str, Any],
    plan_parts: list[str],
) -> Iterator[AgentStreamEvent]:
    """No-tools completion; every content chunk is emitted as ``AgentReasoningDelta``."""
    msgs = _inject_system_suffix(messages, _THINK_TAG_INSTRUCTION)
    extra = {k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")}
    stream = _stream_completion(
        provider,
        backend,
        msgs,
        model=model,
        tools=None,
        tool_choice=None,
        max_tokens=plan_cap,
        kwargs=extra,
    )
    parser = ThinkTagParser()
    for ev in stream:
        if isinstance(ev, AgentContentDelta):
            plan_parts.append(ev.text)
            for _role, chunk in parser.feed(ev.text):
                if chunk:
                    yield AgentReasoningDelta(text=chunk)
        elif isinstance(ev, AgentReasoningDelta):
            plan_parts.append(ev.text)
            yield ev
        elif isinstance(ev, AgentMessageFinish):
            for _role, chunk in parser.flush():
                if chunk:
                    yield AgentReasoningDelta(text=chunk)
            break
        else:
            yield ev


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
    """Hybrid FALLBACK: two-phase for tool selection, think-tags for answer rounds.

    See module docstring.  ``backend`` is ``openai``, ``openrouter``, or ``anthropic``.
    """
    if backend not in ("openai", "openrouter", "anthropic"):
        msg = f"FALLBACK not implemented for backend={backend!r}"
        raise NotImplementedError(msg)

    if not _needs_planning(messages, tools):
        msgs = _inject_system_suffix(messages, _THINK_TAG_INSTRUCTION)
        yield from _yield_think_tag_split_stream(
            _stream_completion(
                provider,
                backend,
                msgs,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                kwargs=kwargs,
            )
        )
        return

    plan_cap = planning_max_tokens if planning_max_tokens is not None else min(512, max_tokens or 1024)
    plan_parts: list[str] = []
    yield from _stream_phase_a_all_reasoning(
        provider,
        backend,
        messages,
        model=model,
        plan_cap=plan_cap,
        kwargs=kwargs,
        plan_parts=plan_parts,
    )

    plan_text = "".join(plan_parts).strip() or "(no planning text)"
    msgs_b = _inject_phase_b_bridge(deepcopy(messages))
    wrapped = _PLAN_WRAPPER_PREFIX + plan_text + _PLAN_WRAPPER_SUFFIX
    msgs_b.append({"role": "assistant", "content": wrapped})

    yield from _stream_completion(
        provider,
        backend,
        msgs_b,
        model=model,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        kwargs=kwargs,
    )


__all__ = ["stream_agent_turn_fallback", "ThinkTagParser"]
