"""Hybrid FALLBACK for :class:`~llm_markdown.reasoning.ReasoningMode.FALLBACK`.

**Tool-selection rounds** (non-empty ``tools`` and no ``tool`` messages in history):
two provider calls. Phase A has no tools, so the model must emit text; that
emits :class:`~llm_markdown.agent_stream.AgentReasoningDelta` only (no think-tag
split; instruction elicits structured intent/tools lines, not a user-facing reply).
The planning text is injected into Phase B as hidden notes (OpenAI-style
``assistant`` prefill; **Anthropic** uses a synthetic final ``user`` message —
Claude rejects assistant prefill on some models), then Phase B runs with tools
and parsed through :class:`ThinkTagParser` (think tags in content split into
reasoning vs reply; tool deltas unchanged).

**Answer rounds** (no tools, or tool results already in history): a single
completion with a thinking-tag instruction; the parser splits streamed text so
that segments inside the think delimiters become ``AgentReasoningDelta`` and
segments outside become ``AgentContentDelta``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
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

_PHASE_A_INSTRUCTION = (
    "[Internal reasoning — never shown to the user]\n"
    "Reply with exactly 3–5 lines in this format only (no other prose, no preamble):\n"
    "- intent: …\n"
    "- language: …\n"
    "- tools: … (none or comma-separated names)\n"
    "- plan: … (one short clause)\n"
    "Forbidden: greetings, offers to help, addressing the user, emoji, or any text that "
    "could appear as a chat reply. If you write anything user-facing, the step fails."
)

_THINK_TAG_INSTRUCTION = (
    "You have a private reasoning step. Before responding, write brief "
    f"analytical notes inside {_OPEN_TAG}...{_CLOSE_TAG} tags.\n\n"
    "RULES:\n"
    f"1. Between {_OPEN_TAG} and {_CLOSE_TAG}: ONLY plain-text analytical notes.\n"
    "   State what the user wants, your observations about data, your approach — 1-3 short sentences.\n"
    "2. FORBIDDEN inside the tags: markdown tables, bullet lists, numbered lists, headings, "
    "HTML tags (<details>, <summary>, etc.), code blocks, bold/italic formatting, emoji, "
    "and any text that could serve as a reply to the user.\n"
    f"3. Your text inside the tags and your response after {_CLOSE_TAG} "
    "must be completely different content.\n\n"
    f"After {_CLOSE_TAG}, write your complete response to the user.\n"
    "Always reply in the same language as the user's message."
)

_FALLBACK_PHASE_B_BRIDGE = (
    "The conversation includes **hidden internal planning notes** (a synthetic message the end user "
    "never saw — assistant-role on OpenAI-compatible APIs, or a labeled user-role block on Anthropic).  "
    "Ignore that block's tone and phrasing.  Respond to the user's actual last message: call tools "
    "if needed, then write one clear reply.  Match the language of the **user's message** (not the notes)."
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
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[AgentStreamEvent]:
    """Map content through :class:`ThinkTagParser`; reasoning vs content by tag position."""
    parser = ThinkTagParser()
    for ev in events:
        if should_cancel and should_cancel():
            yield AgentMessageFinish(finish_reason="cancelled", usage=None)
            return
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
    should_cancel: Callable[[], bool] | None = None,
) -> Iterator[AgentStreamEvent]:
    """No-tools completion; every text chunk is emitted as ``AgentReasoningDelta``."""
    msgs = _inject_system_suffix(messages, _PHASE_A_INSTRUCTION)
    extra = {k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")}
    if should_cancel is not None:
        extra["should_cancel"] = should_cancel
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
    for ev in stream:
        if should_cancel and should_cancel():
            yield AgentMessageFinish(finish_reason="cancelled", usage=None)
            return
        if isinstance(ev, AgentContentDelta):
            plan_parts.append(ev.text)
            yield AgentReasoningDelta(text=ev.text)
        elif isinstance(ev, AgentReasoningDelta):
            plan_parts.append(ev.text)
            yield ev
        elif isinstance(ev, AgentMessageFinish):
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
    should_cancel: Callable[[], bool] | None = None,
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
        fb_kw = dict(kwargs)
        if should_cancel is not None:
            fb_kw["should_cancel"] = should_cancel
        yield from _yield_think_tag_split_stream(
            _stream_completion(
                provider,
                backend,
                msgs,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                kwargs=fb_kw,
            ),
            should_cancel=should_cancel,
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
        should_cancel=should_cancel,
    )

    if should_cancel and should_cancel():
        yield AgentMessageFinish(finish_reason="cancelled", usage=None)
        return

    plan_text = "".join(plan_parts).strip() or "(no planning text)"
    msgs_b = _inject_phase_b_bridge(deepcopy(messages))
    msgs_b = _inject_system_suffix(msgs_b, _THINK_TAG_INSTRUCTION)
    wrapped = _PLAN_WRAPPER_PREFIX + plan_text + _PLAN_WRAPPER_SUFFIX
    # Anthropic Messages API: trailing plain assistant text is "prefill"; several Claude models reject it
    # (conversation must end with a user turn). OpenAI chat accepts assistant prefill here.
    if backend == "anthropic":
        msgs_b.append({"role": "user", "content": wrapped})
    else:
        msgs_b.append({"role": "assistant", "content": wrapped})

    fb_kw_b = dict(kwargs)
    if should_cancel is not None:
        fb_kw_b["should_cancel"] = should_cancel
    yield from _yield_think_tag_split_stream(
        _stream_completion(
            provider,
            backend,
            msgs_b,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            kwargs=fb_kw_b,
        ),
        should_cancel=should_cancel,
    )


__all__ = ["stream_agent_turn_fallback", "ThinkTagParser"]
