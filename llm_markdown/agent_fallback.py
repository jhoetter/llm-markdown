"""Two-phase planning + tools stream for :class:`~llm_markdown.reasoning.ReasoningMode.FALLBACK`."""

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

# Appended to system prompt (or new system message) for phase A only.
_FALLBACK_PLANNING_APPENDIX = (
    "[Planning phase — no tools yet] Write a brief plain-text plan: what the user wants, "
    "which tool you will use next (if any), and why. A few short sentences. "
    "Do not output tool calls or JSON in this turn."
)


def _inject_planning_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = deepcopy(messages)
    for i, m in enumerate(out):
        if m.get("role") == "system":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                out[i] = {
                    **m,
                    "content": c.rstrip() + "\n\n" + _FALLBACK_PLANNING_APPENDIX,
                }
                return out
            break
    return [{"role": "system", "content": _FALLBACK_PLANNING_APPENDIX}, *out]


def _phase_b_events(
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
    """Tool-capable turn; forwards provider stream including ``AgentReasoningDelta``."""
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
        raw = provider.stream_chat_completion_events(msgs, **call_kw)
    elif backend == "anthropic":
        call_kw.pop("thinking", None)
        raw = provider.stream_messages_events(msgs, **call_kw)
    else:
        msg = f"unknown backend: {backend!r}"
        raise ValueError(msg)

    yield from raw


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
    """Phase A: planning stream (no tools) → ``AgentReasoningDelta``; Phase B: tools + pass-through stream.

    ``backend`` is ``openai``, ``openrouter`` (OpenAI-compatible), or ``anthropic``.
    """
    if backend not in ("openai", "openrouter", "anthropic"):
        msg = f"FALLBACK not implemented for backend={backend!r}"
        raise NotImplementedError(msg)

    plan_cap = planning_max_tokens if planning_max_tokens is not None else min(512, max_tokens or 1024)
    plan_msgs = _inject_planning_system(messages)
    extra = {k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")}

    if backend in ("openai", "openrouter"):
        phase_a = provider.stream_chat_completion_events(
            plan_msgs,
            model=model,
            max_tokens=plan_cap,
            **extra,
        )
    else:
        phase_a = provider.stream_messages_events(
            plan_msgs,
            model=model,
            max_tokens=plan_cap,
            **extra,
        )

    plan_parts: list[str] = []
    for ev in phase_a:
        if isinstance(ev, AgentContentDelta):
            plan_parts.append(ev.text)
            yield AgentReasoningDelta(text=ev.text)
        elif isinstance(ev, AgentReasoningDelta):
            plan_parts.append(ev.text)
            yield ev
        elif isinstance(ev, AgentToolCallDelta):
            continue
        elif isinstance(ev, AgentMessageFinish):
            break
        else:
            yield ev

    plan_text = "".join(plan_parts).strip() or "(no planning text)"
    msgs_b = deepcopy(messages)
    msgs_b.append({"role": "assistant", "content": plan_text})

    yield from _phase_b_events(
        provider,
        backend,
        msgs_b,
        model=model,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        kwargs=kwargs,
    )


__all__ = ["stream_agent_turn_fallback", "_inject_planning_system"]
