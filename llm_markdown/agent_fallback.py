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

# ---------------------------------------------------------------------------
# Phase A: metacognitive analysis (no tools).
#
# The key insight: asking a model to "write a plan" for a trivial query like
# "hi" produces answer-shaped text ("Hello! How can I assist you?").  Asking
# it to *analyze* the request in a structured format produces genuinely
# different text ("The user is greeting me.  No tools needed.").
# ---------------------------------------------------------------------------
_FALLBACK_PLANNING_APPENDIX = (
    "[Hidden reasoning step — your output here is never shown to the user.]\n"
    "Analyze the user's latest message. Write short internal notes, not a reply:\n"
    "- Intent: what is the user asking or doing?\n"
    "- Tools: which tool(s) will you call, or \"none\"?\n"
    "- Language: what language is the user writing in?\n"
    "- Approach: one sentence on how you will respond.\n"
    "Do not greet, answer, or address the user here. Do not output tool calls or JSON."
)

# ---------------------------------------------------------------------------
# Phase B bridge: injected into the system prompt so the model knows the
# assistant turn that follows is scratchpad, not prior output.
# ---------------------------------------------------------------------------
_FALLBACK_PHASE_B_BRIDGE = (
    "The next assistant-role message is **hidden internal notes** from a reasoning step — "
    "the user never saw it and it is not part of the conversation.  Ignore its tone and phrasing.  "
    "Respond to the user's last message directly: call tools if needed, then write one clear reply.  "
    "Match the language of the **user's message** (not the notes)."
)

_PLAN_WRAPPER_PREFIX = "[Internal notes — not shown to user]\n"
_PLAN_WRAPPER_SUFFIX = "\n[End internal notes]"


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


def _has_tool_results(messages: list[dict[str, Any]]) -> bool:
    """True when the conversation already contains tool-result messages (post-tool round)."""
    return any(m.get("role") == "tool" for m in messages)


def _needs_planning(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> bool:
    """Decide whether Phase A (metacognitive planning) adds value for this round.

    Skip when:
    - No tools are provided (nothing to plan).
    - Tool results already exist in the message history — the model already
      called tools and just needs to answer from results.  Running a second
      two-phase turn here doubles latency and produces duplicate content
      (the "plan" IS the answer, then the answer streams again).
    """
    if not tools:
        return False
    if _has_tool_results(messages):
        return False
    return True


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

    Phase A is **skipped** when ``tools`` is empty/``None`` or when the message
    history already contains tool results (post-tool summary rounds).  In those
    cases the turn falls through to a single direct completion — same latency
    as native mode.
    """
    if backend not in ("openai", "openrouter", "anthropic"):
        msg = f"FALLBACK not implemented for backend={backend!r}"
        raise NotImplementedError(msg)

    if not _needs_planning(messages, tools):
        yield from _phase_b_events(
            provider,
            backend,
            messages,
            model=model,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            kwargs=kwargs,
        )
        return

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
    msgs_b = _inject_phase_b_bridge(deepcopy(messages))
    wrapped = _PLAN_WRAPPER_PREFIX + plan_text + _PLAN_WRAPPER_SUFFIX
    msgs_b.append({"role": "assistant", "content": wrapped})

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
