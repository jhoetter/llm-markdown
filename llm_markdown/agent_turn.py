"""Single entry point for one agent model turn with configurable reasoning."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from llm_markdown.agent_fallback import stream_agent_turn_fallback
from llm_markdown.agent_stream import (
    AgentContentDelta,
    AgentMessageFinish,
    AgentReasoningDelta,
    AgentSegmentStart,
    AgentStreamEvent,
    AgentToolCallDelta,
)
from llm_markdown.reasoning import BackendName, ReasoningConfig, ReasoningMode


def _filter_reasoning_off(events: Iterator[AgentStreamEvent]) -> Iterator[AgentStreamEvent]:
    for ev in events:
        if isinstance(ev, AgentReasoningDelta):
            continue
        yield ev


def _is_agentic_turn(tools: list[dict[str, Any]] | None) -> bool:
    return bool(tools)


def _apply_agentic_segment_markers(
    events: Iterator[AgentStreamEvent],
    *,
    agentic: bool,
) -> Iterator[AgentStreamEvent]:
    """Insert :class:`AgentSegmentStart` for non-empty ``tools`` turns.

    Opens with ``reasoning``; first :class:`AgentContentDelta` or
    :class:`AgentToolCallDelta` is preceded by ``content``. Reasoning deltas
    stay in the reasoning segment until that transition.
    """
    if not agentic:
        yield from events
        return
    yield AgentSegmentStart(segment="reasoning")
    lane: str = "reasoning"
    for ev in events:
        if isinstance(ev, AgentReasoningDelta):
            yield ev
        elif isinstance(ev, (AgentContentDelta, AgentToolCallDelta)):
            if lane == "reasoning":
                yield AgentSegmentStart(segment="content")
                lane = "content"
            yield ev
        elif isinstance(ev, AgentMessageFinish):
            yield ev
        elif isinstance(ev, AgentSegmentStart):
            yield ev
        else:
            yield ev


def stream_agent_turn(
    provider: Any,
    backend: BackendName,
    messages: list[dict[str, Any]],
    *,
    model: str,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = "auto",
    max_tokens: int | None = None,
    reasoning: ReasoningConfig | None = None,
    planning_max_tokens: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
    **kwargs: Any,
) -> Iterator[AgentStreamEvent]:
    """Stream one tool-capable model turn as :class:`~llm_markdown.agent_stream.AgentStreamEvent`.

    ``should_cancel``: if set, called frequently while streaming; when it returns True the
    turn ends with ``AgentMessageFinish(finish_reason="cancelled")`` and provider streams are
    closed where supported.

    Dispatches to:

    - ``backend="openai"`` or ``"openrouter"`` →
      :meth:`~llm_markdown.providers.openai.OpenAIProvider.stream_chat_completion_events`
    - ``backend="anthropic"`` →
      :meth:`~llm_markdown.providers.anthropic.AnthropicProvider.stream_messages_events`
    - ``backend="gemini"`` → not implemented (raises ``ValueError`` from validation).

    ``openrouter`` uses the same client/stream shape as OpenAI.

    ``reasoning.mode``:

    - ``native`` — forward provider-native reasoning/thinking when the API emits it.
    - ``off`` — filter out ``AgentReasoningDelta``; do not request Anthropic extended thinking.
    - ``fallback`` — single completion with ``<think>`` tag parsing
      (:mod:`llm_markdown.agent_fallback`).  Content inside ``<think>...</think>``
      is streamed as ``AgentReasoningDelta``; content outside as ``AgentContentDelta``.
      Provider-native reasoning is forwarded unchanged.

    **Agentic segment contract:** when ``tools`` is non-empty, the stream includes
    :class:`~llm_markdown.agent_stream.AgentSegmentStart` markers so consumers can
    treat the first segment as reasoning without inferring from the first delta type.

    Raises:
        ValueError: if :meth:`ReasoningConfig.validate_for_backend` fails or backend is ``gemini``.
        NotImplementedError: from :func:`~llm_markdown.agent_fallback.stream_agent_turn_fallback`
            if ``fallback`` is used with an unsupported backend (only ``openai``, ``openrouter``,
            and ``anthropic`` are supported).
    """
    rc = reasoning or ReasoningConfig.native()
    rc.validate_for_backend(backend)

    agentic = _is_agentic_turn(tools)

    if rc.mode is ReasoningMode.FALLBACK:
        yield from _apply_agentic_segment_markers(
            stream_agent_turn_fallback(
                provider,
                backend,
                messages,
                model=model,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
                planning_max_tokens=planning_max_tokens,
                should_cancel=should_cancel,
                **kwargs,
            ),
            agentic=agentic,
        )
        return

    call_kw: dict[str, Any] = dict(kwargs)
    call_kw["model"] = model
    if max_tokens is not None:
        call_kw["max_tokens"] = max_tokens
    if tools is not None:
        call_kw["tools"] = tools
    if tool_choice is not None:
        call_kw["tool_choice"] = tool_choice
    if should_cancel is not None:
        call_kw["should_cancel"] = should_cancel

    if backend in ("openai", "openrouter"):
        if rc.mode is ReasoningMode.NATIVE and rc.openai_extras:
            call_kw.update(rc.openai_extras)
        raw = provider.stream_chat_completion_events(messages, **call_kw)
    elif backend == "anthropic":
        if rc.mode is ReasoningMode.NATIVE and rc.anthropic_thinking:
            call_kw["thinking"] = dict(rc.anthropic_thinking)
        elif rc.mode is ReasoningMode.OFF:
            call_kw.pop("thinking", None)
        raw = provider.stream_messages_events(messages, **call_kw)
    else:
        msg = f"unknown backend: {backend!r}"
        raise ValueError(msg)

    if rc.mode is ReasoningMode.OFF:
        yield from _apply_agentic_segment_markers(
            _filter_reasoning_off(raw),
            agentic=agentic,
        )
        return
    yield from _apply_agentic_segment_markers(raw, agentic=agentic)


__all__ = ["stream_agent_turn"]
