"""Normalized streaming events for agent-style tool + text loops.

Providers yield these dataclasses so callers (e.g. Hof) can map them to their
own wire protocol without parsing vendor-specific SDK chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Union


@dataclass(frozen=True, slots=True)
class AgentContentDelta:
    """Incremental assistant-visible text (maps to chat ``delta.content``)."""

    text: str
    kind: Literal["content_delta"] = "content_delta"


@dataclass(frozen=True, slots=True)
class AgentReasoningDelta:
    """Separate reasoning / thinking text when the API exposes it."""

    text: str
    kind: Literal["reasoning_delta"] = "reasoning_delta"


@dataclass(frozen=True, slots=True)
class AgentToolCallDelta:
    """Partial tool call assembly (same index across deltas for one call)."""

    index: int
    tool_call_id: str | None
    name: str | None
    arguments: str | None
    kind: Literal["tool_call_delta"] = "tool_call_delta"


@dataclass(frozen=True, slots=True)
class AgentMessageFinish:
    """End of one assistant generation (before tool execution if applicable)."""

    finish_reason: str | None
    usage: dict[str, Any] | None
    kind: Literal["message_finish"] = "message_finish"


@dataclass(frozen=True, slots=True)
class AgentSegmentStart:
    """Start of a **reasoning** or **content** segment in an agentic turn.

    Emitted by :func:`~llm_markdown.agent_turn.stream_agent_turn` when ``tools``
    is non-empty. The stream always begins with ``segment="reasoning"``; the
    first :class:`AgentContentDelta` or :class:`AgentToolCallDelta` is preceded
    by ``segment="content"``. :class:`AgentReasoningDelta` events belong to the
    reasoning segment until that transition (including across FALLBACK phase A→B).
    """

    segment: Literal["reasoning", "content"]
    kind: Literal["segment_start"] = "segment_start"


@dataclass(frozen=True, slots=True)
class AgentRateLimitWait:
    """Provider hit a retryable limit or transient error; consumer may show a short wait notice.

    Emitted **before** the provider sleeps and retries opening the stream (e.g. HTTP 429).
    """

    seconds: float
    reason: Literal["rate_limit", "transient_error"] = "transient_error"
    kind: Literal["provider_wait"] = "provider_wait"


AgentStreamEvent = Union[
    AgentContentDelta,
    AgentReasoningDelta,
    AgentToolCallDelta,
    AgentMessageFinish,
    AgentSegmentStart,
    AgentRateLimitWait,
]


def openai_chat_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI ``chat.completions`` tool specs to Anthropic ``tools``."""
    out: list[dict] = []
    for item in tools:
        fn = item.get("function") if item.get("type") == "function" else item
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not name:
            continue
        out.append(
            {
                "name": name,
                "description": (fn.get("description") or "")[:10_000],
                "input_schema": fn.get("parameters")
                or {"type": "object", "properties": {}},
            }
        )
    return out


__all__ = [
    "AgentContentDelta",
    "AgentReasoningDelta",
    "AgentToolCallDelta",
    "AgentMessageFinish",
    "AgentSegmentStart",
    "AgentRateLimitWait",
    "AgentStreamEvent",
    "openai_chat_tools_to_anthropic",
]
