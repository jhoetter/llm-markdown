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


AgentStreamEvent = Union[
    AgentContentDelta,
    AgentReasoningDelta,
    AgentToolCallDelta,
    AgentMessageFinish,
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
    "AgentStreamEvent",
    "openai_chat_tools_to_anthropic",
]
