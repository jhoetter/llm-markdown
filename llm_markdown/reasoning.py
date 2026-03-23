"""Reasoning / extended-thinking configuration for agent-style streams."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class ReasoningMode(str, Enum):
    """How to obtain (or suppress) streamed reasoning / thinking text."""

    NATIVE = "native"
    OFF = "off"
    FALLBACK = "fallback"


BackendName = Literal["openai", "anthropic", "openrouter", "gemini"]


@dataclass(frozen=True, slots=True)
class ReasoningConfig:
    """Provider-aware reasoning policy for :func:`~llm_markdown.agent_turn.stream_agent_turn`.

    Attributes:
        mode: ``native`` forwards API reasoning; ``off`` filters ``AgentReasoningDelta``;
            ``fallback`` uses a hybrid in :mod:`llm_markdown.agent_fallback`: two-phase
            planning (no tools, then tools) for tool-selection rounds, with all Phase A
            text as ``AgentReasoningDelta``; answer rounds use one completion with think-tag
            parsing (inside tags → ``AgentReasoningDelta``, outside → ``AgentContentDelta``).
        openai_extras: Extra kwargs merged into OpenAI ``chat.completions.create`` when
            ``mode`` is ``native`` (e.g. model-specific reasoning parameters).
        anthropic_thinking: Passed as ``thinking=...`` to Anthropic ``messages.stream`` when
            ``mode`` is ``native`` (e.g. ``{"type": "enabled", "budget_tokens": 1024}``).
    """

    mode: ReasoningMode = ReasoningMode.NATIVE
    openai_extras: dict[str, Any] | None = None
    anthropic_thinking: dict[str, Any] | None = None

    @staticmethod
    def native(
        *,
        openai_extras: dict[str, Any] | None = None,
        anthropic_thinking: dict[str, Any] | None = None,
    ) -> ReasoningConfig:
        return ReasoningConfig(
            mode=ReasoningMode.NATIVE,
            openai_extras=dict(openai_extras) if openai_extras else None,
            anthropic_thinking=dict(anthropic_thinking) if anthropic_thinking else None,
        )

    @staticmethod
    def off() -> ReasoningConfig:
        return ReasoningConfig(mode=ReasoningMode.OFF)

    def validate_for_backend(self, backend: BackendName) -> None:
        """Raise ``ValueError`` if options conflict with *backend*."""
        if backend == "gemini":
            msg = (
                "backend 'gemini' is not supported for stream_agent_turn "
                "(no unified AgentStreamEvent tool loop yet); use openai, openrouter, or anthropic"
            )
            raise ValueError(msg)
        if self.mode is ReasoningMode.FALLBACK:
            if self.openai_extras:
                msg = "ReasoningConfig FALLBACK does not use openai_extras; use NATIVE instead"
                raise ValueError(msg)
            if self.anthropic_thinking:
                msg = (
                    "ReasoningConfig FALLBACK does not use anthropic_thinking; "
                    "use NATIVE for extended thinking"
                )
                raise ValueError(msg)
            return
        if self.mode is ReasoningMode.OFF and self.anthropic_thinking:
            msg = "ReasoningConfig.mode=OFF cannot be combined with anthropic_thinking"
            raise ValueError(msg)
        if backend in ("openai", "openrouter"):
            if self.anthropic_thinking:
                msg = "anthropic_thinking is not used with OpenAI-compatible backends"
                raise ValueError(msg)
        elif backend == "anthropic":
            if self.openai_extras:
                msg = "openai_extras is not used with backend='anthropic'"
                raise ValueError(msg)


__all__ = ["BackendName", "ReasoningConfig", "ReasoningMode"]
