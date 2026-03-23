from llm_markdown.agent_stream import (
    AgentContentDelta,
    AgentMessageFinish,
    AgentReasoningDelta,
    AgentStreamEvent,
    AgentToolCallDelta,
    openai_chat_tools_to_anthropic,
)
from llm_markdown.agent_fallback import stream_agent_turn_fallback
from llm_markdown.agent_turn import stream_agent_turn
from llm_markdown.reasoning import BackendName, ReasoningConfig, ReasoningMode

from .openai import OpenAIProvider, OpenAILegacyProvider
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .openrouter import OpenRouterProvider
from .base import LLMProvider, ProviderError
from .router import RouterProvider


def __getattr__(name: str):
    """Lazy import so ``langfuse`` extra is optional for OpenAI-only installs."""
    if name == "LangfuseWrapper":
        from .langfuse import LangfuseWrapper as _LangfuseWrapper

        return _LangfuseWrapper
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "BackendName",
    "ReasoningConfig",
    "ReasoningMode",
    "stream_agent_turn",
    "stream_agent_turn_fallback",
    "AgentContentDelta",
    "AgentMessageFinish",
    "AgentReasoningDelta",
    "AgentStreamEvent",
    "AgentToolCallDelta",
    "OpenAIProvider",
    "OpenAILegacyProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "OpenRouterProvider",
    "RouterProvider",
    "LangfuseWrapper",
    "LLMProvider",
    "ProviderError",
    "openai_chat_tools_to_anthropic",
]
