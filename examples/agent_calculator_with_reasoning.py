#!/usr/bin/env python3
"""Minimal agent loop: ``add`` / ``multiply`` tools with ``stream_agent_turn``.

Loads ``~/repos/llm-markdown/.env`` when ``python-dotenv`` is installed (or cwd ``.env``).

Env:

- ``LLM_MARKDOWN_AGENT_BACKEND`` — ``openai`` (default), ``openrouter``, ``anthropic``
- ``LLM_MARKDOWN_AGENT_MODEL`` — optional override
- ``LLM_MARKDOWN_AGENT_REASONING_MODE`` — **defaults to ``fallback`` in this script** so you see a
  planning stream (``AgentReasoningDelta``) before tools on ordinary chat models (e.g. ``gpt-4o-mini``).
  Use ``native`` for wire-native reasoning only when your model/API emits it; use ``off`` to strip it.
- Provider keys: ``OPENAI_API_KEY``, ``OPENROUTER_API_KEY``, or ``ANTHROPIC_API_KEY``
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    _root = Path(__file__).resolve().parents[1]
    load_dotenv(_root / ".env")
    load_dotenv()
except ImportError:
    pass

from llm_markdown.agent_stream import (
    AgentContentDelta,
    AgentMessageFinish,
    AgentReasoningDelta,
    AgentToolCallDelta,
)
from llm_markdown.agent_turn import stream_agent_turn
from llm_markdown.providers import AnthropicProvider, OpenAIProvider, OpenRouterProvider
from llm_markdown.reasoning import BackendName, ReasoningConfig, ReasoningMode

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Return a + b.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Return a * b.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
]


def _run_tool(name: str, arguments: str) -> str:
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "invalid JSON arguments"})
    a = args.get("a")
    b = args.get("b")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return json.dumps({"error": "a and b must be numbers"})
    if name == "add":
        return json.dumps({"result": a + b})
    if name == "multiply":
        return json.dumps({"result": a * b})
    return json.dumps({"error": f"unknown tool {name!r}"})


def _backend() -> BackendName:
    raw = (os.environ.get("LLM_MARKDOWN_AGENT_BACKEND") or "openai").strip().lower()
    if raw in ("openai", "anthropic", "openrouter", "gemini"):
        return raw  # type: ignore[return-value]
    print(f"Unknown LLM_MARKDOWN_AGENT_BACKEND={raw!r}", file=sys.stderr)
    sys.exit(1)


def _mode() -> ReasoningMode:
    # Default fallback so this demo shows reasoning_stream + tools without a "reasoning" model.
    raw = (os.environ.get("LLM_MARKDOWN_AGENT_REASONING_MODE") or "fallback").strip().lower()
    if raw == "off":
        return ReasoningMode.OFF
    if raw == "fallback":
        return ReasoningMode.FALLBACK
    return ReasoningMode.NATIVE


def _model(backend: BackendName) -> str:
    env = (os.environ.get("LLM_MARKDOWN_AGENT_MODEL") or "").strip()
    if env:
        return env
    if backend == "openai":
        return "gpt-4o-mini"
    if backend == "openrouter":
        return "openai/gpt-4o-mini"
    if backend == "anthropic":
        return "claude-3-5-haiku-latest"
    return "gemini-2.0-flash"


def _provider(backend: BackendName, model: str):
    if backend == "gemini":
        print(
            "backend gemini is not supported for stream_agent_turn in llm-markdown yet.",
            file=sys.stderr,
        )
        sys.exit(1)
    if backend == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("Missing OPENAI_API_KEY", file=sys.stderr)
            sys.exit(1)
        return OpenAIProvider(api_key=key, model=model)
    if backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            print("Missing OPENROUTER_API_KEY", file=sys.stderr)
            sys.exit(1)
        return OpenRouterProvider(api_key=key, model=model)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("Missing ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)
    return AnthropicProvider(api_key=key, model=model)


def _reasoning(backend: BackendName, mode: ReasoningMode) -> ReasoningConfig:
    if mode is ReasoningMode.NATIVE and backend == "anthropic":
        return ReasoningConfig.native(
            anthropic_thinking={"type": "enabled", "budget_tokens": 2048},
        )
    if mode is ReasoningMode.NATIVE:
        return ReasoningConfig.native()
    if mode is ReasoningMode.OFF:
        return ReasoningConfig.off()
    return ReasoningConfig(mode=ReasoningMode.FALLBACK)


def main() -> None:
    backend = _backend()
    mode = _mode()
    model = _model(backend)
    provider = _provider(backend, model)
    rc = _reasoning(backend, mode)

    env_mode = (os.environ.get("LLM_MARKDOWN_AGENT_REASONING_MODE") or "").strip()
    if not env_mode:
        print(
            "Note: LLM_MARKDOWN_AGENT_REASONING_MODE unset — using fallback (planning stream "
            "before tools). Set to native for API-native reasoning only on models that emit it.\n"
        )

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                "First compute (7 + 5) using the add tool, then multiply that sum by 3 "
                "using multiply. Show the final numeric result in your last assistant message."
            ),
        }
    ]

    max_rounds = 6
    for round_i in range(1, max_rounds + 1):
        print(f"\n--- round {round_i} backend={backend} model={model} reasoning={mode.value} ---")
        parts: dict[int, dict[str, str]] = {}
        assistant_text = ""
        finish_reason: str | None = None
        n_reasoning = 0
        n_content = 0
        n_tool_delta = 0

        for ev in stream_agent_turn(
            provider,
            backend,
            messages,
            model=model,
            tools=_TOOLS,
            tool_choice="auto",
            reasoning=rc,
        ):
            if isinstance(ev, AgentContentDelta):
                assistant_text += ev.text
                n_content += 1
                print(f"  content_delta: {ev.text!r}")
            elif isinstance(ev, AgentReasoningDelta):
                n_reasoning += 1
                print(f"  reasoning_delta: {ev.text!r}")
            elif isinstance(ev, AgentToolCallDelta):
                idx = int(ev.index)
                if idx not in parts:
                    parts[idx] = {"id": "", "name": "", "arguments": ""}
                if ev.tool_call_id:
                    parts[idx]["id"] = ev.tool_call_id
                if ev.name:
                    parts[idx]["name"] += ev.name
                if ev.arguments:
                    parts[idx]["arguments"] += ev.arguments
                n_tool_delta += 1
                print(
                    f"  tool_delta: index={ev.index} id={ev.tool_call_id!r} "
                    f"name={ev.name!r} args_fragment={ev.arguments!r}"
                )
            elif isinstance(ev, AgentMessageFinish):
                finish_reason = ev.finish_reason
                print(f"  finish: {finish_reason!r} usage={ev.usage!r}")

        print(f"  assistant_text: {assistant_text!r}")
        print(
            f"  summary: reasoning_deltas={n_reasoning} content_deltas={n_content} "
            f"tool_delta_events={n_tool_delta}"
        )

        if finish_reason != "tool_calls" or not parts:
            print("Done (no tool_calls or empty tool state).")
            break

        tool_calls_payload: list[dict[str, Any]] = []
        for idx in sorted(parts):
            tc = parts[idx]
            tid = tc["id"] or f"call_{idx}"
            tool_calls_payload.append(
                {
                    "id": tid,
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"] or "{}",
                    },
                }
            )
        asst_msg: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_text if assistant_text.strip() else None,
            "tool_calls": tool_calls_payload,
        }
        messages.append(asst_msg)
        for idx in sorted(parts):
            tc = parts[idx]
            tid = tc["id"] or f"call_{idx}"
            name = tc["name"]
            out = _run_tool(name, tc["arguments"])
            print(f"  tool_result: {name} -> {out}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tid,
                    "content": out,
                }
            )


if __name__ == "__main__":
    main()
