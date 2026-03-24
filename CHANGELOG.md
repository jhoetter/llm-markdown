# Changelog

## 0.3.18

### Added

- **`build_llm_provider_for_model`** (and **`infer_llm_markdown_backend_for_model`**, **`resolve_llm_markdown_backend`**) in `llm_markdown.providers.from_env` — pick `OpenAIProvider` vs `AnthropicProvider` from the model id, optional `LLM_MARKDOWN_PROVIDER` env (`openai` / `anthropic` / `claude`), and API keys from args or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.

### Notes

- Publish by tagging **`v0.3.18`** (must match `setup.py` / `llm_markdown.__version__`) and running the **Publish to PyPI** workflow (`docs/versioning.md`).

## 0.3.7

### Added

- **`AgentSegmentStart`** — `segment="reasoning" | "content"` markers for **agentic** turns (`tools` non-empty) from **`stream_agent_turn`**: stream always opens in the reasoning segment; first `AgentContentDelta` or `AgentToolCallDelta` is preceded by `segment="content"`.

### Changed

- **FALLBACK** phase B forwards **`AgentReasoningDelta`** from the provider (no longer stripped) so tool rounds can interleave reasoning like native mode.

### Notes

- Publish by tagging **`v0.3.7`** (must match `setup.py` / `llm_markdown.__version__`) and running the **Publish to PyPI** workflow (`docs/versioning.md`).

## 0.3.6

### Added

- **`stream_agent_turn`** — single entry for one tool-capable model turn with **`ReasoningConfig`**: `native`, `off`, or **`fallback`** (two-phase planning without tools, then tools; see `docs/agent-streaming.md` and `llm_markdown/agent_fallback.py`).
- **`ReasoningMode`** / **`BackendName`** for agent streams; re-exports on `llm_markdown.providers`.

### Notes

- Publish by tagging **`v0.3.6`** (must match `setup.py` / `llm_markdown.__version__`) and running the **Publish to PyPI** workflow (`docs/versioning.md`).
