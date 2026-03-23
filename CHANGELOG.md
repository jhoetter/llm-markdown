# Changelog

## 0.3.6

### Added

- **`stream_agent_turn`** — single entry for one tool-capable model turn with **`ReasoningConfig`**: `native`, `off`, or **`fallback`** (two-phase planning without tools, then tools; see `docs/agent-streaming.md` and `llm_markdown/agent_fallback.py`).
- **`ReasoningMode`** / **`BackendName`** for agent streams; re-exports on `llm_markdown.providers`.

### Notes

- Publish by tagging **`v0.3.6`** (must match `setup.py` / `llm_markdown.__version__`) and running the **Publish to PyPI** workflow (`docs/versioning.md`).
