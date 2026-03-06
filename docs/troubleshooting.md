# Troubleshooting

## Common issues

- **Missing docstring**: each `@prompt` function must include a docstring template.
- **Provider import errors**: install the correct extra (`openai`, `anthropic`, `gemini`, `openrouter`).
- **Integration tests skipped due to missing SDKs**: install development extras with `pip install -e ".[test,all]"`.
- **Preflight failures**: run `python -m llm_markdown.preflight` and fix missing env vars/SDKs before integration runs.
- **Structured output parse failures**: inspect raw model output and simplify schema requirements.
- **Streaming confusion**: streams return chunks and skip structured validation.
- **Image failures**: verify image MIME type and payload size (<20MB).

## Debug tips

- Enable logging for `llm_markdown` to inspect raw or structured responses.
- Start with `stream=False` while debugging output structure.
- Use lower `max_tokens` and fixed temperature for reproducible tests.
