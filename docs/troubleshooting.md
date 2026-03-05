# Troubleshooting

## Common issues

- **Missing docstring**: each `@prompt` function must include a docstring template.
- **Provider import errors**: install the correct extra (`openai`, `anthropic`, `gemini`, `openrouter`).
- **Structured output parse failures**: inspect raw model output and simplify schema requirements.
- **Streaming confusion**: streams return chunks and skip structured validation.
- **Image failures**: verify image MIME type and payload size (<20MB).

## Debug tips

- Enable logging for `llm_markdown` to inspect raw or structured responses.
- Start with `stream=False` while debugging output structure.
- Use lower `max_tokens` and fixed temperature for reproducible tests.
