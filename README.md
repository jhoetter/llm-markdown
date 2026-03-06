# llm-markdown

LLM calls as Python functions. Write a docstring, add a type hint, done.

```python
import os

from llm_markdown import prompt
from llm_markdown.providers import OpenAIProvider

provider = OpenAIProvider(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",
)

@prompt(provider)
def summarize(text: str) -> str:
    """Summarize this text in 2 sentences: {text}"""

result = summarize("Long article text here...")
# "The article discusses... In conclusion, ..."
```

## How it works

Three rules:

1. The **docstring** is the prompt. Use `{param}` to interpolate function arguments.
2. The **return type** controls the output format. `-> str` gives plain text. `-> MyModel` gives validated structured output.
3. **`Image` parameters** are attached as vision inputs automatically -- they don't appear in the docstring.

That's it. No configuration flags, no prompt templates, no output parsers. The function signature *is* the configuration.

## Installation

```bash
pip install llm-markdown[openai]
```

For development with full provider + test support:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,all]"
```

Provider extras: `openai`, `anthropic`, `gemini`, `openrouter`.

Other extras: `langfuse` (observability), `all` (all providers + langfuse), `test` (pytest suite).

## Provider support

| Provider | Included | Native structured output | Images | Streaming | Extra |
| --- | --- | --- | --- | --- | --- |
| OpenAI | Yes | Yes (`response_format`) | Yes | Yes | `openai` |
| Anthropic | Yes | Yes (tool schema) | Yes (data URI images) | Yes | `anthropic` |
| Google Gemini | Yes | Yes (`response_schema`) | Yes (data URI images) | Yes | `gemini` |
| OpenRouter | Yes | Yes (OpenAI-compatible schema) | Model-dependent | Yes | `openrouter` |

Install one provider:

```bash
pip install llm-markdown[anthropic]
pip install llm-markdown[gemini]
pip install llm-markdown[openrouter]
```

Or install all providers:

```bash
pip install llm-markdown[all]
```

Instantiate providers:

```python
import os

from llm_markdown.providers import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    OpenRouterProvider,
)

openai_provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
anthropic_provider = AnthropicProvider(api_key=os.environ["ANTHROPIC_API_KEY"], model="claude-3-5-sonnet-latest")
gemini_provider = GeminiProvider(api_key=os.environ["GOOGLE_API_KEY"], model="gemini-2.0-flash")
openrouter_provider = OpenRouterProvider(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model="inception/mercury-2",
    app_name="llm-markdown",
    app_url="https://example.com",
)
```

OpenRouter access is configured in three places:

- `.env`: `OPENROUTER_API_KEY`
- Provider instance: `OpenRouterProvider(model="...")`
- Integration test model list: `OPENROUTER_TEST_MODELS` in `tests/test_integration_multi_providers.py`

## Structured output

Return a Pydantic model and the response is validated automatically:

```python
from pydantic import BaseModel

class ReviewAnalysis(BaseModel):
    sentiment: str
    rating: float
    key_points: list[str]

@prompt(provider)
def analyze_review(text: str) -> ReviewAnalysis:
    """Analyze this movie review:
    - Overall sentiment (positive/negative/neutral)
    - Rating on a scale of 1.0 to 5.0
    - Key points

    Review: {text}"""

result = analyze_review("A groundbreaking sci-fi film...")
result.sentiment    # "positive"
result.rating       # 4.5
result.key_points   # ["groundbreaking visual effects", ...]
```

The library generates a JSON schema from the Pydantic model and uses the provider's native structured output (e.g. OpenAI's `response_format`). If the provider doesn't support it, it falls back to JSON prompting automatically.

`List[...]` and `Dict[...]` work the same way:

```python
from typing import List

@prompt(provider)
def list_steps(task: str) -> List[str]:
    """List the steps to complete this task: {task}"""

list_steps("bake a cake")
# ["Preheat oven to 350F", "Mix dry ingredients", ...]
```

## Images

`Image` parameters are detected by type and attached to the API call as vision inputs. The docstring is the text part of the prompt:

```python
from llm_markdown import prompt, Image

@prompt(provider)
def answer_about_image(image: Image, question: str) -> str:
    """Answer this question about the image: {question}"""

answer_about_image(
    image=Image("https://example.com/chart.png"),
    question="What trend does this chart show?",
)
```

`Image` accepts URLs, local file paths, base64 strings, or data URIs. Non-image content types and payloads above 20MB are rejected. Use `List[Image]` for multiple images.

## Image generation

Use the high-level helper for provider-backed image generation:

```python
from llm_markdown import generate_image

image_result = generate_image(
    provider=openrouter_provider,
    prompt="A minimalist watercolor mountain scene",
    model="openai/gpt-image-1",
)
print(image_result["images"][0]["url"])
```

Async variant:

```python
from llm_markdown import generate_image_async

image_result = await generate_image_async(
    provider=openrouter_provider,
    prompt="A retro sci-fi city skyline",
    model="openai/gpt-image-1",
)
```

## Streaming

```python
@prompt(provider, stream=True)
def tell_story(topic: str) -> str:
    """Tell a short story about {topic}."""

for chunk in tell_story("a robot learning to paint"):
    print(chunk, end="", flush=True)
```

Structured event streaming is also available:

```python
from pydantic import BaseModel

class Answer(BaseModel):
    value: str

@prompt(provider, stream=True, stream_mode="json_events")
def stream_answer(question: str) -> Answer:
    """Return a JSON answer for: {question}"""

for event in stream_answer("What is 2+2?"):
    print(event["type"])
```

## Async

Async functions work the same way:

```python
@prompt(provider)
async def analyze(text: str) -> str:
    """Analyze: {text}"""

result = await analyze("some text")
```

## Sessions (multi-turn)

```python
from llm_markdown import Session

session = Session(provider, max_messages=12)

@session.prompt()
def ask(question: str) -> str:
    """Answer this: {question}"""

ask("What is retrieval-augmented generation?")
ask("Now explain it to a beginner.")
```

`Session` keeps shared chat history and supports optional `max_messages`/`max_tokens` trimming.

## Generation controls and metadata

Pass default generation controls at decoration time and override per call with `_llm_options`:

```python
from llm_markdown import prompt, PromptResult

@prompt(
    provider,
    generation_options={"temperature": 0.4, "max_tokens": 300},
    return_metadata=True,
)
def summarize(text: str) -> str:
    """Summarize: {text}"""

result: PromptResult[str] = summarize(
    "Some article",
    _llm_options={"temperature": 0.2},
)
print(result.output)
print(result.metadata)  # provider/model/response_id/usage
```

## Observability with Langfuse

Wrap any provider with `LangfuseWrapper` to log every call:

```python
import os

from llm_markdown.providers import OpenAIProvider, LangfuseWrapper

provider = LangfuseWrapper(
    provider=OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"]),
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    host="https://cloud.langfuse.com",
)

@prompt(
    provider,
    langfuse_metadata={"category": "reviews", "use_case": "sentiment"},
)
def analyze(text: str) -> str:
    """Analyze: {text}"""
```

## Custom providers

Subclass `LLMProvider` to use any LLM backend:

```python
from llm_markdown.providers import LLMProvider

class MyProvider(LLMProvider):
    def complete(self, messages, **kwargs):
        ...  # return response string

    async def complete_async(self, messages, **kwargs):
        ...  # return response string

    # Optional -- enables native structured output.
    # Without this, the decorator falls back to JSON prompting.
    def complete_structured(self, messages, schema, **kwargs):
        ...  # return parsed dict
```

Built-in providers:

- `OpenAIProvider`: OpenAI models (GPT-4o, GPT-5, o1/o3/o4) with automatic token parameter detection.
- `AnthropicProvider`: Claude models with native structured output via tool schema.
- `GeminiProvider`: Gemini models with native structured output via response schema.
- `OpenRouterProvider`: OpenAI-compatible models routed through OpenRouter.
- `RouterProvider`: route/fallback wrapper across multiple providers.

## Testing

```bash
python -m llm_markdown.preflight --strict
pytest -m "not integration"      # required quality gate before commit
cp .env.example .env           # fill provider keys
set -a; source .env; set +a
pytest -m integration           # optional real provider API tests
python -m build                 # required quality gate before push
```

If you installed with `.[test,all]`, integration tests can also auto-read `.env` via `python-dotenv`.

Required keys for provider integration tests:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `OPENROUTER_API_KEY`

Integration tests run against a curated model set per provider and skip individual model cases if a model is not enabled for the key/account.

## Troubleshooting

- `ImportError` for provider packages: install matching extras (`llm-markdown[openai]`, `llm-markdown[anthropic]`, `llm-markdown[gemini]`, `llm-markdown[openrouter]`).
- Missing function docstring: every `@prompt` function needs a docstring prompt template.
- Structured outputs not supported by provider: the decorator automatically falls back to JSON prompting.
- `stream=True` returns a stream iterator and bypasses structured parsing.
- Image issues: URLs/local files must resolve to an image MIME type, and payloads above 20MB are rejected.

## Security notes

- Keep secrets in environment variables (`.env`) and never hardcode keys in source.
- Do not commit `.env` files or raw credentials.
- Treat remote image URLs as untrusted input; prefer trusted sources for production.
- Restrict remote image hosts with `LLM_MARKDOWN_IMAGE_URL_ALLOWLIST` (comma-separated domains).
- Keep private-network URL blocking enabled (`LLM_MARKDOWN_IMAGE_BLOCK_PRIVATE_NETWORKS=true`).

## Versioning and release

- Project version currently lives in `llm_markdown/__init__.py` and `setup.py`.
- Document user-visible changes in release notes or PR descriptions.
- Use `docs/release-notes-template.md` for release notes and breaking-change checklist.
- Before publishing, run `pytest -m "not integration"` and `python -m build`.

## Additional docs

- `docs/getting-started.md`
- `docs/providers.md`
- `docs/structured-output.md`
- `docs/images-and-multimodal.md`
- `docs/streaming-and-async.md`
- `docs/troubleshooting.md`
- `docs/security.md`
- `docs/contributing.md`
- `docs/versioning.md`
- `docs/operations-runbook.md`
- `docs/release-notes-template.md`
