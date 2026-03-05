# llm-markdown

LLM calls as Python functions. Write a docstring, add a type hint, done.

```python
from llm_markdown import prompt
from llm_markdown.providers import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...", model="gpt-4o-mini")

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
from llm_markdown.providers import (
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    OpenRouterProvider,
)

openai_provider = OpenAIProvider(api_key="sk-...", model="gpt-4o-mini")
anthropic_provider = AnthropicProvider(api_key="sk-ant-...", model="claude-3-5-sonnet-latest")
gemini_provider = GeminiProvider(api_key="AIza...", model="gemini-2.0-flash")
openrouter_provider = OpenRouterProvider(
    api_key="sk-or-...",
    model="openai/gpt-4o-mini",
    app_name="llm-markdown",
    app_url="https://example.com",
)
```

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

`Image` accepts URLs, base64 strings, or data URIs. Use `List[Image]` for multiple images.

## Streaming

```python
@prompt(provider, stream=True)
def tell_story(topic: str) -> str:
    """Tell a short story about {topic}."""

for chunk in tell_story("a robot learning to paint"):
    print(chunk, end="", flush=True)
```

## Async

Async functions work the same way:

```python
@prompt(provider)
async def analyze(text: str) -> str:
    """Analyze: {text}"""

result = await analyze("some text")
```

## Observability with Langfuse

Wrap any provider with `LangfuseWrapper` to log every call:

```python
from llm_markdown.providers import OpenAIProvider, LangfuseWrapper

provider = LangfuseWrapper(
    provider=OpenAIProvider(api_key="sk-..."),
    secret_key="sk-lf-...",
    public_key="pk-lf-...",
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
    def complete_structured(self, messages, schema):
        ...  # return parsed dict
```

Built-in providers:

- `OpenAIProvider`: OpenAI models (GPT-4o, GPT-5, o1/o3/o4) with automatic token parameter detection.
- `AnthropicProvider`: Claude models with native structured output via tool schema.
- `GeminiProvider`: Gemini models with native structured output via response schema.
- `OpenRouterProvider`: OpenAI-compatible models routed through OpenRouter.

## Testing

```bash
pytest                          # unit tests (no API key)
cp .env.example .env           # fill provider keys
set -a; source .env; set +a
pytest -m integration           # real provider API tests
```

Required keys for provider integration tests:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`
- `OPENROUTER_API_KEY`

Integration tests run against a curated model set per provider and skip individual model cases if a model is not enabled for the key/account.
