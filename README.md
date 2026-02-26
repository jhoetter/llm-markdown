# llm-markdown

Turn Python functions into typed LLM calls using docstrings as prompts.

Write a function, add a `@prompt` decorator, and the docstring becomes the LLM prompt. The return type annotation controls everything -- primitive types get plain text completion, Pydantic models get structured output with automatic JSON schema generation and validation.

## Installation

```bash
pip install llm-markdown[all]
```

Or pick only what you need:

```bash
pip install llm-markdown              # core only (pydantic + requests)
pip install llm-markdown[openai]      # + OpenAI provider
pip install llm-markdown[langfuse]    # + Langfuse observability
```

For local development:

```bash
pip install -e ".[all,test]"
```

## Quick start

```python
from llm_markdown import prompt
from llm_markdown.providers import OpenAIProvider

provider = OpenAIProvider(api_key="sk-...", model="gpt-4o-mini")

@prompt(provider)
def summarize(text: str) -> str:
    """Summarize this text in 2 sentences: {text}"""

result = summarize("Long article text here...")
print(result)  # A plain string summary
```

The return type drives the behavior:
- `-> str` uses plain text completion
- `-> MyPydanticModel` uses structured output with JSON schema enforcement
- No flags, no configuration beyond the type hint.

## Structured output with Pydantic

Return a Pydantic model and the library handles JSON schema generation, structured output, and validation:

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
    - Key points from the review

    Review: {text}"""

result = analyze_review("A groundbreaking sci-fi film...")
print(result.sentiment)    # "positive"
print(result.rating)       # 4.5
print(result.key_points)   # ["groundbreaking visual effects", ...]
```

If the provider supports native structured output (like OpenAI's `response_format`), it's used automatically. If not, the library falls back to JSON prompting and parses the response -- no errors, no extra configuration.

## Returning generic types

`List[...]` and `Dict[...]` also trigger structured output automatically:

```python
from typing import List

@prompt(provider)
def list_steps(task: str) -> List[str]:
    """List the steps to complete this task: {task}"""

steps = list_steps("bake a cake")
# ["Preheat oven", "Mix dry ingredients", ...]
```

## Multimodal (images)

Use the `Image` type for vision tasks:

```python
from llm_markdown import prompt, Image

@prompt(provider)
def describe(image: Image) -> str:
    """Describe this image in detail."""

result = describe(Image("https://example.com/photo.jpg"))
```

`Image` accepts URLs, base64 strings, or data URIs. Multiple images are supported via `List[Image]`.

## Streaming

```python
@prompt(provider, stream=True)
def tell_story(topic: str) -> str:
    """Tell a short story about {topic}."""

for chunk in tell_story("a robot learning to paint"):
    print(chunk, end="", flush=True)
```

## Async support

All decorated functions can be async:

```python
@prompt(provider)
async def analyze(text: str) -> str:
    """Analyze: {text}"""

result = await analyze("some text")
```

## Langfuse integration

Wrap any provider with `LangfuseWrapper` for automatic logging and cost tracking:

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
    langfuse_metadata={"category": "movie-reviews", "use_case": "sentiment-analysis"},
)
def analyze(text: str) -> str:
    """Analyze: {text}"""
```

## Provider interface

`OpenAIProvider` auto-detects the correct token parameter for all model families (GPT-4o, GPT-5, o1/o3/o4 series). To add a custom provider, subclass `LLMProvider`:

```python
from llm_markdown.providers import LLMProvider

class MyProvider(LLMProvider):
    def complete(self, messages, **kwargs):
        ...  # return response string

    async def complete_async(self, messages, **kwargs):
        ...  # return response string

    # Optional: override for native structured output support.
    # If not implemented, the decorator falls back to JSON prompting.
    def complete_structured(self, messages, schema):
        ...  # return parsed dict
```

## Testing

Run the unit tests (no API key needed):

```bash
pip install -e ".[test]"
pytest
```

Run integration tests against the real OpenAI API:

```bash
OPENAI_API_KEY=sk-... pytest -m integration
```

## Migration from v0.2.0

- The `reasoning_first` parameter has been removed. The decorator now automatically chooses between plain completion and structured output based on the return type annotation.
- Pydantic models, `List[...]`, and `Dict[...]` return types trigger structured output. Primitive types (`str`, `int`, `float`, `bool`) use plain completion.
- The `{"reasoning": "...", "answer": ...}` JSON envelope is gone. Structured output schemas now match the return type directly.
- Providers that don't implement `complete_structured()` now get a graceful fallback (JSON prompting via `complete()`) instead of `NotImplementedError`.
- `provider` is still a keyword-only argument in `@prompt(provider=...)`.
