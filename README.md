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

Other extras: `langfuse` (observability), `all` (everything), `test` (pytest suite).

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

`OpenAIProvider` handles all OpenAI model families (GPT-4o, GPT-5, o1/o3/o4) and auto-detects the correct token parameter.

## Testing

```bash
pytest                                          # unit tests (no API key)
OPENAI_API_KEY=sk-... pytest -m integration     # real API tests
```
