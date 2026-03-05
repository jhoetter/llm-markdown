# Structured Output

When a prompt function returns a Pydantic model, `List[...]`, or `Dict[...]`, llm-markdown uses structured output mode.

## Behavior

- Prefer native provider structured APIs when available.
- Fall back to JSON prompting when native structured support is unavailable.
- Validate model outputs with Pydantic.

## Example

```python
from pydantic import BaseModel
from llm_markdown import prompt

class Review(BaseModel):
    sentiment: str
    score: float

@prompt(provider)
def classify(text: str) -> Review:
    """Classify sentiment and score for: {text}"""
```

Note: `stream=True` bypasses structured parsing and returns a raw stream iterator.
