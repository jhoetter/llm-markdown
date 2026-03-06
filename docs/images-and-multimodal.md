# Images and Multimodal

`Image` inputs are detected from type hints and appended to the user message automatically.

Accepted image sources:

- URL
- Local file path
- Data URI
- Raw base64 string

Validation:

- Source must resolve to an `image/*` MIME type
- Payload must be 20MB or less

## Example

```python
from llm_markdown import Image, prompt

@prompt(provider)
def answer(image: Image, question: str) -> str:
    """Answer this about the image: {question}"""

answer(Image("./assets/chart.png"), "What trend is shown?")
```

## Image generation

llm-markdown also supports provider image-generation endpoints:

```python
from llm_markdown import generate_image

result = generate_image(
    provider=provider,
    prompt="A clean isometric illustration of a data pipeline",
    model="openai/gpt-image-1",
)
```

The returned payload includes provider/model metadata and one or more generated images (URL or base64, depending on provider response mode).
