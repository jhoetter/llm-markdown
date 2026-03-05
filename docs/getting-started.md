# Getting Started

## Install

```bash
pip install llm-markdown[openai]
```

## Configure environment

```bash
cp .env.example .env
set -a; source .env; set +a
```

## First prompt

```python
import os
from llm_markdown import prompt
from llm_markdown.providers import OpenAIProvider

provider = OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])

@prompt(provider)
def summarize(text: str) -> str:
    """Summarize this in two sentences: {text}"""
```

Use `_llm_options` per call to override generation settings:

```python
summarize("...", _llm_options={"temperature": 0.2, "max_tokens": 200})
```
