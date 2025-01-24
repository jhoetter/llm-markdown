# llm-markdown

Functionized LLM instructions as markdown

# Installation

```bash
pip install llm-markdown
```

or locally:

```bash
pip uninstall llm-markdown
pip install -e .
```

# Example usage

```python
from llm_markdown import llm
from llm_markdown.providers.openai import OpenAIProvider
from llm_markdown.providers.langfuse import LangfuseWrapper
from dotenv import load_dotenv
import os

load_dotenv()

# Define a LLM provider, e.g. OpenAI
openai_provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
)

# Define a Pydantic model for the input
class MovieReview(BaseModel):
    title: str
    year: int
    content: str

# Define a Pydantic model for the output
class ReviewAnalysis(BaseModel):
    sentiment: str  # "positive", "negative", or "neutral"
    rating: float   # 1.0 to 5.0
    key_points: list[str]

# Define a function that uses the LLM provider and write the prompt as a formatted markdown string
@llm(provider=openai_provider)
def analyze_movie_review(review: MovieReview) -> ReviewAnalysis:
    f"""
    Analyze the movie review and provide:
    - Overall sentiment (positive/negative/neutral)
    - Rating on a scale of 1.0 to 5.0
    - Key points from the review

    Movie: {review.title} ({review.year})
    Review: {review.content}
    """

# You can also wrap the provider with Langfuse to log the LLM interactions
langfuse_provider = LangfuseWrapper(
    provider=openai_provider,
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

# reasoning_first=False means that the LLM will start answering directly, without chain-of-thought reasoning
@llm(provider=langfuse_provider, reasoning_first=False)
def summarize_text(text: str) -> str:
    f"""
    Please provide a concise summary of the following text using only emojis:
    {text}
    """

# You can also use images in the prompt
@llm(provider=langfuse_provider, reasoning_first=False)
def transcribe_image(image_base64: str) -> str:
    """
    Please accurately describe the image in a few sentences.
    !image[{image_base64}]
    """.format(
        image_base64=image_base64
    )
```
