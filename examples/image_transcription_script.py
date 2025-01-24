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

# You can also wrap the provider with Langfuse to log the LLM interactions
langfuse_provider = LangfuseWrapper(
    provider=openai_provider,
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


# You can also use images in the prompt
@llm(provider=langfuse_provider, reasoning_first=False)
def transcribe_image(image_base64: str) -> str:
    """
    Please accurately describe the image in a few sentences.
    !image[{image_base64}]
    """.format(
        image_base64=image_base64
    )


if __name__ == "__main__":
    result = transcribe_image(
        "https://example.com/image.jpg"
    )
    print(result)
