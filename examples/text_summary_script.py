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


# reasoning_first=False means that the LLM will start answering directly, without chain-of-thought reasoning
@llm(provider=langfuse_provider, reasoning_first=False)
def summarize_text(text: str) -> str:
    f"""
    Please provide a concise summary of the following text using only emojis:
    {text}
    """


if __name__ == "__main__":
    result = summarize_text(
        """
Lorem Ipsum is simply dummy text of the printing and typesetting industry.
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
when an unknown printer took a galley of type and scrambled it to make a type specimen book.
It has survived not only five centuries, but also the leap into electronic typesetting,
remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset
sheets containing Lorem Ipsum passages, and more recently with desktop publishing software
ike Aldus PageMaker including versions of Lorem Ipsum.
"""
    )
    print(result)
