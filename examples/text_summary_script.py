from llm_markdown import llm
from llm_markdown.providers.openai import OpenAILegacyProvider
from llm_markdown.providers.langfuse import LangfuseWrapper
from dotenv import load_dotenv
import os

load_dotenv()

# Define a LLM provider using legacy OpenAI models
openai_provider = OpenAILegacyProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    max_tokens=4096,
)

# You can also wrap the provider with Langfuse to log the LLM interactions
langfuse_provider = LangfuseWrapper(
    provider=openai_provider,
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


# reasoning_first=False means that the LLM will start answering directly, without chain-of-thought reasoning
@llm(provider=langfuse_provider, reasoning_first=False, stream=True)
def summarize_text(text: str) -> str:
    f"""
    Please provide a concise summary of the following text using only emojis:
    {text}
    """


if __name__ == "__main__":
    for chunk in summarize_text(
        """
        Lorem Ipsum is simply dummy text of the printing and typesetting industry.
        Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
        when an unknown printer took a galley of type and scrambled it to make a type specimen book.
        """
    ):
        print(chunk, end="", flush=True)
    print()  # Final newline
