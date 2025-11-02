from llm_markdown import llm
from llm_markdown.providers.openai import OpenAILegacyProvider
from llm_markdown.providers.langfuse import LangfuseWrapper
from dotenv import load_dotenv
from pydantic import BaseModel
import os

load_dotenv()

# Define a LLM provider using legacy OpenAI models
openai_provider = OpenAILegacyProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    max_tokens=4096,
)

# Wrap with Langfuse for logging and cost tracking
langfuse_provider = LangfuseWrapper(
    provider=openai_provider,
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


# Define a Pydantic model for the input
class MovieReview(BaseModel):
    title: str
    year: int
    content: str


# Define a Pydantic model for the output
class ReviewAnalysis(BaseModel):
    sentiment: str  # "positive", "negative", or "neutral"
    rating: float  # 1.0 to 5.0
    key_points: list[str]


# Define a function that uses the LLM provider and write the prompt as a formatted markdown string
# Add langfuse_metadata for categorization - you can filter by "category": "movie-reviews" in Langfuse dashboard
@llm(provider=langfuse_provider, langfuse_metadata={"category": "movie-reviews", "use_case": "sentiment-analysis"})
def analyze_movie_review(review: MovieReview) -> ReviewAnalysis:
    f"""
    Analyze the movie review and provide:
    - Overall sentiment (positive/negative/neutral)
    - Rating on a scale of 1.0 to 5.0
    - Key points from the review

    Movie: {review.title} ({review.year})
    Review: {review.content}
    """


if __name__ == "__main__":
    result = analyze_movie_review(
        MovieReview(
            title="The Matrix",
            year=1999,
            content="The Matrix is a science fiction action film that explores the concept of a dystopian future where humans are enslaved by machines. The story follows the journey of a computer programmer who uncovers the truth about the world and joins a rebellion against the machines.",
        )
    )
    print(result)
    print(type(result))
