from llm_markdown import llm
from llm_markdown.providers.openai import OpenAIProvider
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import os

load_dotenv()

# Define a modern LLM provider using GPT-5
# Note: Uses max_completion_tokens instead of max_tokens
openai_provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-5",
    max_completion_tokens=4096,
)


# Define Pydantic models for structured output
class BookRecommendation(BaseModel):
    title: str
    author: str
    genre: str
    reason: str
    difficulty_level: str  # "beginner", "intermediate", "advanced"


class ReadingList(BaseModel):
    theme: str
    recommendations: List[BookRecommendation]
    total_estimated_reading_time: str


# Example using GPT-5's enhanced reasoning for book recommendations
@llm(provider=openai_provider, reasoning_first=True)
def create_reading_list(topic: str, reader_level: str, preferences: str) -> ReadingList:
    f"""
    Create a personalized reading list using GPT-5's enhanced understanding.
    
    Topic of interest: {topic}
    Reader level: {reader_level}
    Preferences: {preferences}
    
    Please recommend 5 books that would create a comprehensive learning journey.
    Consider the progression from foundational to advanced concepts.
    """


if __name__ == "__main__":
    print("📚 GPT-5 Simple Examples\n")
    print("=" * 50)

    # Example 1: Structured output with Pydantic
    print("📖 Creating a personalized reading list...")
    reading_list = create_reading_list(
        topic="artificial intelligence and machine learning",
        reader_level="intermediate programmer",
        preferences="practical applications, recent developments, some theory",
    )

    print(f"Theme: {reading_list.theme}")
    print(f"Total books: {len(reading_list.recommendations)}")
    print(f"Estimated reading time: {reading_list.total_estimated_reading_time}")
    print("\nFirst recommendation:")
    first_book = reading_list.recommendations[0]
    print(f"  📕 {first_book.title} by {first_book.author}")
    print(f"     Genre: {first_book.genre} | Level: {first_book.difficulty_level}")
    print(f"     Reason: {first_book.reason}")
