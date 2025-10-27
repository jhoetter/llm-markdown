from llm_markdown import llm
from llm_markdown.providers.openai import OpenAIProvider
from llm_markdown.providers.langfuse import LangfuseWrapper
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import os

load_dotenv()

# Define a modern LLM provider using GPT-5 with max_completion_tokens
openai_provider = OpenAIProvider(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-5",
    max_completion_tokens=8192,  # GPT-5 supports higher token limits
)

# You can also wrap the provider with Langfuse to log the LLM interactions
langfuse_provider = LangfuseWrapper(
    provider=openai_provider,
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


# Define Pydantic models for structured output
class ResearchInsight(BaseModel):
    topic: str
    confidence_score: float  # 0.0 to 1.0
    evidence: List[str]
    implications: List[str]


class ComprehensiveAnalysis(BaseModel):
    executive_summary: str
    key_insights: List[ResearchInsight]
    methodology: str
    limitations: List[str]
    recommendations: List[str]


# Example 1: Advanced reasoning with structured output
@llm(provider=langfuse_provider, reasoning_first=True)
def analyze_complex_scenario(scenario: str, context: str) -> ComprehensiveAnalysis:
    f"""
    You are an expert analyst with access to GPT-5's advanced reasoning capabilities.
    Perform a comprehensive analysis of the given scenario using multi-step reasoning.
    
    Scenario: {scenario}
    Context: {context}
    
    Please provide:
    1. A clear executive summary
    2. Key insights with confidence scores and supporting evidence
    3. Your analytical methodology
    4. Limitations of your analysis
    5. Actionable recommendations
    
    Use your enhanced reasoning abilities to consider multiple perspectives,
    identify potential biases, and provide nuanced insights.
    """


if __name__ == "__main__":
    print("🚀 GPT-5 Advanced Reasoning Examples\n")
    print("=" * 60)

    # Example 1: Complex scenario analysis
    print("\n📊 Complex Scenario Analysis:")
    scenario = """
    A mid-size tech company is experiencing 40% employee turnover in their engineering team.
    Recent surveys show high job satisfaction but concerns about career growth and compensation.
    The company just secured Series B funding but faces increased competition for talent.
    """

    context = """
    Industry context: Tech talent shortage, remote work trends, increased salary expectations.
    Company context: 150 employees, profitable, strong product-market fit, expanding internationally.
    """

    analysis = analyze_complex_scenario(scenario, context)
    print(f"Executive Summary: {analysis.executive_summary}")
    print(f"Number of insights: {len(analysis.key_insights)}")
    print(f"Recommendations: {len(analysis.recommendations)}")
