from llm_markdown import llm
from llm_markdown.providers.openai import OpenAIProvider
from llm_markdown.providers.langfuse import LangfuseWrapper
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
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


class PlanningInput(BaseModel):
    messages: List[dict]
    tools: List[dict]
    previous_plan: List[str]


@llm(provider=langfuse_provider, reasoning_first=True)
def plan_execution(planning_input: PlanningInput) -> List[str]:
    f"""
    You are a helpful assistant that plans the execution of a sequence of functions.
    You are given a list of messages and tools.
    You need to plan the execution of the functions in the tools by returning the names of the functions that need to be executed.
    If no functions need to be executed, return an empty list.

    Here is the list of messages:
    {planning_input.messages}

    Here is the list of tools:
    {planning_input.tools}

    Here is the list of previous plan:
    {planning_input.previous_plan}
    """


if __name__ == "__main__":
    planning_input = PlanningInput(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant for project 'my-test-project' (ID: 1). You can call specific functions if needed.",
            },
            {"role": "user", "content": "I want to translate the about section of the project to French."},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "update_project_about",
                    "description": "Update the about content of a project.\n\nArgs:\n    project_id (int): The ID of the project to update.\n    content (UpdateProjectAboutRequest): The new about content to be updated.\n        Expected to be a Pydantic model containing the about section structure.\n    db (Session): The database session.\n\nReturns:\n    dict: A status dictionary indicating success.\n\nRaises:\n    HTTPException: If the project is not found or database operation fails.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "number", "format": "integer"},
                            "content": {
                                "additionalProperties": {
                                    "properties": {
                                        "id": {"title": "Id", "type": "string"},
                                        "type": {
                                            "default": "Paragraph",
                                            "title": "Type",
                                            "type": "string",
                                        },
                                        "value": {
                                            "items": {
                                                "properties": {
                                                    "id": {
                                                        "title": "Id",
                                                        "type": "string",
                                                    },
                                                    "type": {
                                                        "default": "paragraph",
                                                        "title": "Type",
                                                        "type": "string",
                                                    },
                                                    "children": {
                                                        "items": {
                                                            "properties": {
                                                                "text": {
                                                                    "title": "Text",
                                                                    "type": "string",
                                                                }
                                                            },
                                                            "required": ["text"],
                                                            "title": "ParagraphText",
                                                            "type": "object",
                                                        },
                                                        "title": "Children",
                                                        "type": "array",
                                                    },
                                                },
                                                "required": ["id", "children"],
                                                "title": "ParagraphChildren",
                                                "type": "object",
                                            },
                                            "title": "Value",
                                            "type": "array",
                                        },
                                        "meta": {
                                            "properties": {
                                                "align": {
                                                    "default": "left",
                                                    "title": "Align",
                                                    "type": "string",
                                                },
                                                "depth": {
                                                    "default": 0,
                                                    "title": "Depth",
                                                    "type": "integer",
                                                },
                                                "order": {
                                                    "title": "Order",
                                                    "type": "integer",
                                                },
                                            },
                                            "required": ["order"],
                                            "title": "BlockMeta",
                                            "type": "object",
                                        },
                                    },
                                    "required": ["id", "value", "meta"],
                                    "title": "Block",
                                    "type": "object",
                                },
                                "type": "object",
                            },
                        },
                        "required": ["project_id", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_project_content",
                    "description": "Retrieve the complete content structure of a project including about, flow, and nodes.\n\nArgs:\n    project_id (int): The unique identifier of the project to retrieve.\n    db (Session): The database session for executing queries.\n\nReturns:\n    ProjectContent: The project's complete content structure.\n\nRaises:\n    HTTPException: 404 error if the project is not found",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_id": {"type": "number", "format": "integer"}
                        },
                        "required": ["project_id"],
                    },
                },
            },
        ],
        previous_plan=[],
    )

    result = plan_execution(planning_input)
    print(result)
    print(type(result))
