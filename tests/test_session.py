from pydantic import BaseModel

from llm_markdown import Session
from tests.conftest import MockProvider


class Movie(BaseModel):
    title: str
    year: int


def test_session_persists_history_across_calls():
    provider = MockProvider(response="ok")
    session = Session(provider, system_prompt="You are helpful")

    @session.prompt()
    def ask(topic: str) -> str:
        """Tell me about {topic}"""

    ask("cats")
    ask("dogs")

    first_call_messages = provider.calls[0][1]
    second_call_messages = provider.calls[1][1]
    assert len(first_call_messages) == 2
    assert len(second_call_messages) == 4
    assert second_call_messages[2]["role"] == "assistant"


def test_session_structured_prompt():
    provider = MockProvider(structured_response={"title": "Matrix", "year": 1999})
    session = Session(provider)

    @session.prompt()
    def get_movie(desc: str) -> Movie:
        """Find movie: {desc}"""

    result = get_movie("sci-fi")
    assert isinstance(result, Movie)
    assert result.title == "Matrix"
    assert any(call[0] == "complete_structured" for call in provider.calls)


def test_session_stream_appends_after_consumption():
    provider = MockProvider(response="abc")
    session = Session(provider)

    @session.prompt(stream=True)
    def stream_story(topic: str) -> str:
        """Tell a story about {topic}"""

    chunks = list(stream_story("robots"))
    assert "".join(chunks) == "abc"
    assert session.history[-1]["role"] == "assistant"
    assert session.history[-1]["content"] == "abc"


def test_session_max_messages_trim():
    provider = MockProvider(response="ok")
    session = Session(provider, max_messages=2)

    @session.prompt()
    def ask(topic: str) -> str:
        """Topic: {topic}"""

    ask("a")
    ask("b")
    ask("c")

    non_system = [m for m in session.history if m["role"] != "system"]
    assert len(non_system) == 2
