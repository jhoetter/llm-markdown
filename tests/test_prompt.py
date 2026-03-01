import pytest
from pydantic import BaseModel
from typing import List

from llm_markdown import prompt
from tests.conftest import MockProvider, BareProvider


# ---- helpers ---------------------------------------------------------------

class Movie(BaseModel):
    title: str
    year: int


# ---- prompt extraction -----------------------------------------------------

def test_prompt_extracts_docstring_and_interpolates(mock_provider):
    @prompt(provider=mock_provider)
    def greet(name: str) -> str:
        """Hello {name}, how are you?"""

    greet("Alice")
    messages = mock_provider.calls[0][1]
    assert messages[1]["content"] == "Hello Alice, how are you?"


def test_prompt_interpolates_multiple_args(mock_provider):
    @prompt(provider=mock_provider)
    def combine(a: str, b: int) -> str:
        """Value a={a} and b={b}"""

    combine("x", 42)
    assert mock_provider.calls[0][1][1]["content"] == "Value a=x and b=42"


def test_missing_docstring_raises_value_error(mock_provider):
    @prompt(provider=mock_provider)
    def no_doc(x: str) -> str:
        pass

    with pytest.raises(ValueError, match="must have a docstring"):
        no_doc("test")


# ---- type-based dispatch ---------------------------------------------------

def test_str_return_uses_plain_complete(mock_provider):
    @prompt(provider=mock_provider)
    def analyze(text: str) -> str:
        """Analyze: {text}"""

    result = analyze("hello")
    assert mock_provider.calls[0][0] == "complete"
    assert result == "mock response"


def test_int_return_uses_plain_complete(mock_provider):
    @prompt(provider=mock_provider)
    def count(text: str) -> int:
        """Count words in: {text}"""

    result = count("hello world")
    assert mock_provider.calls[0][0] == "complete"


def test_pydantic_return_uses_structured_output():
    structured = {"title": "Matrix", "year": 1999}
    provider = MockProvider(structured_response=structured)

    @prompt(provider=provider)
    def get_movie(desc: str) -> Movie:
        """Find movie: {desc}"""

    result = get_movie("sci-fi classic")
    assert any(c[0] == "complete_structured" for c in provider.calls)
    assert isinstance(result, Movie)
    assert result.title == "Matrix"
    assert result.year == 1999


def test_list_return_uses_structured_output():
    structured = ["step1", "step2", "step3"]
    provider = MockProvider(structured_response=structured)

    @prompt(provider=provider)
    def get_steps(task: str) -> List[str]:
        """List steps for: {task}"""

    result = get_steps("bake a cake")
    assert any(c[0] == "complete_structured" for c in provider.calls)
    assert result == ["step1", "step2", "step3"]


def test_bare_provider_falls_back_to_json_prompting():
    """When provider lacks complete_structured, fall back to complete() with JSON prompt."""
    provider = BareProvider(response='["a", "b"]')

    @prompt(provider=provider)
    def get_items(text: str) -> List[str]:
        """List items in: {text}"""

    result = get_items("test")
    assert provider.calls[0][0] == "complete"
    assert result == ["a", "b"]


def test_bare_provider_pydantic_fallback():
    """BareProvider falls back to JSON prompting for Pydantic models too."""
    provider = BareProvider(response='{"title": "Inception", "year": 2010}')

    @prompt(provider=provider)
    def get_movie(desc: str) -> Movie:
        """Find movie: {desc}"""

    result = get_movie("dream movie")
    assert isinstance(result, Movie)
    assert result.title == "Inception"


# ---- streaming -------------------------------------------------------------

def test_stream_returns_iterator(mock_provider):
    @prompt(provider=mock_provider, stream=True)
    def story(topic: str) -> str:
        """Tell about {topic}"""

    result = story("cats")
    chunks = list(result)
    assert "".join(chunks) == "mock response"
    call = mock_provider.calls[0]
    assert call[0] == "complete"
    assert call[2].get("stream") is True


# ---- async variants --------------------------------------------------------

@pytest.mark.asyncio
async def test_async_str_return():
    provider = MockProvider(response="async result")

    @prompt(provider=provider)
    async def analyze(text: str) -> str:
        """Analyze: {text}"""

    result = await analyze("hello")
    assert result == "async result"
    assert provider.calls[0][0] == "complete_async"


@pytest.mark.asyncio
async def test_async_pydantic_return():
    structured = {"title": "Matrix", "year": 1999}
    provider = MockProvider(structured_response=structured)

    @prompt(provider=provider)
    async def get_movie(desc: str) -> Movie:
        """Find movie: {desc}"""

    result = await get_movie("sci-fi")
    assert isinstance(result, Movie)
    assert any(c[0] == "complete_structured_async" for c in provider.calls)


@pytest.mark.asyncio
async def test_async_stream():
    provider = MockProvider(response="abc")

    @prompt(provider=provider, stream=True)
    async def story(topic: str) -> str:
        """Tell about {topic}"""

    result = await story("dogs")
    chunks = [c async for c in result]
    assert "".join(chunks) == "abc"


@pytest.mark.asyncio
async def test_async_bare_provider_fallback():
    provider = BareProvider(response='{"title": "Up", "year": 2009}')

    @prompt(provider=provider)
    async def get_movie(desc: str) -> Movie:
        """Find movie: {desc}"""

    result = await get_movie("balloon movie")
    assert isinstance(result, Movie)
    assert result.title == "Up"


# ---- metadata --------------------------------------------------------------

def test_langfuse_metadata_passed_to_provider():
    class MetadataProvider(MockProvider):
        def __init__(self):
            super().__init__()
            self.received_metadata = None

        def set_request_metadata(self, metadata):
            self.received_metadata = metadata

    provider = MetadataProvider()

    @prompt(
        provider=provider,
        langfuse_metadata={"category": "test"},
    )
    def analyze(text: str) -> str:
        """Analyze: {text}"""

    analyze("hello")
    assert provider.received_metadata == {"category": "test"}


# ---- wrapper preserves function metadata -----------------------------------

def test_wrapper_preserves_name_and_doc(mock_provider):
    @prompt(provider=mock_provider)
    def my_function(x: str) -> str:
        """My docstring about {x}"""

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My docstring about {x}"


# ---- system message --------------------------------------------------------

def test_system_message_is_set(mock_provider):
    @prompt(provider=mock_provider)
    def greet(name: str) -> str:
        """Hello {name}"""

    greet("Bob")
    messages = mock_provider.calls[0][1]
    assert messages[0]["role"] == "system"
    assert "helpful assistant" in messages[0]["content"]


# ---- default arguments -----------------------------------------------------

def test_default_arguments_work(mock_provider):
    @prompt(provider=mock_provider)
    def greet(name: str, greeting: str = "Hi") -> str:
        """Say {greeting} to {name}"""

    greet("Alice")
    content = mock_provider.calls[0][1][1]["content"]
    assert content == "Say Hi to Alice"
