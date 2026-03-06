import pytest
from pydantic import BaseModel

from llm_markdown import prompt
from tests.conftest import MockProvider


class StreamModel(BaseModel):
    answer: str


def test_json_event_stream_emits_done_and_partial():
    provider = MockProvider(response='{"answer":"ok"}')

    @prompt(provider=provider, stream=True, stream_mode="json_events")
    def stream_answer(topic: str) -> StreamModel:
        """Answer about {topic}"""

    events = list(stream_answer("x"))
    event_types = [event["type"] for event in events]
    assert "delta_text" in event_types
    assert "partial_json" in event_types
    assert event_types[-1] == "done"
    assert events[-1]["output"].answer == "ok"


def test_json_event_stream_emits_error_on_invalid_json():
    provider = MockProvider(response="not-json")

    @prompt(provider=provider, stream=True, stream_mode="json_events")
    def stream_answer(topic: str) -> StreamModel:
        """Answer about {topic}"""

    events = list(stream_answer("x"))
    assert events[-1]["type"] == "error"


@pytest.mark.asyncio
async def test_async_json_event_stream():
    provider = MockProvider(response='{"answer":"ok"}')

    @prompt(provider=provider, stream=True, stream_mode="json_events")
    async def stream_answer(topic: str) -> StreamModel:
        """Answer about {topic}"""

    events = [event async for event in await stream_answer("x")]
    assert events[-1]["type"] == "done"
    assert events[-1]["output"].answer == "ok"
