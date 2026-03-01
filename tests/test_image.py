import base64
from typing import List
from unittest.mock import patch

import pytest

from llm_markdown import Image, _is_url, _to_data_uri, _PromptDecorator, prompt
from tests.conftest import MockProvider


# ---- _is_url ---------------------------------------------------------------

def test_is_url_with_http():
    assert _is_url("http://example.com/img.png") is True


def test_is_url_with_https():
    assert _is_url("https://example.com/img.png") is True


def test_is_url_with_plain_string():
    assert _is_url("not a url") is False


def test_is_url_with_base64():
    assert _is_url("aGVsbG8=") is False


def test_is_url_with_data_uri():
    assert _is_url("data:image/png;base64,abc") is False


def test_is_url_with_empty_string():
    assert _is_url("") is False


# ---- _to_data_uri ----------------------------------------------------------

def test_to_data_uri_passthrough_data_uri():
    uri = "data:image/png;base64,iVBORw0KGgo="
    assert _to_data_uri(uri) == uri


def test_to_data_uri_wraps_raw_base64():
    raw = base64.b64encode(b"fake image data").decode()
    result = _to_data_uri(raw)
    assert result.startswith("data:image/jpeg;base64,")
    assert raw in result


def test_to_data_uri_raises_on_invalid_input():
    with pytest.raises(ValueError, match="Image source must be"):
        _to_data_uri("!!!not-base64-and-not-url!!!")


def test_to_data_uri_fetches_url():
    with patch("llm_markdown.requests.get") as mock_get:
        mock_get.return_value.content = b"fake image"
        mock_get.return_value.headers = {"content-type": "image/png"}
        mock_get.return_value.raise_for_status = lambda: None

        result = _to_data_uri("https://example.com/photo.jpg")
        assert result.startswith("data:image/png;base64,")
        mock_get.assert_called_once_with("https://example.com/photo.jpg")


# ---- Image class -----------------------------------------------------------

def test_image_repr_short():
    img = Image("https://example.com/img.png")
    assert "Image(" in repr(img)
    assert "example.com" in repr(img)


def test_image_repr_truncates_long_source():
    long_source = "data:image/png;base64," + "A" * 100
    img = Image(long_source)
    assert "..." in repr(img)


def test_image_to_content_part():
    data_uri = "data:image/png;base64,iVBORw0KGgo="
    img = Image(data_uri)
    part = img.to_content_part()
    assert part["type"] == "image_url"
    assert part["image_url"]["url"] == data_uri


# ---- _collect_images -------------------------------------------------------

def test_collect_single_image():
    img = Image("data:image/png;base64,abc")
    hints = {"image": Image, "text": str, "return": str}
    args = {"image": img, "text": "hello"}
    result = _PromptDecorator._collect_images(args, hints)
    assert len(result) == 1
    assert result[0] is img


def test_collect_list_of_images():
    imgs = [Image("data:image/png;base64,a"), Image("data:image/png;base64,b")]
    hints = {"images": List[Image], "return": str}
    args = {"images": imgs}
    result = _PromptDecorator._collect_images(args, hints)
    assert len(result) == 2


def test_collect_skips_non_image_args():
    hints = {"text": str, "count": int, "return": str}
    args = {"text": "hello", "count": 5}
    result = _PromptDecorator._collect_images(args, hints)
    assert result == []


def test_collect_skips_return_hint():
    hints = {"return": Image}
    args = {}
    result = _PromptDecorator._collect_images(args, hints)
    assert result == []


# ---- _build_user_message ---------------------------------------------------

def test_build_message_text_only():
    result = _PromptDecorator._build_user_message("hello world", [])
    assert result == "hello world"


def test_build_message_with_images():
    img = Image("data:image/png;base64,iVBORw0KGgo=")
    result = _PromptDecorator._build_user_message("describe this", [img])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == {"type": "text", "text": "describe this"}
    assert result[1]["type"] == "image_url"


def test_build_message_with_multiple_images():
    imgs = [
        Image("data:image/png;base64,a"),
        Image("data:image/png;base64,b"),
    ]
    result = _PromptDecorator._build_user_message("compare", imgs)
    assert isinstance(result, list)
    assert len(result) == 3


# ---- decorated function with Image param ----------------------------------

def test_decorated_function_builds_multimodal_message():
    provider = MockProvider()

    @prompt(provider=provider)
    def describe(image: Image) -> str:
        """Describe this image."""

    describe(Image("data:image/png;base64,iVBORw0KGgo="))

    messages = provider.calls[0][1]
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "image_url"


def test_decorated_function_no_image_sends_plain_text():
    provider = MockProvider()

    @prompt(provider=provider)
    def greet(name: str) -> str:
        """Hello {name}"""

    greet("Alice")

    messages = provider.calls[0][1]
    assert isinstance(messages[1]["content"], str)
