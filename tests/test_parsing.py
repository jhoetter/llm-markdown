import pytest
from typing import List

from pydantic import BaseModel

from llm_markdown import _parse_response, _cast_type


# ---- models for testing ----------------------------------------------------

class Person(BaseModel):
    name: str
    age: int


# ---- _parse_response -------------------------------------------------------

def test_parse_none_return_type_passes_through():
    assert _parse_response("anything", None) == "anything"
    assert _parse_response(42, None) == 42


def test_parse_none_return_type_converts_dict_to_json():
    result = _parse_response({"key": "val"}, None)
    assert isinstance(result, str)
    assert "key" in result


def test_parse_pydantic_from_dict():
    result = _parse_response({"name": "Alice", "age": 30}, Person)
    assert isinstance(result, Person)
    assert result.name == "Alice"
    assert result.age == 30


def test_parse_pydantic_from_json_string():
    result = _parse_response('{"name": "Bob", "age": 25}', Person)
    assert isinstance(result, Person)
    assert result.name == "Bob"


def test_parse_pydantic_from_embedded_json():
    text = 'Here is the result: {"name": "Carol", "age": 40} hope that helps!'
    result = _parse_response(text, Person)
    assert isinstance(result, Person)
    assert result.name == "Carol"


def test_parse_pydantic_no_json_raises():
    with pytest.raises(ValueError, match="Could not find JSON"):
        _parse_response("no json here at all", Person)


def test_parse_list_passes_through():
    result = _parse_response(["a", "b", "c"], List[str])
    assert result == ["a", "b", "c"]


def test_parse_list_from_string():
    result = _parse_response('["x", "y"]', List[str])
    assert result == ["x", "y"]


def test_parse_str_return_type():
    result = _parse_response("hello", str)
    assert result == "hello"


def test_parse_int_return_type():
    result = _parse_response("42", int)
    assert result == 42


def test_parse_float_return_type():
    result = _parse_response("3.14", float)
    assert result == pytest.approx(3.14)


def test_parse_bool_return_type():
    assert _parse_response("true", bool) is True
    assert _parse_response("false", bool) is False
    assert _parse_response("yes", bool) is True
    assert _parse_response("no", bool) is False


# ---- _cast_type ------------------------------------------------------------

def test_cast_bool_truthy():
    for val in ("true", "True", "t", "yes", "y", "1"):
        assert _cast_type(val, bool) is True


def test_cast_bool_falsy():
    for val in ("false", "False", "no", "n", "0"):
        assert _cast_type(val, bool) is False


def test_cast_int():
    assert _cast_type("99", int) == 99


def test_cast_float():
    assert _cast_type("2.5", float) == pytest.approx(2.5)


def test_cast_str():
    assert _cast_type("hello", str) == "hello"


def test_cast_list_from_json():
    assert _cast_type('["a", "b"]', List[str]) == ["a", "b"]


def test_cast_list_single_value():
    assert _cast_type("single", List[str]) == ["single"]


def test_cast_list_comma_separated():
    result = _cast_type("[a, b, c]", List[str])
    assert result == ["a", "b", "c"]


def test_cast_empty_string_returns_empty():
    assert _cast_type("", str) == ""


def test_cast_invalid_int_returns_original():
    assert _cast_type("not_a_number", int) == "not_a_number"
