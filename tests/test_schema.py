from typing import List, Dict

from pydantic import BaseModel

from llm_markdown import (
    _make_schema_strict,
    _build_schema,
    _needs_structured_output,
    _system_instructions,
)


# ---- _make_schema_strict ---------------------------------------------------

def test_strict_adds_additional_properties_false():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    result = _make_schema_strict(schema)
    assert result["additionalProperties"] is False


def test_strict_sets_required_to_all_keys():
    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "string"},
            "y": {"type": "integer"},
        },
    }
    result = _make_schema_strict(schema)
    assert set(result["required"]) == {"x", "y"}


def test_strict_recurses_into_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "inner": {
                "type": "object",
                "properties": {"z": {"type": "number"}},
            }
        },
    }
    result = _make_schema_strict(schema)
    inner = result["properties"]["inner"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["z"]


def test_strict_recurses_into_array_items():
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
    }
    result = _make_schema_strict(schema)
    assert result["items"]["additionalProperties"] is False


def test_strict_recurses_into_defs():
    schema = {
        "type": "object",
        "properties": {},
        "$defs": {
            "Sub": {
                "type": "object",
                "properties": {"val": {"type": "string"}},
            }
        },
    }
    result = _make_schema_strict(schema)
    assert result["$defs"]["Sub"]["additionalProperties"] is False


def test_strict_handles_allof():
    schema = {
        "allOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}}
        ]
    }
    result = _make_schema_strict(schema)
    assert result["allOf"][0]["additionalProperties"] is False


def test_strict_passthrough_non_dict():
    assert _make_schema_strict("not a dict") == "not a dict"
    assert _make_schema_strict(42) == 42


# ---- _needs_structured_output ----------------------------------------------

def test_needs_structured_for_pydantic():
    class Item(BaseModel):
        name: str

    assert _needs_structured_output(Item) is True


def test_needs_structured_for_list():
    assert _needs_structured_output(List[str]) is True


def test_needs_structured_for_dict():
    assert _needs_structured_output(Dict[str, str]) is True


def test_no_structured_for_str():
    assert _needs_structured_output(str) is False


def test_no_structured_for_int():
    assert _needs_structured_output(int) is False


def test_no_structured_for_float():
    assert _needs_structured_output(float) is False


def test_no_structured_for_bool():
    assert _needs_structured_output(bool) is False


def test_no_structured_for_none():
    assert _needs_structured_output(None) is False


# ---- _build_schema --------------------------------------------------------

def test_schema_for_pydantic_model():
    class Item(BaseModel):
        name: str
        price: float

    schema = _build_schema(Item)
    assert schema["type"] == "object"
    assert "name" in schema["properties"]
    assert "price" in schema["properties"]
    assert schema["additionalProperties"] is False


def test_schema_for_nested_pydantic_has_defs():
    class Inner(BaseModel):
        value: int

    class Outer(BaseModel):
        items: List[Inner]

    schema = _build_schema(Outer)
    assert "$defs" in schema


def test_schema_for_list_str():
    schema = _build_schema(List[str])
    assert schema["type"] == "array"
    assert schema["items"] == {"type": "string"}


def test_schema_for_list_int():
    schema = _build_schema(List[int])
    assert schema["type"] == "array"
    assert schema["items"] == {"type": "integer"}


def test_schema_for_dict():
    schema = _build_schema(Dict[str, str])
    assert schema["type"] == "object"


def test_schema_for_str():
    schema = _build_schema(str)
    assert schema == {"type": "string"}


# ---- _system_instructions --------------------------------------------------

def test_instructions_for_pydantic_model():
    class Foo(BaseModel):
        bar: str

    msg = _system_instructions(Foo)
    assert "json" in msg.lower()
    assert "schema" in msg.lower()


def test_instructions_for_primitive():
    msg = _system_instructions(int)
    assert "json" in msg.lower()


def test_instructions_for_list():
    msg = _system_instructions(List[str])
    assert "json" in msg.lower()
