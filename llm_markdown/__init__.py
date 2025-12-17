import re
import inspect
import json
import base64
import requests
from io import BytesIO
from urllib.parse import urlparse
from typing import get_type_hints, List, Dict, Any, get_origin
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


def is_url(string: str) -> bool:
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except:
        return False


def get_base64_image(input_data: str) -> str:
    """Convert input to base64 if it's a URL, or validate/return if already base64"""
    if is_url(input_data):
        response = requests.get(input_data)
        response.raise_for_status()
        image_data = BytesIO(response.content)
        content_type = response.headers.get("content-type", "image/jpeg")
        base64_data = base64.b64encode(image_data.read()).decode("utf-8")
        return f"data:{content_type};base64,{base64_data}"

    # If it's already base64 or data URI, validate and return
    try:
        if input_data.startswith("data:"):
            # Already a data URI
            return input_data
        else:
            # Try to decode to verify it's valid base64
            base64.b64decode(input_data)
            return f"data:image/jpeg;base64,{input_data}"
    except:
        raise ValueError(
            "Input must be either a valid URL or base64-encoded image data"
        )


class llm:
    def __init__(
        self,
        provider,
        reasoning_first: bool = True,
        stream: bool = False,
        langfuse_metadata: dict = None,
        max_retries: int = 2,
    ):
        self.provider = provider
        self.reasoning_first = reasoning_first
        self.stream = stream
        self.langfuse_metadata = langfuse_metadata or {}
        self.max_retries = max_retries

    def _make_schema_strict(self, schema: dict) -> dict:
        """
        Recursively add 'additionalProperties: false' to all object types in a schema.

        OpenAI's strict mode requires this for all object types in the schema.
        Also handles $defs for nested Pydantic models.
        """
        if not isinstance(schema, dict):
            return schema

        result = schema.copy()

        # If this is an object type, add additionalProperties: false
        if result.get("type") == "object":
            result["additionalProperties"] = False
            # Recursively process properties
            if "properties" in result:
                result["properties"] = {
                    k: self._make_schema_strict(v)
                    for k, v in result["properties"].items()
                }

        # Handle arrays
        if result.get("type") == "array" and "items" in result:
            result["items"] = self._make_schema_strict(result["items"])

        # Handle $defs (Pydantic puts nested model definitions here)
        if "$defs" in result:
            result["$defs"] = {
                k: self._make_schema_strict(v) for k, v in result["$defs"].items()
            }

        # Handle allOf, anyOf, oneOf
        for key in ["allOf", "anyOf", "oneOf"]:
            if key in result:
                result[key] = [self._make_schema_strict(s) for s in result[key]]

        return result

    def _build_structured_schema(self, return_type) -> dict:
        """
        Build JSON schema for structured output with reasoning.

        Creates a schema that wraps the return type with a reasoning field,
        suitable for use with OpenAI's response_format json_schema.
        """
        # Initialize $defs to potentially hoist from answer schema
        hoisted_defs = {}

        # Get answer schema based on return type
        if return_type is None:
            answer_schema = {"type": "string"}
        elif get_origin(return_type) is not None:
            # Handle typing annotations (List, Dict, etc)
            origin = get_origin(return_type)
            if origin in (list, List):
                answer_schema = {"type": "array", "items": {"type": "string"}}
            elif origin in (dict, Dict):
                answer_schema = {"type": "object", "additionalProperties": False}
            else:
                answer_schema = {"type": "string"}
        else:
            try:
                if issubclass(return_type, BaseModel):
                    # Get Pydantic model schema and make it strict for OpenAI
                    answer_schema = self._make_schema_strict(
                        return_type.model_json_schema()
                    )
                    # Hoist $defs to root level (OpenAI requires $refs to work from root)
                    if "$defs" in answer_schema:
                        hoisted_defs = answer_schema.pop("$defs")
                elif return_type == str:
                    answer_schema = {"type": "string"}
                elif return_type == int:
                    answer_schema = {"type": "integer"}
                elif return_type == float:
                    answer_schema = {"type": "number"}
                elif return_type == bool:
                    answer_schema = {"type": "boolean"}
                else:
                    answer_schema = {"type": "string"}
            except TypeError:
                answer_schema = {"type": "string"}

        # Wrap with reasoning
        result = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning process",
                },
                "answer": answer_schema,
            },
            "required": ["reasoning", "answer"],
            "additionalProperties": False,
        }

        # Add hoisted $defs at root level if any
        if hoisted_defs:
            result["$defs"] = hoisted_defs

        return result

    def _get_structured_system_instructions(self, return_type) -> str:
        """Generate simpler system instructions for structured output mode."""
        if return_type is None:
            return """You are a helpful assistant. You will respond with JSON.
Include your reasoning in the 'reasoning' field and your answer in the 'answer' field."""

        try:
            if get_origin(return_type) is None and issubclass(return_type, BaseModel):
                schema = return_type.model_json_schema()
                return f"""You are a helpful assistant. You will respond with JSON.
The 'reasoning' field should contain your step-by-step thought process.
The 'answer' field must be a valid JSON object matching this schema: {json.dumps(schema)}"""
        except TypeError:
            pass

        return """You are a helpful assistant. You will respond with JSON.
Include your reasoning in the 'reasoning' field and your answer in the 'answer' field."""

    def _parse_answer(self, answer, return_type):
        """Parse the answer into the expected return type."""
        if return_type is None:
            return answer

        # Handle typing annotations (List, Dict, etc)
        if get_origin(return_type) is not None:
            if isinstance(answer, str):
                return self.cast_type(answer, return_type)
            return answer

        try:
            if issubclass(return_type, BaseModel):
                # Handle Pydantic models
                if isinstance(answer, dict):
                    data = answer
                elif isinstance(answer, str):
                    # Try to extract JSON if it's a string
                    if not answer.strip().startswith("{"):
                        json_match = re.search(r"\{.*\}", answer, re.DOTALL)
                        if json_match:
                            answer = json_match.group(0)
                        else:
                            raise ValueError(
                                "Could not find JSON-like content in response"
                            )
                    data = json.loads(answer)
                else:
                    data = answer

                return return_type.model_validate(data)
        except TypeError:
            pass

        # Handle primitive types
        if isinstance(answer, str):
            return self.cast_type(answer, return_type)
        return answer

    def _get_correction_prompt(self) -> str:
        """Get the correction prompt for self-healing retry."""
        return """Your response was missing the required XML tags.
Please provide your COMPLETE response using EXACTLY this structure:

<reasoning>
Your step-by-step thinking here.
</reasoning>
<answer>
Your final answer here (JSON if returning an object).
</answer>

Start with <reasoning> and end with </answer>. No other text."""

    def _query_with_xml_fallback(self, messages: list, return_type) -> Any:
        """
        XML extraction with self-healing retry.

        Tries to extract reasoning and answer tags from the response.
        If extraction fails, retries with a correction prompt.
        """
        last_error = None
        working_messages = messages.copy()

        for attempt in range(self.max_retries + 1):
            raw_response = self.provider.query(working_messages, stream=False)
            raw_response = raw_response.strip()
            logger.debug(f"Raw LLM response (attempt {attempt + 1}):\n{raw_response}")

            try:
                reasoning = self.extract_tag(raw_response, "reasoning")
                logger.debug(f"Extracted reasoning:\n{reasoning}")
                answer = self.extract_tag(raw_response, "answer")
                logger.debug(f"Extracted answer:\n{answer}")
                logger.debug(f"XML extraction succeeded on attempt {attempt + 1}")
                return self._parse_answer(answer, return_type)

            except ValueError as e:
                last_error = e
                logger.warning(f"XML extraction failed (attempt {attempt + 1}): {e}")

                if attempt < self.max_retries:
                    # Self-healing: add correction prompt
                    working_messages.append(
                        {"role": "assistant", "content": raw_response}
                    )
                    working_messages.append(
                        {"role": "user", "content": self._get_correction_prompt()}
                    )

        raise last_error

    async def _query_with_xml_fallback_async(self, messages: list, return_type) -> Any:
        """
        Async version of XML extraction with self-healing retry.
        """
        last_error = None
        working_messages = messages.copy()

        for attempt in range(self.max_retries + 1):
            raw_response = await self.provider.query_async(
                working_messages, stream=False
            )
            raw_response = raw_response.strip()
            logger.debug(f"Raw LLM response (attempt {attempt + 1}):\n{raw_response}")

            try:
                reasoning = self.extract_tag(raw_response, "reasoning")
                logger.debug(f"Extracted reasoning:\n{reasoning}")
                answer = self.extract_tag(raw_response, "answer")
                logger.debug(f"Extracted answer:\n{answer}")
                logger.debug(f"XML extraction succeeded on attempt {attempt + 1}")
                return self._parse_answer(answer, return_type)

            except ValueError as e:
                last_error = e
                logger.warning(f"XML extraction failed (attempt {attempt + 1}): {e}")

                if attempt < self.max_retries:
                    # Self-healing: add correction prompt
                    working_messages.append(
                        {"role": "assistant", "content": raw_response}
                    )
                    working_messages.append(
                        {"role": "user", "content": self._get_correction_prompt()}
                    )

        raise last_error

    def get_system_instructions(self, return_type, use_structured: bool = False) -> str:
        """Generate appropriate system instructions based on return type."""
        # For structured output mode, use simpler instructions
        if use_structured:
            return self._get_structured_system_instructions(return_type)

        if not return_type:
            return "You are a helpful assistant."

        # Check if it's a typing annotation (List, Dict, etc)
        if get_origin(return_type) is not None:
            assert (
                self.reasoning_first
            ), "Reasoning first must be True for typing annotations"
            return """
            You are a helpful assistant. Always structure your output as follows:
            <reasoning>
            Provide the reasoning behind your answer.
            </reasoning>
            <answer>
            Provide the final answer only, formatted as required.
            </answer>

            You MUST follow the schema above exactly, i.e. start with the <reasoning> tag, close it with </reasoning>, then start the <answer> tag and close it with </answer>.
            """

        # Check if it's a Pydantic model
        if issubclass(return_type, BaseModel):
            assert (
                self.reasoning_first
            ), "Reasoning first must be True for Pydantic models"
            # Get the JSON schema for the Pydantic model
            schema = return_type.model_json_schema()
            return f"""
            You are a helpful assistant that always returns JSON output for Pydantic models.
            The expected response MUST exactly match this JSON schema:
            {schema}

            When responding, use the following structure:
            <reasoning>
            Explain the thought process (if necessary).
            </reasoning>
            <answer>
            A valid JSON object matching the schema above exactly.
            </answer>

            You MUST follow the schema above exactly, i.e. start with the <reasoning> tag, close it with </reasoning>, then start the <answer> tag and close it with </answer>.
            In the <answer> tag, you MUST return a valid JSON object matching the schema above exactly.
            """

        # For primitive types
        if self.reasoning_first:
            return """
            You are a helpful assistant. Always structure your output as follows:
            <reasoning>
            Provide the reasoning behind your answer.
            </reasoning>
            <answer>
            Provide the final answer only, formatted as required.
            </answer>

            You MUST follow the schema above exactly, i.e. start with the <reasoning> tag, close it with </reasoning>, then start the <answer> tag and close it with </answer>.
            In the <answer> tag, you MUST return the final answer only, formatted as required.
            """
        else:
            return "You are a helpful assistant."

    def parse_content(self, text: str, template_vars: dict) -> List[Dict[str, Any]]:
        """
        Parse text containing special syntax for multimodal content.
        Supports:
        - !image[url] for images (converts URLs to base64)
        - Regular text
        """
        lines = text.strip().split("\n")
        content = []
        current_text = []

        for line in lines:
            if match := re.match(r"!image\[(.*?)\]", line.strip()):
                # If we have accumulated text, add it first
                if current_text:
                    content.append(
                        {"type": "text", "text": "\n".join(current_text).strip()}
                    )
                    current_text = []

                image_input = match.group(1).strip()
                try:
                    image_data_uri = get_base64_image(image_input)
                    content.append(
                        {"type": "image_url", "image_url": {"url": image_data_uri}}
                    )
                except Exception as e:
                    logger.error(f"Failed to process image {image_input}: {e}")
                    raise
            else:
                current_text.append(line)

        # Add any remaining text
        if current_text:
            content.append({"type": "text", "text": "\n".join(current_text).strip()})

        return content

    def __call__(self, func):
        # Check if the decorated function is async
        is_async = inspect.iscoroutinefunction(func)

        if is_async:
            # If it's an async function, return an async wrapper
            async def async_wrapper(*args, **kwargs):
                # Get return type
                return_type = get_type_hints(func).get("return")

                # Get system instructions & set up the conversation
                system_instructions = self.get_system_instructions(return_type)
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                try:
                    # Attempt to extract user_prompt from function body
                    source = inspect.getsource(func)
                    match = re.search(r'"""(.*?)"""', source, re.DOTALL)
                    if match:
                        user_prompt = match.group(1)
                        # Evaluate as f-string if possible
                        user_prompt = eval(
                            f"f'''{user_prompt}'''", {}, bound_args.arguments
                        )
                    else:
                        # Fallback to docstring if no triple-quoted string was found
                        user_prompt = inspect.getdoc(func)
                        if user_prompt:
                            user_prompt = eval(
                                f"f'''{user_prompt}'''", {}, bound_args.arguments
                            )

                except Exception as e:
                    logger.debug(f"Error extracting string from function body: {e}")
                    user_prompt = inspect.getdoc(func)
                    if user_prompt:
                        user_prompt = eval(
                            f"f'''{user_prompt}'''", {}, bound_args.arguments
                        )

                if not user_prompt:
                    raise ValueError(
                        "Function must have either a string body or a docstring."
                    )

                # Parse the user prompt into content
                content = self.parse_content(user_prompt, bound_args.arguments)

                # Build messages for the LLM
                messages = [{"role": "system", "content": system_instructions}]

                # Handle special case for Pydantic models
                if return_type and get_origin(return_type) is None:
                    try:
                        if issubclass(return_type, BaseModel):
                            # Add a reminder to produce valid JSON
                            content_list = self.parse_content(
                                user_prompt, bound_args.arguments
                            )
                            if (
                                isinstance(content_list, list)
                                and len(content_list) == 1
                                and "text" in content_list[0]
                            ):
                                content_list[0][
                                    "text"
                                ] += "\n\n**Important**: Return as valid JSON matching the model's field names exactly."
                            user_msg = (
                                content_list
                                if len(content_list) > 1
                                else content_list[0]["text"]
                            )
                        else:
                            user_msg = (
                                content if len(content) > 1 else content[0]["text"]
                            )
                    except TypeError:
                        user_msg = content if len(content) > 1 else content[0]["text"]
                else:
                    user_msg = content if len(content) > 1 else content[0]["text"]

                messages.append({"role": "user", "content": user_msg})

                # Set metadata on LangfuseWrapper if it's being used
                if hasattr(self.provider, "set_request_metadata"):
                    self.provider.set_request_metadata(self.langfuse_metadata)

                # Handle streaming separately (no structured output for streaming)
                if self.stream:
                    raw_response = await self.provider.query_async(
                        messages, stream=True
                    )
                    return raw_response

                # PRIMARY PATH: Try structured output first if reasoning_first and provider supports it
                if self.reasoning_first and self.provider.supports_structured_output():
                    try:
                        schema = self._build_structured_schema(return_type)
                        # Use structured system instructions
                        structured_messages = [
                            {
                                "role": "system",
                                "content": self._get_structured_system_instructions(
                                    return_type
                                ),
                            },
                            {"role": "user", "content": user_msg},
                        ]

                        result = await self.provider.query_structured_async(
                            structured_messages, schema
                        )
                        reasoning = result.get("reasoning", "")
                        answer = result.get("answer")
                        logger.debug(f"Structured output reasoning: {reasoning}")
                        logger.debug(f"Structured output answer: {answer}")

                        return self._parse_answer(answer, return_type)

                    except Exception as e:
                        logger.warning(
                            f"Structured output failed, falling back to XML: {e}"
                        )
                        # Fall through to XML fallback path

                # FALLBACK PATH: XML tags with self-healing (or primary if no structured output support)
                if self.reasoning_first:
                    return await self._query_with_xml_fallback_async(
                        messages, return_type
                    )

                # No reasoning required, just query directly
                raw_response = await self.provider.query_async(messages, stream=False)
                raw_response = raw_response.strip()
                logger.debug(f"Raw LLM response:\n{raw_response}")

                return self._parse_answer(raw_response, return_type)

            return async_wrapper
        else:
            # For synchronous functions, use the original wrapper
            def wrapper(*args, **kwargs):
                # Get return type
                return_type = get_type_hints(func).get("return")

                # Get system instructions & set up the conversation
                system_instructions = self.get_system_instructions(return_type)
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                try:
                    # Attempt to extract user_prompt from function body
                    source = inspect.getsource(func)
                    match = re.search(r'"""(.*?)"""', source, re.DOTALL)
                    if match:
                        user_prompt = match.group(1)
                        # Evaluate as f-string if possible
                        user_prompt = eval(
                            f"f'''{user_prompt}'''", {}, bound_args.arguments
                        )
                    else:
                        # Fallback to docstring if no triple-quoted string was found
                        user_prompt = inspect.getdoc(func)
                        if user_prompt:
                            user_prompt = eval(
                                f"f'''{user_prompt}'''", {}, bound_args.arguments
                            )

                except Exception as e:
                    logger.debug(f"Error extracting string from function body: {e}")
                    user_prompt = inspect.getdoc(func)
                    if user_prompt:
                        user_prompt = eval(
                            f"f'''{user_prompt}'''", {}, bound_args.arguments
                        )

                if not user_prompt:
                    raise ValueError(
                        "Function must have either a string body or a docstring."
                    )

                # Parse the user prompt into content
                content = self.parse_content(user_prompt, bound_args.arguments)

                # Build messages for the LLM
                messages = [{"role": "system", "content": system_instructions}]

                # Handle special case for Pydantic models
                if return_type and get_origin(return_type) is None:
                    try:
                        if issubclass(return_type, BaseModel):
                            # Add a reminder to produce valid JSON
                            content_list = self.parse_content(
                                user_prompt, bound_args.arguments
                            )
                            if (
                                isinstance(content_list, list)
                                and len(content_list) == 1
                                and "text" in content_list[0]
                            ):
                                content_list[0][
                                    "text"
                                ] += "\n\n**Important**: Return as valid JSON matching the model's field names exactly."
                            user_msg = (
                                content_list
                                if len(content_list) > 1
                                else content_list[0]["text"]
                            )
                        else:
                            user_msg = (
                                content if len(content) > 1 else content[0]["text"]
                            )
                    except TypeError:
                        user_msg = content if len(content) > 1 else content[0]["text"]
                else:
                    user_msg = content if len(content) > 1 else content[0]["text"]

                messages.append({"role": "user", "content": user_msg})

                # Set metadata on LangfuseWrapper if it's being used
                if hasattr(self.provider, "set_request_metadata"):
                    self.provider.set_request_metadata(self.langfuse_metadata)

                # Handle streaming separately (no structured output for streaming)
                if self.stream:
                    raw_response = self.provider.query(messages, stream=True)
                    return raw_response

                # PRIMARY PATH: Try structured output first if reasoning_first and provider supports it
                if self.reasoning_first and self.provider.supports_structured_output():
                    try:
                        schema = self._build_structured_schema(return_type)
                        # Use structured system instructions
                        structured_messages = [
                            {
                                "role": "system",
                                "content": self._get_structured_system_instructions(
                                    return_type
                                ),
                            },
                            {"role": "user", "content": user_msg},
                        ]

                        result = self.provider.query_structured(
                            structured_messages, schema
                        )
                        reasoning = result.get("reasoning", "")
                        answer = result.get("answer")
                        logger.debug(f"Structured output reasoning: {reasoning}")
                        logger.debug(f"Structured output answer: {answer}")

                        return self._parse_answer(answer, return_type)

                    except Exception as e:
                        logger.warning(
                            f"Structured output failed, falling back to XML: {e}"
                        )
                        # Fall through to XML fallback path

                # FALLBACK PATH: XML tags with self-healing (or primary if no structured output support)
                if self.reasoning_first:
                    return self._query_with_xml_fallback(messages, return_type)

                # No reasoning required, just query directly
                raw_response = self.provider.query(messages, stream=False)
                raw_response = raw_response.strip()
                logger.debug(f"Raw LLM response:\n{raw_response}")

                return self._parse_answer(raw_response, return_type)

            return wrapper

    @staticmethod
    def extract_tag(response: str, tag: str) -> str:
        """
        Extract content between <tag>...</tag>.
        """
        match = re.search(f"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        raise ValueError(f"Tag <{tag}> not found in response: {response}")

    def cast_type(self, value: str, target_type) -> Any:
        """Cast a string value to the target type."""
        if not value:
            return value

        # Handle typing annotations (List, Dict, etc)
        origin = get_origin(target_type)
        if origin is not None:
            if origin in (list, List):
                # If the value looks like a string representation of a list
                try:
                    if value.startswith("[") and value.endswith("]"):
                        # Parse the string as JSON to get a proper list
                        return json.loads(value)
                    else:
                        # Single value, wrap it in a list
                        return [value]
                except json.JSONDecodeError:
                    # If JSON parsing fails, split by commas and strip whitespace
                    return [
                        item.strip()
                        for item in value.strip("[]").split(",")
                        if item.strip()
                    ]
            # Add more type handling here as needed
            return value

        # Handle primitive types
        try:
            if target_type == bool:
                return value.lower() in ("true", "t", "yes", "y", "1")
            return target_type(value)
        except (ValueError, TypeError):
            return value
