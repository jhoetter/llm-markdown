import re
import inspect
from typing import get_type_hints, List, Dict, Any
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class llm:
    def __init__(self, provider, reasoning_first: bool = True):
        self.provider = provider
        self.reasoning_first = reasoning_first
        self.system_instructions = (
            """You are a helpful assistant."""
            if not reasoning_first
            else """
        You are a helpful assistant. Always structure your output as follows:
        <reasoning>
        Provide the reasoning behind your answer.
        </reasoning>
        <answer>
        Provide the final answer only, formatted as required.
        </answer>
        Always start with the reasoning first. Ensure that both reasoning and answer are complete, i.e. have both opening and closing tags.
        """
        )

    def parse_content(self, text: str, template_vars: dict) -> List[Dict[str, Any]]:
        """
        Parse text containing special syntax for multimodal content.
        Supports:
        - !image[url] for images (both HTTP URLs and base64 data)
        - Regular text
        """
        # Text is already formatted by Python, no need for additional templating
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

                image_url = match.group(1).strip()

                # If it's not already a data URL, assume it's a base64 string and convert it
                if not image_url.startswith("data:"):
                    image_url = f"data:image/jpeg;base64,{image_url}"

                # Add the image
                content.append({"type": "image_url", "image_url": {"url": image_url}})
            else:
                current_text.append(line)

        # Add any remaining text
        if current_text:
            content.append({"type": "text", "text": "\n".join(current_text).strip()})

        return content

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Bind arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            try:
                # Get function source and extract the string
                source = inspect.getsource(func)
                # Extract content between the first triple quotes
                match = re.search(r'"""(.*?)"""', source, re.DOTALL)
                if match:
                    # Get the raw prompt template
                    user_prompt = match.group(1)
                    if source.strip().endswith(".format"):
                        # If using .format(), let Python handle it
                        user_prompt = eval(
                            f"'''{user_prompt}'''", {}, bound_args.arguments
                        )
                    else:
                        # Otherwise treat as f-string
                        user_prompt = eval(
                            f"f'''{user_prompt}'''", {}, bound_args.arguments
                        )
                else:
                    # Fallback to docstring
                    user_prompt = inspect.getdoc(func)
                    if user_prompt:
                        user_prompt = eval(
                            f"f'''{user_prompt}'''", {}, bound_args.arguments
                        )

            except Exception as e:
                logger.debug(f"Error extracting string from function body: {e}")
                # Fallback to docstring
                user_prompt = inspect.getdoc(func)
                if user_prompt:
                    user_prompt = eval(
                        f"f'''{user_prompt}'''", {}, bound_args.arguments
                    )

            if not user_prompt:
                raise ValueError(
                    "Function must have either a string body or a docstring."
                )

            # Parse the content
            content = self.parse_content(user_prompt, bound_args.arguments)

            # Construct the conversation
            messages = [
                {"role": "system", "content": self.system_instructions},
                {
                    "role": "user",
                    "content": content if len(content) > 1 else content[0]["text"],
                },
            ]

            # Query the provider
            raw_response = self.provider.query(messages).strip()

            if self.reasoning_first:
                _ = self.extract_tag(raw_response, "reasoning")
                answer = self.extract_tag(raw_response, "answer")
            else:
                answer = raw_response

            # Validate and cast the extracted answer
            return_type = get_type_hints(func).get("return")
            if return_type and issubclass(return_type, BaseModel):
                return return_type.parse_raw(answer)
            return self.cast_type(answer, return_type)

        return wrapper

    @staticmethod
    def extract_tag(response: str, tag: str) -> str:
        """
        Extract content between <tag>...</tag>.
        """
        import re

        match = re.search(f"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
        raise ValueError(f"Tag <{tag}> not found in response: {response}")

    @staticmethod
    def cast_type(value: str, return_type: type):
        """
        Cast the value to the specified return type.
        """
        if return_type == bool:
            return value.lower() in ["true", "yes", "1"]
        elif return_type == int:
            return int(value)
        elif return_type == float:
            return float(value)
        elif return_type == list:
            return eval(value)
        return value
