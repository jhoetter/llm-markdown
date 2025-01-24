import re
import inspect
import json
import base64
import requests
from io import BytesIO
from urllib.parse import urlparse
from typing import get_type_hints, List, Dict, Any
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
        content_type = response.headers.get('content-type', 'image/jpeg')
        base64_data = base64.b64encode(image_data.read()).decode('utf-8')
        return f"data:{content_type};base64,{base64_data}"
    
    # If it's already base64 or data URI, validate and return
    try:
        if input_data.startswith('data:'):
            # Already a data URI
            return input_data
        else:
            # Try to decode to verify it's valid base64
            base64.b64decode(input_data)
            return f"data:image/jpeg;base64,{input_data}"
    except:
        raise ValueError("Input must be either a valid URL or base64-encoded image data")

class llm:
    def __init__(self, provider, reasoning_first: bool = True):
        self.provider = provider
        self.reasoning_first = reasoning_first
        self.base_system_instructions = "You are a helpful assistant."

    def get_system_instructions(self, return_type) -> str:
        """Generate appropriate system instructions based on return type."""
        if return_type and issubclass(return_type, BaseModel):
            if self.reasoning_first:
                return """You are a helpful assistant that always returns JSON output for Pydantic models.
When responding, use the following structure:
<reasoning>
Explain the thought process (if necessary).
</reasoning>
<answer>
A valid JSON object matching the required fields exactly.
</answer>"""
            else:
                return """You are a helpful assistant that always returns JSON output for Pydantic models.
Your entire response must be a single valid JSON object matching the required fields exactly, with no additional text or explanation."""
        else:
            return self.base_system_instructions if not self.reasoning_first else """
You are a helpful assistant. Always structure your output as follows:
<reasoning>
Provide the reasoning behind your answer.
</reasoning>
<answer>
Provide the final answer only, formatted as required.
</answer>
"""

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
                    content.append({
                        "type": "image_url", 
                        "image_url": {"url": image_data_uri}
                    })
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
                    user_prompt = eval(f"f'''{user_prompt}'''", {}, bound_args.arguments)
                else:
                    # Fallback to docstring if no triple-quoted string was found
                    user_prompt = inspect.getdoc(func)
                    if user_prompt:
                        user_prompt = eval(f"f'''{user_prompt}'''", {}, bound_args.arguments)

            except Exception as e:
                logger.debug(f"Error extracting string from function body: {e}")
                user_prompt = inspect.getdoc(func)
                if user_prompt:
                    user_prompt = eval(f"f'''{user_prompt}'''", {}, bound_args.arguments)

            if not user_prompt:
                raise ValueError("Function must have either a string body or a docstring.")

            # Parse the user prompt into content
            content = self.parse_content(user_prompt, bound_args.arguments)

            # Build messages for the LLM
            messages = [
                {"role": "system", "content": system_instructions}
            ]

            if return_type and issubclass(return_type, BaseModel):
                # Add a reminder to produce valid JSON (in case the system prompt is insufficient)
                content_list = self.parse_content(user_prompt, bound_args.arguments)
                if isinstance(content_list, list) and len(content_list) == 1 and "text" in content_list[0]:
                    content_list[0]["text"] += "\n\n**Important**: Return as valid JSON matching the model's field names exactly."
                user_msg = content_list if len(content_list) > 1 else content_list[0]["text"]
            else:
                user_msg = content if len(content) > 1 else content[0]["text"]

            messages.append({"role": "user", "content": user_msg})

            # Query the provider
            raw_response = self.provider.query(messages).strip()
            logger.debug(f"Raw LLM response:\n{raw_response}")

            if self.reasoning_first:
                reasoning = self.extract_tag(raw_response, "reasoning")
                logger.debug(f"Extracted reasoning:\n{reasoning}")
                answer = self.extract_tag(raw_response, "answer")
                logger.debug(f"Extracted answer:\n{answer}")
            else:
                answer = raw_response

            # Validate and cast the extracted answer
            if return_type and issubclass(return_type, BaseModel):
                try:
                    # Attempt to extract JSON content if not well-formed
                    if not answer.strip().startswith("{"):
                        logger.warning(f"Response doesn't look like JSON, attempting to fix:\n{answer}")
                        json_match = re.search(r"\{.*\}", answer, re.DOTALL)
                        if json_match:
                            answer = json_match.group(0)
                            logger.debug(f"Extracted JSON-like content:\n{answer}")
                        else:
                            raise ValueError("Could not find JSON-like content in response")

                    # Parse the JSON to a dictionary so we can do post-processing
                    data = json.loads(answer)

                    # Example: rename "overall_sentiment" to "sentiment" if needed
                    if "overall_sentiment" in data and "sentiment" not in data:
                        logger.info("Renaming 'overall_sentiment' to 'sentiment'.")
                        data["sentiment"] = data.pop("overall_sentiment")

                    # Now parse into the Pydantic model
                    return return_type.parse_obj(data)
                except Exception as e:
                    logger.error(f"Failed to parse response as {return_type.__name__}: {e}")
                    logger.error(f"Raw response was:\n{answer}")
                    raise ValueError(
                        f"LLM response could not be parsed as {return_type.__name__}. "
                        f"Ensure the prompt requests a JSON response matching the model structure. "
                        f"Error: {str(e)}"
                    ) from e

            # For primitive types or no return type
            return self.cast_type(answer, return_type)

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
