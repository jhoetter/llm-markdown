from .openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter provider via OpenAI-compatible Chat Completions API.

    Args:
        api_key: Your OpenRouter API key.
        model: OpenRouter model identifier (e.g. "openai/gpt-4o-mini").
        max_tokens: Maximum response tokens.
        app_name: Optional app name used for OpenRouter attribution.
        app_url: Optional app URL used for OpenRouter attribution.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        max_tokens: int = 4096,
        app_name: str | None = None,
        app_url: str | None = None,
    ):
        headers = {}
        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_name:
            headers["X-Title"] = app_name

        super().__init__(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers or None,
        )
