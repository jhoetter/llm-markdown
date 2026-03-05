# Streaming and Async

## Streaming

Use `@prompt(provider, stream=True)` to return an iterator (sync) or async iterator (async function).

`stream=True` is always plain text streaming, even when the return type is structured.

## Async

Decorating `async def` functions works the same as sync functions:

```python
@prompt(provider)
async def analyze(text: str) -> str:
    """Analyze this text: {text}"""
```
