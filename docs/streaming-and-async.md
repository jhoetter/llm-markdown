# Streaming and Async

## Streaming

Use `@prompt(provider, stream=True)` to return an iterator (sync) or async iterator (async function).

Default mode (`stream_mode="text"`) yields text chunks.  
Use `stream_mode="json_events"` for structured event streams.

Example event stream:

```python
@prompt(provider, stream=True, stream_mode="json_events")
def classify(text: str) -> dict:
    """Return JSON classification for: {text}"""
```

Event types:

- `delta_text`
- `partial_json`
- `done`
- `error`

## Async

Decorating `async def` functions works the same as sync functions:

```python
@prompt(provider)
async def analyze(text: str) -> str:
    """Analyze this text: {text}"""
```
