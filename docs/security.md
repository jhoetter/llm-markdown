# Security

## Secrets handling

- Use environment variables for API keys.
- Keep local secrets in `.env` (already gitignored).
- Never embed credentials in source code or test fixtures.

## Remote content

- `Image` URLs are fetched over HTTP(S); treat them as untrusted input.
- Prefer allowlisted hosts and validate upstream content in production systems.

## Recommended practices

- Rotate provider keys periodically.
- Use least-privilege project-scoped keys where available.
- Add CI secret scanning before release automation.
