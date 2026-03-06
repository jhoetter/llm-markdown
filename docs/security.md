# Security

## Secrets handling

- Use environment variables for API keys.
- Keep local secrets in `.env` (already gitignored).
- Never embed credentials in source code or test fixtures.

## Remote content

- `Image` URLs are fetched over HTTP(S); treat them as untrusted input.
- Prefer allowlisted hosts and validate upstream content in production systems.
- Use `LLM_MARKDOWN_IMAGE_URL_ALLOWLIST` to restrict allowed image hosts.
- Keep `LLM_MARKDOWN_IMAGE_BLOCK_PRIVATE_NETWORKS=true` (default) to block private-network targets.

## Recommended practices

- Rotate provider keys periodically.
- Use least-privilege project-scoped keys where available.
- Add CI secret scanning before release automation.
